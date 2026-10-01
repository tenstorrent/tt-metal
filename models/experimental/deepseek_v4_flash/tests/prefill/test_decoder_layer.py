# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""PCC tests for the ttnn prefill decoder layer (``DeepSeekV4PrefillDecoderLayer``).

Each test builds the HuggingFace-style reference ``DeepseekV4DecoderLayer`` (fp32, CPU; a standalone copy
of the HF modeling code from ``models.demos.deepseek_v3_d_p.reference.deepseek_v4``) with randomised
weights at the real V4-Flash dimensions, runs it over a whole prompt in one pass, and compares the
device block against it. There is one case per layer type -- the attention kind (``sliding``, ``csa``,
``hca``) crossed with the MoE routing kind (``hash``: the first ``num_hash_layers`` layers, ``moe``:
learned top-k) -- so every code path a real layer can take runs:

* ``test_prefill_decoder_layer_single_shot`` -- the whole prompt as one chunk, at a length that fits in
  one sliding window and one that spans several (and past the routed op's 512-token slice),
* ``test_prefill_decoder_layer_chunked``     -- the same prompt as several chunks through one
  :class:`PrefillAttentionState`, each chunk's output compared with the reference's rows for those
  positions (the hyper-connections and the MoE are per-token, so the attention state is the only thing
  crossing a chunk boundary),
* ``test_prefill_decoder_layer_rejects_*``   -- inputs v1 refuses, which must fail loudly before any
  device work.

The building blocks have their own tests (``test_attention.py``, ``test_moe.py``,
``test_hyperconnection.py``); this one checks their composition: the stream layouts between them, the
norms, and the residual fold.

Single-device tests run the *TP=4 slice* of the model (``moe_intermediate_size = 512``), like
``test_moe.py``, because the routed op is built for a per-chip ``I == 512``. Weights are randomised on
purpose: a CSA layer cannot be referenced with the real checkpoint (the HF module carries
lightning-indexer weights the checkpoint does not ship), and random weights keep the norms, the attention
sink and the hyper-connection mixes from being near-identity. CSA is checked in the regime v1 supports:
at most ``index_topk`` compressed entries (2048 tokens), where the indexer's top-k is the identity.
The routed weights are BFloat4_b and the attention projections BFloat8_b (what a production prefill
runs), so the PCC floor is set against that quantization noise, not against bf16 numerics.

Run::

    pytest -s models/experimental/deepseek_v4_flash/tests/prefill/test_decoder_layer.py
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4DecoderLayer,
    DeepseekV4RotaryEmbedding,
)
from models.experimental.deepseek_v4_flash.tt.decode.moe import DeepSeekV4PreloadedExperts
from models.experimental.deepseek_v4_flash.tt.prefill.decoder_layer import DeepSeekV4PrefillDecoderLayer

_SEED = 1234
_HIDDEN = 4096
_HC = 4
_I_LOCAL = 512  # per-chip intermediate width the routed op is built for (I = 2048 at TP = 4)
_NUM_EXPERTS = 64
_TOP_K = 6
_VOCAB = 1024  # hash layers: size of the token-id -> expert-id table
_WEIGHT_STD = 0.02
_EXPERT_DTYPE = ttnn.bfloat4_b
_WEIGHT_DTYPE = ttnn.bfloat8_b

# Layer index -> (attention type, MoE kind). Layers 0-2 are hash-routed, 3-5 learned.
_ATTENTION_TYPES = ["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"]
_LAYER_TYPES = _ATTENTION_TYPES * 2
_MLP_TYPES = ["hash_moe"] * 3 + ["moe"] * 3
_LAYERS = [
    pytest.param(0, id="sliding-hash"),
    pytest.param(1, id="csa-hash"),
    pytest.param(2, id="hca-hash"),
    pytest.param(3, id="sliding-moe"),
    pytest.param(4, id="csa-moe"),
    pytest.param(5, id="hca-moe"),
]

# Output PCC floors over the whole ``[1, T, hc, D]`` stream stack. The sublayers' contribution rides on
# top of the (exactly mixed) input streams, so these sit above the MoE's own floors. Hash routing selects
# the experts exactly; the learned router ranks on bf16 scores, so a token whose 6th and 7th best experts
# are within one bf16 ulp can pick a different expert than the fp32 reference. Those tokens cost a little
# PCC (measured 0.977-0.980 at T=128 on all three attention types, against >= 0.98 for hash), which is why
# the learned floor is looser -- the same split as ``test_moe.py``. A wrong sublayer lands far below either
# (a layer with both sublayers zeroed correlates ~0.6 with the reference).
_PCC_HASH = 0.98
_PCC_LEARNED = 0.97
# Per-token PCC: tokens below _TOKEN_PCC are "bad", and at most this fraction of the tokens may be.
# Learned routing adds routing disagreements on top of the quantization noise. Hash routing selects the
# experts exactly, but the noise alone (bf4 experts, bf8 attention) leaves a thin tail around the cut: at
# T=1024 hca-hash has one token at 0.9486 with the next worst at 0.9634 and a median of 0.988, and the
# outliers sit at no window boundary -- so hash is allowed a ~1% tail rather than none. A broken sublayer
# puts most tokens below the cut, far beyond either allowance.
_TOKEN_PCC = 0.95
_MAX_BAD_TOKEN_FRACTION_LEARNED = 0.1
_MAX_BAD_TOKEN_FRACTION_HASH = 0.01


@dataclass
class Reference:
    """One reference layer plus everything the device run is compared against."""

    module: DeepseekV4DecoderLayer
    config: DeepseekV4Config
    streams: torch.Tensor  # [1, S, hc, D]
    input_ids: torch.Tensor  # [1, S]
    rope: dict  # {"main"|"compress": (cos_half [S, Rd/2], sin_half [S, Rd/2])}
    output: torch.Tensor  # [1, S, hc, D]

    @property
    def weights(self) -> dict:
        return {k: v.detach() for k, v in self.module.state_dict().items()}


def _config() -> DeepseekV4Config:
    """V4-Flash layer dimensions (the TP=4 MoE slice), with one layer of each type and MoE kind."""
    cfg = DeepseekV4Config(
        hidden_size=_HIDDEN,
        head_dim=512,
        num_attention_heads=64,
        q_lora_rank=1024,
        o_groups=8,
        moe_intermediate_size=_I_LOCAL,
        n_routed_experts=_NUM_EXPERTS,
        num_experts_per_tok=_TOP_K,
        vocab_size=_VOCAB,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        num_hidden_layers=len(_LAYER_TYPES),
        layer_types=list(_LAYER_TYPES),
        mlp_layer_types=list(_MLP_TYPES),
    )
    cfg._attn_implementation = "eager"  # V4 is eager-only: the sdpa interface silently drops the sinks
    return cfg


def _randomize(module: DeepseekV4DecoderLayer) -> None:
    """Give every parameter a sane random value (several are ``torch.empty`` in the reference).

    Everything is rounded through bf16, which is all the device holds for the small tensors: the
    comparison is then about compute fidelity, not the weight cast.
    """
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name.endswith("norm.weight"):  # q_a / kv / compressor norms and both layer norms
                param.uniform_(0.5, 1.5)
            elif name == "self_attn.sinks":
                param.normal_(0.0, 1.0)  # comparable to the QK logits' spread
            elif "position_bias" in name:
                param.normal_(0.0, 0.5)
            elif name.endswith("_hc.fn"):
                param.normal_(0.0, _WEIGHT_STD)  # the model's init: mixes of order 1 after the norm
            elif name.endswith("_hc.base"):
                param.normal_(0.0, 0.1)
            elif name.endswith("_hc.scale"):
                param.uniform_(0.5, 1.5)
            elif name.startswith("mlp."):
                param.normal_(0.0, _WEIGHT_STD)
            else:  # attention projections
                param.normal_(0.0, param.shape[-1] ** -0.5)
            param.copy_(param.to(torch.bfloat16).to(torch.float32))
        gate = module.mlp.gate
        if module.mlp.is_hash:
            # k distinct experts per token id
            gate.tid2eid.copy_(torch.stack([torch.randperm(_NUM_EXPERTS)[:_TOP_K] for _ in range(_VOCAB)]))
        else:
            gate.e_score_correction_bias.normal_(0.0, _WEIGHT_STD)


def _sliding_causal_mask(seq_len: int, sliding_window: int) -> torch.Tensor:
    """Additive ``[1, 1, S, S]`` mask: query ``i`` sees the ``sliding_window`` tokens ending at ``i``."""
    i = torch.arange(seq_len).view(seq_len, 1)
    j = torch.arange(seq_len).view(1, seq_len)
    keep = (j <= i) & (i - j < sliding_window)
    mask = torch.zeros(seq_len, seq_len).masked_fill(~keep, torch.finfo(torch.float32).min)
    return mask.view(1, 1, seq_len, seq_len)


@functools.lru_cache(maxsize=2)
def _reference(layer_idx: int, seq_len: int) -> Reference:
    """Run the reference layer over ``seq_len`` random tokens in one pass (cached per layer/length)."""
    torch.manual_seed(_SEED + 100 * layer_idx + seq_len)
    cfg = _config()
    module = DeepseekV4DecoderLayer(cfg, layer_idx).eval()
    _randomize(module)

    streams = torch.randn(1, seq_len, cfg.hc_mult, cfg.hidden_size).to(torch.bfloat16).to(torch.float32)
    input_ids = torch.randint(0, _VOCAB, (1, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0)
    rotary = DeepseekV4RotaryEmbedding(cfg)
    hidden = streams[:, :, 0]  # the rotary embedding only reads the dtype / device
    position_embeddings = {
        kind: rotary(hidden, position_ids=position_ids, layer_type=kind) for kind in ("main", "compress")
    }
    rope = {kind: (cos[0].contiguous(), sin[0].contiguous()) for kind, (cos, sin) in position_embeddings.items()}

    with torch.no_grad():
        # Only the sliding part of the mask is passed: the attention appends its compressor's block bias
        # (per-query causality + indexer validity) itself, which puts the reference's own CSA indexer
        # path under test.
        output = module(
            streams,
            input_ids=input_ids,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask=_sliding_causal_mask(seq_len, cfg.sliding_window),
            past_key_values=None,
        )
    return Reference(module, cfg, streams, input_ids, rope, output)


def _build(device, ref: Reference, layer_idx: int) -> DeepSeekV4PrefillDecoderLayer:
    """The ttnn prefill layer over ``ref``'s weights (routed weights uploaded in the decode layout)."""
    # The MoE block takes bf16 weights (as in test_moe.py); attention and the hyper-connections take fp32
    # and convert to their own dtypes.
    weights = {
        k: v.to(torch.bfloat16) if k.startswith("mlp.") and v.is_floating_point() else v for k, v in ref.weights.items()
    }
    gate_up = weights["mlp.experts.gate_up_proj"]  # [E, 2I, D]
    down = weights["mlp.experts.down_proj"]  # [E, D, I]

    def provider(e: int):
        return gate_up[e], down[e]

    experts = DeepSeekV4PreloadedExperts(ref.config, provider, device, dtype=_EXPERT_DTYPE)
    return DeepSeekV4PrefillDecoderLayer(
        ref.config, layer_idx, weights, device, ref.rope, experts=experts, weight_dtype=_WEIGHT_DTYPE
    )


def _to_device(streams: torch.Tensor, device) -> ttnn.Tensor:
    """``[1, T, hc, D]`` -> bf16 TILE on the device."""
    return ttnn.from_torch(streams, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _to_host(t: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(t).to(torch.float32)


def _assert_pcc(expected: torch.Tensor, actual: torch.Tensor, floor: float, what: str) -> None:
    expected, actual = expected.to(torch.float32), actual.reshape(expected.shape).to(torch.float32)
    passing, message = comp_pcc(expected, actual, pcc=floor)
    logger.info(f"[{what}] PCC: {message}")
    assert passing, f"{what}: PCC below {floor}: {message}"


def _assert_layer_output(
    expected: torch.Tensor,
    actual: torch.Tensor,
    layer: DeepSeekV4PrefillDecoderLayer,
    what: str,
    first_position: int = 0,
) -> None:
    """Output PCC against the floor for the layer's routing kind, plus the per-token spread.

    The per-token PCC (over each token's ``hc * D`` stream stack) shows whether a shortfall is a few
    tokens routed differently (learned routing: a minority far off, the rest fine) or every token a bit
    off (a numerics problem in a block). The worst tokens are logged with their absolute prompt position
    (``first_position`` is the chunk's start), so an outlier can be followed across runs and chunkings.
    """
    floor = _PCC_HASH if layer.is_hash else _PCC_LEARNED
    expected = expected.to(torch.float32)
    actual = actual.reshape(expected.shape).to(torch.float32)
    rows_ref, rows_out = expected.reshape(expected.shape[1], -1), actual.reshape(expected.shape[1], -1)
    token_pcc = torch.stack([torch.corrcoef(torch.stack([r, o]))[0, 1] for r, o in zip(rows_ref, rows_out)])
    bad = (token_pcc < _TOKEN_PCC).float().mean().item()
    logger.info(
        f"[{what}] per-token PCC: min {token_pcc.min():.4f}, median {token_pcc.median():.4f}, bad fraction {bad:.3f}"
    )
    worst = torch.topk(token_pcc, k=min(5, token_pcc.numel()), largest=False)
    logger.info(
        f"[{what}] worst tokens (position: PCC): "
        + ", ".join(f"{first_position + int(i)}: {v:.4f}" for v, i in zip(worst.values, worst.indices))
    )
    _assert_pcc(expected, actual, floor, what)
    limit = _MAX_BAD_TOKEN_FRACTION_HASH if layer.is_hash else _MAX_BAD_TOKEN_FRACTION_LEARNED
    assert bad <= limit, f"{what}: {bad:.1%} of the tokens have PCC < {_TOKEN_PCC} (allowed {limit:.0%})"


def _token_ids(ref: Reference, layer: DeepSeekV4PrefillDecoderLayer, start: int, end: int):
    """The chunk's token ids for a hash layer, ``None`` for a learned one (which ignores them)."""
    return ref.input_ids[:, start:end] if layer.is_hash else None


def _check_state(ref: Reference, state, seq_len: int, layer_type: str) -> None:
    """The attention state a prompt leaves behind: its length, and the compressed entries' count."""
    assert state.seq_len == seq_len
    if ref.module.self_attn.compressor is None:
        assert state.compressed_kv is None
        return
    rate = ref.config.compress_rates[layer_type]
    assert (
        state.num_entries == seq_len // rate
    ), f"{layer_type}: {state.num_entries} compressed entries on device, expected {seq_len // rate}"


@pytest.mark.parametrize("seq_len", (128, 1024))
@pytest.mark.parametrize("layer_idx", _LAYERS)
def test_prefill_decoder_layer_single_shot(device, reset_seeds, layer_idx, seq_len):
    """A whole prompt as one chunk: the layer's stream stack against the reference."""
    ref = _reference(layer_idx, seq_len)
    layer_type = ref.config.layer_types[layer_idx]
    layer = _build(device, ref, layer_idx)
    logger.info(f"[layer] {layer_type} + {'hash' if layer.is_hash else 'learned'} MoE, T={seq_len}")

    state = layer.new_state()
    streams = _to_device(ref.streams, device)
    out = layer(streams, state, _token_ids(ref, layer, 0, seq_len))

    assert tuple(out.shape) == (1, seq_len, _HC, _HIDDEN)
    _assert_layer_output(ref.output, _to_host(out), layer, f"{layer_type} layer output, T={seq_len}")
    _check_state(ref, state, seq_len, layer_type)
    # The input streams are the caller's: the layer must not have consumed them.
    _assert_pcc(ref.streams, _to_host(streams), 0.9999, "input streams after the layer")


# Chunk splits of the same 1024-token prompt: a first chunk of several windows, then a chunk that
# reaches back into an earlier one, and a chunk past the routed op's 512-token slice.
_CHUNKINGS = [
    pytest.param((256, 256, 512), id="256-256-512"),
    pytest.param((128, 896), id="128-896"),
]


@pytest.mark.parametrize("chunks", _CHUNKINGS)
@pytest.mark.parametrize("layer_idx", _LAYERS)
def test_prefill_decoder_layer_chunked(device, reset_seeds, layer_idx, chunks):
    """The prompt as several chunks through one state matches the single-pass reference row for row."""
    seq_len = sum(chunks)
    ref = _reference(layer_idx, seq_len)
    layer_type = ref.config.layer_types[layer_idx]
    layer = _build(device, ref, layer_idx)

    state = layer.new_state()
    start = 0
    for i, chunk in enumerate(chunks):
        end = start + chunk
        out = layer(_to_device(ref.streams[:, start:end], device), state, _token_ids(ref, layer, start, end))
        assert tuple(out.shape) == (1, chunk, _HC, _HIDDEN)
        _assert_layer_output(
            ref.output[:, start:end], _to_host(out), layer, f"{layer_type} chunk {i} rows [{start}, {end})", start
        )
        start = end
        assert state.seq_len == start
    _check_state(ref, state, seq_len, layer_type)


@pytest.mark.parametrize("layer_idx", (0, 3), ids=("hash", "moe"))
def test_prefill_decoder_layer_rejects_bad_chunks(device, reset_seeds, layer_idx, expect_error):
    """Chunks v1 cannot process are refused before any device work (and leave the state untouched)."""
    ref = _reference(layer_idx, 128)
    layer = _build(device, ref, layer_idx)
    state = layer.new_state()
    ids = ref.input_ids

    with expect_error(ValueError, "multiple of"):
        layer(_to_device(torch.randn(1, 64, _HC, _HIDDEN), device), state, ids[:, :64])
    with expect_error(ValueError, "expected streams"):
        layer(_to_device(torch.randn(2, 128, _HC, _HIDDEN), device), state, ids)
    with expect_error(ValueError, "expected streams"):
        layer(_to_device(torch.randn(1, 128, _HC + 1, _HIDDEN), device), state, ids)
    if layer.is_hash:
        with expect_error(ValueError, "token ids"):
            layer(_to_device(ref.streams, device), state, None)
    assert state.seq_len == 0 and state.kv_tail is None
