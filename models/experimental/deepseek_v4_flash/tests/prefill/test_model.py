# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""PCC tests for the ttnn prefill model (``DeepSeekV4PrefillModel``): the whole network over a prompt.

Each test builds the reference ``DeepseekV4ForCausalLM`` (fp32, CPU; a standalone copy of the HF modeling
code from ``models.demos.deepseek_v3_d_p.reference.deepseek_v4``) with randomised weights and runs it over a
whole prompt in one pass, hooking every decoder layer to keep that layer's output stream stack. The device
model is then compared against it:

* ``test_prefill_model_layers``   -- the embedding and every layer's output, stepped layer by layer: the
  PCC after each layer shows where error enters and how it accumulates down the stack,
* ``test_prefill_model_logits``   -- the head on top: logits for every token and for the last token only,
  plus how often the reference's top-1 token is among the device's top few,
* ``test_prefill_model_chunked``  -- the prompt fed as several chunks through one list of per-layer states,
  each chunk's logits compared with the reference's rows for those positions, and the ``prefill`` helper,
* ``test_prefill_model_rejects_*``-- inputs v1 refuses, which must fail loudly before any device work.

The stack is a reduced-depth model with real V4-Flash dimensions: six layers, one of each attention type
(``sliding``, ``csa``, ``hca``) twice, the first three hash-routed and the last three learned -- so every
layer type and both routing kinds run in one network. It is reduced in two ways forced by the prefill
blocks, which are single-device (``tp_size == 1``) and whose routed-expert op is built for a per-chip
``I == 512``: the MoE is the TP=4 slice of the model (``moe_intermediate_size = 512``) and has 16 routed
experts, and the vocabulary is 1024 tokens. The real checkpoint (``I == 2048``, 256 experts) cannot run
through these blocks until they support tensor parallelism, so weights are randomised on purpose (as in
``test_decoder_layer.py``); CSA is checked in the regime v1 supports, at most ``index_topk`` compressed
entries (2048 tokens).

Error grows down the stack (every layer adds its bf8-attention and bf4-expert noise to the residual, and
learned routing flips a few experts per layer), so the floors below fall with depth and are set from
measured runs -- see ``_layer_floor``.

Run::

    pytest -s models/experimental/deepseek_v4_flash/tests/prefill/test_model.py
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
    DeepseekV4ForCausalLM,
    DeepseekV4RotaryEmbedding,
)
from models.experimental.deepseek_v4_flash.tt.model import DeepSeekV4PrefillModel

_SEED = 1234
_HIDDEN = 4096
_I_LOCAL = 512  # per-chip intermediate width the routed op is built for (I = 2048 at TP = 4)
_NUM_EXPERTS = 16
_TOP_K = 6
_VOCAB = 1024
_WEIGHT_STD = 0.02
_EXPERT_DTYPE = ttnn.bfloat4_b
_WEIGHT_DTYPE = ttnn.bfloat8_b

_ATTENTION_TYPES = ["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"]
_LAYER_TYPES = _ATTENTION_TYPES * 2
_MLP_TYPES = ["hash_moe"] * 3 + ["moe"] * 3

# PCC floors. The first layer is held to the single-layer floor of ``test_decoder_layer.py``; each further
# layer may lose a little more. Logits sit on top of the last layer's floor.
_FIRST_LAYER_PCC = 0.98
_PCC_LOSS_PER_LAYER = 0.006
_EMBED_PCC = 0.9999  # the embedding and stream expansion are exact copies of bf16 table rows
_LOGITS_PCC = 0.90
_LAST_LOGITS_PCC = 0.90
# The reference's top-1 token must be among the device's top-K for at least this fraction of positions.
_TOP_K_RANK = 5
_MIN_TOP_K_AGREEMENT = 0.8


@dataclass
class Reference:
    """One reference model plus everything the device run is compared against."""

    model: DeepseekV4ForCausalLM
    config: DeepseekV4Config
    input_ids: torch.Tensor  # [1, S]
    rope: dict  # {"main"|"compress": (cos_half [S, Rd/2], sin_half [S, Rd/2])}
    embeddings: torch.Tensor  # [1, S, D]
    layer_outputs: list  # per layer: [1, S, hc, D]
    logits: torch.Tensor  # [1, S, V]

    @property
    def weights(self) -> dict:
        """The checkpoint-named weights (no ``model.`` prefix) the ttnn model takes."""
        return {k.removeprefix("model."): v.detach() for k, v in self.model.state_dict().items()}


def _config() -> DeepseekV4Config:
    """V4-Flash dimensions (the TP=4 MoE slice, 16 experts, 1024-token vocab), one layer of each type twice."""
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
        tie_word_embeddings=False,
    )
    cfg._attn_implementation = "eager"  # V4 is eager-only: the sdpa interface silently drops the sinks
    return cfg


def _randomize(model: DeepseekV4ForCausalLM) -> None:
    """Give every parameter a sane random value (several are ``torch.empty`` in the reference).

    Everything is rounded through bf16, which is all the device holds for the small tensors: the
    comparison is then about compute fidelity, not the weight cast.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):  # q_a / kv / compressor norms, layer norms and the final norm
                param.uniform_(0.5, 1.5)
            elif name.endswith("self_attn.sinks"):
                param.normal_(0.0, 1.0)  # comparable to the QK logits' spread
            elif "position_bias" in name:
                param.normal_(0.0, 0.5)
            elif name.endswith(("_hc.fn", "hc_head.hc_fn")):
                param.normal_(0.0, _WEIGHT_STD)  # the model's init: mixes of order 1 after the norm
            elif name.endswith(("_hc.base", "hc_head.hc_base")):
                param.normal_(0.0, 0.1)
            elif name.endswith(("_hc.scale", "hc_head.hc_scale")):
                param.uniform_(0.5, 1.5)
            elif name.endswith("embed_tokens.weight"):
                param.normal_(0.0, 1.0)  # unit-scale hidden states, like the layer tests' streams
            elif name.endswith("lm_head.weight") or ".mlp." in name:
                param.normal_(0.0, _WEIGHT_STD)
            else:  # attention projections
                param.normal_(0.0, param.shape[-1] ** -0.5)
            param.copy_(param.to(torch.bfloat16).to(torch.float32))
        for layer in model.model.layers:
            gate = layer.mlp.gate
            if layer.mlp.is_hash:
                # k distinct experts per token id
                gate.tid2eid.copy_(torch.stack([torch.randperm(_NUM_EXPERTS)[:_TOP_K] for _ in range(_VOCAB)]))
            else:
                gate.e_score_correction_bias.normal_(0.0, _WEIGHT_STD)


@functools.lru_cache(maxsize=2)
def _reference(seq_len: int) -> Reference:
    """Run the reference model over ``seq_len`` random tokens in one pass (cached per length)."""
    torch.manual_seed(_SEED + seq_len)
    cfg = _config()
    model = DeepseekV4ForCausalLM(cfg).eval()
    _randomize(model)
    input_ids = torch.randint(0, _VOCAB, (1, seq_len))

    captured: dict[int, torch.Tensor] = {}
    hooks = [
        layer.register_forward_hook(lambda _m, _a, out, i=i: captured.__setitem__(i, out.detach().clone()))
        for i, layer in enumerate(model.model.layers)
    ]
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits
        embeddings = model.model.embed_tokens(input_ids)
        rotary = DeepseekV4RotaryEmbedding(cfg)
        position_ids = torch.arange(seq_len).unsqueeze(0)
        rope = {}
        for kind in ("main", "compress"):
            cos, sin = rotary(embeddings, position_ids=position_ids, layer_type=kind)
            rope[kind] = (cos[0].contiguous(), sin[0].contiguous())
    for hook in hooks:
        hook.remove()
    return Reference(model, cfg, input_ids, rope, embeddings, [captured[i] for i in range(len(hooks))], logits)


def _build(device, ref: Reference) -> DeepSeekV4PrefillModel:
    """The ttnn prefill model over ``ref``'s weights (routed weights uploaded in the decode layout)."""
    # The MoE blocks take bf16 weights (as in test_moe.py); everything else takes fp32 and converts to its
    # own dtype.
    weights = {k: v.to(torch.bfloat16) if ".mlp." in k and v.is_floating_point() else v for k, v in ref.weights.items()}

    def expert_provider(layer_idx: int):
        gate_up = weights[f"layers.{layer_idx}.mlp.experts.gate_up_proj"]  # [E, 2I, D]
        down = weights[f"layers.{layer_idx}.mlp.experts.down_proj"]  # [E, D, I]
        return lambda e: (gate_up[e], down[e])

    return DeepSeekV4PrefillModel(
        ref.config,
        weights,
        device,
        ref.rope,
        expert_provider=expert_provider,
        weight_dtype=_WEIGHT_DTYPE,
        expert_dtype=_EXPERT_DTYPE,
    )


def _to_host(t: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(t).to(torch.float32)


def _assert_pcc(expected: torch.Tensor, actual: torch.Tensor, floor: float, what: str) -> None:
    expected, actual = expected.to(torch.float32), actual.reshape(expected.shape).to(torch.float32)
    passing, message = comp_pcc(expected, actual, pcc=floor)
    logger.info(f"[{what}] PCC: {message}")
    assert passing, f"{what}: PCC below {floor}: {message}"


def _layer_floor(layer_idx: int) -> float:
    """Output PCC floor after layer ``layer_idx``: the single-layer floor, less a loss per layer of depth."""
    return _FIRST_LAYER_PCC - _PCC_LOSS_PER_LAYER * layer_idx


def _top_k_agreement(expected: torch.Tensor, actual: torch.Tensor) -> float:
    """Fraction of positions whose reference top-1 token is among the device's ``_TOP_K_RANK`` best."""
    top1 = expected.reshape(-1, expected.shape[-1]).argmax(dim=-1, keepdim=True)
    ranked = actual.reshape(-1, actual.shape[-1]).topk(_TOP_K_RANK, dim=-1).indices
    return (ranked == top1).any(dim=-1).float().mean().item()


def _check_logits(expected: torch.Tensor, actual: torch.Tensor, floor: float, what: str) -> None:
    """Logits PCC against ``floor``, plus how well the device ranks the reference's top-1 token."""
    _assert_pcc(expected, actual, floor, what)
    agree = _top_k_agreement(expected, actual.reshape(expected.shape))
    logger.info(f"[{what}] reference top-1 within device top-{_TOP_K_RANK}: {agree:.3f}")
    assert agree >= _MIN_TOP_K_AGREEMENT, f"{what}: top-{_TOP_K_RANK} agreement {agree:.3f} < {_MIN_TOP_K_AGREEMENT}"


@pytest.mark.parametrize("seq_len", (128, 1024))
def test_prefill_model_layers(device, reset_seeds, seq_len):
    """The embedding and each layer's stream stack against the reference, stepped layer by layer."""
    ref = _reference(seq_len)
    model = _build(device, ref)
    assert model.num_layers == len(_LAYER_TYPES)

    ids_dev = model._upload_ids(ref.input_ids)
    streams = model.embed(ids_dev)
    assert tuple(streams.shape) == (1, seq_len, ref.config.hc_mult, _HIDDEN)
    expected_streams = ref.embeddings.unsqueeze(2).expand(-1, -1, ref.config.hc_mult, -1)
    _assert_pcc(expected_streams, _to_host(streams), _EMBED_PCC, f"embedding streams, T={seq_len}")

    states = model.new_state()
    for i, (layer, state) in enumerate(zip(model.layers, states)):
        streams = layer(streams, state, ids_dev)
        what = f"layer {i} ({_LAYER_TYPES[i]} + {_MLP_TYPES[i]}) output, T={seq_len}"
        _assert_pcc(ref.layer_outputs[i], _to_host(streams), _layer_floor(i), what)
        assert state.seq_len == seq_len


@pytest.mark.parametrize("seq_len", (128, 1024))
def test_prefill_model_logits(device, reset_seeds, seq_len):
    """Logits for every token and for the last token only, against the reference's."""
    ref = _reference(seq_len)
    model = _build(device, ref)

    all_logits = model(ref.input_ids, last_only=False)
    assert tuple(all_logits.shape) == (1, 1, seq_len, _VOCAB)
    _check_logits(ref.logits, _to_host(all_logits), _LOGITS_PCC, f"logits, all {seq_len} tokens")

    last_logits = model(ref.input_ids)
    assert tuple(last_logits.shape) == (1, 1, 1, _VOCAB)
    last_ref = ref.logits[:, -1]
    _check_logits(last_ref, _to_host(last_logits), _LAST_LOGITS_PCC, f"logits, last token of {seq_len}")
    # The first generated token: the one thing a prompt's prefill exists to produce.
    assert int(_to_host(last_logits).argmax()) in _to_host(last_logits).reshape(-1).topk(_TOP_K_RANK).indices.tolist()


# Chunk splits of the same 1024-token prompt: a first chunk of several windows, then a chunk that reaches
# back into earlier ones, and a chunk past the routed op's 512-token slice.
_CHUNKINGS = [
    pytest.param((256, 256, 512), id="256-256-512"),
    pytest.param((128, 896), id="128-896"),
]


@pytest.mark.parametrize("chunks", _CHUNKINGS)
def test_prefill_model_chunked(device, reset_seeds, chunks):
    """The prompt as several chunks through one list of states matches the single-pass reference row for row."""
    seq_len = sum(chunks)
    ref = _reference(seq_len)
    model = _build(device, ref)

    states = model.new_state()
    start = 0
    for i, chunk in enumerate(chunks):
        end = start + chunk
        out = model(ref.input_ids[:, start:end], states, last_only=False)
        assert tuple(out.shape) == (1, 1, chunk, _VOCAB)
        _check_logits(ref.logits[:, start:end], _to_host(out), _LOGITS_PCC, f"chunk {i} logits, rows [{start}, {end})")
        start = end
        assert all(state.seq_len == start for state in states)


@pytest.mark.parametrize("chunk_size", (256, 1024), ids=("chunk256", "chunk1024"))
def test_prefill_model_prefill_helper(device, reset_seeds, chunk_size):
    """``prefill`` chunks a prompt itself and returns the last token's logits and the states decode continues from."""
    seq_len = 1024
    ref = _reference(seq_len)
    model = _build(device, ref)

    logits, states = model.prefill(ref.input_ids, chunk_size=chunk_size)
    assert tuple(logits.shape) == (1, 1, 1, _VOCAB)
    _check_logits(ref.logits[:, -1], _to_host(logits), _LAST_LOGITS_PCC, f"prefill last-token logits, {chunk_size}")
    assert len(states) == model.num_layers and all(state.seq_len == seq_len for state in states)
    for layer, state in zip(model.layers, states):
        assert (state.compressed_kv is None) == layer.self_attn.is_sliding


def test_prefill_model_rejects_bad_inputs(device, reset_seeds, expect_error):
    """Inputs v1 cannot process are refused before any device work (and leave the states untouched)."""
    ref = _reference(128)
    model = _build(device, ref)
    states = model.new_state()
    ids = ref.input_ids

    with expect_error(ValueError, "multiple of"):
        model(ids[:, :64], states)
    with expect_error(ValueError, "expected token ids"):
        model(torch.zeros(2, 128, dtype=torch.long), states)
    with expect_error(ValueError, "token ids must lie in"):
        model(torch.full((1, 128), _VOCAB, dtype=torch.long), states)
    with expect_error(ValueError, "layer states"):
        model(ids, states[:-1])
    with expect_error(ValueError, "chunk_size"):
        model.prefill(ids, chunk_size=100)
    assert all(state.seq_len == 0 and state.kv_tail is None for state in states)
