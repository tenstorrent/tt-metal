# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""PCC tests for the ttnn prefill attention block (``DeepSeekV4PrefillAttention``).

Each test builds the HuggingFace-style reference ``DeepseekV4Attention`` (fp32, CPU) for one of the
three layer types with randomised weights at the real V4-Flash dimensions, runs it over a whole
prompt in one pass, and compares the device block against it:

* ``test_prefill_attention_single_shot`` -- the whole prompt as one chunk, output PCC plus the
  state the block leaves behind (the K=V tail and the compressed entries),
* ``test_prefill_attention_chunked``     -- the same prompt fed as several chunks through one
  :class:`PrefillAttentionState`, each chunk's output compared with the reference's rows for those
  positions (so the sliding tail, the compressed-entry append and CSA's overlap carry are all
  exercised), plus the final state,
* ``test_prefill_attention_tp4_single_shot`` / ``test_prefill_attention_tp4_chunked`` -- the same two
  checks with the block tensor-parallel over a 1x4 submesh, like the decode block: ``q_b`` and the
  sinks head-sharded, ``o_a`` group-sharded, ``o_b`` row-parallel plus an all-reduce, everything else
  (``q_a``, ``kv``, the compressor and so the whole KV state) replicated. Needs an 8x4 system mesh and
  fabric, like the prefill MoE TP test. The output and every state tensor are replicated, so one
  rank's copy is compared against the same reference.
* ``test_prefill_attention_maskless_chunked`` / ``test_prefill_attention_tp4_maskless_chunked`` -- the chunked
  check of a sliding layer with ``maskless=True`` (dense causal sliding-window SDPA, no mask; CSA / HCA ignore it).
* ``test_prefill_attention_static_chunked`` -- the traced path's :meth:`forward_static` (run eagerly, no trace)
  over its persistent buffers, with the per-chunk step built on the host the way ``TracedPrefill`` builds it on
  device: each chunk against the reference rows, and the front-anchored entry rows (written at device-tensor row
  ids by ``indexed_fill``) against the reference entries. Masked, and mask-free for the sliding layer.

Randomised weights are used on purpose. A CSA layer cannot be referenced with the real checkpoint
(the HF module carries lightning-indexer weights the checkpoint does not ship), and random weights
keep the norms and the attention sink from being near-identity, so a PCC pass cannot come from
degenerate values. CSA is checked in the regime v1 supports: at most ``index_topk`` compressed
entries (2048 tokens), where the indexer's top-k is the identity and CSA is dense over the causally
visible entries.

The reference model comes from ``models.demos.deepseek_v3_d_p.reference.deepseek_v4`` (a standalone
copy of the HF modeling code that imports without the ``deepseek_v4`` transformers build).

Run::

    pytest -s models/experimental/deepseek_v4_flash/tests/prefill/test_attention.py
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
    DeepseekV4Attention,
    DeepseekV4RotaryEmbedding,
    apply_rotary_pos_emb,
)
from models.experimental.deepseek_v4_flash.tt.prefill.attention import (
    ALIGNMENT,
    DeepSeekV4PrefillAttention,
    PrefillAttentionState,
    PrefillStaticStep,
)

_SEED = 1234
_HIDDEN = 4096
# Layer index -> type. One layer of each kind, so ``layer_idx`` selects the block under test.
_LAYER_TYPES = ["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"]
_LAYERS = [
    pytest.param(0, id="sliding"),
    pytest.param(1, id="csa"),
    pytest.param(2, id="hca"),
]
_SLIDING_LAYER = [pytest.param(0, id="sliding")]  # the only layer type ``maskless`` changes

# (weight dtype, output PCC floor, state PCC floor). bf16 weights isolate the block's own numerics;
# bf8 is what a production prefill would run, so its floor is the one that matters.
_WEIGHT_DTYPES = [
    pytest.param(ttnn.bfloat16, 0.998, 0.998, id="bf16"),
    pytest.param(ttnn.bfloat8_b, 0.99, 0.99, id="bf8"),
]
_CHUNKED_PCC = 0.99
_CHUNKED_STATE_PCC = 0.99


@dataclass
class Reference:
    """One reference layer plus everything the device run is compared against."""

    module: DeepseekV4Attention
    config: DeepseekV4Config
    hidden: torch.Tensor  # [1, S, D]
    rope: dict  # {"main"|"compress": (cos_half [S, Rd/2], sin_half [S, Rd/2])}
    output: torch.Tensor  # [1, S, D]
    kv_tail: torch.Tensor  # [1, 1, sliding_window, Dh]: the last window of roped K=V rows
    entries: torch.Tensor | None  # [1, 1, S/rate, Dh]: compressed entries, HCA / CSA only

    @property
    def weights(self) -> dict:
        return dict(self.module.state_dict())


def _config() -> DeepseekV4Config:
    """V4-Flash attention dimensions, with one layer of each type."""
    cfg = DeepseekV4Config(
        hidden_size=_HIDDEN,
        head_dim=512,
        num_attention_heads=64,
        q_lora_rank=1024,
        o_groups=8,
        num_hidden_layers=len(_LAYER_TYPES),
        layer_types=list(_LAYER_TYPES),
    )
    cfg._attn_implementation = "eager"  # V4 is eager-only: the sdpa interface silently drops the sinks
    return cfg


def _randomize(module: DeepseekV4Attention) -> None:
    """Give every parameter a sane random value (several are ``torch.empty`` in the reference)."""
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name.endswith("norm.weight"):
                param.uniform_(0.5, 1.5)
            elif name == "sinks":
                param.normal_(0.0, 1.0)  # comparable to the QK logits' spread
            elif "position_bias" in name:
                param.normal_(0.0, 0.5)
            else:
                param.normal_(0.0, param.shape[-1] ** -0.5)


def _sliding_causal_mask(seq_len: int, sliding_window: int) -> torch.Tensor:
    """Additive ``[1, 1, S, S]`` mask: query ``i`` sees the ``sliding_window`` tokens ending at ``i``."""
    i = torch.arange(seq_len).view(seq_len, 1)
    j = torch.arange(seq_len).view(1, seq_len)
    keep = (j <= i) & (i - j < sliding_window)
    mask = torch.zeros(seq_len, seq_len).masked_fill(~keep, torch.finfo(torch.float32).min)
    return mask.view(1, 1, seq_len, seq_len)


@functools.lru_cache(maxsize=3)
def _reference(layer_idx: int, seq_len: int) -> Reference:
    """Run the reference layer over ``seq_len`` random tokens in one pass (cached per layer/length)."""
    torch.manual_seed(_SEED + 100 * layer_idx + seq_len)
    cfg = _config()
    module = DeepseekV4Attention(cfg, layer_idx).eval()
    _randomize(module)
    layer_type = cfg.layer_types[layer_idx]

    hidden = torch.randn(1, seq_len, cfg.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    rotary = DeepseekV4RotaryEmbedding(cfg)
    position_embeddings = {
        kind: rotary(hidden, position_ids=position_ids, layer_type=kind) for kind in ("main", "compress")
    }
    rope = {kind: (cos[0].contiguous(), sin[0].contiguous()) for kind, (cos, sin) in position_embeddings.items()}

    with torch.no_grad():
        # Only the sliding part of the mask is passed: the layer appends its compressor's block bias
        # (per-query causality + indexer validity) itself, which puts the reference's own CSA
        # indexer path under test.
        output, _ = module(
            hidden,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask=_sliding_causal_mask(seq_len, cfg.sliding_window),
            past_key_values=None,
        )

        kind = "main" if layer_type == "sliding_attention" else "compress"
        kv = module.kv_norm(module.kv_proj(hidden)).view(1, seq_len, -1, cfg.head_dim).transpose(1, 2)
        kv = apply_rotary_pos_emb(kv, *position_embeddings[kind])
        kv_tail = kv[:, :, seq_len - cfg.sliding_window :]

        entries = None
        if module.compressor is not None:
            q_residual = module.q_a_norm(module.q_a_proj(hidden))
            entries, _ = module.compressor(hidden, q_residual, position_ids, None, layer_idx)
    return Reference(module, cfg, hidden, rope, output, kv_tail, entries)


def _build(
    device, ref: Reference, layer_idx: int, weight_dtype, tp_size: int = 1, maskless: bool = False
) -> DeepSeekV4PrefillAttention:
    return DeepSeekV4PrefillAttention(
        ref.config,
        layer_idx,
        ref.weights,
        device,
        ref.rope,
        weight_dtype=weight_dtype,
        tp_size=tp_size,
        maskless=maskless,
    )


def _to_device(hidden: torch.Tensor, device, tp_size: int = 1) -> ttnn.Tensor:
    """``[1, T, D]`` -> ``[1, 1, T, D]`` bf16 TILE on the device (replicated on every TP rank)."""
    return ttnn.from_torch(
        hidden.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if tp_size > 1 else None,
    )


def _to_host(t: ttnn.Tensor, device=None, tp_size: int = 1) -> torch.Tensor:
    """A tensor back on the host. Under TP the block's output and state are replicated, so the ranks
    are stacked on dim 0 and rank 0's copy is returned."""
    if tp_size > 1:
        t = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))[:1]
        return t.to(torch.float32)
    return ttnn.to_torch(t).to(torch.float32)


def _assert_pcc(expected: torch.Tensor, actual: torch.Tensor, floor: float, what: str) -> None:
    expected, actual = expected.to(torch.float32), actual.reshape(expected.shape).to(torch.float32)
    passing, message = comp_pcc(expected, actual, pcc=floor)
    logger.info(f"[{what}] PCC: {message}")
    assert passing, f"{what}: PCC below {floor}: {message}"


def _check_state(
    ref: Reference,
    state: PrefillAttentionState,
    seq_len: int,
    floor: float,
    layer_type: str,
    device=None,
    tp_size: int = 1,
) -> None:
    """The state a prompt leaves behind must be what the reference computes for the same prompt."""
    assert state.seq_len == seq_len
    _assert_pcc(ref.kv_tail, _to_host(state.kv_tail, device, tp_size), floor, f"{layer_type} K=V tail")
    if ref.entries is None:
        assert state.compressed_kv is None
        return
    assert state.num_entries == ref.entries.shape[2], (
        f"{layer_type}: {state.num_entries} compressed entries on device, reference has " f"{ref.entries.shape[2]}"
    )
    _assert_pcc(ref.entries, _to_host(state.compressed_kv, device, tp_size), floor, f"{layer_type} compressed entries")


def _run_single_shot(device, tp_size, layer_idx, seq_len, weight_dtype, output_pcc, state_pcc) -> None:
    """A whole prompt as one chunk: output and resulting state against the reference."""
    ref = _reference(layer_idx, seq_len)
    layer_type = ref.config.layer_types[layer_idx]
    attn = _build(device, ref, layer_idx, weight_dtype, tp_size)

    state = attn.new_state()
    out = attn(_to_device(ref.hidden, device, tp_size), state)

    assert tuple(out.shape) == (1, 1, seq_len, _HIDDEN)
    tag = f"{layer_type} output, T={seq_len}" + (f", TP{tp_size}" if tp_size > 1 else "")
    _assert_pcc(ref.output, _to_host(out, device, tp_size), output_pcc, tag)
    _check_state(ref, state, seq_len, state_pcc, layer_type, device, tp_size)


@pytest.mark.parametrize("weight_dtype, output_pcc, state_pcc", _WEIGHT_DTYPES)
@pytest.mark.parametrize("layer_idx", _LAYERS)
@pytest.mark.parametrize("seq_len", (128, 1024))
def test_prefill_attention_single_shot(device, reset_seeds, layer_idx, seq_len, weight_dtype, output_pcc, state_pcc):
    """A whole prompt as one chunk: output and resulting state against the reference."""
    _run_single_shot(device, 1, layer_idx, seq_len, weight_dtype, output_pcc, state_pcc)


# Chunk splits of the same 1024-token prompt. A first chunk of exactly one window, a chunk that
# spans several compressed windows, and the all-minimum split.
_CHUNKINGS = [
    pytest.param((256, 256, 512), id="256-256-512"),
    pytest.param((128, 896), id="128-896"),
    pytest.param((128,) * 8, id="128x8"),
]


def _run_chunked(device, tp_size, layer_idx, chunks, maskless: bool = False) -> None:
    """The prompt as several chunks through one state matches the single-pass reference row for row."""
    seq_len = sum(chunks)
    ref = _reference(layer_idx, seq_len)
    layer_type = ref.config.layer_types[layer_idx]
    attn = _build(device, ref, layer_idx, ttnn.bfloat8_b, tp_size, maskless=maskless)

    state = attn.new_state()
    start = 0
    for i, chunk in enumerate(chunks):
        out = attn(_to_device(ref.hidden[:, start : start + chunk], device, tp_size), state)
        assert tuple(out.shape) == (1, 1, chunk, _HIDDEN)
        _assert_pcc(
            ref.output[:, start : start + chunk],
            _to_host(out, device, tp_size),
            _CHUNKED_PCC,
            f"{layer_type} chunk {i} rows [{start}, {start + chunk})" + (f", TP{tp_size}" if tp_size > 1 else ""),
        )
        start += chunk
        assert state.seq_len == start
    _check_state(ref, state, seq_len, _CHUNKED_STATE_PCC, layer_type, device, tp_size)


@pytest.mark.parametrize("chunks", _CHUNKINGS)
@pytest.mark.parametrize("layer_idx", _LAYERS)
def test_prefill_attention_chunked(device, reset_seeds, layer_idx, chunks):
    """The prompt as several chunks through one state matches the single-pass reference row for row."""
    _run_chunked(device, 1, layer_idx, chunks)


@pytest.mark.parametrize("chunks", _CHUNKINGS)
@pytest.mark.parametrize("layer_idx", _SLIDING_LAYER)
def test_prefill_attention_maskless_chunked(device, reset_seeds, layer_idx, chunks):
    """Mask-free sliding attention (dense causal sliding window) matches the reference row for row."""
    _run_chunked(device, 1, layer_idx, chunks, maskless=True)


def _static_step(attn: DeepSeekV4PrefillAttention, start: int, num_tokens: int, cap: int) -> PrefillStaticStep:
    """The chunk's :class:`PrefillStaticStep`, built on the host (``TracedPrefill._step_inputs`` makes it on device)."""
    positions = start + torch.arange(num_tokens)
    entry_rope, entry_rows, masks = {}, {}, {}
    if not attn.is_sliding:
        n = num_tokens // attn.rate
        entry_rope[attn.rate] = attn._rope_tables(start + attn.rate * torch.arange(n))
        entry_rows[attn.rate] = ttnn.from_torch(
            start // attn.rate + torch.arange(n, dtype=torch.int64),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=attn.device,
        )
    if not attn.maskless:
        static, threshold = attn.mask_tables_host(num_tokens, cap)
        mask = torch.where(threshold > start, float("-inf"), static)
        masks[attn.layer_type] = attn._to_device(mask.reshape(1, 1, num_tokens, -1))
    return PrefillStaticStep(
        rope={attn.rope_kind: attn._rope_tables(positions)},
        entry_rope=entry_rope,
        masks=masks,
        caps={attn.layer_type: cap},
        entry_rows=entry_rows,
        first_chunk=start == 0,
    )


# A whole number of equal chunks (one trace shape), and entry buffers larger than the prompt needs, as when the
# traces are prepared for a longer ``max_len`` than the prompt.
_STATIC_CHUNK = 256
_STATIC_SEQ_LEN = 1024
_STATIC_MAX_LEN = 2048


@pytest.mark.parametrize("tiered", [False, True], ids=["full-cap", "tiered"])
@pytest.mark.parametrize("maskless", [False, True], ids=["masked", "maskless"])
@pytest.mark.parametrize("layer_idx", _LAYERS)
def test_prefill_attention_static_chunked(device, reset_seeds, layer_idx, maskless, tiered):
    """``forward_static`` over persistent buffers, chunk by chunk, against the single-pass reference.

    ``tiered`` gives each chunk's step only the entry rows that chunk can reach (a position tier of the traced
    prefill: ``cap`` below the buffers' size), so the key axis and the entry reads grow with the position.
    """
    if maskless and layer_idx != 0:
        pytest.skip("maskless only changes sliding layers")
    ref = _reference(layer_idx, _STATIC_SEQ_LEN)
    layer_type = ref.config.layer_types[layer_idx]
    attn = _build(device, ref, layer_idx, ttnn.bfloat8_b, maskless=maskless)
    cap = 0
    if not attn.is_sliding:
        cap = -(-(_STATIC_MAX_LEN // attn.rate) // ALIGNMENT) * ALIGNMENT
    bufs = attn.new_static_buffers(_STATIC_CHUNK, cap)

    for start in range(0, _STATIC_SEQ_LEN, _STATIC_CHUNK):
        step_cap = cap
        if tiered and not attn.is_sliding:
            step_cap = min(cap, -(-((start + _STATIC_CHUNK) // attn.rate) // ALIGNMENT) * ALIGNMENT)
        step = _static_step(attn, start, _STATIC_CHUNK, step_cap)
        out = attn.forward_static(_to_device(ref.hidden[:, start : start + _STATIC_CHUNK], device), bufs, step)
        _assert_pcc(
            ref.output[:, start : start + _STATIC_CHUNK],
            _to_host(out),
            _CHUNKED_PCC,
            f"{layer_type} static chunk rows [{start}, {start + _STATIC_CHUNK})"
            + (", maskless" if maskless else "")
            + (f", cap {step_cap}" if tiered else ""),
        )

    _assert_pcc(ref.kv_tail, _to_host(bufs.tail), _CHUNKED_STATE_PCC, f"{layer_type} static K=V tail")
    if ref.entries is not None:
        emitted = ref.entries.shape[2]
        entries = _to_host(bufs.entries)
        _assert_pcc(ref.entries, entries[:, :, :emitted], _CHUNKED_STATE_PCC, f"{layer_type} static entry rows")
        assert not entries[:, :, emitted:].any(), f"{layer_type}: rows past the {emitted} entries were written"


# --------------------------------------------------------------------------- #
# Tensor parallel: the block split over a 1x4 submesh, as in decode.
# --------------------------------------------------------------------------- #
_TP_SIZE = 4
_PARENT_MESH = (8, 4)
# Sequence length of the TP single-shot runs: long enough for several 128-row SDPA chunks and, for HCA,
# eight closed compressed windows.
_TP_SEQ_LEN = 1024
# A subset of the chunkings: an uneven split that spans several compressed windows, and the all-minimum one.
_TP_CHUNKINGS = [
    pytest.param((256, 256, 512), id="256-256-512"),
    pytest.param((128,) * 8, id="128x8"),
]
_TP_DEVICE_PARAMS = [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}]


def _tp_submesh(mesh_device):
    """The 1x4 TP group carved out of the 8x4 system mesh."""
    if tuple(mesh_device.shape) != _PARENT_MESH:
        pytest.skip(f"need an {_PARENT_MESH[0]}x{_PARENT_MESH[1]} mesh, got {tuple(mesh_device.shape)}")
    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, _TP_SIZE), ttnn.MeshCoordinate(0, 0))
    assert submesh.get_num_devices() == _TP_SIZE
    return submesh


@pytest.mark.parametrize("device_params", _TP_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [_PARENT_MESH], indirect=True, ids=["8x4"])
@pytest.mark.parametrize("weight_dtype, output_pcc, state_pcc", _WEIGHT_DTYPES)
@pytest.mark.parametrize("layer_idx", _LAYERS)
def test_prefill_attention_tp4_single_shot(mesh_device, reset_seeds, layer_idx, weight_dtype, output_pcc, state_pcc):
    """TP4: a whole prompt as one chunk; the all-reduced output and the replicated state match the reference."""
    submesh = _tp_submesh(mesh_device)
    _run_single_shot(submesh, _TP_SIZE, layer_idx, _TP_SEQ_LEN, weight_dtype, output_pcc, state_pcc)


@pytest.mark.parametrize("device_params", _TP_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [_PARENT_MESH], indirect=True, ids=["8x4"])
@pytest.mark.parametrize("chunks", _TP_CHUNKINGS)
@pytest.mark.parametrize("layer_idx", _LAYERS)
def test_prefill_attention_tp4_chunked(mesh_device, reset_seeds, layer_idx, chunks):
    """TP4: the prompt as several chunks through one (replicated) state, row for row against the reference."""
    submesh = _tp_submesh(mesh_device)
    _run_chunked(submesh, _TP_SIZE, layer_idx, chunks)


@pytest.mark.parametrize("device_params", _TP_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [_PARENT_MESH], indirect=True, ids=["8x4"])
@pytest.mark.parametrize("chunks", _TP_CHUNKINGS)
@pytest.mark.parametrize("layer_idx", _SLIDING_LAYER)
def test_prefill_attention_tp4_maskless_chunked(mesh_device, reset_seeds, layer_idx, chunks):
    """TP4 mask-free sliding attention: each rank runs its 16 heads with no collective."""
    submesh = _tp_submesh(mesh_device)
    _run_chunked(submesh, _TP_SIZE, layer_idx, chunks, maskless=True)
