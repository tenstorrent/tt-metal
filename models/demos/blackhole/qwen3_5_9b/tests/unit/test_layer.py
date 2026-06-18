# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unit test for the Qwen3.5 hybrid decoder layer (tt/layer.py) vs the HF golden.

A decoder layer is the whole transformer block — input_layernorm → token mixer → residual →
post_attention_layernorm → MLP → residual — where the token mixer is either full (softmax)
attention OR a Gated DeltaNet, chosen per index by the model's layer_types. This test wires up
ONE HF Qwen3_5DecoderLayer with random weights, hands its (prefixed) state_dict to the TT
Qwen35DecoderLayer, and PCC-checks the TT output against HF. It is a wiring check: the token
mixers, MLP and norms each have their own exhaustive unit tests (test_attention / test_gdn /
test_mlp), so here we only confirm the block glues them together correctly.

Every test is parametrized by `layer_kind` so BOTH block kinds are exercised: `linear` resolves to
the first Gated DeltaNet layer, `full` to the first full-attention layer. Prefill and decode are
tested separately, and `*_trace` variants capture the whole block as a ttnn trace and replay it —
the path the demo runs — so we also prove the decoder layer is trace-compatible.

On the (1, 4) tensor-parallel mesh the residual stream is FRACTURED along the hidden dim (the
embedding shards it, every token-mixer/MLP output reduce-scatters it), so the block's input is
sharded on dim 3 and its output gathered on dim 3 — exactly the layout the real model streams
through the stacked layers. A single device keeps the full hidden dim (the gather is a no-op).

Run:
    pytest models/demos/blackhole/qwen3_5_9b/tests/unit/test_layer.py -v
"""
import os

import pytest
import torch
from loguru import logger
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen3_5_9b.tests.unit.reference import causal_mask, hf_rope, text_config
from models.demos.blackhole.qwen3_5_9b.tt.attention.rope_tp import rot_mats_decode, rot_mats_prefill
from models.demos.blackhole.qwen3_5_9b.tt.layer import Qwen35DecoderLayer
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.tt_transformers.tt.ccl import TT_CCL

### Test Parameters & Fixtures ─────────────────────────────────────────────────────────
os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-9B")

# Batches mirror the real caller, which differs by phase: the model prefills ONE user at a time
# (per-user KV fill / GDN's per-user state), and decodes the whole batch together. These are also the
# only shapes the decoder block supports on TP — a batched prefill's token-mixer output is flattened to
# [1,1,B*S,dim] by the reduce-scatter and no longer lines up with the [B,1,S,dim] residual, and a
# batch-1 decode on the 4-device mesh overflows the SDPA-decode core grid (both reproduce in the
# standalone test_attention).
PREFILL_BATCHES = [1]
DECODE_BATCHES = [32]
PREFILL_SEQLEN = 512  # prefill prompt length (a tile multiple)
DECODE_STEPS = 5  # decode tokens stepped from the fresh state
DECODE_CACHE_SEQ = 128  # KV-cache length for decode: a multiple of the SDPA-decode k-chunk (64)
PCC = 0.99
TRACE_REGION_SIZE = 268435456  # 256 MiB — the block's traced graph (token mixer + MLP + 2 norms) is heavy

# Mesh × device-params matrix shared by every test: (1, 1) single device and the (1, 4) TP mesh with
# the 1D fabric the token-mixer/MLP reduce-scatters ride. trace_region_size is reserved for the trace
# tests (harmless for the eager ones).
_MESH_PARAMS = [
    ((1, 1), {"trace_region_size": TRACE_REGION_SIZE}),
    ((1, 4), {"trace_region_size": TRACE_REGION_SIZE, "fabric_config": ttnn.FabricConfig.FABRIC_1D}),
]


@pytest.fixture
def setup():
    """The HF text config (eager attention, so the reference honors the explicit causal mask) and the
    HF rotary module. The per-layer HF golden is built inside each test because its kind — and so its
    submodules — depend on the parametrized layer_kind; the config/rope here are kind-independent."""
    cfg = text_config()
    return cfg, hf_rope(cfg)


# ── Helpers (build the matched HF/TT layer pair) ─────────────────────────────────────
def _resolve_layer_num(cfg, layer_kind):
    """First layer index of the requested kind. `linear` → Gated DeltaNet block, `full` → softmax
    attention block; the HF layer is built at this index so its layer_type matches the TT block."""
    target = "full_attention" if layer_kind == "full" else "linear_attention"
    return cfg.layer_types.index(target)


def _build_pair(mesh_device, cfg, args, layer_num):
    """Matched (HF, TT) decoder layers sharing one set of random weights.

    The HF Qwen3_5DecoderLayer is the golden; its state_dict is re-prefixed with `layers.{n}.` (what
    layer.py's substate/_make_norm look up) and handed to the TT block, so both compute with identical
    weights. TT_CCL only on a real mesh (it drives the reduce-scatters); the weight cache stays None —
    caching random weights would corrupt a later re-run."""
    hf_layer = Qwen3_5DecoderLayer(cfg, layer_idx=layer_num).to(torch.float32).eval()
    prefixed = {f"layers.{layer_num}.{k}": v.float() for k, v in hf_layer.state_dict().items()}
    tt_ccl = TT_CCL(mesh_device) if mesh_device.get_num_devices() > 1 else None
    tt_layer = Qwen35DecoderLayer(mesh_device, args, prefixed, layer_num, tensor_cache_path=None, tt_ccl=tt_ccl)
    return hf_layer, tt_layer


# ── Helpers (host torch ⇄ possibly-fractured TT residual stream) ─────────────────────
def _residual_mapper(mesh_device):
    """On TP the residual stream is fractured along the hidden dim, so shard dim 3 (DistributedNorm
    reduces+gathers it back before the token mixer); a single device holds the full dim (replicate)."""
    if mesh_device.get_num_devices() > 1:
        return ttnn.ShardTensorToMesh(mesh_device, dim=3)
    return ttnn.ReplicateTensorToMesh(mesh_device)


def _to_device_prefill(x, mesh_device):
    """Prefill residual layout: [B, S, dim] host → [B, 1, S, dim] bf16 DRAM, fractured/replicated."""
    B, S, dim = x.shape
    return ttnn.from_torch(
        x.to(torch.bfloat16).reshape(B, 1, S, dim),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        mesh_mapper=_residual_mapper(mesh_device),
    )


def _to_device_decode(x, mesh_device):
    """Decode residual layout: [B, 1, dim] host → [1, 1, B, dim] bf16 DRAM, same hidden-dim fracture."""
    B, _, dim = x.shape
    return ttnn.from_torch(
        x.to(torch.bfloat16).reshape(1, 1, B, dim),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        mesh_mapper=_residual_mapper(mesh_device),
    )


def _positions_tt(mesh_device, positions):
    """Per-user decode positions [B] (int32) → replicated TT tensor (drives the KV-cache update index
    and the decode RoPE; ignored by the GDN block, which carries its position implicitly)."""
    return ttnn.from_torch(
        positions, dtype=ttnn.int32, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )


def _from_device(out, mesh_device):
    """The block's output is fractured along the hidden dim like its input (the final residual add of
    two reduce-scattered tensors), so gather dim 3 on TP (no-op on a single device)."""
    nd = mesh_device.get_num_devices()
    return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3 if nd > 1 else 0)).float()


# ── Helpers (HF golden for one prefill / decode call) ────────────────────────────────
def _hf_prefill(hf_layer, rope, cfg, x, is_full):
    """The HF block's prefill output for one user. is_full selects the explicit causal mask the eager
    attention path needs (None for GDN). The cache it fills is a throwaway — prefill PCC is the check."""
    B, S, _ = x.shape
    pos_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    cos, sin = rope(x, pos_ids)
    return hf_layer(
        x,
        position_embeddings=(cos, sin),
        attention_mask=causal_mask(S) if is_full else None,
        position_ids=pos_ids,
        past_key_values=DynamicCache(config=cfg),
    )


def _hf_decode_step(hf_layer, rope, x, step, hf_cache):
    """One HF decode step at position `step`, advancing the (shared) cache. attention_mask=None — decode
    reads the whole accumulated history. RoPE/position are consumed by attention; GDN ignores them."""
    pos_ids = torch.full((x.shape[0], 1), step, dtype=torch.long)
    cos_d, sin_d = rope(x, pos_ids)
    return hf_layer(
        x, position_embeddings=(cos_d, sin_d), attention_mask=None, position_ids=pos_ids, past_key_values=hf_cache
    )[
        :, 0
    ]  # [B, dim]


def _tt_decode_step(tt_layer, args, mesh_device, x, step):
    """One TT decode step at position `step`. Per-user positions drive the attention KV-cache update and
    decode RoPE (the GDN block ignores them, carrying its position implicitly in the recurrent state)."""
    positions = torch.full((x.shape[0],), step, dtype=torch.int32)
    cos_dt, sin_dt = rot_mats_decode(mesh_device, args.rope_head_dim, args.max_seq_len, args.rope_theta, positions)
    return tt_layer.forward_decode(
        _to_device_decode(x, mesh_device),
        cos=cos_dt,
        sin=sin_dt,
        position_tensor=_positions_tt(mesh_device, positions),
    )


# ── Tests ────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
@pytest.mark.parametrize("layer_kind", ["linear", "full"], ids=lambda k: k)
@pytest.mark.parametrize("mesh_device, device_params", _MESH_PARAMS, indirect=True)
@pytest.mark.parametrize("batch", PREFILL_BATCHES, ids=lambda b: f"b{b}")
def test_decoder_prefill(mesh_device, batch, layer_kind, setup, reset_seeds, ensure_gc):
    """Eager whole-block prefill PCC vs HF on (1, 1) and the (1, 4) TP mesh, for both the GDN and
    full-attention block kinds. One user (the model's per-user prefill), the production prefill shape."""
    cfg, rope = setup
    is_full = layer_kind == "full"

    # 1. Resolve the layer index of this kind and build the matched HF + TT blocks
    args = Qwen35ModelArgs(mesh_device, max_batch_size=batch, max_seq_len=PREFILL_SEQLEN)
    layer_num = _resolve_layer_num(cfg, layer_kind)
    assert args.is_full_attention_layer(layer_num) == is_full, "TT and HF disagree on layer kind"
    hf_layer, tt_layer = _build_pair(mesh_device, cfg, args, layer_num)

    # 2. Random residual-stream input + HF golden (fills a throwaway cache as a side effect)
    x = torch.randn(batch, PREFILL_SEQLEN, args.dim, dtype=torch.float32)  # [B, seq, dim]
    out_ref = _hf_prefill(hf_layer, rope, cfg, x, is_full)  # [B, seq, dim]

    # 3. TT prefill through the whole block (norm → token mixer → residual → norm → MLP → residual)
    cos_tt, sin_tt = rot_mats_prefill(mesh_device, args.rope_head_dim, PREFILL_SEQLEN, args.rope_theta)
    out_tt = tt_layer.forward_prefill(_to_device_prefill(x, mesh_device), cos=cos_tt, sin=sin_tt)
    out_tt = _from_device(out_tt, mesh_device).reshape(batch, PREFILL_SEQLEN, args.dim)

    # 4. Compare reference vs TT output PCC
    passing, pcc = comp_pcc(out_ref, out_tt, PCC)
    logger.info(f"decoder prefill PCC (mesh={tuple(mesh_device.shape)}, kind={layer_kind}, b={batch}) = {pcc}")
    assert passing, f"prefill output PCC too low (kind={layer_kind}, b={batch}): {pcc}"


@torch.no_grad()
@pytest.mark.parametrize("layer_kind", ["linear", "full"], ids=lambda k: k)
@pytest.mark.parametrize("mesh_device, device_params", _MESH_PARAMS, indirect=True)
@pytest.mark.parametrize("batch", DECODE_BATCHES, ids=lambda b: f"b{b}")
def test_decoder_decode(mesh_device, batch, layer_kind, setup, reset_seeds, ensure_gc):
    """Eager whole-block decode PCC vs HF on (1, 1) and the (1, 4) TP mesh, for both kinds, at the
    production decode batch.

    The batch steps one token at a time from a FRESH (zero) token-mixer state — the block starts with an
    empty KV cache / zeroed GDN conv+recurrent state, and HF decodes from a matching empty cache, so the
    two stay in lock-step as the state advances. This validates the decode wiring (norm → decode token
    mixer → residual → norm → MLP → residual) and the per-step state advancement. Seeding a prefilled
    history instead would need the model's per-user prefill loop (attention) / state injection (GDN);
    decode-from-zero keeps the wiring check simple and exercises the identical forward_decode path."""
    cfg, rope = setup
    is_full = layer_kind == "full"

    # 1. Build the matched blocks (fresh zero state); HF decodes from a matching empty config-aware cache
    args = Qwen35ModelArgs(mesh_device, max_batch_size=batch, max_seq_len=DECODE_CACHE_SEQ)
    layer_num = _resolve_layer_num(cfg, layer_kind)
    assert args.is_full_attention_layer(layer_num) == is_full, "TT and HF disagree on layer kind"
    hf_layer, tt_layer = _build_pair(mesh_device, cfg, args, layer_num)
    hf_cache = DynamicCache(config=cfg)

    # 2. Step the batch through DECODE_STEPS tokens, comparing the HF vs TT output PCC each step
    for step in range(DECODE_STEPS):
        x = torch.randn(batch, 1, args.dim, dtype=torch.float32)
        ref = _hf_decode_step(hf_layer, rope, x, step, hf_cache)  # [B, dim]
        out_tt = _from_device(_tt_decode_step(tt_layer, args, mesh_device, x, step), mesh_device).reshape(
            batch, args.dim
        )

        passing, pcc = comp_pcc(ref, out_tt, PCC)
        logger.info(
            f"decoder decode step {step} PCC (mesh={tuple(mesh_device.shape)}, kind={layer_kind}, b={batch}) = {pcc}"
        )
        assert passing, f"decode PCC too low (kind={layer_kind}, b={batch}, step={step}): {pcc}"


@torch.no_grad()
@pytest.mark.parametrize("layer_kind", ["linear", "full"], ids=lambda k: k)
@pytest.mark.parametrize("mesh_device, device_params", _MESH_PARAMS, indirect=True)
@pytest.mark.parametrize("batch", PREFILL_BATCHES, ids=lambda b: f"b{b}")
def test_decoder_prefill_trace(mesh_device, batch, layer_kind, setup, reset_seeds, ensure_gc):
    """Same prefill PCC as test_decoder_prefill, but the whole block runs as a captured ttnn trace —
    the path the demo replays to collapse the norms, token mixer, MLP and residual adds into one device
    dispatch. The block's prefill is idempotent (the token-mixer state write is an idempotent side
    effect), so the replay reproduces the eager golden with no state juggling."""
    cfg, rope = setup
    is_full = layer_kind == "full"

    # 1. Build the matched blocks and the random input + HF golden
    args = Qwen35ModelArgs(mesh_device, max_batch_size=batch, max_seq_len=PREFILL_SEQLEN)
    layer_num = _resolve_layer_num(cfg, layer_kind)
    assert args.is_full_attention_layer(layer_num) == is_full, "TT and HF disagree on layer kind"
    hf_layer, tt_layer = _build_pair(mesh_device, cfg, args, layer_num)

    x = torch.randn(batch, PREFILL_SEQLEN, args.dim, dtype=torch.float32)  # [B, seq, dim]
    out_ref = _hf_prefill(hf_layer, rope, cfg, x, is_full)  # [B, seq, dim]

    x_tt = _to_device_prefill(x, mesh_device)
    cos_tt, sin_tt = rot_mats_prefill(mesh_device, args.rope_head_dim, PREFILL_SEQLEN, args.rope_theta)

    # 2. Compile the kernels (trace capture cannot compile), then capture once. `out` is the persistent
    #    buffer the trace writes into.
    tt_layer.forward_prefill(x_tt, cos=cos_tt, sin=sin_tt)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = tt_layer.forward_prefill(x_tt, cos=cos_tt, sin=sin_tt)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)

    # 3. Replay recomputes the whole block into `out` with a single dispatch, then read it back
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    out_tt = _from_device(out, mesh_device).reshape(batch, PREFILL_SEQLEN, args.dim)
    ttnn.release_trace(mesh_device, tid)

    passing, pcc = comp_pcc(out_ref, out_tt, PCC)
    logger.info(f"decoder prefill TRACE PCC (mesh={tuple(mesh_device.shape)}, kind={layer_kind}, b={batch}) = {pcc}")
    assert passing, f"prefill trace PCC too low (kind={layer_kind}, b={batch}): {pcc}"


@torch.no_grad()
@pytest.mark.parametrize("layer_kind", ["linear", "full"], ids=lambda k: k)
@pytest.mark.parametrize("mesh_device, device_params", _MESH_PARAMS, indirect=True)
@pytest.mark.parametrize("batch", DECODE_BATCHES, ids=lambda b: f"b{b}")
def test_decoder_decode_trace(mesh_device, batch, layer_kind, setup, reset_seeds, ensure_gc):
    """Same decode PCC as test_decoder_decode, but one decode step runs as a captured ttnn trace — the
    path the demo replays for every generated token. One step at position 0 from the fresh state.

    Decode reads the token-mixer state, and the trace bakes in the persistent state-buffer addresses
    (not their contents), so before replay those buffers must hold the same fresh S0 the HF golden
    assumes. compile + capture each advance the state once, so:
      * full attention: decode at position 0 reads only position 0, and the replayed step writes that
        position itself, so the KV history needs no restore.
      * GDN: zero its conv + recurrent buffers IN PLACE (preserving the captured addresses) back to S0."""
    cfg, rope = setup
    is_full = layer_kind == "full"

    # 1. Build the matched blocks (fresh zero state); HF golden = one decode step from an empty cache
    args = Qwen35ModelArgs(mesh_device, max_batch_size=batch, max_seq_len=DECODE_CACHE_SEQ)
    layer_num = _resolve_layer_num(cfg, layer_kind)
    assert args.is_full_attention_layer(layer_num) == is_full, "TT and HF disagree on layer kind"
    hf_layer, tt_layer = _build_pair(mesh_device, cfg, args, layer_num)

    x = torch.randn(batch, 1, args.dim, dtype=torch.float32)
    out_ref = _hf_decode_step(hf_layer, rope, x, 0, DynamicCache(config=cfg))  # [B, dim]

    # 2. Fixed decode inputs at position 0 — their buffer addresses are baked into the trace, so the
    #    exact same tensors are reused for compile, capture and replay.
    positions = torch.zeros(batch, dtype=torch.int32)
    x_tt = _to_device_decode(x, mesh_device)
    pos_tt = _positions_tt(mesh_device, positions)
    cos_dt, sin_dt = rot_mats_decode(mesh_device, args.rope_head_dim, args.max_seq_len, args.rope_theta, positions)

    def decode_once():
        return tt_layer.forward_decode(x_tt, cos=cos_dt, sin=sin_dt, position_tensor=pos_tt)

    # 3. Compile the decode kernels (capture cannot compile), then capture once. `out` is the persistent
    #    buffer the trace writes into. Both calls advance the GDN state (attention only re-touches pos 0).
    decode_once()
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = decode_once()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)

    # 4. Restore the fresh start state into the buffers the trace reads (GDN only — see docstring), then
    #    replay one decode step from S0 and read it back.
    if not is_full:
        gdn = tt_layer.attention
        for buf in (gdn.conv_state, gdn.last_recurrent_state):
            ttnn.copy(ttnn.mul(buf, 0.0), buf)  # zero in place, preserving the captured buffer address
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    out_tt = _from_device(out, mesh_device).reshape(batch, args.dim)
    ttnn.release_trace(mesh_device, tid)

    passing, pcc = comp_pcc(out_ref, out_tt, PCC)
    logger.info(f"decoder decode TRACE PCC (mesh={tuple(mesh_device.shape)}, kind={layer_kind}, b={batch}) = {pcc}")
    assert passing, f"decode trace PCC too low (kind={layer_kind}, b={batch}): {pcc}"
