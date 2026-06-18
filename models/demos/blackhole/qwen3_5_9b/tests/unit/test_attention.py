# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unit test for the Qwen3.5 gated full (softmax) attention vs the HF golden.

Build ONE HF Qwen3_5Attention with random weights (eager, so it honors the explicit causal
mask we pass), hand its state_dict to the TT Qwen35Attention, then PCC-check the TT output
against HF for prefill and decode.

Every test is parametrized by a `paged` flag: `contig` exercises the contiguous KV cache
(fill_cache / scaled_dot_product_attention_decode), `paged` exercises the block-paged KV cache
through a scrambled page table (paged_fill_cache / paged_scaled_dot_product_attention_decode) —
the path vLLM drives. Decode continues from a cache the real forward_prefill filled (the
prefill→decode hand-off), which also keeps the per-device KV sharding correct on TP without the
test re-deriving the head→device layout.

Every case runs on both a (1, 1) single device and the (1, 4) tensor-parallel mesh (the latter
with the 1D fabric the o_proj reduce-scatter rides), across batch ∈ {1, 32}. test_attention_*
run eagerly; test_attention_*_trace capture the forward as a ttnn trace and replay it, so the
traced path the demo uses is exercised too.

Run:
    pytest models/demos/blackhole/qwen3_5_9b/tests/unit/test_attention.py -v
"""
import os

import pytest
import torch
from loguru import logger
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen3_5_9b.tests.unit.reference import causal_mask, hf_rope
from models.demos.blackhole.qwen3_5_9b.tt.attention import Qwen35Attention
from models.demos.blackhole.qwen3_5_9b.tt.attention.kv_cache import init_kv_cache
from models.demos.blackhole.qwen3_5_9b.tt.attention.rope_tp import rot_mats_decode, rot_mats_prefill
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import PagedAttentionConfig

### Test Parameters & Fixtures ─────────────────────────────────────────────────────────
os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-9B")

BATCHES = [1, 32]
PREFILL_SEQLEN = [512]  # prefill prompt length (a PAGE_BLOCK_SIZE multiple, so paged fill is block-aligned)
DECODE_PREFILL_LEN = 128  # history the decode tests continue from — short but spans multiple paged blocks
DECODE_STEPS = 5
PAGE_BLOCK_SIZE = 64  # block-paged KV-cache page size, in token positions (mirrors the vLLM/rob.py default)
PCC = 0.99
TRACE_REGION_SIZE = 268435456  # 256 MiB — attention's traced graph (QKV, norms, RoPE, SDPA, gate, wo) is heavy


@pytest.fixture
def setup(mesh_device):
    """HF Qwen3_5Attention golden (random weights), its state_dict, and the HF rotary module. No args
    is handed out — unlike the MLP/GDN, the TT module bakes max_batch_size/max_seq_len into its cache
    and decode shard grid at __init__, so each test builds its OWN sized args in _setup_kv. eager
    attention is REQUIRED so the reference honors the explicit causal mask we pass (sdpa/flash
    re-derive masks differently). q_norm/k_norm stay RAW on the HF side — HF adds the zero-centered
    +1 internally while the TT loader bakes it into the weight."""
    args = Qwen35ModelArgs(mesh_device, max_seq_len=max(PREFILL_SEQLEN))  # config-only, to build the HF golden
    cfg = args.hf_config.get_text_config()
    cfg._attn_implementation = "eager"
    hf_attn = Qwen3_5Attention(config=cfg, layer_idx=0).to(torch.float32).eval()
    return hf_attn, hf_attn.state_dict(), hf_rope(cfg)


# ── Helpers (host torch ⇄ possibly-sharded TT tensors) ───────────────────────────────
def _build_page_table(mesh_device, batch, blocks_per_user) -> ttnn.Tensor:
    """Scrambled [B, blocks_per_user] page table: user b's logical block i → a shuffled physical
    block. argsort(randperm) is a bijection over all physical blocks, so users never collide and a
    correct paged read can't accidentally rely on contiguous storage. int32 ROW_MAJOR, replicated."""
    max_num_blocks = batch * blocks_per_user
    generator = torch.Generator().manual_seed(0)
    page_table = torch.argsort(torch.randperm(max_num_blocks, generator=generator))
    page_table = page_table.reshape(batch, blocks_per_user).to(torch.int32)
    return ttnn.from_torch(
        page_table,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _setup_kv(mesh_device, state_dict, batch, need_seq, paged):
    """The TT attention under test, with a KV cache sized to span need_seq positions.

    A FRESH Qwen35ModelArgs is built per case: the cache shape AND the decode KV-update height-shard
    grid (args.kv_update_shard_cfg) bake in max_batch_size/max_seq_len at construction, so they must
    be set before __init__ — not mutated after. TT_CCL only on a real mesh (it drives the o_proj
    reduce-scatter); the cache path stays None — caching random weights would corrupt a re-run.
    Returns (attn, args, page_table); page_table is None for the contiguous cache, else the scrambled
    bijection the paged fill/read follow."""
    if paged:
        blocks_per_user = -(-need_seq // PAGE_BLOCK_SIZE)  # ceil
        cache_seq = blocks_per_user * PAGE_BLOCK_SIZE
    else:
        cache_seq = -(-need_seq // 32) * 32  # round up to a tile

    args = Qwen35ModelArgs(mesh_device, max_seq_len=cache_seq, max_batch_size=batch)
    tt_ccl = TT_CCL(mesh_device) if mesh_device.get_num_devices() > 1 else None

    if paged:
        attn = Qwen35Attention(mesh_device, state_dict, args, tt_ccl, create_kv_cache=False)
        paged_cfg = PagedAttentionConfig(block_size=PAGE_BLOCK_SIZE, max_num_blocks=batch * blocks_per_user)
        attn.kv_cache = init_kv_cache(
            mesh_device=mesh_device,
            args=args,
            max_batch_size=batch,
            max_seq_len=cache_seq,
            paged_attention_config=paged_cfg,
            cache_dtype=ttnn.bfloat16,
        )
        page_table = _build_page_table(mesh_device, batch, blocks_per_user)
    else:
        attn = Qwen35Attention(mesh_device, state_dict, args, tt_ccl, create_kv_cache=True)
        page_table = None

    return attn, args, page_table


def _tt_prefill_input(x, mesh_device) -> ttnn.Tensor:
    """Prefill activation layout: [B, seq, dim] host → [B, 1, seq, dim] bf16 DRAM, replicated."""
    batch, seq, dim = x.shape
    return ttnn.from_torch(
        x.to(torch.bfloat16).reshape(batch, 1, seq, dim),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _tt_decode_input(x, mesh_device) -> ttnn.Tensor:
    """Decode activation layout: [B, 1, dim] host → [1, 1, B, dim] bf16 DRAM, replicated."""
    batch, _, dim = x.shape
    return ttnn.from_torch(
        x.to(torch.bfloat16).reshape(1, 1, batch, dim),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _positions_tt(mesh_device, positions) -> ttnn.Tensor:
    """Per-user decode positions [B] (int32) → replicated TT tensor. Drives both the KV-cache update
    index (paged_update_cache) and the current position the decode SDPA attends up to."""
    return ttnn.from_torch(
        positions,
        dtype=ttnn.int32,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _read_prefill_out(out, mesh_device, batch, seq, dim):
    """forward_prefill returns [B, 1, S, dim]; on TP the o_proj reduce-scatter fractures it along the
    hidden dim, so gather dim=3 (no-op on a single device) → [B, S, dim]."""
    if mesh_device.get_num_devices() > 1:
        t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    else:
        t = ttnn.to_torch(out)
    return t.reshape(batch, seq, dim).float()


def _read_decode_out(out, mesh_device, batch, dim):
    """forward_decode returns [1, 1, B, dim]; same hidden-dim gather as prefill → [B, dim]."""
    if mesh_device.get_num_devices() > 1:
        t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    else:
        t = ttnn.to_torch(out)
    return t.reshape(batch, dim).float()


def _prefill_history(attn, hf_attn, args, rope, mesh_device, batch, prefill_len, page_table):
    """Run a prefill so the TT KV cache and a fresh HF DynamicCache hold the same [0, prefill_len)
    history, and return that HF cache for the decode steps to continue from. The prefill output is a
    side issue here (test_attention_prefill covers its PCC) — this only establishes the decode start
    state. Filling via the REAL forward_prefill (not a hand-built cache) keeps the per-device KV
    sharding correct on TP without the test re-deriving the head→device layout."""
    x = torch.randn(batch, prefill_len, args.dim, dtype=torch.float32)

    hf_cache = DynamicCache()
    position_ids = torch.arange(prefill_len).unsqueeze(0).expand(batch, -1)  # [B, prefill_len]
    cos, sin = rope(x, position_ids)
    hf_attn(x, position_embeddings=(cos, sin), attention_mask=causal_mask(prefill_len), past_key_values=hf_cache)

    cos_tt, sin_tt = rot_mats_prefill(mesh_device, args.rope_head_dim, prefill_len, args.rope_theta)
    attn.forward_prefill(_tt_prefill_input(x, mesh_device), cos_tt, sin_tt, page_table=page_table, user_id=0)
    return hf_cache


# Mesh × device-params matrix shared by every test: (1, 1) single device and the (1, 4) TP mesh with
# the 1D fabric the o_proj reduce-scatter rides. trace_region_size is reserved for the trace tests
# (harmless for the eager ones).
_MESH_PARAMS = [
    ((1, 1), {"trace_region_size": TRACE_REGION_SIZE}),
    ((1, 4), {"trace_region_size": TRACE_REGION_SIZE, "fabric_config": ttnn.FabricConfig.FABRIC_1D}),
]


# ── Tests ────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
@pytest.mark.parametrize("paged", [False, True], ids=lambda p: "paged" if p else "contig")
@pytest.mark.parametrize("seq_len", PREFILL_SEQLEN, ids=lambda s: f"seq{s}")
@pytest.mark.parametrize("mesh_device, device_params", _MESH_PARAMS, indirect=True)
@pytest.mark.parametrize("batch", BATCHES, ids=lambda b: f"b{b}")
def test_attention_prefill(mesh_device, batch, seq_len, paged, setup, reset_seeds, ensure_gc):
    """Eager forward_prefill PCC vs HF across batch on (1, 1) and the (1, 4) TP mesh, with the KV
    cache filled through the contiguous (fill_cache) or paged (paged_fill_cache + page table) path."""
    hf_attn, state_dict, rope = setup

    # 1. Build the TT attention + its KV cache (paged or contiguous, sized to the prompt)
    attn, args, page_table = _setup_kv(mesh_device, state_dict, batch, seq_len, paged)

    # 2. Random prefill input + RoPE tables (positions 0..S-1 shared by all users)
    x = torch.randn(batch, seq_len, args.dim, dtype=torch.float32)  # [B, seq, dim]
    x_tt = _tt_prefill_input(x, mesh_device)
    cos_tt, sin_tt = rot_mats_prefill(mesh_device, args.rope_head_dim, seq_len, args.rope_theta)

    # 3. HF golden (eager + explicit causal mask); the DynamicCache it fills is unused here
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)  # [B, seq]
    cos, sin = rope(x, position_ids)
    out_ref, _ = hf_attn(
        x, position_embeddings=(cos, sin), attention_mask=causal_mask(seq_len), past_key_values=DynamicCache()
    )  # [B, seq, dim]

    # 4. TT prefill (also fills the KV cache as a side write — the paged vs contiguous fill path)
    out_tt = attn.forward_prefill(x_tt, cos_tt, sin_tt, page_table=page_table, user_id=0)
    out_tt = _read_prefill_out(out_tt, mesh_device, batch, seq_len, args.dim)

    # 5. Compare reference vs TT output PCC
    passing, pcc = comp_pcc(out_ref, out_tt, PCC)
    logger.info(
        f"attention prefill PCC (mesh={tuple(mesh_device.shape)}, b={batch}, S={seq_len}, paged={paged}) = {pcc}"
    )
    assert passing, f"prefill output PCC too low (b={batch}, S={seq_len}, paged={paged}): {pcc}"


@torch.no_grad()
@pytest.mark.parametrize("paged", [False, True], ids=lambda p: "paged" if p else "contig")
@pytest.mark.parametrize("seq_len", PREFILL_SEQLEN, ids=lambda s: f"seq{s}")
@pytest.mark.parametrize("mesh_device, device_params", _MESH_PARAMS, indirect=True)
@pytest.mark.parametrize("batch", BATCHES, ids=lambda b: f"b{b}")
def test_attention_prefill_trace(mesh_device, batch, seq_len, paged, setup, reset_seeds, ensure_gc):
    """Same prefill PCC as test_attention_prefill, but forward_prefill runs as a captured ttnn trace —
    the path the demo replays to collapse the projections, q/k-norm, partial RoPE, causal SDPA, gate,
    o_proj reduce-scatter and the (paged) KV-cache fill into one device dispatch. forward_prefill
    recomputes its output every call (the cache fill is an idempotent side write), so the replay
    reproduces the eager golden with no state juggling."""
    hf_attn, state_dict, rope = setup

    # 1. Build the TT attention + its KV cache, and the random prefill input + RoPE tables
    attn, args, page_table = _setup_kv(mesh_device, state_dict, batch, seq_len, paged)
    x = torch.randn(batch, seq_len, args.dim, dtype=torch.float32)  # [B, seq, dim]
    x_tt = _tt_prefill_input(x, mesh_device)
    cos_tt, sin_tt = rot_mats_prefill(mesh_device, args.rope_head_dim, seq_len, args.rope_theta)

    # 2. HF golden
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
    cos, sin = rope(x, position_ids)
    out_ref, _ = hf_attn(
        x, position_embeddings=(cos, sin), attention_mask=causal_mask(seq_len), past_key_values=DynamicCache()
    )  # [B, seq, dim]

    # 3. Compile the kernels (trace capture cannot compile), then capture once. `out_tt` is the
    #    persistent buffer the trace writes into.
    attn.forward_prefill(x_tt, cos_tt, sin_tt, page_table=page_table, user_id=0)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out_tt = attn.forward_prefill(x_tt, cos_tt, sin_tt, page_table=page_table, user_id=0)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)

    # 4. Replay recomputes forward_prefill(x) into `out_tt` with a single dispatch, then read it back
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    out_tt = _read_prefill_out(out_tt, mesh_device, batch, seq_len, args.dim)
    ttnn.release_trace(mesh_device, tid)

    # 5. Compare reference vs TT replayed output PCC
    passing, pcc = comp_pcc(out_ref, out_tt, PCC)
    logger.info(
        f"attention prefill TRACE PCC (mesh={tuple(mesh_device.shape)}, b={batch}, S={seq_len}, paged={paged}) = {pcc}"
    )
    assert passing, f"prefill trace PCC too low (b={batch}, S={seq_len}, paged={paged}): {pcc}"


@torch.no_grad()
@pytest.mark.parametrize("paged", [False, True], ids=lambda p: "paged" if p else "contig")
@pytest.mark.parametrize("mesh_device, device_params", _MESH_PARAMS, indirect=True)
@pytest.mark.parametrize("batch", BATCHES, ids=lambda b: f"b{b}")
def test_attention_decode(mesh_device, batch, paged, setup, reset_seeds, ensure_gc):
    """Eager forward_decode PCC vs HF across batch on (1, 1) and the (1, 4) TP mesh. A prefill seeds a
    DECODE_PREFILL_LEN history into both the TT KV cache and the HF DynamicCache, then DECODE_STEPS
    tokens step the whole batch together — exercising the prefill→decode hand-off and (paged) the
    paged_update_cache + paged-SDPA-decode path a pure prefill PCC can't see."""
    hf_attn, state_dict, rope = setup

    # 1. TT attention + cache sized to span the prefill history plus the decode steps
    attn, args, page_table = _setup_kv(mesh_device, state_dict, batch, DECODE_PREFILL_LEN + DECODE_STEPS, paged)

    # 2. Prefill the start state into both caches (forward_prefill fills the TT cache per-device, so
    #    the test needs no manual KV sharding); hf_cache is what the decode steps continue from
    hf_cache = _prefill_history(attn, hf_attn, args, rope, mesh_device, batch, DECODE_PREFILL_LEN, page_table)

    # 3. Step the batch through DECODE_STEPS tokens, comparing the HF vs TT output PCC each step
    for step in range(DECODE_STEPS):
        pos = DECODE_PREFILL_LEN + step  # 0-based: prefill filled 0..L-1, so the first decode token sits at L
        x = torch.randn(batch, 1, args.dim, dtype=torch.float32)

        # Reference decode step (advances hf_cache every step)
        cos_d, sin_d = rope(x, torch.full((batch, 1), pos, dtype=torch.long))
        ref, _ = hf_attn(x, position_embeddings=(cos_d, sin_d), attention_mask=None, past_key_values=hf_cache)
        ref = ref[:, 0]  # [B, dim]

        # TT decode step: per-user positions drive both the cache update index and the decode RoPE
        positions = torch.full((batch,), pos, dtype=torch.int32)
        cos_dt, sin_dt = rot_mats_decode(mesh_device, args.rope_head_dim, args.max_seq_len, args.rope_theta, positions)
        out_tt = attn.forward_decode(
            _tt_decode_input(x, mesh_device),
            _positions_tt(mesh_device, positions),
            cos_dt,
            sin_dt,
            page_table=page_table,
        )
        out_tt = _read_decode_out(out_tt, mesh_device, batch, args.dim)

        passing, pcc = comp_pcc(ref, out_tt, PCC)
        logger.info(
            f"attention decode step {step} PCC (mesh={tuple(mesh_device.shape)}, b={batch}, paged={paged}) = {pcc}"
        )
        assert passing, f"decode PCC too low (b={batch}, step={step}, paged={paged}): {pcc}"


@torch.no_grad()
@pytest.mark.parametrize("paged", [False, True], ids=lambda p: "paged" if p else "contig")
@pytest.mark.parametrize("mesh_device, device_params", _MESH_PARAMS, indirect=True)
@pytest.mark.parametrize("batch", BATCHES, ids=lambda b: f"b{b}")
def test_attention_decode_trace(mesh_device, batch, paged, setup, reset_seeds, ensure_gc):
    """Same decode PCC as test_attention_decode, but forward_decode runs as a captured ttnn trace —
    the path the demo replays each token (it captures the whole-model decode, so the full-attention
    layers run this exact graph). One step is captured and replayed.

    Unlike the GDN decode trace, NO start-state restore is needed: the KV history is append-only and
    this single step writes a fresh position (cur_pos = L). The compile and capture calls write that
    same slot, and the replay overwrites it once more, so the history the replayed step actually reads
    (0..L-1) is still the untouched prefill state — the trace bakes in the buffer addresses, and the
    contents at those addresses are already correct at replay time."""
    hf_attn, state_dict, rope = setup

    # 1. TT attention + cache, and the prefill start state in both caches
    attn, args, page_table = _setup_kv(mesh_device, state_dict, batch, DECODE_PREFILL_LEN + DECODE_STEPS, paged)
    hf_cache = _prefill_history(attn, hf_attn, args, rope, mesh_device, batch, DECODE_PREFILL_LEN, page_table)

    # 2. One decode token at pos = L (the first position past the prefill history)
    pos = DECODE_PREFILL_LEN
    x = torch.randn(batch, 1, args.dim, dtype=torch.float32)
    x_tt = _tt_decode_input(x, mesh_device)
    positions = torch.full((batch,), pos, dtype=torch.int32)
    pos_tt = _positions_tt(mesh_device, positions)
    cos_dt, sin_dt = rot_mats_decode(mesh_device, args.rope_head_dim, args.max_seq_len, args.rope_theta, positions)

    # 3. HF golden: one decode step from the prefill history
    cos_d, sin_d = rope(x, torch.full((batch, 1), pos, dtype=torch.long))
    out_ref, _ = hf_attn(x, position_embeddings=(cos_d, sin_d), attention_mask=None, past_key_values=hf_cache)
    out_ref = out_ref[:, 0]  # [B, dim]

    # 4. Compile the decode kernels (capture cannot compile; this writes the cache at pos L — fine,
    #    the replay rewrites it), then capture once. `out_tt` is the persistent buffer the trace writes.
    attn.forward_decode(x_tt, pos_tt, cos_dt, sin_dt, page_table=page_table)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out_tt = attn.forward_decode(x_tt, pos_tt, cos_dt, sin_dt, page_table=page_table)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)

    # 5. Replay the single decode step into `out_tt` and read it back (history 0..L-1 is still S0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    out_tt = _read_decode_out(out_tt, mesh_device, batch, args.dim)
    ttnn.release_trace(mesh_device, tid)

    # 6. Compare reference vs TT replayed output PCC
    passing, pcc = comp_pcc(out_ref, out_tt, PCC)
    logger.info(f"attention decode TRACE PCC (mesh={tuple(mesh_device.shape)}, b={batch}, paged={paged}) = {pcc}")
    assert passing, f"decode trace PCC too low (b={batch}, paged={paged}): {pcc}"
