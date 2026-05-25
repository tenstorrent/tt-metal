# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Sanity tests handed off to the ring_joint_scaled_dot_product_attention owners.

Four small tests built around MLA-shaped inputs, with the goal of teasing out
what the op already supports vs. what we'd like it to support cleanly. Each
test is intentionally tiny (seq_len just a few tiles) so the answer comes
back fast.

Test scope:
1. ND-sharded KV cache as K input directly (no slice workaround).
2. Index-based K access — desired API showcase in torch.
3. Per-chunk persistent-buffer reallocation — TODO, needs design input.
4. ISL smaller than chunk_size — TODO, needs design input.

MLA-shaped knobs used throughout:
  - nhk == 1 (single shared K head broadcast across Q heads)
  - kv_lora_rank=512, qk_rope_head_dim=64  -> kvpe_dim = 576 (K head dim)
  - qk_nope_head_dim=128, qk_rope_head_dim=64 -> qk_head_dim = 192 (Q head dim)
  - v_head_dim = 128
  - kv_dtype = bfloat8_b (matches the on-device KVPE cache)
  - q_dtype  = bfloat16
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    BH_NUM_DRAM_BANKS,
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    init_kvpe_cache,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


def _nd_shard_memory_config(head_dim):
    """ND-sharded DRAM memory config matching init_kvpe_cache's layout but with
    a configurable head dim. Kept here so owners can easily probe variants
    (e.g. ND-shard V with head_dim=V_HEAD_DIM=128) — see the cascade in the
    test docstring below."""
    core_ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0)) for bank_id in range(BH_NUM_DRAM_BANKS)
    ]
    grid = ttnn.CoreRangeSet(core_ranges)
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim],
        grid=grid,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd_shard_spec)


# ---------------------------------------------------------------------------
# MLA-shaped constants used by all tests in this file
#
# At the SDPA op call site (after the wkv_b1 absorption that lifts Q's nope
# part from qk_nope_head_dim=128 into kv_lora_rank=512), the op sees:
#   d_q == d_k == KVPE_DIM = kv_lora_rank + qk_rope_head_dim = 576
#   d_v == V_HEAD_DIM = 128
# The "raw" qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 192 is only
# used to compute the attention scale (MLA convention).
# ---------------------------------------------------------------------------
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 192 — only used for scale
V_HEAD_DIM = 128
KVPE_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576 — what the op actually sees


def _mla_sdpa_reference(q, k, v, scale):
    """
    Causal MLA-style SDPA reference, all-torch.

    Shapes at the op call site:
      q: [b, nhq, sq, KVPE_DIM]
      k: [b, 1,   sq, KVPE_DIM]   (single shared K head broadcast across Q heads)
      v: [b, nhq, sq, V_HEAD_DIM]
    Returns [b, nhq, sq, V_HEAD_DIM].
    """
    nhq = q.shape[1]
    k = k.expand(-1, nhq, -1, -1)
    sq = q.shape[2]
    attn_scores = (q.float() @ k.transpose(-2, -1).float()) * scale
    mask = torch.triu(torch.ones(sq, sq, dtype=torch.bool), diagonal=1)
    attn_scores = attn_scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(attn_scores, dim=-1)
    return (attn @ v.float()).to(q.dtype)


def _make_program_config(mesh_device):
    """Smallest valid SDPA program config: 32x32 chunks on the compute grid."""
    grid = mesh_device.compute_with_storage_grid_size()
    sdpa_compute_grid = (grid.x - 1, grid.y)  # last column reserved for CCL
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=sdpa_compute_grid,
        q_chunk_size=32,
        k_chunk_size=32,
        exp_approx_mode=False,
    )


def _topology_from(device_params):
    fabric_config = device_params.get("fabric_config", ttnn.FabricConfig.FABRIC_1D)
    return ttnn.Topology.Ring if fabric_config == ttnn.FabricConfig.FABRIC_1D_RING else ttnn.Topology.Linear


# ===========================================================================
# Test 1: ND-sharded KV cache as K input directly (no ttnn.slice workaround)
# ===========================================================================
#
# Motivation
# ----------
# In the chunked MLA branch (ipotkonjak/chunked_ring_sdpa_testing), the K
# input to ring_joint_scaled_dot_product_attention is built like this:
#
#     ttnn.kv_cache.fill_cache_for_user_(kvpe_cache, tt_kvpe, cache_batch_idx,
#                                        update_idx=local_offset)
#     tt_k_populated = ttnn.slice(
#         kvpe_cache,
#         [cache_batch_idx, 0, 0, 0],
#         [cache_batch_idx + 1, 1, populated_local, kvpe_dim],
#         memory_config=ttnn.DRAM_MEMORY_CONFIG,
#     )
#     # tt_k_populated is now interleaved DRAM — ND-shard layout is hidden away.
#     attn_out = ring_joint_sdpa(..., tt_k_populated, ...)
#
# The slice does two things at once: (a) selects one cache slot from the
# user-major batch dim, and (b) reinterprets the ND-sharded cache as plain
# interleaved DRAM. (b) is the layout-coupling we'd like to drop — it
# forces the caller to know the on-device shard format. This test asks:
# does the op accept the ND-sharded cache directly?
#
# Setup
# -----
# Cache layout (from models/demos/deepseek_v3_d_p/utils/kv_cache_utils.py
# init_kvpe_cache; this is the production layout for the MLA KVPE cache):
#   shard_shape  = [1, 1, 32, kvpe_dim=576]   (32 tokens per shard)
#   grid         = 8 DRAM banks ([(b,0) for b in 0..7])
#   distribution = ROUND_ROBIN_1D, ROW_MAJOR
#   buffer_type  = DRAM
#   dtype        = bfloat8_b
#   layout       = TILE_LAYOUT
#   mesh_mapper  = ReplicateTensorToMesh (per-device data via fill_cache_for_user_)
#
# Mesh = (2,4) on BH: sp=2, tp=4. num_heads=16 (4/device). seq_len=128
# (seq_len_local=64 = 2 ND-shards/device → exercises 2 of the 8 banks).
# Single-shot full prefill: write the whole local slab to cache slot 0,
# then pass tt_cache directly as K to the op. logical_n = full seq_len.
#
# Findings (full cascade — what owners would hit if they probe the same)
# ----------------------------------------------------------------------
# Run as-is on unmodified main → fatal #1 fires. Subsequent fatals were
# uncovered by commenting out the firing check and rebuilding. The cascade:
#
#   Fatal #1  (file: ring_attention_all_gather_async_device_operation.cpp:41)
#       TT_FATAL(input_tensor.memory_config() == memory_config, ...)
#       The AG sub-op gathers K and V *together* and requires all input
#       tensors to share memory_config. K=ND-shard(head_dim=576),
#       V=interleaved → mismatch. Reported as "Input tensor 1 has different
#       memory config".
#
#       Probe: ND-shard V too via _nd_shard_memory_config(V_HEAD_DIM). Did
#       NOT pass the check, because V's spec uses head_dim=128 (its natural
#       shape) which yields a different MemoryConfig than K's head_dim=576
#       spec. Equality is strict — no "compatible layout family" notion.
#       K and V can only share a memory_config if their shard specs are
#       byte-identical, which is impossible when their head dims differ.
#
#   Fatal #2  (file: ring_attention_all_gather_async_device_operation.cpp:52)
#       TT_FATAL(memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED, ...)
#       The first input's memory_layout is required to be INTERLEAVED.
#       ND-shard is rejected outright.
#
#   Fatal #3  (file: ring_attention_all_gather_async_device_operation.cpp:84)
#       TT_FATAL(output_tensor.memory_config() == operation_attributes.output_mem_config, ...)
#       The persistent_output_buffer_k / persistent_output_buffer_v
#       provided by the caller must match the op-attribute
#       output_mem_config — which is inherited from the first input's
#       memory_config (see compute_output_specs at lines ~110-115 of the
#       same file). So with ND-shard input, the op expects ND-shard output
#       buffers; ours were interleaved → mismatch.
#
#   Wall #4  (file: ttnn/cpp/ttnn/operations/transformer/sdpa/device/
#                  kernels/dataflow/ring_joint_reader.cpp:408 and :482)
#       Kernel JIT compile fails (riscv-tt-elf-g++):
#
#         "operands to '?:' have different types
#          'const PaddedAddrGenerator<TensorAccessor<DistributionSpec<1,8,...>>>'
#          and
#          'const PaddedAddrGenerator<TensorAccessor<DistributionSpec<0,0,...>>>'"
#
#       The reader has:
#           return ring_iter == 0 ? local_k_generator : gathered_k_generator;
#           return ring_iter == 0 ? local_v_generator : gathered_v_generator;
#
#       i.e. on ring iter 0 it reads K/V from the *local input* tensor; on
#       later iters it reads from the *gathered* (persistent output) tensor.
#       Both sides of the ternary must be the same C++ type → input and
#       gathered tensors must share their TensorAccessor distribution spec.
#       Mixing ND-shard input + interleaved gathered buffer makes the
#       template instantiations diverge and the ternary can't unify.
#
# Current implementation note
# ---------------------------
# RingJointSDPA now keeps the standalone AG op's validation unchanged, but
# uses a RingJoint-specific fused-AG validation and reader dispatch. This
# allows local K/V inputs to use layouts that differ from the persistent
# gathered K/V buffers, including ND-sharded local K and ND-sharded local V
# with different natural head dims.
#
# How to reproduce this cascade
# -----------------------------
# 1. Run the test as-is → fatal #1 at line 41.
# 2. To reach fatal #2: in the test, swap V's memory_config to
#    _nd_shard_memory_config(V_HEAD_DIM) (helper at top of this file) —
#    fatal #1 fires anyway (different shard spec), but if you also comment
#    out the line-41 check in the AG validate, you reach the line-52 check.
# 3. To reach fatal #3: comment out line-41 AND line-52, rebuild → line-84.
# 4. To reach wall #4: comment out lines 41, 52, AND 84, rebuild → kernel
#    JIT compile fails on ring_joint_reader.cpp:408 and :482.
# ===========================================================================


@pytest.mark.parametrize("mesh_device", [(2, 2), (2, 4)], ids=["2x2", "2x4"], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.parametrize("v_memory_layout", ["interleaved", "nd_sharded"], ids=["v_interleaved", "v_nd_sharded"])
@pytest.mark.timeout(0)
def test_nd_sharded_kv_cache_as_k(mesh_device, device_params, v_memory_layout):
    sp_axis = 0
    tp_axis = 1
    mesh_shape = list(mesh_device.shape)
    sp = mesh_shape[sp_axis]
    tp = mesh_shape[tp_axis]

    # num_heads chosen so num_heads_local >= tp (joint_q/v sharding on the
    # head dim across tp_axis must divide cleanly).
    num_heads = tp * tp  # = 16 for (2,4): 4 heads per TP device
    num_heads_local = num_heads // tp
    seq_len_local = 64  # 2 ND shards per device (NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK=32)
    seq_len = seq_len_local * sp

    scale = QK_HEAD_DIM**-0.5
    topology = _topology_from(device_params)

    torch.manual_seed(0)
    q_host = torch.randn(1, num_heads, seq_len, KVPE_DIM, dtype=torch.bfloat16)
    kvpe_host = torch.randn(1, 1, seq_len, KVPE_DIM, dtype=torch.bfloat16)
    v_host = torch.randn(1, num_heads, seq_len, V_HEAD_DIM, dtype=torch.bfloat16)

    # CCL setup
    tt_ccl = get_tt_ccl(mesh_device)

    # ND-sharded KVPE cache, batch=1 (single user / single layer)
    tt_cache = init_kvpe_cache(
        kvpe_cache_head_dim=KVPE_DIM,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
    )

    # Upload K data and fill cache slot 0
    kvpe_shard_dims = [None, None]
    kvpe_shard_dims[sp_axis] = 2
    tt_kvpe = ttnn.from_torch(
        kvpe_host,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=kvpe_shard_dims),
    )
    ttnn.kv_cache.fill_cache_for_user_(tt_cache, tt_kvpe, 0)

    # Upload Q (sharded on sp seq + tp heads)
    q_shard_dims = [None, None]
    q_shard_dims[sp_axis] = 2
    q_shard_dims[tp_axis] = 1
    tt_q = ttnn.from_torch(
        q_host,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=q_shard_dims),
    )

    # Upload V (sharded on sp seq + tp heads). The nd_sharded case exercises
    # direct local-V reads with a natural V_HEAD_DIM shard spec.
    v_shard_dims = [None, None]
    v_shard_dims[sp_axis] = 2
    v_shard_dims[tp_axis] = 1
    v_memory_config = (
        _nd_shard_memory_config(V_HEAD_DIM) if v_memory_layout == "nd_sharded" else ttnn.DRAM_MEMORY_CONFIG
    )
    tt_v = ttnn.from_torch(
        v_host,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=v_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=v_shard_dims),
    )

    # Empty joint tensors (joint_strategy="rear" with seq_len=0 disables joint)
    joint_shard_dims = [None, None]
    joint_shard_dims[tp_axis] = 1
    joint_q = ttnn.from_torch(
        torch.zeros(1, num_heads_local, 0, KVPE_DIM),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=joint_shard_dims),
    )
    joint_kv = ttnn.from_torch(
        torch.zeros(1, 1, 0, KVPE_DIM),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    joint_v = ttnn.from_torch(
        torch.zeros(1, num_heads_local, 0, V_HEAD_DIM),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=joint_shard_dims),
    )

    # Persistent AG output buffers sized to the full seq_len (single-shot equivalent).
    persistent_k_buf = ttnn.from_torch(
        torch.zeros(1, 1, seq_len, KVPE_DIM),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[None, None]),
    )
    persistent_v_shard_dims = [None, None]
    persistent_v_shard_dims[tp_axis] = 1
    persistent_v_buf = ttnn.from_torch(
        torch.zeros(1, num_heads, seq_len, V_HEAD_DIM),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, mesh_shape=tuple(mesh_device.shape), dims=persistent_v_shard_dims
        ),
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    logger.info(
        f"Calling ring_joint_sdpa with ND-sharded cache as K. "
        f"K shape={list(tt_cache.shape)}, K memory_config={tt_cache.memory_config()}, "
        f"V memory layout={v_memory_layout}, V memory_config={tt_v.memory_config()}"
    )

    # THE EXPERIMENT: pass the ND-sharded cache directly as K, no ttnn.slice.
    attn_out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q,
        tt_cache,  # <-- ND-sharded K, no slice workaround
        tt_v,
        joint_q,
        joint_kv,
        joint_v,
        persistent_output_buffer_k=persistent_k_buf,
        persistent_output_buffer_v=persistent_v_buf,
        joint_strategy="rear",
        logical_n=seq_len,
        program_config=_make_program_config(mesh_device),
        compute_kernel_config=compute_kernel_config,
        dim=2,
        multi_device_global_semaphore=tt_ccl.ring_attention_ccl_semaphore_handles,
        num_links=1,
        cluster_axis=sp_axis,
        mesh_device=mesh_device,
        topology=topology,
        subdevice_id=tt_ccl.worker_sub_device_id,
        ccl_core_grid_offset=tt_ccl.ring_attention_ccl_core_grid_offset,
        use_column_major_ccl=True,
        is_causal=True,
        scale=scale,
        is_balanced=False,
    )

    logger.success(f"Op accepted ND-sharded KV cache as K input with {v_memory_layout} V.")

    # Gather output and verify correctness against a torch reference.
    out_concat_dims = [None, None]
    out_concat_dims[tp_axis] = 1
    out_concat_dims[sp_axis] = 2
    tt_out_host = ttnn.to_torch(
        attn_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=out_concat_dims, mesh_shape=mesh_device.shape),
    ).to(torch.bfloat16)

    ref = _mla_sdpa_reference(q_host, kvpe_host, v_host, scale)
    _, pcc_msg = assert_with_pcc(ref, tt_out_host, 0.97)
    logger.info(f"ND-sharded K with {v_memory_layout} V PCC vs torch reference: {pcc_msg}")


# ===========================================================================
# Test 2: Index-based K access — torch showcase of the desired op API
# ===========================================================================
#
# The chunked MLA branch currently feeds K to ring_joint_sdpa via a slice:
#
#     cache_batch_idx = cache_user_id * num_cache_layers + cache_layer_idx
#     ttnn.kv_cache.fill_cache_for_user_(kvpe_cache, tt_kvpe, cache_batch_idx,
#                                        update_idx=local_offset)
#     tt_k_populated = ttnn.slice(
#         kvpe_cache,
#         [cache_batch_idx, 0, 0, 0],
#         [cache_batch_idx + 1, 1, populated_local, kvpe_dim],
#         memory_config=ttnn.DRAM_MEMORY_CONFIG,
#     )
#     attn = ring_joint_sdpa(tt_q, tt_k_populated, tt_v_populated, ...)
#
# The slice does two things at once: (a) selects one batch slot from a
# user-major cache layout, and (b) reinterprets the ND-sharded cache as plain
# interleaved DRAM. (b) is the layout-coupling that breaks the cache layout
# abstraction and forces the caller to know the on-device shard format.
#
# Desired op API — pass the whole cache plus an integer slot index and a
# populated-length, let the op do the equivalent select+trim internally
# without changing the underlying buffer:
#
#     attn = ring_joint_sdpa(
#         tt_q,
#         kvpe_cache,                      # <-- whole cache, any layout
#         tt_v_populated,
#         ...,
#         cache_batch_idx=cache_batch_idx, # NEW: which user/layer slot
#         n_local_kv=populated_local,      # NEW: how much of that slot is populated
#         ...
#     )
#
# This is a pure-torch test that mocks both call shapes against the same
# inputs and asserts they produce identical output. It serves as the spec
# for what the op-side change should mean mathematically — no device, no
# layout, just call ergonomics.
# ===========================================================================


def _causal_chunked_mla_sdpa_torch(q, k, v, q_abs_offset, scale):
    """
    Chunked causal MLA SDPA in torch.

    Q is one chunk (rows [q_abs_offset, q_abs_offset + sq)); K/V cover the
    full populated prefix [0, sk). Mask: Q-row i attends to K-cols [0, q_abs_offset + i].

      q: [b, nhq, sq, KVPE_DIM]
      k: [b, 1,   sk, KVPE_DIM]   (single shared K head, broadcast)
      v: [b, nhq, sk, V_HEAD_DIM]
    Returns [b, nhq, sq, V_HEAD_DIM].
    """
    nhq = q.shape[1]
    k = k.expand(-1, nhq, -1, -1)
    sq = q.shape[2]
    sk = k.shape[2]
    attn_scores = (q.float() @ k.transpose(-2, -1).float()) * scale
    q_pos = torch.arange(sq) + q_abs_offset
    k_pos = torch.arange(sk)
    mask = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)  # [sq, sk]
    attn_scores = attn_scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(attn_scores, dim=-1)
    return (attn @ v.float()).to(q.dtype)


def _ring_joint_sdpa_torch_with_slice(q, k_input, v, q_abs_offset, scale):
    """
    Current API in torch: caller pre-slices the cache to produce k_input.
    The op just runs SDPA on what it's given.
    """
    return _causal_chunked_mla_sdpa_torch(q, k_input, v, q_abs_offset, scale)


def _to_balanced_growing(t, sp, chunk_size_local, n_chunks):
    """
    Reorder along dim 2 from natural (chunked-then-chip-contiguous) order
    into the "balanced growing" layout the ring_joint_sdpa chunked path
    expects after a naive SP shard.

    Natural layout:  [chunk0 chip0 | chunk0 chip1 | chunk1 chip0 | chunk1 chip1 | ...]
    Balanced layout: [chunk0 chip0 | chunk1 chip0 | chunk0 chip1 | chunk1 chip1 | ...]

    The op's kv_global_tile_for_local() decodes chip-r local tile t as global pos
        (t / slab_tiles) * chunk_size_global + r * slab_tiles + (t % slab_tiles)
    so chip r's local [j*chunk_size_local, (j+1)*chunk_size_local) must hold
    global [j*chunk_size_global + r*chunk_size_local, ... + chunk_size_local).
    After this reorder, ShardTensor2dMesh splitting dim 2 across SP cleanly
    gives each chip the expected slabs.
    """
    out = torch.empty_like(t)
    populated_local = n_chunks * chunk_size_local
    chunk_size_global = chunk_size_local * sp
    for chip in range(sp):
        for chunk in range(n_chunks):
            src = chunk * chunk_size_global + chip * chunk_size_local
            dst = chip * populated_local + chunk * chunk_size_local
            out[..., dst : dst + chunk_size_local, :] = t[..., src : src + chunk_size_local, :]
    return out


def _ring_joint_sdpa_torch_with_cache_index(q, kvpe_cache, v, *, cache_batch_idx, n_local_kv, q_abs_offset, scale):
    """
    Desired API in torch: caller passes the whole cache + (cache_batch_idx,
    n_local_kv); the op does the slot select + populated-prefix trim
    internally. On-device this would be a layout-free view, not a copy.

      kvpe_cache: [num_users * num_cache_layers, 1, seq_len, KVPE_DIM]
    """
    k_view = kvpe_cache[cache_batch_idx : cache_batch_idx + 1, :, :n_local_kv, :]
    return _causal_chunked_mla_sdpa_torch(q, k_view, v, q_abs_offset, scale)


def test_index_based_k_access_torch_showcase():
    # Multi-user, multi-layer cache (user-major layout):
    #   cache_batch_idx = cache_user_id * num_cache_layers + cache_layer_idx
    num_users = 3
    num_cache_layers = 2
    cache_batch = num_users * num_cache_layers  # = 6 slots

    seq_len = 256
    chunk_size = 64
    n_chunks = seq_len // chunk_size
    nhq = 8  # arbitrary — torch only, no device sharding constraints

    # Exercise a non-trivial (user, layer) so we can see the index math matter:
    # cache_batch_idx = 2 * 2 + 1 = 5  (last user, second layer)
    cache_user_id = 2
    cache_layer_idx = 1
    cache_batch_idx = cache_user_id * num_cache_layers + cache_layer_idx
    assert cache_batch_idx == 5, "index math should be user-major"

    scale = QK_HEAD_DIM**-0.5

    torch.manual_seed(7)
    q_full = torch.randn(1, nhq, seq_len, KVPE_DIM, dtype=torch.bfloat16)
    k_full = torch.randn(1, 1, seq_len, KVPE_DIM, dtype=torch.bfloat16)
    v_full = torch.randn(1, nhq, seq_len, V_HEAD_DIM, dtype=torch.bfloat16)

    # Build the multi-user cache. Slot 5 holds k_full for "our" user/layer;
    # other slots hold distinct random data so a wrong index would show up
    # as a PCC failure rather than a coincidental pass.
    cache = torch.empty(cache_batch, 1, seq_len, KVPE_DIM, dtype=torch.bfloat16)
    for i in range(cache_batch):
        if i == cache_batch_idx:
            cache[i] = k_full[0]
        else:
            cache[i] = torch.randn(1, seq_len, KVPE_DIM, dtype=torch.bfloat16) * 100  # noise

    # Pick the middle chunk so neither approach trivially passes (chunk 0
    # has populated_local == chunk_size; last chunk has populated_local == seq_len).
    chunk_i = n_chunks // 2  # = 2 for n_chunks=4
    q_abs_offset = chunk_i * chunk_size
    populated_local = q_abs_offset + chunk_size  # cache populated up to end of this chunk

    q_chunk = q_full[:, :, q_abs_offset : q_abs_offset + chunk_size, :]
    v_populated = v_full[:, :, :populated_local, :]

    # ----- Approach A: current API (pre-slice K from the cache) -----
    k_pre_sliced = cache[cache_batch_idx : cache_batch_idx + 1, :, :populated_local, :]
    out_with_slice = _ring_joint_sdpa_torch_with_slice(q_chunk, k_pre_sliced, v_populated, q_abs_offset, scale)

    # ----- Approach B: desired API (cache + index + n_local_kv) -----
    out_with_index = _ring_joint_sdpa_torch_with_cache_index(
        q_chunk,
        cache,
        v_populated,
        cache_batch_idx=cache_batch_idx,
        n_local_kv=populated_local,
        q_abs_offset=q_abs_offset,
        scale=scale,
    )

    # The two approaches must produce bit-identical output: same math, same
    # data — only the caller-side wiring differs.
    torch.testing.assert_close(out_with_slice, out_with_index, rtol=0, atol=0)
    logger.success(
        f"Index API matches slice API. cache_batch_idx={cache_batch_idx} "
        f"(user={cache_user_id}, layer={cache_layer_idx}), "
        f"populated_local={populated_local}, q_abs_offset={q_abs_offset}"
    )


# ===========================================================================
# Test 3: Persistent buffer size constraint (per-chunk realloc as it stands)
# ===========================================================================
#
# Motivation
# ----------
# In the chunked MLA branch (ipotkonjak/chunked_ring_sdpa_testing), the
# caller allocates fresh persistent K/V output buffers inside forward()
# every chunk, sized to that chunk's chunk_end_global:
#
#     persistent_k_buf = ttnn.from_torch(
#         torch.zeros(1, 1, chunk_end_global, kvpe_dim), ...
#     )
#     persistent_v_buf = ttnn.from_torch(
#         torch.zeros(1, self.num_heads, chunk_end_global, self.v_head_dim), ...
#     )
#
# Per-chunk allocation is wasted host work + device memory churn, and
# also forks the SDPA program cache (program hash depends on buffer
# shape). The non-chunked single-shot path doesn't have this problem —
# it allocates one max-sized buffer at __init__ and reuses it.
#
# We'd like the chunked path to behave the same way: allocate one
# max-sized buffer at __init__, reuse it every chunk, let `logical_n`
# bound the actual data movement.
#
# What stops us
# -------------
# Two validation checks enforce strict equality between the persistent
# buffer's seq dim and N_local_kv * ring_size for the current call:
#
#   (A) ring_attention_all_gather_async_device_operation.cpp:88-93
#       TT_FATAL(output_shape == expected_output_shape, ...)
#       where expected_output_shape[dim] = input_shape[dim] * ring_size.
#
#   (B) ring_joint_sdpa_device_operation.cpp:192
#       TT_FATAL(N_global == N_local_kv * args.ring_size, ...)
#       where N_global = persistent_buffer_k.shape[2],
#             N_local_kv = input_k.shape[2].
#
# Is the strict equality a *real* constraint?
# -------------------------------------------
# We traced through both the AG writer and the SDPA reader, and the
# answer is NO. Both kernels derive their strides from the buffer's
# *actual* logical_shape (read consistently on both sides), and their
# iteration bounds come from input tensor shapes (not buffer shape):
#
#   - AG writer (ring_attention_all_gather_writer.cpp:115-185):
#       per-head jump = output_tensor_Wt * output_tensor_Ht (line 180),
#       row stride = output_tensor_Wt (line 139). Both are read from
#       output_tensor[i].padded_shape() in the program factory
#       (ring_attention_all_gather_async_multi_core_with_workers_program_factory.cpp:414).
#       Loop bound = input_tile_id_end, derived from input tensor
#       page count (program factory line 412), independent of output.
#
#   - SDPA reader (ring_joint_reader.cpp):
#       gathered_k accessor = TensorTileShape(B, NHK, padded_Nt, DHt)
#       where padded_Nt = gathered_k.logical_shape()[2] / TILE_HEIGHT
#       (line 177, 183, 214). Same buffer-shape source as the writer.
#       Iteration uses kv_local_padded_Nt (from input_k shape, line 176)
#       and logical_nt; "K chunk beyond logical_n" is skipped at
#       line 352-356.
#
# Both kernels use the same buffer-shape-derived strides → they agree
# on which tile lives at which DRAM address. Both iterate only over the
# valid prefix (ring_size * N_local_kv tiles per batch*head). Oversize:
# the trailing tiles per head are untouched by writer, unread by reader.
# Bit-correct.
#
# This holds for both K (NHK=1) and V (NHV=multi): the per-head stride
# is computed once from the buffer's actual shape and used identically
# by writer and reader.
#
# So checks (A) and (B) are pure gates — they could be `>=` instead of
# `==` and the kernels would handle it as-is.
#
# What this test does
# -------------------
# A chunked-prefill ring_joint_sdpa call mirroring the chunked MLA branch's
# usage pattern (Q = one current chunk, K/V = populated prefix covering
# multiple chunks), invoked twice:
#
#   Call A: persistent K/V sized to populated_global (= N_local_kv * sp).
#     Expected: succeeds. PCC vs torch chunked-causal MLA-SDPA reference asserted.
#
#   Call B: persistent K/V sized to seq_len_max (= 4x oversize here).
#     Expected today: RuntimeError at AG :93 (the AG sub-op validates
#       its output shape against `input_shape[dim] * ring_size` before
#       SDPA's outer :192 check ever runs; both encode the same rule).
#     Expected after relaxing BOTH validates (AG :88-93 and SDPA :192
#     from `==` to `>=` on the gather dim): same output as Call A.
#
# K/V are pre-shuffled into the op's "balanced growing" per-chip layout
# (see _to_balanced_growing) so naive SP sharding lands each chip's slabs
# at the global positions kv_global_tile_for_local expects. Q is a single
# chunk so its naive SP shard already matches the op's expectation.
#
# What we tried and the relaxations we applied
# ---------------------------------------------
# Running this test on unmodified main hits Fatal #1 (AG :93). We then
# probed the cascade by commenting out one validate at a time, rebuilding,
# and rerunning. Each layer relaxed surfaced the next layer:
#
#   1. Relaxed AG :88-93 (output_shape == expected) -> hit SDPA :192 next.
#      "Gathered K seq length must equal per-device K shard times ring size".
#
#   2. Relaxed SDPA :192 (N_global == N_local_kv * ring_size) -> op ran
#      end-to-end. No further validation, no kernel-JIT, no runtime errors.
#
# Concrete diffs (currently held locally on this branch; not yet committed):
#
#   ring_attention_all_gather_async_device_operation.cpp:88-93
#     Replaced the single `output_shape == expected_output_shape` assert
#     with a per-dim loop:
#       - gather dim: `output_shape[d] >= expected_output_shape[d]` (oversize OK)
#       - other dims: `output_shape[d] == expected_output_shape[d]` (unchanged)
#
#   ring_joint_sdpa_device_operation.cpp:192
#     `N_global == N_local_kv * args.ring_size`  ->  `N_global >= ...`
#
# Both changes are pure validation loosening — no kernel code touched, no
# program-factory changes, no host-side derivation changes.
#
# Outcome (with both relaxations in place)
# ----------------------------------------
#   - Call A (exact-size buffer, seq = N_local_kv * sp = 128):
#       passes; PCC vs torch chunked-causal MLA-SDPA reference = 0.9994.
#   - Call B (max-size buffer, seq = 256, 2x oversize):
#       passes; PCC vs torch = 0.9994; PCC vs Call A = 1.0 (bit-identical).
#   - test_persistent_buffer_reuse_across_chunks (below) runs 4 chunked
#     iterations passing the SAME pair of max-sized buffers to every op
#     call; all four pass PCC >= 0.99 against their per-chunk reference.
#
# This confirms the kernel-level trace was right: nothing in the AG writer
# or SDPA reader requires the buffer to be exactly N_local_kv * ring_size.
# Both use the buffer's actual logical_shape consistently for strides, and
# iteration bounds come from input K shape + logical_n, not from buffer
# shape. Oversize leaves dead trailing tiles per batch*head that neither
# side touches.
#
# Payoff for chunked-MLA: one pair of (seq_len_max-sized) persistent K/V
# buffers allocated at __init__ and reused for every chunked op call —
# mirroring the single-shot path's __init__-time allocation (mla.py:304-324).
# No per-chunk realloc. (Program cache still forks per chunk because the
# SDPA program hash depends on input K shape and logical_n; that's
# orthogonal — the win here is host alloc + device memory churn.)
# ===========================================================================


@pytest.mark.parametrize("mesh_device", [(2, 2), (2, 4)], ids=["2x2", "2x4"], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.timeout(0)
def test_persistent_buffer_size_constraint(mesh_device, device_params):
    sp_axis = 0
    tp_axis = 1
    mesh_shape = list(mesh_device.shape)
    sp = mesh_shape[sp_axis]
    tp = mesh_shape[tp_axis]

    # num_heads chosen so num_heads_local >= tp (joint_q/v sharding on the
    # head dim across tp_axis must divide cleanly). Same convention as Test 1.
    num_heads = tp * tp
    num_heads_local = num_heads // tp

    # Chunked-prefill: Q is the current chunk (one chunk's tokens),
    # K/V are the populated prefix (multiple chunks). seq_len_max is the
    # max prefill horizon we'd allocate at __init__ in the desired API.
    chunk_size_local = 32  # 1 tile per chunk per chip
    chunk_size_global = chunk_size_local * sp
    n_chunks_total = 4
    n_chunks_populated = 2  # processing chunk 1 (0-indexed)
    seq_len_max = n_chunks_total * chunk_size_global
    populated_global = n_chunks_populated * chunk_size_global

    chunk_idx = n_chunks_populated - 1  # current chunk index
    q_abs_offset = chunk_idx * chunk_size_global  # global start of current Q chunk

    scale = QK_HEAD_DIM**-0.5
    topology = _topology_from(device_params)

    torch.manual_seed(0)
    # Q: just the current chunk's tokens, in natural order. For a single chunk,
    # naive SP sharding already matches the op's per-chip expectation.
    q_host = torch.randn(1, num_heads, chunk_size_global, KVPE_DIM, dtype=torch.bfloat16)
    # K/V: full populated prefix in natural order. Will be reshuffled into the
    # balanced-growing layout below before upload.
    k_global = torch.randn(1, 1, populated_global, KVPE_DIM, dtype=torch.bfloat16)
    v_global = torch.randn(1, num_heads, populated_global, V_HEAD_DIM, dtype=torch.bfloat16)
    k_host = _to_balanced_growing(k_global, sp, chunk_size_local, n_chunks_populated)
    v_host = _to_balanced_growing(v_global, sp, chunk_size_local, n_chunks_populated)

    tt_ccl = get_tt_ccl(mesh_device)

    # Q (sp-sharded on seq, tp-sharded on heads).
    q_shard_dims = [None, None]
    q_shard_dims[sp_axis] = 2
    q_shard_dims[tp_axis] = 1
    tt_q = ttnn.from_torch(
        q_host,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=q_shard_dims),
    )

    # K (sp-sharded on seq, NHK=1 so no tp sharding on heads).
    k_shard_dims = [None, None]
    k_shard_dims[sp_axis] = 2
    tt_k = ttnn.from_torch(
        k_host,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=k_shard_dims),
    )

    # V (sp-sharded on seq, tp-sharded on heads).
    v_shard_dims = [None, None]
    v_shard_dims[sp_axis] = 2
    v_shard_dims[tp_axis] = 1
    tt_v = ttnn.from_torch(
        v_host,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=v_shard_dims),
    )

    # Empty joint placeholders (joint_strategy="rear" with seq=0 — no joint compute).
    joint_shard_dims = [None, None]
    joint_shard_dims[tp_axis] = 1
    joint_q = ttnn.from_torch(
        torch.zeros(1, num_heads_local, 0, KVPE_DIM),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=joint_shard_dims),
    )
    joint_kv = ttnn.from_torch(
        torch.zeros(1, 1, 0, KVPE_DIM),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    joint_v = ttnn.from_torch(
        torch.zeros(1, num_heads_local, 0, V_HEAD_DIM),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=joint_shard_dims),
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    def make_persistent_bufs(seq_len_total):
        """Allocate (K, V) persistent buffers with the given seq dim."""
        persistent_v_shard_dims = [None, None]
        persistent_v_shard_dims[tp_axis] = 1
        pk = ttnn.from_torch(
            torch.zeros(1, 1, seq_len_total, KVPE_DIM),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[None, None]),
        )
        pv = ttnn.from_torch(
            torch.zeros(1, num_heads, seq_len_total, V_HEAD_DIM),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=persistent_v_shard_dims
            ),
        )
        return pk, pv

    def call_op(persistent_k_buf, persistent_v_buf):
        return ttnn.transformer.ring_joint_scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            joint_q,
            joint_kv,
            joint_v,
            persistent_output_buffer_k=persistent_k_buf,
            persistent_output_buffer_v=persistent_v_buf,
            joint_strategy="rear",
            logical_n=populated_global,
            program_config=_make_program_config(mesh_device),
            compute_kernel_config=compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=tt_ccl.ring_attention_ccl_semaphore_handles,
            num_links=1,
            cluster_axis=sp_axis,
            mesh_device=mesh_device,
            topology=topology,
            subdevice_id=tt_ccl.worker_sub_device_id,
            ccl_core_grid_offset=tt_ccl.ring_attention_ccl_core_grid_offset,
            use_column_major_ccl=True,
            is_causal=True,
            scale=scale,
            is_balanced=False,
        )

    # ----- Call A: exact-size persistent buffer (current chunked-MLA pattern) -----
    pk_exact, pv_exact = make_persistent_bufs(populated_global)
    logger.info(f"Call A: persistent K/V seq = {populated_global} (== N_local_kv * ring_size, exact)")
    attn_out_exact, _, _ = call_op(pk_exact, pv_exact)
    logger.success("Call A succeeded with exact-size persistent buffer.")

    # Verify Call A correctness against a torch chunked-causal MLA-SDPA reference.
    # Q is in natural order (single chunk), K/V (in `k_global`/`v_global`) are
    # the natural-order full prefix. The op output for chunk `chunk_idx` is in
    # natural order — gathered output matches the reference directly.
    out_concat_dims = [None, None]
    out_concat_dims[tp_axis] = 1
    out_concat_dims[sp_axis] = 2
    tt_out_exact_host = ttnn.to_torch(
        attn_out_exact,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=out_concat_dims, mesh_shape=mesh_device.shape),
    ).to(torch.bfloat16)
    ref = _causal_chunked_mla_sdpa_torch(q_host, k_global, v_global, q_abs_offset, scale)
    _, pcc_msg = assert_with_pcc(ref, tt_out_exact_host, 0.97)
    logger.info(f"Call A PCC vs torch reference: {pcc_msg}")

    # ----- Call B: max-size persistent buffer (the pattern we'd like to use) -----
    #
    # With AG :88-93 and SDPA :192 relaxed from `==` to `>=` on the gather dim,
    # the op should accept an oversize buffer and produce output identical to
    # Call A's (the dead trailing tiles per batch*head are untouched by writer
    # and unread by reader — see kernel trace in the test docstring).
    pk_max, pv_max = make_persistent_bufs(seq_len_max)
    oversize_ratio = seq_len_max / populated_global
    logger.info(f"Call B: persistent K/V seq = {seq_len_max} (max prefill horizon, " f"{oversize_ratio:g}x oversize).")
    attn_out_max, _, _ = call_op(pk_max, pv_max)
    logger.success("Call B succeeded with max-size persistent buffer.")

    # Gather Call B output and verify it matches Call A's (the answer should
    # be independent of how big the persistent buffer is, as long as the
    # populated prefix is the same).
    tt_out_max_host = ttnn.to_torch(
        attn_out_max,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=out_concat_dims, mesh_shape=mesh_device.shape),
    ).to(torch.bfloat16)
    _, pcc_vs_a_msg = assert_with_pcc(tt_out_exact_host, tt_out_max_host, 0.999)
    logger.info(f"Call B vs Call A PCC: {pcc_vs_a_msg}")
    _, pcc_vs_ref_msg = assert_with_pcc(ref, tt_out_max_host, 0.97)
    logger.info(f"Call B vs torch reference PCC: {pcc_vs_ref_msg}")


# ===========================================================================
# Test 3b: Reuse ONE persistent buffer across a full chunked-prefill loop
# ===========================================================================
#
# Test 3 proves that a single chunked op call accepts an oversize persistent
# buffer with bit-identical output. This test goes one step further: it
# allocates ONE pair of max-sized (seq=seq_len_max) persistent K/V buffers
# *before* the chunked-prefill loop and passes the same buffer objects to
# every op call — chunk 0 through chunk N-1.
#
# This is the pattern we'd like chunked MLA to use: __init__-time
# allocation only, no per-chunk realloc inside forward(). Mirrors what
# the non-chunked single-shot path already does today (mla.py:304-324).
#
# Per-chunk behavior:
#   chunk 0: N_local_q == N_local_kv (one-chunk prefix, regular causal).
#   chunk i ≥ 1: N_local_q < N_local_kv (chunked-mode triggered by shapes).
#   logical_n grows: (i+1)*chunk_size_global.
#   AG writes only ring_size*N_local_kv tiles per call (a growing prefix);
#     each chunk's write overwrites the previous prefix and extends it.
#   Per-head stride is constant (= seq_len_max * Wt), so the per-head
#     "active region" stays at the same offset across chunks. Bit-correct.
#
# Each chunk's output is PCC-checked against a torch chunked-causal MLA-SDPA
# reference. If all chunks pass, the kernels handle buffer reuse correctly.
#
# Today (unmodified main): this test fails at AG :93 / SDPA :192 for any
# chunk where populated_global < seq_len_max (i.e. every chunk except the
# last). With the two `==` → `>=` relaxations applied, it passes.
#
# Caveat: the program cache still forks per chunk because the SDPA program
# hash depends on input K shape (and logical_n) — not on buffer shape. So
# this relaxation doesn't deduplicate programs across chunks; the win is
# the host-side allocation + memory churn elimination.
# ===========================================================================


@pytest.mark.parametrize("mesh_device", [(2, 2), (2, 4)], ids=["2x2", "2x4"], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.timeout(0)
def test_persistent_buffer_reuse_across_chunks(mesh_device, device_params):
    sp_axis = 0
    tp_axis = 1
    mesh_shape = list(mesh_device.shape)
    sp = mesh_shape[sp_axis]
    tp = mesh_shape[tp_axis]
    num_heads = tp * tp
    num_heads_local = num_heads // tp

    chunk_size_local = 32  # 1 tile per chunk per chip
    chunk_size_global = chunk_size_local * sp
    n_chunks_total = 4
    seq_len_max = n_chunks_total * chunk_size_global

    scale = QK_HEAD_DIM**-0.5
    topology = _topology_from(device_params)

    torch.manual_seed(0)
    # Full-ISL Q/K/V — chunks will slice into these as the loop progresses.
    q_global_full = torch.randn(1, num_heads, seq_len_max, KVPE_DIM, dtype=torch.bfloat16)
    k_global_full = torch.randn(1, 1, seq_len_max, KVPE_DIM, dtype=torch.bfloat16)
    v_global_full = torch.randn(1, num_heads, seq_len_max, V_HEAD_DIM, dtype=torch.bfloat16)

    tt_ccl = get_tt_ccl(mesh_device)

    # Joint placeholders (built once, reused every call) — joint_strategy="rear" + seq=0 is a no-op.
    joint_shard_dims = [None, None]
    joint_shard_dims[tp_axis] = 1
    joint_q = ttnn.from_torch(
        torch.zeros(1, num_heads_local, 0, KVPE_DIM),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=joint_shard_dims),
    )
    joint_kv = ttnn.from_torch(
        torch.zeros(1, 1, 0, KVPE_DIM),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    joint_v = ttnn.from_torch(
        torch.zeros(1, num_heads_local, 0, V_HEAD_DIM),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=joint_shard_dims),
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    # ===== THE KEY ALLOCATION: one pair of max-sized persistent buffers, built ONCE =====
    persistent_v_shard_dims = [None, None]
    persistent_v_shard_dims[tp_axis] = 1
    pk_shared = ttnn.from_torch(
        torch.zeros(1, 1, seq_len_max, KVPE_DIM),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[None, None]),
    )
    pv_shared = ttnn.from_torch(
        torch.zeros(1, num_heads, seq_len_max, V_HEAD_DIM),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, mesh_shape=tuple(mesh_device.shape), dims=persistent_v_shard_dims
        ),
    )
    logger.info(
        f"Allocated ONE pair of persistent K/V buffers (seq={seq_len_max}). "
        f"Will reuse for all {n_chunks_total} chunked op calls below."
    )

    # ===== Chunked-prefill loop =====
    for chunk_idx in range(n_chunks_total):
        populated_chunks = chunk_idx + 1
        populated_global = populated_chunks * chunk_size_global
        q_abs_offset = chunk_idx * chunk_size_global

        # Q for this chunk: tokens [q_abs_offset, q_abs_offset+chunk_size_global) of the
        # full Q. Single-chunk Q has identical natural and balanced layouts.
        q_host_chunk = q_global_full[:, :, q_abs_offset : q_abs_offset + chunk_size_global, :].contiguous()

        # K/V populated prefix (natural order, then balanced-reshuffled for the SP sharding).
        k_natural_chunk = k_global_full[:, :, :populated_global, :].contiguous()
        v_natural_chunk = v_global_full[:, :, :populated_global, :].contiguous()
        k_host_chunk = _to_balanced_growing(k_natural_chunk, sp, chunk_size_local, populated_chunks)
        v_host_chunk = _to_balanced_growing(v_natural_chunk, sp, chunk_size_local, populated_chunks)

        # Upload Q/K/V to mesh (fresh per chunk because their shapes change).
        q_shard_dims = [None, None]
        q_shard_dims[sp_axis] = 2
        q_shard_dims[tp_axis] = 1
        tt_q_chunk = ttnn.from_torch(
            q_host_chunk,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=q_shard_dims),
        )
        k_shard_dims = [None, None]
        k_shard_dims[sp_axis] = 2
        tt_k_chunk = ttnn.from_torch(
            k_host_chunk,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=k_shard_dims),
        )
        v_shard_dims = [None, None]
        v_shard_dims[sp_axis] = 2
        v_shard_dims[tp_axis] = 1
        tt_v_chunk = ttnn.from_torch(
            v_host_chunk,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=v_shard_dims),
        )

        # Op call: SAME persistent buffers every iteration.
        attn_out_chunk, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            tt_q_chunk,
            tt_k_chunk,
            tt_v_chunk,
            joint_q,
            joint_kv,
            joint_v,
            persistent_output_buffer_k=pk_shared,
            persistent_output_buffer_v=pv_shared,
            joint_strategy="rear",
            logical_n=populated_global,
            program_config=_make_program_config(mesh_device),
            compute_kernel_config=compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=tt_ccl.ring_attention_ccl_semaphore_handles,
            num_links=1,
            cluster_axis=sp_axis,
            mesh_device=mesh_device,
            topology=topology,
            subdevice_id=tt_ccl.worker_sub_device_id,
            ccl_core_grid_offset=tt_ccl.ring_attention_ccl_core_grid_offset,
            use_column_major_ccl=True,
            is_causal=True,
            scale=scale,
            is_balanced=False,
        )

        # Gather output and verify against torch chunked-causal reference for this chunk.
        out_concat_dims = [None, None]
        out_concat_dims[tp_axis] = 1
        out_concat_dims[sp_axis] = 2
        tt_out_chunk_host = ttnn.to_torch(
            attn_out_chunk,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=out_concat_dims, mesh_shape=mesh_device.shape),
        ).to(torch.bfloat16)
        ref_chunk = _causal_chunked_mla_sdpa_torch(q_host_chunk, k_natural_chunk, v_natural_chunk, q_abs_offset, scale)
        _, pcc_msg = assert_with_pcc(ref_chunk, tt_out_chunk_host, 0.97)
        logger.info(
            f"  chunk {chunk_idx}/{n_chunks_total - 1}: "
            f"N_local_q={chunk_size_local}, N_local_kv={populated_global // sp}, "
            f"logical_n={populated_global}, PCC={pcc_msg}"
        )

    logger.success(
        f"All {n_chunks_total} chunked-prefill iterations passed with ONE pair of "
        f"persistent K/V buffers (seq={seq_len_max}). No per-chunk realloc."
    )


# ===========================================================================
# Test 4: KV-pad-aware rotation for ISL smaller than chunk_size (torch showcase)
# ===========================================================================
#
# Production scenario
# -------------------
# The op is always handed an input tensor of fixed shape `chunk_size` in
# the ISL dim (e.g. 5120 in production). Only the first `new_actual_isl`
# tokens are valid; the rest is padding that the next iteration will
# overwrite. The prior cache holds `kv_actual_isl` valid tokens (always
# rounded down to TILE_SIZE), with trailing padding inside the last device's
# OLD slab if `kv_actual_isl` doesn't fill the per-device slab cleanly.
#
# Across iterations the cache state can be: each device's slab is uniform
# in size, valid tokens are dense up to position kv_actual_isl in global
# order, with the trailing padding always on the "pad chip" (the chip
# whose OLD slab has the unfilled cells).
#
# The server, between iterations, REORDERS new tokens so that the pad
# chip gets the first `pad_size_in_chip` tokens (to overwrite its OLD pad
# cells), then the remaining new tokens are distributed across devices'
# NEW slabs (devs 0..pad_chip-1 each get up to chunk_size_local NEW
# tokens, then pad_chip gets the leftover into its NEW slab). When
# new_actual_isl < chunk_size, some NEW slabs end up with their own
# trailing padding too.
#
# What the op needs (per Q3+Q4)
# -----------------------------
# Today the op derives the Q-to-global-position offset from
# `q_start_idx = (N_local_kv - N_local_q) * ring_size`, which assumes a
# clean balanced-growing layout. With this rotation the layout is no
# longer clean — Q rows on a single chip can correspond to two disjoint
# global-position regions (the pad-fill bump + the chip's natural slab
# slot).
#
# The op needs to compute, from `(kv_actual_isl, new_actual_isl, sp_factor,
# chunk_size_local)`:
#
#     pad_chip           = (kv_actual_isl // chunk_size_local) % sp_factor   # first chip with pad
#     pad_offset_in_chip = kv_actual_isl % chunk_size_local
#     total_pad_in_old   = sp_factor * chunk_size_local - kv_actual_isl      # may span multiple chips
#
# Pad fill (phase 1) walks OLD-slab pad cells in *global-position order* —
# pad can span multiple chips when `kv_actual_isl + pad` straddles
# chunk_size_local boundaries (e.g., kv_actual_isl=128 on sp=4 with
# chunk_size_local=64 leaves devs 2 and 3 OLD slabs entirely as pad).
# Phase 2 (NEW-slab fills) only starts after all OLD pad is consumed.
#
# From these the op derives, for each Q row on each chip, the row's global
# position; for each K col on each chip, the col's global position; then
# apply a per-(Q row, K col) mask:
#
#     mask = -inf if (Q row is padding) OR (K col is padding) OR (K_pos > Q_pos)
#     mask = 0    otherwise
#
# Causality applies on the global positions, not the local row indices —
# the kernel's existing kv_global_tile_for_local() math doesn't capture
# the rotation.
#
# What this test does
# -------------------
# Pure torch, no device. Builds the rotated per-device layout for K, V, Q
# explicitly, builds the explicit attention mask using global positions,
# runs torch SDPA with the mask, then verifies the result against a
# natural-order causal reference (no rotation, full sequence).
#
# Owners would implement this on-device: derive the global positions from
# kv_actual_isl + new_actual_isl + sp_factor + chunk_size_local on the
# fly, and either pass an explicit mask CB or fold the masking math
# into the existing chunked-prefill mask path.
#
# Test scenario (small)
# ---------------------
#   sp_factor          = 4
#   chunk_size         = 256       (so chunk_size_local = 64)
#   kv_actual_isl      = 224       (pad on dev 3: 32 valid + 32 pad in its OLD slab)
#   new_actual_isl     = 64        (much smaller than chunk_size — the "small ISL" case)
#
#   Derived:
#     pad_chip            = 3
#     pad_offset_in_chip  = 32     (where pad starts on pad_chip's OLD slab)
#     pad_size_in_chip    = 32     (how many pad cells)
#
#   Server reorder of new tokens [0..64]:
#     tokens [0..32]  -> dev 3 OLD slab cells [32..64]  (global pos 224..256)
#     tokens [32..64] -> dev 0 NEW slab cells [0..32]   (global pos 256..288)
#     all other NEW-slab cells are padding.
#
# ===========================================================================


def _kv_pad_rotation_layout(kv_actual_isl, new_actual_isl, sp_factor, chunk_size_local):
    """
    Compute the per-device K/Q layout after the server's kv-pad-aware rotation.

    OLD-slab pad can span multiple chips when kv_actual_isl + pad straddles
    chunk_size_local boundaries (e.g., kv_actual_isl=128 with sp=4,
    chunk_size_local=64 leaves devs 2 and 3 OLD slabs entirely as pad).
    Phase 1 fills these pad cells in global-position order across all
    affected chips before phase 2 moves to NEW slabs.

    Returns:
      pad_chip            (int): first chip whose OLD slab contains pad
        (= (kv_actual_isl // chunk_size_local) % sp_factor).
      pad_offset_in_chip  (int): position in pad_chip's OLD slab where
        pad starts (= kv_actual_isl % chunk_size_local).
      total_pad_in_old    (int): total number of pad cells across all OLD
        slabs (= sp_factor * chunk_size_local - kv_actual_isl).
      new_token_destinations (list of (token_idx, chip, slab, cell)):
        For each new token (index in natural order, 0..new_actual_isl),
        where it ends up. `slab` is "OLD" or "NEW", `cell` is the position
        in that slab (0..chunk_size_local).
    """
    pad_chip = (kv_actual_isl // chunk_size_local) % sp_factor
    pad_offset_in_chip = kv_actual_isl % chunk_size_local
    total_pad_in_old = max(0, sp_factor * chunk_size_local - kv_actual_isl)

    new_token_destinations = []
    # Phase 1: fill OLD-slab pad cells in global-position order, which can
    # span multiple chips.
    fill_count = min(total_pad_in_old, new_actual_isl)
    for i in range(fill_count):
        pad_global_pos = kv_actual_isl + i
        chip = (pad_global_pos // chunk_size_local) % sp_factor
        cell = pad_global_pos % chunk_size_local
        new_token_destinations.append((i, chip, "OLD", cell))

    # Phase 2: remaining tokens go into NEW slabs, starting from chip 0,
    # filling chunk_size_local cells per chip before moving on.
    remaining = new_actual_isl - fill_count
    token_idx = fill_count
    chip = 0
    cell = 0
    while remaining > 0:
        new_token_destinations.append((token_idx, chip, "NEW", cell))
        token_idx += 1
        remaining -= 1
        cell += 1
        if cell == chunk_size_local:
            cell = 0
            chip += 1  # next chip's NEW slab (chip can equal pad_chip — its NEW slab is a normal slab)

    return pad_chip, pad_offset_in_chip, total_pad_in_old, new_token_destinations


@pytest.mark.parametrize(
    "kv_actual_isl, new_actual_isl, expected_pad_chip, expected_pad_offset, expected_total_pad",
    [
        # Single-device pad: pad on dev 3 only (32 cells).
        (224, 64, 3, 32, 32),
        # Multi-device pad: devs 2 and 3 OLD slabs entirely pad; new fills both fully.
        (128, 128, 2, 0, 128),
        # Multi-device pad, partial fill: fills dev 2's pad fully, dev 3's pad partially.
        (128, 96, 2, 0, 128),
        # Cold start: every OLD slab is pad; new fills first two devices.
        (0, 128, 0, 0, 256),
        # Clean boundary: kv ends on a chunk boundary, no OLD pad anywhere.
        # All new tokens go to NEW slabs starting from dev 0.
        (256, 64, 0, 0, 0),
    ],
    ids=["single_pad_dev3", "multi_pad_2_full", "multi_pad_2_partial", "cold_start", "no_old_pad"],
)
def test_kv_pad_aware_rotation_torch_showcase(
    kv_actual_isl, new_actual_isl, expected_pad_chip, expected_pad_offset, expected_total_pad
):
    """See section header above for the full motivation and math."""
    # ----- Scenario parameters -----
    sp_factor = 4
    chunk_size = 256
    chunk_size_local = chunk_size // sp_factor  # 64
    nhq = 4  # torch only — pick anything

    scale = QK_HEAD_DIM**-0.5

    # ----- Layout math -----
    pad_chip, pad_offset_in_chip, total_pad_in_old, new_token_destinations = _kv_pad_rotation_layout(
        kv_actual_isl=kv_actual_isl,
        new_actual_isl=new_actual_isl,
        sp_factor=sp_factor,
        chunk_size_local=chunk_size_local,
    )
    assert pad_chip == expected_pad_chip
    assert pad_offset_in_chip == expected_pad_offset
    assert total_pad_in_old == expected_total_pad
    assert len(new_token_destinations) == new_actual_isl

    # ----- Generate inputs -----
    torch.manual_seed(0)
    # Prior cache (kv_actual_isl tokens in natural order across global positions 0..kv_actual_isl).
    old_cache_k = torch.randn(1, 1, kv_actual_isl, KVPE_DIM, dtype=torch.bfloat16)
    old_cache_v = torch.randn(1, nhq, kv_actual_isl, V_HEAD_DIM, dtype=torch.bfloat16)
    # This iteration's new tokens in NATURAL order (positions kv_actual_isl..kv_actual_isl+new_actual_isl).
    new_tokens_q = torch.randn(1, nhq, new_actual_isl, KVPE_DIM, dtype=torch.bfloat16)
    new_tokens_k = torch.randn(1, 1, new_actual_isl, KVPE_DIM, dtype=torch.bfloat16)
    new_tokens_v = torch.randn(1, nhq, new_actual_isl, V_HEAD_DIM, dtype=torch.bfloat16)

    # ----- Build per-device K/V cache (OLD slab + NEW slab) after the iteration's writes -----
    # Each device's cache slab is now 2 * chunk_size_local (OLD + NEW).
    # Per-cell metadata: (is_valid, global_pos, source: "old_cache" or "new_tokens", source_idx).
    cache_seq_per_dev = 2 * chunk_size_local  # 128
    k_per_dev = torch.zeros(sp_factor, 1, cache_seq_per_dev, KVPE_DIM, dtype=torch.bfloat16)
    v_per_dev = torch.zeros(sp_factor, nhq, cache_seq_per_dev, V_HEAD_DIM, dtype=torch.bfloat16)
    k_global_pos_per_dev = [[None] * cache_seq_per_dev for _ in range(sp_factor)]  # None = padding

    # OLD slab content from prior cache. Each chip's OLD slab holds positions
    # [chip * chunk_size_local .. (chip+1) * chunk_size_local) of the prior cache;
    # cells past kv_actual_isl are padding (will be overwritten by rotation).
    for chip in range(sp_factor):
        chip_old_start = chip * chunk_size_local
        chip_old_end = min(chip_old_start + chunk_size_local, kv_actual_isl)
        n_valid_in_chip = max(0, chip_old_end - chip_old_start)
        if n_valid_in_chip > 0:
            k_per_dev[chip, :, :n_valid_in_chip, :] = old_cache_k[0, :, chip_old_start:chip_old_end, :]
            v_per_dev[chip, :, :n_valid_in_chip, :] = old_cache_v[0, :, chip_old_start:chip_old_end, :]
            for r in range(n_valid_in_chip):
                k_global_pos_per_dev[chip][r] = chip_old_start + r

    # Apply the server's rotation: new tokens go into the OLD pad slot first, then NEW slabs.
    for token_idx, chip, slab, cell in new_token_destinations:
        global_pos = kv_actual_isl + token_idx
        cache_row = cell if slab == "OLD" else (chunk_size_local + cell)
        k_per_dev[chip, :, cache_row, :] = new_tokens_k[0, :, token_idx, :]
        v_per_dev[chip, :, cache_row, :] = new_tokens_v[0, :, token_idx, :]
        k_global_pos_per_dev[chip][cache_row] = global_pos

    # ----- Build per-device Q (just this iter's new tokens, padded) -----
    q_per_dev = torch.zeros(sp_factor, nhq, chunk_size_local, KVPE_DIM, dtype=torch.bfloat16)
    q_global_pos_per_dev = [[None] * chunk_size_local for _ in range(sp_factor)]
    # Per-device Q layout follows the same distribution as the NEW writes (one Q row per new token).
    # Q's chunk_size_local slab per device gets the new tokens that ALSO populate THIS chip's
    # NEW K/V slab cells (one-to-one mapping by cell index), PLUS the pad-fill tokens get
    # placed on pad_chip's Q slab too (they're new tokens whose Q rows must be computed).
    #
    # Simplest model: Q-on-chip-c row r is THE new token whose K-on-chip-c lands at cell r
    # of EITHER the pad-fill slot (mapping cell in OLD slab [pad_offset..chunk_size_local)
    # back to Q row [0..pad_size_in_chip)) OR the NEW slab (cell in NEW slab maps directly to
    # Q row of the same cell index). pad_chip's Q ends up with both pad-fill tokens (rows
    # 0..pad_size_in_chip) and any NEW-slab tokens (rows pad_size_in_chip..) — Q row indices
    # don't directly correspond to "OLD vs NEW slab", they just enumerate this chip's
    # contribution to the new iteration.
    #
    # The mapping we use: for each new_token_destination, the Q row on `chip` is the
    # next-available row on that chip's Q slab. We track per-chip Q fill cursor.
    q_fill_cursor = [0] * sp_factor
    for token_idx, chip, slab, cell in new_token_destinations:
        global_pos = kv_actual_isl + token_idx
        q_row = q_fill_cursor[chip]
        q_per_dev[chip, :, q_row, :] = new_tokens_q[0, :, token_idx, :]
        q_global_pos_per_dev[chip][q_row] = global_pos
        q_fill_cursor[chip] += 1

    # ----- Concatenate per-device tensors into one flat sequence for torch SDPA -----
    # Combined K shape: [1, 1, sp_factor * cache_seq_per_dev, KVPE_DIM]
    combined_K = k_per_dev.permute(1, 0, 2, 3).reshape(1, 1, sp_factor * cache_seq_per_dev, KVPE_DIM)
    combined_V = v_per_dev.permute(1, 0, 2, 3).reshape(1, nhq, sp_factor * cache_seq_per_dev, V_HEAD_DIM)
    combined_Q = q_per_dev.permute(1, 0, 2, 3).reshape(1, nhq, sp_factor * chunk_size_local, KVPE_DIM)

    # Flatten per-device global position maps to match the combined layout.
    combined_K_global_pos = []
    for chip in range(sp_factor):
        combined_K_global_pos.extend(k_global_pos_per_dev[chip])
    combined_Q_global_pos = []
    for chip in range(sp_factor):
        combined_Q_global_pos.extend(q_global_pos_per_dev[chip])

    # ----- Build the explicit attention mask using global positions -----
    sq_total = len(combined_Q_global_pos)
    sk_total = len(combined_K_global_pos)
    mask = torch.zeros(sq_total, sk_total, dtype=torch.float32)
    for i, q_pos in enumerate(combined_Q_global_pos):
        for j, k_pos in enumerate(combined_K_global_pos):
            if q_pos is None or k_pos is None or k_pos > q_pos:
                mask[i, j] = float("-inf")

    # ----- Run torch SDPA with the explicit mask -----
    # K head dim differs from Q in MLA flat sense, but the K we built here is the absorbed
    # form (head_dim=KVPE_DIM=576), matching Q. Broadcast K's single head to nhq.
    K_b = combined_K.expand(-1, nhq, -1, -1)
    attn_scores = (combined_Q.float() @ K_b.transpose(-2, -1).float()) * scale
    attn_scores = attn_scores + mask  # mask is broadcast across batch + heads
    attn = torch.softmax(attn_scores, dim=-1)
    rotated_out = (attn @ combined_V.float()).to(torch.bfloat16)
    # rotated_out shape: [1, nhq, sq_total, V_HEAD_DIM]

    # Extract the valid Q rows in natural global-position order.
    # For each new token index t (global pos kv_actual_isl+t), find its row in combined_Q.
    valid_rows = [None] * new_actual_isl
    for i, q_pos in enumerate(combined_Q_global_pos):
        if q_pos is not None:
            t = q_pos - kv_actual_isl
            assert 0 <= t < new_actual_isl
            valid_rows[t] = i
    assert all(r is not None for r in valid_rows), "missing some new tokens in rotated layout"
    rotated_out_natural_order = rotated_out[:, :, valid_rows, :]

    # ----- Natural-order reference -----
    natural_K = torch.cat([old_cache_k, new_tokens_k], dim=2)  # [1, 1, kv+new, KVPE_DIM]
    natural_V = torch.cat([old_cache_v, new_tokens_v], dim=2)  # [1, nhq, kv+new, V_HEAD_DIM]
    natural_K_b = natural_K.expand(-1, nhq, -1, -1)
    sq = new_actual_isl
    sk = kv_actual_isl + new_actual_isl
    ref_scores = (new_tokens_q.float() @ natural_K_b.transpose(-2, -1).float()) * scale
    # Causal mask: Q row r (global pos kv_actual_isl+r) attends to K cols [0, kv_actual_isl+r] (inclusive).
    q_pos_vec = torch.arange(sq) + kv_actual_isl
    k_pos_vec = torch.arange(sk)
    ref_mask = (k_pos_vec.unsqueeze(0) > q_pos_vec.unsqueeze(1)).float() * float("-inf")
    ref_mask = torch.nan_to_num(ref_mask, nan=0.0)  # turn 0*-inf into 0
    ref_scores = ref_scores + ref_mask
    ref_attn = torch.softmax(ref_scores, dim=-1)
    ref_out = (ref_attn @ natural_V.float()).to(torch.bfloat16)
    # ref_out shape: [1, nhq, new_actual_isl, V_HEAD_DIM]

    # ----- Compare -----
    torch.testing.assert_close(rotated_out_natural_order, ref_out, rtol=1e-2, atol=1e-2)
    logger.success(
        f"KV-pad-aware rotation showcase passed: rotated output matches natural-order "
        f"causal reference for new_actual_isl={new_actual_isl}, kv_actual_isl={kv_actual_isl}, "
        f"pad_chip={pad_chip}, total_pad_in_old={total_pad_in_old}."
    )
