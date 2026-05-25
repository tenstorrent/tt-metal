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
# Architectural conclusion
# ------------------------
# The three validation checks above are not arbitrary safety rails — they
# reflect the reader kernel's structural assumption that all K-side
# tensors (input K, gathered K) share one accessor type, and likewise for
# V. The kernel literally cannot be JIT-compiled without this. Relaxing
# the validation does not unblock the op.
#
# For the chunked-MLA caller, this means the
# ttnn.slice(..., memory_config=ttnn.DRAM_MEMORY_CONFIG) workaround is
# doing essential work: it produces an interleaved K tensor whose
# accessor distribution matches the (interleaved) gathered buffer's, so
# the reader's ternaries type-unify.
#
# Dropping the slice requires owners to either:
#
#   (a) Make the gathered K buffer ND-shard with the *same* spec as the
#       input K (forcing all K-side tensors into one shard family).
#       Doable but rigid — V would need its own parallel ND-shard story,
#       and the output buffer alloc would need to follow input layout.
#
#   (b) Refactor the reader kernel to handle mixed input/gathered layouts
#       via if-constexpr branches per (input_layout, gathered_layout)
#       combo. More flexible, more code, more JIT variants.
#
# In either case the AG sub-op's three validate checks (lines 41, 52, 84)
# need to be relaxed/replaced to express per-tensor configs instead of
# one config-for-all.
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
@pytest.mark.timeout(0)
def test_nd_sharded_kv_cache_as_k(mesh_device, device_params):
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

    # Upload V (sharded on sp seq + tp heads) — interleaved DRAM, matching
    # how V naturally comes out of wkv_b2 in production MLA.
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
        f"K shape={list(tt_cache.shape)}, K memory_config={tt_cache.memory_config()}"
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

    logger.success("Op accepted ND-sharded KV cache as K input — no validation error.")

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
    logger.info(f"ND-sharded K PCC vs torch reference: {pcc_msg}")


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
# Test 3 — TODO: per-chunk persistent buffer realloc (needs design input)
# Test 4 — TODO: ISL smaller than chunk_size (needs design input)
# ===========================================================================
