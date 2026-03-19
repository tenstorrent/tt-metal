# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Exhaustive fusion build cache tests.

Tests cache key properties (topology fingerprint, Parallel sort),
cache hit correctness (RT args, CB pointers, outputs, PCC), cache
entry structure, topology differentiation, and lifecycle.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.ops.descriptors.fusion import Parallel, Sequential
from models.experimental.ops.descriptors.fusion.fusion import (
    _BUILD_CACHE,
    _CacheEntry,
    _CacheHitOverrideSpec,
    _topology_fingerprint,
    _cache_key_and_ops,
)
from models.experimental.ops.descriptors.normalization import rms_norm


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def torch_rms_norm(x, weight, eps=1e-5):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = x / rms
    if weight is not None:
        out = out * weight
    return out


def cores(x1, y1, x2=None, y2=None):
    if x2 is None:
        x2, y2 = x1, y1
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x1, y1), ttnn.CoreCoord(x2, y2))})


def _make_branches(device, seed=42):
    """Create Q and KV RMS norm branches (DeepSeek V3 style)."""
    torch.manual_seed(seed)

    q_cores = cores(0, 0, 3, 3)
    q_shard_w = 96
    q_total_w = 16 * q_shard_w

    kv_cores = cores(5, 0, 6, 7)
    kv_shard_w = 32
    kv_total_w = 16 * kv_shard_w

    q_shard_spec = ttnn.ShardSpec(q_cores, [32, q_shard_w], ttnn.ShardOrientation.ROW_MAJOR)
    q_mem = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=q_shard_spec,
    )
    q_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(4, 4),
        subblock_w=q_shard_w // 32,
        block_h=1,
        block_w=q_shard_w // 32,
        inplace=False,
    )

    kv_shard_spec = ttnn.ShardSpec(kv_cores, [32, kv_shard_w], ttnn.ShardOrientation.ROW_MAJOR)
    kv_mem = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=kv_shard_spec,
    )
    kv_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(2, 8),
        subblock_w=kv_shard_w // 32,
        block_h=1,
        block_w=kv_shard_w // 32,
        inplace=False,
    )

    torch_q_input = torch.rand(1, 1, 32, q_total_w, dtype=torch.bfloat16)
    torch_q_weight = torch.rand(1, 1, 1, q_total_w, dtype=torch.bfloat16)
    torch_kv_input = torch.rand(1, 1, 32, kv_total_w, dtype=torch.bfloat16)
    torch_kv_weight = torch.rand(1, 1, 1, kv_total_w, dtype=torch.bfloat16)

    tt_q_input = ttnn.from_torch(torch_q_input, device=device, layout=ttnn.TILE_LAYOUT, memory_config=q_mem)
    tt_q_weight = ttnn.from_torch(torch_q_weight, device=device, layout=ttnn.TILE_LAYOUT)
    tt_kv_input = ttnn.from_torch(torch_kv_input, device=device, layout=ttnn.TILE_LAYOUT, memory_config=kv_mem)
    tt_kv_weight = ttnn.from_torch(torch_kv_weight, device=device, layout=ttnn.TILE_LAYOUT)

    q_branch = rms_norm.rms_norm(
        tt_q_input,
        epsilon=1e-5,
        weight=tt_q_weight,
        memory_config=q_mem,
        core_range_set=q_cores,
        program_config=q_pc,
    )
    kv_branch = rms_norm.rms_norm(
        tt_kv_input,
        epsilon=1e-5,
        weight=tt_kv_weight,
        memory_config=kv_mem,
        core_range_set=kv_cores,
        program_config=kv_pc,
    )

    return q_branch, kv_branch, (torch_q_input, torch_q_weight, torch_kv_input, torch_kv_weight)


# ===========================================================================
# D1. Infrastructure tests (no device)
# ===========================================================================


class TestCacheKeyInfra:
    """Test cache key properties without device.

    Tests _topology_fingerprint directly with flat item lists.
    Parallel construction requires real ProgramDescriptors — those
    are tested in the device test classes.
    """

    def test_topology_fingerprint_ops_only(self):
        """Flat op list produces 'O,O,O'."""
        from models.experimental.ops.descriptors.op_descriptor import OpDescriptor

        class _MockDescriptor:
            kernels = []
            cbs = []
            semaphores = []

        a = OpDescriptor(descriptor=_MockDescriptor(), input_tensors=[], output_tensors=[], name="a")
        b = OpDescriptor(descriptor=_MockDescriptor(), input_tensors=[], output_tensors=[], name="b")
        c = OpDescriptor(descriptor=_MockDescriptor(), input_tensors=[], output_tensors=[], name="c")

        fp = _topology_fingerprint([a, b, c])
        assert fp == "O,O,O"

    def test_topology_fingerprint_with_sequential(self):
        """Nested Sequential produces 'S(O,O)' sub-fingerprint."""
        from models.experimental.ops.descriptors.op_descriptor import OpDescriptor

        class _MockDescriptor:
            kernels = []
            cbs = []
            semaphores = []

        a = OpDescriptor(descriptor=_MockDescriptor(), input_tensors=[], output_tensors=[], name="a")
        b = OpDescriptor(descriptor=_MockDescriptor(), input_tensors=[], output_tensors=[], name="b")
        c = OpDescriptor(descriptor=_MockDescriptor(), input_tensors=[], output_tensors=[], name="c")

        inner = Sequential(b, c)
        fp = _topology_fingerprint([a, inner])
        assert fp == "O,S(O,O)", f"Expected 'O,S(O,O)', got {fp!r}"

    def test_cache_key_includes_topology(self, device):
        """Key tuple starts with a topology fingerprint string."""
        q, kv, _ = _make_branches(device, seed=42)
        key, ops = _cache_key_and_ops([q, kv])
        assert isinstance(key[0], str), f"First element should be string, got {type(key[0])}"
        assert key[0] == "O,O"


# ===========================================================================
# D2. Order tolerance (device)
# ===========================================================================


class TestCacheOrderTolerance:
    def test_parallel_order_invariant(self, device):
        """Parallel(Q, KV) and Parallel(KV, Q) hit same cache entry."""
        _BUILD_CACHE.clear()

        q1, kv1, _ = _make_branches(device, seed=100)
        fused1 = Parallel(q1, kv1).build()
        fused1.launch()
        assert len(_BUILD_CACHE) == 1

        q2, kv2, _ = _make_branches(device, seed=200)
        # Reversed order
        fused2 = Parallel(kv2, q2).build()
        fused2.launch()
        assert len(_BUILD_CACHE) == 1, f"Expected cache hit (size 1), got {len(_BUILD_CACHE)}"


# ===========================================================================
# D3. Cache hit correctness (device)
# ===========================================================================


class TestCacheHitCorrectness:
    def test_fresh_descriptor_each_hit(self, device):
        """Each cache hit returns a new FusedOp with distinct descriptor id."""
        _BUILD_CACHE.clear()

        q1, kv1, _ = _make_branches(device, seed=100)
        fused1 = Parallel(q1, kv1).build()
        fused1.launch()
        desc_id_1 = id(fused1.descriptor)

        q2, kv2, _ = _make_branches(device, seed=200)
        fused2 = Parallel(q2, kv2).build()
        desc_id_2 = id(fused2.descriptor)

        assert desc_id_1 != desc_id_2, "Cache hit must return a fresh descriptor copy"

    def test_output_tensors_from_fresh_ops(self, device):
        """Cache hit's output_tensors are from the fresh ops, not stale cached ones."""
        _BUILD_CACHE.clear()

        q1, kv1, _ = _make_branches(device, seed=100)
        fused1 = Parallel(q1, kv1).build()
        fused1.launch()
        out1_ids = {id(t) for t in fused1.output_tensors}

        q2, kv2, _ = _make_branches(device, seed=200)
        fused2 = Parallel(q2, kv2).build()
        out2_ids = {id(t) for t in fused2.output_tensors}

        assert not (out1_ids & out2_ids), "Cache hit outputs must be new tensor objects"

    def test_pcc_across_seeds(self, device):
        """10 iterations with different seeds, PCC >= 0.98 on every hit."""
        _BUILD_CACHE.clear()

        q_cores = cores(0, 0, 3, 3)
        q_shard_w = 96
        q_total_w = 16 * q_shard_w
        kv_cores = cores(5, 0, 6, 7)
        kv_shard_w = 32
        kv_total_w = 16 * kv_shard_w

        q_shard_spec = ttnn.ShardSpec(q_cores, [32, q_shard_w], ttnn.ShardOrientation.ROW_MAJOR)
        q_mem = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=q_shard_spec,
        )
        q_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(4, 4),
            subblock_w=q_shard_w // 32,
            block_h=1,
            block_w=q_shard_w // 32,
            inplace=False,
        )
        kv_shard_spec = ttnn.ShardSpec(kv_cores, [32, kv_shard_w], ttnn.ShardOrientation.ROW_MAJOR)
        kv_mem = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=kv_shard_spec,
        )
        kv_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(2, 8),
            subblock_w=kv_shard_w // 32,
            block_h=1,
            block_w=kv_shard_w // 32,
            inplace=False,
        )

        for i in range(10):
            seed = 1000 + i * 37
            torch.manual_seed(seed)

            torch_q = torch.rand(1, 1, 32, q_total_w, dtype=torch.bfloat16)
            torch_qw = torch.rand(1, 1, 1, q_total_w, dtype=torch.bfloat16)
            torch_kv = torch.rand(1, 1, 32, kv_total_w, dtype=torch.bfloat16)
            torch_kvw = torch.rand(1, 1, 1, kv_total_w, dtype=torch.bfloat16)

            tt_q = ttnn.from_torch(torch_q, device=device, layout=ttnn.TILE_LAYOUT, memory_config=q_mem)
            tt_qw = ttnn.from_torch(torch_qw, device=device, layout=ttnn.TILE_LAYOUT)
            tt_kv = ttnn.from_torch(torch_kv, device=device, layout=ttnn.TILE_LAYOUT, memory_config=kv_mem)
            tt_kvw = ttnn.from_torch(torch_kvw, device=device, layout=ttnn.TILE_LAYOUT)

            q_b = rms_norm.rms_norm(
                tt_q,
                epsilon=1e-5,
                weight=tt_qw,
                memory_config=q_mem,
                core_range_set=q_cores,
                program_config=q_pc,
            )
            kv_b = rms_norm.rms_norm(
                tt_kv,
                epsilon=1e-5,
                weight=tt_kvw,
                memory_config=kv_mem,
                core_range_set=kv_cores,
                program_config=kv_pc,
            )

            fused = Parallel(q_b, kv_b).build()
            fused.launch()

            if i == 0:
                assert len(_BUILD_CACHE) == 1
            else:
                assert len(_BUILD_CACHE) == 1, f"Iteration {i}: expected cache hit"

            q_golden = torch_rms_norm(torch_q.float(), torch_qw.float())
            q_result = ttnn.to_torch(q_b.output_tensors[0])
            passing_q, pcc_q = comp_pcc(q_golden, q_result, pcc=0.98)
            assert passing_q, f"[iter {i}] Q PCC={pcc_q}"

            kv_golden = torch_rms_norm(torch_kv.float(), torch_kvw.float())
            kv_result = ttnn.to_torch(kv_b.output_tensors[0])
            passing_kv, pcc_kv = comp_pcc(kv_golden, kv_result, pcc=0.98)
            assert passing_kv, f"[iter {i}] KV PCC={pcc_kv}"


# ===========================================================================
# D4. Cache entry structure (device)
# ===========================================================================


class TestCacheEntryStructure:
    def test_no_tensor_refs_in_entry(self, device):
        """_CacheEntry fields don't hold any ttnn.Tensor objects."""
        _BUILD_CACHE.clear()

        q, kv, _ = _make_branches(device, seed=42)
        Parallel(q, kv).build().launch()

        assert len(_BUILD_CACHE) == 1
        entry = next(iter(_BUILD_CACHE.values()))

        # Check entry fields
        assert isinstance(entry, _CacheEntry)
        assert isinstance(entry.spec, _CacheHitOverrideSpec)

        # cached_descriptor is a ProgramDescriptor, not a tensor
        assert not isinstance(entry.cached_descriptor, ttnn.Tensor)
        # semaphores should be a tuple of GlobalSemaphore refs, not tensors
        for s in entry.semaphores:
            assert not isinstance(s, ttnn.Tensor)

    def test_override_spec_complete(self, device):
        """_CacheHitOverrideSpec has non-empty origin_kernel_map and barrier_suffix."""
        _BUILD_CACHE.clear()

        q, kv, _ = _make_branches(device, seed=42)
        Parallel(q, kv).build().launch()

        entry = next(iter(_BUILD_CACHE.values()))
        spec = entry.spec

        assert len(spec.origin_kernel_map) > 0, "origin_kernel_map should not be empty"
        assert len(spec.barrier_suffix) > 0, "barrier_suffix should not be empty"
        assert len(spec.output_sources) > 0, "output_sources should not be empty"

    def test_override_spec_has_sharded_cb_map(self, device):
        """spec.sharded_cb_map is populated for sharded parallel RMS norm."""
        _BUILD_CACHE.clear()

        q, kv, _ = _make_branches(device, seed=42)
        Parallel(q, kv).build().launch()

        entry = next(iter(_BUILD_CACHE.values()))
        spec = entry.spec

        assert len(spec.sharded_cb_map) > 0, "sharded_cb_map should be non-empty for sharded ops"
        # Each entry should be (merged_idx, op_idx, orig_cb_idx) — all ints
        for merged_idx, op_idx, orig_cb_idx in spec.sharded_cb_map:
            assert isinstance(merged_idx, int)
            assert isinstance(op_idx, int)
            assert isinstance(orig_cb_idx, int)


# ===========================================================================
# D5. Topology differentiation (device)
# ===========================================================================


class TestTopologyDifferentiation:
    def test_topology_fingerprint_in_key(self, device):
        """Cache keys from Sequential and Parallel start with different topology strings."""
        q1, kv1, _ = _make_branches(device, seed=100)
        q2, kv2, _ = _make_branches(device, seed=200)

        key_par, _ = _cache_key_and_ops([Parallel(q1, kv1)])
        key_seq, _ = _cache_key_and_ops([Sequential(q2, kv2)])

        # Topology fingerprints differ
        assert key_par[0] != key_seq[0], (
            f"Parallel and Sequential should have different topology fingerprints: " f"{key_par[0]!r} vs {key_seq[0]!r}"
        )


# ===========================================================================
# D6. Cache lifecycle
# ===========================================================================


class TestCacheLifecycle:
    def test_clear_cache(self, device):
        """After _BUILD_CACHE.clear(), next build is a cache miss."""
        _BUILD_CACHE.clear()

        q1, kv1, _ = _make_branches(device, seed=100)
        Parallel(q1, kv1).build().launch()
        assert len(_BUILD_CACHE) == 1

        _BUILD_CACHE.clear()
        assert len(_BUILD_CACHE) == 0

        q2, kv2, _ = _make_branches(device, seed=200)
        fused = Parallel(q2, kv2).build()
        fused.launch()
        assert len(_BUILD_CACHE) == 1, "Should have re-added a cache entry after clear"

    def test_single_op_not_cached(self, device):
        """Sequential(single_op) doesn't create a cache entry."""
        _BUILD_CACHE.clear()

        q, _, _ = _make_branches(device, seed=42)
        fused = Sequential(q).build()
        assert len(_BUILD_CACHE) == 0, "Single-op Sequential should not create a cache entry"
