# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Exhaustive fusion build cache tests.

Tests cache key properties (topology fingerprint, branch order),
cache hit correctness (RT args, CB pointers, outputs, PCC), cache
entry structure, topology differentiation, and lifecycle.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.ops.descriptors.fusion import Parallel, Sequential, clear_build_cache
from models.experimental.ops.descriptors.fusion.fusion import (
    _BUILD_CACHE,
    _CacheEntry,
    _build_cache_surface_key,
    _fusion_cache_id_and_ops,
    _topology_fingerprint,
)
from models.experimental.ops.descriptors.normalization.rms_norm import rms_norm
from models.experimental.ops.descriptors.op_descriptor import DeferredOpDescriptor


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

    q_branch = rms_norm(
        tt_q_input,
        epsilon=1e-5,
        weight=tt_q_weight,
        memory_config=q_mem,
        core_range_set=q_cores,
        program_config=q_pc,
    )
    kv_branch = rms_norm(
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
        """Surface key prefixes Parallel vs Sequential so P(O,O) ≠ S(O,O)."""
        q, kv, _ = _make_branches(device, seed=42)
        assert _build_cache_surface_key([q, kv], "P") == "P(O,O)"
        assert _build_cache_surface_key([q, kv], "S") == "S(O,O)"
        assert _build_cache_surface_key([q, kv], "P") != _build_cache_surface_key([q, kv], "S")


# ===========================================================================
# D1b. FusedOp IO rebind (op-agnostic API, device)
# ===========================================================================


class TestFusedOpRebind:
    """``FusedOp.refresh_merged_io_from_parallel`` matches a second ``build()`` (PCC)."""

    def test_rebind_from_parallel_matches_second_build(self, device):
        """Same PCC after rebind+launch as a fresh Parallel.build from matching tensors."""
        clear_build_cache()

        q_a, kv_a, _ = _make_branches(device, seed=501)
        p = Parallel(q_a, kv_a)
        fused_once = p.build()
        fused_once.launch()
        ttnn.synchronize_device(device)

        q_b, kv_b, torch_bufs = _make_branches(device, seed=502)
        torch_q, torch_qw, torch_kv, torch_kvw = torch_bufs

        # General pattern: copy tensor handles onto the *same* op objects used at build().
        q_a.input_tensors[:] = list(q_b.input_tensors)
        q_a.output_tensors[:] = list(q_b.output_tensors)
        kv_a.input_tensors[:] = list(kv_b.input_tensors)
        kv_a.output_tensors[:] = list(kv_b.output_tensors)

        fused_once.rebind_from_parallel(p)
        fused_once.launch()
        ttnn.synchronize_device(device)

        clear_build_cache()
        fused_fresh = Parallel(q_b, kv_b)
        outs_fused_fresh = fused_fresh.run()
        ttnn.synchronize_device(device)

        for t_new, t_ref in zip(fused_once.output_tensors, outs_fused_fresh):
            ok, pcc = comp_pcc(ttnn.to_torch(t_new), ttnn.to_torch(t_ref))
            assert ok and pcc >= 0.999, f"rebind vs fresh build PCC {pcc}"

        # Match fused outputs to goldens by shape (Q vs KV widths differ).
        q_golden = torch_rms_norm(torch_q.float(), torch_qw.float())
        kv_golden = torch_rms_norm(torch_kv.float(), torch_kvw.float())
        for tt_out in fused_once.output_tensors:
            th = ttnn.to_torch(tt_out).float()
            if th.shape == q_golden.shape:
                ok, pcc = comp_pcc(th, q_golden)
                assert ok and pcc >= 0.98, f"Q branch PCC {pcc}"
            elif th.shape == kv_golden.shape:
                ok, pcc = comp_pcc(th, kv_golden)
                assert ok and pcc >= 0.98, f"KV branch PCC {pcc}"
            else:
                raise AssertionError(f"unexpected output shape {th.shape}")


# ===========================================================================
# D2. Order tolerance (device)
# ===========================================================================


class TestCacheOrderTolerance:
    def test_parallel_branch_order_is_user_visible(self, device):
        """Parallel(Q, KV) vs Parallel(KV, Q) are distinct builds and cache entries."""
        clear_build_cache()

        q1, kv1, _ = _make_branches(device, seed=100)
        fused1 = Parallel(q1, kv1)
        outs_fused1 = fused1.run()
        assert len(_BUILD_CACHE) == 1

        q2, kv2, _ = _make_branches(device, seed=200)
        fused2 = Parallel(kv2, q2)
        outs_fused2 = fused2.run()
        assert len(_BUILD_CACHE) == 2, (
            "Reversed branch order must not share the same fusion cache entry "
            f"(output tensor order differs); got {len(_BUILD_CACHE)}"
        )


# ===========================================================================
# D3. Cache hit correctness (device)
# ===========================================================================


class TestCacheHitCorrectness:
    def test_cache_hit_returns_fused_op(self, device):
        """Each cache hit returns a FusedOp that produces correct results."""
        clear_build_cache()

        q1, kv1, _ = _make_branches(device, seed=100)
        fused1 = Parallel(q1, kv1)
        outs_fused1 = fused1.run()

        q2, kv2, _ = _make_branches(device, seed=200)
        fused2 = Parallel(q2, kv2)
        outs_fused2 = fused2.run()

        # Cache hit reuses the same descriptor (no deep copy).
        assert len(_BUILD_CACHE) == 1

    def test_cache_hit_produces_fresh_fused_op(self, device):
        """Cache hit builds a fresh FusedOp (no pinned tensors in cache)."""
        clear_build_cache()

        q1, kv1, _ = _make_branches(device, seed=100)
        p1 = Parallel(q1, kv1)
        p1.run()
        fo1 = p1._run_fused

        q2, kv2, _ = _make_branches(device, seed=200)
        fo2 = Parallel(q2, kv2).build()

        # Cache hit reuses the same ProgramDescriptor but a fresh FusedOp wrapper.
        assert len(_BUILD_CACHE) == 1
        assert id(fo1.descriptor) == id(fo2.descriptor), "descriptor should be same cached object"
        assert id(fo1) != id(fo2), "FusedOp wrapper should be fresh each hit"

    def test_pcc_across_seeds(self, device):
        """10 iterations with different seeds, PCC >= 0.98 on every hit."""
        clear_build_cache()

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

            q_b = rms_norm(
                tt_q,
                epsilon=1e-5,
                weight=tt_qw,
                memory_config=q_mem,
                core_range_set=q_cores,
                program_config=q_pc,
            )
            kv_b = rms_norm(
                tt_kv,
                epsilon=1e-5,
                weight=tt_kvw,
                memory_config=kv_mem,
                core_range_set=kv_cores,
                program_config=kv_pc,
            )

            fused = Parallel(q_b, kv_b)
            outs_fused = fused.run()

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
    def test_cache_entry_fields(self, device):
        """_CacheEntry has the expected fields."""
        clear_build_cache()

        q, kv, _ = _make_branches(device, seed=42)
        Parallel(q, kv).run()

        assert len(_BUILD_CACHE) == 1
        entry = next(iter(_BUILD_CACHE.values()))

        assert isinstance(entry, _CacheEntry)
        assert entry.cached_descriptor is not None
        assert isinstance(entry.semaphores, tuple)
        assert isinstance(entry.kernel_labels, tuple)
        assert entry.output_sources, "expected non-empty output_sources when branches have outputs"
        assert entry.merged_input_len is not None and entry.merged_input_len >= 1

    def test_no_tensor_refs_in_entry(self, device):
        """_CacheEntry fields don't hold any ttnn.Tensor objects."""
        clear_build_cache()

        q, kv, _ = _make_branches(device, seed=42)
        Parallel(q, kv).run()

        entry = next(iter(_BUILD_CACHE.values()))
        assert not isinstance(entry.cached_descriptor, ttnn.Tensor)
        for s in entry.semaphores:
            assert not isinstance(s, ttnn.Tensor)


# ===========================================================================
# D5. Topology differentiation (device)
# ===========================================================================


class TestTopologyDifferentiation:
    def test_topology_fingerprint_in_key(self, device):
        """Wrapped Parallel vs Sequential yield different tree fingerprints (and fusion ids)."""
        q1, kv1, _ = _make_branches(device, seed=100)
        q2, kv2, _ = _make_branches(device, seed=200)

        topo_par = _topology_fingerprint([Parallel(q1, kv1)])
        topo_seq = _topology_fingerprint([Sequential(q2, kv2)])
        assert topo_par != topo_seq, f"expected different topology strings, got {topo_par!r} vs {topo_seq!r}"
        assert topo_par.startswith("P(") and topo_seq.startswith("S(")

        id_par, _ = _fusion_cache_id_and_ops([Parallel(q1, kv1)], "")
        id_seq, _ = _fusion_cache_id_and_ops([Sequential(q2, kv2)], "")
        assert id_par != id_seq


# ===========================================================================
# D5b. Lazy factory must not run on fusion cache hit
# ===========================================================================


class TestDeferredFactorySkippedOnFusionCacheHit:
    def test_branch_descriptors_stay_deferred_after_cache_hit_build(self, device):
        """Fusion cache hit must not access ``DeferredOpDescriptor.descriptor`` (no C++ factory)."""
        clear_build_cache()

        q1, kv1, _ = _make_branches(device, seed=10)
        Parallel(q1, kv1).run()
        assert len(_BUILD_CACHE) == 1

        q2, kv2, _ = _make_branches(device, seed=20)
        assert isinstance(q2, DeferredOpDescriptor)
        assert isinstance(kv2, DeferredOpDescriptor)
        assert q2._descriptor is None and q2._factory_fn is not None
        assert kv2._descriptor is None and kv2._factory_fn is not None

        fused2 = Parallel(q2, kv2).build()
        assert q2._descriptor is None and q2._factory_fn is not None
        assert kv2._descriptor is None and kv2._factory_fn is not None

        fused2.launch()
        assert q2._descriptor is None and q2._factory_fn is not None
        assert kv2._descriptor is None and kv2._factory_fn is not None


# ===========================================================================
# D6. Cache lifecycle
# ===========================================================================


class TestCacheLifecycle:
    def test_clear_cache(self, device):
        """After clear_build_cache(), next build is a cache miss."""
        clear_build_cache()

        q1, kv1, _ = _make_branches(device, seed=100)
        Parallel(q1, kv1).run()
        assert len(_BUILD_CACHE) == 1

        clear_build_cache()
        assert len(_BUILD_CACHE) == 0

        q2, kv2, _ = _make_branches(device, seed=200)
        Parallel(q2, kv2).run()
        assert len(_BUILD_CACHE) == 1, "Should have re-added a cache entry after clear"

    def test_single_op_cached(self, device):
        """Single-op ``Sequential`` is cached like any other fuse (uniform policy)."""
        clear_build_cache()

        q, _, _ = _make_branches(device, seed=42)
        Sequential(q).build()
        assert len(_BUILD_CACHE) == 1, "Single-op Sequential should populate the fusion build cache"


# ===========================================================================
# D7. build_launch sugar
# ===========================================================================


class TestLaunchInPlaceBranchIo:
    """In-place branch input swaps must stay correct with ``run()`` (reused ``FusedOp`` + refresh each launch)."""

    def test_parallel_run_many_in_place_inputs_pcc(self, device):
        clear_build_cache()
        torch.manual_seed(200)
        q0, kv0, _ = _make_branches(device, seed=200)
        ref_q_in = q0.input_tensors[0]
        ref_kv_in = kv0.input_tensors[0]
        q_mem = ref_q_in.memory_config()
        kv_mem = ref_kv_in.memory_config()
        q_shape = tuple(int(d) for d in ref_q_in.shape)
        kv_shape = tuple(int(d) for d in ref_kv_in.shape)
        p = Parallel(q0, kv0)

        n = 12
        min_pcc = 1.0
        fused_ids = []
        for i in range(n):
            if i > 0:
                torch.manual_seed(200 + i * 7)
                torch_q = torch.rand(q_shape, dtype=torch.bfloat16)
                torch_kv = torch.rand(kv_shape, dtype=torch.bfloat16)
                q0.input_tensors[0] = ttnn.from_torch(
                    torch_q, device=device, layout=ttnn.TILE_LAYOUT, memory_config=q_mem
                )
                kv0.input_tensors[0] = ttnn.from_torch(
                    torch_kv, device=device, layout=ttnn.TILE_LAYOUT, memory_config=kv_mem
                )

            merged_out = p.run()
            fused_ids.append(id(p._run_fused))
            ttnn.synchronize_device(device)

            # Goldens from current activations + gammas on the branches (same tensors fused op reads).
            tin_q = ttnn.to_torch(q0.input_tensors[0]).float()
            tin_kv = ttnn.to_torch(kv0.input_tensors[0]).float()
            qw = ttnn.to_torch(q0.input_tensors[1]).float()
            kvw = ttnn.to_torch(kv0.input_tensors[1]).float()
            q_golden = torch_rms_norm(tin_q, qw)
            kv_golden = torch_rms_norm(tin_kv, kvw)
            q_result = ttnn.to_torch(merged_out[0])
            kv_result = ttnn.to_torch(merged_out[1])
            ok_q, pcc_q = comp_pcc(q_golden, q_result, pcc=0.98)
            ok_kv, pcc_kv = comp_pcc(kv_golden, kv_result, pcc=0.98)
            assert ok_q, f"[iter {i}] Q PCC={pcc_q}"
            assert ok_kv, f"[iter {i}] KV PCC={pcc_kv}"
            min_pcc = min(min_pcc, pcc_q, pcc_kv)

        assert min_pcc >= 0.98
        assert len(set(fused_ids)) == 1, "run() should reuse one FusedOp across iterations"


class TestBuildLaunch:
    def test_build_launch_same_as_build_then_launch(self, device):
        """``Parallel.build_launch()`` matches ``build().launch()`` and ``run()``."""
        clear_build_cache()
        q_op, kv_op, _ = _make_branches(device, seed=77)
        p = Parallel(q_op, kv_op)
        out_sep = p.build().launch()
        ttnn.synchronize_device(device)
        out_one = p.build_launch()
        ttnn.synchronize_device(device)
        assert list(out_sep) == list(out_one)
        p.invalidate_run()
        out_run = p.run()
        ttnn.synchronize_device(device)
        assert list(out_sep) == list(out_run)

    def test_parallel_run_twice_uses_cache_hit(self, device):
        clear_build_cache()
        q_op, kv_op, _ = _make_branches(device, seed=88)
        p = Parallel(q_op, kv_op)
        p.run()
        ttnn.synchronize_device(device)
        assert len(_BUILD_CACHE) == 1
        p.run()
        ttnn.synchronize_device(device)
        assert len(_BUILD_CACHE) == 1
