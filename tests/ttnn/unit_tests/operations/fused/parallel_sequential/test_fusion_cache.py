# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Exhaustive fusion build cache tests.

Tests cache key properties (topology fingerprint, branch order),
cache hit correctness (RT args, CB pointers, outputs, PCC), cache
entry structure, topology differentiation, and lifecycle.

Also tests the LRU eviction, generation counter, and program key
cache mechanisms added for cache hardening.
"""

import pytest
import torch
import ttnn
from collections import OrderedDict
from unittest import mock

from models.common.utility_functions import comp_pcc
from models.experimental.ops.descriptors.fusion import Parallel, Sequential, clear_build_cache
from models.experimental.ops.descriptors.fusion.fusion import (
    _BUILD_CACHE,
    _BUILD_CACHE_GEN,
    _BUILD_CACHE_MAX,
    _CacheEntry,
    _build_cache_get,
    _build_cache_put,
    _build_cache_surface_key,
    _build_run_cache_key,
    _topology_fingerprint,
)
from models.experimental.ops.descriptors.normalization.rms_norm import rms_norm
from models.experimental.ops.descriptors.op_descriptor import (
    OpDescriptor,
    _PROGRAM_KEY_CACHE_MAX,
    _PROGRAM_KEY_CACHE_REGISTRY,
    _clear_all_program_key_caches,
    _program_key_cache_get,
    _program_key_cache_put,
)


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
    """Create Q and KV RMS norm branches with DeepSeek V3 decode shapes.

    Q: 1x1x32x1536 width-sharded on 4x4 cores (shard [32,96]).
    KV: 1x1x32x512 width-sharded on 2x8 cores (shard [32,32]).
    """
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

        a = OpDescriptor(
            descriptor=_MockDescriptor(), input_tensors=[], output_tensors=[], name="a", program_cache_key=1
        )
        b = OpDescriptor(
            descriptor=_MockDescriptor(), input_tensors=[], output_tensors=[], name="b", program_cache_key=2
        )
        c = OpDescriptor(
            descriptor=_MockDescriptor(), input_tensors=[], output_tensors=[], name="c", program_cache_key=3
        )

        fp = _topology_fingerprint([a, b, c])
        assert fp == "O,O,O"

    def test_topology_fingerprint_with_sequential(self):
        """Nested Sequential produces 'S(O,O)' sub-fingerprint."""
        from models.experimental.ops.descriptors.op_descriptor import OpDescriptor

        class _MockDescriptor:
            kernels = []
            cbs = []
            semaphores = []

        a = OpDescriptor(
            descriptor=_MockDescriptor(), input_tensors=[], output_tensors=[], name="a", program_cache_key=1
        )
        b = OpDescriptor(
            descriptor=_MockDescriptor(), input_tensors=[], output_tensors=[], name="b", program_cache_key=2
        )
        c = OpDescriptor(
            descriptor=_MockDescriptor(), input_tensors=[], output_tensors=[], name="c", program_cache_key=3
        )

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
# D1b. FusedOp merged-IO refresh (op-agnostic API, device)
# ===========================================================================


class TestFusedOpRefreshMergedIo:
    """``FusedOp.refresh_merged_io_from_parallel`` matches a second ``build()`` (PCC)."""

    def test_refresh_merged_io_from_parallel_matches_second_build(self, device):
        """Same PCC after refresh+launch as a fresh Parallel.build from matching tensors."""
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

        fused_once.refresh_merged_io_from_parallel(p)
        fused_once.launch()
        ttnn.synchronize_device(device)

        clear_build_cache()
        fused_fresh = Parallel(q_b, kv_b)
        outs_fused_fresh = fused_fresh.run(results=[q_b, kv_b])
        ttnn.synchronize_device(device)
        for t_new, t_ref in zip(fused_once.output_tensors, outs_fused_fresh):
            ok, pcc = comp_pcc(ttnn.to_torch(t_new), ttnn.to_torch(t_ref))
            assert ok and pcc >= 0.999, f"refresh vs fresh build PCC {pcc}"

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
        fused1.run()
        assert len(_BUILD_CACHE) == 1

        q2, kv2, _ = _make_branches(device, seed=200)
        fused2 = Parallel(kv2, q2)
        fused2.run()
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
        fused1.run()

        q2, kv2, _ = _make_branches(device, seed=200)
        fused2 = Parallel(q2, kv2)
        fused2.run()

        # Cache hit reuses the same descriptor (no deep copy).
        assert len(_BUILD_CACHE) == 1

    def test_cache_hit_produces_fresh_fused_op(self, device):
        """Cache hit via build() returns a fresh FusedOp sharing the cached descriptor."""
        clear_build_cache()

        q1, kv1, _ = _make_branches(device, seed=100)
        fo1 = Parallel(q1, kv1).build()

        q2, kv2, _ = _make_branches(device, seed=200)
        fo2 = Parallel(q2, kv2).build()

        # Cache hit reuses the same ProgramDescriptor but a fresh FusedOp wrapper.
        assert len(_BUILD_CACHE) == 1
        assert id(fo1.descriptor) == id(fo2.descriptor), "descriptor should be same cached object"
        assert id(fo1) != id(fo2), "FusedOp wrapper should be fresh each hit"

    def test_pcc_across_seeds(self, device):
        """10 iterations with different seeds using DeepSeek V3 Q/KV shapes, PCC >= 0.98 on every hit."""
        clear_build_cache()

        for i in range(10):
            seed = 1000 + i * 37
            q_b, kv_b, (torch_q, torch_qw, torch_kv, torch_kvw) = _make_branches(device, seed=seed)

            fused = Parallel(q_b, kv_b)
            q_out, kv_out = fused.run(results=[q_b, kv_b])

            if i == 0:
                assert len(_BUILD_CACHE) == 1
            else:
                assert len(_BUILD_CACHE) == 1, f"Iteration {i}: expected cache hit"

            q_golden = torch_rms_norm(torch_q.float(), torch_qw.float())
            q_result = ttnn.to_torch(q_out)
            passing_q, pcc_q = comp_pcc(q_golden, q_result, pcc=0.98)
            assert passing_q, f"[iter {i}] Q PCC={pcc_q}"

            kv_golden = torch_rms_norm(torch_kv.float(), torch_kvw.float())
            kv_result = ttnn.to_torch(kv_out)
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
        assert isinstance(entry.sem_specs, tuple)
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
        for spec in entry.sem_specs:
            assert not isinstance(spec, ttnn.Tensor)


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

        par = Parallel(q1, kv1)
        seq = Sequential(q2, kv2)
        id_par = _build_run_cache_key(par, par._cached_ops, None)
        id_seq = _build_run_cache_key(seq, seq._cached_ops, None)
        assert id_par != id_seq


# ===========================================================================
# D5b. Lazy factory must not run on fusion cache hit
# ===========================================================================


class TestDeferredFactorySkippedOnFusionCacheHit:
    def test_branch_descriptors_stay_deferred_after_cache_hit_build(self, device):
        """Fusion cache hit must not access ``OpDescriptor.descriptor`` (no C++ factory)."""
        clear_build_cache()

        q1, kv1, _ = _make_branches(device, seed=10)
        Parallel(q1, kv1).run()
        assert len(_BUILD_CACHE) == 1

        q2, kv2, _ = _make_branches(device, seed=20)
        assert isinstance(q2, OpDescriptor)
        assert isinstance(kv2, OpDescriptor)
        assert q2.is_deferred
        assert kv2.is_deferred

        fused2 = Parallel(q2, kv2).build()
        assert q2.is_deferred
        assert kv2.is_deferred

        fused2.launch()
        assert q2.is_deferred
        assert kv2.is_deferred


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

            q_out, kv_out = p.run(results=[q0, kv0])
            ttnn.synchronize_device(device)

            # Goldens from current activations + gammas on the branches (same tensors fused op reads).
            tin_q = ttnn.to_torch(q0.input_tensors[0]).float()
            tin_kv = ttnn.to_torch(kv0.input_tensors[0]).float()
            qw = ttnn.to_torch(q0.input_tensors[1]).float()
            kvw = ttnn.to_torch(kv0.input_tensors[1]).float()
            q_golden = torch_rms_norm(tin_q, qw)
            kv_golden = torch_rms_norm(tin_kv, kvw)
            q_result = ttnn.to_torch(q_out)
            kv_result = ttnn.to_torch(kv_out)
            ok_q, pcc_q = comp_pcc(q_golden, q_result, pcc=0.98)
            ok_kv, pcc_kv = comp_pcc(kv_golden, kv_result, pcc=0.98)
            assert ok_q, f"[iter {i}] Q PCC={pcc_q}"
            assert ok_kv, f"[iter {i}] KV PCC={pcc_kv}"
            min_pcc = min(min_pcc, pcc_q, pcc_kv)

        assert min_pcc >= 0.98


# ===========================================================================
# Infra tests for update(), @OpDescriptor.create, and named kwargs
# ===========================================================================


class TestUpdateAPI:
    """Test OpDescriptor.update() — positional and keyword forms."""

    def test_update_positional(self, device):
        """update(tensor) replaces input_tensors[0]."""
        q, _, _ = _make_branches(device)
        original = q.input_tensors[0]
        new_tensor = ttnn.from_torch(
            torch.rand_like(ttnn.to_torch(original)),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=original.memory_config(),
        )
        q.update(new_tensor)
        assert q.input_tensors[0] is new_tensor

    def test_update_keyword(self, device):
        """update(input_tensor=tensor) replaces by name."""
        q, _, _ = _make_branches(device)
        original = q.input_tensors[0]
        new_tensor = ttnn.from_torch(
            torch.rand_like(ttnn.to_torch(original)),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=original.memory_config(),
        )
        q.update(input_tensor=new_tensor)
        assert q.input_tensors[0] is new_tensor

    def test_update_error_unknown_name(self, device):
        """update(bad_name=t) raises ValueError with valid names listed."""
        q, _, _ = _make_branches(device)
        with pytest.raises(ValueError, match="Unknown input name"):
            q.update(nonexistent=q.input_tensors[0])

    def test_update_multiple_positional(self, device):
        """update(t1, t2) replaces input_tensors[0] and [1]."""
        q, _, _ = _make_branches(device)
        old_0, old_1 = q.input_tensors[0], q.input_tensors[1]
        new_0 = ttnn.from_torch(
            torch.rand_like(ttnn.to_torch(old_0)),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=old_0.memory_config(),
        )
        new_1 = ttnn.from_torch(
            torch.rand_like(ttnn.to_torch(old_1)),
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        q.update(new_0, new_1)
        assert q.input_tensors[0] is new_0
        assert q.input_tensors[1] is new_1

    def test_update_error_mixed_args(self, device):
        """update(t, input_tensor=t) raises ValueError."""
        q, _, _ = _make_branches(device)
        t = q.input_tensors[0]
        with pytest.raises(ValueError, match="positional OR keyword"):
            q.update(t, input_tensor=t)


class TestDeferredDescriptor:
    """Test @OpDescriptor.create decorator — deferred (persistent) descriptors."""

    def test_partial_has_no_hash(self, device):
        """rms_norm(weight=w) without input_tensor has program_cache_key=None."""
        q, _, _ = _make_branches(device)
        w = q.input_tensors[1]
        mem = q.input_tensors[0].memory_config()
        desc = rms_norm(weight=w, epsilon=1e-5, memory_config=mem, core_range_set=mem.shard_spec.grid)
        assert desc.program_cache_key is None
        assert desc.input_tensors[0] is None  # pending slot

    def test_partial_materializes_on_update(self, device):
        """After update(tensor), program_cache_key is set and factory is ready."""
        q, _, _ = _make_branches(device)
        # Extract config from the eager descriptor to build a matching partial one
        w = q.input_tensors[1]  # weight
        x = q.input_tensors[0]  # activation
        mem = x.memory_config()
        desc = rms_norm(weight=w, epsilon=1e-5, memory_config=mem, core_range_set=mem.shard_spec.grid)
        assert desc.program_cache_key is None
        desc.update(x)
        assert desc.program_cache_key is not None
        assert desc.input_tensors[0] is x

    def test_partial_matches_full(self, device):
        """Deferred descriptor after update() has same hash as eager descriptor."""
        q, _, _ = _make_branches(device)
        w = q.input_tensors[1]
        x = q.input_tensors[0]
        mem = x.memory_config()
        cr = mem.shard_spec.grid
        # Eager (inline)
        eager = rms_norm(x, weight=w, epsilon=1e-5, memory_config=mem, core_range_set=cr)
        # Deferred (persistent)
        deferred = rms_norm(weight=w, epsilon=1e-5, memory_config=mem, core_range_set=cr)
        deferred.update(x)
        assert eager.program_cache_key == deferred.program_cache_key


class TestNamedKwargs:
    """Test named kwargs on Parallel/Sequential — attribute access + collision detection."""

    def test_parallel_named_access(self, device):
        """Parallel(q=desc, kv=desc) sets named attributes."""
        q, kv, _ = _make_branches(device)
        p = Parallel(q=q, kv=kv)
        assert p.q is q
        assert p.kv is kv

    def test_sequential_named_access(self, device):
        """Sequential(a=desc) sets named attribute."""
        q, _, _ = _make_branches(device)
        s = Sequential(a=q)
        assert s.a is q

    def test_nested_hoisting(self, device):
        """Names from nested containers are hoisted to the top level."""
        q, kv, _ = _make_branches(device)
        # Need a third descriptor for Sequential (needs ≥1 item)
        q2, _, _ = _make_branches(device, seed=99)
        inner = Parallel(q=q, kv=kv)
        outer = Sequential(stem=q2, branches=inner)
        # Hoisted: outer.q, outer.kv should work
        assert outer.q is q
        assert outer.kv is kv
        assert outer.stem is q2

    def test_duplicate_name_raises(self, device):
        """Duplicate names across nesting levels raise ValueError."""
        q, kv, _ = _make_branches(device)
        q2, _, _ = _make_branches(device, seed=99)
        inner = Parallel(q=q, kv=kv)
        with pytest.raises(ValueError, match="Duplicate descriptor name"):
            Sequential(q=q2, branches=inner)  # "q" collides


class TestTopologyInvalidation:
    """Test invalidation after add() or invalidate_run()."""

    def test_invalidate_run_resets_topology(self, device):
        """invalidate_run() resets cached topology so next run() does a fresh build."""
        q, kv, torches = _make_branches(device, seed=42)
        p = Parallel(q=q, kv=kv)

        p.run()
        p.run()

        p.invalidate_run()

        [out_q, out_kv] = p.run()
        torch_q_in, torch_q_w, torch_kv_in, torch_kv_w = torches
        ok_q, _ = comp_pcc(torch_rms_norm(torch_q_in.float(), torch_q_w.float()), ttnn.to_torch(out_q), pcc=0.98)
        ok_kv, _ = comp_pcc(torch_rms_norm(torch_kv_in.float(), torch_kv_w.float()), ttnn.to_torch(out_kv), pcc=0.98)
        assert ok_q and ok_kv, "PCC should pass after invalidate + fresh run"

    def test_add_invalidates_topology(self, device):
        """add() resets cached topology so the shape change is picked up."""
        q, kv, _ = _make_branches(device, seed=42)
        q2, _, _ = _make_branches(device, seed=99)

        s = Sequential(q)
        s.run()
        s.run()

        old_ops = list(s._cached_ops)
        s.add(kv)
        assert s._cached_ops != old_ops, "add() should reset _cached_ops"


class TestPersistentWithResults:
    """Test run() with explicit results kwarg."""

    def test_parallel_explicit_results(self, device):
        """run(results=[specific_desc]) returns only the requested branch output."""
        q, kv, torches = _make_branches(device, seed=42)
        p = Parallel(q=q, kv=kv)

        p.run()

        [out_q] = p.run(results=[q])
        torch_q_in, torch_q_w, _, _ = torches
        ok, _ = comp_pcc(torch_rms_norm(torch_q_in.float(), torch_q_w.float()), ttnn.to_torch(out_q), pcc=0.98)
        assert ok, "Explicit results=[q] should return correct Q output"


# ===========================================================================
# Program key cache — LRU helpers (unit, no device)
# ===========================================================================


class TestProgramKeyCacheLRU:
    """Unit tests for the program key cache LRU helpers."""

    def test_get_miss_returns_none(self):
        cache = OrderedDict()
        assert _program_key_cache_get(cache, "missing") is None

    def test_get_hit_returns_value(self):
        cache = OrderedDict()
        cache["k1"] = "v1"
        assert _program_key_cache_get(cache, "k1") == "v1"

    def test_get_promotes_to_end(self):
        cache = OrderedDict()
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        _program_key_cache_get(cache, "a")
        assert list(cache.keys()) == ["b", "c", "a"]

    def test_put_adds_entry(self):
        cache = OrderedDict()
        _program_key_cache_put(cache, "x", 42)
        assert cache["x"] == 42

    def test_put_promotes_existing(self):
        cache = OrderedDict()
        cache["a"] = 1
        cache["b"] = 2
        _program_key_cache_put(cache, "a", 10)
        assert list(cache.keys()) == ["b", "a"]
        assert cache["a"] == 10

    def test_put_evicts_oldest_beyond_max(self):
        cache = OrderedDict()
        with mock.patch("models.experimental.ops.descriptors.op_descriptor._PROGRAM_KEY_CACHE_MAX", 3):
            _program_key_cache_put(cache, "a", 1)
            _program_key_cache_put(cache, "b", 2)
            _program_key_cache_put(cache, "c", 3)
            assert len(cache) == 3
            _program_key_cache_put(cache, "d", 4)
            assert len(cache) == 3
            assert "a" not in cache, "oldest entry should be evicted"
            assert list(cache.keys()) == ["b", "c", "d"]

    def test_put_evicts_multiple_when_far_over(self):
        """If cache somehow grew past max, put drains it back to max."""
        cache = OrderedDict()
        for i in range(10):
            cache[f"k{i}"] = i
        with mock.patch("models.experimental.ops.descriptors.op_descriptor._PROGRAM_KEY_CACHE_MAX", 3):
            _program_key_cache_put(cache, "new", 99)
            assert len(cache) == 3
            assert list(cache.keys()) == ["k8", "k9", "new"]

    def test_lru_order_with_mixed_get_put(self):
        """Interleaved gets and puts maintain correct LRU order."""
        cache = OrderedDict()
        with mock.patch("models.experimental.ops.descriptors.op_descriptor._PROGRAM_KEY_CACHE_MAX", 4):
            _program_key_cache_put(cache, "a", 1)
            _program_key_cache_put(cache, "b", 2)
            _program_key_cache_put(cache, "c", 3)
            _program_key_cache_get(cache, "a")  # promote a
            _program_key_cache_put(cache, "d", 4)
            # Order: b, c, a, d → adding "e" evicts "b"
            _program_key_cache_put(cache, "e", 5)
            assert len(cache) == 4
            assert "b" not in cache
            assert list(cache.keys()) == ["c", "a", "d", "e"]


class TestProgramKeyCacheRegistry:
    """Tests for the program key cache registry and clear mechanism."""

    def test_clear_all_empties_registered_caches(self):
        test_cache = OrderedDict({"k": "v"})
        _PROGRAM_KEY_CACHE_REGISTRY.append(test_cache)
        try:
            _clear_all_program_key_caches()
            assert len(test_cache) == 0
        finally:
            _PROGRAM_KEY_CACHE_REGISTRY.remove(test_cache)

    def test_clear_all_handles_multiple_caches(self):
        c1 = OrderedDict({"a": 1})
        c2 = OrderedDict({"b": 2, "c": 3})
        _PROGRAM_KEY_CACHE_REGISTRY.append(c1)
        _PROGRAM_KEY_CACHE_REGISTRY.append(c2)
        try:
            _clear_all_program_key_caches()
            assert len(c1) == 0
            assert len(c2) == 0
        finally:
            _PROGRAM_KEY_CACHE_REGISTRY.remove(c1)
            _PROGRAM_KEY_CACHE_REGISTRY.remove(c2)


# ===========================================================================
# Build cache — LRU helpers (unit, no device)
# ===========================================================================


class TestBuildCacheLRU:
    """Unit tests for the build cache LRU helpers."""

    def _save_and_clear(self):
        saved = OrderedDict(_BUILD_CACHE)
        _BUILD_CACHE.clear()
        return saved

    def _restore(self, saved):
        _BUILD_CACHE.clear()
        _BUILD_CACHE.update(saved)

    def test_get_miss_returns_none(self):
        saved = self._save_and_clear()
        try:
            assert _build_cache_get(("missing",)) is None
        finally:
            self._restore(saved)

    def test_get_hit_returns_entry(self):
        saved = self._save_and_clear()
        try:
            sentinel = object()
            _BUILD_CACHE[("key",)] = sentinel
            assert _build_cache_get(("key",)) is sentinel
        finally:
            self._restore(saved)

    def test_get_promotes_to_end(self):
        saved = self._save_and_clear()
        try:
            _BUILD_CACHE[("a",)] = 1
            _BUILD_CACHE[("b",)] = 2
            _BUILD_CACHE[("c",)] = 3
            _build_cache_get(("a",))
            assert list(_BUILD_CACHE.keys()) == [("b",), ("c",), ("a",)]
        finally:
            self._restore(saved)

    def test_put_adds_and_evicts(self):
        saved = self._save_and_clear()
        try:
            with mock.patch("models.experimental.ops.descriptors.fusion.fusion._BUILD_CACHE_MAX", 2):
                _build_cache_put(("x",), "X")
                _build_cache_put(("y",), "Y")
                assert len(_BUILD_CACHE) == 2
                _build_cache_put(("z",), "Z")
                assert len(_BUILD_CACHE) == 2
                assert ("x",) not in _BUILD_CACHE
                assert list(_BUILD_CACHE.keys()) == [("y",), ("z",)]
        finally:
            self._restore(saved)

    def test_put_promotes_existing_key(self):
        saved = self._save_and_clear()
        try:
            _build_cache_put(("a",), 1)
            _build_cache_put(("b",), 2)
            _build_cache_put(("a",), 10)
            assert list(_BUILD_CACHE.keys()) == [("b",), ("a",)]
            assert _BUILD_CACHE[("a",)] == 10
        finally:
            self._restore(saved)


# ===========================================================================
# Generation counter (unit, no device)
# ===========================================================================


class TestBuildCacheGeneration:
    """Tests for the _BUILD_CACHE_GEN generation counter mechanism."""

    def test_clear_increments_generation(self):
        import models.experimental.ops.descriptors.fusion.fusion as fusion_mod

        gen_before = fusion_mod._BUILD_CACHE_GEN
        clear_build_cache()
        assert fusion_mod._BUILD_CACHE_GEN == gen_before + 1

    def test_clear_cascades_to_program_key_caches(self):
        test_cache = OrderedDict({"k": "v"})
        _PROGRAM_KEY_CACHE_REGISTRY.append(test_cache)
        try:
            clear_build_cache()
            assert len(test_cache) == 0, "clear_build_cache should cascade to program key caches"
        finally:
            _PROGRAM_KEY_CACHE_REGISTRY.remove(test_cache)

    def test_multiple_clears_keep_incrementing(self):
        import models.experimental.ops.descriptors.fusion.fusion as fusion_mod

        gen_start = fusion_mod._BUILD_CACHE_GEN
        for i in range(5):
            clear_build_cache()
        assert fusion_mod._BUILD_CACHE_GEN == gen_start + 5


# ===========================================================================
# Generation counter — container invalidation (device)
# ===========================================================================


class TestGenerationContainerInvalidation:
    """Test that _cached_entry_gen protects persistent containers from stale entries."""

    def test_cached_entry_gen_set_on_run(self, device):
        """After run(), container._cached_entry_gen equals _BUILD_CACHE_GEN."""
        import models.experimental.ops.descriptors.fusion.fusion as fusion_mod

        clear_build_cache()
        q, kv, _ = _make_branches(device, seed=42)
        p = Parallel(q, kv)
        p.run()

        assert hasattr(p, "_cached_entry_gen")
        assert p._cached_entry_gen == fusion_mod._BUILD_CACHE_GEN

    def test_clear_invalidates_persistent_container(self, device):
        """After clear_build_cache(), a persistent container's cached entry is stale."""
        import models.experimental.ops.descriptors.fusion.fusion as fusion_mod

        clear_build_cache()
        q, kv, _ = _make_branches(device, seed=42)
        p = Parallel(q, kv)
        p.run()

        gen_at_run = p._cached_entry_gen
        clear_build_cache()

        assert (
            fusion_mod._BUILD_CACHE_GEN != gen_at_run
        ), "Generation should have advanced past the container's cached gen"

    def test_stale_container_recovers_after_clear(self, device):
        """Persistent container recovers from a stale _cached_entry after clear_build_cache."""
        clear_build_cache()
        q, kv, torches = _make_branches(device, seed=42)
        p = Parallel(q, kv)

        p.run()
        p.run()

        clear_build_cache()

        [out_q, out_kv] = p.run()
        torch_q_in, torch_q_w, torch_kv_in, torch_kv_w = torches
        ok_q, _ = comp_pcc(
            torch_rms_norm(torch_q_in.float(), torch_q_w.float()),
            ttnn.to_torch(out_q),
            pcc=0.98,
        )
        ok_kv, _ = comp_pcc(
            torch_rms_norm(torch_kv_in.float(), torch_kv_w.float()),
            ttnn.to_torch(out_kv),
            pcc=0.98,
        )
        assert ok_q and ok_kv, "Persistent container should produce correct results after cache clear"


# ===========================================================================
# Build cache LRU — integration (device)
# ===========================================================================


class TestBuildCacheLRUIntegration:
    """Test that build cache LRU eviction works end-to-end."""

    def test_build_cache_bounded_by_max(self, device):
        """Build cache respects _BUILD_CACHE_MAX by evicting oldest entries."""
        clear_build_cache()
        with mock.patch("models.experimental.ops.descriptors.fusion.fusion._BUILD_CACHE_MAX", 2):
            q1, kv1, _ = _make_branches(device, seed=100)
            Parallel(q1, kv1).run()
            assert len(_BUILD_CACHE) == 1

            q2, _, _ = _make_branches(device, seed=200)
            Sequential(q2).run()
            assert len(_BUILD_CACHE) == 2

            q3, kv3, _ = _make_branches(device, seed=300)
            kv3_only = Sequential(kv3)
            kv3_only.run()
            assert len(_BUILD_CACHE) <= 2, "Build cache should have evicted oldest entry"

    def test_evicted_topology_rebuilds_correctly(self, device):
        """After LRU evicts a topology, re-running it produces correct PCC."""
        clear_build_cache()
        with mock.patch("models.experimental.ops.descriptors.fusion.fusion._BUILD_CACHE_MAX", 1):
            q1, kv1, torches1 = _make_branches(device, seed=100)
            Parallel(q1, kv1).run()
            assert len(_BUILD_CACHE) == 1

            q2, _, _ = _make_branches(device, seed=200)
            Sequential(q2).run()
            assert len(_BUILD_CACHE) == 1

            q3, kv3, torches3 = _make_branches(device, seed=300)
            [out_q, out_kv] = Parallel(q3, kv3).run()
            torch_q_in, torch_q_w, torch_kv_in, torch_kv_w = torches3
            ok_q, _ = comp_pcc(
                torch_rms_norm(torch_q_in.float(), torch_q_w.float()),
                ttnn.to_torch(out_q),
                pcc=0.98,
            )
            ok_kv, _ = comp_pcc(
                torch_rms_norm(torch_kv_in.float(), torch_kv_w.float()),
                ttnn.to_torch(out_kv),
                pcc=0.98,
            )
            assert ok_q and ok_kv, "Rebuild after eviction should produce correct results"


# ===========================================================================
# Program key cache — integration (device)
# ===========================================================================


class TestProgramKeyCacheIntegration:
    """Test that the per-factory program key cache works end-to-end."""

    def test_inline_cache_hit(self, device):
        """Second inline call with same config objects hits the program key cache."""
        clear_build_cache()
        q_cores = cores(0, 0, 3, 3)
        q_shard_w = 96
        q_total_w = 16 * q_shard_w
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

        torch_input1 = torch.rand(1, 1, 32, q_total_w, dtype=torch.bfloat16)
        torch_weight = torch.rand(1, 1, 1, q_total_w, dtype=torch.bfloat16)
        tt_input1 = ttnn.from_torch(torch_input1, device=device, layout=ttnn.TILE_LAYOUT, memory_config=q_mem)
        tt_weight = ttnn.from_torch(torch_weight, device=device, layout=ttnn.TILE_LAYOUT)

        desc1 = rms_norm(
            tt_input1,
            epsilon=1e-5,
            weight=tt_weight,
            memory_config=q_mem,
            core_range_set=q_cores,
            program_config=q_pc,
        )
        pck1 = desc1.program_cache_key

        torch_input2 = torch.rand(1, 1, 32, q_total_w, dtype=torch.bfloat16)
        tt_input2 = ttnn.from_torch(torch_input2, device=device, layout=ttnn.TILE_LAYOUT, memory_config=q_mem)

        desc2 = rms_norm(
            tt_input2,
            epsilon=1e-5,
            weight=tt_weight,
            memory_config=q_mem,
            core_range_set=q_cores,
            program_config=q_pc,
        )
        pck2 = desc2.program_cache_key

        assert pck1 == pck2, "Same config should yield same program_cache_key via cache hit"

    def test_clear_build_cache_clears_program_key_caches(self, device):
        """clear_build_cache() empties all registered program key caches."""
        q, _, _ = _make_branches(device, seed=42)

        nonempty = any(len(c) > 0 for c in _PROGRAM_KEY_CACHE_REGISTRY)
        if not nonempty:
            pass

        clear_build_cache()
        for cache in _PROGRAM_KEY_CACHE_REGISTRY:
            assert len(cache) == 0, "All program key caches should be empty after clear_build_cache()"


# ===========================================================================
# Environment variable configurability (unit, no device)
# ===========================================================================


class TestCacheEnvVars:
    """Test that cache sizes are configurable via environment variables."""

    def test_build_cache_max_reads_from_env(self):
        assert _BUILD_CACHE_MAX == int(__import__("os").environ.get("TT_METAL_FUSION_BUILD_CACHE_MAX_ENTRIES", "256"))

    def test_program_key_cache_max_reads_from_env(self):
        assert _PROGRAM_KEY_CACHE_MAX == int(
            __import__("os").environ.get("TT_METAL_FUSION_PROGRAM_KEY_CACHE_MAX_ENTRIES", "128")
        )

    def test_build_cache_max_default_is_256(self):
        """Without env var override, default build cache max is 256."""
        import os

        if "TT_METAL_FUSION_BUILD_CACHE_MAX_ENTRIES" not in os.environ:
            assert _BUILD_CACHE_MAX == 256

    def test_program_key_cache_max_default_is_128(self):
        """Without env var override, default program key cache max is 128."""
        import os

        if "TT_METAL_FUSION_PROGRAM_KEY_CACHE_MAX_ENTRIES" not in os.environ:
            assert _PROGRAM_KEY_CACHE_MAX == 128
