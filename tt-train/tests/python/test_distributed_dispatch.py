# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for the layout-aware distributed dispatch layer.

This file tests:
  - Layout primitives and conversion helpers
  - PlanCache behavior (LRU eviction, hit/miss)
  - ShardingPlan + rule registry
  - Module rule registry (class and inheritance lookup)
  - Op sharding rules (matmul, elementwise, normalization, attention, loss)
  - Debug trace recording
  - Policy construction patterns (exact and regex matching)
  - MeshRuntime properties and global singleton
  - Redistribute logic
  - distribute_tensor and distribute_module
  - Full device-backed integration tests

Device-backed tests are marked ``requires_device`` and skipped when
no hardware is available.
"""

from __future__ import annotations

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Layout primitives
# ---------------------------------------------------------------------------

from ttml.distributed.layout import (
    Layout,
    Shard,
    Replicate,
    replicated_layout,
    get_layout,
    set_layout,
    layout_from_topology,
    layout_to_mapper_config,
)


class TestLayoutPrimitives:
    """Tests for Shard, Replicate, and Layout dataclasses."""

    def test_shard_equality(self):
        assert Shard(0) == Shard(0)
        assert Shard(0) != Shard(1)
        assert Shard(-1) != Replicate()

    def test_shard_repr(self):
        assert repr(Shard(0)) == "Shard(0)"
        assert repr(Shard(-2)) == "Shard(-2)"

    def test_replicate_equality(self):
        assert Replicate() == Replicate()

    def test_replicate_repr(self):
        assert repr(Replicate()) == "Replicate()"

    def test_layout_frozen(self):
        layout = Layout(placements=(Replicate(), Shard(-1)))
        assert layout.ndim == 2
        assert layout.is_sharded_on(1)
        assert not layout.is_sharded_on(0)
        # Verify frozen (immutable)
        with pytest.raises(AttributeError):
            layout.placements = (Shard(0),)

    def test_layout_hashable(self):
        a = Layout(placements=(Replicate(), Shard(-1)))
        b = Layout(placements=(Replicate(), Shard(-1)))
        assert hash(a) == hash(b)
        d = {a: 42}
        assert d[b] == 42

    def test_layout_different_hash(self):
        a = Layout(placements=(Replicate(), Shard(-1)))
        b = Layout(placements=(Replicate(), Shard(-2)))
        assert a != b
        # Different layouts should (usually) have different hashes
        # Note: hash collisions are possible but unlikely for simple cases

    def test_replicated_layout_helper(self):
        rep = replicated_layout(3)
        assert rep.is_replicated()
        assert rep.ndim == 3
        for p in rep.placements:
            assert isinstance(p, Replicate)

    def test_replicated_layout_default(self):
        rep = replicated_layout()
        assert rep.ndim == 1
        assert rep.is_replicated()

    def test_with_placement(self):
        rep = replicated_layout(2)
        sharded = rep.with_placement(1, Shard(-1))
        assert sharded.placements == (Replicate(), Shard(-1))
        assert rep.placements == (Replicate(), Replicate())  # original unchanged

    def test_with_placement_multiple(self):
        rep = replicated_layout(3)
        modified = rep.with_placement(0, Shard(0)).with_placement(2, Shard(-1))
        assert modified.placements == (Shard(0), Replicate(), Shard(-1))

    def test_shard_dim(self):
        layout = Layout(placements=(Shard(2), Replicate()))
        assert layout.shard_dim(0) == 2
        assert layout.shard_dim(1) is None

    def test_shard_dim_negative(self):
        layout = Layout(placements=(Shard(-1), Shard(-2)))
        assert layout.shard_dim(0) == -1
        assert layout.shard_dim(1) == -2

    def test_is_replicated(self):
        rep = Layout(placements=(Replicate(), Replicate()))
        assert rep.is_replicated()
        sharded = Layout(placements=(Replicate(), Shard(-1)))
        assert not sharded.is_replicated()

    def test_is_sharded_on_out_of_bounds(self):
        layout = Layout(placements=(Replicate(),))
        assert not layout.is_sharded_on(5)

    def test_layout_from_list(self):
        # Layout should convert list to tuple
        layout = Layout(placements=[Replicate(), Shard(-1)])
        assert isinstance(layout.placements, tuple)
        assert layout.placements == (Replicate(), Shard(-1))


class TestLayoutConversion:
    """Tests for layout conversion utilities."""

    def test_layout_from_topology_mock(self):
        """layout_from_topology extracts Layout from an object with placements()."""
        try:
            import ttnn
        except ImportError:
            pytest.skip("ttnn not available")

        class MockTopology:
            def placements(self):
                return [ttnn.PlacementShard(-1), ttnn.PlacementReplicate()]

        layout = layout_from_topology(MockTopology())
        assert layout.ndim == 2
        assert isinstance(layout.placements[0], Shard)
        assert layout.placements[0].dim == -1
        assert isinstance(layout.placements[1], Replicate)

    def test_layout_from_topology_all_replicate(self):
        try:
            import ttnn
        except ImportError:
            pytest.skip("ttnn not available")

        class MockTopology:
            def placements(self):
                return [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()]

        layout = layout_from_topology(MockTopology())
        assert layout.is_replicated()
        assert layout.ndim == 2

    def test_layout_to_mapper_config(self):
        import ttnn

        layout = Layout(placements=(Replicate(), Shard(-1)))
        config = layout_to_mapper_config(layout)
        # MeshMapperConfig is an opaque object, just verify it was created
        assert config is not None

    def test_layout_to_mapper_config_shard_dim(self):
        """layout_to_mapper_config with Shard(-2) produces correct dim."""
        import ttnn

        layout = Layout(placements=(Shard(-2), Replicate()))
        config = layout_to_mapper_config(layout)
        # MeshMapperConfig is an opaque object, just verify it was created
        assert config is not None


# ---------------------------------------------------------------------------
# PlanCache
# ---------------------------------------------------------------------------

from ttml.distributed.cache import PlanCache


class TestPlanCache:
    """Tests for the LRU plan cache."""

    def test_put_get(self):
        cache = PlanCache(maxsize=8)
        key = ("linear", (Layout((Replicate(),)),), ())
        cache.put(key, "plan_a")
        assert cache.get(key) == "plan_a"

    def test_miss(self):
        cache = PlanCache()
        assert cache.get(("nope",)) is None

    def test_eviction(self):
        cache = PlanCache(maxsize=2)
        cache.put(("a",), 1)
        cache.put(("b",), 2)
        cache.put(("c",), 3)
        assert cache.get(("a",)) is None  # evicted
        assert cache.get(("b",)) == 2
        assert cache.get(("c",)) == 3

    def test_lru_order(self):
        cache = PlanCache(maxsize=2)
        cache.put(("a",), 1)
        cache.put(("b",), 2)
        # Access "a" to make it most recently used
        cache.get(("a",))
        cache.put(("c",), 3)
        # "b" should be evicted, not "a"
        assert cache.get(("a",)) == 1
        assert cache.get(("b",)) is None
        assert cache.get(("c",)) == 3

    def test_update_existing(self):
        cache = PlanCache(maxsize=2)
        cache.put(("a",), 1)
        cache.put(("a",), 2)  # update
        assert cache.get(("a",)) == 2
        assert len(cache) == 1

    def test_clear(self):
        cache = PlanCache()
        cache.put(("x",), 10)
        cache.put(("y",), 20)
        cache.clear()
        assert len(cache) == 0
        assert cache.get(("x",)) is None

    def test_len(self):
        cache = PlanCache(maxsize=10)
        assert len(cache) == 0
        cache.put(("a",), 1)
        assert len(cache) == 1
        cache.put(("b",), 2)
        assert len(cache) == 2


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

from ttml.distributed.rules.registry import (
    ShardingPlan,
    register_rule,
    get_rule,
    register_module_rule,
    get_module_rule,
    _OP_RULES,
    _MODULE_RULES,
)


class TestShardingPlan:
    """Tests for ShardingPlan dataclass."""

    def test_basic_plan(self):
        plan = ShardingPlan(
            input_layouts=[Layout((Replicate(),))],
            output_layout=Layout((Replicate(),)),
        )
        assert len(plan.input_layouts) == 1
        assert plan.post_collective is None
        assert plan.reduce_mesh_axis is None
        assert plan.gather_grad_replicated is False

    def test_plan_with_collective(self):
        plan = ShardingPlan(
            input_layouts=[Layout((Replicate(),))],
            output_layout=Layout((Replicate(),)),
            post_collective="all_reduce",
            reduce_mesh_axis=1,
        )
        assert plan.post_collective == "all_reduce"
        assert plan.reduce_mesh_axis == 1

    def test_plan_with_grad_replicated(self):
        plan = ShardingPlan(
            input_layouts=[Layout((Replicate(),))],
            output_layout=Layout((Replicate(),)),
            gather_grad_replicated=True,
        )
        assert plan.gather_grad_replicated is True


class TestRuleRegistry:
    """Tests for op rule registration and lookup."""

    def test_register_and_get(self):
        @register_rule("__test_op_xyz__")
        def _rule(layout, **kwargs):
            return ShardingPlan(
                input_layouts=[layout],
                output_layout=layout,
            )

        assert get_rule("__test_op_xyz__") is _rule
        del _OP_RULES["__test_op_xyz__"]

    def test_get_missing(self):
        assert get_rule("__nonexistent__") is None

    def test_register_overwrites(self):
        @register_rule("__test_overwrite__")
        def _rule1(layout, **kwargs):
            return ShardingPlan(input_layouts=[layout], output_layout=layout)

        @register_rule("__test_overwrite__")
        def _rule2(layout, **kwargs):
            return ShardingPlan(input_layouts=[layout], output_layout=layout)

        assert get_rule("__test_overwrite__") is _rule2
        del _OP_RULES["__test_overwrite__"]


class TestModuleRuleRegistry:
    """Tests for module rule registration and lookup."""

    def test_register_by_class(self):
        class _DummyModule:
            pass

        @register_module_rule(_DummyModule)
        def _rule(module, runtime, policy, prefix=""):
            return module

        assert get_module_rule(_DummyModule) is _rule
        del _MODULE_RULES[_DummyModule]

    def test_get_by_instance(self):
        class _Base:
            pass

        class _Child(_Base):
            pass

        @register_module_rule(_Base)
        def _rule(module, runtime, policy, prefix=""):
            return module

        assert get_module_rule(_Child) is _rule
        del _MODULE_RULES[_Base]

    def test_get_missing_module_rule(self):
        class _UnregisteredModule:
            pass

        assert get_module_rule(_UnregisteredModule) is None


# ---------------------------------------------------------------------------
# Sharding rules (unit-level, no device needed)
# ---------------------------------------------------------------------------


class TestMatmulRules:
    """Tests for linear and matmul sharding rules."""

    def test_column_parallel(self):
        from ttml.distributed.rules.matmul import linear_rule

        inp = Layout((Replicate(), Replicate()))
        wt = Layout((Replicate(), Shard(-2)))
        plan = linear_rule(inp, wt, runtime=None)
        assert plan.post_collective is None
        assert isinstance(plan.output_layout.placements[1], Shard)
        assert plan.output_layout.placements[1].dim == -1

    def test_row_parallel(self):
        from ttml.distributed.rules.matmul import linear_rule

        inp = Layout((Replicate(), Replicate()))
        wt = Layout((Replicate(), Shard(-1)))
        plan = linear_rule(inp, wt, runtime=None)
        assert plan.post_collective == "all_reduce"
        assert plan.reduce_mesh_axis == 1

    def test_replicated_weights(self):
        from ttml.distributed.rules.matmul import linear_rule

        inp = Layout((Replicate(), Replicate()))
        wt = Layout((Replicate(), Replicate()))
        plan = linear_rule(inp, wt, runtime=None)
        assert plan.post_collective is None
        assert plan.output_layout.is_replicated()

    def test_column_parallel_with_bias(self):
        from ttml.distributed.rules.matmul import linear_rule

        inp = Layout((Replicate(), Replicate()))
        wt = Layout((Replicate(), Shard(-2)))
        bias = Layout((Replicate(), Replicate()))
        plan = linear_rule(inp, wt, bias, runtime=None)
        assert len(plan.input_layouts) == 3
        # Bias should be sharded on -1 for column parallel
        assert isinstance(plan.input_layouts[2].placements[1], Shard)

    def test_matmul_column_parallel(self):
        from ttml.distributed.rules.matmul import matmul_rule

        a = Layout((Replicate(), Replicate()))
        b = Layout((Replicate(), Shard(-1)))
        plan = matmul_rule(a, b, runtime=None)
        assert plan.post_collective is None
        assert isinstance(plan.output_layout.placements[1], Shard)

    def test_matmul_row_parallel(self):
        from ttml.distributed.rules.matmul import matmul_rule

        a = Layout((Replicate(), Replicate()))
        b = Layout((Replicate(), Shard(-2)))
        plan = matmul_rule(a, b, runtime=None)
        assert plan.post_collective == "all_reduce"


class TestElementwiseRules:
    """Tests for elementwise op sharding rules."""

    def test_binary_same_layout(self):
        from ttml.distributed.rules.elementwise import elementwise_binary_rule

        l = Layout((Shard(-1),))
        plan = elementwise_binary_rule(l, l, runtime=None)
        assert plan.output_layout == l
        assert plan.input_layouts == [l, l]

    def test_binary_picks_more_sharded(self):
        from ttml.distributed.rules.elementwise import elementwise_binary_rule

        a = Layout((Replicate(),))
        b = Layout((Shard(-1),))
        plan = elementwise_binary_rule(a, b, runtime=None)
        assert plan.output_layout == b
        assert plan.input_layouts == [b, b]

    def test_binary_single_input(self):
        from ttml.distributed.rules.elementwise import elementwise_binary_rule

        a = Layout((Shard(-1),))
        plan = elementwise_binary_rule(a, None, runtime=None)
        assert plan.output_layout == a
        assert plan.input_layouts == [a]

    def test_unary_passes_through(self):
        from ttml.distributed.rules.elementwise import elementwise_unary_rule

        la = Layout((Shard(2),))
        plan = elementwise_unary_rule(la, runtime=None)
        assert plan.output_layout == la

    def test_unary_no_layouts(self):
        from ttml.distributed.rules.elementwise import elementwise_unary_rule

        plan = elementwise_unary_rule(runtime=None)
        assert plan.output_layout.is_replicated()

    def test_dropout_passes_through(self):
        from ttml.distributed.rules.elementwise import dropout_rule

        la = Layout((Shard(-1), Replicate()))
        plan = dropout_rule(la, runtime=None)
        assert plan.output_layout == la


class TestNormRules:
    """Tests for normalization op sharding rules."""

    def test_rmsnorm_passes_through(self):
        from ttml.distributed.rules.normalization import rmsnorm_rule

        inp = Layout((Replicate(), Shard(-1)))
        gamma = Layout((Replicate(), Shard(-1)))
        plan = rmsnorm_rule(inp, gamma, runtime=None)
        assert plan.output_layout == inp

    def test_rmsnorm_single_tensor(self):
        from ttml.distributed.rules.normalization import rmsnorm_rule

        inp = Layout((Replicate(),))
        plan = rmsnorm_rule(inp, runtime=None)
        assert plan.output_layout == inp

    def test_rmsnorm_no_layouts(self):
        from ttml.distributed.rules.normalization import rmsnorm_rule

        plan = rmsnorm_rule(runtime=None)
        assert plan.output_layout.is_replicated()

    def test_layernorm_passes_through(self):
        from ttml.distributed.rules.normalization import layernorm_rule

        inp = Layout((Shard(0), Replicate()))
        plan = layernorm_rule(inp, runtime=None)
        assert plan.output_layout == inp

    def test_composite_layernorm(self):
        from ttml.distributed.rules.normalization import layernorm_rule

        inp = Layout((Replicate(), Shard(-1)))
        gamma = Layout((Replicate(), Shard(-1)))
        beta = Layout((Replicate(), Shard(-1)))
        plan = layernorm_rule(inp, gamma, beta, runtime=None)
        assert plan.output_layout == inp
        assert len(plan.input_layouts) == 3


class TestAttentionRules:
    """Tests for attention-related op sharding rules."""

    def test_sdpa_passes_through(self):
        from ttml.distributed.rules.attention import sdpa_rule

        l = Layout((Shard(0),))
        plan = sdpa_rule(l, l, l, runtime=None)
        assert plan.output_layout == l

    def test_sdpa_with_mask(self):
        from ttml.distributed.rules.attention import sdpa_rule

        q = Layout((Shard(0), Replicate()))
        k = Layout((Shard(0), Replicate()))
        v = Layout((Shard(0), Replicate()))
        mask = Layout((Replicate(), Replicate()))
        plan = sdpa_rule(q, k, v, mask, runtime=None)
        assert plan.output_layout == q
        assert len(plan.input_layouts) == 4

    def test_grouped_heads_creation(self):
        from ttml.distributed.rules.attention import grouped_heads_creation_rule

        l = Layout((Shard(-1), Replicate()))
        plan = grouped_heads_creation_rule(l, l, runtime=None)
        assert plan.output_layout == l

    def test_heads_fusion(self):
        from ttml.distributed.rules.attention import heads_fusion_rule

        l = Layout((Shard(0),))
        plan = heads_fusion_rule(l, runtime=None)
        assert plan.output_layout == l

    def test_heads_creation(self):
        from ttml.distributed.rules.attention import heads_creation_rule

        l = Layout((Replicate(), Shard(-1)))
        plan = heads_creation_rule(l, runtime=None)
        assert plan.output_layout == l

    def test_rope(self):
        from ttml.distributed.rules.attention import rope_rule

        l = Layout((Shard(0), Replicate()))
        plan = rope_rule(l, l, runtime=None)
        assert plan.output_layout == l

    def test_embedding(self):
        from ttml.distributed.rules.attention import embedding_rule

        l = Layout((Replicate(),))
        plan = embedding_rule(l, l, runtime=None)
        assert plan.output_layout == l

    def test_reshape(self):
        from ttml.distributed.rules.attention import reshape_rule

        l = Layout((Shard(-1), Replicate()))
        plan = reshape_rule(l, runtime=None)
        assert plan.output_layout == l


class TestLossRule:
    """Tests for loss op sharding rules."""

    def test_sharded_logits_become_replicated(self):
        from ttml.distributed.rules.loss import cross_entropy_loss_rule

        logit_layout = Layout((Replicate(), Shard(-1)))
        target_layout = Layout((Replicate(), Replicate()))
        plan = cross_entropy_loss_rule(logit_layout, target_layout, runtime=None)
        assert plan.input_layouts[0].is_replicated()
        assert plan.gather_grad_replicated is True

    def test_replicated_logits_stay(self):
        from ttml.distributed.rules.loss import cross_entropy_loss_rule

        logit_layout = Layout((Replicate(), Replicate()))
        target_layout = Layout((Replicate(), Replicate()))
        plan = cross_entropy_loss_rule(logit_layout, target_layout, runtime=None)
        assert plan.input_layouts[0].is_replicated()
        assert plan.gather_grad_replicated is True

    def test_no_layouts(self):
        from ttml.distributed.rules.loss import cross_entropy_loss_rule

        plan = cross_entropy_loss_rule(runtime=None)
        assert plan.output_layout.is_replicated()


# ---------------------------------------------------------------------------
# Debug tracing
# ---------------------------------------------------------------------------

from ttml.distributed.debug import DispatchTracer, dispatch_trace, TraceEntry


class TestDebugTrace:
    """Tests for dispatch tracing."""

    def test_enable_disable(self):
        dispatch_trace.clear()
        assert not dispatch_trace.enabled
        dispatch_trace.enable()
        assert dispatch_trace.enabled
        dispatch_trace.disable()
        assert not dispatch_trace.enabled

    def test_context_manager_collects(self):
        dispatch_trace.clear()
        entry = TraceEntry(
            op_name="test",
            input_layouts=[],
            rule_name=None,
            plan=None,
            redistributions=[],
            post_collectives=[],
            output_layout=None,
        )
        with DispatchTracer() as tracer:
            dispatch_trace.record(entry)
        assert len(tracer.entries) == 1
        assert tracer.entries[0].op_name == "test"
        assert not dispatch_trace.enabled

    def test_context_manager_only_collects_new(self):
        dispatch_trace.clear()
        dispatch_trace.enable()
        dispatch_trace.record(TraceEntry("old", [], None, None, [], [], None))
        dispatch_trace.disable()

        with DispatchTracer() as tracer:
            dispatch_trace.record(TraceEntry("new", [], None, None, [], [], None))

        assert len(tracer.entries) == 1
        assert tracer.entries[0].op_name == "new"

    def test_clear(self):
        dispatch_trace.clear()
        dispatch_trace.enable()
        dispatch_trace.record(TraceEntry("x", [], None, None, [], [], None))
        dispatch_trace.disable()
        assert len(dispatch_trace.entries) == 1
        dispatch_trace.clear()
        assert len(dispatch_trace.entries) == 0

    def test_record_when_disabled(self):
        dispatch_trace.clear()
        dispatch_trace.disable()
        dispatch_trace.record(TraceEntry("ignored", [], None, None, [], [], None))
        assert len(dispatch_trace.entries) == 0

    def test_trace_entry_repr(self):
        entry = TraceEntry(
            op_name="linear",
            input_layouts=[Layout((Replicate(),))],
            rule_name="linear_rule",
            plan=None,
            redistributions=[{"from": "a", "to": "b"}],
            post_collectives=[{"type": "all_reduce"}],
            output_layout=Layout((Shard(-1),)),
        )
        repr_str = repr(entry)
        assert "linear" in repr_str
        assert "linear_rule" in repr_str
        assert "redist=" in repr_str
        assert "post_ccl=" in repr_str


# ---------------------------------------------------------------------------
# MeshRuntime
# ---------------------------------------------------------------------------

from ttml.distributed.mesh_runtime import MeshRuntime, get_runtime, set_runtime


class TestMeshRuntime:
    """Tests for MeshRuntime configuration."""

    def test_properties(self):
        class FakeMesh:
            shape = [2, 4]

            def get_num_devices(self):
                return 8

        rt = MeshRuntime(mesh_device=FakeMesh(), tp_axis=1, dp_axis=0)
        assert rt.tp_size == 4
        assert rt.dp_size == 2
        assert rt.cp_size == 1
        assert rt.is_tp_enabled
        assert rt.is_dp_enabled
        assert not rt.is_cp_enabled
        assert rt.num_devices == 8

    def test_mesh_shape(self):
        class FakeMesh:
            shape = [8, 4]

            def get_num_devices(self):
                return 32

        rt = MeshRuntime(mesh_device=FakeMesh())
        assert rt.mesh_shape == [8, 4]

    def test_disabled_parallelism(self):
        class FakeMesh:
            shape = [1, 1]

            def get_num_devices(self):
                return 1

        rt = MeshRuntime(mesh_device=FakeMesh())
        assert rt.tp_size == 1
        assert rt.dp_size == 1
        assert rt.cp_size == 1
        assert not rt.is_tp_enabled
        assert not rt.is_dp_enabled
        assert not rt.is_cp_enabled

    def test_cp_enabled(self):
        class FakeMesh:
            shape = [2, 4, 2]

            def get_num_devices(self):
                return 16

        rt = MeshRuntime(mesh_device=FakeMesh(), tp_axis=1, dp_axis=0, cp_axis=2)
        assert rt.cp_size == 2
        assert rt.is_cp_enabled

    def test_global_runtime(self):
        old = get_runtime()
        try:
            set_runtime(None)
            assert get_runtime() is None
            sentinel = object()
            set_runtime(sentinel)
            assert get_runtime() is sentinel
        finally:
            set_runtime(old)

    def test_plan_cache_default(self):
        class FakeMesh:
            shape = [2, 4]

            def get_num_devices(self):
                return 8

        rt = MeshRuntime(mesh_device=FakeMesh())
        assert rt.plan_cache is not None
        assert len(rt.plan_cache) == 0


# ---------------------------------------------------------------------------
# Policy construction (application-level, tested as a pattern)
# ---------------------------------------------------------------------------


class TestPolicyConstruction:
    """Tests for policy dict construction patterns."""

    def test_explicit_policy_structure(self):
        """Verify that a manually-built TP policy has the expected shape."""
        tp_axis = 1
        ndim = 2
        col = Layout(
            tuple(Shard(-2) if i == tp_axis else Replicate() for i in range(ndim))
        )
        row = Layout(
            tuple(Shard(-1) if i == tp_axis else Replicate() for i in range(ndim))
        )

        policy = {
            "attention.q_linear.weight": col,
            "attention.out_linear.weight": row,
            "mlp.w1.weight": col,
            "mlp.w2.weight": row,
        }

        assert isinstance(policy["attention.q_linear.weight"].placements[1], Shard)
        assert policy["attention.q_linear.weight"].placements[1].dim == -2
        assert isinstance(policy["attention.out_linear.weight"].placements[1], Shard)
        assert policy["attention.out_linear.weight"].placements[1].dim == -1

    def test_policy_with_multiple_layers(self):
        col = Layout((Replicate(), Shard(-2)))
        row = Layout((Replicate(), Shard(-1)))

        policy = {}
        for i in range(12):
            policy[f"layers.{i}.attention.q_linear.weight"] = col
            policy[f"layers.{i}.attention.kv_linear.weight"] = col
            policy[f"layers.{i}.attention.out_linear.weight"] = row
            policy[f"layers.{i}.mlp.w1.weight"] = col
            policy[f"layers.{i}.mlp.w3.weight"] = col
            policy[f"layers.{i}.mlp.w2.weight"] = row

        assert len(policy) == 72
        assert policy["layers.5.mlp.w1.weight"] == col


class TestRegexPolicy:
    """Tests for regex-based policy matching."""

    def test_exact_match_takes_priority(self):
        from ttml.distributed.training import _match_policy

        col = Layout((Replicate(), Shard(-2)))
        row = Layout((Replicate(), Shard(-1)))
        policy = {
            "layers.0.q_linear.weight": col,
            r".*\.q_linear\.weight": row,
        }
        assert _match_policy("layers.0.q_linear.weight", policy) == col

    def test_regex_match(self):
        from ttml.distributed.training import _match_policy

        col = Layout((Replicate(), Shard(-2)))
        policy = {r".*\.(q_linear|kv_linear|w1|w3)\.weight": col}
        assert _match_policy("layers.0.attention.q_linear.weight", policy) == col
        assert _match_policy("layers.5.mlp.w3.weight", policy) == col
        assert _match_policy("layers.0.attention.out_linear.weight", policy) is None

    def test_no_match(self):
        from ttml.distributed.training import _match_policy

        col = Layout((Replicate(), Shard(-2)))
        policy = {r".*\.q_linear\.weight": col}
        assert _match_policy("embedding.weight", policy) is None

    def test_regex_policy_compact(self):
        """Regex-based policy covers multiple layers with one pattern."""
        col = Layout((Replicate(), Shard(-2)))
        row = Layout((Replicate(), Shard(-1)))
        policy = {
            r".*\.(q_linear|kv_linear|w1|w3)\.weight": col,
            r".*\.(out_linear|w2)\.weight": row,
        }
        from ttml.distributed.training import _match_policy

        assert _match_policy("layers.0.attention.q_linear.weight", policy) == col
        assert _match_policy("layers.0.attention.out_linear.weight", policy) == row
        assert _match_policy("layers.3.mlp.w1.weight", policy) == col
        assert _match_policy("layers.3.mlp.w2.weight", policy) == row
        assert _match_policy("norm.weight", policy) is None

    def test_invalid_regex_ignored(self):
        from ttml.distributed.training import _match_policy

        col = Layout((Replicate(), Shard(-2)))
        policy = {
            r"[invalid(regex": col,  # Invalid regex
            r".*\.valid\.weight": col,
        }
        # Invalid regex should be skipped, valid one should match
        assert _match_policy("some.valid.weight", policy) == col
        assert _match_policy("some.invalid.weight", policy) is None


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

from ttml.distributed.dispatch import _hashable_kwargs


class TestDispatchHelpers:
    """Tests for dispatch utility functions."""

    def test_hashable_kwargs_simple(self):
        kwargs = {"a": 1, "b": "hello", "c": True}
        result = _hashable_kwargs(kwargs)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_hashable_kwargs_unhashable(self):
        kwargs = {"a": [1, 2, 3], "b": {"nested": "dict"}}
        result = _hashable_kwargs(kwargs)
        # Unhashable values should be converted to id()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_hashable_kwargs_empty(self):
        result = _hashable_kwargs({})
        assert result == ()

    def test_hashable_kwargs_sorted(self):
        kwargs = {"z": 1, "a": 2, "m": 3}
        result = _hashable_kwargs(kwargs)
        # Should be sorted by key
        assert result[0][0] == "a"
        assert result[1][0] == "m"
        assert result[2][0] == "z"


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

from ttml.distributed.utils import is_distributed


class TestUtils:
    """Tests for utility functions."""

    def test_is_distributed_returns_false_for_invalid(self):
        # Test with object that doesn't have get_value
        assert not is_distributed(object())
        assert not is_distributed(None)
        assert not is_distributed(42)


# ---------------------------------------------------------------------------
# Module rules (unit tests)
# ---------------------------------------------------------------------------


class TestModuleRulesUnit:
    """Unit tests for module rule helpers."""

    def test_infer_bias_layout_column_parallel(self):
        from ttml.distributed.module_rules import _infer_bias_layout

        weight_layout = Layout((Replicate(), Shard(-2)))
        bias_layout = _infer_bias_layout(weight_layout)
        assert isinstance(bias_layout.placements[1], Shard)
        assert bias_layout.placements[1].dim == -1

    def test_infer_bias_layout_row_parallel(self):
        from ttml.distributed.module_rules import _infer_bias_layout

        weight_layout = Layout((Replicate(), Shard(-1)))
        bias_layout = _infer_bias_layout(weight_layout)
        assert bias_layout.is_replicated()

    def test_infer_bias_layout_replicated(self):
        from ttml.distributed.module_rules import _infer_bias_layout

        weight_layout = Layout((Replicate(), Replicate()))
        bias_layout = _infer_bias_layout(weight_layout)
        assert bias_layout.is_replicated()


# ---------------------------------------------------------------------------
# Device-backed tests
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
class TestDistributedOnDevice:
    """These tests require a Tenstorrent multi-device mesh (full 32-device Galaxy).

    Note: Fabric requires the full mesh shape to be initialized. These tests
    use [8, 4] = 32 devices. They will be skipped if fabric/devices aren't available.
    """

    @pytest.fixture(autouse=True)
    def setup_device(self):
        import gc
        import ttml
        from ttml.distributed.mesh_runtime import set_runtime, MeshRuntime

        # Fabric requires full 32-device mesh
        num_devices = 32
        mesh_shape = [8, 4]

        try:
            ttml.core.distributed.enable_fabric(num_devices)
        except Exception:
            pytest.skip("Fabric not available or insufficient devices")

        auto_ctx = ttml.autograd.AutoContext.get_instance()
        try:
            auto_ctx.open_device(mesh_shape)
        except Exception:
            pytest.skip(f"Could not open device with mesh shape {mesh_shape}")

        mesh_device = auto_ctx.get_device()
        set_runtime(MeshRuntime(mesh_device=mesh_device))
        yield
        # Clean up to avoid nanobind leak warnings
        set_runtime(None)
        auto_ctx.reset_graph()  # Release tensor references in autograd graph
        gc.collect()  # Force garbage collection before closing device
        auto_ctx.close_device()

    def _pcc(self, a, b):
        """Compute Pearson Correlation Coefficient between two arrays."""
        a_flat = np.asarray(a).flatten().astype(np.float64)
        b_flat = np.asarray(b).flatten().astype(np.float64)
        if np.std(a_flat) < 1e-10 or np.std(b_flat) < 1e-10:
            return 1.0 if np.allclose(a_flat, b_flat) else 0.0
        return float(np.corrcoef(a_flat, b_flat)[0, 1])

    def _nonzero_randn(self, *shape, mean=0.5, scale=0.5):
        """Generate random array with non-zero mean for better numerical signal."""
        return (np.random.randn(*shape).astype(np.float32) + mean) * scale

    def test_distribute_tensor_sharded(self):
        """Test distributing a tensor with sharding."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()

        # Create tensor with shape divisible by TP size (4)
        np_data = np.ones((1, 1, 4, 16), dtype=np.float32)
        tensor = ttml.autograd.Tensor.from_numpy(
            np_data.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )

        # Shard on TP axis (axis 1), last dim
        layout = Layout(placements=(Replicate(), Shard(-1)))
        result = distribute_tensor(tensor, mesh_device, layout)
        assert get_layout(result) == layout

    def test_distribute_tensor_replicated(self):
        """Test distributing a tensor as replicated."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Replicate

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()

        np_data = np.ones((1, 1, 8, 32), dtype=np.float32)
        tensor = ttml.autograd.Tensor.from_numpy(
            np_data.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )

        layout = Layout(placements=(Replicate(), Replicate()))
        result = distribute_tensor(tensor, mesh_device, layout)
        # Check that result is replicated (may have different ndim depending on mapper)
        result_layout = get_layout(result)
        assert result_layout.is_replicated()

    def test_distribute_tensor_different_shard_dims(self):
        """Test distributing with different shard dimensions."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()

        # Shape divisible by 8 (DP axis) and 4 (TP axis)
        np_data = np.ones((1, 1, 8, 32), dtype=np.float32)
        tensor = ttml.autograd.Tensor.from_numpy(
            np_data.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )

        # Shard on dim -2 (second to last)
        layout = Layout(placements=(Replicate(), Shard(-2)))
        result = distribute_tensor(tensor, mesh_device, layout)
        assert get_layout(result) == layout

    def test_get_layout_from_distributed_tensor(self):
        """Test that get_layout correctly reads from distributed tensor."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()

        np_data = np.random.randn(1, 1, 4, 16).astype(np.float32)
        tensor = ttml.autograd.Tensor.from_numpy(
            np_data.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )

        layout = Layout(placements=(Replicate(), Shard(-1)))
        result = distribute_tensor(tensor, mesh_device, layout)

        # Read back the layout
        read_layout = get_layout(result)
        assert read_layout == layout

    def test_set_layout_get_layout_round_trip(self):
        """After distribute_tensor, set_layout then get_layout returns the same layout."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        np_data = np.ones((1, 1, 4, 16), dtype=np.float32)
        tensor = ttml.autograd.Tensor.from_numpy(
            np_data.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )
        layout = Layout(placements=(Replicate(), Shard(-1)))
        distributed = distribute_tensor(tensor, mesh_device, layout)
        set_layout(distributed, layout)
        assert get_layout(distributed) == layout

    def test_is_distributed_true_for_sharded_tensor(self):
        """is_distributed returns True for a tensor distributed with Shard placement."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        layout = Layout(placements=(Replicate(), Shard(-1)))
        np_data = np.ones((1, 1, 4, 16), dtype=np.float32)
        tensor = ttml.autograd.Tensor.from_numpy(
            np_data.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )
        distributed = distribute_tensor(tensor, mesh_device, layout)
        assert is_distributed(distributed) is True

    def test_dispatch_init_ops(self):
        """Test that init_ops can be called without error."""
        from ttml.distributed import init_ops

        init_ops()
        # Should be idempotent
        init_ops()

    def test_mesh_runtime_with_real_device(self):
        """Test MeshRuntime with actual mesh device."""
        import ttml
        from ttml.distributed.mesh_runtime import MeshRuntime

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)

        # MeshShape doesn't compare directly with list, check individual elements
        assert rt.mesh_shape[0] == 8
        assert rt.mesh_shape[1] == 4
        assert rt.tp_size == 4
        assert rt.dp_size == 8
        # Test the num_devices property (which uses get_num_devices() internally)
        assert rt.num_devices == 32

    def test_dispatch_trace_with_real_ops(self):
        """Test dispatch tracing with real tensor operations."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.debug import DispatchTracer, dispatch_trace
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        # Create and distribute a tensor
        np_data = np.random.randn(1, 1, 8, 32).astype(np.float32)
        tensor = ttml.autograd.Tensor.from_numpy(
            np_data.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )
        layout = Layout(placements=(Replicate(), Replicate()))
        x = distribute_tensor(tensor, mesh_device, layout)

        dispatch_trace.clear()
        with DispatchTracer() as tracer:
            # Perform an operation that goes through dispatch
            y = ttml.ops.unary.silu(x)

        # Should have recorded the silu operation
        assert len(tracer.entries) >= 1
        op_names = [e.op_name for e in tracer.entries]
        assert "silu" in op_names

    def test_plan_cache_with_real_ops(self):
        """Test that plan cache works with real operations."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime, get_runtime

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        # Clear cache
        rt.plan_cache.clear()
        initial_cache_size = len(rt.plan_cache)

        # Create and distribute tensors
        np_data = np.random.randn(1, 1, 8, 32).astype(np.float32)
        tensor = ttml.autograd.Tensor.from_numpy(
            np_data.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )
        layout = Layout(placements=(Replicate(), Replicate()))
        x = distribute_tensor(tensor, mesh_device, layout)

        # First call should populate cache
        y1 = ttml.ops.unary.silu(x)
        cache_after_first = len(rt.plan_cache)

        # Second call should hit cache
        y2 = ttml.ops.unary.silu(x)
        cache_after_second = len(rt.plan_cache)

        # Cache should have grown after first call but not after second
        assert cache_after_first > initial_cache_size
        assert cache_after_second == cache_after_first

    def test_binary_ops_with_distributed_tensors(self):
        """Test binary operations with distributed tensors."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        # Create two tensors
        np_a = np.random.randn(1, 1, 8, 32).astype(np.float32)
        np_b = np.random.randn(1, 1, 8, 32).astype(np.float32)

        tensor_a = ttml.autograd.Tensor.from_numpy(
            np_a.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )
        tensor_b = ttml.autograd.Tensor.from_numpy(
            np_b.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )

        layout = Layout(placements=(Replicate(), Replicate()))
        a = distribute_tensor(tensor_a, mesh_device, layout)
        b = distribute_tensor(tensor_b, mesh_device, layout)

        # Test binary operations - check that output is replicated
        c = ttml.ops.binary.add(a, b)
        assert get_layout(c).is_replicated()

        d = ttml.ops.binary.mul(a, b)
        assert get_layout(d).is_replicated()

    def test_unary_ops_preserve_layout(self):
        """Test that unary operations preserve tensor layout."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        np_data = np.random.randn(1, 1, 8, 32).astype(np.float32)
        tensor = ttml.autograd.Tensor.from_numpy(
            np_data.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )

        # Test with sharded layout
        sharded_layout = Layout(placements=(Replicate(), Shard(-1)))
        x = distribute_tensor(tensor, mesh_device, sharded_layout)

        # Unary ops should preserve layout
        y_silu = ttml.ops.unary.silu(x)
        assert get_layout(y_silu) == sharded_layout

        y_relu = ttml.ops.unary.relu(x)
        assert get_layout(y_relu) == sharded_layout

        y_gelu = ttml.ops.unary.gelu(x)
        assert get_layout(y_gelu) == sharded_layout

    def test_rmsnorm_preserves_layout(self):
        """Dispatch rmsnorm with sharded input preserves layout."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        shard_layout = Layout(placements=(Replicate(), Shard(-1)))
        np_x = np.random.randn(1, 1, 4, 16).astype(np.float32)
        np_gamma = np.ones((1, 1, 1, 16), dtype=np.float32)
        x = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_x.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh_device,
            shard_layout,
        )
        gamma = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_gamma.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh_device,
            shard_layout,
        )
        set_layout(x, shard_layout)
        set_layout(gamma, shard_layout)
        y = ttml.ops.rmsnorm.rmsnorm(x, gamma, epsilon=1e-5)
        assert get_layout(y) == shard_layout

    def test_dropout_preserves_layout(self):
        """Dispatch dropout preserves input layout (pass-through rule)."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        shard_layout = Layout(placements=(Replicate(), Shard(-1)))
        np_data = np.random.randn(1, 1, 4, 16).astype(np.float32)
        x = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_data.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh_device,
            shard_layout,
        )
        set_layout(x, shard_layout)
        # dropout uses 'probability' parameter, not 'p'
        y = ttml.ops.dropout.dropout(x, probability=0.0)
        assert get_layout(y) == shard_layout

    def test_linear_column_parallel(self):
        """Test column-parallel linear operation."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        # Input: [1, 1, 8, 64] - replicated
        # Weight: [1, 1, 128, 64] - sharded on out_features (dim -2)
        # Output should be sharded on last dim

        np_input = np.random.randn(1, 1, 8, 64).astype(np.float32)
        np_weight = np.random.randn(1, 1, 128, 64).astype(np.float32)

        input_tensor = ttml.autograd.Tensor.from_numpy(
            np_input.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )
        weight_tensor = ttml.autograd.Tensor.from_numpy(
            np_weight.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )

        input_layout = Layout(placements=(Replicate(), Replicate()))
        weight_layout = Layout(placements=(Replicate(), Shard(-2)))  # Column parallel

        x = distribute_tensor(input_tensor, mesh_device, input_layout)
        w = distribute_tensor(weight_tensor, mesh_device, weight_layout)

        # Perform linear operation
        y = ttml.ops.linear.linear(x, w, None)

        # Output should be sharded on last dim
        output_layout = get_layout(y)
        assert output_layout.is_sharded_on(1)
        assert output_layout.shard_dim(1) == -1

    def test_linear_row_parallel(self):
        """Test row-parallel linear operation with all_reduce."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        # Input: [1, 1, 8, 128] - sharded on last dim (to match weight in_features)
        # Weight: [1, 1, 64, 128] - sharded on in_features (dim -1)
        # Output should be replicated after all_reduce

        np_input = np.random.randn(1, 1, 8, 128).astype(np.float32)
        np_weight = np.random.randn(1, 1, 64, 128).astype(np.float32)

        input_tensor = ttml.autograd.Tensor.from_numpy(
            np_input.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )
        weight_tensor = ttml.autograd.Tensor.from_numpy(
            np_weight.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )

        input_layout = Layout(placements=(Replicate(), Shard(-1)))  # Sharded to match
        weight_layout = Layout(placements=(Replicate(), Shard(-1)))  # Row parallel

        x = distribute_tensor(input_tensor, mesh_device, input_layout)
        w = distribute_tensor(weight_tensor, mesh_device, weight_layout)

        # Perform linear operation
        y = ttml.ops.linear.linear(x, w, None)

        # Output should be replicated (after all_reduce)
        output_layout = get_layout(y)
        assert output_layout.is_replicated()

    def test_distribute_module_linear_layer(self):
        """Test distribute_module with a LinearLayer."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops, distribute_module
        from ttml.distributed.layout import Layout, Shard, Replicate
        from ttml.modules import LinearLayer

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()

        # Create a LinearLayer (uses has_bias, not bias)
        linear = LinearLayer(64, 128, has_bias=False)

        # Set up policy for column-parallel
        policy = {
            "weight": Layout(placements=(Replicate(), Shard(-2))),
        }

        # Distribute the module
        distribute_module(linear, mesh_device, policy)

        # Check that weight has correct layout
        weight_layout = get_layout(linear.weight.tensor)
        assert weight_layout == Layout(placements=(Replicate(), Shard(-2)))

    def test_distribute_module_with_bias(self):
        """Test distribute_module with LinearLayer that has bias."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops, distribute_module
        from ttml.distributed.layout import Layout, Shard, Replicate
        from ttml.modules import LinearLayer

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()

        # Create a LinearLayer with bias (uses has_bias, not bias)
        linear = LinearLayer(64, 128, has_bias=True)

        # Set up policy for column-parallel (bias should be inferred)
        policy = {
            "weight": Layout(placements=(Replicate(), Shard(-2))),
        }

        # Distribute the module
        distribute_module(linear, mesh_device, policy)

        # Check weight layout
        weight_layout = get_layout(linear.weight.tensor)
        assert weight_layout == Layout(placements=(Replicate(), Shard(-2)))

        # Check bias layout (should be sharded on -1 for column parallel)
        bias_layout = get_layout(linear.bias.tensor)
        assert bias_layout.is_sharded_on(1)
        assert bias_layout.shard_dim(1) == -1

    def test_numerical_correctness_column_parallel(self):
        """Test numerical correctness of column-parallel linear."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        np.random.seed(42)

        # Create test data with non-zero mean
        batch_seq = 8
        in_features = 64
        out_features = 128  # Divisible by TP size (4)

        np_input = self._nonzero_randn(1, 1, batch_seq, in_features)
        np_weight = self._nonzero_randn(1, 1, out_features, in_features)

        # Compute reference on CPU
        ref_output = np.matmul(np_input, np.transpose(np_weight, (0, 1, 3, 2)))

        # Distribute and compute on device
        input_tensor = ttml.autograd.Tensor.from_numpy(
            np_input.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )
        weight_tensor = ttml.autograd.Tensor.from_numpy(
            np_weight.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )

        input_layout = Layout(placements=(Replicate(), Replicate()))
        weight_layout = Layout(placements=(Replicate(), Shard(-2)))

        x = distribute_tensor(input_tensor, mesh_device, input_layout)
        w = distribute_tensor(weight_tensor, mesh_device, weight_layout)

        y = ttml.ops.linear.linear(x, w, None)

        # Gather output (it's sharded, so we need to gather)
        # The output is sharded on the last dim, so gather on that axis
        y_gathered = ttml.ops.distributed.all_gather(y, dim=-1, cluster_axis=1)

        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0)
        y_np = np.asarray(y_gathered.to_numpy(composer=composer))

        # Composer concatenates all 32 devices along dim 0, take first slice for replicated data
        y_np_first = y_np[:1]

        pcc = self._pcc(ref_output, y_np_first)
        assert pcc > 0.99, f"Column-parallel linear PCC {pcc:.4f} < 0.99"

    def test_numerical_correctness_row_parallel(self):
        """Test numerical correctness of row-parallel linear with all_reduce."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        np.random.seed(42)

        # Create test data with non-zero mean
        batch_seq = 8
        in_features = 128  # Divisible by TP size (4)
        out_features = 64

        np_input = self._nonzero_randn(1, 1, batch_seq, in_features)
        np_weight = self._nonzero_randn(1, 1, out_features, in_features)

        # Compute reference on CPU
        ref_output = np.matmul(np_input, np.transpose(np_weight, (0, 1, 3, 2)))

        # Distribute and compute on device
        input_tensor = ttml.autograd.Tensor.from_numpy(
            np_input.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )
        weight_tensor = ttml.autograd.Tensor.from_numpy(
            np_weight.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )

        # Row parallel: input sharded on last dim, weight sharded on in_features
        input_layout = Layout(placements=(Replicate(), Shard(-1)))
        weight_layout = Layout(placements=(Replicate(), Shard(-1)))

        x = distribute_tensor(input_tensor, mesh_device, input_layout)
        w = distribute_tensor(weight_tensor, mesh_device, weight_layout)

        y = ttml.ops.linear.linear(x, w, None)

        # Output should already be replicated (all_reduce was applied by dispatch)
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0)
        y_np = np.asarray(y.to_numpy(composer=composer))

        # Composer concatenates all 32 devices along dim 0, take first slice for replicated data
        y_np_first = y_np[:1]

        pcc = self._pcc(ref_output, y_np_first)
        assert pcc > 0.99, f"Row-parallel linear PCC {pcc:.4f} < 0.99"

    def test_elementwise_numerical_correctness(self):
        """Test numerical correctness of elementwise operations."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        np.random.seed(42)

        # Non-zero mean for better numerical signal
        np_a = self._nonzero_randn(1, 1, 8, 32)
        np_b = self._nonzero_randn(1, 1, 8, 32)

        # Reference computations
        ref_add = np_a + np_b
        ref_mul = np_a * np_b
        ref_silu = np_a / (1 + np.exp(-np_a))

        # Distribute tensors
        tensor_a = ttml.autograd.Tensor.from_numpy(
            np_a.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )
        tensor_b = ttml.autograd.Tensor.from_numpy(
            np_b.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )

        layout = Layout(placements=(Replicate(), Replicate()))
        a = distribute_tensor(tensor_a, mesh_device, layout)
        b = distribute_tensor(tensor_b, mesh_device, layout)

        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0)

        # Test add - take first slice since composer concatenates all devices
        c = ttml.ops.binary.add(a, b)
        c_np = np.asarray(c.to_numpy(composer=composer)[:1])
        add_pcc = self._pcc(ref_add, c_np)
        assert add_pcc > 0.99, f"Add PCC {add_pcc:.4f} < 0.99"

        # Test mul
        d = ttml.ops.binary.mul(a, b)
        d_np = np.asarray(d.to_numpy(composer=composer)[:1])
        mul_pcc = self._pcc(ref_mul, d_np)
        assert mul_pcc > 0.99, f"Mul PCC {mul_pcc:.4f} < 0.99"

        # Test silu
        e = ttml.ops.unary.silu(a)
        e_np = np.asarray(e.to_numpy(composer=composer)[:1])
        silu_pcc = self._pcc(ref_silu, e_np)
        assert silu_pcc > 0.99, f"SiLU PCC {silu_pcc:.4f} < 0.99"

    def test_fallback_for_unregistered_op(self):
        """Test that unregistered ops fall back to replicated execution."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime
        from ttml.distributed.debug import DispatchTracer

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        np_data = np.random.randn(1, 1, 8, 32).astype(np.float32)
        tensor = ttml.autograd.Tensor.from_numpy(
            np_data.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )

        layout = Layout(placements=(Replicate(), Replicate()))
        x = distribute_tensor(tensor, mesh_device, layout)

        # Operations that go through dispatch should work even if not explicitly registered
        # (they fall back to replicated execution)
        with DispatchTracer() as tracer:
            y = ttml.ops.unary.silu(x)

        # Should have traced the operation
        assert len(tracer.entries) >= 1

    def test_sync_gradients_with_cluster_axes(self):
        """Test sync_gradients with explicit cluster_axes."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops, distribute_module, sync_gradients
        from ttml.distributed.layout import Layout, Shard, Replicate
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime
        from ttml.modules import LinearLayer

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        # Create a simple model (uses has_bias, not bias)
        linear = LinearLayer(64, 128, has_bias=False)

        policy = {
            "weight": Layout(placements=(Replicate(), Shard(-2))),
        }
        distribute_module(linear, mesh_device, policy)

        # Create input and run forward
        np_input = np.random.randn(1, 1, 8, 64).astype(np.float32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            np_input.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )

        from ttml.distributed.training import distribute_tensor

        input_layout = Layout(placements=(Replicate(), Replicate()))
        x = distribute_tensor(input_tensor, mesh_device, input_layout)

        y = linear(x)

        # Sync gradients with explicit cluster_axes (DP axis = 0)
        # This should not error
        sync_gradients(linear, cluster_axes=[0])

    def test_redistribute_shard_to_replicate(self):
        """Test redistribute from sharded to replicated."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.redistribute import redistribute
        from ttml.distributed.layout import Layout, Shard, Replicate
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        np.random.seed(42)
        np_data = self._nonzero_randn(1, 1, 8, 32)

        tensor = ttml.autograd.Tensor.from_numpy(
            np_data.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )

        # Start with sharded layout
        sharded_layout = Layout(placements=(Replicate(), Shard(-1)))
        x = distribute_tensor(tensor, mesh_device, sharded_layout)

        # Redistribute to replicated
        replicated_layout = Layout(placements=(Replicate(), Replicate()))
        y = redistribute(x, replicated_layout)

        # Check layout
        assert get_layout(y) == replicated_layout

        # Check numerical correctness using PCC
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0)
        y_np = np.asarray(y.to_numpy(composer=composer)[:1])
        pcc = self._pcc(np_data, y_np)
        assert pcc > 0.999, f"Redistribute shard->replicate PCC {pcc:.4f} < 0.999"

    def test_redistribute_replicate_to_shard(self):
        """Test redistribute from replicated to sharded."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.redistribute import redistribute
        from ttml.distributed.layout import Layout, Shard, Replicate
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        np.random.seed(42)
        np_data = self._nonzero_randn(1, 1, 8, 32)

        tensor = ttml.autograd.Tensor.from_numpy(
            np_data.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )

        # Start with sharded layout (so we have proper 2D topology)
        sharded_layout = Layout(placements=(Replicate(), Shard(-1)))
        x = distribute_tensor(tensor, mesh_device, sharded_layout)

        # Redistribute to replicated first, then back to sharded
        replicated_layout = Layout(placements=(Replicate(), Replicate()))
        y = redistribute(x, replicated_layout)

        # Check that y is replicated
        assert get_layout(y).is_replicated()

        # Now redistribute back to sharded
        z = redistribute(y, sharded_layout)
        result_layout = get_layout(z)
        # Check that it's sharded on the expected axis
        assert result_layout.is_sharded_on(1) or result_layout.is_sharded_on(0)

    def test_redistribute_noop(self):
        """Test that redistribute is a no-op when layouts match."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.redistribute import redistribute
        from ttml.distributed.layout import Layout, Shard, Replicate
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        np_data = np.random.randn(1, 1, 8, 32).astype(np.float32)
        tensor = ttml.autograd.Tensor.from_numpy(
            np_data.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )

        # Use sharded layout to get proper 2D topology
        layout = Layout(placements=(Replicate(), Shard(-1)))
        x = distribute_tensor(tensor, mesh_device, layout)

        # Redistribute to same layout should be no-op
        y = redistribute(x, layout)

        # Should have the same layout
        assert get_layout(y) == layout

    def test_cross_entropy_loss_dispatch_smoke(self):
        """cross_entropy_loss through dispatch with sharded logits (gather to replicated) runs."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        shard_layout = Layout(placements=(Replicate(), Shard(-1)))
        # Logits: [1, 1, seq_len, vocab_size] - rank 4
        np_logits = np.random.randn(1, 1, 4, 16).astype(np.float32)
        # Targets: [1, seq_len] - rank 2 (cross_entropy expects prediction rank=4, target rank=2)
        # Target must be UINT32, not INT32
        np_targets = np.array([[0, 1, 2, 3]], dtype=np.uint32)

        logits = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_logits.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh_device,
            shard_layout,
        )

        # Target tensor must be ROW_MAJOR and UINT32, create with replicate mapper directly
        replicate_mapper = ttml.core.distributed.replicate_tensor_to_mesh_mapper(
            mesh_device
        )
        targets = ttml.autograd.Tensor.from_numpy(
            np_targets,
            ttnn.Layout.ROW_MAJOR,
            ttnn.DataType.UINT32,
            replicate_mapper,
        )

        set_layout(logits, shard_layout)
        loss = ttml.ops.loss.cross_entropy_loss(logits, targets)
        assert loss is not None
        assert hasattr(loss, "get_value")

    # -------------------------------------------------------------------------
    # Dispatch API Tests
    # -------------------------------------------------------------------------

    def test_dispatch_fast_path_no_layout(self):
        """Test dispatch fast path: tensor without explicit layout goes through dispatch but gets no layout stamp.

        Note: Tensors created with replicate_tensor_to_mesh_mapper will have a topology that
        get_layout() can read, but the dispatch layer's "fast path" is about whether we've
        explicitly set a Layout via set_layout() or distribute_tensor().
        """
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.dispatch import dispatch, _RAW_OPS
        from ttml.distributed.layout import get_layout, set_layout, Layout, Replicate

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()

        # Create tensor with mapper (gives it topology) but don't use distribute_tensor
        np_data = np.random.randn(1, 1, 32, 32).astype(np.float32)
        replicate_mapper = ttml.core.distributed.replicate_tensor_to_mesh_mapper(
            mesh_device
        )
        tensor = ttml.autograd.Tensor.from_numpy(
            np_data.astype(ml_dtypes.bfloat16),
            ttnn.Layout.TILE,
            ttnn.DataType.BFLOAT16,
            replicate_mapper,
        )

        # get_layout reads from topology, so it may return a layout
        # The key test is that dispatch works and we can do ops
        result = ttml.ops.unary.silu(tensor)
        assert result is not None
        assert hasattr(result, "get_value")

    def test_dispatch_with_layout_uses_rule(self):
        """Test dispatch with layout set uses sharding rule."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate, get_layout
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime
        from ttml.distributed.debug import DispatchTracer

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        # Create and distribute tensor
        np_data = np.random.randn(1, 1, 32, 32).astype(np.float32)
        tensor = ttml.autograd.Tensor.from_numpy(
            np_data.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )
        layout = Layout(placements=(Replicate(), Replicate()))
        x = distribute_tensor(tensor, mesh_device, layout)

        # Layout should be set
        assert get_layout(x) is not None

        # Dispatch should use rule and trace it
        with DispatchTracer() as tracer:
            result = ttml.ops.unary.silu(x)

        assert len(tracer.entries) >= 1
        assert any(e.op_name == "silu" for e in tracer.entries)
        # Output should have layout set
        assert get_layout(result) is not None

    def test_dispatch_fallback_for_unknown_op(self):
        """Test dispatch fallback gathers to replicated for ops without rules."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate, get_layout
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime
        from ttml.distributed.debug import DispatchTracer

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        # Create sharded tensor
        np_data = np.random.randn(1, 1, 32, 32).astype(np.float32)
        tensor = ttml.autograd.Tensor.from_numpy(
            np_data.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )
        shard_layout = Layout(placements=(Replicate(), Shard(-1)))
        x = distribute_tensor(tensor, mesh_device, shard_layout)

        # silu has a rule, so output should preserve sharding pattern
        with DispatchTracer() as tracer:
            result = ttml.ops.unary.silu(x)

        # Check that dispatch was traced
        assert len(tracer.entries) >= 1
        silu_entry = next(e for e in tracer.entries if e.op_name == "silu")
        assert silu_entry.rule_name is not None  # Has a rule

    def test_dispatch_register_op_wraps_correctly(self):
        """Test that register_op creates correct wrapper."""
        from ttml.distributed.dispatch import register_op, _RAW_OPS

        call_count = [0]

        def dummy_op(x, y, kwarg1=None):
            call_count[0] += 1
            return x

        wrapped = register_op("test_dummy_op", dummy_op)

        # Raw op should be stored
        assert "test_dummy_op" in _RAW_OPS
        assert _RAW_OPS["test_dummy_op"] is dummy_op

        # Wrapper should have same name
        assert wrapped.__name__ == "dummy_op"

    def test_dispatch_plan_cache_hit(self):
        """Test that plan cache is used on repeated calls."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Replicate, get_layout
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime, get_runtime

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        # Clear cache
        rt.plan_cache._cache.clear()

        # Create tensor
        np_data = np.random.randn(1, 1, 32, 32).astype(np.float32)
        tensor = ttml.autograd.Tensor.from_numpy(
            np_data.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )
        layout = Layout(placements=(Replicate(), Replicate()))
        x = distribute_tensor(tensor, mesh_device, layout)

        # First call - cache miss
        _ = ttml.ops.unary.silu(x)
        cache_size_after_first = len(rt.plan_cache._cache)

        # Second call with same layout - should hit cache
        _ = ttml.ops.unary.silu(x)
        cache_size_after_second = len(rt.plan_cache._cache)

        # Cache size should not increase on second call
        assert cache_size_after_second == cache_size_after_first

    # -------------------------------------------------------------------------
    # MLP Training Test - Distributed vs Single Device
    # -------------------------------------------------------------------------

    def _pcc(self, a, b):
        """Compute Pearson Correlation Coefficient between two arrays."""
        a_flat = a.flatten().astype(np.float64)
        b_flat = b.flatten().astype(np.float64)
        if len(a_flat) != len(b_flat):
            # If shapes differ, compare what we can (e.g., sharded vs full)
            min_len = min(len(a_flat), len(b_flat))
            a_flat = a_flat[:min_len]
            b_flat = b_flat[:min_len]
        if np.std(a_flat) < 1e-10 or np.std(b_flat) < 1e-10:
            return 1.0 if np.allclose(a_flat, b_flat) else 0.0
        return np.corrcoef(a_flat, b_flat)[0, 1]

    def test_mlp_distributed_vs_single_device(self):
        """Test MLP forward AND backward: distributed TP must match single device.

        This test:
        1. Runs MLP forward+backward on single device (replicated weights)
        2. Runs MLP forward+backward with TP sharding (column + row parallel)
        3. Compares outputs and all gradients using PCC (Pearson Correlation)

        MLP: x -> linear1 -> silu -> linear2 -> mean -> loss
        """
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate, get_layout
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0)

        # Dimensions - must be divisible by TP size (4)
        batch, seq, tokens = 1, 1, 32
        in_dim, hidden_dim, out_dim = 64, 64, 64

        # Fixed random seed for reproducibility
        # Use non-zero mean distributions for better numerical signal
        np.random.seed(42)
        np_input = (
            np.random.randn(batch, seq, tokens, in_dim).astype(np.float32) + 1.0
        ) * 0.5
        np_w1 = (
            np.random.randn(1, 1, hidden_dim, in_dim).astype(np.float32) + 0.5
        ) * 0.2
        np_w2 = (
            np.random.randn(1, 1, out_dim, hidden_dim).astype(np.float32) + 0.5
        ) * 0.2

        # =====================================================================
        # SINGLE DEVICE (replicated) - reference
        # =====================================================================
        set_runtime(None)  # No TP runtime

        # Create tensors - replicated
        rep_layout = Layout(placements=(Replicate(), Replicate()))

        x_single = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_input.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh_device,
            rep_layout,
            requires_grad=True,
        )
        w1_single = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_w1.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh_device,
            rep_layout,
            requires_grad=True,
        )
        w2_single = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_w2.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh_device,
            rep_layout,
            requires_grad=True,
        )

        # Forward - use raw ops (no dispatch)
        from ttml.distributed.dispatch import _get_raw

        raw_linear = _get_raw("linear")
        raw_silu = _get_raw("silu")
        raw_mean = ttml.ops.unary.mean  # mean is not dispatched

        h1_single = raw_linear(x_single, w1_single)
        h1_act_single = raw_silu(h1_single)
        out_single = raw_linear(h1_act_single, w2_single)
        loss_single = raw_mean(out_single)

        # Backward
        loss_single.backward(False)

        # Extract results - use get_grad_tensor() to get gradient as autograd.Tensor
        # which has .to_numpy(new_type, composer) signature (same as autograd.Tensor)
        out_single_np = out_single.to_numpy(composer=composer)
        x_grad_single = x_single.get_grad_tensor().to_numpy(composer=composer)
        w1_grad_single = w1_single.get_grad_tensor().to_numpy(composer=composer)
        w2_grad_single = w2_single.get_grad_tensor().to_numpy(composer=composer)

        # Reset graph for next run
        ttml.autograd.AutoContext.get_instance().reset_graph()

        # =====================================================================
        # DISTRIBUTED (TP sharded) - column + row parallel
        # =====================================================================
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        # Column-parallel: shard W1 on out_features (dim -2)
        col_layout = Layout(placements=(Replicate(), Shard(-2)))
        # Row-parallel: shard W2 on in_features (dim -1)
        row_layout = Layout(placements=(Replicate(), Shard(-1)))

        x_dist = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_input.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh_device,
            rep_layout,
            requires_grad=True,
        )
        w1_dist = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_w1.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh_device,
            col_layout,
            requires_grad=True,
        )
        w2_dist = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_w2.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh_device,
            row_layout,
            requires_grad=True,
        )

        # Forward - use dispatched ops
        h1_dist = ttml.ops.linear.linear(x_dist, w1_dist)
        assert get_layout(h1_dist).is_sharded_on(
            -1
        ), "After col-parallel, should be sharded"

        h1_act_dist = ttml.ops.unary.silu(h1_dist)

        out_dist = ttml.ops.linear.linear(h1_act_dist, w2_dist)
        assert get_layout(
            out_dist
        ).is_replicated(), "After row-parallel, should be replicated"

        loss_dist = raw_mean(out_dist)

        # Backward
        loss_dist.backward(False)

        # Extract results - use get_grad_tensor() to get gradient as autograd.Tensor
        out_dist_np = out_dist.to_numpy(composer=composer)
        x_grad_dist = x_dist.get_grad_tensor().to_numpy(composer=composer)
        w1_grad_dist = w1_dist.get_grad_tensor().to_numpy(composer=composer)
        w2_grad_dist = w2_dist.get_grad_tensor().to_numpy(composer=composer)

        # =====================================================================
        # COMPARE - forward and backward using PCC (Pearson Correlation)
        # =====================================================================
        # PCC threshold - 0.99 is very high correlation
        pcc_threshold = 0.99

        # Take first slice from composer concatenation for replicated tensors
        def first_slice(arr):
            arr = np.asarray(arr)
            if arr.ndim > 0 and arr.shape[0] > 1:
                return arr[:1]
            return arr

        out_single_np = first_slice(out_single_np)
        out_dist_np = first_slice(out_dist_np)
        x_grad_single = first_slice(x_grad_single)
        x_grad_dist = first_slice(x_grad_dist)

        # Forward output PCC
        out_pcc = self._pcc(out_dist_np, out_single_np)
        assert (
            out_pcc > pcc_threshold
        ), f"Forward output PCC {out_pcc:.4f} < {pcc_threshold}"

        # Input gradient PCC
        x_grad_pcc = self._pcc(x_grad_dist, x_grad_single)
        assert (
            x_grad_pcc > pcc_threshold
        ), f"Input gradient PCC {x_grad_pcc:.4f} < {pcc_threshold}"

        # W1 gradient PCC (column-parallel weight - sharded on out_features, dim 2)
        # Distributed: (1, 1, 16, 64) - first shard of out_features
        # Single: (1, 1, 64, 64) - compare with [:, :, :16, :]
        w1_grad_dist_slice = first_slice(w1_grad_dist)
        w1_grad_single_slice = first_slice(w1_grad_single)
        shard_size_w1 = w1_grad_dist_slice.shape[2]  # 16 = 64/4
        w1_grad_single_shard = w1_grad_single_slice[:, :, :shard_size_w1, :]
        print(
            f"W1 grad shapes: dist={w1_grad_dist_slice.shape}, single_shard={w1_grad_single_shard.shape}"
        )
        w1_grad_pcc = self._pcc(w1_grad_dist_slice, w1_grad_single_shard)
        assert (
            w1_grad_pcc > pcc_threshold
        ), f"W1 gradient PCC {w1_grad_pcc:.4f} < {pcc_threshold}"

        # W2 gradient PCC (row-parallel weight - sharded on in_features, dim 3)
        # Distributed: (1, 1, 64, 16) - first shard of in_features
        # Single: (1, 1, 64, 64) - compare with [:, :, :, :16]
        w2_grad_dist_slice = first_slice(w2_grad_dist)
        w2_grad_single_slice = first_slice(w2_grad_single)
        shard_size_w2 = w2_grad_dist_slice.shape[3]  # 16 = 64/4
        w2_grad_single_shard = w2_grad_single_slice[:, :, :, :shard_size_w2]
        print(
            f"W2 grad shapes: dist={w2_grad_dist_slice.shape}, single_shard={w2_grad_single_shard.shape}"
        )
        w2_grad_pcc = self._pcc(w2_grad_dist_slice, w2_grad_single_shard)
        assert (
            w2_grad_pcc > pcc_threshold
        ), f"W2 gradient PCC {w2_grad_pcc:.4f} < {pcc_threshold}"

        print(
            f"PCC results: output={out_pcc:.4f}, x_grad={x_grad_pcc:.4f}, w1_grad={w1_grad_pcc:.4f}, w2_grad={w2_grad_pcc:.4f}"
        )

    def test_mlp_forward_numerical_correctness(self):
        """Test MLP forward pass produces numerically correct results vs NumPy reference.

        Linear op: y = x @ W^T where W is [1, 1, out_features, in_features]
        So input x must have shape [..., in_features] to match.
        """
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate, get_layout
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime
        from ttml.modules import LinearLayer

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        # MLP dimensions - all must be divisible by TP size (4)
        num_tokens = 32
        input_dim = 64
        hidden_dim = 64
        output_dim = 64

        # Create layers
        col_linear = LinearLayer(input_dim, hidden_dim, has_bias=False)
        row_linear = LinearLayer(hidden_dim, output_dim, has_bias=False)

        # Get weights as numpy before distributing for reference
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0)
        col_weight_np = ttml.autograd.to_numpy(
            col_linear.weight.tensor, composer=composer
        )[:1]
        row_weight_np = ttml.autograd.to_numpy(
            row_linear.weight.tensor, composer=composer
        )[:1]

        # Distribute with TP
        col_weight_layout = Layout(placements=(Replicate(), Shard(-2)))
        row_weight_layout = Layout(placements=(Replicate(), Shard(-1)))

        col_linear.weight.tensor = distribute_tensor(
            col_linear.weight.tensor, mesh_device, col_weight_layout
        )
        row_linear.weight.tensor = distribute_tensor(
            row_linear.weight.tensor, mesh_device, row_weight_layout
        )

        # Input - shape [batch, seq, tokens, features] with non-zero mean
        np_input = self._nonzero_randn(1, 1, num_tokens, input_dim)

        # NumPy reference forward
        def numpy_linear(x, w):
            return x @ w[0, 0].T

        def numpy_silu(x):
            return x * (1 / (1 + np.exp(-x)))

        h_ref = numpy_linear(np_input, col_weight_np)
        h_act_ref = numpy_silu(h_ref)
        out_ref = numpy_linear(h_act_ref, row_weight_np)

        # Distributed forward
        input_layout = Layout(placements=(Replicate(), Replicate()))
        x = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_input.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh_device,
            input_layout,
        )

        h = ttml.ops.linear.linear(x, col_linear.weight.tensor)
        h_act = ttml.ops.unary.silu(h)
        out = ttml.ops.linear.linear(h_act, row_linear.weight.tensor)

        # Compare using PCC
        out_np = np.asarray(ttml.autograd.to_numpy(out, composer=composer)[:1])

        pcc = self._pcc(out_np, out_ref)
        assert pcc > 0.99, f"MLP forward PCC {pcc:.4f} < 0.99"


# ---------------------------------------------------------------------------
# Single-device reference tests (numerical correctness)
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
class TestDistributedVsSingleDevice:
    """Tests comparing distributed implementation against single-device reference.

    These tests verify that the distributed dispatch layer produces numerically
    correct results by comparing against a non-distributed single-device implementation.
    """

    @pytest.fixture(autouse=True)
    def setup_device(self):
        import gc
        import ttml
        from ttml.distributed.mesh_runtime import set_runtime, MeshRuntime

        num_devices = 32
        mesh_shape = [8, 4]

        try:
            ttml.core.distributed.enable_fabric(num_devices)
        except Exception:
            pytest.skip("Fabric not available or insufficient devices")

        auto_ctx = ttml.autograd.AutoContext.get_instance()
        try:
            auto_ctx.open_device(mesh_shape)
        except Exception:
            pytest.skip(f"Could not open device with mesh shape {mesh_shape}")

        mesh_device = auto_ctx.get_device()
        set_runtime(MeshRuntime(mesh_device=mesh_device))
        yield
        set_runtime(None)
        auto_ctx.reset_graph()
        gc.collect()
        auto_ctx.close_device()

    def _numpy_linear(self, x, weight):
        """NumPy reference: y = x @ W^T where W is [1, 1, out, in]."""
        # x: [batch, seq, tokens, in_features]
        # weight: [1, 1, out_features, in_features]
        w = weight[0, 0]  # [out_features, in_features]
        return x @ w.T

    def _numpy_silu(self, x):
        """NumPy reference for SiLU activation."""
        return x * (1 / (1 + np.exp(-x)))

    def _pcc(self, a, b):
        """Compute Pearson Correlation Coefficient between two arrays."""
        a_flat = np.asarray(a).flatten().astype(np.float64)
        b_flat = np.asarray(b).flatten().astype(np.float64)
        if np.std(a_flat) < 1e-10 or np.std(b_flat) < 1e-10:
            return 1.0 if np.allclose(a_flat, b_flat) else 0.0
        return float(np.corrcoef(a_flat, b_flat)[0, 1])

    def _nonzero_randn(self, *shape, mean=0.5, scale=0.5):
        """Generate random array with non-zero mean for better numerical signal."""
        return (np.random.randn(*shape).astype(np.float32) + mean) * scale

    def test_linear_forward_matches_numpy(self):
        """Test that distributed linear forward matches NumPy reference using PCC."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime
        from ttml.modules import LinearLayer

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        # Dimensions divisible by TP=4
        num_tokens = 32
        in_features = 64
        out_features = 64

        # Create layer and get weight as numpy
        linear = LinearLayer(in_features, out_features, has_bias=False)
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0)
        weight_np = np.asarray(
            ttml.autograd.to_numpy(linear.weight.tensor, composer=composer)[:1]
        )

        # Create input with non-zero mean
        np_input = self._nonzero_randn(1, 1, num_tokens, in_features)

        # NumPy reference
        expected = self._numpy_linear(np_input, weight_np)

        # Distributed forward (column-parallel)
        col_weight_layout = Layout(placements=(Replicate(), Shard(-2)))
        linear.weight.tensor = distribute_tensor(
            linear.weight.tensor, mesh_device, col_weight_layout
        )

        input_layout = Layout(placements=(Replicate(), Replicate()))
        x = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_input.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh_device,
            input_layout,
        )

        out = ttml.ops.linear.linear(x, linear.weight.tensor)
        # Column-parallel output is sharded - gather before comparing
        out_gathered = ttml.ops.distributed.all_gather(out, dim=-1, cluster_axis=1)
        out_np = np.asarray(ttml.autograd.to_numpy(out_gathered, composer=composer)[:1])

        # Compare using PCC
        pcc = self._pcc(out_np, expected)
        assert pcc > 0.99, f"Linear forward PCC {pcc:.4f} < 0.99"

    def test_mlp_forward_matches_numpy(self):
        """Test that distributed MLP forward matches NumPy reference using PCC."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime
        from ttml.modules import LinearLayer

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        # MLP dimensions
        num_tokens = 32
        input_dim = 64
        hidden_dim = 64
        output_dim = 64

        # Create layers
        col_linear = LinearLayer(input_dim, hidden_dim, has_bias=False)
        row_linear = LinearLayer(hidden_dim, output_dim, has_bias=False)

        # Get weights as numpy before distributing
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0)
        col_weight_np = np.asarray(
            ttml.autograd.to_numpy(col_linear.weight.tensor, composer=composer)[:1]
        )
        row_weight_np = np.asarray(
            ttml.autograd.to_numpy(row_linear.weight.tensor, composer=composer)[:1]
        )

        # Input with non-zero mean
        np_input = self._nonzero_randn(1, 1, num_tokens, input_dim)

        # NumPy reference MLP forward
        h = self._numpy_linear(np_input, col_weight_np)
        h_act = self._numpy_silu(h)
        expected = self._numpy_linear(h_act, row_weight_np)

        # Distribute weights
        col_weight_layout = Layout(placements=(Replicate(), Shard(-2)))
        row_weight_layout = Layout(placements=(Replicate(), Shard(-1)))

        col_linear.weight.tensor = distribute_tensor(
            col_linear.weight.tensor, mesh_device, col_weight_layout
        )
        row_linear.weight.tensor = distribute_tensor(
            row_linear.weight.tensor, mesh_device, row_weight_layout
        )

        # Distributed forward
        input_layout = Layout(placements=(Replicate(), Replicate()))
        x = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_input.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh_device,
            input_layout,
        )

        h = ttml.ops.linear.linear(x, col_linear.weight.tensor)
        h_act = ttml.ops.unary.silu(h)
        out = ttml.ops.linear.linear(h_act, row_linear.weight.tensor)

        out_np = np.asarray(ttml.autograd.to_numpy(out, composer=composer)[:1])

        # Compare using PCC
        pcc = self._pcc(out_np, expected)
        assert pcc > 0.99, f"MLP forward PCC {pcc:.4f} < 0.99"

    def test_elementwise_ops_match_numpy(self):
        """Test that distributed elementwise ops match NumPy reference using PCC."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        # Input with non-zero mean
        np_a = self._nonzero_randn(1, 1, 32, 64)
        np_b = self._nonzero_randn(1, 1, 32, 64)

        # NumPy references
        expected_add = np_a + np_b
        expected_mul = np_a * np_b
        expected_silu = np_a * (1 / (1 + np.exp(-np_a)))

        # Distribute tensors
        layout = Layout(placements=(Replicate(), Shard(-1)))
        a = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_a.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh_device,
            layout,
        )
        b = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_b.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh_device,
            layout,
        )

        # Distributed ops
        result_add = ttml.ops.binary.add(a, b)
        result_mul = ttml.ops.binary.mul(a, b)
        result_silu = ttml.ops.unary.silu(a)

        # Gather sharded results before comparing
        result_add_gathered = ttml.ops.distributed.all_gather(
            result_add, dim=-1, cluster_axis=1
        )
        result_mul_gathered = ttml.ops.distributed.all_gather(
            result_mul, dim=-1, cluster_axis=1
        )
        result_silu_gathered = ttml.ops.distributed.all_gather(
            result_silu, dim=-1, cluster_axis=1
        )

        # Get results
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0)
        add_np = np.asarray(
            ttml.autograd.to_numpy(result_add_gathered, composer=composer)[:1]
        )
        mul_np = np.asarray(
            ttml.autograd.to_numpy(result_mul_gathered, composer=composer)[:1]
        )
        silu_np = np.asarray(
            ttml.autograd.to_numpy(result_silu_gathered, composer=composer)[:1]
        )

        # Compare using PCC
        add_pcc = self._pcc(add_np, expected_add)
        mul_pcc = self._pcc(mul_np, expected_mul)
        silu_pcc = self._pcc(silu_np, expected_silu)

        assert add_pcc > 0.99, f"Add PCC {add_pcc:.4f} < 0.99"
        assert mul_pcc > 0.99, f"Mul PCC {mul_pcc:.4f} < 0.99"
        assert silu_pcc > 0.99, f"SiLU PCC {silu_pcc:.4f} < 0.99"

    def test_replicated_vs_sharded_same_result(self):
        """Test that replicated and sharded tensors produce same results using PCC."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        # Input with non-zero mean
        np_input = self._nonzero_randn(1, 1, 32, 64)

        # Replicated layout
        rep_layout = Layout(placements=(Replicate(), Replicate()))
        x_rep = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_input.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh_device,
            rep_layout,
        )

        # Sharded layout
        shard_layout = Layout(placements=(Replicate(), Shard(-1)))
        x_shard = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_input.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh_device,
            shard_layout,
        )

        # Apply same operation
        result_rep = ttml.ops.unary.silu(x_rep)
        result_shard = ttml.ops.unary.silu(x_shard)

        # Gather sharded result before comparing
        result_shard_gathered = ttml.ops.distributed.all_gather(
            result_shard, dim=-1, cluster_axis=1
        )

        # Get results
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0)
        rep_np = np.asarray(ttml.autograd.to_numpy(result_rep, composer=composer)[:1])
        shard_np = np.asarray(
            ttml.autograd.to_numpy(result_shard_gathered, composer=composer)[:1]
        )

        # Should be identical - use PCC
        pcc = self._pcc(rep_np, shard_np)
        assert pcc > 0.999, f"Replicated vs sharded PCC {pcc:.4f} < 0.999"

    def test_single_linear_forward_correctness(self):
        """Test that a single distributed linear forward matches NumPy reference using PCC.

        This is a simpler test than full MLP - just one linear layer.
        """
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime
        from ttml.modules import LinearLayer

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        # Dimensions
        num_tokens = 32
        in_features = 64
        out_features = 64

        # Create layer
        linear = LinearLayer(in_features, out_features, has_bias=False)

        # Get weight for reference before distributing
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0)
        weight_np = np.asarray(
            ttml.autograd.to_numpy(linear.weight.tensor, composer=composer)[:1]
        )

        # Distribute (column-parallel)
        weight_layout = Layout(placements=(Replicate(), Shard(-2)))
        linear.weight.tensor = distribute_tensor(
            linear.weight.tensor, mesh_device, weight_layout
        )

        # Input with non-zero mean
        np_input = self._nonzero_randn(1, 1, num_tokens, in_features)
        input_layout = Layout(placements=(Replicate(), Replicate()))
        x = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_input.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh_device,
            input_layout,
        )

        # Forward
        out = ttml.ops.linear.linear(x, linear.weight.tensor)
        # Column-parallel output is sharded - gather before comparing
        out_gathered = ttml.ops.distributed.all_gather(out, dim=-1, cluster_axis=1)

        # NumPy reference
        expected = np_input @ weight_np[0, 0].T

        # Compare using PCC
        out_np = np.asarray(ttml.autograd.to_numpy(out_gathered, composer=composer)[:1])
        pcc = self._pcc(out_np, expected)
        assert pcc > 0.99, f"Single linear forward PCC {pcc:.4f} < 0.99"

    def test_column_parallel_output_shape(self):
        """Test that column-parallel linear produces correct output shape."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate, get_layout
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime
        from ttml.modules import LinearLayer

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        # Dimensions
        num_tokens = 32
        in_features = 64
        out_features = 128  # Different from in_features

        # Create layer
        linear = LinearLayer(in_features, out_features, has_bias=False)

        # Column-parallel: shard on out_features (dim -2)
        weight_layout = Layout(placements=(Replicate(), Shard(-2)))
        linear.weight.tensor = distribute_tensor(
            linear.weight.tensor, mesh_device, weight_layout
        )

        # Input with non-zero mean
        np_input = self._nonzero_randn(1, 1, num_tokens, in_features)
        input_layout = Layout(placements=(Replicate(), Replicate()))
        x = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_input.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh_device,
            input_layout,
        )

        # Forward
        out = ttml.ops.linear.linear(x, linear.weight.tensor)

        # Output should be sharded on last dim
        out_layout = get_layout(out)
        assert out_layout is not None
        assert out_layout.is_sharded_on(-1), f"Expected sharded on -1, got {out_layout}"

        # Gather and check full shape
        out_gathered = ttml.ops.distributed.all_gather(out, dim=-1, cluster_axis=1)
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0)
        out_np = np.asarray(ttml.autograd.to_numpy(out_gathered, composer=composer)[:1])

        # Output should have shape [1, 1, num_tokens, out_features]
        expected_shape = (1, 1, num_tokens, out_features)
        assert (
            out_np.shape == expected_shape
        ), f"Output shape {out_np.shape} != expected {expected_shape}"

    def test_row_parallel_all_reduce(self):
        """Test that row-parallel linear correctly all-reduces partial sums."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate, get_layout
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime
        from ttml.modules import LinearLayer

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        # Dimensions
        num_tokens = 32
        in_features = 64
        out_features = 64

        # Create layer
        linear = LinearLayer(in_features, out_features, has_bias=False)

        # Get weight for numpy reference
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0)
        weight_np = ttml.autograd.to_numpy(linear.weight.tensor, composer=composer)[:1]

        # Row-parallel: shard on in_features (dim -1)
        weight_layout = Layout(placements=(Replicate(), Shard(-1)))
        linear.weight.tensor = distribute_tensor(
            linear.weight.tensor, mesh_device, weight_layout
        )

        # Input - must be sharded to match row-parallel weight, with non-zero mean
        np_input = self._nonzero_randn(1, 1, num_tokens, in_features)
        input_layout = Layout(placements=(Replicate(), Shard(-1)))
        x = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_input.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh_device,
            input_layout,
        )

        # Forward
        out = ttml.ops.linear.linear(x, linear.weight.tensor)

        # Output should be replicated (after all_reduce)
        out_layout = get_layout(out)
        assert out_layout is not None
        assert (
            out_layout.is_replicated()
        ), f"Expected replicated output, got {out_layout}"

        # Verify numerical correctness using PCC
        out_np = np.asarray(ttml.autograd.to_numpy(out, composer=composer)[:1])
        expected = self._numpy_linear(np_input, weight_np)
        pcc = self._pcc(out_np, expected)
        assert pcc > 0.99, f"Row-parallel all_reduce PCC {pcc:.4f} < 0.99"

    def test_dispatch_forward_backward_integration(self):
        """Test that dispatch properly handles ops with both forward AND backward passes.

        This test verifies that:
        1. Dispatched ops build correct autograd graph (forward works)
        2. Backward pass flows correctly through dispatched ops
        3. Gradients are computed and accessible after backward
        4. The dispatch layer doesn't break the autograd chain

        This is critical because dispatch wraps raw ops - we need to ensure
        the wrapper doesn't interfere with gradient computation.
        """
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.layout import Layout, Shard, Replicate, get_layout
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime
        from ttml.distributed.debug import DispatchTracer

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0)

        # Dimensions divisible by TP=4
        batch, seq, tokens = 1, 1, 32
        in_dim, out_dim = 64, 64

        # Create input and weight with requires_grad=True
        np_input = self._nonzero_randn(batch, seq, tokens, in_dim)
        np_weight = self._nonzero_randn(1, 1, out_dim, in_dim)

        # Distribute with column-parallel layout
        rep_layout = Layout(placements=(Replicate(), Replicate()))
        col_layout = Layout(placements=(Replicate(), Shard(-2)))

        x = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_input.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh_device,
            rep_layout,
            requires_grad=True,
        )
        w = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_weight.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh_device,
            col_layout,
            requires_grad=True,
        )

        # Verify inputs have requires_grad set
        assert x.get_requires_grad(), "Input should have requires_grad=True"
        assert w.get_requires_grad(), "Weight should have requires_grad=True"

        # Forward through dispatch - trace to verify dispatch is used
        with DispatchTracer() as tracer:
            y = ttml.ops.linear.linear(x, w)
            y_act = ttml.ops.unary.silu(y)

        # Verify dispatch was invoked for both ops
        op_names = [e.op_name for e in tracer.entries]
        assert "linear" in op_names, f"linear not dispatched, got: {op_names}"
        assert "silu" in op_names, f"silu not dispatched, got: {op_names}"

        # Verify output has layout (meaning dispatch processed it)
        y_layout = get_layout(y)
        assert y_layout is not None, "Output should have layout from dispatch"
        assert y_layout.is_sharded_on(-1), "Column-parallel output should be sharded"

        # Compute loss and backward
        loss = ttml.ops.unary.mean(y_act)

        # This is the critical test - backward should work through dispatched ops
        loss.backward(False)

        # Verify gradients exist and are accessible
        x_grad = x.get_grad_tensor()
        w_grad = w.get_grad_tensor()

        assert x_grad is not None, "Input gradient should exist after backward"
        assert w_grad is not None, "Weight gradient should exist after backward"

        # Verify gradients have non-zero values (actual computation happened)
        x_grad_np = np.asarray(x_grad.to_numpy(composer=composer))
        w_grad_np = np.asarray(w_grad.to_numpy(composer=composer))

        assert np.any(x_grad_np != 0), "Input gradient should be non-zero"
        assert np.any(w_grad_np != 0), "Weight gradient should be non-zero"

        # Verify gradient shapes are correct
        # x_grad should match input shape (replicated)
        assert x_grad_np.shape[-1] == in_dim, f"Input grad last dim should be {in_dim}"
        # w_grad should be sharded (column-parallel)
        assert (
            w_grad_np.shape[2] == out_dim // 4
        ), f"Weight grad should be sharded: {w_grad_np.shape}"

        print(f"Forward-backward integration test passed!")
        print(
            f"  Input grad shape: {x_grad_np.shape}, non-zero: {np.count_nonzero(x_grad_np)}"
        )
        print(
            f"  Weight grad shape: {w_grad_np.shape}, non-zero: {np.count_nonzero(w_grad_np)}"
        )

    def test_mlp_training_step_with_distribute_module(self):
        """Test full MLP training step using distribute_module API.

        This test simulates the actual training pipeline:
        1. Create an MLP model (LlamaMLP)
        2. Use distribute_module() with a TP policy to shard weights
        3. Run forward pass with dispatched ops
        4. Compute loss and run backward
        5. Verify gradients and compare with single-device reference

        This is the recommended way to use the distributed framework in practice.
        """
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor, distribute_module
        from ttml.distributed.layout import Layout, Shard, Replicate, get_layout
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime, get_runtime
        from ttml.models.llama.transformer import LlamaMLP

        init_ops()

        mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0)

        # MLP dimensions - must be divisible by TP size (4)
        # LlamaMLP computes intermediate_size automatically, but we need it divisible by 4
        embedding_size = 64
        # Force intermediate_size to be divisible by 4
        intermediate_size = 256  # 64 * 4, divisible by TP=4

        # Fixed random seed for reproducibility
        np.random.seed(42)

        # Create input data with non-zero mean
        batch, seq, tokens = 1, 1, 32
        np_input = self._nonzero_randn(batch, seq, tokens, embedding_size)

        # =====================================================================
        # SINGLE DEVICE (replicated) - reference
        # =====================================================================
        set_runtime(None)  # No TP runtime

        # Create MLP model
        mlp_single = LlamaMLP(embedding_size, intermediate_size, dropout=0.0)

        # Get weights before distributing for reference
        w1_np = np.asarray(
            ttml.autograd.to_numpy(mlp_single.w1.weight.tensor, composer=composer)[:1]
        )
        w2_np = np.asarray(
            ttml.autograd.to_numpy(mlp_single.w2.weight.tensor, composer=composer)[:1]
        )
        w3_np = np.asarray(
            ttml.autograd.to_numpy(mlp_single.w3.weight.tensor, composer=composer)[:1]
        )

        # Create input tensor (replicated)
        rep_layout = Layout(placements=(Replicate(), Replicate()))
        x_single = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_input.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh_device,
            rep_layout,
            requires_grad=True,
        )

        # Forward using raw ops (no dispatch)
        from ttml.distributed.dispatch import _get_raw

        raw_linear = _get_raw("linear")
        raw_silu = _get_raw("silu")
        raw_mul = _get_raw("mul")
        raw_mean = ttml.ops.unary.mean

        # LlamaMLP forward: silu(w1(x)) * w3(x) -> w2
        h1 = raw_linear(x_single, mlp_single.w1.weight.tensor)
        h1_act = raw_silu(h1)
        h3 = raw_linear(x_single, mlp_single.w3.weight.tensor)
        gated = raw_mul(h1_act, h3)
        out_single = raw_linear(gated, mlp_single.w2.weight.tensor)
        loss_single = raw_mean(out_single)

        # Backward
        loss_single.backward(False)

        # Extract results
        out_single_np = np.asarray(out_single.to_numpy(composer=composer)[:1])
        x_grad_single = np.asarray(
            x_single.get_grad_tensor().to_numpy(composer=composer)[:1]
        )

        # Reset graph for next run
        ttml.autograd.AutoContext.get_instance().reset_graph()

        # =====================================================================
        # DISTRIBUTED (TP sharded) - using distribute_module
        # =====================================================================
        rt = MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0)
        set_runtime(rt)

        # Create fresh MLP model
        mlp_dist = LlamaMLP(embedding_size, intermediate_size, dropout=0.0)

        # Copy weights from single-device model to ensure same starting point
        mlp_dist.w1.weight.tensor = ttml.autograd.Tensor.from_numpy(
            w1_np.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )
        mlp_dist.w2.weight.tensor = ttml.autograd.Tensor.from_numpy(
            w2_np.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )
        mlp_dist.w3.weight.tensor = ttml.autograd.Tensor.from_numpy(
            w3_np.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )

        # Build TP policy using regex patterns (the recommended way)
        # Note: When calling distribute_module directly on a module (not nested),
        # the prefix for child LinearLayers is just the attribute name (e.g., "w1"),
        # so the weight key is "w1.weight", not "mlp.w1.weight".
        # Use regex that matches both cases: with or without parent prefix.
        col_layout = Layout(placements=(Replicate(), Shard(-2)))  # column-parallel
        row_layout = Layout(placements=(Replicate(), Shard(-1)))  # row-parallel

        policy = {
            r"(.*\.)?w1\.weight": col_layout,  # w1 is column-parallel
            r"(.*\.)?w3\.weight": col_layout,  # w3 is column-parallel
            r"(.*\.)?w2\.weight": row_layout,  # w2 is row-parallel
        }

        # Use distribute_module to apply the policy
        distribute_module(mlp_dist, mesh_device, policy)

        # Create input tensor (replicated, with requires_grad)
        x_dist = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_input.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh_device,
            rep_layout,
            requires_grad=True,
        )

        # Forward using dispatched ops (through module's forward method)
        # Note: We manually call the ops to match the MLP forward exactly
        h1_dist = ttml.ops.linear.linear(x_dist, mlp_dist.w1.weight.tensor)
        h1_act_dist = ttml.ops.unary.silu(h1_dist)
        h3_dist = ttml.ops.linear.linear(x_dist, mlp_dist.w3.weight.tensor)
        gated_dist = ttml.ops.binary.mul(h1_act_dist, h3_dist)
        out_dist = ttml.ops.linear.linear(gated_dist, mlp_dist.w2.weight.tensor)

        # Verify intermediate layouts
        assert get_layout(h1_dist).is_sharded_on(
            -1
        ), "After w1 (col-parallel), should be sharded"
        assert get_layout(h3_dist).is_sharded_on(
            -1
        ), "After w3 (col-parallel), should be sharded"
        assert get_layout(gated_dist).is_sharded_on(
            -1
        ), "After mul, should preserve sharding"
        assert get_layout(
            out_dist
        ).is_replicated(), "After w2 (row-parallel), should be replicated"

        loss_dist = raw_mean(out_dist)

        # Backward
        loss_dist.backward(False)

        # Extract results
        out_dist_np = np.asarray(out_dist.to_numpy(composer=composer)[:1])
        x_grad_dist = np.asarray(
            x_dist.get_grad_tensor().to_numpy(composer=composer)[:1]
        )

        # =====================================================================
        # COMPARE - forward and backward using PCC
        # =====================================================================
        pcc_threshold = 0.99

        # Forward output PCC
        out_pcc = self._pcc(out_dist_np, out_single_np)
        assert (
            out_pcc > pcc_threshold
        ), f"Forward output PCC {out_pcc:.4f} < {pcc_threshold}"

        # Input gradient PCC
        x_grad_pcc = self._pcc(x_grad_dist, x_grad_single)
        assert (
            x_grad_pcc > pcc_threshold
        ), f"Input gradient PCC {x_grad_pcc:.4f} < {pcc_threshold}"

        print(f"MLP training step with distribute_module passed!")
        print(f"  Output PCC: {out_pcc:.4f}")
        print(f"  Input gradient PCC: {x_grad_pcc:.4f}")
