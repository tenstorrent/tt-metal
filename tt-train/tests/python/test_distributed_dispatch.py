# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for ttml.distributed: layout, rules, registry, trace, and mesh integration.

- Unit tests: no device.
- ``@pytest.mark.requires_device``: 32-device fabric mesh [8, 4] (skipped if unavailable).
"""

from __future__ import annotations

import pytest
import numpy as np

from ttml.distributed.layout import (
    Layout,
    Shard,
    Replicate,
    replicated_layout,
    get_layout,
    layout_from_topology,
    layout_to_mapper_config,
)
from ttml.distributed.utils import is_distributed
from ttml.distributed.cache import PlanCache
from ttml.distributed.rules.registry import (
    ShardingPlan,
    register_rule,
    get_rule,
    _OP_RULES,
    register_module_rule,
    get_module_rule,
    _MODULE_RULES,
    AllReduce,
)
from ttml.distributed.debug import TraceEntry, dispatch_trace, DispatchTracer


def _layout_rep_shard(tp_axis: int = 1, dim: int = -1) -> Layout:
    return Layout(ndim=2, axis_placements={tp_axis: Shard(dim)})


def _layout_col_parallel(tp_axis: int = 1) -> Layout:
    return Layout(ndim=2, axis_placements={tp_axis: Shard(-2)})


def _layout_row_parallel(tp_axis: int = 1) -> Layout:
    return Layout(ndim=2, axis_placements={tp_axis: Shard(-1)})


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


class TestLayout:
    def test_shard_replicate_equality(self):
        assert Shard(0) == Shard(0)
        assert Shard(0) != Shard(1)
        assert Replicate() == Replicate()

    def test_layout_with_placement_and_shard_dim(self):
        rep = replicated_layout(2)
        sh = rep.with_placement(1, Shard(-1))
        assert sh.placements == (Replicate(), Shard(-1))
        assert sh.shard_dim(1) == -1
        assert sh.is_sharded_on(1)
        assert not sh.is_sharded_on(0)

    def test_layout_hash_equality(self):
        a = _layout_rep_shard()
        b = Layout(ndim=2, axis_placements={1: Shard(-1)})
        assert a == b
        assert hash(a) == hash(b)

    def test_layout_from_topology_mock(self):
        import ttnn

        class MockTopology:
            def placements(self):
                return [ttnn.PlacementShard(-1), ttnn.PlacementReplicate()]

        layout = layout_from_topology(MockTopology(), mesh_ndim=2)
        assert layout.ndim == 2
        assert isinstance(layout.placements[0], Shard)
        assert isinstance(layout.placements[1], Replicate)

    def test_layout_to_mapper_config(self):
        layout = Layout(ndim=2, axis_placements={1: Shard(-1)})
        assert layout_to_mapper_config(layout) is not None


# ---------------------------------------------------------------------------
# Plan cache & linear rule
# ---------------------------------------------------------------------------


class TestPlanCache:
    def test_lru_put_get(self):
        cache = PlanCache(maxsize=2)
        la = replicated_layout(2)
        lb = _layout_rep_shard()
        lc = Layout(ndim=2, axis_placements={0: Shard(0)})
        key_a = ("op", (la,), ())
        key_b = ("op", (lb,), ())
        key_c = ("op", (lc,), ())
        cache.put(key_a, "p1")
        cache.put(key_b, "p2")
        cache.put(key_c, "p3")
        assert cache.get(key_a) is None
        assert cache.get(key_b) == "p2"
        assert cache.get(key_c) == "p3"


class TestLinearMatmulRule:
    def test_column_parallel_plan_output_sharded_last_dim(self):
        from ttml.distributed.rules.matmul import _linear_matmul_plan

        inp = replicated_layout(2)
        wt = _layout_col_parallel(1)
        plan = _linear_matmul_plan(inp, wt, runtime=None)
        assert plan.output_layout.is_sharded_on(1)
        assert plan.output_layout.shard_dim(1) == -1

    def test_row_parallel_plan_output_replicated(self):
        from ttml.distributed.rules.matmul import _linear_matmul_plan

        inp = _layout_rep_shard(1, -1)
        wt = _layout_row_parallel(1)
        plan = _linear_matmul_plan(inp, wt, runtime=None)
        assert plan.output_layout.is_replicated()


# ---------------------------------------------------------------------------
# Registry & policy
# ---------------------------------------------------------------------------


class TestRuleRegistry:
    def test_register_and_get_rule(self):
        @register_rule("__test_tmp_rule__")
        def _rule(x, *a, runtime=None, **k):
            return ShardingPlan(input_layouts=[x], output_layout=x)

        assert get_rule("__test_tmp_rule__") is _rule
        del _OP_RULES["__test_tmp_rule__"]

    def test_module_rule_inheritance(self):
        class Base:
            pass

        class Derived(Base):
            pass

        @register_module_rule(Base)
        def _dist(m, mesh, tp_axis, cp_axis=None):
            m.ok = True
            return m

        assert get_module_rule(Derived) is _dist
        d = Derived()
        _dist(d, None, 0, None)
        assert d.ok
        del _MODULE_RULES[Base]


class TestPolicyMatch:
    def test_exact_and_regex(self):
        from ttml.distributed.training import _match_plan

        policy = {
            "a.weight": Layout(ndim=2, axis_placements={1: Shard(-2)}),
            r".*\.bias": Layout(ndim=2, axis_placements={1: Shard(-1)}),
        }
        assert _match_plan("a.weight", policy).is_sharded_on(1)
        assert _match_plan("layer.a.bias", policy).shard_dim(1) == -1


class TestBiasLayoutInvariants:
    """Column-parallel bias is sharded on last dim; row-parallel bias stays replicated."""

    def test_column_vs_row_expected_layouts(self):
        col_bias = Layout(ndim=2, axis_placements={1: Shard(-1)})
        assert col_bias.shard_dim(1) == -1
        row_bias = replicated_layout(2)
        assert row_bias.is_replicated()


class TestIsDistributedHelper:
    def test_non_tensors(self):
        assert not is_distributed(None)
        assert not is_distributed(object())


# ---------------------------------------------------------------------------
# Trace entry
# ---------------------------------------------------------------------------


class TestTraceEntry:
    def test_to_dict_includes_op_kwargs(self):
        entry = TraceEntry(
            op_name="all_gather",
            input_layouts=[],
            rule_name="all_gather_rule",
            plan=None,
            pre_collectives=[],
            redistributions=[],
            post_collectives=[],
            output_layout=None,
            op_kwargs={"dim": -1, "cluster_axis": 1},
        )
        d = entry.to_dict()
        assert d["op_kwargs"] == {"dim": -1, "cluster_axis": 1}
        assert d["input_shapes"] == []


# ---------------------------------------------------------------------------
# Custom op rule (unit)
# ---------------------------------------------------------------------------


class TestCustomOpRuleUnit:
    def test_rule_with_all_reduce_cleanup_registry(self):
        @register_rule("__test_arule__")
        def _r(layout: Layout, *e, runtime=None, **kw):
            if layout.is_sharded_on(1):
                return ShardingPlan(
                    input_layouts=[layout],
                    output_layout=replicated_layout(layout.ndim),
                    post_collectives=[AllReduce(mesh_axis=1)],
                )
            return ShardingPlan(input_layouts=[layout], output_layout=layout)

        sh = _layout_rep_shard()
        plan = _r(sh, runtime=None)
        assert isinstance(plan.post_collectives[0], AllReduce)
        del _OP_RULES["__test_arule__"]


# ---------------------------------------------------------------------------
# Custom module rule (unit)
# ---------------------------------------------------------------------------


class TestCustomModuleRuleUnit:
    def test_register_and_invoke(self):
        class Box:
            def __init__(self):
                self.marked = False

        @register_module_rule(Box)
        def _dist(m, mesh_device, tp_axis, cp_axis=None):
            m.marked = True
            return m

        b = Box()
        _dist(b, None, 0, None)
        assert b.marked
        del _MODULE_RULES[Box]


# ---------------------------------------------------------------------------
# Device-backed integration (32-device mesh)
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
class TestDistributedMesh:
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
        set_runtime(MeshRuntime(mesh_device=mesh_device, tp_axis=1, dp_axis=0))
        yield
        set_runtime(None)
        auto_ctx.reset_graph()
        gc.collect()
        auto_ctx.close_device()

    def _pcc(self, a, b):
        a_flat = np.asarray(a).flatten().astype(np.float64)
        b_flat = np.asarray(b).flatten().astype(np.float64)
        if np.std(a_flat) < 1e-10 or np.std(b_flat) < 1e-10:
            return 1.0 if np.allclose(a_flat, b_flat) else 0.0
        return float(np.corrcoef(a_flat, b_flat)[0, 1])

    def _nz(self, *shape, mean=0.5, scale=0.5):
        return (np.random.randn(*shape).astype(np.float32) + mean) * scale

    def test_distribute_tensor_preserves_layout_metadata(self):
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed.training import distribute_tensor

        mesh = ttml.autograd.AutoContext.get_instance().get_device()
        layout = _layout_rep_shard(1, -1)
        np_data = np.ones((1, 1, 4, 16), dtype=np.float32)
        t = ttml.autograd.Tensor.from_numpy(
            np_data.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
        )
        out = distribute_tensor(t, mesh, layout)
        assert get_layout(out) == layout

    def test_dispatch_trace_records_op(self):
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor

        init_ops()
        mesh = ttml.autograd.AutoContext.get_instance().get_device()
        x = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                self._nz(1, 1, 8, 32).astype(ml_dtypes.bfloat16),
                layout=ttnn.Layout.TILE,
            ),
            mesh,
            replicated_layout(2),
        )
        dispatch_trace.clear()
        with DispatchTracer():
            ttml.ops.unary.silu(x)
        assert any(e.op_name == "silu" for e in dispatch_trace.entries)

    def test_unary_preserves_sharded_layout(self):
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor

        init_ops()
        mesh = ttml.autograd.AutoContext.get_instance().get_device()
        layout = _layout_rep_shard(1, -1)
        x = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                self._nz(1, 1, 4, 16).astype(ml_dtypes.bfloat16),
                layout=ttnn.Layout.TILE,
            ),
            mesh,
            layout,
        )
        y = ttml.ops.unary.silu(x)
        assert get_layout(y) == layout

    def test_linear_column_and_row_output_layouts(self):
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor

        init_ops()
        mesh = ttml.autograd.AutoContext.get_instance().get_device()
        np_x = self._nz(1, 1, 8, 64)
        np_w = self._nz(1, 1, 128, 64)
        x = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_x.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh,
            replicated_layout(2),
        )
        w_col = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_w.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh,
            _layout_col_parallel(1),
        )
        y_col = ttml.ops.linear.linear(x, w_col, None)
        assert get_layout(y_col).is_sharded_on(1)
        assert get_layout(y_col).shard_dim(1) == -1

        np_x2 = self._nz(1, 1, 8, 128)
        np_w2 = self._nz(1, 1, 64, 128)
        x2 = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_x2.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh,
            _layout_rep_shard(1, -1),
        )
        w_row = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_w2.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh,
            _layout_row_parallel(1),
        )
        y_row = ttml.ops.linear.linear(x2, w_row, None)
        assert get_layout(y_row).is_replicated()

    def test_mlp_parallelize_module_forward_backward_matches_reference(self):
        """Two-layer SiLU MLP: TP via parallelize_module vs replicated reference (forward + grads)."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import (
            init_ops,
            parallelize_module,
            ColwiseParallel,
            RowwiseParallel,
        )
        from ttml.distributed.training import distribute_tensor
        from ttml.distributed.mesh_runtime import set_runtime, get_runtime
        from ttml.modules import AbstractModuleBase, LinearLayer

        init_ops()
        mesh = ttml.autograd.AutoContext.get_instance().get_device()
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(mesh, 0)
        tokens, d_in, d_h, d_out = 32, 64, 64, 64
        rep = replicated_layout(2)
        tp_axis = 1

        class TwoLayerSiluMLP(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                self.fc1 = LinearLayer(d_in, d_h, has_bias=False)
                self.fc2 = LinearLayer(d_h, d_out, has_bias=False)

            def forward(self, x):
                # Call submodules so ColwiseParallel/RowwiseParallel wrappers run (broadcast, all_reduce).
                h = self.fc1(x)
                h = ttml.ops.unary.silu(h)
                return self.fc2(h)

        def _clone_weights_bf16(template: TwoLayerSiluMLP):
            w1 = np.asarray(
                ttml.autograd.to_numpy(template.fc1.weight.tensor, composer=composer)[
                    :1
                ]
            )
            w2 = np.asarray(
                ttml.autograd.to_numpy(template.fc2.weight.tensor, composer=composer)[
                    :1
                ]
            )
            return w1, w2

        def _load_weights(m: TwoLayerSiluMLP, w1_np, w2_np):
            m.fc1.weight.tensor = ttml.autograd.Tensor.from_numpy(
                np.asarray(w1_np), layout=ttnn.Layout.TILE
            )
            m.fc2.weight.tensor = ttml.autograd.Tensor.from_numpy(
                np.asarray(w2_np), layout=ttnn.Layout.TILE
            )
            # from_numpy defaults to no grad; need True for weight grads after backward
            m.fc1.weight.tensor.set_requires_grad(True)
            m.fc2.weight.tensor.set_requires_grad(True)

        np.random.seed(42)
        donor = TwoLayerSiluMLP()
        w1_np, w2_np = _clone_weights_bf16(donor)
        np_x = self._nz(1, 1, tokens, d_in)

        rt = get_runtime()
        set_runtime(None)
        m_ref = TwoLayerSiluMLP()
        _load_weights(m_ref, w1_np, w2_np)
        m_ref.fc1.weight.tensor = distribute_tensor(
            m_ref.fc1.weight.tensor, mesh, rep, requires_grad=True
        )
        m_ref.fc2.weight.tensor = distribute_tensor(
            m_ref.fc2.weight.tensor, mesh, rep, requires_grad=True
        )
        x_ref = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(np_x, layout=ttnn.Layout.TILE),
            mesh,
            rep,
            requires_grad=True,
        )
        out_ref = m_ref(x_ref)
        loss_ref = ttml.ops.unary.mean(out_ref)
        loss_ref.backward(False)
        out_ref_np = np.asarray(ttml.autograd.to_numpy(out_ref, composer=composer)[:1])
        xg_ref = np.asarray(x_ref.get_grad_tensor().to_numpy(composer=composer)[:1])
        w1g = m_ref.fc1.weight.tensor.get_grad_tensor()
        w2g = m_ref.fc2.weight.tensor.get_grad_tensor()
        assert w1g is not None and w2g is not None, "reference weight grads missing"
        w1g_ref = np.asarray(w1g.to_numpy(composer=composer)[:1])
        w2g_ref = np.asarray(w2g.to_numpy(composer=composer)[:1])
        set_runtime(rt)

        ttml.autograd.AutoContext.get_instance().reset_graph()

        m_tp = TwoLayerSiluMLP()
        _load_weights(m_tp, w1_np, w2_np)
        tp_plan = {
            "fc1": ColwiseParallel(),
            "fc2": RowwiseParallel(),
        }
        parallelize_module(m_tp, mesh, tp_plan, tp_axis=tp_axis)
        x_tp = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_x.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh,
            rep,
            requires_grad=True,
        )
        out_tp = m_tp(x_tp)
        loss_tp = ttml.ops.unary.mean(out_tp)
        loss_tp.backward(False)
        out_tp_np = np.asarray(ttml.autograd.to_numpy(out_tp, composer=composer)[:1])
        xg_tp = np.asarray(x_tp.get_grad_tensor().to_numpy(composer=composer)[:1])

        w1g_tp = m_tp.fc1.weight.tensor.get_grad_tensor()
        w2g_tp = m_tp.fc2.weight.tensor.get_grad_tensor()
        assert w1g_tp is not None and w2g_tp is not None, "TP weight grads missing"
        w1g_tp_full = ttml.ops.distributed.all_gather(
            w1g_tp, dim=2, cluster_axis=tp_axis
        )
        w1g_tp_np = np.asarray(w1g_tp_full.to_numpy(composer=composer)[:1])

        w2g_tp_full = ttml.ops.distributed.all_gather(
            w2g_tp, dim=3, cluster_axis=tp_axis
        )
        w2g_tp_np = np.asarray(w2g_tp_full.to_numpy(composer=composer)[:1])

        thr = 0.99
        assert self._pcc(out_tp_np, out_ref_np) > thr
        assert self._pcc(xg_tp, xg_ref) > thr
        assert self._pcc(w1g_tp_np, w1g_ref) > thr
        assert self._pcc(w2g_tp_np, w2g_ref) > thr

    def test_linear_regression_training_step_backward(self):
        """Single linear layer: forward, mean loss, backward — gradients exist and shapes match."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.training import distribute_tensor
        from ttml.modules import LinearLayer

        init_ops()
        mesh = ttml.autograd.AutoContext.get_instance().get_device()
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(mesh, 0)
        linear = LinearLayer(64, 64, has_bias=False)
        linear.weight.tensor = distribute_tensor(
            linear.weight.tensor, mesh, _layout_col_parallel(1), requires_grad=True
        )
        np_x = self._nz(1, 1, 16, 64)
        x = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_x.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh,
            replicated_layout(2),
            requires_grad=True,
        )
        y = ttml.ops.linear.linear(x, linear.weight.tensor)
        loss = ttml.ops.unary.mean(y)
        loss.backward(False)
        xg = x.get_grad_tensor()
        wg = linear.weight.tensor.get_grad_tensor()
        assert xg is not None and wg is not None
        xg_np = np.asarray(xg.to_numpy(composer=composer))
        wg_np = np.asarray(wg.to_numpy(composer=composer))
        assert xg_np.shape[-1] == 64
        assert wg_np.shape[2] == 64 // 4

    def test_parallelize_module_colwise_weight_layout(self):
        import ttml
        from ttml.distributed import init_ops, parallelize_module, ColwiseParallel
        from ttml.modules import LinearLayer

        init_ops()
        mesh = ttml.autograd.AutoContext.get_instance().get_device()
        linear = LinearLayer(64, 128, has_bias=False)
        parallelize_module(linear, mesh, {"": ColwiseParallel()}, tp_axis=1)
        assert get_layout(linear.weight.tensor) == _layout_col_parallel(1)

    def test_custom_op_register_rule_dispatch_layout(self):
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.distributed.dispatch import dispatch, _get_raw, _RAW_OPS
        from ttml.distributed.training import distribute_tensor

        init_ops()
        mesh = ttml.autograd.AutoContext.get_instance().get_device()
        name = "__test_dispatch_custom__"

        def raw_pass(x):
            return _get_raw("silu")(x)

        _RAW_OPS[name] = raw_pass

        @register_rule(name)
        def _rule(layout: Layout, *e, runtime=None, **kw):
            return ShardingPlan(input_layouts=[layout], output_layout=layout)

        layout = _layout_rep_shard(1, -1)
        np_data = self._nz(1, 1, 8, 32)
        x = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_data.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh,
            layout,
        )
        y = dispatch(name, x)
        assert get_layout(y) == layout
        del _OP_RULES[name]
        del _RAW_OPS[name]

    def test_custom_module_parallelize_and_forward_layout(self):
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import (
            init_ops,
            parallelize_module,
            ColwiseParallel,
            RowwiseParallel,
        )
        from ttml.distributed.training import distribute_tensor
        from ttml.modules import AbstractModuleBase, LinearLayer

        init_ops()
        mesh = ttml.autograd.AutoContext.get_instance().get_device()
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(mesh, 0)

        class SmallMLP(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                self.fc1 = LinearLayer(64, 64, has_bias=False)
                self.fc2 = LinearLayer(64, 64, has_bias=False)

            def forward(self, x):
                h = ttml.ops.unary.silu(self.fc1(x))
                return self.fc2(h)

        m = SmallMLP()
        # Full logical weights before TP sharding (composer [:1] is one replica, not a TP shard).
        w1_full = np.asarray(
            ttml.autograd.to_numpy(m.fc1.weight.tensor, composer=composer)[:1]
        )
        w2_full = np.asarray(
            ttml.autograd.to_numpy(m.fc2.weight.tensor, composer=composer)[:1]
        )

        parallelize_module(
            m,
            mesh,
            {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()},
            tp_axis=1,
        )
        assert get_layout(m.fc1.weight.tensor) == _layout_col_parallel(1)
        assert get_layout(m.fc2.weight.tensor) == _layout_row_parallel(1)

        np_x = self._nz(1, 1, 16, 64).astype(np.float32)
        x = distribute_tensor(
            ttml.autograd.Tensor.from_numpy(
                np_x.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ),
            mesh,
            replicated_layout(2),
        )
        out = m(x)
        assert get_layout(out).is_replicated()
        # Reference: same op order as forward (fc1 -> silu -> fc2) using full weights.
        h = np_x @ w1_full[0, 0].T
        h = h * (1.0 / (1.0 + np.exp(-h)))
        expected = (h @ w2_full[0, 0].T).astype(np.float32)
        out_np = np.asarray(ttml.autograd.to_numpy(out, composer=composer)[:1]).astype(
            np.float32
        )
        assert self._pcc(out_np, expected) > 0.99

    def test_custom_module_rule_invoked_by_parallelize_module(self):
        """@register_module_rule on a non-GQA composite: rule runs, then children get plan styles."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import (
            init_ops,
            parallelize_module,
            ColwiseParallel,
        )
        from ttml.distributed.rules.registry import register_module_rule, _MODULE_RULES
        from ttml.distributed.training import distribute_tensor
        from ttml.modules import AbstractModuleBase, LinearLayer

        class CustomPairBlock(AbstractModuleBase):
            """User-defined composite: two linears; rule can adjust per-instance state before TP."""

            def __init__(self):
                super().__init__()
                self.branch_a = LinearLayer(64, 64, has_bias=False)
                self.branch_b = LinearLayer(64, 64, has_bias=False)
                self.constant = 1

            def forward(self, x):
                ya = self.branch_a(x)
                yb = self.branch_b(x)
                return ttml.ops.binary.add(ya, yb)

        @register_module_rule(CustomPairBlock)
        def _custom_pair_rule(module, mesh_device, tp_axis, cp_axis=None):
            # Must use ``module``, not ``self`` — this is a plain function, not a bound method.
            module.constant = tp_axis + 4
            return module

        init_ops()
        mesh = ttml.autograd.AutoContext.get_instance().get_device()
        m = CustomPairBlock()

        tp_axis = 1
        parallelize_module(
            m,
            mesh,
            {
                "branch_a": ColwiseParallel(),
                "branch_b": ColwiseParallel(),
            },
            tp_axis=tp_axis,
            cp_axis=None,
        )
        assert m.constant == tp_axis + 4
        assert get_layout(m.branch_a.weight.tensor) == _layout_col_parallel(tp_axis)
        assert get_layout(m.branch_b.weight.tensor) == _layout_col_parallel(tp_axis)
        other = CustomPairBlock()
        assert other.constant == 1
        del _MODULE_RULES[CustomPairBlock]
