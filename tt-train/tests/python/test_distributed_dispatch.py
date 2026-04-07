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
    DistributedLayout,
    Shard,
    Replicate,
    replicated_layout,
    get_layout,
    layout_from_topology,
    layout_to_mapper_config,
)
from ttml.distributed.utils import is_distributed
from ttml.distributed.rules.registry import (
    ShardingPlan,
    register_rule,
    get_rule,
    _OP_RULES,
    AllReduce,
)
from ttml.distributed.debug import TraceEntry, dispatch_trace, DispatchTracer


def _layout_rep_shard(tp_axis: int = 1, dim: int = -1) -> DistributedLayout:
    return DistributedLayout(ndim=2, axis_placements={tp_axis: Shard(dim)})


def _layout_col_parallel(tp_axis: int = 1) -> DistributedLayout:
    return DistributedLayout(ndim=2, axis_placements={tp_axis: Shard(-2)})


def _layout_row_parallel(tp_axis: int = 1) -> DistributedLayout:
    return DistributedLayout(ndim=2, axis_placements={tp_axis: Shard(-1)})


# ---------------------------------------------------------------------------
# DistributedLayout
# ---------------------------------------------------------------------------


class TestDistributedLayout:
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
        b = DistributedLayout(ndim=2, axis_placements={1: Shard(-1)})
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
        layout = DistributedLayout(ndim=2, axis_placements={1: Shard(-1)})
        assert layout_to_mapper_config(layout) is not None


# ---------------------------------------------------------------------------
# Plan cache & linear rule
# ---------------------------------------------------------------------------


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
# Registry
# ---------------------------------------------------------------------------


class TestRuleRegistry:
    def test_register_and_get_rule(self):
        @register_rule("__test_tmp_rule__")
        def _rule(x, *a, runtime=None, **k):
            return ShardingPlan(input_layouts=[x], output_layout=x)

        assert get_rule("__test_tmp_rule__") is _rule
        del _OP_RULES["__test_tmp_rule__"]


class TestBiasDistributedLayoutInvariants:
    """Column-parallel bias is sharded on last dim; row-parallel bias stays replicated."""

    def test_column_vs_row_expected_layouts(self):
        col_bias = DistributedLayout(ndim=2, axis_placements={1: Shard(-1)})
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
        def _r(layout: DistributedLayout, *e, runtime=None, **kw):
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
# Device-backed integration (32-device mesh)
# ---------------------------------------------------------------------------


def _distribute_tensor(tensor, mesh, layout, requires_grad=None):
    """Test helper: distribute a tensor to mesh with layout."""
    import ttml
    import ttnn
    import ml_dtypes
    from ttml.distributed.layout import set_layout

    orig_requires_grad = tensor.get_requires_grad()
    final_requires_grad = requires_grad if requires_grad is not None else orig_requires_grad
    orig_dtype = tensor.dtype()

    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(mesh, 0)
    np_data = tensor.to_numpy(orig_dtype, composer)
    if np_data.shape[0] > 1:
        np_data = np_data[:1]

    mapper = layout.build_mapper(mesh, tensor_rank=len(np_data.shape))
    result = ttml.autograd.Tensor.from_numpy(np_data, ttnn.Layout.TILE, orig_dtype, mapper)
    result.set_requires_grad(final_requires_grad)
    set_layout(result, layout)
    return result


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
        auto_ctx.open_device(mesh_shape)

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

        mesh = ttml.autograd.AutoContext.get_instance().get_device()
        layout = _layout_rep_shard(1, -1)
        np_data = np.ones((1, 1, 4, 16), dtype=np.float32)
        t = ttml.autograd.Tensor.from_numpy(np_data.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE)
        out = _distribute_tensor(t, mesh, layout)
        assert get_layout(out) == layout

    def test_dispatch_trace_records_op(self):
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops

        init_ops()
        mesh = ttml.autograd.AutoContext.get_instance().get_device()
        x = _distribute_tensor(
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

        init_ops()
        mesh = ttml.autograd.AutoContext.get_instance().get_device()
        layout = _layout_rep_shard(1, -1)
        x = _distribute_tensor(
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

        init_ops()
        mesh = ttml.autograd.AutoContext.get_instance().get_device()
        np_x = self._nz(1, 1, 8, 64)
        np_w = self._nz(1, 1, 128, 64)
        x = _distribute_tensor(
            ttml.autograd.Tensor.from_numpy(np_x.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE),
            mesh,
            replicated_layout(2),
        )
        w_col = _distribute_tensor(
            ttml.autograd.Tensor.from_numpy(np_w.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE),
            mesh,
            _layout_col_parallel(1),
        )
        y_col = ttml.ops.linear.linear(x, w_col, None)
        assert get_layout(y_col).is_sharded_on(1)
        assert get_layout(y_col).shard_dim(1) == -1

        np_x2 = self._nz(1, 1, 8, 128)
        np_w2 = self._nz(1, 1, 64, 128)
        x2 = _distribute_tensor(
            ttml.autograd.Tensor.from_numpy(np_x2.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE),
            mesh,
            _layout_rep_shard(1, -1),
        )
        w_row = _distribute_tensor(
            ttml.autograd.Tensor.from_numpy(np_w2.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE),
            mesh,
            _layout_row_parallel(1),
        )
        y_row = ttml.ops.linear.linear(x2, w_row, None)
        assert get_layout(y_row).is_replicated()

    def test_linear_regression_training_step_backward(self):
        """Single linear layer: forward, mean loss, backward — gradients exist and shapes match."""
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops
        from ttml.modules import LinearLayer

        init_ops()
        mesh = ttml.autograd.AutoContext.get_instance().get_device()
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(mesh, 0)
        np_w = self._nz(1, 1, 64, 64)
        w = _distribute_tensor(
            ttml.autograd.Tensor.from_numpy(np_w.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE),
            mesh,
            _layout_col_parallel(1),
            requires_grad=True,
        )
        np_x = self._nz(1, 1, 16, 64)
        x = _distribute_tensor(
            ttml.autograd.Tensor.from_numpy(np_x.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE),
            mesh,
            replicated_layout(2),
            requires_grad=True,
        )
        y = ttml.ops.linear.linear(x, w)
        loss = ttml.ops.unary.mean(y)
        loss.backward(False)
        xg = x.get_grad_tensor()
        wg = w.get_grad_tensor()
        assert xg is not None and wg is not None
        xg_np = np.asarray(xg.to_numpy(composer=composer))
        wg_np = np.asarray(wg.to_numpy(composer=composer))
        assert xg_np.shape[-1] == 64
        assert wg_np.shape[2] == 64 // 4

    def test_custom_op_register_rule_dispatch_layout(self):
        import ttml
        import ttnn
        import ml_dtypes
        from ttml.distributed import init_ops, register_op
        from ttml.distributed.dispatch import _get_raw, _RAW_OPS

        init_ops()
        mesh = ttml.autograd.AutoContext.get_instance().get_device()
        name = "__test_dispatch_custom__"

        @register_op(name)
        def custom_pass(x):
            return _get_raw("silu")(x)

        @register_rule(name)
        def _rule(layout: DistributedLayout, *e, runtime=None, **kw):
            return ShardingPlan(input_layouts=[layout], output_layout=layout)

        layout = _layout_rep_shard(1, -1)
        np_data = self._nz(1, 1, 8, 32)
        x = _distribute_tensor(
            ttml.autograd.Tensor.from_numpy(np_data.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE),
            mesh,
            layout,
        )
        y = custom_pass(x)
        assert get_layout(y) == layout
        del _OP_RULES[name]
        del _RAW_OPS[name]
