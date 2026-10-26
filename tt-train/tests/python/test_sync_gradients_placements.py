# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gradient synchronisation trusts TensorTopology placements: this module pins what it reads.

``synchronize_gradients`` decides per (parameter, mesh axis) whether to all-reduce the gradient along that axis by
reading the parameter's label through ``is_sharded_on_axis``: sharded there (TP, FSDP) means the gradient is already
per-shard-correct, replicated means it must be averaged. The reader takes an N-D label as is, expands the default
mappers' collapsed 1-D label per axis, and refuses a label it cannot read.

For tiny TP / DDP / FSDP models after one backward, the tests assert that every parameter label has one
placement per mesh axis, which axes ``is_sharded_on_axis`` reports per parameter, and that after
``synchronize_gradients`` each gradient reads Replicate on the reduced axis, keeps its Shard on the other axis and
holds the mean of the per-device gradients. Negative controls: a collapsed Shard label reads as Shard on the
non-trivial axis, and a label that does not cover the mesh raises instead of being guessed.

The ParallelismContext overload of ``synchronize_gradients`` cannot be exercised for several layouts in one process
(the context is a one-shot singleton); it only picks the axes and runs the explicit-axes overload's loop, which the
tests call.

Meshes: the conftest's ``tp_mesh`` / ``fsdp_mesh`` ``[1, 2]`` fixtures and a local ``[1, 2]`` with the data-parallel
axis on axis 1. Every fixture is module-scoped and reopens the device, so each mesh has its own class, in fixture
order. DP x TP and HSDP on a ``[2, 2]`` are in test_sync_gradients_placements_2x2.py.
"""

from __future__ import annotations

import numpy as np
import pytest

import ttnn
import ttml
from ttml.modules import LinearLayer
from sync_gradients_placements_common import (
    DIM,
    NATIVE,
    TP_SHARDED_SUFFIXES,
    Model,
    TPBlock,
    _assert_mean_over_axis,
    _assert_per_axis_labels,
    _backward,
    _build_fsdp_model,
    _device,
    _input,
    _placements,
    _reduce_set,
    _shards,
    is_sharded_on_axis,
)

pytestmark = [pytest.mark.requires_device, pytest.mark.timeout(1800)]


# --- local mesh ------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ddp_mesh(fresh_device_mesh):
    """``[1, 2]`` with axes ``("tp", "dp")``: TP of size 1, the data-parallel axis on the two devices."""
    with fresh_device_mesh((1, 2), ("tp", "dp"), what="DDP placements tests") as mesh:
        yield mesh


# --- TP on [1, 2] ("dp", "tp") ----------------------------------------------------------------------------------


class TestTP:
    def test_tp_params_read_sharded_on_tp_only(self, tp_mesh):
        tp, dp = tp_mesh.axis_index("tp"), tp_mesh.axis_index("dp")
        model = TPBlock()
        params = model.parameters()
        x = _input((1, 1, DIM, DIM))
        assert len(_placements(x.get_value(NATIVE))) == 2, "a tensor created without a mapper is labelled per axis"
        _backward(model, x)

        _assert_per_axis_labels(params, 2)
        for name, t in params.items():
            value = t.get_value(NATIVE)
            tp_sharded = name.endswith(TP_SHARDED_SUFFIXES)
            assert is_sharded_on_axis(value, tp) == tp_sharded, name
            assert not is_sharded_on_axis(value, dp), name
            # DDP (were it enabled on "dp") reduces every parameter; nothing is ever reduced along "tp".
            assert _reduce_set(value, (dp,)) == {dp}, name
            assert _reduce_set(value, (dp, tp)) == ({dp} if tp_sharded else {dp, tp}), name
            assert t.is_grad_initialized(), name
            assert len(_placements(t.get_grad())) == 2, f"{name}: gradient label is not per axis"

    def test_collapsed_shard_label_reads_as_shard_on_the_tp_axis(self, tp_mesh, expect_error):
        """Negative control. ``{2}, [Shard(2)]`` from the whole-mesh 1-D mapper reads as Shard on the tp axis (axis 1)
        and Replicate on the size-1 dp axis, so a TP-sharded parameter created this way is never reduced along a DDP
        axis with its own shard. A collapsed label that does not cover the mesh is refused, not guessed."""
        tp, dp = tp_mesh.axis_index("tp"), tp_mesh.axis_index("dp")
        collapsed = _input(
            (1, 1, DIM, DIM), ttml.core.distributed.shard_tensor_to_mesh_mapper(_device(), 2, None)
        ).get_value(NATIVE)
        assert len(_placements(collapsed)) == 1, "precondition: collapsed 1-D label"
        assert is_sharded_on_axis(collapsed, tp)
        assert not is_sharded_on_axis(collapsed, dp)

        forged = _input((1, 1, DIM, DIM)).get_value(NATIVE)
        coords = list(forged.tensor_topology().mesh_coords())
        forged.update_tensor_topology(ttnn.TensorTopology(ttnn.MeshShape([1]), [ttnn.PlacementShard(0)], coords[:1]))
        with expect_error(RuntimeError, "cannot tell whether the tensor is sharded"):
            is_sharded_on_axis(forged, tp)

    def test_constant_inits_are_labelled_per_axis(self, tp_mesh):
        """Eager ``ttml.init.ones / zeros / constant`` (norm weights, zero-init biases) replicate with one Replicate
        per mesh axis, like every other tensor created without a mapper. Negative control: they moved a host tensor to
        the mesh, which carries the collapsed ``{2}, [Replicate]``."""
        for name, init, value in (
            ("ones", ttml.init.ones(), 1.0),
            ("zeros", ttml.init.zeros(), 0.0),
            ("constant", ttml.init.constant(0.5), 0.5),
        ):
            t = init((1, 1, 1, DIM)).get_value(NATIVE)
            placements = _placements(t)
            assert len(placements) == 2, f"{name}: label {t.tensor_topology()} is not per axis"
            assert all(isinstance(p, ttnn.PlacementReplicate) for p in placements), f"{name}: {placements}"
            for device, shard in enumerate(_shards(t)):
                np.testing.assert_array_equal(
                    shard, np.full((1, 1, 1, DIM), value, dtype=shard.dtype), err_msg=f"{name}: device {device}"
                )

    def test_synchronize_over_tp_leaves_the_tp_shards_alone(self, tp_mesh):
        """``synchronize_gradients(params, [tp])`` averages only the gradients of parameters replicated on tp. A
        TP-sharded parameter holds a different piece of the weight on each tp device, so averaging along tp would add
        those pieces together; its gradient comes back unchanged, bit for bit.

        Negative control: without the per-parameter filter, this call averaged each TP-sharded gradient with the
        other device's piece."""
        tp = tp_mesh.axis_index("tp")
        model = TPBlock()
        params = model.parameters()
        _backward(model, _input((1, 1, DIM, DIM)))
        before = {name: _shards(t.get_grad()) for name, t in params.items()}
        sharded = [name for name in params if name.endswith(TP_SHARDED_SUFFIXES)]
        assert sharded and any(
            not np.array_equal(*before[name]) for name in sharded
        ), "precondition: the TP shards differ, so averaging them would show"

        ttml.core.distributed.synchronize_gradients(params, [tp])

        for name, t in params.items():
            after = _shards(t.get_grad())
            if name in sharded:
                for device, (b, a) in enumerate(zip(before[name], after)):
                    np.testing.assert_array_equal(a, b, err_msg=f"{name}: device {device}'s piece was averaged on tp")
            else:
                _assert_mean_over_axis(before[name], after, tp_mesh.shape, tp, name)


# --- FSDP on [1, 2] ("dp", "fsdp") ------------------------------------------------------------------------------


class TestFSDP:
    @pytest.mark.parametrize("init", ["eager", "lazy"])
    def test_fsdp_params_skip_the_fsdp_axis(self, fsdp_mesh, init):
        fsdp, dp = fsdp_mesh.axis_index("fsdp"), fsdp_mesh.axis_index("dp")
        model = _build_fsdp_model(init)
        params = model.parameters()
        _backward(model, _input((1, 1, DIM, DIM)))

        _assert_per_axis_labels(params, 2)
        assert any(ttml.fsdp.is_fsdp_managed(t) for t in params.values())
        assert not all(ttml.fsdp.is_fsdp_managed(t) for t in params.values())
        for name, t in params.items():
            value = t.get_value(NATIVE)
            managed = ttml.fsdp.is_fsdp_managed(t)
            assert is_sharded_on_axis(value, fsdp) == managed, name
            assert not is_sharded_on_axis(value, dp), name
            # HSDP reduces a managed parameter on "dp" only (its backward hook reduce-scattered "fsdp"), the
            # unmanaged head on both.
            assert _reduce_set(value, (dp, fsdp)) == ({dp} if managed else {dp, fsdp}), name
            # The reduce-scattered gradient of a managed parameter is labelled Shard on "fsdp" by the collective.
            assert is_sharded_on_axis(t.get_grad(), fsdp) == managed, name

    def test_forged_shard_label_drops_the_axis_from_the_reduce_set(self, fsdp_mesh):
        """Negative control: the reader trusts the label. Forging Shard on the fsdp axis of the replicated head flips
        its reduce set from {dp, fsdp} to {dp}."""
        fsdp, dp = fsdp_mesh.axis_index("fsdp"), fsdp_mesh.axis_index("dp")
        model = Model()
        head = model.parameters()["Model/head/weight"].get_value(NATIVE)
        assert _reduce_set(head, (dp, fsdp)) == {dp, fsdp}
        placements = _placements(head)
        placements[fsdp] = ttnn.PlacementShard(2)
        topology = head.tensor_topology()
        head.update_tensor_topology(
            ttnn.TensorTopology(topology.distribution_shape(), placements, list(topology.mesh_coords()))
        )
        assert _reduce_set(head, (dp, fsdp)) == {dp}

    def test_fully_shard_refuses_a_collapsed_shard_label(self, fsdp_mesh, expect_error):
        """A parameter carrying the collapsed ``{2}, [Shard(2)]`` label of the whole-mesh 1-D mapper is sharded over
        every mesh axis at once, so ``fully_shard`` refuses it with the collapsed-label error, whatever index the
        FSDP axis has: the placement count is checked before any index is read."""
        from ttml import fsdp as fsdp_module

        linear = LinearLayer(DIM, DIM, has_bias=False)
        weight = linear.weight.tensor.get_value(NATIVE)
        coords = list(weight.tensor_topology().mesh_coords())
        weight.update_tensor_topology(
            ttnn.TensorTopology(ttnn.MeshShape([len(coords)]), [ttnn.PlacementShard(2)], coords)
        )
        with expect_error(RuntimeError, "collapsed 1-D topology label"):
            ttml.fsdp.fully_shard(linear)

        n_axes = len(fsdp_mesh.shape)
        with expect_error(RuntimeError, "collapsed 1-D topology label"):  # FSDP axis at index 0
            fsdp_module._already_fsdp_sharded([ttnn.PlacementShard(2)], 0, n_axes)
        assert fsdp_module._already_fsdp_sharded([ttnn.PlacementReplicate()], 0, n_axes) is False
        assert fsdp_module._already_fsdp_sharded([ttnn.PlacementReplicate(), ttnn.PlacementShard(2)], 1, n_axes) is True
        assert (
            fsdp_module._already_fsdp_sharded([ttnn.PlacementReplicate(), ttnn.PlacementShard(2)], 0, n_axes) is False
        )


# --- DDP on [1, 2] ("tp", "dp") ---------------------------------------------------------------------------------


class TestDDP:
    def test_grads_are_averaged_and_read_replicate_on_the_dp_axis(self, ddp_mesh):
        """Batch of two sharded along "dp": every device computes a different gradient. After the reduction every
        gradient holds the mean of the two and reads Replicate on both axes, with one placement per axis."""
        dp = ddp_mesh.axis_index("dp")
        model = Model()
        params = model.parameters()
        x = _input((2, 1, DIM, DIM), ddp_mesh.axis_mapper("dp", tdim=0))
        _backward(model, x)

        before = {name: _shards(t.get_grad()) for name, t in params.items()}
        for name, shards in before.items():
            if name.endswith("weight"):
                assert not np.array_equal(shards[0], shards[1]), f"{name}: precondition, the two devices agree"
        for name, t in params.items():
            assert _reduce_set(t.get_value(NATIVE), (dp,)) == {dp}, name

        ttml.core.distributed.synchronize_gradients(params, [dp])

        for name, t in params.items():
            grad = t.get_grad()
            assert len(_placements(grad)) == 2, name
            assert not is_sharded_on_axis(grad, dp), name
            assert not is_sharded_on_axis(grad, 1 - dp), name
            _assert_mean_over_axis(before[name], _shards(grad), ddp_mesh.shape, dp, name)

    def test_collapsed_grad_label_is_read_reduced_and_relabelled(self, ddp_mesh):
        """Negative control. A gradient carrying the collapsed ``{2}, [Shard(0)]`` label (what a whole-mesh
        ShardTensorToMesh batch hands every downstream gradient) reads as Shard on the dp axis and comes back from the
        reduction as Replicate on every axis with the averaged bytes, labelled by the collective itself. On this line
        mesh the all_reduce keeps the collapsed spelling, so the result is ``{2}, [Replicate]``
        with ONE placement (on a 2-D mesh it would be the N-D ``[Replicate, ...]``); hence no ``len == 2`` assertion
        here, by design."""
        dp = ddp_mesh.axis_index("dp")
        model = Model()
        params = model.parameters()
        _backward(model, _input((2, 1, DIM, DIM), ddp_mesh.axis_mapper("dp", tdim=0)))

        before = {}
        for name, t in params.items():
            grad = t.get_grad()
            before[name] = _shards(grad)
            coords = list(grad.tensor_topology().mesh_coords())
            grad.update_tensor_topology(
                ttnn.TensorTopology(ttnn.MeshShape([len(coords)]), [ttnn.PlacementShard(0)], coords)
            )
            assert len(_placements(t.get_grad())) == 1, "precondition: the gradient carries the collapsed label"
            assert is_sharded_on_axis(t.get_grad(), dp), name
            assert not is_sharded_on_axis(t.get_grad(), 1 - dp), name

        ttml.core.distributed.synchronize_gradients(params, [dp])

        for name, t in params.items():
            grad = t.get_grad()
            assert not is_sharded_on_axis(grad, dp), name
            assert not is_sharded_on_axis(grad, 1 - dp), name
            assert all(not isinstance(p, ttnn.PlacementShard) for p in _placements(grad)), name
            _assert_mean_over_axis(before[name], _shards(grad), ddp_mesh.shape, dp, name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
