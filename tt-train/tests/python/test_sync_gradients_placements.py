# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gradient synchronisation trusts TensorTopology placements: this module pins what it reads.

``ttml.core.distributed.synchronize_gradients`` decides per (parameter, mesh axis) whether to all-reduce the gradient
along that axis by reading the *parameter's* TensorTopology through ``is_sharded_on_axis``: a parameter sharded on an
axis (TP, FSDP) already holds a per-shard-correct gradient there, a replicated one must be averaged. That reader used
to index ``placements[axis]`` directly, which is only meaningful for an N-D label (one placement per mesh axis); the
collapsed 1-D ``{N}, [placement]`` of the default mappers -- and of every tensor created without a mapper -- read as
"not sharded" on every axis but 0. After the reduction a relabel by index (force_replicate_axes) then rewrote the
result's label, over-claiming Replicate for a collapsed Shard. Now every tensor created without a mapper carries one
Replicate per mesh axis, the reader expands a collapsed label per axis (a whole-mesh Shard is Shard on every axis of
size > 1) and refuses one it cannot read, and the all_reduce result carries its own per-axis label.

For tiny TP / DDP / FSDP / HSDP models after one backward this module asserts:

  * every parameter label has one placement per mesh axis (tensors created without a mapper included);
  * ``is_sharded_on_axis`` per (parameter, axis), hence the set of axes each gradient is reduced along;
  * after ``synchronize_gradients`` the gradient reads Replicate on the reduced axis, keeps its Shard on the other
    axis, and holds the mean of the per-device gradients read before the reduction;
  * negative controls: a collapsed Shard label reads as Shard on the non-trivial axis (it read as Replicate before
    this change, the one placement sat at index 0), and a label that does not cover the mesh raises instead of being
    guessed.

The ParallelismContext-driven overload of ``synchronize_gradients`` cannot be exercised for several layouts in one
process (the context is a one-shot singleton), so its axis filter is checked through the reader it uses and the
collectives through the explicit-axes overload.

Meshes: the shared ``tp_mesh`` / ``fsdp_mesh`` ``[1, 2]`` fixtures (conftest), a local ``[1, 2]`` mesh with the
data-parallel axis on axis 1 so a DDP reduction really runs over two devices, and ``[2, 2]`` meshes for DP x TP and
HSDP that open-or-skip like test_ccl_topology.py (4 devices and a bundled MGD needed). Every fixture is module-scoped
and reopens the device, so each mesh's tests live in their own class, in fixture order: TestTP, TestFSDP, TestDDP,
TestDPxTP, TestHSDP.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pytest

import ttnn
import ttml
from ttml.modules import AbstractModuleBase, ColumnParallelLinear, LinearLayer, RowParallelLinear

pytestmark = [pytest.mark.requires_device, pytest.mark.timeout(1800)]

DIM = 64  # tile-aligned per device after a 2-way split
NATIVE = ttml.autograd.PreferredPrecision.NATIVE

is_sharded_on_axis = ttml.core.distributed.is_sharded_on_axis


# --- local meshes ----------------------------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MGD_FOR_ARCH_AND_SHAPE = {
    ("blackhole", (1, 2)): os.path.join(_REPO_ROOT, "configs", "mgd", "bh_galaxy_1_2_line_line.textproto"),
    ("wormhole_b0", (1, 2)): os.path.join(_REPO_ROOT, "configs", "mgd", "n300_1_2_line_line.textproto"),
    ("blackhole", (2, 2)): os.path.join(_REPO_ROOT, "configs", "mgd", "bh_galaxy_2_2_line_line.textproto"),
}


def _detect_arch() -> Optional[str]:
    try:
        name = ttnn.get_arch_name().lower()
    except Exception:  # noqa: BLE001
        return None
    if "blackhole" in name:
        return "blackhole"
    if "wormhole_b0" in name:
        return "wormhole_b0"
    return None


def _num_available_devices() -> int:
    try:
        return int(ttnn.distributed.get_num_devices())
    except Exception:  # noqa: BLE001
        return 0


def _close_device_mesh_quietly() -> None:
    try:
        ttml.close_device_mesh()
    except Exception:  # noqa: BLE001
        pass


def _restore_mgd_path(previous: Optional[str]) -> None:
    if previous is None:
        os.environ.pop("TT_MESH_GRAPH_DESC_PATH", None)
    else:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = previous


def _open_named_mesh_or_skip(shape: tuple[int, ...], axis_names: tuple[str, ...]):
    """Open ``shape`` with ``axis_names`` for a module-scoped fixture, or skip its tests; returns the previous MGD.

    A caller-set ``TT_MESH_GRAPH_DESC_PATH`` wins for a 2-device shape (the conftest fixtures' rule: the bench box
    runs its own descriptor). For the 4-device shape the bundled descriptor is installed when one exists, as
    test_ccl_topology.py does for its 2x2 mesh, since a path set for a line cannot describe it."""
    needed = int(np.prod(shape))
    available = _num_available_devices()
    if available and available < needed:
        pytest.skip(f"needs a {shape} mesh ({needed} chips); host has {available}.")
    previous = os.environ.get("TT_MESH_GRAPH_DESC_PATH")
    arch = _detect_arch()
    candidate = _MGD_FOR_ARCH_AND_SHAPE.get((arch, shape)) if arch is not None else None
    if candidate and os.path.isfile(candidate) and (needed > 2 or not previous):
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = candidate
    elif not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        pytest.skip(f"needs a bundled MGD for arch={arch!r} shape={shape}, or TT_MESH_GRAPH_DESC_PATH.")
    _close_device_mesh_quietly()
    try:
        ttml.open_device_mesh(ttml.Mesh(shape, axis_names))
    except Exception as e:  # noqa: BLE001
        _restore_mgd_path(previous)
        pytest.skip(f"needs a {shape} {axis_names} mesh: {e}")
    return previous


@pytest.fixture(scope="module")
def ddp_mesh():
    """``[1, 2]`` with axes ``("tp", "dp")``: TP of size 1, the data-parallel axis on the two devices."""
    previous = _open_named_mesh_or_skip((1, 2), ("tp", "dp"))
    yield ttml.mesh()
    _close_device_mesh_quietly()
    _restore_mgd_path(previous)


@pytest.fixture(scope="module")
def dp_tp_mesh():
    """``[2, 2]`` with axes ``("dp", "tp")``."""
    previous = _open_named_mesh_or_skip((2, 2), ("dp", "tp"))
    yield ttml.mesh()
    _close_device_mesh_quietly()
    _restore_mgd_path(previous)


@pytest.fixture(scope="module")
def hsdp_mesh():
    """``[2, 2]`` with axes ``("dp", "fsdp")``."""
    previous = _open_named_mesh_or_skip((2, 2), ("dp", "fsdp"))
    yield ttml.mesh()
    _close_device_mesh_quietly()
    _restore_mgd_path(previous)


# --- models and helpers ----------------------------------------------------------------------------------------


class TPBlock(AbstractModuleBase):
    """Replicated linear -> column-parallel (weight Shard(2), bias Shard(3) on tp) -> row-parallel (weight Shard(3),
    bias replicated)."""

    def __init__(self) -> None:
        super().__init__()
        self.inp = LinearLayer(DIM, DIM)
        self.up = ColumnParallelLinear(DIM, DIM, has_bias=True)
        self.down = RowParallelLinear(DIM, DIM, has_bias=True, input_is_parallel=True)

    def forward(self, x):
        return self.down(self.up(self.inp(x)))


TP_SHARDED_SUFFIXES = ("up/weight", "up/bias", "down/weight")


class Block(AbstractModuleBase):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = LinearLayer(DIM, DIM)
        self.fc2 = LinearLayer(DIM, DIM)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class Model(AbstractModuleBase):
    """``block`` is the FSDP target where FSDP is used; ``head`` is never FSDP-managed."""

    def __init__(self) -> None:
        super().__init__()
        self.block = Block()
        self.head = LinearLayer(DIM, DIM, has_bias=False)

    def forward(self, x):
        return self.head(self.block(x))


def _build_fsdp_model(init: str) -> Model:
    if init == "eager":
        model = Model()
        ttml.fsdp.fully_shard(model.block)
        return model
    with ttml.lazy_init():
        model = Model()
    ttml.fsdp.fully_shard(model.block)
    ttml.materialize_module(model)
    return model


def _device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def _input(shape, mapper=None, *, seed: int = 0, scale: float = 0.1):
    data = np.random.default_rng(seed).standard_normal(shape, dtype=np.float32) * scale
    return ttml.autograd.Tensor.from_numpy(data, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)


def _backward(model, x) -> None:
    ctx = ttml.autograd.AutoContext.get_instance()
    loss = ttml.ops.unary.mean(model(x))
    loss.backward(False)
    ctx.reset_graph()


def _placements(value) -> list:
    return list(value.tensor_topology().placements())


def _shards(value) -> list:
    """Per-device arrays in mesh row-major order (every label built here lists its coordinates in that order)."""
    return [ttnn.to_torch(shard).float().numpy() for shard in ttnn.get_device_tensors(value)]


def _reduce_set(value, data_axes) -> set:
    """The axes synchronize_gradients reduces a gradient along: the data axes the parameter is not sharded on."""
    return {axis for axis in data_axes if not is_sharded_on_axis(value, axis)}


def _assert_mean_over_axis(before: list, after: list, mesh_shape, axis: int) -> None:
    """``after`` holds, on every device, the mean of ``before`` over mesh axis ``axis`` (bf16 sum then exact scale)."""
    shape = tuple(mesh_shape)
    b = np.stack(before).reshape(shape + before[0].shape)
    a = np.stack(after).reshape(shape + after[0].shape)
    expected = np.broadcast_to(b.mean(axis=axis, keepdims=True), b.shape)
    np.testing.assert_allclose(a, expected, rtol=2e-2, atol=1e-4)


def _assert_per_axis_labels(params, n_axes: int) -> None:
    """Every parameter carries one placement per mesh axis: none of them is a collapsed 1-D label."""
    wrong = {name: _placements(t.get_value(NATIVE)) for name, t in params.items()}
    wrong = {name: p for name, p in wrong.items() if len(p) != n_axes}
    assert not wrong, f"parameters without one placement per mesh axis: {wrong}"


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
        and Replicate on the size-1 dp axis. Before this change axis 1 was out of range of the single placement and
        read as not sharded, so a TP-sharded parameter created this way would have been reduced along a DDP axis
        with its own shard. A collapsed label that does not cover the mesh is refused, not guessed."""
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
        FSDP axis has. Negative control: the reader used to look at ``placements[axis_index]`` first, so with the FSDP
        axis at index 0 the single placement "existed" and the diagnosis was "already sharded on mesh axis 'fsdp'",
        while at index 1 the label read as not sharded and one device's shard was redistributed as the full tensor."""
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
        with expect_error(RuntimeError, "collapsed 1-D topology label"):  # FSDP axis at index 0: refused, not misread
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
            _assert_mean_over_axis(before[name], _shards(grad), ddp_mesh.shape, dp)

    def test_collapsed_grad_label_is_read_reduced_and_relabelled(self, ddp_mesh):
        """Negative control. A gradient carrying the collapsed ``{2}, [Shard(0)]`` label (what a whole-mesh
        ShardTensorToMesh batch hands every downstream gradient) reads as Shard on the dp axis -- before this change
        it read as not sharded there, the one placement sat at index 0 -- and comes back from the reduction as
        Replicate on every axis with the averaged bytes, labelled by the collective itself (the relabel-by-index this
        replaces would have produced the same label here, but over-claimed Replicate for the same input on a 2-D
        mesh). On this line mesh the all_reduce keeps the collapsed spelling, so the result is ``{2}, [Replicate]``
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
            _assert_mean_over_axis(before[name], _shards(grad), ddp_mesh.shape, dp)


# --- DP x TP on [2, 2] ("dp", "tp") -----------------------------------------------------------------------------


class TestDPxTP:
    def test_dp_reduction_keeps_tp_shards(self, dp_tp_mesh):
        """Batch sharded along "dp", TP weights sharded along "tp": every parameter is reduced along "dp" only, the
        result averages over the dp axis, reads Replicate there and keeps Shard on the tp axis for TP parameters."""
        dp, tp = dp_tp_mesh.axis_index("dp"), dp_tp_mesh.axis_index("tp")
        model = TPBlock()
        params = model.parameters()
        _backward(model, _input((2, 1, DIM, DIM), dp_tp_mesh.axis_mapper("dp", tdim=0)))

        _assert_per_axis_labels(params, 2)
        before = {name: _shards(t.get_grad()) for name, t in params.items()}
        for name, t in params.items():
            value = t.get_value(NATIVE)
            assert is_sharded_on_axis(value, tp) == name.endswith(TP_SHARDED_SUFFIXES), name
            assert _reduce_set(value, (dp,)) == {dp}, name

        ttml.core.distributed.synchronize_gradients(params, [dp])

        for name, t in params.items():
            grad = t.get_grad()
            assert not is_sharded_on_axis(grad, dp), name
            if name.endswith(TP_SHARDED_SUFFIXES):
                assert is_sharded_on_axis(grad, tp), f"{name}: the TP shard did not survive the dp all_reduce"
            _assert_mean_over_axis(before[name], _shards(grad), dp_tp_mesh.shape, dp)


# --- HSDP on [2, 2] ("dp", "fsdp") ------------------------------------------------------------------------------


class TestHSDP:
    def _batch_mapper(self, mesh):
        """The HSDP batch mapper train.py builds: Shard(0) along "dp", Shard(1) along "fsdp"."""
        placements = [ttnn.PlacementReplicate() for _ in mesh.shape]
        placements[mesh.axis_index("dp")] = ttnn.PlacementShard(0)
        placements[mesh.axis_index("fsdp")] = ttnn.PlacementShard(1)
        return ttnn.create_mesh_mapper(_device(), ttnn.MeshMapperConfig(placements))

    def test_managed_params_are_reduced_on_dp_only(self, hsdp_mesh):
        """The batch is [D, F, DIM, DIM] sharded dim 0 along "dp" and dim 1 along "fsdp" (one sample per device,
        label [Shard(0), Shard(1)]). FSDP-managed parameters are Shard on "fsdp", so they are reduced on "dp" only;
        the unmanaged head on both. After the dp reduction a managed gradient reads Replicate on "dp" and Shard on
        "fsdp" (the parameter's own label), the head's still Shard on "fsdp" until it is reduced there too."""
        dp, fsdp = hsdp_mesh.axis_index("dp"), hsdp_mesh.axis_index("fsdp")
        D, F = hsdp_mesh.axis_size("dp"), hsdp_mesh.axis_size("fsdp")
        model = _build_fsdp_model("eager")
        params = model.parameters()
        x = _input((D, F, DIM, DIM), self._batch_mapper(hsdp_mesh))
        assert list(x.get_value(NATIVE).shape) == [1, 1, DIM, DIM]
        _backward(model, x)

        _assert_per_axis_labels(params, 2)
        before = {name: _shards(t.get_grad()) for name, t in params.items()}
        for name, t in params.items():
            value = t.get_value(NATIVE)
            managed = ttml.fsdp.is_fsdp_managed(t)
            assert is_sharded_on_axis(value, fsdp) == managed, name
            assert not is_sharded_on_axis(value, dp), name
            assert _reduce_set(value, (dp, fsdp)) == ({dp} if managed else {dp, fsdp}), name
            assert is_sharded_on_axis(t.get_grad(), dp), f"{name}: every device computed a different gradient"

        ttml.core.distributed.synchronize_gradients(params, [dp])

        for name, t in params.items():
            grad = t.get_grad()
            assert not is_sharded_on_axis(grad, dp), name
            assert is_sharded_on_axis(grad, fsdp), name
            _assert_mean_over_axis(before[name], _shards(grad), hsdp_mesh.shape, dp)

        head = params["Model/head/weight"]
        reduced = ttnn.multiply(ttml.core.distributed.all_reduce(head.get_grad(), cluster_axis=fsdp), 1.0 / F)
        assert not is_sharded_on_axis(reduced, dp)
        assert not is_sharded_on_axis(reduced, fsdp)
        _assert_mean_over_axis(_shards(head.get_grad()), _shards(reduced), hsdp_mesh.shape, fsdp)

    def test_causal_lm_collate_batch_split(self, hsdp_mesh):
        """``causal_lm_collate_fn(batch_split=D)`` with the HSDP mapper: the standard per-device shapes, a label that
        is Shard on both data axes, and device (d, f) holding examples d*B/D + f*B/(D*F) + j."""
        from ttml.datasets import causal_lm_collate_fn

        dp, fsdp = hsdp_mesh.axis_index("dp"), hsdp_mesh.axis_index("fsdp")
        D, F = hsdp_mesh.axis_size("dp"), hsdp_mesh.axis_size("fsdp")
        seq_len, batch = 32, 2 * D * F
        examples = [
            {
                "input_ids": list(range(100 * i, 100 * i + seq_len)),
                "labels": list(range(100 * i + 1, 100 * i + seq_len + 1)),
            }
            for i in range(batch)
        ]

        out = causal_lm_collate_fn(examples, seq_len=seq_len, mapper=self._batch_mapper(hsdp_mesh), batch_split=D)

        per_device = batch // (D * F)
        ids, labels, mask = (
            out.input_ids.get_value(NATIVE),
            out.labels.get_value(NATIVE),
            out.loss_mask.get_value(NATIVE),
        )
        assert list(ids.shape) == [per_device, 1, 1, seq_len]
        assert list(labels.shape) == [per_device, seq_len]
        assert list(mask.shape) == [per_device, 1, seq_len, 1]
        for tensor in (ids, labels, mask):
            assert is_sharded_on_axis(tensor, dp) and is_sharded_on_axis(tensor, fsdp)
            assert len(_placements(tensor)) == 2

        shards = _shards(ids)
        rows = batch // D
        for index in range(len(shards)):  # shards are in mesh row-major order, like the label's coordinates
            coord = np.unravel_index(index, tuple(hsdp_mesh.shape))
            d, f = int(coord[dp]), int(coord[fsdp])
            first = d * rows + f * per_device
            expected = np.stack(
                [np.array(examples[first + j]["input_ids"], dtype=np.float32) for j in range(per_device)]
            )
            np.testing.assert_array_equal(shards[index][:, 0, 0, :], expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
