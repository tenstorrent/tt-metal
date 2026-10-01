# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests that CCL collective ops produce correct ``tensor_topology()`` placements.

Each test creates a tensor with known placements (sharded or replicated),
calls a collective op via ``ttml.core.distributed``, and asserts the
output tensor's placement matches the op's semantics:

  * **all_reduce** — output is replicated on the collective axis.
  * **reduce_scatter** — output is sharded on the collective axis.
  * **all_gather** — output is replicated on the collective axis.

In all cases, placements on mesh axes *other* than ``cluster_axis`` must
be preserved unchanged.

These tests exercise the ``ttml.core.distributed`` wrappers (non-autograd)
which call the ttnn experimental async CCL ops.  They exist to catch ops
that silently drop or fail to update ``TensorTopology`` placements on
their output tensors.

The fixture opens a 2x2 line-line mesh (4 devices).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pytest

import ttnn
import ttml

pytestmark = pytest.mark.requires_device

# Default mesh for the main tests: 2x2 with both axes addressable as the
# collective axis. Both axes are size 2 so any axis can be used as the
# cluster axis and a non-collective axis simultaneously.
MESH_SHAPE_2X2 = (2, 2)
AXIS_SIZE = 2
CLUSTER_AXIS = 1  # axis used as the collective axis in the main suite
OTHER_AXIS = 0  # axis we will sometimes shard on to verify it stays put

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MGD_FOR_ARCH_AND_SHAPE = {
    ("blackhole", MESH_SHAPE_2X2): os.path.join(_REPO_ROOT, "configs", "mgd", "bh_galaxy_2_2_line_line.textproto"),
    # Lines and the full galaxy, so _open_mesh_or_skip can open them for suites that parametrise the mesh.
    ("blackhole", (1, 8)): os.path.join(_REPO_ROOT, "configs", "mgd", "bh_galaxy_1_8_line_line.textproto"),
    ("blackhole", (8, 4)): os.path.join(_REPO_ROOT, "configs", "mgd", "bh_galaxy_8_4_line_line.textproto"),
}


def _detect_arch() -> Optional[str]:
    """Return ``"blackhole"`` or ``"wormhole_b0"`` for the host, or ``None``.

    Uses ``ttnn.get_arch_name()`` which reads the cluster yaml at
    process start and does not require any device to be open.
    """
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
    """Total devices visible to the cluster, or 0 if we can't tell."""
    try:
        return int(ttnn.distributed.get_num_devices())
    except Exception:  # noqa: BLE001
        return 0


# ---------------------------------------------------------------------------
# Multi-device mesh fixtures
# ---------------------------------------------------------------------------


def _close_device_mesh_quietly() -> None:
    """Reverse ``open_device_mesh`` (close device, disable fabric, clear the global mesh),
    swallowing errors so it is safe on the pre-open and teardown paths."""
    try:
        ttml.close_device_mesh()
    except Exception:  # noqa: BLE001
        pass


def _ensure_mgd_path(shape: tuple[int, ...]) -> Optional[str]:
    """Set ``TT_MESH_GRAPH_DESC_PATH`` for ``shape`` and return the previous value.

    Returns the previous env var so a caller can restore it on teardown.
    Always overwrites the env var when an MGD file is bundled for the
    host arch + requested shape, so that switching meshes inside a single
    test session works correctly. Does not overwrite when no bundled MGD
    matches — that lets a user-supplied ``TT_MESH_GRAPH_DESC_PATH`` win.
    """
    previous = os.environ.get("TT_MESH_GRAPH_DESC_PATH")
    arch = _detect_arch()
    if arch is None:
        return previous
    candidate = _MGD_FOR_ARCH_AND_SHAPE.get((arch, shape))
    if candidate and os.path.isfile(candidate):
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = candidate
    return previous


def _restore_mgd_path(previous: Optional[str]) -> None:
    if previous is None:
        os.environ.pop("TT_MESH_GRAPH_DESC_PATH", None)
    else:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = previous


def _skip_if_unsupported(shape: tuple[int, ...]) -> None:
    """Skip the test up-front when the host can't run a ``shape`` mesh.

    Two conditions trip a skip:
      * Cluster has fewer than ``prod(shape)`` chips — e.g. N300 (2)
        can't host a 2x2 (4) mesh.
      * Host arch has no bundled MGD entry for ``shape`` and no
        ``TT_MESH_GRAPH_DESC_PATH`` was supplied — we'd just open and
        crash inside the fabric layer otherwise.

    Skipping here, before any device or fabric state has been touched,
    keeps the rest of the test session clean (no leaked fabric config,
    no half-open mesh).
    """
    needed = 1
    for d in shape:
        needed *= d
    available = _num_available_devices()
    if available and available < needed:
        pytest.skip(f"CCL topology tests need a {shape} mesh ({needed} chips); host has {available}.")

    arch = _detect_arch()
    if os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        return  # user override wins; assume they know what they're doing
    if arch is None:
        return  # unknown arch, let the open path try
    if (arch, shape) not in _MGD_FOR_ARCH_AND_SHAPE:
        pytest.skip(
            f"CCL topology tests need a bundled MGD for arch={arch!r} shape={shape}; "
            f"none available. Either add one under tt-train/configs/mgd/ and update "
            f"_MGD_FOR_ARCH_AND_SHAPE, or export TT_MESH_GRAPH_DESC_PATH yourself."
        )


def _open_mesh_or_skip(shape: tuple[int, ...]):
    """Open a fresh mesh of ``shape``, skipping the test if not possible.

    Returns the previous MGD path so a teardown can restore it.
    """
    _skip_if_unsupported(shape)
    previous_mgd = _ensure_mgd_path(shape)
    _close_device_mesh_quietly()
    try:
        ttml.open_device_mesh(shape)
    except Exception as e:  # noqa: BLE001
        _restore_mgd_path(previous_mgd)
        pytest.skip(f"CCL topology tests need a {shape} mesh: {e}")
    return previous_mgd


@pytest.fixture(scope="module")
def ccl_mesh():
    """Open the default 2x2 mesh used by the main test classes."""
    previous_mgd = _open_mesh_or_skip(MESH_SHAPE_2X2)
    yield ttml.mesh()
    _close_device_mesh_quietly()
    _restore_mgd_path(previous_mgd)


# ---------------------------------------------------------------------------
# Tensor factories
# ---------------------------------------------------------------------------


def _make_tensor(shape: tuple[int, ...], placements, *, seed: int = 0):
    """Create a raw ttnn tensor with explicit per-mesh-axis ``placements``.

    Caller is responsible for ensuring the per-axis sharded dims divide
    evenly across the corresponding mesh axis sizes.
    """
    device = ttml.autograd.AutoContext.get_instance().get_device()
    mapper = ttnn.create_mesh_mapper(device, ttnn.MeshMapperConfig(list(placements)))
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(shape).astype(np.float32) * 0.1
    return ttml.autograd.Tensor.from_numpy(data, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper).get_value()


def _replicated_tensor(shape: tuple[int, ...], num_axes: int = 2, *, seed: int = 0):
    placements = [ttnn.PlacementReplicate() for _ in range(num_axes)]
    return _make_tensor(shape, placements, seed=seed)


def _sharded_tensor(
    shape: tuple[int, ...],
    shard_dim: int,
    *,
    cluster_axis: int = CLUSTER_AXIS,
    num_axes: int = 2,
    seed: int = 0,
):
    """Shard along ``shard_dim`` on ``cluster_axis``, replicate other axes."""
    placements = [ttnn.PlacementReplicate() for _ in range(num_axes)]
    placements[cluster_axis] = ttnn.PlacementShard(shard_dim)
    return _make_tensor(shape, placements, seed=seed)


# ---------------------------------------------------------------------------
# Placement assertion helpers
# ---------------------------------------------------------------------------


def _placement_repr(p) -> str:
    if isinstance(p, ttnn.PlacementShard):
        return f"Shard(dim={p.dim})"
    if isinstance(p, ttnn.PlacementReplicate):
        return "Replicate()"
    return repr(p)


def _assert_replicated(tensor, axis: int, msg: str = ""):
    placements = tensor.tensor_topology().placements()
    assert isinstance(placements[axis], ttnn.PlacementReplicate), (
        f"Expected PlacementReplicate on axis {axis}, " f"got {_placement_repr(placements[axis])}. {msg}"
    )


def _assert_shard(tensor, axis: int, expected_dim: int, msg: str = ""):
    placements = tensor.tensor_topology().placements()
    assert isinstance(placements[axis], ttnn.PlacementShard), (
        f"Expected PlacementShard on axis {axis}, " f"got {_placement_repr(placements[axis])}. {msg}"
    )
    assert placements[axis].dim == expected_dim, (
        f"Expected Shard(dim={expected_dim}) on axis {axis}, " f"got Shard(dim={placements[axis].dim}). {msg}"
    )


# ---------------------------------------------------------------------------
# all_reduce
# ---------------------------------------------------------------------------


class TestAllReduceTopology:
    """all_reduce output should be replicated on the collective axis and
    must leave placements on every other mesh axis unchanged."""

    @pytest.fixture(autouse=True)
    def _setup(self, ccl_mesh):
        self.mesh = ccl_mesh

    def test_sharded_input_becomes_replicated(self):
        shard_dim = 2
        tensor = _sharded_tensor((1, 1, 64, 128), shard_dim, seed=0)
        _assert_shard(tensor, CLUSTER_AXIS, shard_dim, "input sanity check")

        result = ttml.core.distributed.all_reduce(tensor, cluster_axis=CLUSTER_AXIS)

        _assert_replicated(result, CLUSTER_AXIS, "all_reduce output")

    def test_replicated_input_stays_replicated(self):
        tensor = _replicated_tensor((1, 1, 64, 128), seed=1)
        _assert_replicated(tensor, CLUSTER_AXIS, "input sanity check")

        result = ttml.core.distributed.all_reduce(tensor, cluster_axis=CLUSTER_AXIS)

        _assert_replicated(result, CLUSTER_AXIS, "all_reduce output")

    def test_other_axis_sharding_preserved(self):
        """Sharding on OTHER_AXIS must survive an all_reduce on CLUSTER_AXIS."""
        other_axis_shard_dim = 2
        cluster_axis_shard_dim = 3
        placements = [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()]
        placements[OTHER_AXIS] = ttnn.PlacementShard(other_axis_shard_dim)
        placements[CLUSTER_AXIS] = ttnn.PlacementShard(cluster_axis_shard_dim)
        tensor = _make_tensor((1, 1, 64, 128), placements, seed=2)
        _assert_shard(tensor, OTHER_AXIS, other_axis_shard_dim, "input sanity check")
        _assert_shard(tensor, CLUSTER_AXIS, cluster_axis_shard_dim, "input sanity check")

        result = ttml.core.distributed.all_reduce(tensor, cluster_axis=CLUSTER_AXIS)

        _assert_shard(result, OTHER_AXIS, other_axis_shard_dim, "other axis preserved")
        _assert_replicated(result, CLUSTER_AXIS, "all_reduce output")

    def test_replicated_input_with_other_axis_sharded(self):
        """Input replicated on CLUSTER_AXIS, sharded on OTHER_AXIS: output
        stays replicated on CLUSTER_AXIS and keeps Shard on OTHER_AXIS."""
        other_axis_shard_dim = 2
        placements = [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()]
        placements[OTHER_AXIS] = ttnn.PlacementShard(other_axis_shard_dim)
        tensor = _make_tensor((1, 1, 64, 128), placements, seed=3)
        _assert_shard(tensor, OTHER_AXIS, other_axis_shard_dim, "input sanity check")
        _assert_replicated(tensor, CLUSTER_AXIS, "input sanity check")

        result = ttml.core.distributed.all_reduce(tensor, cluster_axis=CLUSTER_AXIS)

        _assert_shard(result, OTHER_AXIS, other_axis_shard_dim, "other axis preserved")
        _assert_replicated(result, CLUSTER_AXIS, "all_reduce output")


# ---------------------------------------------------------------------------
# reduce_scatter
# ---------------------------------------------------------------------------


class TestReduceScatterTopology:
    """reduce_scatter output should be sharded on the collective axis along
    the scatter dim, and leave other mesh axes' placements unchanged."""

    @pytest.fixture(autouse=True)
    def _setup(self, ccl_mesh):
        self.mesh = ccl_mesh

    def test_replicated_input_becomes_sharded(self):
        scatter_dim = 3
        tensor = _replicated_tensor((1, 1, 64, 128), seed=10)
        _assert_replicated(tensor, CLUSTER_AXIS, "input sanity check")

        result = ttml.core.distributed.reduce_scatter(tensor, scatter_dim, cluster_axis=CLUSTER_AXIS)

        _assert_shard(result, CLUSTER_AXIS, scatter_dim, "reduce_scatter output")

    def test_sharded_input_stays_sharded(self):
        shard_dim = 3
        tensor = _sharded_tensor((1, 1, 64, 128), shard_dim, seed=11)
        _assert_shard(tensor, CLUSTER_AXIS, shard_dim, "input sanity check")

        result = ttml.core.distributed.reduce_scatter(tensor, shard_dim, cluster_axis=CLUSTER_AXIS)

        _assert_shard(result, CLUSTER_AXIS, shard_dim, "reduce_scatter output")

    def test_other_axis_sharding_preserved(self):
        """OTHER_AXIS sharding must survive reduce_scatter on CLUSTER_AXIS."""
        other_axis_shard_dim = 2
        scatter_dim = 3
        placements = [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()]
        placements[OTHER_AXIS] = ttnn.PlacementShard(other_axis_shard_dim)
        tensor = _make_tensor((1, 1, 64, 128), placements, seed=12)
        _assert_shard(tensor, OTHER_AXIS, other_axis_shard_dim, "input sanity check")
        _assert_replicated(tensor, CLUSTER_AXIS, "input sanity check")

        result = ttml.core.distributed.reduce_scatter(tensor, scatter_dim, cluster_axis=CLUSTER_AXIS)

        _assert_shard(result, OTHER_AXIS, other_axis_shard_dim, "other axis preserved")
        _assert_shard(result, CLUSTER_AXIS, scatter_dim, "reduce_scatter output")

    def test_negative_scatter_dim(self):
        """A negative ``dim`` (e.g. -1) must produce a Shard with the
        normalized dim, not the literal negative integer."""
        scatter_dim_input = -1  # logical dim 3 for a 4D tensor
        normalized = 3
        tensor = _replicated_tensor((1, 1, 64, 128), seed=13)
        _assert_replicated(tensor, CLUSTER_AXIS, "input sanity check")

        result = ttml.core.distributed.reduce_scatter(tensor, scatter_dim_input, cluster_axis=CLUSTER_AXIS)

        _assert_shard(result, CLUSTER_AXIS, normalized, "negative scatter_dim")


# ---------------------------------------------------------------------------
# all_gather
# ---------------------------------------------------------------------------


class TestAllGatherTopology:
    """all_gather output should be replicated on the collective axis and
    leave placements on every other mesh axis unchanged."""

    @pytest.fixture(autouse=True)
    def _setup(self, ccl_mesh):
        self.mesh = ccl_mesh

    def test_sharded_input_becomes_replicated(self):
        shard_dim = 2
        tensor = _sharded_tensor((1, 1, 64, 128), shard_dim, seed=20)
        _assert_shard(tensor, CLUSTER_AXIS, shard_dim, "input sanity check")

        result = ttml.core.distributed.all_gather(tensor, shard_dim, cluster_axis=CLUSTER_AXIS)

        _assert_replicated(result, CLUSTER_AXIS, "all_gather output")

    def test_other_axis_sharding_preserved(self):
        """OTHER_AXIS Shard must survive an all_gather on CLUSTER_AXIS."""
        other_axis_shard_dim = 2
        gather_dim = 3
        placements = [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()]
        placements[OTHER_AXIS] = ttnn.PlacementShard(other_axis_shard_dim)
        placements[CLUSTER_AXIS] = ttnn.PlacementShard(gather_dim)
        tensor = _make_tensor((1, 1, 64, 128), placements, seed=21)
        _assert_shard(tensor, OTHER_AXIS, other_axis_shard_dim, "input sanity check")
        _assert_shard(tensor, CLUSTER_AXIS, gather_dim, "input sanity check")

        result = ttml.core.distributed.all_gather(tensor, gather_dim, cluster_axis=CLUSTER_AXIS)

        _assert_shard(result, OTHER_AXIS, other_axis_shard_dim, "other axis preserved")
        _assert_replicated(result, CLUSTER_AXIS, "all_gather output")


# ---------------------------------------------------------------------------------------------------------------------
# Collapsed (1-D) input labels
# ---------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def strict_ccl_topology():
    """Promote the CCL topology helper's refusals from warnings to errors for the test (what CI runs)."""
    previous = ttnn.CONFIG.strict_ccl_topology
    ttnn.CONFIG.strict_ccl_topology = True
    try:
        yield
    finally:
        ttnn.CONFIG.strict_ccl_topology = previous


def _collapsed_tensor(shape: tuple[int, ...], make_mapper, *, seed: int = 0):
    """Tensor distributed by a 1-D mapper (``ttnn.shard_tensor_to_mesh_mapper`` / ``ttnn.replicate_tensor_to_mesh_mapper``
    -- the raw ``TensorToMesh`` that ``Tensor.from_numpy`` takes, not the ``ttnn.ReplicateTensorToMesh`` Python
    wrapper), whose label is the collapsed ``{N}, [placement]`` over the mesh in row-major device order. Small
    integers, so every sum and concatenation below is exact in bfloat16. Returns the device tensor and the full numpy
    array."""
    device = ttml.autograd.AutoContext.get_instance().get_device()
    rng = np.random.default_rng(seed)
    data = rng.integers(-3, 4, size=shape).astype(np.float32)
    tensor = ttml.autograd.Tensor.from_numpy(data, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, make_mapper(device))
    return tensor.get_value(), data


def _shards(tensor):
    """Per-device shards in the tensor's coordinate order (row-major over the mesh for everything built here)."""
    return [ttnn.to_torch(shard).float().numpy() for shard in ttnn.get_device_tensors(tensor)]


def _compose_by_label(tensor):
    """Reassemble the full array the way the label says the shards fit together: concatenate along a Shard axis,
    require identical bytes along a Replicate axis (an over-claimed Replicate fails here). Row-major over the
    distribution shape; a collapsed label is the one-axis case."""
    shards = _shards(tensor)
    dist_shape = tuple(int(d) for d in tensor.tensor_topology().distribution_shape())
    placements = list(tensor.tensor_topology().placements())
    assert len(shards) == int(np.prod(dist_shape)), f"{len(shards)} shards for distribution shape {dist_shape}"

    def compose_axis(parts, placement):
        if isinstance(placement, ttnn.PlacementShard):
            return np.concatenate(parts, axis=placement.dim)
        for index, part in enumerate(parts[1:], start=1):
            assert np.array_equal(part, parts[0]), f"Replicate axis claims identical bytes but part {index} differs"
        return parts[0]

    def compose(parts, shape, placements_):
        if len(shape) == 1:
            return compose_axis(parts, placements_[0])
        stride = int(np.prod(shape[1:]))
        blocks = [compose(parts[i * stride : (i + 1) * stride], shape[1:], placements_[1:]) for i in range(shape[0])]
        return compose_axis(blocks, placements_[0])

    return compose(shards, dist_shape, placements)


def _assert_collapsed(tensor, expected_dim: int, msg: str = ""):
    placements = tensor.tensor_topology().placements()
    assert len(placements) == 1, f"Expected a collapsed 1-D label, got {len(placements)} placements. {msg}"
    _assert_shard(tensor, 0, expected_dim, msg)


class TestCollapsedInputs:
    """The default mappers produce one placement for the whole mesh; before the CCL topology helper every
    collective edited placements by mesh-axis index, so cluster_axis=1 did nothing to the label and cluster_axis=0
    overwrote the single placement. These tests pin the per-axis labels the helper derives and check them against
    the bytes by composing the shards the way the label says. All run with strict mode on, so a label the helper
    refuses is an error (asserted explicitly in the negative controls)."""

    ROWS, COLS = MESH_SHAPE_2X2

    @pytest.fixture(autouse=True)
    def _setup(self, ccl_mesh, strict_ccl_topology):
        self.mesh = ccl_mesh

    def _pieces(self, data, dim):
        return np.split(data, self.ROWS * self.COLS, axis=dim)

    def test_all_gather_same_dim_inner_axis_keeps_outer_shard(self):
        """Same-dim case of rule (e): {4}, [Shard(3)] expands to Shard(3) on both axes; gathering dim 3 along axis 1
        rebuilds each row's chunk, so axis 0 keeps Shard(3) -- it must not be cleared to Replicate (the rows differ).
        Negative control: before this change the label stayed {4}, [Shard(3)] (cluster_axis=1 was out of range of
        the one placement), composing to four copies of the row chunks."""
        tensor, data = _collapsed_tensor((1, 1, 64, 128), lambda d: ttnn.shard_tensor_to_mesh_mapper(d, 3), seed=30)
        _assert_collapsed(tensor, 3, "input sanity check")

        result = ttml.core.distributed.all_gather(tensor, 3, cluster_axis=1)

        _assert_shard(result, OTHER_AXIS, 3, "outer axis still holds distinct chunks")
        _assert_replicated(result, CLUSTER_AXIS, "all_gather output")
        assert tuple(int(d) for d in result.tensor_topology().distribution_shape()) == MESH_SHAPE_2X2
        assert np.array_equal(_compose_by_label(result), data)

    def test_all_gather_same_dim_outer_axis_is_refused(self):
        """Rule (d), negative control: {4}, [Shard(3)] puts piece 2r+c on device (r, c); gathering dim 3 along axis 0
        gives device (r, c) pieces c and 2+c side by side, which is no slice of any tensor. Strict mode makes that
        an error; before this change the label silently became {4}, [Replicate], claiming all four devices equal."""
        tensor, _ = _collapsed_tensor((1, 1, 64, 128), lambda d: ttnn.shard_tensor_to_mesh_mapper(d, 3), seed=31)

        with pytest.raises(RuntimeError, match="would interleave"):  # allow-pytest.raises: no expect_error here
            ttml.core.distributed.all_gather(tensor, 3, cluster_axis=0)

    def test_all_gather_other_dim_outer_axis(self):
        """Rule (d) relaxation: gathering dim 2 (not the sharded dim) along axis 0 leaves the dim-3 pieces in place:
        [Replicate, Shard(3)], and column c composes to concat_2(piece c, piece 2+c)."""
        tensor, data = _collapsed_tensor((1, 1, 64, 128), lambda d: ttnn.shard_tensor_to_mesh_mapper(d, 3), seed=32)

        result = ttml.core.distributed.all_gather(tensor, 2, cluster_axis=0)

        _assert_replicated(result, 0, "gathered axis")
        _assert_shard(result, 1, 3, "untouched axis")
        pieces = self._pieces(data, 3)
        expected = np.concatenate(
            [np.concatenate([pieces[r * self.COLS + c] for r in range(self.ROWS)], axis=2) for c in range(self.COLS)],
            axis=3,
        )
        assert np.array_equal(_compose_by_label(result), expected)

    def test_reduce_scatter_collapsed_replicate_shards_the_axis(self):
        """{4}, [Replicate] reduce-scattered on dim 3 along axis 1: [Replicate, Shard(3)] (only an N-D label can say
        Shard here, Replicate there), composing to 2 * data. Negative control: before this change the label stayed
        {4}, [Replicate], claiming the two columns identical."""
        tensor, data = _collapsed_tensor((1, 1, 64, 128), ttnn.replicate_tensor_to_mesh_mapper, seed=33)

        result = ttml.core.distributed.reduce_scatter(tensor, 3, cluster_axis=CLUSTER_AXIS)

        _assert_replicated(result, OTHER_AXIS, "untouched axis")
        _assert_shard(result, CLUSTER_AXIS, 3, "reduce_scatter output")
        assert np.array_equal(_compose_by_label(result), data * self.COLS)

    def test_reduce_scatter_collapsed_shard_same_dim_collapses(self):
        """Rule (c): {4}, [Shard(3)] reduce-scattered on dim 3 along the inner axis: row r sums its two pieces and
        splits the sum back across the row -- row-major hierarchical sharding, the collapsed label again."""
        tensor, data = _collapsed_tensor((1, 1, 64, 256), lambda d: ttnn.shard_tensor_to_mesh_mapper(d, 3), seed=34)

        result = ttml.core.distributed.reduce_scatter(tensor, 3, cluster_axis=CLUSTER_AXIS)

        _assert_collapsed(result, 3, "reduce_scatter output")
        pieces = self._pieces(data, 3)
        expected = np.concatenate([sum(pieces[r * self.COLS : (r + 1) * self.COLS]) for r in range(self.ROWS)], axis=3)
        assert np.array_equal(_compose_by_label(result), expected)

    def test_all_reduce_collapsed_shard_inner_axis(self):
        """{4}, [Shard(2)] all-reduced along axis 1: Replicate on axis 1, Shard(2) kept on axis 0; row r composes to
        the sum of its two pieces."""
        tensor, data = _collapsed_tensor((1, 1, 128, 128), lambda d: ttnn.shard_tensor_to_mesh_mapper(d, 2), seed=35)

        result = ttml.core.distributed.all_reduce(tensor, cluster_axis=CLUSTER_AXIS)

        _assert_shard(result, OTHER_AXIS, 2, "outer axis still holds distinct chunks")
        _assert_replicated(result, CLUSTER_AXIS, "all_reduce output")
        pieces = self._pieces(data, 2)
        expected = np.concatenate([sum(pieces[r * self.COLS : (r + 1) * self.COLS]) for r in range(self.ROWS)], axis=2)
        assert np.array_equal(_compose_by_label(result), expected)

    def test_all_reduce_collapsed_shard_outer_axis_through_the_composite_branch(self):
        """The tt-train DDP shape: a grad distributed {4}, [Shard(0)] (data-parallel on the OUTER axis) all-reduced
        along axis 0. (4, 1, 1, 32) leaves no per-device dim divisible by the axis size, so all_reduce_async takes its
        composite branch (all_gather of the unsqueezed tensor + local sum + reshape). The result is Replicate on axis 0
        and keeps Shard(0) on axis 1: column c holds piece c + piece 2+c. Negative control: before this change the
        composite branch returned {4}, [Replicate] (the inner all_broadcast edited index 0 and the local sum kept it),
        claiming all four devices identical -- the over-claim tt-train's force_replicate_axes then repeated. With the
        inner gather now refused as interleaving under strict mode, all_reduce_async passes
        require_contiguous_gather=false for it and relabels the result itself."""
        tensor, data = _collapsed_tensor((4, 1, 1, 32), lambda d: ttnn.shard_tensor_to_mesh_mapper(d, 0), seed=36)
        _assert_collapsed(tensor, 0, "input sanity check")

        result = ttml.core.distributed.all_reduce(tensor, cluster_axis=0)

        _assert_replicated(result, 0, "reduced axis")
        _assert_shard(result, 1, 0, "untouched axis")
        pieces = self._pieces(data, 0)
        expected = np.concatenate(
            [sum(pieces[r * self.COLS + c] for r in range(self.ROWS)) for c in range(self.COLS)], axis=0
        )
        assert np.array_equal(_compose_by_label(result), expected)

    def test_all_reduce_collapsed_replicate(self):
        """{4}, [Replicate] all-reduced along axis 1 stays Replicate on both axes and composes to 2 * data."""
        tensor, data = _collapsed_tensor((1, 1, 64, 128), ttnn.replicate_tensor_to_mesh_mapper, seed=37)

        result = ttml.core.distributed.all_reduce(tensor, cluster_axis=CLUSTER_AXIS)

        _assert_replicated(result, OTHER_AXIS, "untouched axis")
        _assert_replicated(result, CLUSTER_AXIS, "all_reduce output")
        assert np.array_equal(_compose_by_label(result), data * self.COLS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
