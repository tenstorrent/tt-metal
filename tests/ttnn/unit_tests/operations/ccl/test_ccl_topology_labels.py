# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Output TensorTopology labels of the collective family, checked against the bytes they describe.

Every test builds a mesh tensor whose label is either the collapsed 1-D form (``ShardTensorToMesh`` /
``ReplicateTensorToMesh``: ``{N}, [placement]``) or the N-D form (``ShardTensor2dMesh``: one placement per mesh axis),
runs one collective, and asserts two things about the result:

* the placements (and distribution shape) are the ones ``ttnn/operations/ccl/common/host/ccl_topology_utils`` promises;
* composing the per-device shards *by that label* (concatenate along a Shard axis, require identical bytes along a
  Replicate axis) gives the tensor the collective semantically produced. A label that over-claims Replicate fails the
  identical-bytes check; one that mislabels the Shard dim or order fails the equality with the reference.

Before this change every op edited the input's placements by mesh-axis index, so a collapsed label was edited at index
0 whatever ``cluster_axis`` was (or not at all for ``cluster_axis=1``); each test's docstring names the label the same
call produced then. All tests run with ``ttnn.CONFIG.strict_ccl_topology`` on, so any label the helper refuses is an
error rather than a warning; the negative controls assert that error.

Meshes: 1x2 (N300), 1x8 (T3K line), 2x4 (T3K / galaxy submesh), 8x4 (galaxy; the mesh_device fixture skips it on a
smaller box). Cases that need two non-trivial mesh axes skip on the lines; whole-mesh (cluster_axis=None) cases run on
the lines only. Fabric: 1D, Linear topology (the axis-wise collectives on a 2-D mesh run per row / column), except
strided_all_gather_async, the whole-mesh all_to_all_async and the two direct / strided reduce_scatters, which run on a
wrapping axis (1D ring fabric, Ring topology -- the combination their nightlies exercise) and skip when
``ttnn.get_usable_topology`` says the axis does not wrap. The two strided_all_gather_async cases that run the op are
Wormhole-only (``skip_for_blackhole``, like the op's own nightly): on a Blackhole 1x8 the op hung in both the Linear and
the Ring topology during bench.
"""

import math

import pytest
import torch

import ttnn
from models.common.utility_functions import skip_for_blackhole

FABRIC_1D = [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}]
FABRIC_1D_RING = [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}]
MESHES = [2, 8, (2, 4), (8, 4)]
MESH_IDS = ["1x2", "1x8", "2x4", "8x4"]
TWO_AXIS_MESHES = [(2, 4), (8, 4)]
TWO_AXIS_MESH_IDS = ["2x4", "8x4"]

REPLICATE = "PlacementReplicate()"


def SHARD(dim):
    return f"PlacementShard({dim})"


def _with_strict_ccl_topology(strict):
    previous = ttnn.CONFIG.strict_ccl_topology
    ttnn.CONFIG.strict_ccl_topology = strict
    try:
        yield
    finally:
        ttnn.CONFIG.strict_ccl_topology = previous


@pytest.fixture
def strict_ccl_topology():
    """Run the test with the helper's refusals promoted to errors (what CI runs), restoring the previous mode after."""
    yield from _with_strict_ccl_topology(True)


@pytest.fixture
def warn_only_ccl_topology():
    """Run the test in the shipping default (warn and keep the input's label), whatever TTNN_CONFIG_OVERRIDES says."""
    yield from _with_strict_ccl_topology(False)


# ---------------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------------


def _placements(tensor):
    return [repr(p) for p in tensor.tensor_topology().placements()]


def _dist_shape(tensor):
    return tuple(int(d) for d in tensor.tensor_topology().distribution_shape())


def _shards(tensor):
    """Per-device shards in the tensor's coordinate order (row-major over the mesh for every tensor built here)."""
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]


def _compose_axis(parts, placement):
    if isinstance(placement, ttnn.PlacementShard):
        return torch.cat(parts, dim=placement.dim)
    for index, part in enumerate(parts[1:], start=1):
        assert torch.equal(part, parts[0]), f"Replicate axis claims identical bytes but part {index} differs"
    return parts[0]


def _compose_by_label(tensor):
    """Reassemble the full tensor the way the label says the shards fit together.

    Row-major over the distribution shape: the outermost axis groups the shards into contiguous blocks, the innermost
    axis composes each block. A collapsed label is the one-axis case.
    """
    shards = _shards(tensor)
    dist_shape = _dist_shape(tensor)
    placements = list(tensor.tensor_topology().placements())
    assert len(shards) == math.prod(dist_shape), f"{len(shards)} shards for distribution shape {dist_shape}"

    def compose(parts, shape, placements_):
        if len(shape) == 1:
            return _compose_axis(parts, placements_[0])
        stride = math.prod(shape[1:])
        blocks = [compose(parts[i * stride : (i + 1) * stride], shape[1:], placements_[1:]) for i in range(shape[0])]
        return _compose_axis(blocks, placements_[0])

    return compose(shards, dist_shape, placements)


def _from_torch(tensor, mesh_device, mapper):
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def _integers(shape):
    """Small integers: every sum and concatenation below is exact in bfloat16, so equality checks are exact."""
    return torch.randint(-3, 4, shape).bfloat16()


def _semaphores(mesh_device, count):
    grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    return [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(count)]


def _mesh(mesh_device):
    rows, cols = tuple(mesh_device.shape)
    return rows, cols, rows * cols


def _pieces(tensor, count, dim=3):
    return list(torch.chunk(tensor, count, dim=dim))


def _expected_collapsed_shard_after_inner_gather(rows, cols):
    """Label of a collapsed Shard{d} after a gather / reduce along the inner axis: [Shard{d}, Replicate] on a 2-D
    mesh, the collapsed spelling on a line (the collapsed axis is the collective axis)."""
    return ([SHARD(3), REPLICATE], (rows, cols)) if rows > 1 else ([REPLICATE], (cols,))


def _all_gather_async(tensor, mesh_device, dim, cluster_axis):
    return ttnn.experimental.all_gather_async(
        tensor,
        dim=dim,
        multi_device_global_semaphore=_semaphores(mesh_device, 2),
        topology=ttnn.Topology.Linear,
        cluster_axis=cluster_axis,
    )


def _all_gather(tensor, mesh_device, dim, cluster_axis):
    return ttnn.all_gather(tensor, dim=dim, cluster_axis=cluster_axis, topology=ttnn.Topology.Linear)


def _reduce_scatter_minimal_async(tensor, mesh_device, dim, cluster_axis):
    return ttnn.experimental.reduce_scatter_minimal_async(
        tensor,
        dim=dim,
        multi_device_global_semaphore=_semaphores(mesh_device, 3),
        topology=ttnn.Topology.Linear,
        cluster_axis=cluster_axis,
    )


def _all_reduce_async(tensor, mesh_device, cluster_axis):
    return ttnn.experimental.all_reduce_async(
        tensor,
        cluster_axis=cluster_axis,
        mesh_device=mesh_device,
        math_op=ttnn.ReduceType.Sum,
        topology=ttnn.Topology.Linear,
    )


def _all_reduce_async_whole_mesh(tensor, mesh_device):
    """The `num_devices` overload: no cluster_axis, the ring is the whole mesh; explicit semaphores are required."""
    return ttnn.experimental.all_reduce_async(
        tensor,
        num_devices=mesh_device.get_num_devices(),
        barrier_semaphores=_semaphores(mesh_device, 2),
        rs_global_semaphores=_semaphores(mesh_device, 3),
        ag_global_semaphores=_semaphores(mesh_device, 2),
        math_op=ttnn.ReduceType.Sum,
        topology=ttnn.Topology.Linear,
    )


def _all_reduce_height(rs_ag_branch, ring_size):
    """all_reduce_async picks reduce_scatter + all_gather when some per-device dim has a tile count divisible by the
    ring size (finding_scatter_dim) and the composite all_gather + local sum otherwise. Inputs here are one tile
    wide per device, so the height decides: 32 (one tile) is never divisible, 32 * ring_size always is."""
    return 32 * ring_size if rs_ag_branch else 32


ALL_GATHERS = [_all_gather_async, _all_gather]
ALL_GATHER_IDS = ["all_gather_async", "all_gather"]


# ---------------------------------------------------------------------------------------------------------------------
# all_gather (prim ccl/all_gather and experimental all_gather_async)
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, indirect=True, ids=MESH_IDS)
@pytest.mark.parametrize("all_gather", ALL_GATHERS, ids=ALL_GATHER_IDS)
def test_all_gather_collapsed_shard_inner_axis(mesh_device, all_gather, strict_ccl_topology):
    """ShardTensorToMesh(dim=3) gives {N}, [Shard(3)] in row-major device order; gathering dim 3 along the innermost
    mesh axis puts each row's chunk back together: [Shard(3), Replicate] on a 2-D mesh, {N}, [Replicate] on a line,
    and the composed tensor is the original. Negative control: before this change the label stayed {N}, [Shard(3)]
    on a 2-D mesh (cluster_axis=1 was out of range of the single placement), so composing it concatenated eight
    copies of the row chunks."""
    torch.manual_seed(0)
    rows, cols, num_devices = _mesh(mesh_device)
    full = _integers([1, 1, 32, 32 * num_devices])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=3))

    tt_output = all_gather(tt_input, mesh_device, dim=3, cluster_axis=1)

    expected_placements, expected_shape = _expected_collapsed_shard_after_inner_gather(rows, cols)
    assert _placements(tt_output) == expected_placements
    assert _dist_shape(tt_output) == expected_shape
    assert torch.equal(_compose_by_label(tt_output), full)

    # The label does not depend on the program cache: a second call (cache hit) is labelled the same.
    tt_again = all_gather(tt_input, mesh_device, dim=3, cluster_axis=1)
    assert _placements(tt_again) == expected_placements


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, indirect=True, ids=MESH_IDS)
@pytest.mark.parametrize("all_gather", ALL_GATHERS, ids=ALL_GATHER_IDS)
@pytest.mark.parametrize("cluster_axis", [0, 1])
def test_all_gather_nd_input_replicates_only_the_gathered_axis(
    mesh_device, all_gather, cluster_axis, strict_ccl_topology
):
    """ShardTensor2dMesh dims=(2, 3): rows shard dim 2, columns shard dim 3. Gathering the dim a mesh axis shards
    along that axis makes it Replicate and leaves the other axis alone; the composed tensor is the original."""
    torch.manual_seed(1)
    rows, cols, _ = _mesh(mesh_device)
    if tuple(mesh_device.shape)[cluster_axis] == 1:
        pytest.skip("gathering along a size-1 mesh axis is a no-op the ops reject")
    dims = (2 if rows > 1 else None, 3)
    full = _integers([1, 1, 32 * rows, 32 * cols])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=dims))

    tt_output = all_gather(tt_input, mesh_device, dim=dims[cluster_axis], cluster_axis=cluster_axis)

    expected = [SHARD(2) if dims[0] is not None else REPLICATE, SHARD(3)]
    expected[cluster_axis] = REPLICATE
    assert _placements(tt_output) == expected
    assert _dist_shape(tt_output) == (rows, cols)
    assert torch.equal(_compose_by_label(tt_output), full)


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
@pytest.mark.parametrize("all_gather", ALL_GATHERS, ids=ALL_GATHER_IDS)
def test_all_gather_keeps_a_same_dim_shard_on_the_other_axis(mesh_device, all_gather, strict_ccl_topology):
    """Rule (e): ShardTensor2dMesh dims=(3, None) shards dim 3 across rows and replicates across columns. Gathering
    dim 3 along the columns concatenates four identical copies of the row's chunk; the rows still hold distinct
    chunks, so the label stays [Shard(3), Replicate]. Negative control: prim all_gather used to replicate every axis
    whose Shard dim equalled the gather dim, giving [Replicate, Replicate]; composing that fails the identical-bytes
    check because row 0 and row 1 differ."""
    torch.manual_seed(2)
    rows, cols, _ = _mesh(mesh_device)
    full = _integers([1, 1, 32, 32 * rows])
    tt_input = _from_torch(
        full, mesh_device, ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(3, None))
    )

    tt_output = all_gather(tt_input, mesh_device, dim=3, cluster_axis=1)

    assert _placements(tt_output) == [SHARD(3), REPLICATE]
    row_chunks = _pieces(full, rows)
    expected = torch.cat([torch.cat([chunk] * cols, dim=3) for chunk in row_chunks], dim=3)
    assert torch.equal(_compose_by_label(tt_output), expected)


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
@pytest.mark.parametrize("all_gather", ALL_GATHERS, ids=ALL_GATHER_IDS)
def test_all_gather_collapsed_shard_outer_axis_is_refused(mesh_device, all_gather, strict_ccl_topology, expect_error):
    """Rule (d), negative control. {8}, [Shard(3)] puts piece 4r+c on device (r, c); gathering dim 3 along the rows
    gives device (r, c) the pieces c and C+c side by side, which is no slice of any tensor: no label describes it.
    Before this change the label was {N}, [Replicate] (the edit landed on index 0), which claims every device holds
    the same bytes; the serialiser would have kept one column."""
    torch.manual_seed(3)
    _, _, num_devices = _mesh(mesh_device)
    tt_input = _from_torch(
        _integers([1, 1, 32, 32 * num_devices]), mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=3)
    )

    with expect_error(RuntimeError, "would interleave"):
        all_gather(tt_input, mesh_device, dim=3, cluster_axis=0)


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
@pytest.mark.parametrize("all_gather", ALL_GATHERS, ids=ALL_GATHER_IDS)
def test_all_gather_collapsed_shard_outer_axis_other_dim(mesh_device, all_gather, strict_ccl_topology):
    """Rule (d) relaxation: gathering a dim the collapsed label does not shard leaves the dim-3 pieces where they
    are, so the outer axis is honest: [Replicate, Shard(3)], and column c composes to concat_2 of pieces c, C+c, ..."""
    torch.manual_seed(4)
    rows, cols, num_devices = _mesh(mesh_device)
    full = _integers([1, 1, 32, 32 * num_devices])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=3))

    tt_output = all_gather(tt_input, mesh_device, dim=2, cluster_axis=0)

    assert _placements(tt_output) == [REPLICATE, SHARD(3)]
    pieces = _pieces(full, num_devices)
    expected = torch.cat([torch.cat([pieces[r * cols + c] for r in range(rows)], dim=2) for c in range(cols)], dim=3)
    assert torch.equal(_compose_by_label(tt_output), expected)


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
def test_all_gather_collapsed_shard_outer_axis_warn_only_keeps_input_label(mesh_device, warn_only_ccl_topology):
    """The default (warn-only) mode: the refused gather still runs, logs a warning, and the output keeps the input's
    label -- {N}, [Shard(3)] -- instead of the old {N}, [Replicate] over-claim."""
    torch.manual_seed(5)
    _, _, num_devices = _mesh(mesh_device)
    tt_input = _from_torch(
        _integers([1, 1, 32, 32 * num_devices]), mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=3)
    )

    tt_output = _all_gather(tt_input, mesh_device, dim=3, cluster_axis=0)

    assert _placements(tt_output) == [SHARD(3)]
    assert _dist_shape(tt_output) == (num_devices,)


# ---------------------------------------------------------------------------------------------------------------------
# all_broadcast
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, indirect=True, ids=MESH_IDS)
@pytest.mark.parametrize("cluster_axis", [0, 1])
def test_all_broadcast_collapsed_shard(mesh_device, cluster_axis, strict_ccl_topology):
    """Output k of all_broadcast along an axis holds device k's shard on every device of that axis: Replicate on
    the axis, the input's placement elsewhere. Nothing is concatenated, so both axes of a collapsed Shard are honest
    (no contiguity requirement). Negative control: before this change cluster_axis=1 left {N}, [Shard(3)] in place,
    claiming eight distinct pieces where each output holds only `rows` distinct ones."""
    torch.manual_seed(6)
    rows, cols, num_devices = _mesh(mesh_device)
    if tuple(mesh_device.shape)[cluster_axis] == 1:
        pytest.skip("broadcast along a size-1 mesh axis is rejected by the op")
    full = _integers([1, 1, 32, 32 * num_devices])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=3))

    outputs = ttnn.all_broadcast(tt_input, cluster_axis=cluster_axis, topology=ttnn.Topology.Linear)

    pieces = _pieces(full, num_devices)
    axis_size = tuple(mesh_device.shape)[cluster_axis]
    assert len(outputs) == axis_size
    for k, tt_output in enumerate(outputs):
        if rows == 1:
            assert _placements(tt_output) == [REPLICATE]
            expected = pieces[k]
        elif cluster_axis == 1:
            assert _placements(tt_output) == [SHARD(3), REPLICATE]
            expected = torch.cat([pieces[r * cols + k] for r in range(rows)], dim=3)
        else:
            assert _placements(tt_output) == [REPLICATE, SHARD(3)]
            expected = torch.cat(pieces[k * cols : (k + 1) * cols], dim=3)
        assert torch.equal(_compose_by_label(tt_output), expected), f"output {k}"


# ---------------------------------------------------------------------------------------------------------------------
# reduce_scatter_minimal_async
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, indirect=True, ids=MESH_IDS)
def test_reduce_scatter_collapsed_replicate(mesh_device, strict_ccl_topology):
    """ReplicateTensorToMesh gives {N}, [Replicate]. Reduce-scattering dim -1 along the innermost axis leaves each
    device with its slice of `cols` * T: [Replicate, Shard(3)] on a 2-D mesh, {N}, [Shard(3)] on a line (rule (f):
    the normalised dim is written, not -1). Negative control: before this change the label stayed {N}, [Replicate]
    on a 2-D mesh; composing it asserts identical columns."""
    torch.manual_seed(7)
    rows, cols, _ = _mesh(mesh_device)
    full = _integers([1, 1, 32, 32 * cols])
    tt_input = _from_torch(full, mesh_device, ttnn.ReplicateTensorToMesh(mesh_device))

    tt_output = _reduce_scatter_minimal_async(tt_input, mesh_device, dim=-1, cluster_axis=1)

    if rows > 1:
        assert _placements(tt_output) == [REPLICATE, SHARD(3)]
        assert _dist_shape(tt_output) == (rows, cols)
    else:
        assert _placements(tt_output) == [SHARD(3)]
        assert _dist_shape(tt_output) == (cols,)
    assert torch.equal(_compose_by_label(tt_output), full * cols)


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, indirect=True, ids=MESH_IDS)
def test_reduce_scatter_collapsed_shard_same_dim_inner_axis(mesh_device, strict_ccl_topology):
    """{N}, [Shard(3)] reduce-scattered on dim 3 along the innermost axis: row r sums its `cols` pieces and splits
    the sum back across the row -- row-major hierarchical sharding, i.e. the collapsed label again (rule (c))."""
    torch.manual_seed(8)
    rows, cols, num_devices = _mesh(mesh_device)
    full = _integers([1, 1, 32, 32 * num_devices * cols])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=3))

    tt_output = _reduce_scatter_minimal_async(tt_input, mesh_device, dim=3, cluster_axis=1)

    assert _placements(tt_output) == [SHARD(3)]
    assert _dist_shape(tt_output) == (num_devices,)
    pieces = _pieces(full, num_devices)
    expected = torch.cat([sum(pieces[r * cols : (r + 1) * cols]) for r in range(rows)], dim=3)
    assert torch.equal(_compose_by_label(tt_output), expected)


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
def test_reduce_scatter_nd_other_dim_keeps_other_axis(mesh_device, strict_ccl_topology):
    """ShardTensor2dMesh dims=(2, None) then reduce_scatter dim 3 along the columns: [Shard(2), Shard(3)]."""
    torch.manual_seed(9)
    rows, cols, _ = _mesh(mesh_device)
    full = _integers([1, 1, 32 * rows, 32 * cols])
    tt_input = _from_torch(
        full, mesh_device, ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(2, None))
    )

    tt_output = _reduce_scatter_minimal_async(tt_input, mesh_device, dim=3, cluster_axis=1)

    assert _placements(tt_output) == [SHARD(2), SHARD(3)]
    assert torch.equal(_compose_by_label(tt_output), full * cols)


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
def test_reduce_scatter_nd_same_dim_inner_axis_collapses(mesh_device, strict_ccl_topology):
    """Rule (c): ShardTensor2dMesh dims=(3, None) holds chunk r of dim 3 on row r; reduce-scattering dim 3 along the
    columns splits each row's (summed) chunk into C -- device (r, c) holds piece Cr+c of C * T: the collapsed
    {N}, [Shard(3)]. Negative control: before this change the op replicated the other axis that shared the dim and
    labelled [Replicate, Shard(3)], claiming both rows identical; the serialiser would have kept row 0 only."""
    torch.manual_seed(10)
    rows, cols, num_devices = _mesh(mesh_device)
    full = _integers([1, 1, 32, 32 * num_devices])
    tt_input = _from_torch(
        full, mesh_device, ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(3, None))
    )

    tt_output = _reduce_scatter_minimal_async(tt_input, mesh_device, dim=3, cluster_axis=1)

    assert _placements(tt_output) == [SHARD(3)]
    assert _dist_shape(tt_output) == (num_devices,)
    assert torch.equal(_compose_by_label(tt_output), full * cols)


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
def test_reduce_scatter_nd_same_dim_outer_axis_is_refused(mesh_device, strict_ccl_topology, expect_error):
    """Rule (c) as amended, negative control: ShardTensor2dMesh dims=(None, 3) holds chunk c on column c;
    reduce-scattering dim 3 along the rows gives device (r, c) part r of chunk c -- piece Rc+r, column-major, which
    no row-major label describes. Before this change the op emitted [Shard(3), Replicate], claiming identical
    columns."""
    torch.manual_seed(11)
    rows, cols, num_devices = _mesh(mesh_device)
    tt_input = _from_torch(
        _integers([1, 1, 32, 32 * num_devices]),
        mesh_device,
        ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(None, 3)),
    )

    with expect_error(RuntimeError, "no TensorTopology can express"):
        _reduce_scatter_minimal_async(tt_input, mesh_device, dim=3, cluster_axis=0)


# ---------------------------------------------------------------------------------------------------------------------
# all_reduce_async (host overload: composite all_gather + local sum, or reduce_scatter + all_gather)
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, indirect=True, ids=MESH_IDS)
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize("rs_ag_branch", [False, True], ids=["composite_branch", "reduce_scatter_all_gather_branch"])
def test_all_reduce_async_collapsed_shard(mesh_device, cluster_axis, rs_ag_branch, strict_ccl_topology):
    """all_reduce of {N}, [Shard(3)] along either axis: the reduced axis becomes Replicate, the other keeps Shard(3).
    The height selects the branch (see _all_reduce_height): the composite all_gather of the unsqueezed tensor + local
    sum + reshape, or reduce_scatter + all_gather. The label is the same on both -- the returned tensor is relabelled
    from the original input -- so the branch is not observable here beyond the height that forces it. Negative
    control: before this change the composite branch returned {N}, [Replicate] for either axis (the inner
    all_broadcast edited index 0, the local sum kept it) -- along the columns that claims the two rows identical, and
    the same over-claim is what tt-train's force_replicate_axes papered over. Along the rows the gather itself would
    now be refused as interleaving; all_reduce runs its intermediates inside the helper's CallerRelabelsScope because
    it sums the gathered pieces rather than keeping them and labels the result itself."""
    torch.manual_seed(12)
    rows, cols, num_devices = _mesh(mesh_device)
    if tuple(mesh_device.shape)[cluster_axis] == 1:
        pytest.skip("reducing along a size-1 mesh axis is rejected by the op")
    height = _all_reduce_height(rs_ag_branch, tuple(mesh_device.shape)[cluster_axis])
    full = _integers([1, 1, height, 32 * num_devices])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=3))

    tt_output = _all_reduce_async(tt_input, mesh_device, cluster_axis=cluster_axis)

    pieces = _pieces(full, num_devices)
    if rows == 1:
        assert _placements(tt_output) == [REPLICATE]
        expected = sum(pieces)
    elif cluster_axis == 1:
        assert _placements(tt_output) == [SHARD(3), REPLICATE]
        expected = torch.cat([sum(pieces[r * cols : (r + 1) * cols]) for r in range(rows)], dim=3)
    else:
        assert _placements(tt_output) == [REPLICATE, SHARD(3)]
        expected = torch.cat([sum(pieces[r * cols + c] for r in range(rows)) for c in range(cols)], dim=3)
    assert torch.equal(_compose_by_label(tt_output), expected)


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, indirect=True, ids=MESH_IDS)
def test_all_reduce_async_collapsed_shard_scattered_on_the_sharded_dim(mesh_device, strict_ccl_topology):
    """all_reduce of {N}, [Shard(3)] along the outer axis when finding_scatter_dim picks dim 3 itself: with `rows`
    tiles per device along dim 3 the reduce_scatter + all_gather branch scatters the sharded dim. The intermediates
    have no honest label -- the reduce_scatter leaves device (r, c) with piece c * R + r (column-major) and the
    all_gather of it interleaves -- but the result is labelled from the input: [Replicate, Shard(3)], one column sum
    per column. Negative control: before CallerRelabelsScope the prim all_gather hook refused the intermediate
    ("would interleave") and this call failed under strict mode although the label it returns is right."""
    torch.manual_seed(15)
    rows, cols, num_devices = _mesh(mesh_device)
    if rows == 1:
        pytest.skip("reducing along a size-1 mesh axis is rejected by the op")
    full = _integers([1, 1, 32, 32 * rows * num_devices])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=3))

    tt_output = _all_reduce_async(tt_input, mesh_device, cluster_axis=0)

    pieces = _pieces(full, num_devices)
    assert _placements(tt_output) == [REPLICATE, SHARD(3)]
    expected = torch.cat([sum(pieces[r * cols + c] for r in range(rows)) for c in range(cols)], dim=3)
    assert torch.equal(_compose_by_label(tt_output), expected)


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, indirect=True, ids=MESH_IDS)
def test_all_reduce_async_collapsed_replicate(mesh_device, strict_ccl_topology):
    """{N}, [Replicate] all-reduced along the innermost axis stays Replicate everywhere and composes to `cols` * T."""
    torch.manual_seed(13)
    rows, cols, _ = _mesh(mesh_device)
    full = _integers([1, 1, 32, 64])
    tt_input = _from_torch(full, mesh_device, ttnn.ReplicateTensorToMesh(mesh_device))

    tt_output = _all_reduce_async(tt_input, mesh_device, cluster_axis=1)

    assert _placements(tt_output) == ([REPLICATE, REPLICATE] if rows > 1 else [REPLICATE])
    assert torch.equal(_compose_by_label(tt_output), full * cols)


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, indirect=True, ids=MESH_IDS)
@pytest.mark.parametrize("sharded", [True, False], ids=["collapsed_shard", "collapsed_replicate"])
@pytest.mark.parametrize("rs_ag_branch", [False, True], ids=["composite_branch", "reduce_scatter_all_gather_branch"])
def test_all_reduce_async_whole_mesh_overload(mesh_device, sharded, rs_ag_branch, strict_ccl_topology):
    """The `num_devices` overload (no cluster_axis; what tt-train uses on a line mesh) reduces over the whole mesh, so
    every device ends up with the sum: {N}, [Replicate] whatever the input was, keeping the input's distribution shape
    and coordinates (the whole-mesh ring walks them in order). Both branches (see _all_reduce_height) already
    produced that label for these inputs -- the whole-mesh edits replicate everything -- so this pins the explicit
    host relabel added for the overload and the whole-mesh labels of the reduce_scatter_minimal_async /
    all_gather_async it runs, with the bytes: the sum of the pieces for a collapsed Shard, N * T for Replicate. A
    whole-mesh ring over a 2-D mesh is not a physical line under 1D fabric, so this runs on the line meshes only."""
    torch.manual_seed(14)
    rows, _, num_devices = _mesh(mesh_device)
    if rows > 1:
        pytest.skip("whole-mesh (cluster_axis=None) all_reduce needs a line mesh")
    height = _all_reduce_height(rs_ag_branch, num_devices)
    full = _integers([1, 1, height, 32 * num_devices if sharded else 32])
    mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3) if sharded else ttnn.ReplicateTensorToMesh(mesh_device)
    tt_input = _from_torch(full, mesh_device, mapper)

    tt_output = _all_reduce_async_whole_mesh(tt_input, mesh_device)

    assert _placements(tt_output) == [REPLICATE]
    assert _dist_shape(tt_output) == (num_devices,)
    expected = sum(_pieces(full, num_devices)) if sharded else full * num_devices
    assert torch.equal(_compose_by_label(tt_output), expected)


# ---------------------------------------------------------------------------------------------------------------------
# experimental/ccl ops that had no compute_output_topologies until now (strided_all_gather_async,
# all_to_all_async_generic, reduce_scatter_minimal_direct, strided_reduce_scatter_async). Before this change every one
# of them took launch()'s union default -- the INPUT's label -- so the negative control for each test below is the
# same: composing the output by the input's label (Shard where the collective produced Replicate, or the input's dim
# where the collective produced another) either fails the identical-bytes check or reassembles the wrong tensor.
# ---------------------------------------------------------------------------------------------------------------------


def _skip_unless_ring(tensor, cluster_axis):
    """reduce_scatter_minimal_direct and strided_reduce_scatter_async only run on an axis that wraps into a ring."""
    if ttnn.get_usable_topology(tensor, cluster_axis=cluster_axis) != ttnn.Topology.Ring:
        pytest.skip("needs a mesh axis that wraps into a ring (1D ring fabric)")


def _strided_all_gather_async(tensor, mesh_device, dim, cluster_axis):
    # The strided variant is the matmul-fused all_gather's reader; stand-alone it still needs the matmul block
    # geometry because the factory divides by it. As in its nightly: Ring topology, 2 semaphores, 1 link, one matmul
    # block per device slice (the slices here are one tile high and one tile wide, so 1 x 1 tiles).
    return ttnn.experimental.strided_all_gather_async(
        tensor,
        persistent_output_buffer=None,
        dim=dim,
        multi_device_global_semaphore=_semaphores(mesh_device, 2),
        topology=ttnn.Topology.Ring,
        cluster_axis=cluster_axis,
        mm_cores_y=1,
        mm_block_ht=1,
        mm_block_wt=1,
    )


def _all_to_all_async_generic(tensor, in_dim, out_dim, cluster_axis):
    return ttnn.experimental.all_to_all_async_generic(
        tensor, in_dim=in_dim, out_dim=out_dim, topology=ttnn.Topology.Linear, cluster_axis=cluster_axis
    )


def _all_to_all_async(tensor, mesh_device, full_shape, in_dim, out_dim):
    """The whole-mesh all_to_all with the caller-owned intermediate and output buffers its nightly uses (both the
    per-device output shape: the full tensor with out_dim split across the ring)."""
    _, _, num_devices = _mesh(mesh_device)
    output_shape = list(full_shape)
    output_shape[out_dim] //= num_devices
    intermediate, output = (
        _from_torch(torch.zeros(output_shape).bfloat16(), mesh_device, ttnn.ReplicateTensorToMesh(mesh_device))
        for _ in range(2)
    )
    return ttnn.experimental.all_to_all_async(
        tensor,
        persistent_intermediate_buffer=intermediate,
        persistent_output_buffer=output,
        in_dim=in_dim,
        out_dim=out_dim,
        multi_device_global_semaphore=_semaphores(mesh_device, 1)[0],
        topology=ttnn.Topology.Ring,
    )


def _reduce_scatter_minimal_direct(tensor, dim, cluster_axis):
    return ttnn.experimental.reduce_scatter_minimal_direct(
        tensor, dim=dim, cluster_axis=cluster_axis, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _strided_reduce_scatter_async(tensor, mesh_device, dim, cluster_axis):
    # Each device's slice is 2 x 2 tiles: one matmul block (mm_block_ht x mm_block_wt) of the fused matmul the strided
    # reduce_scatter is the writer of; stand-alone it still needs that geometry.
    return ttnn.experimental.strided_reduce_scatter_async(
        tensor,
        dim=dim,
        multi_device_global_semaphore=_semaphores(mesh_device, 3),
        mm_block_ht=2,
        mm_block_wt=2,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Ring,
        cluster_axis=cluster_axis,
        mm_cores_y=1,
        mm_N_full_block_wt=2,
        chunk_width_in_mm_blocks=1,
    )


# ---------------------------------------------------------------------------------------------------------------------
# strided_all_gather_async (ring fabric)
# ---------------------------------------------------------------------------------------------------------------------


@skip_for_blackhole("strided_all_gather_async hung on a Blackhole 1x8 in both topologies; its nightly is WH-only")
@pytest.mark.parametrize("device_params", FABRIC_1D_RING, indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, indirect=True, ids=MESH_IDS)
def test_strided_all_gather_async_collapsed_shard_inner_axis(mesh_device, strict_ccl_topology):
    """{N}, [Shard(3)] gathered on dim 3 along the innermost axis: [Shard(3), Replicate] on a 2-D mesh, {N},
    [Replicate] on a line, composing to the original. Negative control: the union default kept {N}, [Shard(3)], so
    composing concatenated N copies of the gathered row. Wormhole-only: the op's own nightly is skip_for_blackhole;
    on a Blackhole 1x8 the op hung in both Linear and Ring topologies during bench."""
    torch.manual_seed(20)
    rows, cols, num_devices = _mesh(mesh_device)
    full = _integers([1, 1, 32, 32 * num_devices])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=3))
    _skip_unless_ring(tt_input, 1)

    tt_output = _strided_all_gather_async(tt_input, mesh_device, dim=3, cluster_axis=1)

    expected_placements, expected_shape = _expected_collapsed_shard_after_inner_gather(rows, cols)
    assert _placements(tt_output) == expected_placements
    assert _dist_shape(tt_output) == expected_shape
    assert torch.equal(_compose_by_label(tt_output), full)

    # The label does not depend on the program cache: a second call (cache hit) is labelled the same.
    tt_again = _strided_all_gather_async(tt_input, mesh_device, dim=3, cluster_axis=1)
    assert _placements(tt_again) == expected_placements


@skip_for_blackhole("strided_all_gather_async hung on a Blackhole 1x8 in both topologies; its nightly is WH-only")
@pytest.mark.parametrize("device_params", FABRIC_1D_RING, indirect=True)
@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
def test_strided_all_gather_async_nd_input_replicates_only_the_gathered_axis(mesh_device, strict_ccl_topology):
    """ShardTensor2dMesh dims=(2, 3) gathered on dim 3 along the columns: [Shard(2), Replicate]; the rows' Shard(2) is
    untouched and the composed tensor is the original. Wormhole-only: the op's own nightly is skip_for_blackhole; on a
    Blackhole 1x8 the op hung in both Linear and Ring topologies during bench."""
    torch.manual_seed(21)
    rows, cols, _ = _mesh(mesh_device)
    full = _integers([1, 1, 32 * rows, 32 * cols])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(2, 3)))
    _skip_unless_ring(tt_input, 1)

    tt_output = _strided_all_gather_async(tt_input, mesh_device, dim=3, cluster_axis=1)

    assert _placements(tt_output) == [SHARD(2), REPLICATE]
    assert _dist_shape(tt_output) == (rows, cols)
    assert torch.equal(_compose_by_label(tt_output), full)


@pytest.mark.parametrize("device_params", FABRIC_1D_RING, indirect=True)
@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
def test_strided_all_gather_async_collapsed_shard_outer_axis_is_refused(mesh_device, strict_ccl_topology, expect_error):
    """Rule (d), negative control: gathering dim 3 of {N}, [Shard(3)] along the rows interleaves pieces c and C+c;
    the hook refuses before the op runs (so no ring is needed, nothing is skipped, and this runs on Blackhole too).
    Before this change the output silently kept {N}, [Shard(3)]."""
    torch.manual_seed(22)
    _, _, num_devices = _mesh(mesh_device)
    tt_input = _from_torch(
        _integers([1, 1, 32, 32 * num_devices]), mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=3)
    )

    with expect_error(RuntimeError, "would interleave"):
        _strided_all_gather_async(tt_input, mesh_device, dim=3, cluster_axis=0)


# ---------------------------------------------------------------------------------------------------------------------
# all_to_all_async_generic
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, indirect=True, ids=MESH_IDS)
def test_all_to_all_async_generic_collapsed_shard_inner_axis(mesh_device, strict_ccl_topology):
    """all_to_all is an all_gather of in_dim followed by each device keeping its ring-index piece of out_dim, and so is
    its label. {N}, [Shard(2)] exchanged along the innermost axis with in_dim=2, out_dim=3: on a line every device
    ends up with all rows and its own column block, {N}, [Shard(3)]; on a 2-D mesh each row keeps its (contiguous)
    row block and splits the columns, [Shard(2), Shard(3)]. Either way composing by the label gives the original.
    Negative control: the union default kept {N}, [Shard(2)], which stacks the column blocks along the rows."""
    torch.manual_seed(23)
    rows, cols, num_devices = _mesh(mesh_device)
    full = _integers([1, 1, 32 * num_devices, 32 * cols])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=2))

    tt_output = _all_to_all_async_generic(tt_input, in_dim=2, out_dim=3, cluster_axis=1)

    if rows > 1:
        assert _placements(tt_output) == [SHARD(2), SHARD(3)]
        assert _dist_shape(tt_output) == (rows, cols)
    else:
        assert _placements(tt_output) == [SHARD(3)]
        assert _dist_shape(tt_output) == (cols,)
    assert torch.equal(_compose_by_label(tt_output), full)


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
def test_all_to_all_async_generic_nd_transposes_the_exchanged_axis(mesh_device, strict_ccl_topology):
    """The nightly test's 2-D layout: ShardTensor2dMesh dims=(3, 2) holds column block r and row block c on device
    (r, c). Exchanging along the columns with in_dim=2, out_dim=3 gathers the rows and leaves device (r, c) with
    column sub-block Cr+c -- row-major hierarchical sharding of dim 3, the collapsed {N}, [Shard(3)] the nightly
    composes with ConcatMeshToTensor(dim=3). Negative control: the union default kept [Shard(3), Shard(2)], whose
    composition concatenates full-height blocks along dim 2 and reads the wrong tensor."""
    torch.manual_seed(24)
    rows, cols, num_devices = _mesh(mesh_device)
    full = _integers([1, 1, 32 * cols, 32 * num_devices])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(3, 2)))

    tt_output = _all_to_all_async_generic(tt_input, in_dim=2, out_dim=3, cluster_axis=1)

    assert _placements(tt_output) == [SHARD(3)]
    assert _dist_shape(tt_output) == (num_devices,)
    assert torch.equal(_compose_by_label(tt_output), full)


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
def test_all_to_all_async_generic_collapsed_shard_outer_axis_is_refused(mesh_device, strict_ccl_topology, expect_error):
    """Negative control: {N}, [Shard(2)] exchanged along the rows with in_dim=2 gathers row pieces c and C+c side by
    side -- the all_gather half of the label refuses by rule (d), "would interleave" (gather family: a warning in
    warn-only mode), before the op runs. Routing the input label through the reduce_scatter-shaped alias alone would
    have labelled this [Shard(3), Shard(2)], a permuted tensor. This is a different refusal from the nightly's
    cluster_axis=0 cases: those exchange an N-D [Shard(in), Shard(out)], whose gather half succeeds
    ([Replicate, Shard(out)]) and whose scatter half rule (c) then refuses -- out_dim already sharded on the inner
    axis while the new Shard axis is the outer one, "no TensorTopology can express" (scatter family: logged at error
    level) -- so the output keeps the union default there."""
    torch.manual_seed(25)
    _, cols, num_devices = _mesh(mesh_device)
    tt_input = _from_torch(
        _integers([1, 1, 32 * num_devices, 32 * cols]), mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=2)
    )

    with expect_error(RuntimeError, "would interleave"):
        _all_to_all_async_generic(tt_input, in_dim=2, out_dim=3, cluster_axis=0)


# ---------------------------------------------------------------------------------------------------------------------
# all_to_all_async (whole mesh, ring fabric)
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", FABRIC_1D_RING, indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, indirect=True, ids=MESH_IDS)
def test_all_to_all_async_whole_mesh(mesh_device, strict_ccl_topology):
    """The whole-mesh all_to_all of {N}, [Shard(2)] with in_dim=2, out_dim=3 leaves device k with every row and
    column block k. Its program factory numbers the ring by MeshDeviceView::get_ring_devices() -- the walk around the
    mesh boundary -- which is the coordinate order only on a line, so: on 1x2 / 1x8 the label is {N}, [Shard(3)] and
    composing by it gives the original (negative control: the union default kept {N}, [Shard(2)], which stacks the
    column blocks along the rows); on a 2-D mesh the hook deliberately keeps the union default -- the input's
    {N}, [Shard(2)] -- because {N}, [Shard(3)] over row-major coordinates would put device (1, 0)'s piece 7 where
    piece 4 belongs, and no label can carry the boundary order. The caller's persistent output buffer (replicated
    zeros before the call) is the tensor relabelled either way. Shape: each device's input is 4 x 4N tiles so every
    destination shard is 4 x 4 tiles = eight two-tile packets; the op's factory statically partitions the
    intermediate buffer by packet id and rejects shards of a single packet (its final packet would overflow the
    buffer), which is what the nightly's shapes also satisfy."""
    torch.manual_seed(30)
    rows, cols, num_devices = _mesh(mesh_device)
    full = _integers([1, 1, 128 * num_devices, 128 * num_devices])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=2))
    _skip_unless_ring(tt_input, None)

    tt_output = _all_to_all_async(tt_input, mesh_device, full.shape, in_dim=2, out_dim=3)

    if rows > 1 and cols > 1:
        assert _placements(tt_output) == [SHARD(2)]
        assert _dist_shape(tt_output) == (num_devices,)
    else:
        assert _placements(tt_output) == [SHARD(3)]
        assert _dist_shape(tt_output) == (num_devices,)
        assert torch.equal(_compose_by_label(tt_output), full)


# ---------------------------------------------------------------------------------------------------------------------
# reduce_scatter_minimal_direct (ring fabric)
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", FABRIC_1D_RING, indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, indirect=True, ids=MESH_IDS)
def test_reduce_scatter_minimal_direct_collapsed_replicate(mesh_device, strict_ccl_topology):
    """{N}, [Replicate] reduce-scattered on dim 3 along the innermost axis: [Replicate, Shard(3)] on a 2-D mesh,
    {N}, [Shard(3)] on a line, composing to `cols` * T. Negative control: the union default kept {N}, [Replicate],
    whose composition asserts identical columns. The staging buffer (return index 1, not surfaced here) keeps the
    input's label."""
    torch.manual_seed(26)
    rows, cols, _ = _mesh(mesh_device)
    full = _integers([1, 1, 32, 32 * cols])
    tt_input = _from_torch(full, mesh_device, ttnn.ReplicateTensorToMesh(mesh_device))
    _skip_unless_ring(tt_input, 1)

    tt_output = _reduce_scatter_minimal_direct(tt_input, dim=3, cluster_axis=1)

    if rows > 1:
        assert _placements(tt_output) == [REPLICATE, SHARD(3)]
        assert _dist_shape(tt_output) == (rows, cols)
    else:
        assert _placements(tt_output) == [SHARD(3)]
        assert _dist_shape(tt_output) == (cols,)
    assert torch.equal(_compose_by_label(tt_output), full * cols)


@pytest.mark.parametrize("device_params", FABRIC_1D_RING, indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, indirect=True, ids=MESH_IDS)
def test_reduce_scatter_minimal_direct_collapsed_shard_same_dim_inner_axis(mesh_device, strict_ccl_topology):
    """{N}, [Shard(3)] reduce-scattered on dim 3 along the innermost axis: each row sums its pieces and splits the sum
    back across the row -- the collapsed label again (rule (c)), composing to the concatenated row sums."""
    torch.manual_seed(27)
    rows, cols, num_devices = _mesh(mesh_device)
    full = _integers([1, 1, 32, 32 * num_devices * cols])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=3))
    _skip_unless_ring(tt_input, 1)

    tt_output = _reduce_scatter_minimal_direct(tt_input, dim=3, cluster_axis=1)

    assert _placements(tt_output) == [SHARD(3)]
    assert _dist_shape(tt_output) == (num_devices,)
    pieces = _pieces(full, num_devices)
    expected = torch.cat([sum(pieces[r * cols : (r + 1) * cols]) for r in range(rows)], dim=3)
    assert torch.equal(_compose_by_label(tt_output), expected)


# ---------------------------------------------------------------------------------------------------------------------
# strided_reduce_scatter_async (ring fabric)
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", FABRIC_1D_RING, indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, indirect=True, ids=MESH_IDS)
def test_strided_reduce_scatter_async_collapsed_replicate(mesh_device, strict_ccl_topology):
    """{N}, [Replicate] reduce-scattered on dim 3 along the innermost axis: [Replicate, Shard(3)] on a 2-D mesh,
    {N}, [Shard(3)] on a line, composing to `cols` * T. Negative control: the union default kept {N}, [Replicate].
    The ring intermediate (return index 0, not surfaced here) keeps the input's label."""
    torch.manual_seed(28)
    rows, cols, _ = _mesh(mesh_device)
    full = _integers([1, 1, 64, 64 * cols])
    tt_input = _from_torch(full, mesh_device, ttnn.ReplicateTensorToMesh(mesh_device))
    _skip_unless_ring(tt_input, 1)

    tt_output = _strided_reduce_scatter_async(tt_input, mesh_device, dim=3, cluster_axis=1)

    if rows > 1:
        assert _placements(tt_output) == [REPLICATE, SHARD(3)]
        assert _dist_shape(tt_output) == (rows, cols)
    else:
        assert _placements(tt_output) == [SHARD(3)]
        assert _dist_shape(tt_output) == (cols,)
    assert torch.equal(_compose_by_label(tt_output), full * cols)


@pytest.mark.parametrize("device_params", FABRIC_1D_RING, indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, indirect=True, ids=MESH_IDS)
def test_strided_reduce_scatter_async_collapsed_shard_same_dim_inner_axis(mesh_device, strict_ccl_topology):
    """{N}, [Shard(3)] reduce-scattered on dim 3 along the innermost axis: the collapsed label again (rule (c)),
    composing to the concatenated row sums."""
    torch.manual_seed(29)
    rows, cols, num_devices = _mesh(mesh_device)
    full = _integers([1, 1, 64, 64 * num_devices * cols])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=3))
    _skip_unless_ring(tt_input, 1)

    tt_output = _strided_reduce_scatter_async(tt_input, mesh_device, dim=3, cluster_axis=1)

    assert _placements(tt_output) == [SHARD(3)]
    assert _dist_shape(tt_output) == (num_devices,)
    pieces = _pieces(full, num_devices)
    expected = torch.cat([sum(pieces[r * cols : (r + 1) * cols]) for r in range(rows)], dim=3)
    assert torch.equal(_compose_by_label(tt_output), expected)
