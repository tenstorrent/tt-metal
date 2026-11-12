# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Output TensorTopology labels of the collective family, and of mesh_partition, which shares the scatter rule, checked
against the bytes they describe.

Every test builds a mesh tensor whose label is either the collapsed 1-D form (``ShardTensorToMesh`` /
``ReplicateTensorToMesh``: ``{N}, [placement]``) or the N-D form (``ShardTensor2dMesh``: one placement per mesh axis),
runs one collective, and asserts two things about the result:

* the placements (and distribution shape) are the ones ``ttnn/operations/ccl/common/host/ccl_topology_utils`` promises;
* composing the per-device shards *by that label* (concatenate along a Shard axis, require identical bytes along a
  Replicate axis) gives the tensor the collective semantically produced. A label that over-claims Replicate fails the
  identical-bytes check; one that mislabels the Shard dim or order fails the equality with the reference.

Meshes: 1x2 (N300), 1x8 (T3K line), 2x4 (T3K / galaxy submesh), 8x4 (galaxy; the mesh_device fixture skips it on a
smaller box).
"""

import math

import pytest
import torch

import ttnn

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
    and the composed tensor is the original."""
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
    """all_gather_output_topology replicates only the gathered axis. ShardTensor2dMesh dims=(3, None) shards dim 3
    across rows and replicates across columns. Gathering dim 3 along the columns concatenates four identical copies
    of the row's chunk; the rows still hold distinct chunks, so the label stays [Shard(3), Replicate]."""
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
    """all_gather_output_topology's interleave refusal (require_contiguous_gather): {8},
    [Shard(3)] puts piece 4r+c on device (r, c); gathering dim 3 along the rows gives device (r, c) the pieces c and
    C+c side by side, which is no slice of any tensor: no label describes it."""
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
    """The interleave refusal applies only to the dim the input shards: gathering a dim the collapsed label does not
    shard leaves the dim-3 pieces where they are, so the outer axis is honest: [Replicate, Shard(3)], and column c
    composes to concat_2 of pieces c, C+c, ..."""
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
    label -- {N}, [Shard(3)]."""
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
    (no contiguity requirement)."""
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
    device with its slice of `cols` * T: [Replicate, Shard(3)] on a 2-D mesh, {N}, [Shard(3)] on a line (the helper
    writes the normalised dim, not -1)."""
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
    the sum back across the row -- row-major hierarchical sharding, i.e. the collapsed label again
    (reduce_scatter_output_topology's same-dim inner-axis collapse)."""
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
    """reduce_scatter_output_topology collapses a same-dim scatter along the inner axis. ShardTensor2dMesh
    dims=(3, None) holds chunk r of dim 3 on row r; reduce-scattering dim 3 along the columns splits each row's
    (summed) chunk into C -- device (r, c) holds piece Cr+c of C * T: the collapsed {N}, [Shard(3)]."""
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
    """reduce_scatter_output_topology refuses a same-dim scatter along the outer axis.
    ShardTensor2dMesh dims=(None, 3) holds chunk c on column c; reduce-scattering dim 3 along the rows gives device
    (r, c) part r of chunk c -- piece Rc+r, column-major, which no row-major label describes."""
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
# mesh_partition (a local scatter: every device keeps one piece of its own tensor, nothing moves between chips)
# ---------------------------------------------------------------------------------------------------------------------
#
# mesh_partition labels through the scatter rule reduce_scatter uses (mesh_partition_output_topology, the partition
# kind): Shard(dim) on the partitioned axis, the input's placement elsewhere, a same-dim Shard on the outer axis
# collapsing to the row-major {N}, [Shard(dim)]. Nothing is summed, so a Shard of another dim is kept rather than
# consumed, which is one refusal more than reduce_scatter has: neither the partitioned axis nor, for a whole-mesh
# partition, any non-trivial axis may already shard another dim. The op needs no fabric, so these tests take no
# device_params. Besides the label and the bytes it composes to, each test checks that device (r, c) holds its own
# chunk. The device-free twins are the MeshPartition cases of CclTopologyUtils in
# tests/ttnn/unit_tests/gtests/ccl/test_ccl_helpers.cpp.

_LINE_MESHES = [2, 8]
_LINE_MESH_IDS = ["1x2", "1x8"]


def _coords(tensor):
    return list(tensor.tensor_topology().mesh_coords())


def _shard_2d(full, mesh_device, dims):
    rows, cols, _ = _mesh(mesh_device)
    return _from_torch(full, mesh_device, ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=dims))


def _assert_partition_slices(tt_output, tt_input, dim, cluster_axis, mesh_device):
    """Device (r, c) holds chunk r (cluster_axis 0), c (cluster_axis 1) or r * cols + c (whole mesh) of its own input
    shard along `dim`, whatever the label says; get_device_tensors enumerates both tensors in row-major device order."""
    rows, cols, num_devices = _mesh(mesh_device)
    chunks = num_devices if cluster_axis is None else (rows, cols)[cluster_axis]
    out_shards, in_shards = _shards(tt_output), _shards(tt_input)
    assert len(out_shards) == len(in_shards) == num_devices
    for index, (out_shard, in_shard) in enumerate(zip(out_shards, in_shards)):
        r, c = divmod(index, cols)
        chunk = index if cluster_axis is None else (r if cluster_axis == 0 else c)
        assert torch.equal(out_shard, torch.chunk(in_shard, chunks, dim=dim)[chunk]), f"device ({r}, {c})"


@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
@pytest.mark.parametrize(
    "input_dims, dim, cluster_axis, expected",
    [
        ((2, None), 3, 1, [SHARD(2), SHARD(3)]),  # rows shard dim 2; partition dim 3 across the columns
        ((2, None), -1, 1, [SHARD(2), SHARD(3)]),  # the label carries the normalised dim, not -1
        ((None, 3), 2, 0, [SHARD(2), SHARD(3)]),  # columns shard dim 3; partition dim 2 across the rows
        ((None, None), 1, 1, [REPLICATE, SHARD(1)]),  # replicated N-D label: the other axis stays Replicate
        ((None, None), 1, 0, [SHARD(1), REPLICATE]),
    ],
    ids=["rows_shard_2", "rows_shard_2_negative_dim", "cols_shard_3", "replicated_axis_1", "replicated_axis_0"],
)
def test_mesh_partition_nd_label(mesh_device, input_dims, dim, cluster_axis, expected, strict_ccl_topology):
    """A ShardTensor2dMesh input partitioned along one axis: Shard(dim) on that axis, the input's placement on the
    other, over the input's coordinates. Composing by the label reassembles the input. The label is computed on the
    host launch path, so a program-cache hit must carry the same one."""
    torch.manual_seed(15)
    rows, cols, _ = _mesh(mesh_device)
    full = _integers([1, 8, 32 * rows, 32 * cols])
    tt_input = _shard_2d(full, mesh_device, input_dims)

    mesh_device.enable_program_cache()
    tt_output = ttnn.mesh_partition(tt_input, dim, cluster_axis=cluster_axis)
    cache_entries = mesh_device.num_program_cache_entries()

    assert _placements(tt_output) == expected
    assert _dist_shape(tt_output) == (rows, cols)
    assert _coords(tt_output) == _coords(tt_input)
    _assert_partition_slices(tt_output, tt_input, dim % full.ndim, cluster_axis, mesh_device)
    assert torch.equal(_compose_by_label(tt_output), full)

    tt_again = ttnn.mesh_partition(tt_input, dim, cluster_axis=cluster_axis)
    assert mesh_device.num_program_cache_entries() == cache_entries
    assert _placements(tt_again) == expected and _dist_shape(tt_again) == (rows, cols)


@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
@pytest.mark.parametrize("dim", [3, -1])
def test_mesh_partition_nd_same_dim_inner_axis_collapses(mesh_device, dim, strict_ccl_topology):
    """ShardTensor2dMesh dims=(3, None): row r holds coarse slice r of dim 3; partitioning dim 3 along the columns
    cuts each into fine slices, so device (r, c) holds slice r * cols + c -- row-major hierarchical sharding, which
    only the collapsed {N}, [Shard(3)] states (reduce_scatter's same-dim inner-axis collapse). With dim=-1 the input
    label carries Shard(-1) and the same-dim match normalises first."""
    torch.manual_seed(16)
    rows, cols, num_devices = _mesh(mesh_device)
    full = _integers([1, 1, 32, 32 * num_devices])
    tt_input = _shard_2d(full, mesh_device, (dim, None))

    tt_output = ttnn.mesh_partition(tt_input, dim, cluster_axis=1)

    assert _placements(tt_output) == [SHARD(3)]
    assert _dist_shape(tt_output) == (num_devices,)
    assert _coords(tt_output) == _coords(tt_input)
    _assert_partition_slices(tt_output, tt_input, 3, 1, mesh_device)
    assert torch.equal(_compose_by_label(tt_output), full)


@pytest.mark.parametrize("mesh_device", _LINE_MESHES, indirect=True, ids=_LINE_MESH_IDS)
def test_mesh_partition_nd_same_dim_on_a_size_1_axis_replicates(mesh_device, strict_ccl_topology):
    """ShardTensor2dMesh dims=(3, None) on a 1xN mesh labels the size-1 row axis Shard(3): one chunk, the whole
    extent. Partitioning dim 3 across the columns turns that axis into Replicate rather than leaving a second
    Shard(3): [Replicate, Shard(3)]."""
    torch.manual_seed(17)
    _, cols, _ = _mesh(mesh_device)
    full = _integers([1, 1, 32, 32 * cols])
    tt_input = _shard_2d(full, mesh_device, (3, None))
    assert _placements(tt_input) == [SHARD(3), REPLICATE]

    tt_output = ttnn.mesh_partition(tt_input, 3, cluster_axis=1)

    assert _placements(tt_output) == [REPLICATE, SHARD(3)]
    assert _dist_shape(tt_output) == (1, cols)
    assert _coords(tt_output) == _coords(tt_input)
    _assert_partition_slices(tt_output, tt_input, 3, 1, mesh_device)
    assert torch.equal(_compose_by_label(tt_output), full)


@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
@pytest.mark.parametrize("dim", [1, 3])
def test_mesh_partition_nd_replicated_whole_mesh_collapses(mesh_device, dim, strict_ccl_topology):
    """Whole mesh (cluster_axis=None): device r * cols + c keeps chunk r * cols + c of its tensor. An N-D label would
    need Shard(dim) on both axes, so a replicated N-D input yields the collapsed {N}, [Shard(dim)] over the input's
    row-major coordinates -- what ShardTensorToMesh(dim) would have produced -- and composing by it reassembles the
    input."""
    torch.manual_seed(18)
    _, _, num_devices = _mesh(mesh_device)
    full = _integers([1, num_devices, 32, 32 * num_devices])
    tt_input = _shard_2d(full, mesh_device, (None, None))

    tt_output = ttnn.mesh_partition(tt_input, dim, cluster_axis=None)

    assert _placements(tt_output) == [SHARD(dim)]
    assert _dist_shape(tt_output) == (num_devices,)
    assert _coords(tt_output) == _coords(tt_input)
    _assert_partition_slices(tt_output, tt_input, dim, None, mesh_device)
    assert torch.equal(_compose_by_label(tt_output), full)


@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
def test_mesh_partition_nd_whole_mesh_same_dim_shard_is_overwritten(mesh_device, strict_ccl_topology):
    """ShardTensor2dMesh dims=(3, None) gives [Shard(3), Replicate]; a whole-mesh partition of dim 3 yields the
    collapsed {N}, [Shard(3)]. Unlike the replicated input, the label is not exact against the input: device (r, c)
    holds chunk r * cols + c of its own row's slice, so the N pieces concatenate to each row's own pieces, 1/rows of
    the input's extent. A label says how the outputs assemble, not that they tile the input (the reduce_scatter
    stance); a Shard of another dim is refused instead (test_mesh_partition_refused_labels)."""
    torch.manual_seed(19)
    rows, cols, num_devices = _mesh(mesh_device)
    full = _integers([1, 1, 32, 32 * num_devices * rows])
    tt_input = _shard_2d(full, mesh_device, (3, None))

    tt_output = ttnn.mesh_partition(tt_input, 3, cluster_axis=None)

    assert _placements(tt_output) == [SHARD(3)]
    assert _dist_shape(tt_output) == (num_devices,)
    assert _coords(tt_output) == _coords(tt_input)
    _assert_partition_slices(tt_output, tt_input, 3, None, mesh_device)
    row_slices = _pieces(full, rows)
    expected = torch.cat(
        [_pieces(row_slices[r], num_devices)[r * cols + c] for r in range(rows) for c in range(cols)], dim=3
    )
    assert torch.equal(_compose_by_label(tt_output), expected)


@pytest.mark.parametrize("mesh_device", MESHES, indirect=True, ids=MESH_IDS)
@pytest.mark.parametrize("dim", [1, 3, -1])
def test_mesh_partition_collapsed_replicate_whole_mesh(mesh_device, dim, strict_ccl_topology):
    """ReplicateTensorToMesh gives {N}, [Replicate]; a whole-mesh partition yields {N}, [Shard(dim)] (the normalised
    dim) over the same coordinates. A [Replicate] label here would be data loss: the serialiser dedups a Replicate
    axis to one shard."""
    torch.manual_seed(20)
    _, _, num_devices = _mesh(mesh_device)
    full = _integers([1, num_devices, 32, 32 * num_devices])
    tt_input = _from_torch(full, mesh_device, ttnn.ReplicateTensorToMesh(mesh_device))

    tt_output = ttnn.mesh_partition(tt_input, dim, cluster_axis=None)

    assert _placements(tt_output) == [SHARD(dim % full.ndim)]
    assert _dist_shape(tt_output) == (num_devices,)
    assert _coords(tt_output) == _coords(tt_input)
    _assert_partition_slices(tt_output, tt_input, dim % full.ndim, None, mesh_device)
    assert torch.equal(_compose_by_label(tt_output), full)


@pytest.mark.parametrize("mesh_device", _LINE_MESHES, indirect=True, ids=_LINE_MESH_IDS)
@pytest.mark.parametrize("dim", [1, 3, -1])
def test_mesh_partition_collapsed_replicate_along_the_line(mesh_device, dim, strict_ccl_topology):
    """On a 1xN mesh with cluster_axis=1 the collapsed axis is the partitioned axis, so {N}, [Replicate] becomes
    {N}, [Shard(dim)] over the same coordinates, exactly as for the whole-mesh partition."""
    torch.manual_seed(21)
    _, _, num_devices = _mesh(mesh_device)
    full = _integers([1, num_devices, 32, 32 * num_devices])
    tt_input = _from_torch(full, mesh_device, ttnn.ReplicateTensorToMesh(mesh_device))

    tt_output = ttnn.mesh_partition(tt_input, dim, cluster_axis=1)

    assert _placements(tt_output) == [SHARD(dim % full.ndim)]
    assert _dist_shape(tt_output) == (num_devices,)
    assert _coords(tt_output) == _coords(tt_input)
    _assert_partition_slices(tt_output, tt_input, dim % full.ndim, 1, mesh_device)
    assert torch.equal(_compose_by_label(tt_output), full)


@pytest.mark.parametrize("mesh_device", _LINE_MESHES, indirect=True, ids=_LINE_MESH_IDS)
def test_mesh_partition_collapsed_shard_along_the_line_is_overwritten(mesh_device, strict_ccl_topology):
    """ShardTensorToMesh(dim=1) gives {N}, [Shard(1)]; partitioning dim 3 along the line (the label's only axis)
    relabels it {N}, [Shard(3)]: the partitioned axis takes the new placement whatever it held (the reduce_scatter
    stance, which tt-train relies on). Composing by the label gives the output bytes: device k's chunk k along dim 3
    of its own dim-1 slice. On a multi-axis mesh the same input is read per axis and refused instead
    (test_mesh_partition_refused_labels)."""
    torch.manual_seed(22)
    _, _, num_devices = _mesh(mesh_device)
    full = _integers([1, num_devices, 32, 32 * num_devices])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=1))

    tt_output = ttnn.mesh_partition(tt_input, 3, cluster_axis=1)

    assert _placements(tt_output) == [SHARD(3)]
    assert _dist_shape(tt_output) == (num_devices,)
    assert _coords(tt_output) == _coords(tt_input)
    _assert_partition_slices(tt_output, tt_input, 3, 1, mesh_device)
    dim1_slices = _pieces(full, num_devices, dim=1)
    expected = torch.cat([_pieces(dim1_slices[k], num_devices)[k] for k in range(num_devices)], dim=3)
    assert torch.equal(_compose_by_label(tt_output), expected)


@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
@pytest.mark.parametrize("dim", [1, 3])
@pytest.mark.parametrize("cluster_axis", [0, 1])
def test_mesh_partition_collapsed_replicate_uncollapses_on_a_2d_mesh(
    mesh_device, dim, cluster_axis, strict_ccl_topology
):
    """ReplicateTensorToMesh on a 2-D mesh gives {N}, [Replicate]. Partitioning along one axis leaves the tensor
    replicated along the other, which only an N-D label over the device mesh states; the collapsed row-major
    coordinates already enumerate that mesh and carry over. The deepseek_v3 mesh_partition path. Composing by the
    label reassembles the input on every line of the replicated axis."""
    torch.manual_seed(23)
    rows, cols, num_devices = _mesh(mesh_device)
    full = _integers([1, num_devices, 32, 32 * num_devices])
    tt_input = _from_torch(full, mesh_device, ttnn.ReplicateTensorToMesh(mesh_device))

    tt_output = ttnn.mesh_partition(tt_input, dim, cluster_axis=cluster_axis)

    expected = [REPLICATE, REPLICATE]
    expected[cluster_axis] = SHARD(dim)
    assert _placements(tt_output) == expected
    assert _dist_shape(tt_output) == (rows, cols)
    assert _coords(tt_output) == _coords(tt_input)
    _assert_partition_slices(tt_output, tt_input, dim, cluster_axis, mesh_device)
    assert torch.equal(_compose_by_label(tt_output), full)


@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
def test_mesh_partition_collapsed_shard_same_dim_inner_axis_collapses(mesh_device, strict_ccl_topology):
    """{N}, [Shard(3)] on a 2-D mesh reads per axis as row-major hierarchical sharding; partitioning dim 3 along
    the columns cuts each device's piece again in that order, so the result is the collapsed label once more
    (reduce_scatter's same-dim inner-axis collapse; along the rows it is refused, test_mesh_partition_refused_labels).
    Composing by the label gives the output bytes: device k's chunk k % cols of its own piece."""
    torch.manual_seed(24)
    _, cols, num_devices = _mesh(mesh_device)
    full = _integers([1, 1, 32, 32 * num_devices * cols])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=3))

    tt_output = ttnn.mesh_partition(tt_input, 3, cluster_axis=1)

    assert _placements(tt_output) == [SHARD(3)]
    assert _dist_shape(tt_output) == (num_devices,)
    assert _coords(tt_output) == _coords(tt_input)
    _assert_partition_slices(tt_output, tt_input, 3, 1, mesh_device)
    expected = torch.cat([_pieces(piece, cols)[k % cols] for k, piece in enumerate(_pieces(full, num_devices))], dim=3)
    assert torch.equal(_compose_by_label(tt_output), expected)


def _refused_partition(mesh_device, scenario):
    """Inputs whose partition has no honest label: (tt_input, dim, cluster_axis, fragment of the helper's reason)."""
    rows, cols, num_devices = _mesh(mesh_device)
    if scenario == "nd_same_dim_outer_axis":
        # Columns shard dim 3 (fine slices); partitioning dim 3 along the rows cuts each again, so device (r, c) holds
        # slice c * rows + r: column-major, which neither an N-D label (dim 3 on two axes) nor the row-major collapsed
        # label states.
        full = _integers([1, 1, 32, 32 * num_devices])
        return _shard_2d(full, mesh_device, (None, 3)), 3, 0, "no TensorTopology can express"
    if scenario == "nd_partitioned_axis_shards_another_dim":
        # Rows shard dim 3, columns shard dim 2; partitioning dim 3 along the columns leaves each device with its chunk
        # of dim 2 next to its new piece of dim 3: two dims on one axis.
        full = _integers([1, 1, 32 * cols, 32 * num_devices])
        return _shard_2d(full, mesh_device, (3, 2)), 3, 1, "already shards dim"
    if scenario == "nd_whole_mesh_other_shards":
        # Sharded on dims 0 and 1 (the nightly test_mesh_partition input); a whole-mesh partition of dim 3 leaves those
        # Shards next to the new piece, which no collapsed label states.
        full = _integers([rows, cols, 32, 32 * num_devices])
        return _shard_2d(full, mesh_device, (0, 1)), 3, None, "still sharded on dim"
    if scenario == "collapsed_shard_other_dim":
        # {N}, [Shard(1)] on a 2-D mesh reads per axis as [Shard(1), Shard(1)]; partitioning dim 3 along the columns
        # meets a Shard of another dim on the partitioned axis (on a line the same input is relabelled instead,
        # test_mesh_partition_collapsed_shard_along_the_line_is_overwritten).
        full = _integers([1, num_devices, 32, 32 * cols])
        return _from_torch(full, mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=1)), 3, 1, "already shards dim"
    if scenario == "collapsed_shard_same_dim_outer_axis":
        # {N}, [Shard(3)] partitioned along the rows: the outer axis would cut what the inner axis keeps apart.
        full = _integers([1, 1, 32, 32 * num_devices * rows])
        tt_input = _from_torch(full, mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=3))
        return tt_input, 3, 0, "no TensorTopology can express"
    raise AssertionError(scenario)


_REFUSED_PARTITIONS = [
    "nd_same_dim_outer_axis",
    "nd_partitioned_axis_shards_another_dim",
    "nd_whole_mesh_other_shards",
    "collapsed_shard_other_dim",
    "collapsed_shard_same_dim_outer_axis",
]


@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
@pytest.mark.parametrize("scenario", _REFUSED_PARTITIONS)
def test_mesh_partition_refused_labels(mesh_device, scenario, strict_ccl_topology, expect_error):
    """Partitions with no honest label (see _refused_partition): the partitioned axis, or any non-trivial axis of a
    whole-mesh partition, already shards another dim (kept, not consumed as a reduction would consume it), or a
    same-dim Shard sits on the inner axis. Strict mode makes each an error."""
    tt_input, dim, cluster_axis, message = _refused_partition(mesh_device, scenario)

    with expect_error(RuntimeError, message):
        ttnn.mesh_partition(tt_input, dim, cluster_axis=cluster_axis)


@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
@pytest.mark.parametrize("scenario", _REFUSED_PARTITIONS)
def test_mesh_partition_refused_labels_warn_only_keep_the_input_label(mesh_device, scenario, warn_only_ccl_topology):
    """The shipping default for the same inputs: a warning, and the output keeps the input's label (placements,
    shape and coordinates). The data is partitioned regardless."""
    tt_input, dim, cluster_axis, _ = _refused_partition(mesh_device, scenario)

    tt_output = ttnn.mesh_partition(tt_input, dim, cluster_axis=cluster_axis)

    assert _placements(tt_output) == _placements(tt_input)
    assert _dist_shape(tt_output) == _dist_shape(tt_input)
    assert _coords(tt_output) == _coords(tt_input)
    _assert_partition_slices(tt_output, tt_input, dim, cluster_axis, mesh_device)


def _forged_label(mesh_device, kind):
    """A replicated tensor whose label is overwritten (the hook reads only the label). "fewer_devices": {C}, [Shard(2)]
    over the first row of an R x C mesh. "sub_mesh": 1 x C [Replicate, Replicate] over that row, what a
    mesh_shape_override that fits per axis produces."""
    _, cols, num_devices = _mesh(mesh_device)
    tt_input = _from_torch(
        _integers([1, 1, 32, 32 * num_devices]), mesh_device, ttnn.ReplicateTensorToMesh(mesh_device)
    )
    block = _coords(tt_input)[:cols]
    if kind == "fewer_devices":
        label = ttnn.TensorTopology(ttnn.MeshShape([cols]), [ttnn.PlacementShard(2)], block)
    else:
        placements = [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()]
        label = ttnn.TensorTopology(ttnn.MeshShape([1, cols]), placements, block)
    tt_input.update_tensor_topology(label)
    return tt_input


_REFUSED_FORGED_LABELS = [
    ("fewer_devices", 1, "cover"),
    ("fewer_devices", None, "cover"),
    ("sub_mesh", 0, "devices the label does not describe"),
    ("sub_mesh", None, "devices the label does not describe"),
]
_REFUSED_FORGED_LABEL_IDS = [
    "fewer_devices_axis_1",
    "fewer_devices_whole_mesh",
    "sub_mesh_axis_0",
    "sub_mesh_whole_mesh",
]


@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
@pytest.mark.parametrize("kind, cluster_axis, message", _REFUSED_FORGED_LABELS, ids=_REFUSED_FORGED_LABEL_IDS)
def test_mesh_partition_label_not_covering_the_devices_is_refused(
    mesh_device, kind, cluster_axis, message, strict_ccl_topology, expect_error
):
    """A label over the first row of an R x C mesh describes C of the R * C devices the op partitions across, so no
    label can describe the output. {C}, [Shard(2)] with cluster_axis=1: the label's N equals the axis size by
    coincidence and the line rule must not fire; whole mesh: the rule must not emit {C}, [Shard(3)] for an R * C-piece
    output (a serialiser would save one row's pieces). The 1 x C sub-mesh label with cluster_axis=0 (the groups reach
    row 1) or over the whole mesh (the block holds C of R * C pieces) is refused for the same reason."""
    torch.manual_seed(25)
    tt_input = _forged_label(mesh_device, kind)

    with expect_error(RuntimeError, message):
        ttnn.mesh_partition(tt_input, 3, cluster_axis=cluster_axis)


@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
@pytest.mark.parametrize("kind, cluster_axis, _message", _REFUSED_FORGED_LABELS, ids=_REFUSED_FORGED_LABEL_IDS)
def test_mesh_partition_label_not_covering_the_devices_warn_only_keeps_the_input_label(
    mesh_device, kind, cluster_axis, _message, warn_only_ccl_topology
):
    """The shipping default for the same forged labels: the output keeps the input's label, and every device holds its
    chunk of the (replicated) input."""
    torch.manual_seed(25)
    tt_input = _forged_label(mesh_device, kind)

    tt_output = ttnn.mesh_partition(tt_input, 3, cluster_axis=cluster_axis)

    assert _placements(tt_output) == _placements(tt_input)
    assert _dist_shape(tt_output) == _dist_shape(tt_input)
    assert _coords(tt_output) == _coords(tt_input)
    _assert_partition_slices(tt_output, tt_input, 3, cluster_axis, mesh_device)


@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
def test_mesh_partition_sub_mesh_label_is_partitioned_within_its_block(mesh_device, strict_ccl_topology):
    """The 1 x C [Replicate, Replicate] label over row 0 with cluster_axis=1: the axis spans the mesh, so row 0 is a
    complete partition group and the output is labelled within the block, 1 x C [Replicate, Shard(3)] over the same C
    coordinates."""
    torch.manual_seed(25)
    _, cols, _ = _mesh(mesh_device)
    tt_input = _forged_label(mesh_device, "sub_mesh")

    tt_output = ttnn.mesh_partition(tt_input, 3, cluster_axis=1)

    assert _placements(tt_output) == [REPLICATE, SHARD(3)]
    assert _dist_shape(tt_output) == (1, cols)
    assert _coords(tt_output) == _coords(tt_input)
    _assert_partition_slices(tt_output, tt_input, 3, 1, mesh_device)


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
    from the original input -- so the branch is not observable here beyond the height that forces it. Along the rows
    the gather itself would be refused as interleaving; all_reduce runs its intermediates inside the helper's
    SilenceIntermediateRefusals because it sums the gathered pieces rather than keeping them and labels the result
    itself."""
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
    per column."""
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

    # The `num_devices` overload takes no cluster_axis (the ring is the whole mesh) and requires explicit semaphores.
    tt_output = ttnn.experimental.all_reduce_async(
        tt_input,
        num_devices=num_devices,
        barrier_semaphores=_semaphores(mesh_device, 2),
        rs_global_semaphores=_semaphores(mesh_device, 3),
        ag_global_semaphores=_semaphores(mesh_device, 2),
        math_op=ttnn.ReduceType.Sum,
        topology=ttnn.Topology.Linear,
    )

    assert _placements(tt_output) == [REPLICATE]
    assert _dist_shape(tt_output) == (num_devices,)
    expected = sum(_pieces(full, num_devices)) if sharded else full * num_devices
    assert torch.equal(_compose_by_label(tt_output), expected)


# ---------------------------------------------------------------------------------------------------------------------
# reduce_scatter (public ttnn.reduce_scatter: the prim ccl/reduce_scatter path, and the composite path that ends in it)
# ---------------------------------------------------------------------------------------------------------------------


def _reduce_scatter(tensor, mesh_device, dim, cluster_axis):
    """ttnn.reduce_scatter with the Linear topology, which keeps it off the one-shot reduce_scatter_minimal_direct
    op (an experimental op, labelled in #59589 rather than here). A tile-aligned per-device output takes the prim
    (ccl/reduce_scatter) path; a non-aligned one (test_reduce_scatter_composite_*) the composite split / pad / concat
    / prim / slice route, which ends in the same prim. `dim` may be negative: the wrapper normalises it before the
    prim sees it."""
    return ttnn.reduce_scatter(tensor, dim=dim, cluster_axis=cluster_axis, topology=ttnn.Topology.Linear)


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, indirect=True, ids=MESH_IDS)
def test_reduce_scatter_prim_collapsed_replicate(mesh_device, strict_ccl_topology):
    """ReplicateTensorToMesh gives {N}, [Replicate]. ttnn.reduce_scatter of dim -1 along the innermost axis leaves
    each device with its slice of `cols` * T: [Replicate, Shard(3)] on a 2-D mesh, {N}, [Shard(3)] on a line (the
    normalised dim is written). The second call is a program-cache hit (no new entry) and is labelled the same: the
    label does not depend on the cache."""
    torch.manual_seed(20)
    rows, cols, _ = _mesh(mesh_device)
    full = _integers([1, 1, 32, 32 * cols])
    tt_input = _from_torch(full, mesh_device, ttnn.ReplicateTensorToMesh(mesh_device))

    tt_output = _reduce_scatter(tt_input, mesh_device, dim=-1, cluster_axis=1)
    entries = mesh_device.num_program_cache_entries()
    tt_again = _reduce_scatter(tt_input, mesh_device, dim=-1, cluster_axis=1)
    assert mesh_device.num_program_cache_entries() == entries

    expected_placements = [REPLICATE, SHARD(3)] if rows > 1 else [SHARD(3)]
    expected_shape = (rows, cols) if rows > 1 else (cols,)
    assert _placements(tt_output) == expected_placements
    assert _placements(tt_again) == expected_placements
    assert _dist_shape(tt_output) == expected_shape
    assert torch.equal(_compose_by_label(tt_output), full * cols)
    assert torch.equal(_compose_by_label(tt_again), full * cols)


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, indirect=True, ids=MESH_IDS)
def test_reduce_scatter_composite_collapsed_replicate(mesh_device, strict_ccl_topology):
    """The composite ttnn.reduce_scatter path (composite_common::use_composite_reduce_scatter: a TILE input whose
    per-device output along dim 3 is not a tile multiple -- 16 wide here): split, pad, concat, prim reduce_scatter,
    slice. The prim call is labelled exactly as on the prim path and the slice / typecast tail passes its label
    through, so the composite result is labelled with no edit of its own: [Replicate, Shard(3)] on a 2-D mesh,
    {N}, [Shard(3)] on a line, bytes `cols` * T."""
    torch.manual_seed(29)
    rows, cols, _ = _mesh(mesh_device)
    full = _integers([1, 1, 32, 16 * cols])
    tt_input = _from_torch(full, mesh_device, ttnn.ReplicateTensorToMesh(mesh_device))

    tt_output = _reduce_scatter(tt_input, mesh_device, dim=3, cluster_axis=1)

    assert tuple(tt_output.shape) == (1, 1, 32, 16)  # a 16-wide TILE output: only the composite path produces it
    assert _placements(tt_output) == ([REPLICATE, SHARD(3)] if rows > 1 else [SHARD(3)])
    assert _dist_shape(tt_output) == ((rows, cols) if rows > 1 else (cols,))
    assert torch.equal(_compose_by_label(tt_output), full * cols)


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, indirect=True, ids=MESH_IDS)
def test_reduce_scatter_prim_collapsed_shard_same_dim_inner_axis(mesh_device, strict_ccl_topology):
    """{N}, [Shard(3)] reduce-scattered on dim 3 along the innermost axis: row r sums its `cols` pieces and splits
    the sum back across the row -- row-major hierarchical sharding, i.e. the collapsed label again
    (reduce_scatter_output_topology's same-dim inner-axis collapse), with the bytes to match. Under strict mode a
    refusal would raise, so this pins that the collapse rule produces the label and the bytes it describes."""
    torch.manual_seed(21)
    rows, cols, num_devices = _mesh(mesh_device)
    full = _integers([1, 1, 32, 32 * num_devices * cols])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=3))

    tt_output = _reduce_scatter(tt_input, mesh_device, dim=3, cluster_axis=1)

    assert _placements(tt_output) == [SHARD(3)]
    assert _dist_shape(tt_output) == (num_devices,)
    pieces = _pieces(full, num_devices)
    expected = torch.cat([sum(pieces[r * cols : (r + 1) * cols]) for r in range(rows)], dim=3)
    assert torch.equal(_compose_by_label(tt_output), expected)


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
def test_reduce_scatter_prim_nd_same_dim_inner_axis_collapses(mesh_device, strict_ccl_topology):
    """reduce_scatter_output_topology collapses a same-dim scatter along the inner axis. ShardTensor2dMesh
    dims=(3, None) holds chunk r of dim 3 on row r; reduce-scattering dim 3 along the columns splits each row's
    (summed) chunk into C -- device (r, c) holds piece Cr+c of C * T: the collapsed {N}, [Shard(3)]."""
    torch.manual_seed(22)
    rows, cols, num_devices = _mesh(mesh_device)
    full = _integers([1, 1, 32, 32 * num_devices])
    tt_input = _from_torch(
        full, mesh_device, ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(3, None))
    )

    tt_output = _reduce_scatter(tt_input, mesh_device, dim=3, cluster_axis=1)

    assert _placements(tt_output) == [SHARD(3)]
    assert _dist_shape(tt_output) == (num_devices,)
    assert torch.equal(_compose_by_label(tt_output), full * cols)


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
@pytest.mark.parametrize("cluster_axis", [0, 1])
def test_reduce_scatter_prim_nd_other_dim_keeps_other_axis(mesh_device, cluster_axis, strict_ccl_topology):
    """An N-D input sharded on one dim along one axis, reduce-scattered on a different dim along the other axis:
    the scattered axis takes Shard(scattered dim), the other axis keeps its Shard -- [Shard(2), Shard(3)] both ways
    (dims=(2, None) scattered on 3 along the columns; dims=(None, 3) scattered on 2 along the rows)."""
    torch.manual_seed(23)
    rows, cols, _ = _mesh(mesh_device)
    axis_size = (rows, cols)[cluster_axis]
    dims = (2, None) if cluster_axis == 1 else (None, 3)
    scatter_dim = 3 if cluster_axis == 1 else 2
    full = _integers([1, 1, 32 * rows, 32 * cols])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=dims))

    tt_output = _reduce_scatter(tt_input, mesh_device, dim=scatter_dim, cluster_axis=cluster_axis)

    assert _placements(tt_output) == [SHARD(2), SHARD(3)]
    assert _dist_shape(tt_output) == (rows, cols)
    assert torch.equal(_compose_by_label(tt_output), full * axis_size)


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
def test_reduce_scatter_prim_collapsed_shard_outer_axis_is_refused(mesh_device, strict_ccl_topology, expect_error):
    """reduce_scatter_output_topology refuses a same-dim scatter along the outer axis. {N},
    [Shard(3)] puts piece Cr+c on device (r, c); reduce-scattering dim 3 along the rows leaves device (r, c) with
    part r of its column's summed piece -- column-major, which no row-major label describes, so strict mode refuses
    before any device work."""
    torch.manual_seed(24)
    rows, _, num_devices = _mesh(mesh_device)
    tt_input = _from_torch(
        _integers([1, 1, 32, 32 * num_devices * rows]), mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=3)
    )

    with expect_error(RuntimeError, "no TensorTopology can express"):
        _reduce_scatter(tt_input, mesh_device, dim=3, cluster_axis=0)


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
def test_reduce_scatter_prim_collapsed_shard_outer_axis_warn_only_keeps_input_label(
    mesh_device, warn_only_ccl_topology
):
    """The default (warn-only) mode: the refused scatter still runs, logs (at error level: a scatter-family
    fallback can over-claim Replicate) and the output keeps the input's {N}, [Shard(3)], the hook's explicit
    fallback (not the framework's union default, which would also fold in a preallocated output's label). Here
    that label only under-claims (every device does hold a distinct piece, just not in row-major order), so the
    per-device bytes are checked directly: device (r, c) holds part r of the sum over the rows of column c's
    pieces."""
    torch.manual_seed(25)
    rows, cols, num_devices = _mesh(mesh_device)
    full = _integers([1, 1, 32, 32 * num_devices * rows])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=3))

    tt_output = _reduce_scatter(tt_input, mesh_device, dim=3, cluster_axis=0)

    assert _placements(tt_output) == [SHARD(3)]
    assert _dist_shape(tt_output) == (num_devices,)
    pieces = _pieces(full, num_devices)
    shards = _shards(tt_output)
    for r in range(rows):
        for c in range(cols):
            column_sum = sum(pieces[row * cols + c] for row in range(rows))
            assert torch.equal(shards[r * cols + c], _pieces(column_sum, rows)[r]), f"device ({r}, {c})"


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, indirect=True, ids=MESH_IDS)
def test_reduce_scatter_prim_whole_mesh(mesh_device, strict_ccl_topology):
    """cluster_axis=None. On a line the wrapper hands the prim the whole mesh: every device ends up with a distinct
    piece in coordinate order, the collapsed {N}, [Shard(3)]. On a 2-D mesh the wrapper runs one reduce_scatter per
    axis (cluster_axis 0, then 1): the first gives [Shard(3), Replicate], the second scatters dim 3 again along the
    inner axis and the helper collapses it to {N}, [Shard(3)] -- device (r, c) holds piece Cr+c of N * T."""
    torch.manual_seed(26)
    _, _, num_devices = _mesh(mesh_device)
    full = _integers([1, 1, 32, 32 * num_devices])
    tt_input = _from_torch(full, mesh_device, ttnn.ReplicateTensorToMesh(mesh_device))

    tt_output = _reduce_scatter(tt_input, mesh_device, dim=3, cluster_axis=None)

    assert _placements(tt_output) == [SHARD(3)]
    assert _dist_shape(tt_output) == (num_devices,)
    assert torch.equal(_compose_by_label(tt_output), full * num_devices)


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, indirect=True, ids=MESH_IDS)
def test_reduce_scatter_prim_preallocated_output_is_relabelled(mesh_device, strict_ccl_topology):
    """A caller-preallocated output (`output_tensor=`, the prim's optional_output_tensor): create_output_tensors
    returns that tensor itself, so the hook's label lands on the caller's object through the shared tensor
    attributes -- after the op it *is* the reduce_scatter result. {N}, [Replicate] in, a {N}, [Replicate] output
    buffer, dim 3 along the inner axis: the returned tensor and the caller's handle both read [Replicate, Shard(3)]
    ({N}, [Shard(3)] on a line) and compose to `cols` * T."""
    torch.manual_seed(30)
    rows, cols, _ = _mesh(mesh_device)
    full = _integers([1, 1, 32, 32 * cols])
    tt_input = _from_torch(full, mesh_device, ttnn.ReplicateTensorToMesh(mesh_device))
    tt_preallocated = _from_torch(
        torch.zeros([1, 1, 32, 32], dtype=torch.bfloat16), mesh_device, ttnn.ReplicateTensorToMesh(mesh_device)
    )

    tt_output = ttnn.reduce_scatter(
        tt_input, dim=3, cluster_axis=1, topology=ttnn.Topology.Linear, output_tensor=tt_preallocated
    )

    expected_placements = [REPLICATE, SHARD(3)] if rows > 1 else [SHARD(3)]
    expected_shape = (rows, cols) if rows > 1 else (cols,)
    for tensor in (tt_output, tt_preallocated):
        assert _placements(tensor) == expected_placements
        assert _dist_shape(tensor) == expected_shape
        assert torch.equal(_compose_by_label(tensor), full * cols)


# ---------------------------------------------------------------------------------------------------------------------
# broadcast (ccl/broadcast: one sender, every device on its line receives)
# ---------------------------------------------------------------------------------------------------------------------

LINE_MESHES = [2, 8]
LINE_MESH_IDS = ["1x2", "1x8"]


def _broadcast(tensor, sender, cluster_axis):
    row, col = sender
    return ttnn.broadcast(
        tensor, ttnn.MeshCoordinate(row, col), cluster_axis=cluster_axis, topology=ttnn.Topology.Linear
    )


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", LINE_MESHES, indirect=True, ids=LINE_MESH_IDS)
@pytest.mark.parametrize("cluster_axis", [1, None], ids=["axis_1", "whole_mesh"])
def test_broadcast_collapsed_shard_on_a_line(mesh_device, cluster_axis, strict_ccl_topology):
    """ttnn.broadcast copies the sender's shard to every other device of its line. {N}, [Shard(3)] in, with the last
    device as the sender: every device now holds piece N-1, so the label is {N}, [Replicate] -- along cluster_axis 1
    (the collapsed axis is the broadcast axis) and for the whole-mesh call alike (every placement Replicate,
    distribution shape kept)."""
    torch.manual_seed(27)
    _, _, num_devices = _mesh(mesh_device)
    full = _integers([1, 1, 32, 32 * num_devices])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=3))
    sender = num_devices - 1

    tt_output = _broadcast(tt_input, (0, sender), cluster_axis)

    assert _placements(tt_output) == [REPLICATE]
    assert _dist_shape(tt_output) == (num_devices,)
    assert torch.equal(_compose_by_label(tt_output), _pieces(full, num_devices)[sender])


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
@pytest.mark.parametrize("cluster_axis", [0, 1])
def test_broadcast_nd_along_one_axis_keeps_the_other(mesh_device, cluster_axis, strict_ccl_topology):
    """broadcast has one sender and signals only the devices on its line, so on a 2-D mesh the tensor spans that
    line: a (1, C) or (R, 1) sub-range (the nightly test_broadcast_2d pattern). ShardTensor2dMesh over it, sharding
    dim 3 along the broadcast axis and dim 2 on the trivial other axis; broadcasting from the last device makes the
    broadcast axis Replicate and keeps the other axis's placement verbatim: [Shard(2), Replicate] along the columns,
    [Replicate, Shard(2)] along the rows, with the sender's piece on every device."""
    torch.manual_seed(28)
    rows, cols, _ = _mesh(mesh_device)
    axis_size = (rows, cols)[cluster_axis]
    line_shape = (1, cols) if cluster_axis == 1 else (rows, 1)
    dims = (2, 3) if cluster_axis == 1 else (3, 2)
    full = _integers([1, 1, 32, 32 * axis_size])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=line_shape, dims=dims))
    sender = (0, axis_size - 1) if cluster_axis == 1 else (axis_size - 1, 0)

    tt_output = _broadcast(tt_input, sender, cluster_axis)

    expected = [SHARD(2), REPLICATE] if cluster_axis == 1 else [REPLICATE, SHARD(2)]
    assert _placements(tt_output) == expected
    assert _dist_shape(tt_output) == line_shape
    assert torch.equal(_compose_by_label(tt_output), _pieces(full, axis_size)[axis_size - 1])


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
def test_broadcast_misaligned_nd_label_is_refused(mesh_device, strict_ccl_topology, expect_error):
    """An N-D label whose axes are not the mesh's: `ShardTensor2dMesh(mesh_shape=(C, 1))` on an RxC mesh with C > R
    does not fit, so the mapper lays the C chunks out row-major over row 0 -- physically the (1, C) row the previous
    test broadcasts along, but labelled (C, 1), [Shard(3), Replicate]. A per-axis edit of that label keeps the Shard,
    claiming C distinct pieces where every device holds the sender's; the helper refuses it instead ("does not align
    with the axes"), an error under strict mode before any device work."""
    rows, cols, _ = _mesh(mesh_device)
    if cols <= rows:
        pytest.skip("needs more columns than rows so that a (cols, 1) mapper does not fit the mesh")
    full = _integers([1, 1, 32, 32 * cols])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(cols, 1), dims=(3, None)))
    assert _placements(tt_input) == [SHARD(3), REPLICATE]
    assert _dist_shape(tt_input) == (cols, 1)

    with expect_error(RuntimeError, "does not align with the axes"):
        _broadcast(tt_input, (0, cols - 1), cluster_axis=1)


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
def test_broadcast_misaligned_nd_label_warn_only_keeps_input_label(mesh_device, warn_only_ccl_topology):
    """The default (warn-only) mode for the label above: the refused broadcast still runs (its devices are row 0, a
    line), logs, and the output keeps the input's (C, 1), [Shard(3), Replicate] -- the hook's explicit fallback, not
    the framework's union. That label under-claims (every device holds the sender's piece), so the per-device bytes
    are checked directly rather than composed by it."""
    rows, cols, _ = _mesh(mesh_device)
    if cols <= rows:
        pytest.skip("needs more columns than rows so that a (cols, 1) mapper does not fit the mesh")
    full = _integers([1, 1, 32, 32 * cols])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(cols, 1), dims=(3, None)))

    tt_output = _broadcast(tt_input, (0, cols - 1), cluster_axis=1)

    assert _placements(tt_output) == [SHARD(3), REPLICATE]
    assert _dist_shape(tt_output) == (cols, 1)
    sender_piece = _pieces(full, cols)[cols - 1]
    for shard in _shards(tt_output):
        assert torch.equal(shard, sender_piece)


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


def _strided_reduce_scatter_async(tensor, mesh_device, dim, cluster_axis, persistent_output_buffers=None):
    # Each device's slice is 2 x 2 tiles: one matmul block (mm_block_ht x mm_block_wt) of the fused matmul the strided
    # reduce_scatter is the writer of; stand-alone it still needs that geometry. `persistent_output_buffers` is the
    # op's caller-owned [intermediate, output] pair.
    return ttnn.experimental.strided_reduce_scatter_async(
        tensor,
        persistent_output_buffers=persistent_output_buffers,
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


@pytest.mark.parametrize("device_params", FABRIC_1D_RING, indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, indirect=True, ids=MESH_IDS)
def test_strided_all_gather_async_collapsed_shard_inner_axis(mesh_device, strict_ccl_topology):
    """{N}, [Shard(3)] gathered on dim 3 along the innermost axis: [Shard(3), Replicate] on a 2-D mesh, {N},
    [Replicate] on a line, composing to the original."""
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


@pytest.mark.parametrize("device_params", FABRIC_1D_RING, indirect=True)
@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
def test_strided_all_gather_async_nd_input_replicates_only_the_gathered_axis(mesh_device, strict_ccl_topology):
    """ShardTensor2dMesh dims=(2, 3) gathered on dim 3 along the columns: [Shard(2), Replicate]; the rows' Shard(2) is
    untouched and the composed tensor is the original."""
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
    """all_gather's interleave refusal (require_contiguous_gather): gathering dim 3 of
    {N}, [Shard(3)] along the rows interleaves pieces c and C+c; the hook refuses before the op runs (so no ring is
    needed, nothing is skipped, and this runs on Blackhole too)."""
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
    row block and splits the columns, [Shard(2), Shard(3)]. Either way composing by the label gives the original."""
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
    composes with ConcatMeshToTensor(dim=3)."""
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
    """{N}, [Shard(2)] exchanged along the rows with in_dim=2 gathers row pieces c and C+c side by side -- the
    all_gather half of the label refuses as interleaving, "would interleave" (gather family: a warning in warn-only
    mode), before the op runs. This is a different refusal from the nightly's
    cluster_axis=0 cases: those exchange an N-D [Shard(in), Shard(out)], whose gather half succeeds
    ([Replicate, Shard(out)]) and whose scatter half then refuses -- out_dim already sharded on the inner
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
def test_all_to_all_async_whole_mesh(mesh_device, strict_ccl_topology, expect_error):
    """The whole-mesh all_to_all of {N}, [Shard(2)] with in_dim=2, out_dim=3 leaves device k with every row and
    column block k. Its program factory numbers the ring by MeshDeviceView::get_ring_devices() -- the walk around the
    mesh boundary -- which is the coordinate order only on a line, so: on 1x2 / 1x8 the label is {N}, [Shard(3)] and
    composing by it gives the original; on a 2-D mesh the hook refuses -- an error under strict mode, and in the default
    mode (next test) the input's {N}, [Shard(2)] kept -- because {N}, [Shard(3)] over row-major coordinates would put
    device (1, 0)'s piece 7 where piece 4 belongs, and no label can carry the boundary order. The caller's persistent output buffer (replicated
    zeros before the call) is the tensor relabelled either way. Shape: each device's input is 4 x 4N tiles so every
    destination shard is 4 x 4 tiles = eight two-tile packets; the op's factory statically partitions the
    intermediate buffer by packet id and rejects shards of a single packet (its final packet would overflow the
    buffer), which is what the nightly's shapes also satisfy."""
    torch.manual_seed(30)
    rows, cols, num_devices = _mesh(mesh_device)
    full = _integers([1, 1, 128 * num_devices, 128 * num_devices])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=2))
    _skip_unless_ring(tt_input, None)

    if rows > 1 and cols > 1:
        with expect_error(RuntimeError, "walk around the mesh boundary"):
            _all_to_all_async(tt_input, mesh_device, full.shape, in_dim=2, out_dim=3)
        return

    tt_output = _all_to_all_async(tt_input, mesh_device, full.shape, in_dim=2, out_dim=3)

    assert _placements(tt_output) == [SHARD(3)]
    assert _dist_shape(tt_output) == (num_devices,)
    assert torch.equal(_compose_by_label(tt_output), full)


@pytest.mark.parametrize("device_params", FABRIC_1D_RING, indirect=True)
@pytest.mark.parametrize("mesh_device", TWO_AXIS_MESHES, indirect=True, ids=TWO_AXIS_MESH_IDS)
def test_all_to_all_async_whole_mesh_2d_warn_only_keeps_input_label(mesh_device, warn_only_ccl_topology):
    """The default (warn-only) mode on a mesh with two non-trivial axes: the refused all_to_all still runs, logs (at
    error level: a scatter-family fallback) and the output -- the caller's persistent output buffer -- keeps the
    input's {N}, [Shard(2)]: the hook's explicit fallback, not the framework's union over the buffers' stale labels.
    That label under-claims (every device does hold a distinct piece, in boundary order), so nothing is composed by
    it. The refused op itself runs only where the boundary walk visits every device (2x4); with interior devices (8x4)
    its factory indexes past the boundary ring, so that mesh is skipped."""
    torch.manual_seed(30)
    rows, cols, num_devices = _mesh(mesh_device)
    if rows > 2 and cols > 2:
        pytest.skip(
            "the op's whole-mesh ring is the walk around the mesh boundary; a mesh with interior devices cannot run it"
        )
    full = _integers([1, 1, 128 * num_devices, 128 * num_devices])
    tt_input = _from_torch(full, mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=2))
    _skip_unless_ring(tt_input, None)

    tt_output = _all_to_all_async(tt_input, mesh_device, full.shape, in_dim=2, out_dim=3)

    assert _placements(tt_output) == [SHARD(2)]
    assert _dist_shape(tt_output) == (num_devices,)


# ---------------------------------------------------------------------------------------------------------------------
# reduce_scatter_minimal_direct (ring fabric)
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", FABRIC_1D_RING, indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, indirect=True, ids=MESH_IDS)
def test_reduce_scatter_minimal_direct_collapsed_replicate(mesh_device, strict_ccl_topology):
    """{N}, [Replicate] reduce-scattered on dim 3 along the innermost axis: [Replicate, Shard(3)] on a 2-D mesh,
    {N}, [Shard(3)] on a line, composing to `cols` * T. The staging buffer (return index 1, not surfaced here) keeps
    the input's label."""
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
    back across the row -- the collapsed label again (the helper's same-dim inner-axis collapse), composing to the
    concatenated row sums."""
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
    {N}, [Shard(3)] on a line, composing to `cols` * T. The ring intermediate (return index 0, not surfaced here) keeps
    the input's label."""
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
    """{N}, [Shard(3)] reduce-scattered on dim 3 along the innermost axis: the collapsed label again (the helper's
    same-dim inner-axis collapse), composing to the concatenated row sums."""
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


@pytest.mark.parametrize("device_params", FABRIC_1D_RING, indirect=True)
@pytest.mark.parametrize("mesh_device", [8], indirect=True, ids=["1x8"])
def test_strided_reduce_scatter_async_refused_label_falls_back_to_the_input_not_the_buffers(
    mesh_device, warn_only_ccl_topology
):
    """A refused label falls back to the INPUT's label, whatever the caller's persistent buffers carry (the convention
    ccl_topology_utils.hpp documents): the result and the caller's output handle (the same tensor) both read the
    input's (2, 4), [Shard(3), Replicate], not a union with the (1, 8), [Replicate, Shard(2)] the buffers' mapper left
    on them.

    The refusal: the input is mapped with a (2, 4) mesh_shape override on the 1x8, a label whose axes are not the
    mesh's (the mapper distributes it row-major over the line: devices 0-3 hold the first half of dim 3, 4-7 the
    second), which the helper refuses in every mode; warn-only, so the op still runs. Eight devices on a line are
    needed for that label, hence 1x8 only. The bytes are checked too: device d holds chunk d of the ring sum, four
    copies of each half."""
    torch.manual_seed(31)
    rows, cols, num_devices = _mesh(mesh_device)
    halves = _integers([1, 1, 64, 64 * num_devices * 2])
    tt_input = _from_torch(halves, mesh_device, ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(2, 4), dims=(3, None)))
    _skip_unless_ring(tt_input, 1)
    assert _placements(tt_input) == [SHARD(3), REPLICATE]
    assert _dist_shape(tt_input) == (2, 4)
    stale_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(None, 2))
    intermediate = _from_torch(
        torch.zeros([1, 1, 64 * num_devices, 64 * num_devices]).bfloat16(), mesh_device, stale_mapper
    )
    output = _from_torch(torch.zeros([1, 1, 64 * num_devices, 64]).bfloat16(), mesh_device, stale_mapper)
    assert _placements(output) == [REPLICATE, SHARD(2)]

    tt_output = _strided_reduce_scatter_async(
        tt_input, mesh_device, dim=3, cluster_axis=1, persistent_output_buffers=[intermediate, output]
    )

    for tensor in (tt_output, output):
        assert _placements(tensor) == [SHARD(3), REPLICATE]
        assert _dist_shape(tensor) == (2, 4)
    first, second = _pieces(halves, 2)
    for shard, piece in zip(_shards(tt_output), _pieces(4 * (first + second), num_devices)):
        assert torch.equal(shard, piece)
