# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TensorTopology labels produced by ttnn.mesh_partition.

Contract (``MeshPartitionDeviceOperation::compute_output_topologies``; the rule table lives in the op header and is
unit-tested without devices in ``tests/ttnn/unit_tests/gtests/ccl/test_mesh_partition_topology_rules.cpp``):

0. The label must describe the devices the partition acts on. A collapsed label has no axes and must cover the mesh.
   An N-D sub-mesh label (``mesh_shape_override`` that fits per axis) is kept, with rule 1 applied within its block,
   when the cluster axis spans the mesh along that axis so every partition group lies inside the block; otherwise
   (whole-mesh partition, or a shorter axis) the output keeps the input label with a warning.
1. N-D input label + ``cluster_axis``: ``Shard(dim)`` on the partitioned axis, the input's placement elsewhere. A
   tensor dim must never be sharded on two mesh axes (the N-D composer rejects it), so when another axis already
   shards ``dim``: a size-1 axis becomes ``Replicate`` (exact); an OUTER axis whose partitioned axis held
   ``Replicate`` or ``Shard(dim)`` collapses to ``{N}, [Shard(dim)]`` over the input's row-major coordinates
   (row-major hierarchical sharding, exact); anything else (inner axis, partitioned axis holding another Shard) is
   not expressible and the output keeps the input label (union default) with a warning.
2. ``cluster_axis=None`` (whole mesh, devices linearised row-major): ``{N}, [Shard(dim)]`` over the input's
   coordinates when every non-trivial axis of the input is ``Replicate`` or ``Shard(dim)``; otherwise the input label.
   Exact for a replicated input; a ``Shard(dim)`` already on a non-trivial axis is overwritten and the label describes
   the output bytes (the reduce_scatter_minimal_async stance).
3. Collapsed ``{N}, [p]`` input whose N equals both the mesh size and the cluster-axis size (1xN ring,
   ``cluster_axis=1``): ``{N}, [Shard(dim)]`` whatever ``p`` was. The guard identifies the label's only axis with the
   partitioned axis, so a ``Shard(k)`` is overwritten (reduce_scatter_minimal_async stance); rule 2 has no cluster axis
   to make that identification and falls back for a ``Shard(k)`` on a non-trivial axis.
4. Collapsed ``{N}, [Replicate]`` covering a multi-axis mesh: uncollapsed to the device mesh shape, ``Shard(dim)``
   on the partitioned axis and ``Replicate`` elsewhere.
5. A collapsed ``Shard`` label (``ShardTensorToMesh`` on a multi-axis mesh) partitioned along one axis: the input
   label, silently, i.e. the pre-existing behaviour (no exact label exists in either form).

Negative controls: before ``compute_output_topologies`` existed the framework's union default copied the input label
onto the output, so a replicated input stayed ``[Replicate]`` (the serialiser then dedups shards on that axis and
drops data) and a ``ShardTensor2dMesh`` input kept its pre-partition placements; every test asserting a changed label
fails on that code (``_assert_label_changed`` makes it explicit). The same-dim tests additionally fail on the first
version of the hook, which emitted ``[Shard(dim), Shard(dim)]`` (``_assert_no_duplicate_shard_dims`` pins that), and
the fewer-devices tests fail on the version whose rule 3 fired on any collapsed label of the cluster-axis size and
on the version whose whole-mesh rule relabelled a four-device label over an eight-device partition.

``ConcatMeshToTensor`` is not label-driven (it concatenates the device shards in device order); where it is used it
checks that the row-major device order a collapsed label records really reassembles the input.
"""

import pytest
import torch

import ttnn


def _placements_equal(actual, expected):
    # PlacementShard has no __eq__ binding, so compare structurally.
    if isinstance(expected, ttnn.PlacementShard):
        return isinstance(actual, ttnn.PlacementShard) and actual.dim == expected.dim
    return isinstance(expected, ttnn.PlacementReplicate) and isinstance(actual, ttnn.PlacementReplicate)


def _assert_topology(output, expected_shape, expected_placements, expected_coords):
    topology = output.tensor_topology()
    actual_shape = list(topology.distribution_shape())
    assert actual_shape == list(expected_shape), f"distribution shape {actual_shape} != {list(expected_shape)}"
    placements = list(topology.placements())
    assert len(placements) == len(expected_placements), f"{placements} vs {expected_placements}"
    for actual, expected in zip(placements, expected_placements):
        assert _placements_equal(actual, expected), f"{placements} != {expected_placements}"
    assert list(topology.mesh_coords()) == list(expected_coords), "mesh_coords must carry over from the input"


def _assert_label_changed(output, input_tensor):
    # Negative control: the union default (pre-change behaviour) would have produced exactly the input's label.
    assert output.tensor_topology() != input_tensor.tensor_topology()


def _assert_no_duplicate_shard_dims(output):
    # Negative control for the same-dim rules: the first version of the hook emitted [Shard(dim), Shard(dim)].
    shard_dims = [p.dim for p in output.tensor_topology().placements() if isinstance(p, ttnn.PlacementShard)]
    assert len(shard_dims) == len(set(shard_dims)), f"a tensor dim is sharded on two mesh axes: {shard_dims}"


def _assert_falls_back_to_input_label(output, input_tensor):
    """Documented fallback: the partitioned result has no exact TensorTopology, so the op returns {} and launch()
    keeps the union default, which for a single full-mesh input is the input's own label (same row-major coords)."""
    assert output.tensor_topology() == input_tensor.tensor_topology()
    _assert_no_duplicate_shard_dims(output)


def _assert_device_slices(output, input_tensor, dim, cluster_axis, mesh_shape):
    """Device (r, c) must hold chunk r (cluster_axis=0), c (cluster_axis=1) or r*cols+c (whole mesh) of its own
    input shard along ``dim``. Shards of both tensors are enumerated in the same row-major device order."""
    rows, cols = mesh_shape
    num_chunks = rows * cols if cluster_axis is None else mesh_shape[cluster_axis]
    output_shards = ttnn.get_device_tensors(output)
    input_shards = ttnn.get_device_tensors(input_tensor)
    coords = list(output.tensor_topology().mesh_coords())
    assert len(output_shards) == len(input_shards) == len(coords) == rows * cols
    for coord, out_shard, in_shard in zip(coords, output_shards, input_shards):
        r, c = coord[0], coord[1]
        index = r * cols + c if cluster_axis is None else (r if cluster_axis == 0 else c)
        expected = torch.chunk(ttnn.to_torch(in_shard), num_chunks, dim=dim)[index]
        assert torch.equal(ttnn.to_torch(out_shard), expected), f"data mismatch on device {coord}"


def _from_torch(torch_tensor, mesh_device, mesh_mapper):
    return ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )


def _shard_2d(torch_tensor, mesh_device, dims):
    mesh_shape = tuple(mesh_device.shape)
    return _from_torch(torch_tensor, mesh_device, ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims))


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4")], indirect=True)
@pytest.mark.parametrize(
    "input_dims, dim, cluster_axis, expected_placements",
    [
        # Rows shard dim 2, columns replicate; partition dim 3 across columns.
        ((2, None), 3, 1, [ttnn.PlacementShard(2), ttnn.PlacementShard(3)]),
        # Same, with a negative dim: the label must carry the normalised dim, not -1.
        ((2, None), -1, 1, [ttnn.PlacementShard(2), ttnn.PlacementShard(3)]),
        # Columns shard dim 3, rows replicate; partition dim 2 across rows.
        ((None, 3), 2, 0, [ttnn.PlacementShard(2), ttnn.PlacementShard(3)]),
        # Fully replicated N-D label; partition dim 1 across one axis, the other axis stays Replicate.
        ((None, None), 1, 1, [ttnn.PlacementReplicate(), ttnn.PlacementShard(1)]),
        ((None, None), 1, 0, [ttnn.PlacementShard(1), ttnn.PlacementReplicate()]),
    ],
)
def test_mesh_partition_2d_label(mesh_device, input_dims, dim, cluster_axis, expected_placements):
    mesh_shape = tuple(mesh_device.shape)
    torch.manual_seed(0)
    torch_input = torch.rand((1, 8, 64, 256), dtype=torch.bfloat16)
    tt_input = _shard_2d(torch_input, mesh_device, input_dims)
    input_coords = list(tt_input.tensor_topology().mesh_coords())

    mesh_device.enable_program_cache()
    tt_output = ttnn.mesh_partition(tt_input, dim, cluster_axis=cluster_axis)
    cache_entries = mesh_device.num_program_cache_entries()

    _assert_topology(tt_output, mesh_shape, expected_placements, input_coords)
    _assert_label_changed(tt_output, tt_input)
    _assert_no_duplicate_shard_dims(tt_output)
    _assert_device_slices(tt_output, tt_input, dim % torch_input.ndim, cluster_axis, mesh_shape)

    # The label is applied on the host launch path, so a program-cache hit must produce the same label.
    tt_output_cached = ttnn.mesh_partition(tt_input, dim, cluster_axis=cluster_axis)
    assert mesh_device.num_program_cache_entries() == cache_entries
    assert tt_output_cached.tensor_topology() == tt_output.tensor_topology()


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4")], indirect=True)
@pytest.mark.parametrize("dim", [3, -1])
def test_mesh_partition_same_dim_outer_axis_collapses(mesh_device, dim):
    """Rule 1(ii): rows already shard dim 3 (coarse slices), the partition across columns cuts each of them into
    fine slices, so device (r, c) holds slice r*cols+c of dim 3: row-major hierarchical sharding, which only the
    collapsed {8}, [Shard(3)] label states exactly. With dim=-1 the input label carries Shard(-1), so the same-dim
    match must normalise before comparing. Negative control: the first hook version emitted [Shard(3), Shard(3)]."""
    mesh_shape = tuple(mesh_device.shape)
    rows, cols = mesh_shape
    torch.manual_seed(0)
    torch_input = torch.rand((1, 8, 64, 256), dtype=torch.bfloat16)
    tt_input = _shard_2d(torch_input, mesh_device, (dim, None))
    input_coords = list(tt_input.tensor_topology().mesh_coords())
    normalised_dim = dim % torch_input.ndim

    tt_output = ttnn.mesh_partition(tt_input, dim, cluster_axis=1)

    _assert_topology(tt_output, (rows * cols,), [ttnn.PlacementShard(normalised_dim)], input_coords)
    _assert_label_changed(tt_output, tt_input)
    _assert_no_duplicate_shard_dims(tt_output)
    _assert_device_slices(tt_output, tt_input, normalised_dim, 1, mesh_shape)

    # Against the global tensor: device (r, c) holds slice r*cols+c of the input along dim, and the row-major device
    # order the collapsed label records reassembles the input.
    global_chunks = torch.chunk(torch_input, rows * cols, dim=normalised_dim)
    for coord, shard in zip(input_coords, ttnn.get_device_tensors(tt_output)):
        assert torch.equal(ttnn.to_torch(shard), global_chunks[coord[0] * cols + coord[1]]), f"device {coord}"
    composed = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=normalised_dim))
    assert torch.equal(composed, torch_input)


@pytest.mark.parametrize("mesh_device", [pytest.param(2, id="1x2"), pytest.param(8, id="1x8")], indirect=True)
def test_mesh_partition_same_dim_trivial_axis_replicates(mesh_device):
    """Rule 1(i): an N-D label over a 1xN mesh may carry Shard(3) on the size-1 row axis (one chunk = the whole
    extent). Partitioning dim 3 across the columns must turn that axis into Replicate, not leave a second Shard(3).
    Negative controls: the first hook version emitted [Shard(3), Shard(3)]; the union default kept
    [Shard(3), Replicate]."""
    mesh_shape = tuple(mesh_device.shape)
    assert mesh_shape[0] == 1
    torch.manual_seed(0)
    torch_input = torch.rand((1, 8, 64, 256), dtype=torch.bfloat16)
    tt_input = _shard_2d(torch_input, mesh_device, (3, None))
    input_coords = list(tt_input.tensor_topology().mesh_coords())
    assert _placements_equal(list(tt_input.tensor_topology().placements())[0], ttnn.PlacementShard(3))

    tt_output = ttnn.mesh_partition(tt_input, 3, cluster_axis=1)

    _assert_topology(tt_output, mesh_shape, [ttnn.PlacementReplicate(), ttnn.PlacementShard(3)], input_coords)
    _assert_label_changed(tt_output, tt_input)
    _assert_no_duplicate_shard_dims(tt_output)
    _assert_device_slices(tt_output, tt_input, 3, 1, mesh_shape)
    composed = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    assert torch.equal(composed, torch_input)


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4")], indirect=True)
def test_mesh_partition_same_dim_inner_axis_falls_back(mesh_device):
    """Rule 1(iii): columns shard dim 3 (fine slices first), the partition across rows cuts each again, so device
    (r, c) holds slice c*rows+r: column-major, which neither an N-D label (duplicate dim) nor the row-major collapsed
    label can state. The op keeps the input label and logs a warning (documented fallback). Negative control: the
    first hook version emitted [Shard(3), Shard(3)]. The data is still right."""
    mesh_shape = tuple(mesh_device.shape)
    torch.manual_seed(0)
    torch_input = torch.rand((1, 8, 64, 256), dtype=torch.bfloat16)
    tt_input = _shard_2d(torch_input, mesh_device, (None, 3))

    tt_output = ttnn.mesh_partition(tt_input, 3, cluster_axis=0)

    _assert_falls_back_to_input_label(tt_output, tt_input)
    _assert_device_slices(tt_output, tt_input, 3, 0, mesh_shape)


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4")], indirect=True)
def test_mesh_partition_same_dim_partitioned_axis_sharded_falls_back(mesh_device):
    """Rule 1(iii): rows shard dim 3 (outer axis, same dim) but the partitioned column axis already shards dim 2, so
    the result would need Shard(2) and Shard(3) on the columns at once. Not expressible: the op keeps the input
    label. Negative control: the first hook version emitted [Shard(3), Shard(3)]."""
    mesh_shape = tuple(mesh_device.shape)
    torch.manual_seed(0)
    torch_input = torch.rand((1, 8, 128, 256), dtype=torch.bfloat16)
    tt_input = _shard_2d(torch_input, mesh_device, (3, 2))

    tt_output = ttnn.mesh_partition(tt_input, 3, cluster_axis=1)

    _assert_falls_back_to_input_label(tt_output, tt_input)
    _assert_device_slices(tt_output, tt_input, 3, 1, mesh_shape)


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4")], indirect=True)
@pytest.mark.parametrize("dim", [1, 3])
def test_mesh_partition_2d_input_whole_mesh_collapses(mesh_device, dim):
    """Rule 2: cluster_axis=None hands device r*cols+c chunk r*cols+c. An N-D label would need Shard(dim) on both
    axes, which the composer rejects, so a replicated N-D input yields the collapsed {N}, [Shard(dim)] label over the
    same row-major coordinates (identical to what ShardTensorToMesh(dim) would have produced)."""
    mesh_shape = tuple(mesh_device.shape)
    num_devices = mesh_shape[0] * mesh_shape[1]
    torch.manual_seed(0)
    torch_input = torch.rand((1, 8, 64, 256), dtype=torch.bfloat16)
    tt_input = _shard_2d(torch_input, mesh_device, (None, None))
    input_coords = list(tt_input.tensor_topology().mesh_coords())

    tt_output = ttnn.mesh_partition(tt_input, dim, cluster_axis=None)

    _assert_topology(tt_output, (num_devices,), [ttnn.PlacementShard(dim)], input_coords)
    _assert_label_changed(tt_output, tt_input)
    _assert_device_slices(tt_output, tt_input, dim, None, mesh_shape)

    # ConcatMeshToTensor concatenates in device order, which is the row-major order the collapsed label records.
    composed = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=dim))
    assert torch.equal(composed, torch_input)


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4")], indirect=True)
def test_mesh_partition_whole_mesh_other_shard_falls_back(mesh_device):
    """Rule 2 restriction: the input is sharded on dims 0 and 1 (ShardTensor2dMesh dims=(0, 1), the nightly
    test_mesh_partition input); a whole-mesh partition of dim 3 leaves those Shards on the device next to the new
    slice, which no label can state. The op keeps the input label. Negative control: the first hook version emitted
    {8}, [Shard(3)], claiming the eight shards concatenate along dim 3 alone."""
    mesh_shape = tuple(mesh_device.shape)
    torch.manual_seed(0)
    torch_input = torch.rand((2, 8, 64, 256), dtype=torch.bfloat16)
    tt_input = _shard_2d(torch_input, mesh_device, (0, 1))

    tt_output = ttnn.mesh_partition(tt_input, 3, cluster_axis=None)

    _assert_falls_back_to_input_label(tt_output, tt_input)
    _assert_device_slices(tt_output, tt_input, 3, None, mesh_shape)


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4")], indirect=True)
def test_mesh_partition_whole_mesh_same_dim_shard_overwrites(mesh_device):
    """Rule 2 on an input already Shard(3) on a non-trivial axis: ShardTensor2dMesh(dims=(3, None)) gives
    [Shard(3), Replicate]; a whole-mesh partition of dim 3 yields the collapsed {8}, [Shard(3)]. Unlike the replicated
    input this label is not exact: device (r, c) holds chunk r*cols+c of its own row's half of dim 3, so the eight
    shards do not reassemble the input. The label describes the output bytes and overwrites the Shard(3) the input
    carried (the reduce_scatter_minimal_async stance); a Shard on another dim falls back instead
    (test_mesh_partition_whole_mesh_other_shard_falls_back). Negative control: the union default kept
    [Shard(3), Replicate]."""
    mesh_shape = tuple(mesh_device.shape)
    num_devices = mesh_shape[0] * mesh_shape[1]
    dim = 3
    torch.manual_seed(0)
    # 512 along dim 3: each row holds 256, and the whole-mesh partition leaves one 32-wide tile per device.
    torch_input = torch.rand((1, 8, 64, 512), dtype=torch.bfloat16)
    tt_input = _shard_2d(torch_input, mesh_device, (dim, None))
    input_coords = list(tt_input.tensor_topology().mesh_coords())
    assert _placements_equal(list(tt_input.tensor_topology().placements())[0], ttnn.PlacementShard(dim))

    tt_output = ttnn.mesh_partition(tt_input, dim, cluster_axis=None)

    _assert_topology(tt_output, (num_devices,), [ttnn.PlacementShard(dim)], input_coords)
    _assert_label_changed(tt_output, tt_input)
    _assert_no_duplicate_shard_dims(tt_output)
    # Per-device data: device (r, c) holds chunk r*cols+c of its own row's half.
    _assert_device_slices(tt_output, tt_input, dim, None, mesh_shape)
    # The semantic the label pins: concatenating the eight shards in label order gives the output bytes (each row's
    # own slices, half the input's extent), NOT the input's global view. A consumer that composes by this label gets
    # exactly that tensor.
    rows, cols = mesh_shape
    reassembled = torch.cat([ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tt_output)], dim=dim)
    assert reassembled.shape[dim] == torch_input.shape[dim] // rows
    row_halves = torch.chunk(torch_input, rows, dim=dim)
    expected = torch.cat(
        [torch.chunk(row_halves[r], num_devices, dim=dim)[r * cols + c] for r in range(rows) for c in range(cols)],
        dim=dim,
    )
    assert torch.equal(reassembled, expected)


@pytest.mark.parametrize("mesh_device", [pytest.param(2, id="1x2"), pytest.param(8, id="1x8")], indirect=True)
@pytest.mark.parametrize("dim", [1, 3, -1])
@pytest.mark.parametrize("cluster_axis", [1, None])
def test_mesh_partition_collapsed_replicate_on_ring(mesh_device, dim, cluster_axis):
    """Rules 3 and 2 on a ring: ReplicateTensorToMesh gives the collapsed {N}, [Replicate] label. With cluster_axis=1
    on a 1xN mesh the collapsed axis is the partitioned axis, so the honest label is {N}, [Shard(dim)] over the same
    coordinates. Before the change the output kept [Replicate], which the serialiser dedups (dropping slices)."""
    mesh_shape = tuple(mesh_device.shape)
    num_devices = mesh_shape[0] * mesh_shape[1]
    torch.manual_seed(0)
    torch_input = torch.rand((1, 8, 64, 256), dtype=torch.bfloat16)
    tt_input = _from_torch(torch_input, mesh_device, ttnn.ReplicateTensorToMesh(mesh_device))
    input_coords = list(tt_input.tensor_topology().mesh_coords())
    assert list(tt_input.tensor_topology().distribution_shape()) == [num_devices]
    assert _placements_equal(list(tt_input.tensor_topology().placements())[0], ttnn.PlacementReplicate())

    tt_output = ttnn.mesh_partition(tt_input, dim, cluster_axis=cluster_axis)

    normalised_dim = dim % torch_input.ndim
    _assert_topology(tt_output, (num_devices,), [ttnn.PlacementShard(normalised_dim)], input_coords)
    _assert_label_changed(tt_output, tt_input)
    _assert_device_slices(tt_output, tt_input, normalised_dim, cluster_axis, mesh_shape)

    composed = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=normalised_dim))
    assert torch.equal(composed, torch_input)


@pytest.mark.parametrize("mesh_device", [pytest.param(8, id="1x8")], indirect=True)
def test_mesh_partition_collapsed_shard_on_ring_overwrites(mesh_device):
    """Rule 3 on a collapsed Shard label: ShardTensorToMesh(dim=1) gives {8}, [Shard(1)]; partitioning dim 3 across
    the ring (cluster_axis=1, the label's only axis) relabels it {8}, [Shard(3)] over the same coordinates. This is
    the reduce_scatter_minimal_async stance (pinned by tt-train test_ccl_topology.py): the label describes the output
    bytes and the partitioned axis takes the collective's placement whatever it held. The ring guard (N == mesh size
    == cluster-axis size) is what identifies the label's axis with the partitioned axis; rule 2 (cluster_axis=None)
    has no such identification and falls back for this input. Negative control: the union default kept [Shard(1)]."""
    mesh_shape = tuple(mesh_device.shape)
    assert mesh_shape[0] == 1
    num_devices = mesh_shape[0] * mesh_shape[1]
    shard_dim, dim = 1, 3
    torch.manual_seed(0)
    torch_input = torch.rand((1, 8, 64, 256), dtype=torch.bfloat16)
    tt_input = _from_torch(torch_input, mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=shard_dim))
    input_coords = list(tt_input.tensor_topology().mesh_coords())
    assert list(tt_input.tensor_topology().distribution_shape()) == [num_devices]
    assert _placements_equal(list(tt_input.tensor_topology().placements())[0], ttnn.PlacementShard(shard_dim))

    tt_output = ttnn.mesh_partition(tt_input, dim, cluster_axis=1)

    _assert_topology(tt_output, (num_devices,), [ttnn.PlacementShard(dim)], input_coords)
    _assert_label_changed(tt_output, tt_input)
    _assert_no_duplicate_shard_dims(tt_output)
    _assert_device_slices(tt_output, tt_input, dim, 1, mesh_shape)


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4")], indirect=True)
@pytest.mark.parametrize("dim", [1, 3])
@pytest.mark.parametrize("cluster_axis", [0, 1])
def test_mesh_partition_collapsed_replicate_uncollapses_on_2d_mesh(mesh_device, dim, cluster_axis):
    """Rule 4: ReplicateTensorToMesh on a 2x4 mesh gives {8}, [Replicate]. Partitioning along one axis leaves the
    tensor replicated along the other, which only an N-D label over the device mesh can state; the collapsed
    row-major coordinates already enumerate that mesh, so they carry over. This is the deepseek_v3 mesh_partition
    path."""
    mesh_shape = tuple(mesh_device.shape)
    torch.manual_seed(0)
    torch_input = torch.rand((1, 8, 64, 256), dtype=torch.bfloat16)
    tt_input = _from_torch(torch_input, mesh_device, ttnn.ReplicateTensorToMesh(mesh_device))
    input_coords = list(tt_input.tensor_topology().mesh_coords())
    assert list(tt_input.tensor_topology().distribution_shape()) == [mesh_shape[0] * mesh_shape[1]]

    tt_output = ttnn.mesh_partition(tt_input, dim, cluster_axis=cluster_axis)

    expected_placements = [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()]
    expected_placements[cluster_axis] = ttnn.PlacementShard(dim)
    _assert_topology(tt_output, mesh_shape, expected_placements, input_coords)
    _assert_label_changed(tt_output, tt_input)
    _assert_device_slices(tt_output, tt_input, dim, cluster_axis, mesh_shape)

    # Concatenating along the partitioned axis reassembles the input on every line of the replicated axis.
    shards = {
        (coord[0], coord[1]): ttnn.to_torch(shard)
        for coord, shard in zip(tt_output.tensor_topology().mesh_coords(), ttnn.get_device_tensors(tt_output))
    }
    rows, cols = mesh_shape
    for other in range(cols if cluster_axis == 0 else rows):
        if cluster_axis == 0:
            line = [shards[(i, other)] for i in range(rows)]
        else:
            line = [shards[(other, j)] for j in range(cols)]
        assert torch.equal(torch.cat(line, dim=dim), torch_input)


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4")], indirect=True)
def test_mesh_partition_collapsed_replicate_whole_mesh(mesh_device):
    """Rule 2 on a collapsed replicated input: {8}, [Replicate] -> {8}, [Shard(dim)]."""
    mesh_shape = tuple(mesh_device.shape)
    num_devices = mesh_shape[0] * mesh_shape[1]
    dim = 3
    torch.manual_seed(0)
    torch_input = torch.rand((1, 8, 64, 256), dtype=torch.bfloat16)
    tt_input = _from_torch(torch_input, mesh_device, ttnn.ReplicateTensorToMesh(mesh_device))
    input_coords = list(tt_input.tensor_topology().mesh_coords())

    tt_output = ttnn.mesh_partition(tt_input, dim, cluster_axis=None)

    _assert_topology(tt_output, (num_devices,), [ttnn.PlacementShard(dim)], input_coords)
    _assert_label_changed(tt_output, tt_input)
    _assert_device_slices(tt_output, tt_input, dim, None, mesh_shape)
    composed = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=dim))
    assert torch.equal(composed, torch_input)


def _assert_partitioned_data(tt_output, torch_input, dim, cluster_axis, mesh_shape):
    """The data is partitioned across the mesh regardless of the label: device index i (row-major) holds chunk
    i % cols (cluster_axis=1), i // cols (cluster_axis=0) or i (whole mesh) of the replicated input."""
    rows, cols = mesh_shape
    num_chunks = rows * cols if cluster_axis is None else mesh_shape[cluster_axis]
    chunks = torch.chunk(torch_input, num_chunks, dim=dim)
    for index, shard in enumerate(ttnn.get_device_tensors(tt_output)):
        chunk = index if cluster_axis is None else (index // cols if cluster_axis == 0 else index % cols)
        assert torch.equal(ttnn.to_torch(shard), chunks[chunk]), f"data mismatch on device index {index}"


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4")], indirect=True)
@pytest.mark.parametrize("cluster_axis", [1, None])
def test_mesh_partition_fewer_shards_collapsed_label_falls_back(mesh_device, cluster_axis):
    """Rule 0: a {4}, [Shard(2)] label on a 2x4 mesh covers four of the eight devices the op partitions across, so no
    label can describe the output and the op keeps the input label. With cluster_axis=1 the label's N equals the
    cluster-axis size by coincidence (rule 3 must not fire); with cluster_axis=None the whole-mesh rule must not emit
    {4}, [Shard(3)] for an eight-slice output (a serialiser would then save four of the eight slices). The label is
    forged with update_tensor_topology on a fully replicated tensor because the hook reads only the label. Negative
    controls: the first hook version returned {4}, [Shard(3)] for cluster_axis=1, the second for cluster_axis=None.
    The union default keeps the output's own (full-mesh) coordinates, so only shape and placements are compared."""
    mesh_shape = tuple(mesh_device.shape)
    rows, cols = mesh_shape
    torch.manual_seed(0)
    torch_input = torch.rand((1, 8, 64, 256), dtype=torch.bfloat16)
    tt_input = _from_torch(torch_input, mesh_device, ttnn.ReplicateTensorToMesh(mesh_device))
    input_coords = list(tt_input.tensor_topology().mesh_coords())
    tt_input.update_tensor_topology(
        ttnn.TensorTopology(ttnn.MeshShape([cols]), [ttnn.PlacementShard(2)], input_coords[:cols])
    )
    assert list(tt_input.tensor_topology().distribution_shape()) == [cols]

    tt_output = ttnn.mesh_partition(tt_input, 3, cluster_axis=cluster_axis)

    output_topology = tt_output.tensor_topology()
    assert list(output_topology.distribution_shape()) == [cols]
    output_placements = list(output_topology.placements())
    assert len(output_placements) == 1 and _placements_equal(output_placements[0], ttnn.PlacementShard(2))
    _assert_partitioned_data(tt_output, torch_input, 3, cluster_axis, mesh_shape)


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4")], indirect=True)
@pytest.mark.parametrize("cluster_axis", [1, 0, None])
def test_mesh_partition_sub_mesh_nd_label(mesh_device, cluster_axis):
    """Rule 0 for an N-D sub-mesh label: a 1x4 [Replicate, Replicate] label over row 0 of a 2x4 mesh (what a
    mesh_shape_override that fits per axis produces). cluster_axis=1 spans the mesh along that axis, so row 0 is a
    complete partition group and rule 1 applies within the block: 1x4 [Replicate, Shard(3)] over the same four
    coordinates. cluster_axis=0 (groups reach row 1) and cluster_axis=None (the block holds four of eight slices)
    have no honest label and keep the input label. Negative controls: the first hook version emitted {4}, [Shard(3)]
    for the whole-mesh case; the second fell back for cluster_axis=1 as well."""
    mesh_shape = tuple(mesh_device.shape)
    rows, cols = mesh_shape
    torch.manual_seed(0)
    torch_input = torch.rand((1, 8, 64, 256), dtype=torch.bfloat16)
    tt_input = _from_torch(torch_input, mesh_device, ttnn.ReplicateTensorToMesh(mesh_device))
    input_coords = list(tt_input.tensor_topology().mesh_coords())
    tt_input.update_tensor_topology(
        ttnn.TensorTopology(
            ttnn.MeshShape([1, cols]), [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], input_coords[:cols]
        )
    )
    assert list(tt_input.tensor_topology().distribution_shape()) == [1, cols]

    tt_output = ttnn.mesh_partition(tt_input, 3, cluster_axis=cluster_axis)

    if cluster_axis == 1:
        _assert_topology(tt_output, (1, cols), [ttnn.PlacementReplicate(), ttnn.PlacementShard(3)], input_coords[:cols])
        _assert_label_changed(tt_output, tt_input)
    else:
        output_topology = tt_output.tensor_topology()
        assert list(output_topology.distribution_shape()) == [1, cols]
        output_placements = list(output_topology.placements())
        assert len(output_placements) == 2 and all(isinstance(p, ttnn.PlacementReplicate) for p in output_placements)
    _assert_partitioned_data(tt_output, torch_input, 3, cluster_axis, mesh_shape)
