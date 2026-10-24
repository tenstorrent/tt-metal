# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Mesh-topology labels of caller-owned outputs for the data-movement in-place ops.

Every device op relabels the tensors it returns in ``launch()``. Unless the op defines
``compute_output_topologies`` that label is the *union* of all tensor inputs, where any Shard
placement beats Replicate. Four ops here hand back a tensor the caller already owns and label it by
the rule shared with the in-place norm ops (``ttnn/operations/core/caller_owned_topology.hpp``,
also used by PR #59331):

* ``sharded_to_interleaved_partial`` / ``slice_write`` write a slice into the caller's cache/output.
  The caller's label stays while it still describes the data: an input that is replicated, or
  sharded only along mesh axes the output is sharded along too, leaves the output's distribution as
  labelled. An input sharded along a mesh axis on which the output is replicated writes a different
  slice on every device there, so the result takes the union (the input's label) -- on the returned
  handle and, because it aliases the caller's storage, on the caller's handle. A ``Replicate`` label
  kept there would make the flatbuffer serialiser deduplicate shards that differ.
* ``copy`` / ``assign`` / ``typecast`` with a preallocated output overwrite dst on every mesh
  coordinate src occupies. When src and dst span the same *set* of coordinates (any order) dst now
  holds src's shard on every device and adopts src's topology. When src spans a strict subset of
  dst's coordinates only those devices are rewritten: the shared rule declines, the result takes the
  union on a new handle restricted to src's coordinates, and the caller's dst handle keeps its label.

Each test states in its docstring what the label was before the fix (the negative control) and,
where the op runs twice, asserts the program cache is untouched by the second dispatch: topology is
not part of the program hash, so the relabelling must not cost a recompile.

Neither branch of the copy / typecast coverage decision can be driven from Python beyond the
full-coverage case these tests use. A full-mesh src whose ``mesh_coords()`` list the same devices as
dst in a different order -- what the set comparison exists for -- cannot be built: every mesh mapper
emits its coordinates in row-major order of the region it maps
(``ttnn/core/distributed/distributed_tensor.cpp``), so on a 1x2 mesh the only full-mesh order is
``[(0, 0), (0, 1)]``. A src whose label spans a strict subset of dst's coordinates cannot be built
either: a tensor restricted to some coordinates (``ttnn.get_device_tensors(t)[i]``, or what
``launch()`` hands back after a partial dispatch) shares its parent's ``TensorTopology`` and so lists
the whole mesh, and a host tensor mapped onto one device with ``mesh_shape_override=MeshShape([1])``
reaches the mesh with one shard in its storage but a label listing every coordinate (the
host-to-mesh write path labels a single-shard upload as replicated over the whole mesh --
pre-existing, not touched here). The partial branch is therefore defensive today; it is exercised
by reasoning in the hook comments, not by a test.
"""

import pytest
import torch

import ttnn

MESH_PARAMS = pytest.mark.parametrize("mesh_device", [2], indirect=True)


def _placement_names(topology):
    return [f"Shard({p.dim})" if isinstance(p, ttnn.PlacementShard) else "Replicate" for p in topology.placements()]


def _mesh_coords(topology):
    return [tuple(coord) for coord in topology.mesh_coords()]


def _from_torch(torch_tensor, mesh_device, mesh_mapper, **kwargs):
    kwargs.setdefault("dtype", ttnn.bfloat16)
    kwargs.setdefault("layout", ttnn.TILE_LAYOUT)
    kwargs.setdefault("memory_config", ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.from_torch(torch_tensor, device=mesh_device, mesh_mapper=mesh_mapper, **kwargs)


def _replicated(torch_tensor, mesh_device, **kwargs):
    return _from_torch(torch_tensor, mesh_device, ttnn.ReplicateTensorToMesh(mesh_device), **kwargs)


def _sharded(torch_tensor, mesh_device, dim, **kwargs):
    return _from_torch(torch_tensor, mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=dim), **kwargs)


def _shard0_per_mesh_axis(mesh_device):
    """Dim-0 sharding declared with one placement per mesh axis: Shard(0) on the axis that has more than one device.

    Describes the same distribution as ``ShardTensorToMesh(dim=0)``'s collapsed 1-D label; only the label's rank
    differs.
    """
    return [ttnn.PlacementShard(0) if size > 1 else ttnn.PlacementReplicate() for size in mesh_device.shape]


# How a tensor is distributed over the mesh in the partial-write tests.
REPLICATED = "replicated"  # ReplicateTensorToMesh: collapsed 1-D label [Replicate]
SHARD0 = "shard0"  # ShardTensorToMesh(dim=0): collapsed 1-D label [Shard(0)]
SHARD0_2D = "shard0_2d"  # same data, one placement per mesh axis, e.g. [Replicate, Shard(0)] on a 1x2 mesh


def _distribute(torch_tensor, mesh_device, label, **kwargs):
    if label == REPLICATED:
        return _replicated(torch_tensor, mesh_device, **kwargs)
    if label == SHARD0:
        return _sharded(torch_tensor, mesh_device, 0, **kwargs)
    assert label == SHARD0_2D
    mapper = ttnn.create_mesh_mapper(mesh_device, ttnn.MeshMapperConfig(_shard0_per_mesh_axis(mesh_device)))
    return _from_torch(torch_tensor, mesh_device, mapper, **kwargs)


def _global_shape(label, per_device_shape, num_devices):
    return per_device_shape if label == REPLICATED else [num_devices] + per_device_shape[1:]


def _per_device_slice(label, torch_tensor, device_index):
    return torch_tensor if label == REPLICATED else torch_tensor[device_index : device_index + 1]


def _per_device(tt_tensor):
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tt_tensor)]


def _assert_labels(tt_tensor, expected_names, what):
    names = _placement_names(tt_tensor.tensor_topology())
    assert names == expected_names, f"{what}: expected placements {expected_names}, got {names}"


def _reset_program_cache(mesh_device):
    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()


# (input label, caller's cache/output label, whether the caller's label is replaced by the union). The union case
# is the only one in which the input is sharded along a mesh axis the caller's tensor is replicated on.
_PARTIAL_WRITE_CASES = [
    pytest.param(REPLICATED, REPLICATED, False, id="replicated_input_keeps_label"),
    pytest.param(SHARD0, REPLICATED, True, id="sharded_input_into_replicated_output_takes_union"),
    pytest.param(SHARD0, SHARD0, False, id="sharded_input_into_output_sharded_on_the_same_axis_keeps_label"),
    pytest.param(REPLICATED, SHARD0, False, id="replicated_input_into_sharded_output_keeps_label"),
    pytest.param(SHARD0_2D, SHARD0, False, id="rank_mixed_input_sharded_on_the_same_axis_keeps_label"),
]


# ---------------------------------------------------------------------------
# sharded_to_interleaved_partial
# ---------------------------------------------------------------------------


@MESH_PARAMS
@pytest.mark.parametrize("input_label, cache_label, takes_union", _PARTIAL_WRITE_CASES)
def test_sharded_to_interleaved_partial_label_follows_the_data(mesh_device, input_label, cache_label, takes_union):
    """Per-slice writes from the input into the caller's cache; the cache's label follows the data.

    Compatible cases keep the cache's own label: a replicated input; an input sharded over the mesh on the same
    axis as the cache, whether its label is the collapsed 1-D form or one placement per mesh axis (the union would
    relabel the 1-D cache with the input's 2-D form); and a replicated input into a sharded cache. A dim-0-sharded
    input into a Replicate cache writes a different slice on every device, so the cache takes the union -- the
    input's ``Shard(0)`` -- on the caller's handle as well as the returned one. The second slice write then finds
    a cache sharded on the same axis and keeps that label.

    Negative control: the previous revision kept the cache's label unconditionally, leaving ``Replicate`` on a
    cache whose shards differ (data loss on serialisation); before this PR the union relabelled the cache on every
    write, which is wrong for the rank-mixed compatible case.
    """
    num_devices = mesh_device.get_num_devices()
    num_slices = 2
    H, W = 64, 64
    per_device_shape = [1, 1, H, W]
    grid = mesh_device.compute_with_storage_grid_size()
    grid_size = (grid.x, grid.y)
    height_shard_spec = [H // num_slices, W]

    def run(seed):
        torch.manual_seed(seed)
        in0 = torch.randn(_global_shape(input_label, per_device_shape, num_devices)).bfloat16()
        in0_t = _distribute(in0, mesh_device, input_label, memory_config=ttnn.L1_MEMORY_CONFIG)
        cache = torch.zeros(_global_shape(cache_label, per_device_shape, num_devices)).bfloat16()
        cache_t = _distribute(cache, mesh_device, cache_label, memory_config=ttnn.L1_MEMORY_CONFIG)
        cache_names_before = _placement_names(cache_t.tensor_topology())
        input_names = _placement_names(in0_t.tensor_topology())
        # The union replaces the cache's label exactly when the input is sharded on an axis the cache is not.
        assert takes_union == (input_label != REPLICATED and cache_label == REPLICATED)
        expected_names = input_names if takes_union else cache_names_before
        expected_topology = in0_t.tensor_topology() if takes_union else cache_t.tensor_topology()

        for slice_index in range(num_slices):
            in0_slice = ttnn.interleaved_to_sharded_partial(
                in0_t,
                grid_size,
                height_shard_spec,
                num_slices,
                slice_index,
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            _assert_labels(in0_slice, input_names, "precondition: the slice carries the input's label")
            returned = ttnn.sharded_to_interleaved_partial(
                in0_slice, cache_t, num_slices, slice_index, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            _assert_labels(cache_t, expected_names, f"cache after slice {slice_index} (caller's handle)")
            _assert_labels(returned, expected_names, f"cache after slice {slice_index} (returned handle)")
            assert cache_t.tensor_topology() == expected_topology
            assert returned.tensor_topology() == cache_t.tensor_topology()
            assert returned.buffer_address() == cache_t.buffer_address()

        # Each device's cache now holds the slices that device was given.
        for device_index, cache_shard in enumerate(_per_device(cache_t)):
            assert torch.equal(cache_shard, _per_device_slice(input_label, in0, device_index))
        return in0_t, cache_t

    _reset_program_cache(mesh_device)
    keep_alive = [run(1)]
    entries = mesh_device.num_program_cache_entries()
    assert entries > 0
    keep_alive.append(run(2))
    assert mesh_device.num_program_cache_entries() == entries, "relabelling must not miss the program cache"


# ---------------------------------------------------------------------------
# experimental.slice_write
# ---------------------------------------------------------------------------


@MESH_PARAMS
@pytest.mark.parametrize("input_memory_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("input_label, output_label, takes_union", _PARTIAL_WRITE_CASES)
def test_slice_write_label_follows_the_data(mesh_device, input_memory_config, input_label, output_label, takes_union):
    """A slice of the caller's row-major output is written from the input; the output's label follows the data.

    Same cases as for sharded_to_interleaved_partial: compatible inputs (replicated, or sharded over the mesh on
    the same axis as the output, in either label rank) keep the output's own label; a dim-0-sharded input into a
    Replicate output writes a different slice on every device, so the output takes the union -- the input's
    ``Shard(0)`` -- on both handles. Row-major on both sides so the wrapper hands the caller's handle straight to
    the device op (a TILE output would first be converted to row-major, which allocates a new tensor).

    Negative control: the previous revision kept the output's label unconditionally (``Replicate`` over shards
    that differ); before this PR the union relabelled the output on every write.
    """
    num_devices = mesh_device.get_num_devices()
    out_shape = [1, 1, 64, 64]
    in_shape = [1, 1, 32, 64]
    start = [0, 0, 0, 0]
    end = in_shape
    step = [1, 1, 1, 1]

    def run(seed):
        torch.manual_seed(seed)
        src = torch.randn(_global_shape(input_label, in_shape, num_devices)).bfloat16()
        src_t = _distribute(
            src, mesh_device, input_label, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=input_memory_config
        )
        out = torch.zeros(_global_shape(output_label, out_shape, num_devices)).bfloat16()
        out_t = _distribute(out, mesh_device, output_label, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_names_before = _placement_names(out_t.tensor_topology())
        input_names = _placement_names(src_t.tensor_topology())
        # The union replaces the output's label exactly when the input is sharded on an axis the output is not.
        assert takes_union == (input_label != REPLICATED and output_label == REPLICATED)
        expected_names = input_names if takes_union else output_names_before
        expected_topology = src_t.tensor_topology() if takes_union else out_t.tensor_topology()

        returned = ttnn.experimental.slice_write(src_t, out_t, start, end, step)

        _assert_labels(out_t, expected_names, "output after slice_write (caller's handle)")
        _assert_labels(returned, expected_names, "output after slice_write (returned handle)")
        assert out_t.tensor_topology() == expected_topology
        assert returned.tensor_topology() == out_t.tensor_topology()
        assert returned.buffer_address() == out_t.buffer_address()
        for device_index, out_shard in enumerate(_per_device(out_t)):
            expected = torch.zeros(out_shape).bfloat16()
            expected[:, :, : in_shape[2], :] = _per_device_slice(input_label, src, device_index)
            assert torch.equal(out_shard, expected)
        return src_t, out_t

    _reset_program_cache(mesh_device)
    keep_alive = [run(1)]
    entries = mesh_device.num_program_cache_entries()
    assert entries > 0
    keep_alive.append(run(2))
    assert mesh_device.num_program_cache_entries() == entries, "relabelling must not miss the program cache"


# ---------------------------------------------------------------------------
# copy / assign / typecast into a preallocated output
# ---------------------------------------------------------------------------


def _copy_via_copy(src_t, dst_t):
    return ttnn.copy(src_t, dst_t)


def _copy_via_assign(src_t, dst_t):
    return ttnn.assign(src_t, memory_config=dst_t.memory_config(), dtype=dst_t.dtype, output_tensor=dst_t)


def _typecast_into(src_t, dst_t):
    return ttnn.typecast(src_t, dst_t.dtype, output_tensor=dst_t)


_WRITE_INTO_DST = [
    pytest.param(_copy_via_copy, ttnn.bfloat16, torch.bfloat16, id="copy"),
    pytest.param(_copy_via_assign, ttnn.bfloat16, torch.bfloat16, id="assign"),
    pytest.param(_typecast_into, ttnn.float32, torch.float32, id="typecast"),
]


@MESH_PARAMS
@pytest.mark.parametrize("write_into, dst_dtype, dst_torch_dtype", _WRITE_INTO_DST)
def test_replicated_src_into_sharded_dst_adopts_replicate(mesh_device, write_into, dst_dtype, dst_torch_dtype):
    """Replicated src written into a dst labelled Shard(0): dst becomes Replicate, on both handles.

    src and dst span the same set of mesh coordinates (full coverage), so after the write every device holds the
    same data and Replicate is the only label that describes dst.

    Negative control (behaviour before the fix): the union over ``{src, dst}`` let dst's Shard(0) win, so dst kept
    a Shard label although every device now holds an identical copy; composing it along dim 0 would have produced
    ``num_devices`` stacked copies of src.
    """
    num_devices = mesh_device.get_num_devices()
    shape = [1, 1, 64, 64]

    def run(seed):
        torch.manual_seed(seed)
        src = torch.randn(shape).bfloat16()
        src_t = _replicated(src, mesh_device)
        # Per-device shape matches src; the label is what differs.
        dst_t = _sharded(torch.zeros([num_devices] + shape[1:], dtype=dst_torch_dtype), mesh_device, 0, dtype=dst_dtype)
        _assert_labels(src_t, ["Replicate"], "precondition: src is replicated")
        _assert_labels(dst_t, ["Shard(0)"], "precondition: dst is labelled Shard(0)")
        assert _mesh_coords(src_t.tensor_topology()) == _mesh_coords(dst_t.tensor_topology()), "full coverage"

        returned = write_into(src_t, dst_t)

        _assert_labels(dst_t, ["Replicate"], "dst after write (caller's handle)")
        _assert_labels(returned, ["Replicate"], "dst after write (returned handle)")
        assert returned.tensor_topology() == src_t.tensor_topology()
        assert returned.buffer_address() == dst_t.buffer_address()
        assert returned.dtype == dst_dtype
        expected = src.to(dst_torch_dtype)
        for dst_shard in _per_device(dst_t):
            assert torch.equal(dst_shard, expected)
        return src_t, dst_t

    _reset_program_cache(mesh_device)
    keep_alive = [run(1)]
    entries = mesh_device.num_program_cache_entries()
    assert entries > 0
    keep_alive.append(run(2))
    assert mesh_device.num_program_cache_entries() == entries, "relabelling must not miss the program cache"


@MESH_PARAMS
@pytest.mark.parametrize("write_into, dst_dtype, dst_torch_dtype", _WRITE_INTO_DST)
def test_sharded_src_into_replicated_dst_adopts_shard(mesh_device, write_into, dst_dtype, dst_torch_dtype):
    """Dim-0-sharded src written into a Replicate dst (full coverage): dst becomes Shard(0), on both handles.

    Every device now holds a distinct shard of src; a Replicate label would make the serialiser keep only one of
    them. Before the fix the union over ``{src, dst}`` already produced Shard(0) here, so this direction is the
    correct-by-construction check that the new rule agrees with the union where the union happened to be right.
    """
    num_devices = mesh_device.get_num_devices()
    per_device_shape = [1, 1, 64, 64]
    full_shape = [num_devices] + per_device_shape[1:]

    def run(seed):
        torch.manual_seed(seed)
        src = torch.randn(full_shape).bfloat16()
        src_t = _sharded(src, mesh_device, 0)
        dst_t = _replicated(torch.zeros(per_device_shape, dtype=dst_torch_dtype), mesh_device, dtype=dst_dtype)
        _assert_labels(src_t, ["Shard(0)"], "precondition: src is sharded on dim 0")
        _assert_labels(dst_t, ["Replicate"], "precondition: dst is replicated")

        returned = write_into(src_t, dst_t)

        _assert_labels(dst_t, ["Shard(0)"], "dst after write (caller's handle)")
        assert returned.tensor_topology() == src_t.tensor_topology()
        assert dst_t.tensor_topology() == src_t.tensor_topology()
        assert returned.buffer_address() == dst_t.buffer_address()
        composed = ttnn.to_torch(dst_t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        assert list(composed.shape) == full_shape
        assert torch.equal(composed, src.to(dst_torch_dtype))
        return src_t, dst_t

    _reset_program_cache(mesh_device)
    keep_alive = [run(1)]
    entries = mesh_device.num_program_cache_entries()
    assert entries > 0
    keep_alive.append(run(2))
    assert mesh_device.num_program_cache_entries() == entries, "relabelling must not miss the program cache"


@MESH_PARAMS
def test_fresh_copy_and_typecast_outputs_take_source_topology(mesh_device):
    """Without a preallocated output, copy/assign and typecast label the fresh tensor like the source.

    The union over the single input gives the same answer, so this is a guard that the new hook does not change the
    fresh-output path.
    """
    num_devices = mesh_device.get_num_devices()
    src = torch.randn([num_devices, 1, 64, 64]).bfloat16()
    src_t = _sharded(src, mesh_device, 0)

    copied = ttnn.assign(src_t, memory_config=ttnn.L1_MEMORY_CONFIG)
    assert copied.tensor_topology() == src_t.tensor_topology()
    assert copied.buffer_address() != src_t.buffer_address()
    assert torch.equal(ttnn.to_torch(copied, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)), src)

    cast = ttnn.typecast(src_t, ttnn.float32)
    assert cast.tensor_topology() == src_t.tensor_topology()
    assert torch.equal(ttnn.to_torch(cast, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)), src.float())
