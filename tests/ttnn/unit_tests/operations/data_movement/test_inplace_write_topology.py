# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Mesh-topology labels of the data-movement ops that write into the caller's tensor.

``sharded_to_interleaved_partial`` and ``slice_write`` write a slice into the caller's cache / output, which keeps
its label unless the input is sharded along a mesh axis on which that label is replicated; the result then takes the
union of the two labels, on the caller's handle too. ``copy`` / ``assign`` / ``typecast`` overwrite a preallocated
output stored on the same mesh coordinates as the source on every device, so it takes the source's label. Running an
op a second time with other labels must hit the program cache: the label is not part of the program hash.

A preallocated output stored on more mesh coordinates than the source, such as a source taken from one device of a
mesh tensor, is overwritten on the source's devices only and labelled by the framework default: by the caller-owned
rule against a source labelled over the same mesh coordinates, with the union of the two labels against one labelled
over fewer.
"""

import pytest
import torch

import ttnn

MESH_PARAMS = pytest.mark.parametrize("mesh_device", [2], indirect=True)


def _placement_names(topology):
    return [f"Shard({p.dim})" if isinstance(p, ttnn.PlacementShard) else "Replicate" for p in topology.placements()]


def _storage_coords(tt_tensor):
    return [tuple(coord) for coord in tt_tensor.device_coords()]


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


# How a tensor is distributed over the mesh, and under which label.
REPLICATED = "replicated"  # ReplicateTensorToMesh: collapsed 1-D label [Replicate]
REPLICATED_2D = "replicated_2d"  # same data, one placement per mesh axis: [Replicate, Replicate] on a 1x2 mesh
SHARD0 = "shard0"  # ShardTensorToMesh(dim=0): collapsed 1-D label [Shard(0)]
SHARD0_2D = "shard0_2d"  # same data, one placement per mesh axis, e.g. [Replicate, Shard(0)] on a 1x2 mesh

# The same distribution under the label of the other rank.
_OTHER_LABEL = {REPLICATED: REPLICATED_2D, REPLICATED_2D: REPLICATED, SHARD0: SHARD0_2D, SHARD0_2D: SHARD0}


def _is_replicated(label):
    return label in (REPLICATED, REPLICATED_2D)


def _distribute(torch_tensor, mesh_device, label, **kwargs):
    if label == REPLICATED:
        return _replicated(torch_tensor, mesh_device, **kwargs)
    if label == SHARD0:
        return _sharded(torch_tensor, mesh_device, 0, **kwargs)
    if label == REPLICATED_2D:
        placements = [ttnn.PlacementReplicate() for _ in mesh_device.shape]
    else:
        assert label == SHARD0_2D
        placements = _shard0_per_mesh_axis(mesh_device)
    mapper = ttnn.create_mesh_mapper(mesh_device, ttnn.MeshMapperConfig(placements))
    return _from_torch(torch_tensor, mesh_device, mapper, **kwargs)


def _global_shape(label, per_device_shape, num_devices):
    return per_device_shape if _is_replicated(label) else [num_devices] + per_device_shape[1:]


def _per_device_slice(label, torch_tensor, device_index):
    return torch_tensor if _is_replicated(label) else torch_tensor[device_index : device_index + 1]


def _per_device(tt_tensor):
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tt_tensor)]


def _assert_labels(tt_tensor, expected_names, what):
    names = _placement_names(tt_tensor.tensor_topology())
    assert names == expected_names, f"{what}: expected placements {expected_names}, got {names}"


def _check_program_cache(mesh_device, run, label):
    """Calls ``run(seed, label)``, then ``run`` with the source's other label: the second call must hit the cache."""
    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()
    first = run(1, label)
    entries = mesh_device.num_program_cache_entries()
    assert entries > 0
    second = run(2, _OTHER_LABEL[label])
    assert first[0].tensor_topology() != second[0].tensor_topology(), "precondition: another source label"
    assert mesh_device.num_program_cache_entries() == entries, "a call with other labels must hit the program cache"


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
    """
    num_devices = mesh_device.get_num_devices()
    num_slices = 2
    H, W = 64, 64
    per_device_shape = [1, 1, H, W]
    grid = mesh_device.compute_with_storage_grid_size()
    grid_size = (grid.x, grid.y)
    height_shard_spec = [H // num_slices, W]

    def run(seed, in_label):
        torch.manual_seed(seed)
        in0 = torch.randn(_global_shape(in_label, per_device_shape, num_devices)).bfloat16()
        in0_t = _distribute(in0, mesh_device, in_label, memory_config=ttnn.L1_MEMORY_CONFIG)
        cache = torch.zeros(_global_shape(cache_label, per_device_shape, num_devices)).bfloat16()
        cache_t = _distribute(cache, mesh_device, cache_label, memory_config=ttnn.L1_MEMORY_CONFIG)
        cache_names_before = _placement_names(cache_t.tensor_topology())
        input_names = _placement_names(in0_t.tensor_topology())
        # The union replaces the cache's label exactly when the input is sharded on an axis the cache is not.
        assert takes_union == (not _is_replicated(in_label) and _is_replicated(cache_label))
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
            assert torch.equal(cache_shard, _per_device_slice(in_label, in0, device_index))
        return in0_t, cache_t

    _check_program_cache(mesh_device, run, input_label)


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
    """
    num_devices = mesh_device.get_num_devices()
    out_shape = [1, 1, 64, 64]
    in_shape = [1, 1, 32, 64]
    start = [0, 0, 0, 0]
    end = in_shape
    step = [1, 1, 1, 1]

    def run(seed, in_label):
        torch.manual_seed(seed)
        src = torch.randn(_global_shape(in_label, in_shape, num_devices)).bfloat16()
        src_t = _distribute(src, mesh_device, in_label, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=input_memory_config)
        out = torch.zeros(_global_shape(output_label, out_shape, num_devices)).bfloat16()
        out_t = _distribute(out, mesh_device, output_label, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_names_before = _placement_names(out_t.tensor_topology())
        input_names = _placement_names(src_t.tensor_topology())
        # The union replaces the output's label exactly when the input is sharded on an axis the output is not.
        assert takes_union == (not _is_replicated(in_label) and _is_replicated(output_label))
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
            expected[:, :, : in_shape[2], :] = _per_device_slice(in_label, src, device_index)
            assert torch.equal(out_shard, expected)
        return src_t, out_t

    _check_program_cache(mesh_device, run, input_label)


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

    src and dst are stored on the same mesh coordinates (full coverage), so after the write every device holds the
    same data and a Replicate label is the only one that describes dst. With src's collapsed label the union over
    ``{src, dst}`` is dst's Shard(0), which would compose into ``num_devices`` stacked copies of src.
    """
    num_devices = mesh_device.get_num_devices()
    shape = [1, 1, 64, 64]

    def run(seed, src_label):
        torch.manual_seed(seed)
        src = torch.randn(shape).bfloat16()
        src_t = _distribute(src, mesh_device, src_label)
        # Per-device shape matches src; the label is what differs.
        dst_t = _sharded(torch.zeros([num_devices] + shape[1:], dtype=dst_torch_dtype), mesh_device, 0, dtype=dst_dtype)
        src_names = _placement_names(src_t.tensor_topology())
        assert set(src_names) == {"Replicate"}, f"precondition: src is replicated, got {src_names}"
        _assert_labels(dst_t, ["Shard(0)"], "precondition: dst is labelled Shard(0)")
        assert _storage_coords(src_t) == _storage_coords(dst_t), "full coverage"

        returned = write_into(src_t, dst_t)

        _assert_labels(dst_t, src_names, "dst after write (caller's handle)")
        _assert_labels(returned, src_names, "dst after write (returned handle)")
        assert returned.tensor_topology() == src_t.tensor_topology()
        assert dst_t.tensor_topology() == src_t.tensor_topology()
        assert returned.buffer_address() == dst_t.buffer_address()
        assert returned.dtype == dst_dtype
        expected = src.to(dst_torch_dtype)
        for dst_shard in _per_device(dst_t):
            assert torch.equal(dst_shard, expected)
        return src_t, dst_t

    _check_program_cache(mesh_device, run, REPLICATED)


@MESH_PARAMS
@pytest.mark.parametrize("write_into, dst_dtype, dst_torch_dtype", _WRITE_INTO_DST)
def test_sharded_src_into_replicated_dst_adopts_shard(mesh_device, write_into, dst_dtype, dst_torch_dtype):
    """Dim-0-sharded src written into a Replicate dst (full coverage): dst becomes Shard(0), on both handles.

    Every device now holds a distinct shard of src; a Replicate label would make the serialiser keep only one of
    them.
    """
    num_devices = mesh_device.get_num_devices()
    per_device_shape = [1, 1, 64, 64]
    full_shape = [num_devices] + per_device_shape[1:]

    def run(seed, src_label):
        torch.manual_seed(seed)
        src = torch.randn(full_shape).bfloat16()
        src_t = _distribute(src, mesh_device, src_label)
        dst_t = _replicated(torch.zeros(per_device_shape, dtype=dst_torch_dtype), mesh_device, dtype=dst_dtype)
        src_names = _placement_names(src_t.tensor_topology())
        assert "Shard(0)" in src_names, f"precondition: src is sharded on dim 0, got {src_names}"
        _assert_labels(dst_t, ["Replicate"], "precondition: dst is replicated")

        returned = write_into(src_t, dst_t)

        _assert_labels(dst_t, src_names, "dst after write (caller's handle)")
        assert returned.tensor_topology() == src_t.tensor_topology()
        assert dst_t.tensor_topology() == src_t.tensor_topology()
        assert returned.buffer_address() == dst_t.buffer_address()
        composed = ttnn.to_torch(dst_t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        assert list(composed.shape) == full_shape
        assert torch.equal(composed, src.to(dst_torch_dtype))
        return src_t, dst_t

    _check_program_cache(mesh_device, run, SHARD0)


# Sources stored on one device of the mesh, built from device 0's tensor of a mesh tensor: (how, what device 0 of
# dst holds afterwards, given device 0's slice of the mesh tensor).
_ONE_DEVICE_SOURCES = [
    pytest.param(ttnn.neg, lambda x0: -x0, id="op_on_one_device_tensor"),
    pytest.param(lambda device_tensor: device_tensor, lambda x0: x0, id="one_device_tensor"),
]


@MESH_PARAMS
@pytest.mark.parametrize("make_src, expected_src", _ONE_DEVICE_SOURCES)
@pytest.mark.parametrize("write_into, dst_dtype, dst_torch_dtype", _WRITE_INTO_DST)
def test_src_on_one_device_overwrites_dst_there_only(
    mesh_device, make_src, expected_src, write_into, dst_dtype, dst_torch_dtype
):
    """A source stored on one device overwrites a Shard(0) dst on that device only, and dst keeps its label.

    dst is stored on every device and written in part, so the framework default labels it. The result of an op on one
    device tensor of a mesh tensor is stored on that device, though its label can span the whole mesh: dst keeps its
    Shard(0) next to the replicated source. That device tensor itself is labelled over its own mesh coordinate: dst
    takes the union of the two labels, which is its own Shard(0).
    """
    num_devices = mesh_device.get_num_devices()
    shape = [num_devices, 1, 64, 64]
    torch.manual_seed(0)
    x = torch.randn(shape).bfloat16()
    x_t = _sharded(x, mesh_device, 0)
    src_t = make_src(ttnn.get_device_tensors(x_t)[0])
    dst = torch.randn(shape).to(dst_torch_dtype)
    dst_t = _sharded(dst, mesh_device, 0, dtype=dst_dtype)
    dst_label = dst_t.tensor_topology()
    assert _storage_coords(src_t) == _storage_coords(dst_t)[:1], "precondition: src is stored on dst's first device"

    returned = write_into(src_t, dst_t)

    assert dst_t.tensor_topology() == dst_label, "dst after write (caller's handle)"
    assert returned.tensor_topology() == dst_label, "dst after write (returned handle)"
    assert returned.buffer_address() == dst_t.buffer_address()
    dst_shards = _per_device(dst_t)
    assert torch.equal(dst_shards[0], expected_src(x[0:1]).to(dst_torch_dtype))
    for device_index in range(1, num_devices):
        assert torch.equal(dst_shards[device_index], dst[device_index : device_index + 1]), f"device {device_index}"


# The device op's error for a preallocated output of another per-device shape.
_SHAPE_MISMATCH_ERRORS = {
    _copy_via_copy: "does not match output tensor shape",
    _copy_via_assign: "does not match output tensor shape",
    _typecast_into: "logical shapes to match",
}


@MESH_PARAMS
@pytest.mark.parametrize("write_into, dst_dtype, dst_torch_dtype", _WRITE_INTO_DST)
def test_rejected_write_keeps_dst_label(mesh_device, write_into, dst_dtype, dst_torch_dtype, expect_error):
    """A write the op rejects leaves dst's label alone: a replicated src and a dst labelled Shard(0) with another
    per-device shape. The labels are applied before the op validates the call, so taking src's label first would
    leave dst labelled Replicate over its unchanged shards."""
    num_devices = mesh_device.get_num_devices()
    src_t = _replicated(torch.randn([1, 1, 64, 64]).bfloat16(), mesh_device)
    dst = torch.randn([num_devices, 1, 64, 32]).to(dst_torch_dtype)
    dst_t = _sharded(dst, mesh_device, 0, dtype=dst_dtype)
    label = dst_t.tensor_topology()

    with expect_error(RuntimeError, _SHAPE_MISMATCH_ERRORS[write_into]):
        write_into(src_t, dst_t)

    assert dst_t.tensor_topology() == label
    for device_index, dst_shard in enumerate(_per_device(dst_t)):
        assert torch.equal(dst_shard, dst[device_index : device_index + 1]), f"device {device_index}"


@MESH_PARAMS
def test_fresh_copy_and_typecast_outputs_take_source_topology(mesh_device):
    """Without a preallocated output, copy/assign and typecast label the fresh tensor like the source."""
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
