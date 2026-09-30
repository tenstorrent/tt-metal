# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Mesh-topology labels of caller-owned outputs for the data-movement in-place ops.

Every device op relabels the tensors it returns in ``launch()``. Unless the op defines
``compute_output_topologies`` that label is the *union* of all tensor inputs, where any Shard
placement beats Replicate. For an op that writes into a tensor the caller already owns that
union is wrong in two different ways:

* ``sharded_to_interleaved_partial`` / ``slice_write`` return the caller's cache/output. Writing a
  slice into it does not change how it is distributed, but the union relabels a Replicate
  cache/output with the (sharded) input's placement because the input is a tensor arg too.
  Fix: keep the caller tensor's own topology.
* ``copy`` / ``assign`` / ``typecast`` with a preallocated output overwrite dst on every mesh
  coordinate with src's shard for that coordinate, so dst now really is distributed like src.
  The union instead keeps a stale Shard label on a dst that is now replicated (or, in the other
  direction, coincidentally gets it right). Fix: dst adopts src's topology.

Each test states in its docstring what the label was before the fix (the negative control) and
asserts the program cache is untouched by a second dispatch: topology is not part of the program
hash, so the relabelling must not cost a recompile.
"""

import pytest
import torch

import ttnn

MESH_PARAMS = pytest.mark.parametrize("mesh_device", [2], indirect=True)


def _placement_names(topology):
    return [f"Shard({p.dim})" if isinstance(p, ttnn.PlacementShard) else "Replicate" for p in topology.placements()]


def _replicated(torch_tensor, mesh_device, **kwargs):
    kwargs.setdefault("dtype", ttnn.bfloat16)
    kwargs.setdefault("layout", ttnn.TILE_LAYOUT)
    kwargs.setdefault("memory_config", ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.from_torch(
        torch_tensor, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device), **kwargs
    )


def _sharded(torch_tensor, mesh_device, dim, **kwargs):
    kwargs.setdefault("dtype", ttnn.bfloat16)
    kwargs.setdefault("layout", ttnn.TILE_LAYOUT)
    kwargs.setdefault("memory_config", ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.from_torch(
        torch_tensor, device=mesh_device, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim), **kwargs
    )


def _per_device(tt_tensor):
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tt_tensor)]


def _assert_labels(tt_tensor, expected_names, what):
    names = _placement_names(tt_tensor.tensor_topology())
    assert names == expected_names, f"{what}: expected placements {expected_names}, got {names}"


def _reset_program_cache(mesh_device):
    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()


# ---------------------------------------------------------------------------
# sharded_to_interleaved_partial
# ---------------------------------------------------------------------------


@MESH_PARAMS
def test_sharded_to_interleaved_partial_keeps_cache_topology(mesh_device):
    """The cache keeps its own (Replicate) label after per-slice writes from a dim-0-sharded input.

    Negative control (behaviour before the fix): the op's tensor args are ``{input_tensor, cache_tensor}`` with the
    input first, so the union relabelled the cache ``Shard(0)`` after the first slice write. Now the caller's label
    on the cache is untouched by the write.
    """
    torch.manual_seed(0)
    num_devices = mesh_device.get_num_devices()
    num_slices = 2
    H, W = 64, 64
    grid = mesh_device.compute_with_storage_grid_size()
    grid_size = (grid.x, grid.y)
    height_shard_spec = [H // num_slices, W]

    def run(seed):
        torch.manual_seed(seed)
        # One [1, 1, H, W] block per device, sharded over the mesh on dim 0.
        in0 = torch.randn([num_devices, 1, H, W]).bfloat16()
        in0_t = _sharded(in0, mesh_device, 0, memory_config=ttnn.L1_MEMORY_CONFIG)
        cache = torch.zeros([1, 1, H, W]).bfloat16()
        cache_t = _replicated(cache, mesh_device, memory_config=ttnn.L1_MEMORY_CONFIG)
        _assert_labels(cache_t, ["Replicate"], "precondition: cache is replicated")

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
            _assert_labels(in0_slice, ["Shard(0)"], "precondition: the slice carries the sharded input's label")
            returned = ttnn.sharded_to_interleaved_partial(
                in0_slice, cache_t, num_slices, slice_index, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            _assert_labels(cache_t, ["Replicate"], f"cache after slice {slice_index}")
            assert returned.tensor_topology() == cache_t.tensor_topology()
            assert returned.buffer_address() == cache_t.buffer_address()

        # Each device's cache now holds that device's shard of in0.
        for device_index, cache_shard in enumerate(_per_device(cache_t)):
            assert torch.equal(cache_shard, in0[device_index : device_index + 1])
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
def test_slice_write_keeps_output_topology(mesh_device, input_memory_config):
    """A Replicate output written from a dim-0-sharded input keeps its Replicate label.

    Negative control (behaviour before the fix): the union over ``{input, output}`` relabelled the output
    ``Shard(0)``. Row-major on both sides so the wrapper hands the caller's handle straight to the device op (a TILE
    output would first be converted to row-major, which allocates a new tensor).
    """
    num_devices = mesh_device.get_num_devices()
    out_shape = [1, 1, 64, 64]
    in_shape = [1, 1, 32, 64]
    start = [0, 0, 0, 0]
    end = in_shape
    step = [1, 1, 1, 1]

    def run(seed):
        torch.manual_seed(seed)
        src = torch.randn([num_devices] + in_shape[1:]).bfloat16()
        src_t = _sharded(src, mesh_device, 0, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=input_memory_config)
        _assert_labels(src_t, ["Shard(0)"], "precondition: input is sharded on dim 0")
        out_t = _replicated(torch.zeros(out_shape).bfloat16(), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)
        _assert_labels(out_t, ["Replicate"], "precondition: output is replicated")

        returned = ttnn.experimental.slice_write(src_t, out_t, start, end, step)

        _assert_labels(out_t, ["Replicate"], "output after slice_write")
        assert returned.tensor_topology() == out_t.tensor_topology()
        assert returned.buffer_address() == out_t.buffer_address()
        for device_index, out_shard in enumerate(_per_device(out_t)):
            expected = torch.zeros(out_shape).bfloat16()
            expected[:, :, : in_shape[2], :] = src[device_index : device_index + 1]
            assert torch.equal(out_shard, expected)
        return src_t, out_t

    _reset_program_cache(mesh_device)
    keep_alive = [run(1)]
    entries = mesh_device.num_program_cache_entries()
    assert entries > 0
    keep_alive.append(run(2))
    assert mesh_device.num_program_cache_entries() == entries, "relabelling must not miss the program cache"


# ---------------------------------------------------------------------------
# copy / assign
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

    After the write every device holds the same data, so Replicate is the only label that describes dst.

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
    """Dim-0-sharded src written into a Replicate dst: dst becomes Shard(0), on both handles.

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
