# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Layout-changing ops must preserve experimental per-core L1 allocation.

tilize / tilize_with_val_padding / untilize_with_unpadding rebuild the output
MemoryConfig when their optimized sharded program factory is selected, substituting the
shard geometry inherited from the input. That rebuild must carry over the rest of the
caller's requested config -- in particular the per-core allocation bit, which was
otherwise silently downgraded to a lockstep allocation.

See https://github.com/tenstorrent/tt-metal/issues/51133.

The `device` fixture comes from this directory's conftest.py, which sets
TT_METAL_ALLOCATOR_MODE_HYBRID=1 before opening the device. The allocator mode is
latched process-globally on first use, so this suite has to run in its own pytest
process -- which is how CI invokes it.
"""

import pytest
import torch

import ttnn


def _per_core_mem_config(memory_layout, core_range_set, shard_shape):
    """L1 sharded MemoryConfig with the experimental per-core allocation bit set."""
    mem_config = ttnn.MemoryConfig(
        memory_layout,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_range_set, list(shard_shape), ttnn.ShardOrientation.ROW_MAJOR),
    )
    mem_config.experimental_set_per_core_allocation(True)
    return mem_config


def _core_range_set(start, end):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*start), ttnn.CoreCoord(*end))])


def _cores_in(core_range_set):
    cores = []
    for core_range in core_range_set.ranges():
        for y in range(core_range.start.y, core_range.end.y + 1):
            for x in range(core_range.start.x, core_range.end.x + 1):
                cores.append(ttnn.CoreCoord(x, y))
    return cores


def _assert_uniform_addresses(tensor, core_range_set, label):
    """Assert a multi-core per-core tensor has the same L1 address on every core.

    Buffer::address() does not properly handle per-core allocated tensors: it returns the
    first core's address, and the host write, the host read and CB aliasing all use that
    one value for every core. So a multi-core per-core buffer is only handled correctly
    when its addresses happen to agree, which is why callers assert it too (see
    models/demos/deepseek_v3_b1/weights/overlap/packing.py). Checking it here means
    preserving the per-core bit cannot silently produce a mis-addressed tensor.
    """
    cores = _cores_in(core_range_set)
    addrs = [tensor.experimental_per_core_buffer_address(c) for c in cores]
    assert len(set(addrs)) == 1, f"{label}: expected uniform per-core addresses, got " + ", ".join(
        f"({c.x},{c.y})={a:#x}" for c, a in zip(cores, addrs)
    )


def _row_major_per_core_input(device, torch_tensor, memory_layout, core_range_set, shard_shape):
    """Build a ROW_MAJOR, per-core allocated L1 input tensor on device.

    A sharded memory config always takes the host-construction path in
    create_tt_tensor_from_host_data, so the per-core bit survives to the device here.
    """
    mem_config = _per_core_mem_config(memory_layout, core_range_set, shard_shape)
    tensor = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mem_config,
    )
    assert tensor.is_per_core_allocated(), "precondition failed: ROW_MAJOR input is not per-core allocated"
    _assert_uniform_addresses(tensor, core_range_set, "row-major input")
    return tensor, mem_config


@pytest.mark.parametrize(
    "memory_layout, grid_start, grid_end, tensor_shape, shard_shape",
    [
        # WIDTH_SHARDED over 8 cores in a row -- the shape class reported in #51133
        # (gate_mm: a full-height, one-tile-wide shard).
        (ttnn.TensorMemoryLayout.WIDTH_SHARDED, (0, 0), (7, 0), (512, 256), (512, 32)),
        (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (0, 0), (7, 0), (512, 256), (64, 256)),
        (ttnn.TensorMemoryLayout.BLOCK_SHARDED, (0, 0), (1, 1), (256, 256), (128, 128)),
    ],
    ids=["width_sharded", "height_sharded", "block_sharded"],
)
def test_tilize_preserves_per_core_allocation(device, memory_layout, grid_start, grid_end, tensor_shape, shard_shape):
    """tilize must not downgrade a per-core output request to a lockstep allocation.

    All three parametrizations are eligible for the optimized sharded program factory,
    which is the path that rebuilds the output MemoryConfig from the input's shard spec.
    """
    core_range_set = _core_range_set(grid_start, grid_end)
    torch_input = torch.randn(*tensor_shape, dtype=torch.bfloat16)

    row_major, mem_config = _row_major_per_core_input(device, torch_input, memory_layout, core_range_set, shard_shape)

    tiled = ttnn.tilize(row_major, memory_config=mem_config, dtype=ttnn.bfloat16)

    assert tiled.is_per_core_allocated(), (
        "tilize dropped the experimental per-core allocation bit: the output buffer is "
        "lockstep-allocated even though the requested output memory_config asked for per-core"
    )
    _assert_uniform_addresses(tiled, core_range_set, "tilized output")
    assert torch.equal(ttnn.to_torch(tiled), torch_input), "tilize corrupted the data"


def test_to_layout_preserves_per_core_allocation(device):
    """The same drop site reached through the public ttnn.to_layout entry point."""
    core_range_set = _core_range_set((0, 0), (7, 0))
    torch_input = torch.randn(512, 256, dtype=torch.bfloat16)

    row_major, mem_config = _row_major_per_core_input(
        device, torch_input, ttnn.TensorMemoryLayout.WIDTH_SHARDED, core_range_set, (512, 32)
    )

    tiled = ttnn.to_layout(row_major, ttnn.TILE_LAYOUT, memory_config=mem_config)

    assert tiled.is_per_core_allocated(), "ttnn.to_layout dropped the per-core allocation bit"
    _assert_uniform_addresses(tiled, core_range_set, "tilized output")
    assert torch.equal(ttnn.to_torch(tiled), torch_input)


def test_tilize_with_val_padding_preserves_per_core_allocation(device):
    """tilize_with_val_padding rebuilds the output MemoryConfig the same way tilize does.

    A WIDTH_SHARDED input whose logical height is not tile-aligned (500 -> 512) routes
    through the optimized sharded factory, which substitutes a reshaped input shard spec.
    """
    core_range_set = _core_range_set((0, 0), (7, 0))
    torch_input = torch.randn(500, 256, dtype=torch.bfloat16)

    row_major, _ = _row_major_per_core_input(
        device, torch_input, ttnn.TensorMemoryLayout.WIDTH_SHARDED, core_range_set, (500, 32)
    )
    output_mem_config = _per_core_mem_config(ttnn.TensorMemoryLayout.WIDTH_SHARDED, core_range_set, (512, 32))

    padded = ttnn.tilize_with_val_padding(row_major, ttnn.Shape([512, 256]), 0.0, memory_config=output_mem_config)

    assert padded.is_per_core_allocated(), "tilize_with_val_padding dropped the per-core allocation bit"
    _assert_uniform_addresses(padded, core_range_set, "padded output")
    assert torch.equal(ttnn.to_torch(padded), torch_input)


def test_untilize_with_unpadding_preserves_per_core_allocation(device):
    """untilize_with_unpadding rebuilds the output MemoryConfig for sharded in/out.

    A padded TILE tensor is unpadded back to a shorter logical height, which reshapes
    the shard spec inherited from the input.
    """
    core_range_set = _core_range_set((0, 0), (7, 0))
    torch_input = torch.randn(512, 256, dtype=torch.bfloat16)

    row_major, mem_config = _row_major_per_core_input(
        device, torch_input, ttnn.TensorMemoryLayout.WIDTH_SHARDED, core_range_set, (512, 32)
    )
    tiled = ttnn.tilize(row_major, memory_config=mem_config, dtype=ttnn.bfloat16)

    unpadded = ttnn.untilize_with_unpadding(tiled, ttnn.Shape([479, 255]), memory_config=mem_config)

    assert unpadded.is_per_core_allocated(), "untilize_with_unpadding dropped the per-core allocation bit"
    _assert_uniform_addresses(unpadded, core_range_set, "unpadded output")
    assert torch.equal(ttnn.to_torch(unpadded), torch_input[:480, :256])


def test_from_torch_on_device_construction_preserves_per_core_allocation(device):
    """The reported symptom: from_torch(device=..., memory_config=<per-core L1>).

    With a mesh_mapper and a device-constructible dtype, from_torch builds the tensor
    on device in ROW_MAJOR and tilizes it there. The per-core bit rides on the caller's
    memory_config through that whole chain and must survive the on-device tilize.

    ttnn.CreateDevice returns a unit MeshDevice, so the mesh_mapper branch of
    create_tt_tensor_from_host_data -- the only one that constructs a sharded tensor
    on device -- is reachable with a single device.
    """
    core_range_set = _core_range_set((0, 0), (7, 0))
    shard_shape = (512, 32)
    torch_input = torch.randn(shard_shape[0], shard_shape[1] * 8, dtype=torch.bfloat16)
    mem_config = _per_core_mem_config(ttnn.TensorMemoryLayout.WIDTH_SHARDED, core_range_set, shard_shape)

    on_device = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    assert on_device.is_per_core_allocated(), (
        "from_torch(device=..., memory_config=<per-core L1>) returned a lockstep-allocated "
        "tensor; the on-device construction path dropped the per-core request"
    )
    _assert_uniform_addresses(on_device, core_range_set, "from_torch output")
    # Unit mesh + ReplicateTensorToMesh: to_torch returns the single replica.
    assert torch.equal(ttnn.to_torch(on_device), torch_input)

    # The same request via allocate_tensor_on_device is the route that already worked --
    # both must agree.
    host = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=None,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    allocated = ttnn.allocate_tensor_on_device(host.spec, device)
    assert allocated.is_per_core_allocated(), "precondition failed: allocate_tensor_on_device is not per-core"


_PAD_BYTES = 8192


def _pad_core(device, core):
    """One per-core buffer on a single core, used to skew that core's free list."""
    c = ttnn.CoreCoord(*core)
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ttnn.CoreRangeSet([ttnn.CoreRange(c, c)]), [1, _PAD_BYTES], ttnn.ShardOrientation.ROW_MAJOR),
    )
    mem_config.experimental_set_per_core_allocation(True)
    return ttnn.from_torch(
        torch.zeros(1, _PAD_BYTES, dtype=torch.uint8),
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mem_config,
    )


def _skew_free_lists(device, cores):
    """Core i gets i padding buffers, so every core's free list ends up somewhere different."""
    return [_pad_core(device, core) for i, core in enumerate(cores) for _ in range(i)]


def test_per_core_addresses_diverge_where_lockstep_shares_one(device):
    """The defining difference between the two allocators.

    Given cores whose free lists sit at different heights, the per-core allocator hands
    each core its own address, while the lockstep allocator hands out a single address
    shared by every core.

    Note this documents the *allocator*, not a supported tensor configuration: generic
    ops cannot handle a multi-core per-core buffer whose addresses diverge. Callers
    wanting genuinely independent addresses allocate one single-core buffer per core.
    """
    cores = [(0, 0), (1, 0), (2, 0), (3, 0)]
    core_range_set = _core_range_set((0, 0), (3, 0))
    shard_shape = (512, 32)
    torch_input = torch.randn(shard_shape[0], shard_shape[1] * len(cores), dtype=torch.bfloat16)

    def build(per_core):
        mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_range_set, list(shard_shape), ttnn.ShardOrientation.ROW_MAJOR),
        )
        if per_core:
            mem_config.experimental_set_per_core_allocation(True)
        return ttnn.from_torch(
            torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
        )

    pads = _skew_free_lists(device, cores)  # noqa: F841 - held alive to keep the skew in place
    per_core_tensor = build(per_core=True)
    addrs = [per_core_tensor.experimental_per_core_buffer_address(ttnn.CoreCoord(*c)) for c in cores]
    assert per_core_tensor.is_per_core_allocated()
    assert len(set(addrs)) == len(
        cores
    ), "per-core allocator should give each core its own address once their free lists differ, got " + ", ".join(
        f"({c[0]},{c[1]})={a:#x}" for c, a in zip(cores, addrs)
    )
    del per_core_tensor

    lockstep_tensor = build(per_core=False)
    assert not lockstep_tensor.is_per_core_allocated()
    # buffer_address() only exists for lockstep tensors -- it is the single shared address,
    # and it fatals for per-core ones precisely because those have no single address.
    assert isinstance(lockstep_tensor.buffer_address(), int)
