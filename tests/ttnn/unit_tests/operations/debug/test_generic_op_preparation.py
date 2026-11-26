# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

SHAPE = [1, 1, 32, 32]
# Each unrolled store adds instructions to the kernel binary. 6,000 stores already exceed Blackhole's 70,656-byte
# kernel-config buffer; twice that leaves margin for other architectures.
STORES_EXCEEDING_KERNEL_CONFIG_BUFFER = 12000


def _io_tensors(device):
    input_tensor = ttnn.from_torch(
        torch.zeros(SHAPE, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(SHAPE), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    return [input_tensor, output_tensor]


def _single_core_program(kernel_source, descriptor_type):
    core = ttnn.CoreCoord(0, 0)
    runtime_args = ttnn.RuntimeArgs()
    runtime_args[0][0] = [0]
    program = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=kernel_source,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
                compile_time_args=[],
                runtime_args=runtime_args,
                # BRISC avoids Wormhole's 16 KiB NCRISC code limit so oversized
                # kernels reach the aggregate kernel-config-buffer check.
                config=ttnn.WriterConfigDescriptor(),
            )
        ],
        semaphores=[],
        cbs=[],
    )
    if descriptor_type == "program":
        return program
    coord = ttnn.MeshCoordinate(0, 0)
    return ttnn.MeshProgramDescriptor({ttnn.MeshCoordinateRange(coord, coord): program})


def _oversized_kernel_source(descriptor_type):
    return (
        f"// {descriptor_type}\n"
        "#include <cstdint>\n"
        "void kernel_main() {\n"
        "    volatile uint32_t* sink = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(0));\n"
        f"#pragma GCC unroll {STORES_EXCEEDING_KERNEL_CONFIG_BUFFER}\n"
        f"    for (uint32_t i = 0; i < {STORES_EXCEEDING_KERNEL_CONFIG_BUFFER}; ++i) {{ sink[0] = i * 2654435761u; }}\n"
        "}\n"
    )


def test_prepare_generic_op_is_experimental():
    assert not hasattr(ttnn, "prepare_generic_op"), "prepare_generic_op must only be exported under ttnn.experimental"


@pytest.mark.parametrize("descriptor_type", ["program", "mesh_program"])
def test_prepare_generic_op_warms_program_cache(device, descriptor_type):
    io_tensors = _io_tensors(device)
    # The comment makes each parametrization a distinct program-cache entry.
    program = _single_core_program(f"// {descriptor_type}\nvoid kernel_main() {{}}\n", descriptor_type)

    entries_before = device.num_program_cache_entries()
    ttnn.experimental.prepare_generic_op(io_tensors, program)
    assert (
        device.num_program_cache_entries() == entries_before + 1
    ), "a successful preparation must add the workload to the program cache"

    ttnn.experimental.prepare_generic_op(io_tensors, program)
    ttnn.generic_op(io_tensors, program)
    assert (
        device.num_program_cache_entries() == entries_before + 1
    ), "repeated preparation and the launch after it must reuse the prepared workload"


@pytest.mark.parametrize("descriptor_type", ["program", "mesh_program"])
def test_program_exceeding_kernel_config_buffer_leaves_cache_unchanged(device, expect_error, descriptor_type):
    io_tensors = _io_tensors(device)
    program = _single_core_program(_oversized_kernel_source(descriptor_type), descriptor_type)

    entries_before = device.num_program_cache_entries()
    with expect_error(RuntimeError, "too large for kernel config buffer"):
        ttnn.experimental.prepare_generic_op(io_tensors, program)
    assert device.num_program_cache_entries() == entries_before, "a failed preparation must not change the cache"

    with expect_error(RuntimeError, "too large for kernel config buffer"):
        ttnn.generic_op(io_tensors, program)
    assert device.num_program_cache_entries() == entries_before, "a failed launch must not change the cache"


def test_prepare_generic_op_with_cb_bound_later(device, expect_error):
    io_tensors = _io_tensors(device)
    backing = ttnn.from_torch(
        torch.zeros(SHAPE, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.create_sharded_memory_config(
            shape=SHAPE[-2:],
            core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )
    bound_cb = ttnn.cb_descriptor_from_sharded_tensor(0, backing)
    # The same CB, but its tensor is named by a later descriptor.
    unbound_cb = ttnn.CBDescriptor(
        total_size=bound_cb.total_size,
        core_ranges=bound_cb.core_ranges,
        format_descriptors=bound_cb.format_descriptors,
    )
    unbound_cb.globally_allocated_bound_later = True

    def program(cb):
        program = _single_core_program("// cb bound later\nvoid kernel_main() {}\n", "program")
        program.cbs = [cb]
        return program

    entries_before = device.num_program_cache_entries()
    ttnn.experimental.prepare_generic_op(io_tensors, program(unbound_cb))
    assert device.num_program_cache_entries() == entries_before + 1

    # A launch that names the tensor binds the CB on the prepared workload.
    ttnn.generic_op(io_tensors, program(bound_cb))
    ttnn.synchronize_device(device)

    # A launch that names no tensor must fail, not run with the previous launch's tensor.
    with expect_error(RuntimeError, "bound later but the descriptor names no buffer"):
        ttnn.generic_op(io_tensors, program(unbound_cb))
    assert device.num_program_cache_entries() == entries_before + 1, "every launch must reuse the prepared workload"
