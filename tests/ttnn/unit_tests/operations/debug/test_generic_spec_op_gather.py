# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""A two-core gather through ttnn.experimental.generic_op, with the buffers at planned offsets of an L1 arena tensor.

The arena is height-sharded over the two cores, so it sits at the same address on both. On the gather core, a region
at REGION_OFFSET holds one slice per core, at slot offset core_index * SLICE_BYTES. Every iteration:

- The gather kernel reserves the region, which is a DFB borrowed from the arena at REGION_OFFSET, so it waits until
  the drain kernel has copied the previous iteration out. It then gives the other core a "ready" credit, copies its own
  slice into slot 0, waits for the other core's "valid" signal and pushes the region to the drain kernel.
- The other core waits for "ready", stages its slice in its own part of the arena, writes it into slot 1 of the gather
  core's region over the NoC (the region's address is the same on both cores, as blaze's planner places it) and
  increments "valid" on the gather core.
- The drain kernel copies both slices out to DRAM and pops the region.

No RISC-V reads L1 here: the data only moves over the NoC, so Quasar's data movement caches need no scoped_lock.
Implicit DFB sync is off on every kernel, as in blaze's lowering.
"""

import pytest
import torch
import ttnn

E = ttnn.experimental

GATHER_CORE = ttnn.CoreCoord(0, 0)
OTHER_CORE = ttnn.CoreCoord(1, 0)
NUM_CORES = 2
ROW_WORDS = 256  # one slice: a row of 256 uint32
SLICE_BYTES = ROW_WORDS * 4
ARENA_ROWS_PER_CORE = 16  # 16 KiB of arena per core
REGION_OFFSET = 4 * SLICE_BYTES  # the gathered region, on the gather core
STAGE_OFFSET = 8 * SLICE_BYTES  # where the other core stages its slice
NUM_ITERS = 4

KERNEL_INCLUDES = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/local_tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
"""

GATHER_SOURCE = (
    KERNEL_INCLUDES
    + """
void kernel_main() {
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    constexpr uint32_t num_cores = get_arg(args::num_cores);
    constexpr uint32_t slice_bytes = get_arg(args::slice_bytes);
    constexpr uint32_t region_offset = get_arg(args::region_offset);
    const uint32_t other_x = get_arg(args::other_x);
    const uint32_t other_y = get_arg(args::other_y);
    Noc noc;
    DataflowBuffer region(dfb::region);
    Semaphore<> ready(sem::ready);
    Semaphore<> valid(sem::valid);
    const auto in = TensorAccessor(tensor::in);
    const LocalTensorAccessor<uint32_t> arena(tensor::arena);
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        region.reserve_back(1);  // The drain kernel has copied the previous iteration out.
#ifndef SKIP_READY_CREDIT
        ready.up(noc, other_x, other_y, 1);
#endif
        noc.async_read(in, arena, slice_bytes, {.page_id = iter * num_cores}, {.offset_bytes = region_offset});
        noc.async_read_barrier();
#ifndef SKIP_VALID_WAIT
        valid.down(num_cores - 1);
#endif
        region.push_back(1);
    }
}
"""
)

DRAIN_SOURCE = (
    KERNEL_INCLUDES
    + """
void kernel_main() {
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    constexpr uint32_t num_cores = get_arg(args::num_cores);
    constexpr uint32_t slice_bytes = get_arg(args::slice_bytes);
    Noc noc;
    DataflowBuffer region(dfb::region);
    const auto out = TensorAccessor(tensor::out);
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        region.wait_front(1);
        for (uint32_t core = 0; core < num_cores; ++core) {
            noc.async_write(
                region, out, slice_bytes, {.offset_bytes = core * slice_bytes}, {.page_id = iter * num_cores + core});
        }
        noc.async_write_barrier();
        region.pop_front(1);
    }
}
"""
)

SEND_SOURCE = (
    KERNEL_INCLUDES
    + """
void kernel_main() {
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    constexpr uint32_t num_cores = get_arg(args::num_cores);
    constexpr uint32_t slice_bytes = get_arg(args::slice_bytes);
    constexpr uint32_t stage_offset = get_arg(args::stage_offset);
    constexpr uint32_t region_page = get_arg(args::region_page);
    const uint32_t core_index = get_arg(args::core_index);
    const uint32_t gather_x = get_arg(args::gather_x);
    const uint32_t gather_y = get_arg(args::gather_y);
    Noc noc;
    Semaphore<> ready(sem::ready);
    Semaphore<> valid(sem::valid);
    const auto in = TensorAccessor(tensor::in);
    const LocalTensorAccessor<uint32_t> stage(tensor::arena);
    const auto arena = TensorAccessor(tensor::arena);  // page p is row p of the arena, on the core that holds it
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        noc.async_read(in, stage, slice_bytes, {.page_id = iter * num_cores + core_index}, {.offset_bytes = stage_offset});
        noc.async_read_barrier();
#ifndef SKIP_READY_CREDIT
        ready.down(1);  // The gather core's region is free for this iteration.
#endif
        noc.async_write(stage, arena, slice_bytes, {.offset_bytes = stage_offset}, {.page_id = region_page + core_index});
        noc.async_write_barrier();  // The slice lands before the signal that says it is there.
        valid.up(noc, gather_x, gather_y, 1);
    }
}
"""
)


def make_io(device, seed):
    torch.manual_seed(seed)
    rows = NUM_ITERS * NUM_CORES
    torch_in = torch.randint(0, 1 << 30, (rows, ROW_WORDS), dtype=torch.int32)
    to_dram = dict(dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(GATHER_CORE, OTHER_CORE)})
    arena_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [ARENA_ROWS_PER_CORE, ROW_WORDS], ttnn.ShardOrientation.ROW_MAJOR),
    )
    tensors = {
        "in": ttnn.from_torch(torch_in, **to_dram),
        "arena": ttnn.allocate_tensor_on_device(
            ttnn.Shape([NUM_CORES * ARENA_ROWS_PER_CORE, ROW_WORDS]),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            device,
            arena_config,
        ),
        "out": ttnn.from_torch(torch.zeros_like(torch_in), **to_dram),
    }
    return torch_in, tensors


def dm_kernel(unique_id, source, hw_config, defines, **fields):
    return E.KernelSpec(
        unique_id=unique_id,
        source=E.KernelSpec.SourceCode(code=source),
        hw_config=hw_config,
        compiler_options=E.KernelSpec.CompilerOptions(defines=defines),
        **fields,
    )


def gather_program(device, tensors, defines):
    gather_noc = device.worker_core_from_logical_core(GATHER_CORE)
    other_noc = device.worker_core_from_logical_core(OTHER_CORE)
    common = {"num_iters": NUM_ITERS, "num_cores": NUM_CORES, "slice_bytes": SLICE_BYTES}
    semaphore_bindings = [
        E.SemaphoreBinding(semaphore_spec_name=name, accessor_name=name) for name in ("ready", "valid")
    ]
    reader = E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True)
    writer = E.create_writer_datamovement_config(disable_dfb_implicit_sync_for_all=True)
    spec = E.ProgramSpec(
        name="gather",
        kernels=[
            dm_kernel(
                "gather",
                GATHER_SOURCE,
                reader,
                defines,
                dfb_bindings=[E.producer_of("region", "region")],
                semaphore_bindings=semaphore_bindings,
                tensor_bindings=[
                    E.TensorBinding(tensor_parameter_name=name, accessor_name=name) for name in ("in", "arena")
                ],
                compile_time_args={**common, "region_offset": REGION_OFFSET},
                runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["other_x", "other_y"]),
            ),
            dm_kernel(
                "drain",
                DRAIN_SOURCE,
                writer,
                defines,
                dfb_bindings=[E.consumer_of("region", "region")],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name="out", accessor_name="out")],
                compile_time_args=common,
            ),
            dm_kernel(
                "send",
                SEND_SOURCE,
                reader,
                defines,
                semaphore_bindings=semaphore_bindings,
                tensor_bindings=[
                    E.TensorBinding(tensor_parameter_name=name, accessor_name=name) for name in ("in", "arena")
                ],
                compile_time_args={**common, "stage_offset": STAGE_OFFSET, "region_page": REGION_OFFSET // SLICE_BYTES},
                runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
                    runtime_arg_names=["core_index", "gather_x", "gather_y"]
                ),
            ),
        ],
        dataflow_buffers=[
            E.DataflowBufferSpec(
                unique_id="region",
                entry_size=NUM_CORES * SLICE_BYTES,
                num_entries=1,
                borrowed_from="arena",
                advanced_options=E.DFBAdvancedOptions(borrowed_offset=REGION_OFFSET),
            )
        ],
        semaphores=[
            E.SemaphoreSpec(unique_id=name, target_nodes=ttnn.CoreRange(GATHER_CORE, OTHER_CORE))
            for name in ("ready", "valid")
        ],
        tensor_parameters=[E.TensorParameter(unique_id=name, spec=tensor.spec) for name, tensor in tensors.items()],
        work_units=[
            E.WorkUnitSpec(name="gather_core", kernels=["gather", "drain"], target_nodes=GATHER_CORE),
            E.WorkUnitSpec(name="other_core", kernels=["send"], target_nodes=OTHER_CORE),
        ],
    )
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.ProgramRunArgs.KernelRunArgs(
                kernel="gather",
                runtime_arg_values={"other_x": {GATHER_CORE: other_noc.x}, "other_y": {GATHER_CORE: other_noc.y}},
            ),
            E.ProgramRunArgs.KernelRunArgs(
                kernel="send",
                runtime_arg_values={
                    "core_index": {OTHER_CORE: 1},
                    "gather_x": {OTHER_CORE: gather_noc.x},
                    "gather_y": {OTHER_CORE: gather_noc.y},
                },
            ),
        ],
        tensor_args=tensors,
    )
    return spec, run_args


def run_gather(device, seed, defines=None):
    torch_in, tensors = make_io(device, seed)
    spec, run_args = gather_program(device, tensors, defines or {})
    E.generic_op(list(tensors.values()), spec, run_args)
    return torch_in, ttnn.to_torch(tensors["out"]).to(torch.int32)


def test_generic_spec_op_gather(device):
    if device.compute_with_storage_grid_size().x < 2:
        pytest.skip("Needs two cores in a row")
    # The second call reuses the cached program with new tensors.
    entries = device.num_program_cache_entries()
    for seed in range(2):
        expected, result = run_gather(device, seed)
        assert torch.equal(result, expected)
    assert device.num_program_cache_entries() == entries + 1
