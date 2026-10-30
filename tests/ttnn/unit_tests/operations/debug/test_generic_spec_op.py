# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# ttnn.generic_op with a Metal 2.0 ProgramSpec built in Python. It rebuilds the tilized, sharded,
# same-dtype branch of ttnn.clone's program factory (clone_program_factory.cpp) on its kernels.

import os

import pytest
import torch
import ttnn
from tt_py_test_utils_common.utility_functions import is_quasar

CLONE_KERNELS = "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels"
READER_SOURCE = f"{CLONE_KERNELS}/read_kernel_sharded.cpp"
WRITER_SOURCE = f"{CLONE_KERNELS}/write_kernel_sharded.cpp"


def make_sharded_copy_program(input_tensor, output_tensor, shard_grid, tiles_per_shard, inline_writer_source):
    tile_size = input_tensor.tile.get_tile_size(input_tensor.dtype)
    if inline_writer_source:
        with open(os.path.join(os.environ["TT_METAL_HOME"], WRITER_SOURCE)) as f:
            writer_source = {"source_code": f.read()}
    else:
        writer_source = {"source": WRITER_SOURCE}

    spec = ttnn.ProgramSpec(
        name="sharded_copy",
        dataflow_buffers=[
            ttnn.DataflowBufferSpec(
                unique_id="src", entry_size=tile_size, num_entries=2, data_format=input_tensor.dtype
            )
        ],
        tensor_parameters=[
            ttnn.TensorParameter(unique_id="input", spec=input_tensor.spec),
            ttnn.TensorParameter(unique_id="output", spec=output_tensor.spec),
        ],
        kernels=[
            ttnn.KernelSpec(
                unique_id="reader",
                source=READER_SOURCE,
                hw_config=ttnn.DataMovementHardwareConfig.reader(),
                dfb_bindings=[
                    ttnn.DFBBinding(
                        dfb_spec_name="src", accessor_name="src", endpoint_type=ttnn.DFBEndpointType.PRODUCER
                    )
                ],
                tensor_bindings=[ttnn.TensorBinding(tensor_parameter_name="input", accessor_name="input")],
                runtime_arg_names=["num_tiles"],
            ),
            ttnn.KernelSpec(
                unique_id="writer",
                **writer_source,
                hw_config=ttnn.DataMovementHardwareConfig.writer(),
                # The writer drains the reader's DFB directly: there is no compute stage.
                dfb_bindings=[
                    ttnn.DFBBinding(
                        dfb_spec_name="src", accessor_name="dst", endpoint_type=ttnn.DFBEndpointType.CONSUMER
                    )
                ],
                tensor_bindings=[ttnn.TensorBinding(tensor_parameter_name="output", accessor_name="output")],
                runtime_arg_names=["num_tiles"],
            ),
        ],
        work_units=[ttnn.WorkUnitSpec(name="main", kernels=["reader", "writer"], target_nodes=shard_grid)],
    )

    num_tiles = {}
    for core_range in shard_grid.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                num_tiles[ttnn.CoreCoord(x, y)] = tiles_per_shard
    run_args = ttnn.ProgramSpecRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(kernel="reader", runtime_arg_values={"num_tiles": num_tiles}),
            ttnn.KernelRunArgs(kernel="writer", runtime_arg_values={"num_tiles": num_tiles}),
        ],
        tensor_args={"input": input_tensor, "output": output_tensor},
    )
    return spec, run_args


@pytest.mark.parametrize("inline_writer_source", [False, True], ids=["source_path", "source_code"])
@pytest.mark.parametrize(
    "shape, shard_grid",
    [
        ([1, 1, 128, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})),
        ([1, 1, 128, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})),
    ],
    ids=["1_core", "2_cores"],
)
def test_generic_spec_op_sharded_copy(device, shape, shard_grid, inline_writer_source):
    num_cores = shard_grid.num_cores()
    shard_shape = (shape[-2] // num_cores, shape[-1])
    tiles_per_shard = (shard_shape[0] // 32) * (shard_shape[1] // 32)
    memory_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Two dispatches: the second is a program-cache hit that must rebind the new tensors.
    cache_entries = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        torch_input = torch.randn(shape, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(
            torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )
        output_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory_config
        )
        spec, run_args = make_sharded_copy_program(
            input_tensor, output_tensor, shard_grid, tiles_per_shard, inline_writer_source
        )

        result = ttnn.generic_op([input_tensor, output_tensor], spec, run_args)

        assert torch.equal(ttnn.to_torch(result), torch_input)
        cache_entries.append(device.num_program_cache_entries())
    assert cache_entries[0] == cache_entries[1]


# tt-blaze's single-core L1 -> L1 Copy (blaze/ops/copy/kernels/metal2/{stage,flush}.cpp), given inline.
BLAZE_STAGE_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_pages = get_arg(args::num_pages);

    Noc noc;
    DataflowBuffer staged(dfb::staged);
    const LocalTensorAccessor<uint32_t> src(tensor::src);

    const uint32_t num_bytes = staged.get_tile_size() * num_pages;
    staged.reserve_back(num_pages);
    noc.async_read(src, staged, num_bytes, {.offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    staged.push_back(num_pages);
}
"""

BLAZE_FLUSH_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_pages = get_arg(args::num_pages);

    Noc noc;
    DataflowBuffer staged(dfb::staged);
    const LocalTensorAccessor<uint32_t> dst(tensor::dst);

    const uint32_t num_bytes = staged.get_tile_size() * num_pages;
    staged.wait_front(num_pages);
    noc.async_write(staged, dst, num_bytes, {.offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_write_barrier();
    staged.pop_front(num_pages);
}
"""


def test_generic_spec_op_blaze_copy(device):
    width = 512
    tile = ttnn.Tile((1, 32))
    num_pages = width // 32
    core = ttnn.CoreCoord(1, 0)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [1, width], ttnn.ShardOrientation.ROW_MAJOR),
    )

    torch_input = torch.arange(width, dtype=torch.float32).reshape(1, width).mul(0.5).sub(7.0).to(torch.bfloat16)
    src = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=tile, device=device, memory_config=memory_config
    )
    dst = ttnn.from_torch(
        torch.zeros_like(torch_input),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        tile=tile,
        device=device,
        memory_config=memory_config,
    )

    spec = ttnn.ProgramSpec(
        name="blaze_copy_metal2",
        dataflow_buffers=[
            ttnn.DataflowBufferSpec(
                unique_id="staged",
                entry_size=src.tile.get_tile_size(ttnn.bfloat16),
                num_entries=num_pages,
                data_format=ttnn.bfloat16,
                # Without the tile shape, get_tile_size() on Wormhole / Blackhole reports a 32x32 tile.
                tile=tile,
            )
        ],
        tensor_parameters=[
            ttnn.TensorParameter(unique_id="src", spec=src.spec),
            ttnn.TensorParameter(unique_id="dst", spec=dst.spec),
        ],
        kernels=[
            ttnn.KernelSpec(
                unique_id="stage",
                source_code=BLAZE_STAGE_SOURCE,
                hw_config=ttnn.DataMovementHardwareConfig.reader(),
                dfb_bindings=[
                    ttnn.DFBBinding(
                        dfb_spec_name="staged", accessor_name="staged", endpoint_type=ttnn.DFBEndpointType.PRODUCER
                    )
                ],
                tensor_bindings=[ttnn.TensorBinding(tensor_parameter_name="src", accessor_name="src")],
                compile_time_args={"num_pages": num_pages},
            ),
            ttnn.KernelSpec(
                unique_id="flush",
                source_code=BLAZE_FLUSH_SOURCE,
                hw_config=ttnn.DataMovementHardwareConfig.writer(),
                dfb_bindings=[
                    ttnn.DFBBinding(
                        dfb_spec_name="staged", accessor_name="staged", endpoint_type=ttnn.DFBEndpointType.CONSUMER
                    )
                ],
                tensor_bindings=[ttnn.TensorBinding(tensor_parameter_name="dst", accessor_name="dst")],
                compile_time_args={"num_pages": num_pages},
            ),
        ],
        work_units=[ttnn.WorkUnitSpec(name="main", kernels=["stage", "flush"], target_nodes=core)],
    )
    run_args = ttnn.ProgramSpecRunArgs(tensor_args={"src": src, "dst": dst})

    result = ttnn.generic_op([src, dst], spec, run_args)

    assert torch.equal(ttnn.to_torch(result), torch_input)


COMPUTE_SOURCE = f"{CLONE_KERNELS}/compute_kernel.cpp"


# Rebuilds clone's dtype-converting branch: reader -> src DFB -> compute copy -> dst DFB -> writer.
@pytest.mark.parametrize(
    "input_dtype, output_dtype, unpack_mode",
    [
        (ttnn.bfloat16, ttnn.bfloat16, None),
        (ttnn.bfloat16, ttnn.float32, None),
        (ttnn.float32, ttnn.float32, "UnpackToSrc"),
        (ttnn.float32, ttnn.float32, "UnpackToDest"),
    ],
    ids=["bf16_to_bf16", "bf16_to_fp32", "fp32_unpack_to_src", "fp32_unpack_to_dest"],
)
def test_generic_spec_op_compute_copy(device, input_dtype, output_dtype, unpack_mode):
    shape = [1, 1, 64, 64]
    num_tiles = (shape[-2] // 32) * (shape[-1] // 32)
    core = ttnn.CoreCoord(0, 0)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})
    memory_config = ttnn.create_sharded_memory_config(
        shape=(shape[-2], shape[-1]),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32)
    input_tensor = ttnn.from_torch(
        torch_input, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), output_dtype, ttnn.TILE_LAYOUT, device, memory_config
    )

    fp32_dest = ttnn.float32 in (input_dtype, output_dtype)
    if unpack_mode is None:
        compute_hw = ttnn.ComputeHardwareConfig.from_compute_kernel_config(
            ttnn.init_device_compute_kernel_config(
                device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_dest
            )
        )
    else:
        compute_hw = ttnn.ComputeHardwareConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            enable_32_bit_dest=fp32_dest,
            unpack_modes={"src": getattr(ttnn.UnpackMode, unpack_mode)},
        )

    def dfb(unique_id, dtype):
        tile_size = input_tensor.tile.get_tile_size(dtype)
        return ttnn.DataflowBufferSpec(unique_id=unique_id, entry_size=tile_size, num_entries=2, data_format=dtype)

    spec = ttnn.ProgramSpec(
        name="compute_copy",
        dataflow_buffers=[dfb("src", input_dtype), dfb("dst", output_dtype)],
        tensor_parameters=[
            ttnn.TensorParameter(unique_id="input", spec=input_tensor.spec),
            ttnn.TensorParameter(unique_id="output", spec=output_tensor.spec),
        ],
        kernels=[
            ttnn.KernelSpec(
                unique_id="reader",
                source=READER_SOURCE,
                hw_config=ttnn.DataMovementHardwareConfig.reader(),
                dfb_bindings=[
                    ttnn.DFBBinding(
                        dfb_spec_name="src", accessor_name="src", endpoint_type=ttnn.DFBEndpointType.PRODUCER
                    )
                ],
                tensor_bindings=[ttnn.TensorBinding(tensor_parameter_name="input", accessor_name="input")],
                runtime_arg_names=["num_tiles"],
            ),
            ttnn.KernelSpec(
                unique_id="compute",
                source=COMPUTE_SOURCE,
                hw_config=compute_hw,
                dfb_bindings=[
                    ttnn.DFBBinding(
                        dfb_spec_name="src", accessor_name="src", endpoint_type=ttnn.DFBEndpointType.CONSUMER
                    ),
                    ttnn.DFBBinding(
                        dfb_spec_name="dst", accessor_name="dst", endpoint_type=ttnn.DFBEndpointType.PRODUCER
                    ),
                ],
                compile_time_args={"num_tiles": num_tiles},
                opt_level=ttnn.KernelBuildOptLevel.O3,
            ),
            ttnn.KernelSpec(
                unique_id="writer",
                source=WRITER_SOURCE,
                hw_config=ttnn.DataMovementHardwareConfig.writer(),
                dfb_bindings=[
                    ttnn.DFBBinding(
                        dfb_spec_name="dst", accessor_name="dst", endpoint_type=ttnn.DFBEndpointType.CONSUMER
                    )
                ],
                tensor_bindings=[ttnn.TensorBinding(tensor_parameter_name="output", accessor_name="output")],
                runtime_arg_names=["num_tiles"],
            ),
        ],
        work_units=[ttnn.WorkUnitSpec(name="main", kernels=["reader", "compute", "writer"], target_nodes=grid)],
    )
    run_args = ttnn.ProgramSpecRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(kernel="reader", runtime_arg_values={"num_tiles": {core: num_tiles}}),
            ttnn.KernelRunArgs(kernel="writer", runtime_arg_values={"num_tiles": {core: num_tiles}}),
        ],
        tensor_args={"input": input_tensor, "output": output_tensor},
    )

    result = ttnn.to_torch(ttnn.generic_op([input_tensor, output_tensor], spec, run_args)).to(torch.float32)

    expected = ttnn.to_torch(input_tensor).to(torch.float32)
    if unpack_mode == "UnpackToSrc":
        # Unpacking to the source registers truncates fp32 to TF32, so the result is close but not exact.
        assert torch.allclose(result, expected, rtol=2e-3, atol=1e-3)
        assert not torch.equal(result, expected)
    else:
        assert torch.equal(result, expected)


# Reads this core's input shard into the scratchpad, adds a constant on the RISC-V, and writes the output shard.
SCRATCHPAD_ADD_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/scratchpad.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t addend = get_arg(args::addend);

    Noc noc;
    Scratchpad<uint32_t> pad(scratch::pad);
    const LocalTensorAccessor<uint32_t> src(tensor::src);
    const LocalTensorAccessor<uint32_t> dst(tensor::dst);

    noc.async_read(src, pad, pad.size_in_bytes(), {.offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    {
        // On Quasar the RISC-V reaches L1 through write-back caches: the lock invalidates them on entry, so the
        // loads see what the NoC wrote, and flushes them on exit, so the NoC write below sends the new values.
        // It is a no-op on Wormhole / Blackhole.
        auto lock = pad.scoped_lock(0, pad.size());
        for (uint32_t i = 0; i < pad.size(); ++i) {
            pad[i] += addend;
        }
    }
    noc.async_write(pad, dst, pad.size_in_bytes(), {.offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_write_barrier();
}
"""


@pytest.mark.parametrize(
    "shard_grid",
    [
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
    ],
    ids=["1_core", "2_cores"],
)
def test_generic_spec_op_scratchpad(device, shard_grid):
    shape = [1, 1, 64, 64]
    addend = 7
    shard_shape = (shape[-2] // shard_grid.num_cores(), shape[-1])
    memory_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    torch_input = torch.randint(0, 1 << 20, shape, dtype=torch.int32)
    src = ttnn.from_torch(
        torch_input, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    dst = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.uint32, ttnn.TILE_LAYOUT, device, memory_config)

    spec = ttnn.ProgramSpec(
        name="scratchpad_add",
        # No data_format: only compute kernels need one, and Quasar rejects UInt32 there.
        scratchpads=[ttnn.ScratchpadSpec(unique_id="pad", size_per_node=shard_shape[0] * shard_shape[1] * 4)],
        tensor_parameters=[
            ttnn.TensorParameter(unique_id="src", spec=src.spec),
            ttnn.TensorParameter(unique_id="dst", spec=dst.spec),
        ],
        kernels=[
            ttnn.KernelSpec(
                unique_id="add",
                source_code=SCRATCHPAD_ADD_SOURCE,
                hw_config=ttnn.DataMovementHardwareConfig.reader(),
                scratchpad_bindings=[ttnn.ScratchpadBinding(scratchpad_spec_name="pad", accessor_name="pad")],
                tensor_bindings=[
                    ttnn.TensorBinding(tensor_parameter_name="src", accessor_name="src"),
                    ttnn.TensorBinding(tensor_parameter_name="dst", accessor_name="dst"),
                ],
                compile_time_args={"addend": addend},
            )
        ],
        work_units=[ttnn.WorkUnitSpec(name="main", kernels=["add"], target_nodes=shard_grid)],
    )
    run_args = ttnn.ProgramSpecRunArgs(tensor_args={"src": src, "dst": dst})

    result = ttnn.to_torch(ttnn.generic_op([src, dst], spec, run_args)).to(torch.int32)

    assert torch.equal(result, torch_input + addend)


# Same core, two kernels: "stage" copies the input shard to the output shard, then signals "ready"; "finish" waits
# for it and adds a constant to the output shard in place. Each kernel has its own scratchpad.
SEMAPHORE_STAGE_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/scratchpad.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;
    Scratchpad<uint32_t> pad(scratch::pad);
    Semaphore ready(sem::ready);
    const LocalTensorAccessor<uint32_t> src(tensor::src);
    const LocalTensorAccessor<uint32_t> dst(tensor::dst);

    noc.async_read(src, pad, pad.size_in_bytes(), {.offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    noc.async_write(pad, dst, pad.size_in_bytes(), {.offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_write_barrier();
    ready.up(1);
}
"""

SEMAPHORE_FINISH_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/scratchpad.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t addend = get_arg(args::addend);

    Noc noc;
    Scratchpad<uint32_t> pad(scratch::pad);
    Semaphore ready(sem::ready);
    const LocalTensorAccessor<uint32_t> dst(tensor::dst);

    ready.down(1);  // Blocks until "stage" has written the output shard.
    noc.async_read(dst, pad, pad.size_in_bytes(), {.offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    {
        auto lock = pad.scoped_lock(0, pad.size());
        for (uint32_t i = 0; i < pad.size(); ++i) {
            pad[i] += addend;
        }
    }
    noc.async_write(pad, dst, pad.size_in_bytes(), {.offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_write_barrier();
}
"""


def make_uint32_shards(device, shard_grid, shape=(1, 1, 64, 64)):
    shard_shape = (shape[-2] // shard_grid.num_cores(), shape[-1])
    memory_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    torch_input = torch.randint(0, 1 << 20, shape, dtype=torch.int32)
    src = ttnn.from_torch(
        torch_input, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    dst = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.uint32, ttnn.TILE_LAYOUT, device, memory_config)
    return torch_input, src, dst, shard_shape[0] * shard_shape[1] * 4


def test_generic_spec_op_semaphore_same_core(device):
    addend = 5
    core = ttnn.CoreCoord(0, 0)
    torch_input, src, dst, shard_bytes = make_uint32_shards(device, ttnn.CoreRangeSet({ttnn.CoreRange(core, core)}))

    spec = ttnn.ProgramSpec(
        name="semaphore_same_core",
        semaphores=[ttnn.SemaphoreSpec(unique_id="ready", target_nodes=core)],
        scratchpads=[
            ttnn.ScratchpadSpec(unique_id="stage_pad", size_per_node=shard_bytes),
            ttnn.ScratchpadSpec(unique_id="finish_pad", size_per_node=shard_bytes),
        ],
        tensor_parameters=[
            ttnn.TensorParameter(unique_id="src", spec=src.spec),
            ttnn.TensorParameter(unique_id="dst", spec=dst.spec),
        ],
        kernels=[
            ttnn.KernelSpec(
                unique_id="stage",
                source_code=SEMAPHORE_STAGE_SOURCE,
                hw_config=ttnn.DataMovementHardwareConfig.reader(),
                semaphore_bindings=[ttnn.SemaphoreBinding(semaphore_spec_name="ready", accessor_name="ready")],
                scratchpad_bindings=[ttnn.ScratchpadBinding(scratchpad_spec_name="stage_pad", accessor_name="pad")],
                tensor_bindings=[
                    ttnn.TensorBinding(tensor_parameter_name="src", accessor_name="src"),
                    ttnn.TensorBinding(tensor_parameter_name="dst", accessor_name="dst"),
                ],
            ),
            ttnn.KernelSpec(
                unique_id="finish",
                source_code=SEMAPHORE_FINISH_SOURCE,
                hw_config=ttnn.DataMovementHardwareConfig.writer(),
                semaphore_bindings=[ttnn.SemaphoreBinding(semaphore_spec_name="ready", accessor_name="ready")],
                scratchpad_bindings=[ttnn.ScratchpadBinding(scratchpad_spec_name="finish_pad", accessor_name="pad")],
                tensor_bindings=[ttnn.TensorBinding(tensor_parameter_name="dst", accessor_name="dst")],
                compile_time_args={"addend": addend},
            ),
        ],
        work_units=[ttnn.WorkUnitSpec(name="main", kernels=["stage", "finish"], target_nodes=core)],
    )
    run_args = ttnn.ProgramSpecRunArgs(tensor_args={"src": src, "dst": dst})

    result = ttnn.to_torch(ttnn.generic_op([src, dst], spec, run_args)).to(torch.int32)

    assert torch.equal(result, torch_input + addend)


# Two cores: each adds a constant to its own shard, then signals the other core over the NoC and waits for the other
# core's signal. A semaphore that cannot reach the peer hangs the test.
SEMAPHORE_PING_PONG_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/scratchpad.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t addend = get_arg(args::addend);
    const uint32_t peer_x = get_arg(args::peer_x);
    const uint32_t peer_y = get_arg(args::peer_y);

    Noc noc;
    Scratchpad<uint32_t> pad(scratch::pad);
    Semaphore done(sem::done);
    const LocalTensorAccessor<uint32_t> src(tensor::src);
    const LocalTensorAccessor<uint32_t> dst(tensor::dst);

    noc.async_read(src, pad, pad.size_in_bytes(), {.offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    {
        auto lock = pad.scoped_lock(0, pad.size());
        for (uint32_t i = 0; i < pad.size(); ++i) {
            pad[i] += addend;
        }
    }
    noc.async_write(pad, dst, pad.size_in_bytes(), {.offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_write_barrier();

    done.up(noc, peer_x, peer_y, 1);
    done.down(1);  // Blocks until the peer has signalled; leaves the semaphore at 0.
}
"""


def test_generic_spec_op_semaphore_cross_core(device):
    addend = 9
    cores = [ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)]
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(cores[0], cores[1])})
    torch_input, src, dst, shard_bytes = make_uint32_shards(device, grid)

    spec = ttnn.ProgramSpec(
        name="semaphore_cross_core",
        semaphores=[ttnn.SemaphoreSpec(unique_id="done", target_nodes=grid)],
        scratchpads=[ttnn.ScratchpadSpec(unique_id="pad", size_per_node=shard_bytes)],
        tensor_parameters=[
            ttnn.TensorParameter(unique_id="src", spec=src.spec),
            ttnn.TensorParameter(unique_id="dst", spec=dst.spec),
        ],
        kernels=[
            ttnn.KernelSpec(
                unique_id="ping_pong",
                source_code=SEMAPHORE_PING_PONG_SOURCE,
                hw_config=ttnn.DataMovementHardwareConfig.reader(),
                semaphore_bindings=[ttnn.SemaphoreBinding(semaphore_spec_name="done", accessor_name="done")],
                scratchpad_bindings=[ttnn.ScratchpadBinding(scratchpad_spec_name="pad", accessor_name="pad")],
                tensor_bindings=[
                    ttnn.TensorBinding(tensor_parameter_name="src", accessor_name="src"),
                    ttnn.TensorBinding(tensor_parameter_name="dst", accessor_name="dst"),
                ],
                compile_time_args={"addend": addend},
                runtime_arg_names=["peer_x", "peer_y"],
            )
        ],
        work_units=[ttnn.WorkUnitSpec(name="main", kernels=["ping_pong"], target_nodes=grid)],
    )
    ping_pong = ttnn.KernelRunArgs(kernel="ping_pong")
    for core, peer in zip(cores, reversed(cores)):
        peer_noc = device.worker_core_from_logical_core(peer)
        ping_pong.add_runtime_args_for_node(core, {"peer_x": peer_noc.x, "peer_y": peer_noc.y})
    run_args = ttnn.ProgramSpecRunArgs(kernel_run_args=[ping_pong], tensor_args={"src": src, "dst": dst})

    result = ttnn.to_torch(ttnn.generic_op([src, dst], spec, run_args)).to(torch.int32)

    assert torch.equal(result, torch_input + addend)


# Negative tests: specs that tt-metal rejects must reach Python as a RuntimeError that names the problem.
# Several are Quasar (Gen2) rules that Wormhole / Blackhole accept.

EMPTY_KERNEL_SOURCE = "void kernel_main() {}\n"


def dm_kernel(unique_id, dfb_bindings=(), num_threads=1, hw_config=None):
    return ttnn.KernelSpec(
        unique_id=unique_id,
        source_code=EMPTY_KERNEL_SOURCE,
        hw_config=hw_config or ttnn.DataMovementHardwareConfig.reader(),
        num_threads=num_threads,
        dfb_bindings=list(dfb_bindings),
    )


def producer(dfb_name):
    return ttnn.DFBBinding(dfb_spec_name=dfb_name, accessor_name=dfb_name, endpoint_type=ttnn.DFBEndpointType.PRODUCER)


def consumer(dfb_name):
    return ttnn.DFBBinding(dfb_spec_name=dfb_name, accessor_name=dfb_name, endpoint_type=ttnn.DFBEndpointType.CONSUMER)


def small_dfb(unique_id, dtype=ttnn.bfloat16):
    return ttnn.DataflowBufferSpec(unique_id=unique_id, entry_size=2048, num_entries=2, data_format=dtype)


def run_on_one_core(device, kernels, dataflow_buffers=(), scratchpads=()):
    """Runs a spec of the given kernels on core (0, 0); every test below expects it to be rejected."""
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
    )
    spec = ttnn.ProgramSpec(
        name="invalid",
        kernels=kernels,
        dataflow_buffers=list(dataflow_buffers),
        scratchpads=list(scratchpads),
        work_units=[
            ttnn.WorkUnitSpec(name="main", kernels=[k.unique_id for k in kernels], target_nodes=ttnn.CoreCoord(0, 0))
        ],
    )
    ttnn.generic_op([output], spec, ttnn.ProgramSpecRunArgs())


def test_generic_spec_op_rejects_source_and_source_code(expect_error):
    with expect_error(RuntimeError, "give exactly one of source"):
        ttnn.KernelSpec(
            unique_id="k",
            source=READER_SOURCE,
            source_code=EMPTY_KERNEL_SOURCE,
            hw_config=ttnn.DataMovementHardwareConfig.reader(),
        )


def test_generic_spec_op_rejects_tensor_outside_io_tensors(device, expect_error):
    output = ttnn.allocate_tensor_on_device(ttnn.Shape([32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    stray = ttnn.allocate_tensor_on_device(ttnn.Shape([32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    spec = ttnn.ProgramSpec(
        name="stray_tensor",
        kernels=[dm_kernel("k")],
        tensor_parameters=[ttnn.TensorParameter(unique_id="t", spec=stray.spec)],
        work_units=[ttnn.WorkUnitSpec(name="main", kernels=["k"], target_nodes=ttnn.CoreCoord(0, 0))],
    )
    with expect_error(RuntimeError, "the tensor bound to TensorParameter 't' is not in io_tensors"):
        ttnn.generic_op([output], spec, ttnn.ProgramSpecRunArgs(tensor_args={"t": stray}))


def test_generic_spec_op_rejects_fp32_unpack_to_dest_without_32_bit_dest(device, expect_error):
    compute = ttnn.KernelSpec(
        unique_id="compute",
        source_code=EMPTY_KERNEL_SOURCE,
        hw_config=ttnn.ComputeHardwareConfig(
            enable_32_bit_dest=False, unpack_modes={"in": ttnn.UnpackMode.UnpackToDest}
        ),
        dfb_bindings=[consumer("in")],
    )
    with expect_error(RuntimeError, "specifies UnpackToDest, but the DFB entries use a 32-bit format"):
        run_on_one_core(
            device,
            [dm_kernel("reader", [producer("in")]), compute],
            dataflow_buffers=[
                ttnn.DataflowBufferSpec(unique_id="in", entry_size=4096, num_entries=2, data_format=ttnn.float32)
            ],
        )


def skip_unless_quasar():
    if not is_quasar():
        pytest.skip("Quasar (Gen2) rule; Wormhole / Blackhole accept this spec")


def test_generic_spec_op_rejects_dm_self_loop_on_quasar(device, expect_error):
    skip_unless_quasar()
    with expect_error(RuntimeError, "Self-loop DFBs are not supported for data-movement kernels on Gen2"):
        run_on_one_core(device, [dm_kernel("loop", [producer("buf"), consumer("buf")])], [small_dfb("buf")])


def test_generic_spec_op_rejects_two_producers(device, expect_error):
    kernels = [
        dm_kernel("producer_a", [producer("buf")]),
        dm_kernel("producer_b", [producer("buf")], hw_config=ttnn.DataMovementHardwareConfig.writer()),
        dm_kernel("consumer", [consumer("buf")]),
    ]
    with expect_error(RuntimeError, "must run exactly one producer and one consumer kernel instance"):
        run_on_one_core(device, kernels, [small_dfb("buf")])


# The spec may place at most 32 DFBs on a core. DM-to-DM DFBs run out sooner, when the program is built: each takes
# 2 of the 24 NoC transaction IDs reserved for DFBs, so a core holds at most 12 (before its 16 tile counters run out).
@pytest.mark.parametrize(
    "num_dfbs, message",
    [(33, "supports at most 32 device slots per core"), (13, "TxnIdAllocator exhausted")],
    ids=["slots", "noc_transaction_ids"],
)
def test_generic_spec_op_rejects_too_many_dfbs_on_quasar(device, expect_error, num_dfbs, message):
    skip_unless_quasar()
    names = [f"buf{i}" for i in range(num_dfbs)]
    kernels = [
        dm_kernel("producer", [producer(n) for n in names]),
        dm_kernel("consumer", [consumer(n) for n in names], hw_config=ttnn.DataMovementHardwareConfig.writer()),
    ]
    dfbs = [ttnn.DataflowBufferSpec(unique_id=n, entry_size=32, num_entries=2) for n in names]
    with expect_error(RuntimeError, message):
        run_on_one_core(device, kernels, dfbs)


def test_generic_spec_op_rejects_too_many_dm_threads(device, expect_error):
    # Quasar caps the threads per data movement kernel; Wormhole / Blackhole allow only one.
    message = "too many data movement threads" if is_quasar() else "does not support multi-threaded kernels"
    with expect_error(RuntimeError, message):
        run_on_one_core(device, [dm_kernel("wide", num_threads=16)])


def test_generic_spec_op_rejects_uint32_scratchpad_on_quasar(device, expect_error):
    skip_unless_quasar()
    pad = ttnn.ScratchpadSpec(unique_id="pad", size_per_node=4096, data_format=ttnn.uint32)
    kernel = ttnn.KernelSpec(
        unique_id="k",
        source_code=EMPTY_KERNEL_SOURCE,
        hw_config=ttnn.DataMovementHardwareConfig.reader(),
        scratchpad_bindings=[ttnn.ScratchpadBinding(scratchpad_spec_name="pad", accessor_name="pad")],
    )
    with expect_error(RuntimeError, "has data format 'UInt32' which is not supported on architecture quasar"):
        run_on_one_core(device, [kernel], scratchpads=[pad])
