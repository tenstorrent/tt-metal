# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# ttnn.generic_op with a Metal 2.0 ProgramSpec built in Python. It rebuilds the tilized, sharded,
# same-dtype branch of ttnn.clone's program factory (clone_program_factory.cpp) on its kernels.

import os

import pytest
import torch
import ttnn

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
