# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Decode-shaped matmul through ttnn.generic_op with a Metal 2.0 ProgramSpec.

out[M, N] = a[M, K] @ b[K, N], with a and out in tiny Tile([M, 32]) tiles and b in 32x32 tiles. Every tensor is an
L1 shard. b is width-sharded over the program's cores, and each core computes its own columns of out. b and out are
borrowed by their dataflow buffers, so no data is copied: one data movement kernel only announces the resident tiles,
the compute kernel reads them in place by index, and a second data movement kernel retires the output tiles, which
compute packed into the output shard. a is borrowed too when it is resident on the core; otherwise the first data
movement kernel reads it over the NoC.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

# Announces the resident tiles of b, then the tiles of a for each iteration: from its resident shard, or read from
# another core's shard.
PUSH_INPUTS_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t Kt = get_arg(args::Kt);
    constexpr uint32_t Nt = get_arg(args::Nt);
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    DataflowBuffer in0(dfb::in0);
    DataflowBuffer in1(dfb::in1);

    in1.reserve_back(Kt * Nt);
    in1.push_back(Kt * Nt);
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
#ifdef READ_IN0
        Noc noc;
        const auto a = TensorAccessor(tensor::in0);
        const uint32_t tile_size = in0.get_tile_size();
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            in0.reserve_back(1);
            noc.async_read(a, in0, tile_size, {.page_id = iter * Kt + kt, .offset_bytes = 0}, {.offset_bytes = 0});
            noc.async_read_barrier();
            in0.push_back(1);
        }
#else
        in0.reserve_back(Kt);
        in0.push_back(Kt);
#endif
    }
}
"""

# One output row block of Nt tiles per iteration. b stays at the front of in1 for every iteration.
MATMUL_SOURCE = """
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t Kt = get_arg(args::Kt);
    constexpr uint32_t Nt = get_arg(args::Nt);
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    DataflowBuffer in0(dfb::in0);
    DataflowBuffer in1(dfb::in1);
    DataflowBuffer out(dfb::out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::in0, dfb::in1, dfb::out);
#ifdef MATMUL_BLOCK
    matmul_block_init(dfb::in0, dfb::in1, /*transpose=*/0, /*ct_dim=*/Nt, /*rt_dim=*/1, /*kt_dim=*/Kt);
#else
    matmul_init(dfb::in0, dfb::in1, /*transpose=*/0);
#endif

    in1.wait_front(Kt * Nt);
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        in0.wait_front(Kt);
#ifdef MATMUL_BLOCK
        // All Nt output tiles at once, in DST 0..Nt-1. b's tiles are row-major, so row kt starts at kt * Nt.
        tile_regs_acquire();
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            matmul_block(dfb::in0, dfb::in1, kt, kt * Nt, 0, /*transpose=*/0, Nt, /*rt_dim=*/1, Kt);
        }
        tile_regs_commit();
        out.reserve_back(Nt);
        tile_regs_wait();
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            pack_tile(nt, dfb::out);
        }
        tile_regs_release();
        out.push_back(Nt);
#else
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                matmul_tiles(dfb::in0, dfb::in1, kt, kt * Nt + nt, 0);
            }
            tile_regs_commit();
            out.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, dfb::out);
            tile_regs_release();
            out.push_back(1);
        }
#endif
        in0.pop_front(Kt);
    }
    in1.pop_front(Kt * Nt);
}
"""

# Retires the output tiles: compute already packed them into the output shard.
POP_OUTPUT_SOURCE = """
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t Nt = get_arg(args::Nt);
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    DataflowBuffer out(dfb::out);
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        out.wait_front(Nt);
        out.pop_front(Nt);
    }
}
"""

FULL_TILE = ttnn.Tile([32, 32])


def core_grid(num_cores):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})


def to_device(device, tensor, dtype, tile, num_cores=1):
    """Width-shards tensor over num_cores cores in row 0 (one shard on core (0, 0) for num_cores=1)."""
    height, width = tensor.shape
    memory_config = ttnn.create_sharded_memory_config(
        shape=(height, width // num_cores),
        core_grid=core_grid(num_cores),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.from_torch(
        tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, tile=tile, device=device, memory_config=memory_config
    )


def make_weights(device, Kt, Nt, num_cores=1):
    """Returns (torch_b, b): b is [32 * Kt, 32 * Nt * num_cores], Nt tile columns per core."""
    torch_b = torch.randn((32 * Kt, 32 * Nt * num_cores), dtype=torch.bfloat16)
    return torch_b, to_device(device, torch_b, ttnn.bfloat16, FULL_TILE, num_cores)


def make_activation(device, M, Kt, num_iters=1):
    """Returns (torch_a, a): a holds one [M, 32 * Kt] activation per iteration, stacked by rows, on core (0, 0)."""
    torch_a = torch.randn((num_iters * M, 32 * Kt), dtype=torch.bfloat16)
    return torch_a, to_device(device, torch_a, ttnn.bfloat16, ttnn.Tile([M, 32]))


def make_output(device, M, b, num_iters=1, dtype=ttnn.bfloat16):
    num_cores = b.memory_config().shard_spec.grid.num_cores()
    zeros = torch.zeros((num_iters * M, b.shape[-1]), dtype=torch.float32)
    return to_device(device, zeros, dtype, ttnn.Tile([M, 32]), num_cores)


def run_decode_matmul(a, b, out, *, num_iters=1, matmul_block=False, enable_32_bit_dest=False):
    """Runs out = a @ b for each of a's num_iters row blocks; returns out as a float32 torch tensor."""
    grid = b.memory_config().shard_spec.grid
    tiny_tile = a.tile
    Kt = a.shape[-1] // 32
    Nt = b.memory_config().shard_spec.shape[1] // 32
    # a is one shard on core (0, 0), so it is resident only when the program runs on that core alone.
    read_in0 = grid.num_cores() > 1

    def dfb(name, tensor, tile, num_entries, borrowed=True):
        return ttnn.DataflowBufferSpec(
            unique_id=name,
            entry_size=tile.get_tile_size(tensor.dtype),
            num_entries=num_entries,
            data_format=tensor.dtype,
            tile=tile,
            borrowed_from=name if borrowed else None,
        )

    def binding(name, endpoint_type):
        return ttnn.DFBBinding(dfb_spec_name=name, accessor_name=name, endpoint_type=endpoint_type)

    producer, consumer = ttnn.DFBEndpointType.PRODUCER, ttnn.DFBEndpointType.CONSUMER
    sizes = {"Kt": Kt, "Nt": Nt, "num_iters": num_iters}
    spec = ttnn.ProgramSpec(
        name="decode_matmul",
        dataflow_buffers=[
            dfb("in0", a, tiny_tile, Kt if read_in0 else num_iters * Kt, borrowed=not read_in0),
            dfb("in1", b, FULL_TILE, Kt * Nt),
            dfb("out", out, tiny_tile, num_iters * Nt),
        ],
        tensor_parameters=[
            ttnn.TensorParameter(unique_id="in0", spec=a.spec),
            ttnn.TensorParameter(unique_id="in1", spec=b.spec),
            ttnn.TensorParameter(unique_id="out", spec=out.spec),
        ],
        kernels=[
            ttnn.KernelSpec(
                unique_id="push_inputs",
                source_code=PUSH_INPUTS_SOURCE,
                hw_config=ttnn.DataMovementHardwareConfig.reader(),
                dfb_bindings=[binding("in0", producer), binding("in1", producer)],
                tensor_bindings=(
                    [ttnn.TensorBinding(tensor_parameter_name="in0", accessor_name="in0")] if read_in0 else []
                ),
                compile_time_args=sizes,
                defines={"READ_IN0": "1"} if read_in0 else {},
            ),
            ttnn.KernelSpec(
                unique_id="matmul",
                source_code=MATMUL_SOURCE,
                hw_config=ttnn.ComputeHardwareConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4, enable_32_bit_dest=enable_32_bit_dest
                ),
                dfb_bindings=[binding("in0", consumer), binding("in1", consumer), binding("out", producer)],
                compile_time_args=sizes,
                defines={"MATMUL_BLOCK": "1"} if matmul_block else {},
            ),
            ttnn.KernelSpec(
                unique_id="pop_output",
                source_code=POP_OUTPUT_SOURCE,
                hw_config=ttnn.DataMovementHardwareConfig.writer(),
                dfb_bindings=[binding("out", consumer)],
                compile_time_args={"Nt": Nt, "num_iters": num_iters},
            ),
        ],
        work_units=[ttnn.WorkUnitSpec(name="main", kernels=["push_inputs", "matmul", "pop_output"], target_nodes=grid)],
    )
    run_args = ttnn.ProgramSpecRunArgs(tensor_args={"in0": a, "in1": b, "out": out})
    return ttnn.to_torch(ttnn.generic_op([a, b, out], spec, run_args)).float()


def expected_output(torch_a, torch_b, M):
    """Each M-row block of torch_a times torch_b, stacked like the device output."""
    return torch.cat([block.float() @ torch_b.float() for block in torch_a.split(M)])


@pytest.mark.parametrize("matmul_block", [False, True], ids=["matmul_tiles", "matmul_block"])
@pytest.mark.parametrize("M", [1, 8])
@pytest.mark.parametrize("Kt", [1, 4])
@pytest.mark.parametrize("Nt", [1, 2])
def test_generic_spec_op_decode_matmul_resident_weights(device, M, Kt, Nt, matmul_block):
    torch.manual_seed(0)
    torch_a, a = make_activation(device, M, Kt)
    torch_b, b = make_weights(device, Kt, Nt)
    result = run_decode_matmul(a, b, make_output(device, M, b), matmul_block=matmul_block)
    assert_with_pcc(expected_output(torch_a, torch_b, M), result, 0.999)


@pytest.mark.parametrize("matmul_block", [False, True], ids=["matmul_tiles", "matmul_block"])
@pytest.mark.parametrize("M", [1, 8])
def test_generic_spec_op_decode_matmul_fp32_accumulation(device, M, matmul_block):
    """Accumulates in a 32-bit DST and packs Float32 output tiles."""
    torch.manual_seed(0)
    torch_a, a = make_activation(device, M, Kt=8)
    torch_b, b = make_weights(device, Kt=8, Nt=2)
    out = make_output(device, M, b, dtype=ttnn.float32)
    result = run_decode_matmul(a, b, out, matmul_block=matmul_block, enable_32_bit_dest=True)
    assert_with_pcc(expected_output(torch_a, torch_b, M), result, 0.9999)


@pytest.mark.parametrize("matmul_block", [False, True], ids=["matmul_tiles", "matmul_block"])
@pytest.mark.parametrize("M", [1, 8])
def test_generic_spec_op_decode_matmul_two_cores(device, M, matmul_block):
    """b and out are split by columns over two cores; each core reads a from core (0, 0) over the NoC."""
    torch.manual_seed(0)
    torch_a, a = make_activation(device, M, Kt=4)
    torch_b, b = make_weights(device, Kt=4, Nt=1, num_cores=2)
    result = run_decode_matmul(a, b, make_output(device, M, b), matmul_block=matmul_block)
    assert_with_pcc(expected_output(torch_a, torch_b, M), result, 0.999)


@pytest.mark.parametrize("matmul_block", [False, True], ids=["matmul_tiles", "matmul_block"])
def test_generic_spec_op_decode_matmul_persistent_weights(device, matmul_block):
    """Several activations in one program run against weights pushed once and never re-read; then a second program
    run with new activations reuses the same weight tensor."""
    M, Kt, Nt, num_iters = 1, 4, 2, 3
    torch.manual_seed(0)
    torch_b, b = make_weights(device, Kt, Nt)
    for _ in range(2):
        torch_a, a = make_activation(device, M, Kt, num_iters)
        out = make_output(device, M, b, num_iters)
        result = run_decode_matmul(a, b, out, num_iters=num_iters, matmul_block=matmul_block)
        assert_with_pcc(expected_output(torch_a, torch_b, M), result, 0.999)


def test_generic_spec_op_decode_matmul_cache_hit_rebinds_borrowed_buffers(device):
    """A second run on new tensors hits the program cache and must read and write the new shards."""
    torch.manual_seed(0)
    runs = []
    for _ in range(2):
        torch_a, a = make_activation(device, 8, 4)
        torch_b, b = make_weights(device, 4, 2)
        runs.append((torch_a, torch_b, a, b, make_output(device, 8, b)))
    assert runs[1][3].buffer_address() != runs[0][3].buffer_address()

    for torch_a, torch_b, a, b, out in runs:
        assert_with_pcc(expected_output(torch_a, torch_b, 8), run_decode_matmul(a, b, out), 0.999)
    assert device.num_program_cache_entries() == 1
