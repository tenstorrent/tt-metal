# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Decode-shaped matmul through ttnn.generic_op with a Metal 2.0 ProgramSpec.

out[M, 32*Nt] = a[M, 32*Kt] @ b[32*Kt, 32*Nt] on one core, with a and out in tiny Tile([M, 32]) tiles and b in
32x32 tiles. Every tensor is an L1 shard, and each dataflow buffer is borrowed from its tensor's shard: no data is
copied. One data movement kernel only announces the resident a and b tiles, the compute kernel reads them in place
by index, and a second data movement kernel retires the output tiles, which compute packed into the output shard.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

# Announces the resident tiles of a and b: their data is already in the shards the DFBs are borrowed from.
PUSH_RESIDENT_SOURCE = """
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t Kt = get_arg(args::Kt);
    constexpr uint32_t Nt = get_arg(args::Nt);
    DataflowBuffer in0(dfb::in0);
    DataflowBuffer in1(dfb::in1);
    in0.push_back(Kt);
    in1.push_back(Kt * Nt);
}
"""

MATMUL_SOURCE = """
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t Kt = get_arg(args::Kt);
    constexpr uint32_t Nt = get_arg(args::Nt);
    DataflowBuffer in0(dfb::in0);
    DataflowBuffer in1(dfb::in1);
    DataflowBuffer out(dfb::out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::in0, dfb::in1, dfb::out);
    matmul_init(dfb::in0, dfb::in1, /*transpose=*/0);

    in0.wait_front(Kt);
    in1.wait_front(Kt * Nt);
    for (uint32_t nt = 0; nt < Nt; ++nt) {
        tile_regs_acquire();
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            // b's tiles are row-major: tile (kt, nt) is at kt * Nt + nt.
            matmul_tiles(dfb::in0, dfb::in1, kt, kt * Nt + nt, 0);
        }
        tile_regs_commit();

        out.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, dfb::out);
        tile_regs_release();
        out.push_back(1);
    }
    in0.pop_front(Kt);
    in1.pop_front(Kt * Nt);
}
"""

# Retires the output tiles: compute already packed them into the output shard.
POP_OUTPUT_SOURCE = """
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t Nt = get_arg(args::Nt);
    DataflowBuffer out(dfb::out);
    out.wait_front(Nt);
    out.pop_front(Nt);
}
"""


def l1_shard_on_core_0(height, width):
    core = ttnn.CoreCoord(0, 0)
    return ttnn.create_sharded_memory_config(
        shape=(height, width),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(core, core)}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def binding(name, endpoint_type):
    return ttnn.DFBBinding(dfb_spec_name=name, accessor_name=name, endpoint_type=endpoint_type)


DTYPE = ttnn.bfloat16
FULL_TILE = ttnn.Tile([32, 32])


def make_operands(device, M, Kt, Nt):
    """Returns (torch_a, torch_b, a, b, out): a and out in Tile([M, 32]) tiles, b in 32x32 tiles."""
    tiny_tile = ttnn.Tile([M, 32])
    K, N = 32 * Kt, 32 * Nt
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, N), dtype=torch.bfloat16)

    def to_device(tensor, tile):
        return ttnn.from_torch(
            tensor,
            dtype=DTYPE,
            layout=ttnn.TILE_LAYOUT,
            tile=tile,
            device=device,
            memory_config=l1_shard_on_core_0(*tensor.shape),
        )

    out = to_device(torch.zeros((M, N), dtype=torch.bfloat16), tiny_tile)
    return torch_a, torch_b, to_device(torch_a, tiny_tile), to_device(torch_b, FULL_TILE), out


def run_decode_matmul(a, b, out):
    dtype = DTYPE
    tiny_tile, full_tile = a.tile, FULL_TILE
    Kt = a.shape[-1] // 32
    Nt = b.shape[-1] // 32

    def borrowed_dfb(name, tile, num_entries):
        return ttnn.DataflowBufferSpec(
            unique_id=name,
            entry_size=tile.get_tile_size(dtype),
            num_entries=num_entries,
            data_format=dtype,
            tile=tile,
            borrowed_from=name,
        )

    producer, consumer = ttnn.DFBEndpointType.PRODUCER, ttnn.DFBEndpointType.CONSUMER
    sizes = {"Kt": Kt, "Nt": Nt}
    spec = ttnn.ProgramSpec(
        name="decode_matmul_resident_weights",
        dataflow_buffers=[
            borrowed_dfb("in0", tiny_tile, Kt),
            borrowed_dfb("in1", full_tile, Kt * Nt),
            borrowed_dfb("out", tiny_tile, Nt),
        ],
        tensor_parameters=[
            ttnn.TensorParameter(unique_id="in0", spec=a.spec),
            ttnn.TensorParameter(unique_id="in1", spec=b.spec),
            ttnn.TensorParameter(unique_id="out", spec=out.spec),
        ],
        kernels=[
            ttnn.KernelSpec(
                unique_id="push_resident",
                source_code=PUSH_RESIDENT_SOURCE,
                hw_config=ttnn.DataMovementHardwareConfig.reader(),
                dfb_bindings=[binding("in0", producer), binding("in1", producer)],
                compile_time_args=sizes,
            ),
            ttnn.KernelSpec(
                unique_id="matmul",
                source_code=MATMUL_SOURCE,
                hw_config=ttnn.ComputeHardwareConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
                dfb_bindings=[binding("in0", consumer), binding("in1", consumer), binding("out", producer)],
                compile_time_args=sizes,
            ),
            ttnn.KernelSpec(
                unique_id="pop_output",
                source_code=POP_OUTPUT_SOURCE,
                hw_config=ttnn.DataMovementHardwareConfig.writer(),
                dfb_bindings=[binding("out", consumer)],
                compile_time_args={"Nt": Nt},
            ),
        ],
        work_units=[
            ttnn.WorkUnitSpec(
                name="main",
                kernels=["push_resident", "matmul", "pop_output"],
                target_nodes=a.memory_config().shard_spec.grid,
            )
        ],
    )
    run_args = ttnn.ProgramSpecRunArgs(tensor_args={"in0": a, "in1": b, "out": out})
    return ttnn.to_torch(ttnn.generic_op([a, b, out], spec, run_args)).float()


@pytest.mark.parametrize("M", [1, 8])
@pytest.mark.parametrize("Kt", [1, 4])
@pytest.mark.parametrize("Nt", [1, 2])
def test_generic_spec_op_decode_matmul_resident_weights(device, M, Kt, Nt):
    torch.manual_seed(0)
    torch_a, torch_b, a, b, out = make_operands(device, M, Kt, Nt)
    result = run_decode_matmul(a, b, out)
    assert_with_pcc(torch_a.float() @ torch_b.float(), result, 0.999)


def test_generic_spec_op_decode_matmul_cache_hit_rebinds_borrowed_buffers(device):
    """A second run on new tensors hits the program cache and must read and write the new shards."""
    torch.manual_seed(0)
    first = make_operands(device, 8, 4, 2)
    second = make_operands(device, 8, 4, 2)
    assert second[3].buffer_address() != first[3].buffer_address()

    for torch_a, torch_b, a, b, out in (first, second):
        result = run_decode_matmul(a, b, out)
        assert_with_pcc(torch_a.float() @ torch_b.float(), result, 0.999)
    assert device.num_program_cache_entries() == 1
