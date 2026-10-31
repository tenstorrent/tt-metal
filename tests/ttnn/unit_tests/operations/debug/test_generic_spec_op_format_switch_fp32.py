# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""A bf16 part and an fp32 part in one compute kernel, where the source-register format really changes, through
ttnn.experimental.generic_op.

Per token: ox = x + x with x in bf16, and oy = y + y with y in fp32, each part with its own DFBs. compute_kernel_hw_startup
runs once, for the bf16 part. On every iteration each part's entry runs reconfig_data_format and
pack_reconfig_data_format for its own operands, then its op init and pack_init.

Unlike test_generic_spec_op_format_switch.py, where every operand unpacks to Float16_b source registers, y unpacks to
Tf32 source registers (UnpackToSrc) and x to Float16_b, so here reconfig_data_format has to rewrite the unpacker's
output format. y holds multiples of 1/1024 below 2 in magnitude: they survive the Tf32 source path exactly but not a
bf16 one, so the fp32 part is checked for exact equality (PCC cannot tell the two apart).

NO_PART2_RECONFIG (a mutant, not part of the test) drops the fp32 part's reconfig_data_format. On Quasar craq-sim, with
the watcher and LLK asserts on, nothing asserts: y is unpacked at bf16 precision on every iteration (1.298828125 comes
back as 1.296875) and about two thirds of oy is wrong, while its PCC stays above 0.99999.
"""

import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

E = ttnn.experimental

CORE = ttnn.CoreCoord(0, 0)
ROWS, COLS = 32, 64
TILES = COLS // 32
NUM_ITERS = 3

DM_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t tiles = get_arg(args::tiles);
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    Noc noc;
    auto load = [&](auto& dfb, const auto& tensor, uint32_t first_page) {
        dfb.reserve_back(tiles);
        for (uint32_t t = 0; t < tiles; ++t) {
            noc.async_read(
                tensor, dfb, dfb.get_entry_size(), {.page_id = first_page + t}, {.offset_bytes = t * dfb.get_entry_size()});
        }
        noc.async_read_barrier();
        dfb.push_back(tiles);
    };
    auto store = [&](auto& dfb, const auto& tensor, uint32_t first_page) {
        dfb.wait_front(tiles);
        for (uint32_t t = 0; t < tiles; ++t) {
            noc.async_write(
                dfb, tensor, dfb.get_entry_size(), {.offset_bytes = t * dfb.get_entry_size()}, {.page_id = first_page + t});
        }
        noc.async_write_barrier();
        dfb.pop_front(tiles);
    };
    DataflowBuffer x(dfb::x);
    DataflowBuffer y(dfb::y);
    DataflowBuffer ox(dfb::ox);
    DataflowBuffer oy(dfb::oy);
    const auto x_tensor = TensorAccessor(tensor::x);
    const auto y_tensor = TensorAccessor(tensor::y);
    const auto ox_tensor = TensorAccessor(tensor::ox);
    const auto oy_tensor = TensorAccessor(tensor::oy);
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        load(x, x_tensor, iter * tiles);
        load(y, y_tensor, iter * tiles);
        store(ox, ox_tensor, iter * tiles);
        store(oy, oy_tensor, iter * tiles);
    }
}
"""

COMPUTE_SOURCE = """
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

template <typename In, typename Out>
void double_tiles(In& in, Out& out, uint32_t in_id, uint32_t out_id, uint32_t tiles) {
    in.wait_front(tiles);
    tile_regs_acquire();
    for (uint32_t t = 0; t < tiles; ++t) {
        add_tiles(in_id, in_id, t, t, t);
    }
    tile_regs_commit();
    out.reserve_back(tiles);
    tile_regs_wait();
    for (uint32_t t = 0; t < tiles; ++t) {
        pack_tile(t, out_id);
    }
    tile_regs_release();
    out.push_back(tiles);
    in.pop_front(tiles);
}

void kernel_main() {
    constexpr uint32_t tiles = get_arg(args::tiles);
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    DataflowBuffer x(dfb::x);
    DataflowBuffer y(dfb::y);
    DataflowBuffer ox(dfb::ox);
    DataflowBuffer oy(dfb::oy);
    compute_kernel_hw_startup(dfb::x, dfb::x, dfb::ox);
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        // Part 1: ox = x + x, bf16.
        reconfig_data_format(dfb::x, dfb::x);
        pack_reconfig_data_format(dfb::ox);
        binary_tiles_init<true, EltwiseBinaryType::ELWADD>(dfb::x, dfb::x);
        pack_init(dfb::ox);
        double_tiles(x, ox, dfb::x, dfb::ox, tiles);

        // Part 2: oy = y + y, fp32.
#ifndef NO_PART2_RECONFIG
        reconfig_data_format(dfb::y, dfb::y);
#endif
        pack_reconfig_data_format(dfb::oy);
        binary_tiles_init<true, EltwiseBinaryType::ELWADD>(dfb::y, dfb::y);
        pack_init(dfb::oy);
        double_tiles(y, oy, dfb::y, dfb::oy, tiles);
    }
}
"""


def run_format_switch_fp32(device, defines=None):
    """Returns (torch x, torch y, device ox, device oy) as float32 torch tensors."""
    torch.manual_seed(0)
    torch_x = torch.randn(NUM_ITERS * ROWS, COLS).bfloat16()
    torch_y = torch.randint(-2047, 2048, (NUM_ITERS * ROWS, COLS)).float() / 1024

    def tile_tensor(t, dtype):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    tensors = {
        "x": tile_tensor(torch_x, ttnn.bfloat16),
        "y": tile_tensor(torch_y, ttnn.float32),
        "ox": tile_tensor(torch.zeros(NUM_ITERS * ROWS, COLS).bfloat16(), ttnn.bfloat16),
        "oy": tile_tensor(torch.zeros(NUM_ITERS * ROWS, COLS), ttnn.float32),
    }

    def dfb(name, dtype):
        return E.DataflowBufferSpec(
            unique_id=name,
            entry_size=ttnn.Tile([32, 32]).get_tile_size(dtype),
            num_entries=TILES,
            data_format_metadata=dtype,
        )

    common = {"tiles": TILES, "num_iters": NUM_ITERS}
    spec = E.ProgramSpec(
        name="format_switch_fp32",
        kernels=[
            E.KernelSpec(
                unique_id="dm",
                source=E.KernelSpec.SourceCode(code=DM_SOURCE),
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                dfb_bindings=[E.producer_of(n, n) for n in ("x", "y")] + [E.consumer_of(n, n) for n in ("ox", "oy")],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name=t, accessor_name=t) for t in tensors],
                compile_time_args=common,
            ),
            E.KernelSpec(
                unique_id="compute",
                source=E.KernelSpec.SourceCode(code=COMPUTE_SOURCE),
                hw_config=E.ComputeHardwareConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    enable_32_bit_dest=True,
                    unpack_modes={"y": ttnn.UnpackMode.UnpackToSrc},  # add_tiles reads source registers
                ),
                compiler_options=E.KernelSpec.CompilerOptions(defines=defines or {}),
                dfb_bindings=[E.consumer_of(n, n) for n in ("x", "y")] + [E.producer_of(n, n) for n in ("ox", "oy")],
                compile_time_args=common,
            ),
        ],
        dataflow_buffers=[
            dfb("x", ttnn.bfloat16),
            dfb("y", ttnn.float32),
            dfb("ox", ttnn.bfloat16),
            dfb("oy", ttnn.float32),
        ],
        tensor_parameters=[E.TensorParameter(unique_id=t, spec=v.spec) for t, v in tensors.items()],
        work_units=[E.WorkUnitSpec(kernels=["dm", "compute"], target_nodes=CORE)],
    )
    E.generic_op(list(tensors.values()), spec, E.ProgramRunArgs(tensor_args=tensors))
    return torch_x.float(), torch_y, ttnn.to_torch(tensors["ox"]).float(), ttnn.to_torch(tensors["oy"]).float()


def test_generic_spec_op_format_switch_fp32(device):
    x, y, ox, oy = run_format_switch_fp32(device)
    assert_with_pcc(x * 2, ox, 0.999)
    assert torch.equal(oy, y * 2)
