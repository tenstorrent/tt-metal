# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""One DFB carrying two logical FIFOs in sequence, through ttnn.experimental.generic_op.

tt-blaze's DFB-pressure lever for fused programs: two like-typed FIFOs whose lifetimes do not overlap within an
iteration share one DFB (one slot, one TRISC-active DFB) instead of two. Here `inq` (DM -> compute) carries logical
FIFO x then logical FIFO y, and `outq` (compute -> DM) carries ox then oy. Each iteration: x's tiles are pushed and
fully popped (compute applies x * 2), then y's tiles (compute applies silu, a different op with its own init). Each
user balances its own pushes and pops, so the DFB is empty at every hand-over. The producer kernel, consumer kernel and
entry format are the same for both users, which is what a merge needs.
"""

import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

E = ttnn.experimental

CORE = ttnn.CoreCoord(0, 0)
TILE_BYTES = 2 * 32 * 32
TILES = 2  # tiles per logical FIFO per iteration
ITERS = 2

DM_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

constexpr uint32_t TILES = get_arg(args::tiles);
constexpr uint32_t ITERS = get_arg(args::iters);

template <typename Q, typename Src>
void feed(Noc& noc, Q& q, const Src& src, uint32_t first_page) {
    for (uint32_t t = 0; t < TILES; t++) {
        q.reserve_back(1);
        noc.async_read(src, q, q.get_entry_size(), {.page_id = first_page + t}, {.offset_bytes = 0});
        noc.async_read_barrier();
        q.push_back(1);
    }
}

template <typename Q, typename Dst>
void drain(Noc& noc, Q& q, const Dst& dst, uint32_t first_page) {
    for (uint32_t t = 0; t < TILES; t++) {
        q.wait_front(1);
        noc.async_write(q, dst, q.get_entry_size(), {.offset_bytes = 0}, {.page_id = first_page + t});
        noc.async_write_barrier();
        q.pop_front(1);
    }
}

void kernel_main() {
    Noc noc;
    DataflowBuffer inq(dfb::inq);
    DataflowBuffer outq(dfb::outq);
    const auto x = TensorAccessor(tensor::x);
    const auto y = TensorAccessor(tensor::y);
    const auto ox = TensorAccessor(tensor::ox);
    const auto oy = TensorAccessor(tensor::oy);
    for (uint32_t it = 0; it < ITERS; it++) {
        const uint32_t page = it * TILES;
        feed(noc, inq, x, page);    // logical FIFO x
        feed(noc, inq, y, page);    // logical FIFO y, same DFB, after x has been consumed
        drain(noc, outq, ox, page);  // logical FIFO ox
        drain(noc, outq, oy, page);  // logical FIFO oy
    }
}
"""

COMPUTE_SOURCE = """
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

constexpr uint32_t TILES = get_arg(args::tiles);
constexpr uint32_t ITERS = get_arg(args::iters);

template <bool SILU, typename In, typename Out>
void consume(In& inq, Out& outq) {
    for (uint32_t t = 0; t < TILES; t++) {
        inq.wait_front(1);
        tile_regs_acquire();
        copy_init(dfb::inq);
        copy_tile(dfb::inq, 0, 0);
        if constexpr (SILU) {
            silu_tile_init();
            silu_tile(0);
        } else {
            binop_with_scalar_tile_init();
            mul_unary_tile(0, 0x40000000u);  // 2.0f
        }
        tile_regs_commit();
        outq.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, dfb::outq);
        tile_regs_release();
        outq.push_back(1);
        inq.pop_front(1);
    }
}

void kernel_main() {
    DataflowBuffer inq(dfb::inq);
    DataflowBuffer outq(dfb::outq);
    compute_kernel_hw_startup(dfb::inq, dfb::outq);
    for (uint32_t it = 0; it < ITERS; it++) {
        consume<false>(inq, outq);  // x -> ox
        consume<true>(inq, outq);   // y -> oy
    }
}
"""


def tile_tensor(device, t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def test_generic_spec_op_fifo_multiplex(device):
    torch.manual_seed(0)
    width = 32 * TILES * ITERS
    torch_x = torch.randn(32, width).bfloat16()
    torch_y = torch.randn(32, width).bfloat16()
    tensors = {
        "x": tile_tensor(device, torch_x),
        "y": tile_tensor(device, torch_y),
        "ox": tile_tensor(device, torch.zeros(32, width).bfloat16()),
        "oy": tile_tensor(device, torch.zeros(32, width).bfloat16()),
    }
    ctas = {"tiles": TILES, "iters": ITERS}
    # One entry per tile of the larger user; both users have TILES tiles here.
    dfbs = [
        E.DataflowBufferSpec(unique_id=n, entry_size=TILE_BYTES, num_entries=TILES, data_format_metadata=ttnn.bfloat16)
        for n in ("inq", "outq")
    ]
    spec = E.ProgramSpec(
        name="fifo_multiplex",
        kernels=[
            E.KernelSpec(
                unique_id="dm",
                source=E.KernelSpec.SourceCode(code=DM_SOURCE),
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                compile_time_args=ctas,
                dfb_bindings=[E.producer_of("inq", "inq"), E.consumer_of("outq", "outq")],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name=t, accessor_name=t) for t in tensors],
            ),
            E.KernelSpec(
                unique_id="compute",
                source=E.KernelSpec.SourceCode(code=COMPUTE_SOURCE),
                hw_config=E.ComputeHardwareConfig(),
                compile_time_args=ctas,
                dfb_bindings=[E.consumer_of("inq", "inq"), E.producer_of("outq", "outq")],
            ),
        ],
        dataflow_buffers=dfbs,
        tensor_parameters=[E.TensorParameter(unique_id=t, spec=v.spec) for t, v in tensors.items()],
        work_units=[E.WorkUnitSpec(kernels=["dm", "compute"], target_nodes=CORE)],
    )
    E.generic_op(list(tensors.values()), spec, E.ProgramRunArgs(tensor_args=tensors))

    assert_with_pcc(torch_x.float() * 2, ttnn.to_torch(tensors["ox"]).float(), 0.999)
    assert_with_pcc(torch.nn.functional.silu(torch_y.float()), ttnn.to_torch(tensors["oy"]).float(), 0.999)
