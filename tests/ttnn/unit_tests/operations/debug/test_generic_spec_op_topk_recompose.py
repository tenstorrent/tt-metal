# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Top-k values recomposed from compute APIs that build for Quasar, through ttnn.experimental.generic_op.

tt-blaze's sampling top-k (topk_xl) exists only for Blackhole. This finds the K largest of 64 values in each of 32 rows
by repeating, K times:

    m = rowmax(x)                      (reduce MAX over rows, both tiles accumulated)
    x = x - BIG * (x - m == 0)         (sub_tiles_bcast_cols, eqz_tile, mul_unary_tile, sub_binary_tile)

Step i writes m as out tile i (column 0 holds each row's i-th largest value). The working copy of x lives in a compute
self-loop DFB. Values only: indices are not tracked. Equal values leave together, so a row with a tie among its top K
loses the duplicate (torch.topk keeps it); the test uses distinct values.
"""

import struct

import torch
import ttnn


E = ttnn.experimental

CORE = ttnn.CoreCoord(0, 0)
WIDTH = 64
WT = WIDTH // 32
TOP_K = 4
BIG = 1e30
TILE_BYTES = 2 * 32 * 32

DM_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t wt = get_arg(args::wt);
    constexpr uint32_t top_k = get_arg(args::top_k);
    Noc noc;
    auto load = [&](auto& dfb, const auto& tensor, uint32_t num_tiles) {
        dfb.reserve_back(num_tiles);
        for (uint32_t t = 0; t < num_tiles; ++t) {
            noc.async_read(tensor, dfb, dfb.get_entry_size(), {.page_id = t}, {.offset_bytes = t * dfb.get_entry_size()});
        }
        noc.async_read_barrier();
        dfb.push_back(num_tiles);
    };
    DataflowBuffer x(dfb::x);
    DataflowBuffer ones(dfb::ones);
    DataflowBuffer top(dfb::top);
    load(x, TensorAccessor(tensor::x), wt);
    load(ones, TensorAccessor(tensor::ones), 1);
    const auto result = TensorAccessor(tensor::result);
    for (uint32_t i = 0; i < top_k; ++i) {
        top.wait_front(1);
        noc.async_write(top, result, top.get_entry_size(), {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        top.pop_front(1);
    }
}
"""

COMPUTE_SOURCE = """
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t wt = get_arg(args::wt);
    constexpr uint32_t top_k = get_arg(args::top_k);
    constexpr uint32_t big = get_arg(args::big);  // fp32 bits of BIG
    DataflowBuffer x(dfb::x);
    DataflowBuffer ones(dfb::ones);
    DataflowBuffer xs(dfb::xs);  // working copy of x
    DataflowBuffer m(dfb::m);
    DataflowBuffer top(dfb::top);
    compute_kernel_hw_startup(dfb::x, dfb::ones, dfb::xs);
    x.wait_front(wt);
    ones.wait_front(1);

    // xs = x
    copy_init(dfb::x);
    tile_regs_acquire();
    for (uint32_t t = 0; t < wt; ++t) {
        copy_tile(dfb::x, t, t);
    }
    tile_regs_commit();
    xs.reserve_back(wt);
    tile_regs_wait();
    pack_init(dfb::xs);
    for (uint32_t t = 0; t < wt; ++t) {
        pack_tile(t, dfb::xs);
    }
    tile_regs_release();
    xs.push_back(wt);
    x.pop_front(wt);
    xs.wait_front(wt);

    for (uint32_t i = 0; i < top_k; ++i) {
        // m = rowmax(xs), out to top and to m
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(dfb::xs, dfb::ones, dfb::m);
        tile_regs_acquire();
        for (uint32_t t = 0; t < wt; ++t) {
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(dfb::xs, dfb::ones, t, 0, 0);
        }
        reduce_uninit();
        tile_regs_commit();
        tile_regs_wait();
        top.reserve_back(1);
        pack_init(dfb::top);
        pack_tile(0, dfb::top);
        top.push_back(1);
        m.reserve_back(1);
        pack_init(dfb::m);
        pack_tile(0, dfb::m);
        m.push_back(1);
        tile_regs_release();
        m.wait_front(1);

        // xs -= BIG * (xs == m)
        tile_regs_acquire();
        sub_bcast_cols_init(dfb::xs, dfb::m);
        for (uint32_t t = 0; t < wt; ++t) {
            sub_tiles_bcast_cols(dfb::xs, dfb::m, t, 0, t);
        }
        eqz_tile_init();
        for (uint32_t t = 0; t < wt; ++t) {
            eqz_tile(t);
        }
        binop_with_scalar_tile_init();
        for (uint32_t t = 0; t < wt; ++t) {
            mul_unary_tile(t, big);
        }
        copy_init(dfb::xs);
        for (uint32_t t = 0; t < wt; ++t) {
            copy_tile(dfb::xs, t, wt + t);
        }
        sub_binary_tile_init();
        for (uint32_t t = 0; t < wt; ++t) {
            sub_binary_tile(wt + t, t, t);
        }
        tile_regs_commit();
        xs.reserve_back(wt);
        tile_regs_wait();
        pack_init(dfb::xs);
        for (uint32_t t = 0; t < wt; ++t) {
            pack_tile(t, dfb::xs);
        }
        tile_regs_release();
        xs.push_back(wt);
        xs.pop_front(wt);  // the previous working copy
        m.pop_front(1);
        xs.wait_front(wt);
    }
    dummy_unpack(dfb::xs);  // nothing reads the last working copy
    xs.pop_front(wt);
    ones.pop_front(1);
}
"""


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def test_generic_spec_op_topk_recompose(device):
    torch.manual_seed(0)
    # Distinct, exactly representable values in each row: a shuffled 0, 1/8, ..., 63/8 minus 4.
    torch_x = torch.stack([torch.randperm(WIDTH) for _ in range(32)]).float() / 8 - 4
    torch_x = torch_x.bfloat16()
    expected = torch.topk(torch_x.float(), TOP_K, dim=-1).values  # [32, TOP_K], descending

    def tile_tensor(t):
        return ttnn.from_torch(t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tensors = {
        "x": tile_tensor(torch_x),
        "ones": tile_tensor(torch.ones(32, 32)),
        "result": tile_tensor(torch.zeros(32, 32 * TOP_K)),
    }
    inputs = {"x": WT, "ones": 1}
    loops = {"xs": 2 * WT, "m": 1}

    def dfb(name, num_entries):
        return E.DataflowBufferSpec(
            unique_id=name, entry_size=TILE_BYTES, num_entries=num_entries, data_format_metadata=ttnn.bfloat16
        )

    args = {"wt": WT, "top_k": TOP_K}
    spec = E.ProgramSpec(
        name="topk_recompose",
        kernels=[
            E.KernelSpec(
                unique_id="dm",
                source=E.KernelSpec.SourceCode(code=DM_SOURCE),
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                dfb_bindings=[E.producer_of(n, n) for n in inputs] + [E.consumer_of("top", "top")],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name=t, accessor_name=t) for t in tensors],
                compile_time_args=args,
            ),
            E.KernelSpec(
                unique_id="compute",
                source=E.KernelSpec.SourceCode(code=COMPUTE_SOURCE),
                hw_config=E.ComputeHardwareConfig(),
                dfb_bindings=[E.consumer_of(n, n) for n in inputs]
                + [E.producer_of(n, n) for n in loops]
                + [E.consumer_of(n, n) for n in loops]
                + [E.producer_of("top", "top")],
                compile_time_args={**args, "big": f32_bits(BIG)},
            ),
        ],
        dataflow_buffers=[dfb(n, e) for n, e in {**inputs, **loops, "top": 2}.items()],
        tensor_parameters=[E.TensorParameter(unique_id=t, spec=v.spec) for t, v in tensors.items()],
        work_units=[E.WorkUnitSpec(kernels=["dm", "compute"], target_nodes=CORE)],
    )
    E.generic_op(list(tensors.values()), spec, E.ProgramRunArgs(tensor_args=tensors))

    result = ttnn.to_torch(tensors["result"]).float()
    values = result[:, ::32]  # column 0 of each out tile
    assert torch.equal(values, expected), (values - expected).abs().max()
