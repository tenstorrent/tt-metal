# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Top-k values and indices recomposed from compute APIs that build for Quasar, through ttnn.experimental.generic_op.

tt-blaze's sampling top-k (topk_xl) exists only for Blackhole. This finds the K largest of 64 values in each of 32 rows,
and their column indices, removing exactly one element per step so that equal values survive as in torch.topk:

    m     = rowmax(x)                                   reduce MAX over rows
    score = 128 * (x - m == 0) - idx                    sub_tiles_bcast_cols, eqz_tile, mul_unary_tile, sub_binary_tile
    i     = 128 - rowmax(score)                         the lowest index holding m (ties score 65..128, others <= 0)
    x     = x - BIG * (idx - i == 0)                    knock out that one element

idx holds each element's column index (0..63), and every intermediate is an integer below 256, exact in bf16. Step s
writes m as value tile s and i as index tile s (column 0 holds each row's s-th value and its index). Among equal values
the lowest index comes first. The working x, m, score and i live in compute self-loop DFBs.
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
    DataflowBuffer idx(dfb::idx);
    DataflowBuffer ones(dfb::ones);
    DataflowBuffer c128(dfb::c128);
    DataflowBuffer top_values(dfb::top_values);
    DataflowBuffer top_indices(dfb::top_indices);
    load(x, TensorAccessor(tensor::x), wt);
    load(idx, TensorAccessor(tensor::idx), wt);
    load(ones, TensorAccessor(tensor::ones), 1);
    load(c128, TensorAccessor(tensor::c128), 1);
    const auto values = TensorAccessor(tensor::values);
    const auto indices = TensorAccessor(tensor::indices);
    for (uint32_t s = 0; s < top_k; ++s) {
        top_values.wait_front(1);
        top_indices.wait_front(1);
        noc.async_write(top_values, values, top_values.get_entry_size(), {.offset_bytes = 0}, {.page_id = s});
        noc.async_write(top_indices, indices, top_indices.get_entry_size(), {.offset_bytes = 0}, {.page_id = s});
        noc.async_write_barrier();
        top_values.pop_front(1);
        top_indices.pop_front(1);
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
    constexpr uint32_t big = get_arg(args::big);                  // fp32 bits of BIG
    constexpr uint32_t tie_weight = get_arg(args::tie_weight);    // fp32 bits of 128
    constexpr uint32_t minus_one = get_arg(args::minus_one);      // fp32 bits of -1
    DataflowBuffer x(dfb::x);
    DataflowBuffer idx(dfb::idx);
    DataflowBuffer ones(dfb::ones);
    DataflowBuffer c128(dfb::c128);
    DataflowBuffer xs(dfb::xs);        // working copy of x
    DataflowBuffer m(dfb::m);          // this step's row max
    DataflowBuffer score(dfb::score);  // 128 * (x == m) - idx
    DataflowBuffer sel(dfb::sel);      // this step's index
    DataflowBuffer top_values(dfb::top_values);
    DataflowBuffer top_indices(dfb::top_indices);
    compute_kernel_hw_startup(dfb::x, dfb::ones, dfb::xs);
    x.wait_front(wt);
    idx.wait_front(wt);
    ones.wait_front(1);
    c128.wait_front(1);

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

    for (uint32_t s = 0; s < top_k; ++s) {
        // m = rowmax(xs), to m and out as the value
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(dfb::xs, dfb::ones, dfb::m);
        tile_regs_acquire();
        for (uint32_t t = 0; t < wt; ++t) {
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(dfb::xs, dfb::ones, t, 0, 0);
        }
        reduce_uninit();
        tile_regs_commit();
        tile_regs_wait();
        top_values.reserve_back(1);
        pack_init(dfb::top_values);
        pack_tile(0, dfb::top_values);
        top_values.push_back(1);
        m.reserve_back(1);
        pack_init(dfb::m);
        pack_tile(0, dfb::m);
        m.push_back(1);
        tile_regs_release();
        m.wait_front(1);

        // score = 128 * (xs == m) - idx
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
            mul_unary_tile(t, tie_weight);
        }
        copy_init(dfb::idx);
        for (uint32_t t = 0; t < wt; ++t) {
            copy_tile(dfb::idx, t, wt + t);
        }
        sub_binary_tile_init();
        for (uint32_t t = 0; t < wt; ++t) {
            sub_binary_tile(t, wt + t, t);
        }
        tile_regs_commit();
        score.reserve_back(wt);
        tile_regs_wait();
        pack_init(dfb::score);
        for (uint32_t t = 0; t < wt; ++t) {
            pack_tile(t, dfb::score);
        }
        tile_regs_release();
        score.push_back(wt);
        m.pop_front(1);
        score.wait_front(wt);

        // i = 128 - rowmax(score), to sel and out as the index
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(dfb::score, dfb::ones, dfb::sel);
        tile_regs_acquire();
        for (uint32_t t = 0; t < wt; ++t) {
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(dfb::score, dfb::ones, t, 0, 0);
        }
        reduce_uninit();
        binop_with_scalar_tile_init();
        mul_unary_tile(0, minus_one);
        copy_init(dfb::c128);
        copy_tile(dfb::c128, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        top_indices.reserve_back(1);
        pack_init(dfb::top_indices);
        pack_tile(0, dfb::top_indices);
        top_indices.push_back(1);
        sel.reserve_back(1);
        pack_init(dfb::sel);
        pack_tile(0, dfb::sel);
        sel.push_back(1);
        tile_regs_release();
        score.pop_front(wt);
        sel.wait_front(1);

        // xs -= BIG * (idx == i): exactly one element per row
        tile_regs_acquire();
        sub_bcast_cols_init(dfb::idx, dfb::sel);
        for (uint32_t t = 0; t < wt; ++t) {
            sub_tiles_bcast_cols(dfb::idx, dfb::sel, t, 0, t);
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
        sel.pop_front(1);
        xs.wait_front(wt);
    }
    dummy_unpack(dfb::xs);  // nothing reads the last working copy
    xs.pop_front(wt);
    idx.pop_front(wt);
    ones.pop_front(1);
    c128.pop_front(1);
}
"""


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def make_values():
    """Random bf16 values, with ties forced into the top K of most rows."""
    torch.manual_seed(0)
    x = torch.randn(32, WIDTH).bfloat16()
    for row in range(0, 32, 2):
        order = torch.argsort(x[row].float(), descending=True)
        x[row, order[row % TOP_K + 3]] = x[row, order[row % TOP_K]]  # a lower value now ties one in the top K
    x[1, :] = x[1, 0]  # a whole row of equal values
    return x


def test_generic_spec_op_topk_recompose(device):
    torch_x = make_values()

    def tile_tensor(t):
        return ttnn.from_torch(t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tensors = {
        "x": tile_tensor(torch_x),
        "idx": tile_tensor(torch.arange(WIDTH, dtype=torch.float32).expand(32, WIDTH)),
        "ones": tile_tensor(torch.ones(32, 32)),
        "c128": tile_tensor(torch.full((32, 32), 128.0)),
        "values": tile_tensor(torch.zeros(32, 32 * TOP_K)),
        "indices": tile_tensor(torch.zeros(32, 32 * TOP_K)),
    }
    inputs = {"x": WT, "idx": WT, "ones": 1, "c128": 1}
    loops = {"xs": 2 * WT, "m": 1, "score": WT, "sel": 1}
    outputs = {"top_values": 2, "top_indices": 2}

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
                dfb_bindings=[E.producer_of(n, n) for n in inputs] + [E.consumer_of(n, n) for n in outputs],
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
                + [E.producer_of(n, n) for n in outputs],
                compile_time_args={
                    **args,
                    "big": f32_bits(BIG),
                    "tie_weight": f32_bits(128.0),
                    "minus_one": f32_bits(-1.0),
                },
            ),
        ],
        dataflow_buffers=[dfb(n, e) for n, e in {**inputs, **loops, **outputs}.items()],
        tensor_parameters=[E.TensorParameter(unique_id=t, spec=v.spec) for t, v in tensors.items()],
        work_units=[E.WorkUnitSpec(kernels=["dm", "compute"], target_nodes=CORE)],
    )
    E.generic_op(list(tensors.values()), spec, E.ProgramRunArgs(tensor_args=tensors))

    values = ttnn.to_torch(tensors["values"]).float()[:, ::32]  # column 0 of each out tile
    indices = ttnn.to_torch(tensors["indices"]).float()[:, ::32].long()
    expected = torch.topk(torch_x.float(), TOP_K, dim=-1)
    # Values, duplicates included, match torch.topk bit for bit.
    assert torch.equal(values, expected.values), (values - expected.values).abs().max()
    # Each index points at its value, no element is picked twice, and among equal values the lowest index comes
    # first (torch.topk does not promise an order among ties).
    assert torch.equal(torch.gather(torch_x.float(), 1, indices), values)
    for row in range(32):
        assert len(set(indices[row].tolist())) == TOP_K, (row, indices[row])
        for s in range(TOP_K):
            tied = (torch_x[row].float() == values[row, s]).nonzero().flatten().tolist()
            earlier = indices[row, :s].tolist()
            assert indices[row, s].item() == min(i for i in tied if i not in earlier), (row, s, indices[row])
    # Where none of the row's top K values occurs twice in the row, the indices match torch.topk exactly.
    for row in range(32):
        if all((torch_x[row].float() == v).sum() == 1 for v in values[row]):
            assert torch.equal(indices[row], expected.indices[row]), (row, indices[row], expected.indices[row])
