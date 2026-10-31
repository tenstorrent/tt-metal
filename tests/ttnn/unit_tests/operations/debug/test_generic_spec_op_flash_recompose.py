# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Decode flash attention recomposed from compute APIs that build for Quasar, through ttnn.experimental.generic_op.

tt-blaze's SDPA (sdpa.h, sdpa_custom_mm.h) exists only for Blackhole. This is one head with head_dim d = 64 and a KV
length of 64 in two chunks of 32 keys, with online softmax across the chunks:

    S = q k_c^T;  m' = max(m, rowmax S);  alpha = exp(m - m');  P = exp(S - m')
    l' = alpha l + rowsum P;  O' = alpha O + P v_c;  out = O / l

q holds 32 query rows (a decode token's query is one of them; the others are other tokens of the same head), already
scaled by 1 / sqrt(d) on the host, and k comes in transposed. Each step uses matmul_tiles, reduce (MAX and SUM over
rows), sub_tiles_bcast_cols, exp_tile, binary_max_tile, the SFPU binary ops, recip_tile and mul_tiles_bcast_cols. The
running m, l and O and the intermediates live in compute self-loop DFBs. On Quasar every pack to a different DFB is
preceded by pack_init, and the final m, which nothing reads, is popped after dummy_unpack.
"""

import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

E = ttnn.experimental

CORE = ttnn.CoreCoord(0, 0)
D = 64
SEQ = 64
DT = D // 32  # head_dim tiles
CHUNKS = SEQ // 32
TILE_BYTES = 2 * 32 * 32

DM_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t dt = get_arg(args::dt);
    constexpr uint32_t chunks = get_arg(args::chunks);
    Noc noc;
    auto load = [&](auto& dfb, const auto& tensor, uint32_t num_tiles) {
        dfb.reserve_back(num_tiles);
        for (uint32_t t = 0; t < num_tiles; ++t) {
            noc.async_read(tensor, dfb, dfb.get_entry_size(), {.page_id = t}, {.offset_bytes = t * dfb.get_entry_size()});
        }
        noc.async_read_barrier();
        dfb.push_back(num_tiles);
    };
    DataflowBuffer q(dfb::q);
    DataflowBuffer kt(dfb::kt);
    DataflowBuffer v(dfb::v);
    DataflowBuffer ones(dfb::ones);
    DataflowBuffer out(dfb::out);
    load(q, TensorAccessor(tensor::q), dt);
    load(kt, TensorAccessor(tensor::kt), dt * chunks);
    load(v, TensorAccessor(tensor::v), chunks * dt);
    load(ones, TensorAccessor(tensor::ones), 1);
    const auto result = TensorAccessor(tensor::result);
    out.wait_front(dt);
    for (uint32_t t = 0; t < dt; ++t) {
        noc.async_write(out, result, out.get_entry_size(), {.offset_bytes = t * out.get_entry_size()}, {.page_id = t});
    }
    noc.async_write_barrier();
    out.pop_front(dt);
}
"""

COMPUTE_SOURCE = """
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// Packs DST tile `dst` as one new entry of `dfb` (each pack to a different DFB needs its own pack_init on Quasar).
#define PACK_ONE(dst, buf, token) \\
    buf.reserve_back(1);          \\
    pack_init(token);             \\
    pack_tile(dst, token);        \\
    buf.push_back(1);

void kernel_main() {
    constexpr uint32_t dt = get_arg(args::dt);
    constexpr uint32_t chunks = get_arg(args::chunks);
    DataflowBuffer q(dfb::q);
    DataflowBuffer kt(dfb::kt);
    DataflowBuffer v(dfb::v);
    DataflowBuffer ones(dfb::ones);
    DataflowBuffer s(dfb::s);          // scores of this chunk
    DataflowBuffer p(dfb::p);          // exp(S - m')
    DataflowBuffer m(dfb::m);          // running row max: [old, new]
    DataflowBuffer l(dfb::l);          // running row sum: [old, new]
    DataflowBuffer o(dfb::o);          // running output: dt tiles per generation
    DataflowBuffer alpha(dfb::alpha);  // exp(m - m'), and 1 / l at the end
    DataflowBuffer out(dfb::out);
    compute_kernel_hw_startup(dfb::q, dfb::kt, dfb::out);
    q.wait_front(dt);
    kt.wait_front(dt * chunks);
    v.wait_front(chunks * dt);
    ones.wait_front(1);

    for (uint32_t c = 0; c < chunks; ++c) {
        // S = q @ kT[:, c]
        matmul_init(dfb::q, dfb::kt, 0);
        tile_regs_acquire();
        for (uint32_t k = 0; k < dt; ++k) {
            matmul_tiles(dfb::q, dfb::kt, k, k * chunks + c, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        PACK_ONE(0, s, dfb::s);
        tile_regs_release();
        s.wait_front(1);

        // m' = max(m, rowmax S) in DST 0; alpha = exp(m - m') in DST 2 (column 0 holds the row values)
        tile_regs_acquire();
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(dfb::s, dfb::ones, dfb::m);
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(dfb::s, dfb::ones, 0, 0, 0);
        reduce_uninit();
        if (c > 0) {
            copy_init(dfb::m);
            copy_tile(dfb::m, 0, 1);
            binary_max_tile_init();
            binary_max_tile(1, 0, 0);
            sub_binary_tile_init();
            sub_binary_tile(1, 0, 2);
            exp_tile_init();
            exp_tile(2);
        }
        tile_regs_commit();
        tile_regs_wait();
        PACK_ONE(0, m, dfb::m);
        if (c > 0) {
            PACK_ONE(2, alpha, dfb::alpha);
        }
        tile_regs_release();
        if (c > 0) {
            m.pop_front(1);  // the old max (read by copy_tile above)
        }
        m.wait_front(1);

        // P = exp(S - m')
        tile_regs_acquire();
        sub_bcast_cols_init(dfb::s, dfb::m);
        sub_tiles_bcast_cols(dfb::s, dfb::m, 0, 0, 0);
        exp_tile_init();
        exp_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        PACK_ONE(0, p, dfb::p);
        tile_regs_release();
        s.pop_front(1);
        p.wait_front(1);
        if (c > 0) {
            alpha.wait_front(1);
        }

        // l' = alpha l + rowsum P
        tile_regs_acquire();
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb::p, dfb::ones, dfb::l);
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb::p, dfb::ones, 0, 0, 0);
        reduce_uninit();
        if (c > 0) {
            copy_init(dfb::l);
            copy_tile(dfb::l, 0, 1);
            copy_init(dfb::alpha);
            copy_tile(dfb::alpha, 0, 2);
            mul_binary_tile_init();
            mul_binary_tile(1, 2, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        PACK_ONE(0, l, dfb::l);
        tile_regs_release();
        if (c > 0) {
            l.pop_front(1);
        }

        // O' = alpha O + P @ v[c, :]
        tile_regs_acquire();
        matmul_init(dfb::p, dfb::v, 0);
        for (uint32_t j = 0; j < dt; ++j) {
            matmul_tiles(dfb::p, dfb::v, 0, c * dt + j, j);
        }
        if (c > 0) {
            mul_bcast_cols_init(dfb::o, dfb::alpha);
            for (uint32_t j = 0; j < dt; ++j) {
                mul_tiles_bcast_cols(dfb::o, dfb::alpha, j, 0, dt + j);
            }
            add_binary_tile_init();
            for (uint32_t j = 0; j < dt; ++j) {
                add_binary_tile(j, dt + j, j);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        o.reserve_back(dt);
        pack_init(dfb::o);
        for (uint32_t j = 0; j < dt; ++j) {
            pack_tile(j, dfb::o);
        }
        o.push_back(dt);
        tile_regs_release();
        p.pop_front(1);
        if (c > 0) {
            o.pop_front(dt);
            alpha.pop_front(1);
        }
        o.wait_front(dt);
        l.wait_front(1);
    }

    // out = O / l
    tile_regs_acquire();
    copy_init(dfb::l);
    copy_tile(dfb::l, 0, 0);
    recip_tile_init();
    recip_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    PACK_ONE(0, alpha, dfb::alpha);
    tile_regs_release();
    alpha.wait_front(1);
    tile_regs_acquire();
    mul_bcast_cols_init(dfb::o, dfb::alpha);
    for (uint32_t j = 0; j < dt; ++j) {
        mul_tiles_bcast_cols(dfb::o, dfb::alpha, j, 0, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    out.reserve_back(dt);
    pack_init(dfb::out);
    for (uint32_t j = 0; j < dt; ++j) {
        pack_tile(j, dfb::out);
    }
    out.push_back(dt);
    tile_regs_release();

    dummy_unpack(dfb::m);  // nothing reads the final max
    m.pop_front(1);
    l.pop_front(1);
    o.pop_front(dt);
    alpha.pop_front(1);
    q.pop_front(dt);
    kt.pop_front(dt * chunks);
    v.pop_front(chunks * dt);
    ones.pop_front(1);
}
"""


def test_generic_spec_op_flash_recompose(device):
    torch.manual_seed(0)
    torch_q = (torch.randn(32, D) / D**0.5).bfloat16()
    torch_k = torch.randn(SEQ, D).bfloat16()
    torch_v = torch.randn(SEQ, D).bfloat16()
    scores = torch_q.float() @ torch_k.float().T
    expected = torch.softmax(scores, dim=-1) @ torch_v.float()

    def tile_tensor(t):
        return ttnn.from_torch(t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tensors = {
        "q": tile_tensor(torch_q),
        "kt": tile_tensor(torch_k.T.contiguous()),
        "v": tile_tensor(torch_v),
        "ones": tile_tensor(torch.ones(32, 32)),
        "result": tile_tensor(torch.zeros(32, D)),
    }
    inputs = {"q": DT, "kt": DT * CHUNKS, "v": CHUNKS * DT, "ones": 1}
    loops = {"s": 1, "p": 1, "m": 2, "l": 2, "o": 2 * DT, "alpha": 1}

    def dfb(name, num_entries):
        return E.DataflowBufferSpec(
            unique_id=name, entry_size=TILE_BYTES, num_entries=num_entries, data_format_metadata=ttnn.bfloat16
        )

    args = {"dt": DT, "chunks": CHUNKS}
    spec = E.ProgramSpec(
        name="flash_recompose",
        kernels=[
            E.KernelSpec(
                unique_id="dm",
                source=E.KernelSpec.SourceCode(code=DM_SOURCE),
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                dfb_bindings=[E.producer_of(n, n) for n in inputs] + [E.consumer_of("out", "out")],
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
                + [E.producer_of("out", "out")],
                compile_time_args=args,
            ),
        ],
        dataflow_buffers=[dfb(n, e) for n, e in {**inputs, **loops, "out": DT}.items()],
        tensor_parameters=[E.TensorParameter(unique_id=t, spec=v.spec) for t, v in tensors.items()],
        work_units=[E.WorkUnitSpec(kernels=["dm", "compute"], target_nodes=CORE)],
    )
    E.generic_op(list(tensors.values()), spec, E.ProgramRunArgs(tensor_args=tensors))

    assert_with_pcc(expected, ttnn.to_torch(tensors["result"]).float(), 0.99)
