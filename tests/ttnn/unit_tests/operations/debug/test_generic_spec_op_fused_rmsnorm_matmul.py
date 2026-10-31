# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Two tt-blaze ops fused into one compute kernel: deferred RMSNorm, then matmul, through ttnn.experimental.generic_op.

Per decode token, x[8, 128] is normalized and multiplied by resident weights w[128, 32]:

    xn = x * rsqrt(mean(x^2, over K) + eps);  out = xn @ w

The compute kernel runs the parts in sequence, each through its own DFB:
  1. x * x (mul_tiles) -> xsq
  2. reduce SUM over rows of xsq, * 1/K (binop_with_scalar), + eps (an eps tile and the SFPU add_binary_tile, as
     Quasar has no add_unary_tile), rsqrt_tile -> rinv
  3. x * rinv (mul_tiles_bcast_cols) -> xn
  4. matmul_block(xn, w) -> out
compute_kernel_hw_startup runs once, and every part re-runs its op init and the pack_init of its output DFB on every
iteration, as blaze's codegen emits a fused kernel: on Quasar each op init and each pack to another DFB reprograms the
unpacker, math and packer. NO_REINIT (a mutant, not part of the test) runs the inits only on the first iteration.

The 8 token rows are padded to full 32x32 tiles (rows 8-31 zero). On Quasar the binary (mul, bcast, reduce) and unary
broadcast unpack paths support full tiles only for now (tt-metal #47597; LLK asserts in llk_unpack_AB_init and
llk_unpack_A_init), although copy and the standard matmul take Tile([8, 32]).
"""

import struct

import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

E = ttnn.experimental

CORE = ttnn.CoreCoord(0, 0)
M, K, N = 8, 128, 32
KT = K // 32
NUM_ITERS = 3
EPS = 1e-5
ROWS = 32  # M padded to a full tile: see the module docstring
FULL_BYTES = 32 * 32 * 2

DM_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t kt = get_arg(args::kt);
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    Noc noc;
    auto load = [&](auto& dfb, const auto& tensor, uint32_t first_page, uint32_t num_tiles) {
        dfb.reserve_back(num_tiles);
        for (uint32_t t = 0; t < num_tiles; ++t) {
            noc.async_read(
                tensor, dfb, dfb.get_entry_size(), {.page_id = first_page + t}, {.offset_bytes = t * dfb.get_entry_size()});
        }
        noc.async_read_barrier();
        dfb.push_back(num_tiles);
    };
    DataflowBuffer x(dfb::x);
    DataflowBuffer w(dfb::w);
    DataflowBuffer scaler(dfb::scaler);
    DataflowBuffer out(dfb::out);
    const auto x_tensor = TensorAccessor(tensor::x);
    const auto result = TensorAccessor(tensor::result);
    load(w, TensorAccessor(tensor::w), 0, kt);  // resident for every token
    load(scaler, TensorAccessor(tensor::scaler), 0, 1);
    DataflowBuffer eps(dfb::eps);
    load(eps, TensorAccessor(tensor::eps), 0, 1);
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        load(x, x_tensor, iter * kt, kt);
        out.wait_front(1);
        noc.async_write(out, result, out.get_entry_size(), {.offset_bytes = 0}, {.page_id = iter});
        noc.async_write_barrier();
        out.pop_front(1);
    }
}
"""

COMPUTE_SOURCE = """
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#ifdef NO_REINIT
#define REINIT (iter == 0)
#else
#define REINIT true
#endif

void kernel_main() {
    constexpr uint32_t kt = get_arg(args::kt);
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    constexpr uint32_t inv_k = get_arg(args::inv_k);  // fp32 bits of 1 / K
    DataflowBuffer x(dfb::x);
    DataflowBuffer w(dfb::w);
    DataflowBuffer scaler(dfb::scaler);
    DataflowBuffer eps(dfb::eps);
    DataflowBuffer xsq(dfb::xsq);
    DataflowBuffer rinv(dfb::rinv);
    DataflowBuffer xn(dfb::xn);
    DataflowBuffer out(dfb::out);
    compute_kernel_hw_startup(dfb::x, dfb::x, dfb::xsq);
    w.wait_front(kt);
    scaler.wait_front(1);
    eps.wait_front(1);
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        x.wait_front(kt);

        // 1. xsq = x * x
        if (REINIT) {
            binary_tiles_init<true, EltwiseBinaryType::ELWMUL>(dfb::x, dfb::x);
        }
        tile_regs_acquire();
        for (uint32_t t = 0; t < kt; ++t) {
            mul_tiles(dfb::x, dfb::x, t, t, t);
        }
        tile_regs_commit();
        xsq.reserve_back(kt);
        tile_regs_wait();
        if (REINIT) {
            pack_init(dfb::xsq);
        }
        for (uint32_t t = 0; t < kt; ++t) {
            pack_tile(t, dfb::xsq);
        }
        tile_regs_release();
        xsq.push_back(kt);
        xsq.wait_front(kt);

        // 2. rinv = rsqrt(sum(xsq over the row) / K + eps)
        if (REINIT) {
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb::xsq, dfb::scaler, dfb::rinv);
        }
        tile_regs_acquire();
        for (uint32_t t = 0; t < kt; ++t) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb::xsq, dfb::scaler, t, 0, 0);
        }
        if (REINIT) {
            reduce_uninit();
            binop_with_scalar_tile_init();
        }
        mul_unary_tile(0, inv_k);
        // + eps: Quasar has no add_unary_tile (binop_with_scalar.h offers only mul_unary_tile there), so add an eps
        // tile with the SFPU binary add.
        if (REINIT) {
            copy_init(dfb::eps);
        }
        copy_tile(dfb::eps, 0, 1);
        if (REINIT) {
            add_binary_tile_init();
        }
        add_binary_tile(0, 1, 0);
        if (REINIT) {
            rsqrt_tile_init();
        }
        rsqrt_tile(0);
        tile_regs_commit();
        rinv.reserve_back(1);
        tile_regs_wait();
        if (REINIT) {
            pack_init(dfb::rinv);
        }
        pack_tile(0, dfb::rinv);
        tile_regs_release();
        rinv.push_back(1);
        xsq.pop_front(kt);
        rinv.wait_front(1);

        // 3. xn = x * rinv (each row by its own factor)
        if (REINIT) {
            mul_bcast_cols_init(dfb::x, dfb::rinv);
        }
        tile_regs_acquire();
        for (uint32_t t = 0; t < kt; ++t) {
            mul_tiles_bcast_cols(dfb::x, dfb::rinv, t, 0, t);
        }
        tile_regs_commit();
        xn.reserve_back(kt);
        tile_regs_wait();
        if (REINIT) {
            pack_init(dfb::xn);
        }
        for (uint32_t t = 0; t < kt; ++t) {
            pack_tile(t, dfb::xn);
        }
        tile_regs_release();
        xn.push_back(kt);
        rinv.pop_front(1);
        x.pop_front(kt);
        xn.wait_front(kt);

        // 4. out = xn @ w
        if (REINIT) {
            matmul_block_init(dfb::xn, dfb::w, /*transpose=*/0, /*ct_dim=*/1, /*rt_dim=*/1, /*kt_dim=*/kt);
        }
        tile_regs_acquire();
        for (uint32_t k = 0; k < kt; ++k) {
            matmul_block(dfb::xn, dfb::w, k, k, 0, /*transpose=*/0, 1, 1, kt);
        }
        tile_regs_commit();
        out.reserve_back(1);
        tile_regs_wait();
        if (REINIT) {
            pack_init(dfb::out);
        }
        pack_tile(0, dfb::out);
        tile_regs_release();
        out.push_back(1);
        xn.pop_front(kt);
    }
    w.pop_front(kt);
    scaler.pop_front(1);
    eps.pop_front(1);
}
"""


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def run_fused_rmsnorm_matmul(device, defines=None):
    """Returns (expected, result) for NUM_ITERS tokens."""
    torch.manual_seed(0)
    torch_x = torch.zeros(NUM_ITERS, ROWS, K)
    torch_x[:, :M] = torch.randn(NUM_ITERS, M, K)
    torch_x = torch_x.reshape(NUM_ITERS * ROWS, K).bfloat16()
    torch_w = (torch.randn(K, N) / 8).bfloat16()
    x = torch_x.float()
    xn = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    expected = xn @ torch_w.float()

    def tile_tensor(t):
        return ttnn.from_torch(t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tensors = {
        "x": tile_tensor(torch_x),
        "w": tile_tensor(torch_w),
        "scaler": tile_tensor(torch.ones(32, 32)),
        "eps": tile_tensor(torch.full((32, 32), EPS)),
        "result": tile_tensor(torch.zeros(NUM_ITERS * ROWS, N)),
    }

    def dfb(name, num_entries):
        return E.DataflowBufferSpec(
            unique_id=name, entry_size=FULL_BYTES, num_entries=num_entries, data_format_metadata=ttnn.bfloat16
        )

    inputs = ["x", "w", "scaler", "eps"]
    loops = ["xsq", "rinv", "xn"]
    common = {"kt": KT, "num_iters": NUM_ITERS}
    spec = E.ProgramSpec(
        name="fused_rmsnorm_matmul",
        kernels=[
            E.KernelSpec(
                unique_id="dm",
                source=E.KernelSpec.SourceCode(code=DM_SOURCE),
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                dfb_bindings=[E.producer_of(n, n) for n in inputs] + [E.consumer_of("out", "out")],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name=t, accessor_name=t) for t in tensors],
                compile_time_args=common,
            ),
            E.KernelSpec(
                unique_id="compute",
                source=E.KernelSpec.SourceCode(code=COMPUTE_SOURCE),
                hw_config=E.ComputeHardwareConfig(),
                compiler_options=E.KernelSpec.CompilerOptions(defines=defines or {}),
                dfb_bindings=[E.consumer_of(n, n) for n in inputs]
                + [E.producer_of(n, n) for n in loops]
                + [E.consumer_of(n, n) for n in loops]
                + [E.producer_of("out", "out")],
                compile_time_args={**common, "inv_k": f32_bits(1.0 / K)},
            ),
        ],
        dataflow_buffers=[
            dfb("x", KT),
            dfb("w", KT),
            dfb("scaler", 1),
            dfb("eps", 1),
            dfb("xsq", KT),
            dfb("rinv", 1),
            dfb("xn", KT),
            dfb("out", 1),
        ],
        tensor_parameters=[E.TensorParameter(unique_id=t, spec=v.spec) for t, v in tensors.items()],
        work_units=[E.WorkUnitSpec(kernels=["dm", "compute"], target_nodes=CORE)],
    )
    E.generic_op(list(tensors.values()), spec, E.ProgramRunArgs(tensor_args=tensors))
    return expected, ttnn.to_torch(tensors["result"]).float()


def test_generic_spec_op_fused_rmsnorm_matmul(device):
    expected, result = run_fused_rmsnorm_matmul(device)
    assert_with_pcc(expected, result, 0.999)
