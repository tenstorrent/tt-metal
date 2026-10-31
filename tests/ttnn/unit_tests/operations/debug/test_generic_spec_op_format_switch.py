# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Two parts with different data formats fused into one compute kernel, through ttnn.experimental.generic_op.

Per token, x[32, 128] (bf16) and resident block-float weights w[128, 32]:

    a = x + x  (bf16 in, bf16 out);  out = a @ w  (bf16 in0, block-float in1, fp32 out)

The weights are bfloat8_b on Blackhole and mxfp8_e4m3 on Quasar, which has no bfloat8_b
(tt::is_data_format_supported); both are 8-bit elements with a shared exponent per block.

compute_kernel_hw_startup runs once, for part 1. On every iteration each part's entry switches the formats to its own
operands, the way tt-blaze's per-part reinit() would for parts that do not agree on format:
reconfig_data_format for its unpack operands, pack_reconfig_data_format for its output DFB, then its op init and
pack_init. Part 1 switches back from part 2's formats on iterations 1 and 2.

On Quasar the op init and pack_init program the buffer descriptors, which carry each DFB's L1 format;
reconfig_data_format reprograms the unpacker's output (source register) format, and the math half of it is a no-op
(llk_math_common_api.h). NO_PART2_RECONFIG (a mutant, not part of the test) drops part 2's reconfig_data_format, and
still passes on Quasar: mxfp8_e4m3 and bf16 both unpack to Float16_b source registers, so the reconfig rewrites the
same value, and the L1 formats travel in the buffer descriptors. On Blackhole, bfloat8_b unpacks to Bfp8_b source
registers, so there the reconfig does change srcA.

Full 32x32 tiles throughout: on Quasar the binary unpack path supports full tiles only for now (tt-metal #47597).
"""

import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from tt_py_test_utils_common.utility_functions import is_quasar

E = ttnn.experimental

CORE = ttnn.CoreCoord(0, 0)
M, K, N = 32, 128, 32
KT = K // 32
NUM_ITERS = 3

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
    DataflowBuffer out(dfb::out);
    const auto x_tensor = TensorAccessor(tensor::x);
    const auto result = TensorAccessor(tensor::result);
    load(w, TensorAccessor(tensor::w), 0, kt);  // resident for every token
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
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t kt = get_arg(args::kt);
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    DataflowBuffer x(dfb::x);
    DataflowBuffer w(dfb::w);
    DataflowBuffer a(dfb::a);
    DataflowBuffer out(dfb::out);
    compute_kernel_hw_startup(dfb::x, dfb::x, dfb::a);
    w.wait_front(kt);
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        x.wait_front(kt);

        // Part 1: a = x + x, bf16 -> bf16.
        reconfig_data_format(dfb::x, dfb::x);
        pack_reconfig_data_format(dfb::a);
        binary_tiles_init<true, EltwiseBinaryType::ELWADD>(dfb::x, dfb::x);
        pack_init(dfb::a);
        tile_regs_acquire();
        for (uint32_t t = 0; t < kt; ++t) {
            add_tiles(dfb::x, dfb::x, t, t, t);
        }
        tile_regs_commit();
        a.reserve_back(kt);
        tile_regs_wait();
        for (uint32_t t = 0; t < kt; ++t) {
            pack_tile(t, dfb::a);
        }
        tile_regs_release();
        a.push_back(kt);
        x.pop_front(kt);
        a.wait_front(kt);

        // Part 2: out = a @ w, bf16 x block float -> fp32. Matmul unpacks in0 to srcB and in1 to srcA.
#ifndef NO_PART2_RECONFIG
        reconfig_data_format<SrcOrder::Reverse>(dfb::a, dfb::w);
#endif
        pack_reconfig_data_format(dfb::out);
        matmul_block_init(dfb::a, dfb::w, /*transpose=*/0, /*ct_dim=*/1, /*rt_dim=*/1, /*kt_dim=*/kt);
        pack_init(dfb::out);
        tile_regs_acquire();
        for (uint32_t k = 0; k < kt; ++k) {
            matmul_block(dfb::a, dfb::w, k, k, 0, /*transpose=*/0, 1, 1, kt);
        }
        tile_regs_commit();
        out.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, dfb::out);
        tile_regs_release();
        out.push_back(1);
        a.pop_front(kt);
    }
    w.pop_front(kt);
}
"""


def run_format_switch(device, defines=None):
    """Returns (expected with the block-float weights as stored, expected with the bf16 weights, result)."""
    w_dtype = ttnn.mxfp8_e4m3 if is_quasar() else ttnn.bfloat8_b
    torch.manual_seed(0)
    torch_x = torch.randn(NUM_ITERS * M, K).bfloat16()
    torch_w = (torch.randn(K, N) / 8).bfloat16()

    def tile_tensor(t, dtype):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    tensors = {
        "x": tile_tensor(torch_x, ttnn.bfloat16),
        "w": tile_tensor(torch_w, w_dtype),
        "result": tile_tensor(torch.zeros(NUM_ITERS * M, N), ttnn.float32),
    }
    a = (torch_x.float() * 2).reshape(NUM_ITERS, M, K)  # exact in bf16
    expected_stored = (a @ ttnn.to_torch(tensors["w"]).float()).reshape(NUM_ITERS * M, N)
    expected_bf16 = (a @ torch_w.float()).reshape(NUM_ITERS * M, N)

    def dfb(name, num_entries, dtype):
        return E.DataflowBufferSpec(
            unique_id=name,
            entry_size=ttnn.Tile([32, 32]).get_tile_size(dtype),
            num_entries=num_entries,
            data_format_metadata=dtype,
        )

    common = {"kt": KT, "num_iters": NUM_ITERS}
    spec = E.ProgramSpec(
        name="format_switch",
        kernels=[
            E.KernelSpec(
                unique_id="dm",
                source=E.KernelSpec.SourceCode(code=DM_SOURCE),
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                dfb_bindings=[E.producer_of("x", "x"), E.producer_of("w", "w"), E.consumer_of("out", "out")],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name=t, accessor_name=t) for t in tensors],
                compile_time_args=common,
            ),
            E.KernelSpec(
                unique_id="compute",
                source=E.KernelSpec.SourceCode(code=COMPUTE_SOURCE),
                hw_config=E.ComputeHardwareConfig(math_fidelity=ttnn.MathFidelity.HiFi4, enable_32_bit_dest=True),
                compiler_options=E.KernelSpec.CompilerOptions(defines=defines or {}),
                dfb_bindings=[
                    E.consumer_of("x", "x"),
                    E.consumer_of("w", "w"),
                    E.producer_of("a", "a"),
                    E.consumer_of("a", "a"),
                    E.producer_of("out", "out"),
                ],
                compile_time_args=common,
            ),
        ],
        dataflow_buffers=[
            dfb("x", KT, ttnn.bfloat16),
            dfb("w", KT, w_dtype),
            dfb("a", KT, ttnn.bfloat16),
            dfb("out", 1, ttnn.float32),
        ],
        tensor_parameters=[E.TensorParameter(unique_id=t, spec=v.spec) for t, v in tensors.items()],
        work_units=[E.WorkUnitSpec(kernels=["dm", "compute"], target_nodes=CORE)],
    )
    E.generic_op(list(tensors.values()), spec, E.ProgramRunArgs(tensor_args=tensors))
    return expected_stored, expected_bf16, ttnn.to_torch(tensors["result"]).float()


def test_generic_spec_op_format_switch(device):
    expected_stored, expected_bf16, result = run_format_switch(device)
    # Against the weights as stored, only the matmul's bf16 source registers round; against the bf16 weights, the
    # block format's shared exponents cost a little more.
    assert_with_pcc(expected_stored, result, 0.999)
    assert_with_pcc(expected_bf16, result, 0.99)
