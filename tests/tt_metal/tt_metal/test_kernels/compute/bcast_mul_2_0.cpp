// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"  // compute_kernel_hw_startup (CB-id) + tile_regs_* handshake
#include "api/dataflow/circular_buffer.h"
#include "api/compute/experimental/2_0/bcast.h"
#include "api/compute/experimental/2_0/pack.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"
#include "api/compute/experimental/2_0/hw_startup.h"

// Id-free (2.0) binary broadcast-MUL kernel, classic circular buffers. Per tile: C = A * broadcast(B) -> c_16.
// The broadcast axis is selected at compile time via get_compile_time_arg_val(1) (BroadcastType COL/ROW/SCALAR),
// so this ONE kernel covers bcast_mul_{cols,rows,scalar} -- the three differed only in that template arg. The
// broadcast uses experimental::bcast_init / any_tiles_bcast built from LLKOperands (data format + tile geometry
// as NTTPs, absolute L1 address the only runtime state) -- NO CB id on the op surface. hw_startup / pack_tile
// stay the legacy CB-id API so the differential isolates the bcast op. Output must be bit-identical to the
// legacy kernel for the selected broadcast axis.
namespace {
constexpr auto kBcast = static_cast<BroadcastType>(get_compile_time_arg_val(1));
}

void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    CircularBuffer c0(tt::CBIndex::c_0);
    CircularBuffer c1(tt::CBIndex::c_1);
    CircularBuffer c16(tt::CBIndex::c_16);

    constexpr auto in0_cb = experimental::Cb<tt::CBIndex::c_0>{};
    constexpr auto in1_cb = experimental::Cb<tt::CBIndex::c_1>{};
    constexpr auto out_cb = experimental::Cb<tt::CBIndex::c_16>{};
    constexpr auto in0_desc = experimental::to_llk_mem_descriptor(in0_cb);
    constexpr auto in1_desc = experimental::to_llk_mem_descriptor(in1_cb);
    constexpr auto out_desc = experimental::to_llk_mem_descriptor(out_cb);
    using AOp = experimental::LLKOperand<static_cast<DataFormat>(in0_desc.format), in0_desc.shape>;
    using BOp = experimental::LLKOperand<static_cast<DataFormat>(in1_desc.format), in1_desc.shape>;
    using OutOp = experimental::LLKOperand<static_cast<DataFormat>(out_desc.format), out_desc.shape>;

    compute_kernel_hw_startup(AOp(in0_cb.read_address()), BOp(in1_cb.read_address()), OutOp(out_cb.write_address()));
    experimental::bcast_init<EltwiseBinaryType::ELWMUL, kBcast>(AOp(in0_cb.read_address()));

    for (std::uint32_t t = 0; t < per_core_tile_cnt; ++t) {
        c0.wait_front(1);
        c1.wait_front(1);
        c16.reserve_back(1);

        tile_regs_acquire();
        experimental::any_tiles_bcast<EltwiseBinaryType::ELWMUL, kBcast>(
            AOp(in0_cb.read_address()), BOp(in1_cb.read_address()), 0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        experimental::pack_tile(OutOp(out_cb.write_address()), 0, 0);
        tile_regs_release();

        c0.pop_front(1);
        c1.pop_front(1);
        c16.push_back(1);
    }
}
