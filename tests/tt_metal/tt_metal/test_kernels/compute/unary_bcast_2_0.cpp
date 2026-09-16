// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "defines_generated.h"
#ifndef BCAST_DIM
#define BCAST_DIM BroadcastType::ROW
#endif
#ifndef ACC_TO_DEST
#define ACC_TO_DEST 0
#endif

#include "api/compute/common.h"
#include "api/dataflow/circular_buffer.h"

// Id-free (2.0) unary_bcast kernel: per tile, broadcast c_0 -> DST (ROW by default), pack -> c_16.
// ACC_TO_DEST adds a second broadcast to the first before packing, exercising the public accumulation API.
// IDENTICAL to unary_bcast_legacy.cpp except the broadcast uses experimental::unary_bcast[_init] built from an
// LLKOperand (data format + tile geometry as NTTPs, absolute L1 address as the only runtime state) -- NO CB id
// on the op surface. The register format is derived on-device from the L1 format. hw_startup / pack_tile stay
// the legacy CB-id API so the differential isolates unary_bcast. Output must be bit-identical to the legacy kernel.
#include "api/compute/experimental/2_0/bcast.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"

void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    // Compile-time CB accessor -> folded descriptor; the operand bundles descriptor + runtime L1 address.
    constexpr auto in_cb = experimental::Cb<tt::CBIndex::c_0>{};
    constexpr auto in_desc = experimental::to_llk_mem_descriptor(in_cb);
    using InOp = experimental::LLKOperand<static_cast<DataFormat>(in_desc.format), in_desc.shape>;

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    experimental::unary_bcast_init<BCAST_DIM>(InOp(in_cb.read_address()));

    for (std::uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        tile_regs_acquire();
        cb0.wait_front(1);
        cb16.reserve_back(1);

        if constexpr (ACC_TO_DEST) {
            experimental::unary_bcast_init<BCAST_DIM>(InOp(in_cb.read_address()));
        }
        experimental::unary_bcast<BCAST_DIM>(InOp(in_cb.read_address()), 0);
        if constexpr (ACC_TO_DEST) {
            experimental::unary_bcast_init<BCAST_DIM, DST_ACCUM_MODE, true>(InOp(in_cb.read_address()));
            experimental::unary_bcast<BCAST_DIM, DST_ACCUM_MODE, true>(InOp(in_cb.read_address()), 0);
        }

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_16);

        cb0.pop_front(1);
        cb16.push_back(1);
        tile_regs_release();
    }
}
