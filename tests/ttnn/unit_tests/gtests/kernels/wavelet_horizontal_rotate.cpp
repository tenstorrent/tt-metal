// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/operations/wavelet/device/kernels/sfpi/horizontal_stencil_sfpi.h"

void kernel_main() {
    constexpr uint32_t input_cb = tt::CBIndex::c_0;
    constexpr uint32_t output_cb = tt::CBIndex::c_16;
    CircularBuffer input_buffer(input_cb);
    CircularBuffer output_buffer(output_cb);

    compute_kernel_hw_startup(input_cb, output_cb);
    copy_init(input_cb);

    input_buffer.wait_front(2);
    output_buffer.reserve_back(1);
    tile_regs_acquire();
    copy_tile(input_cb, 0, 0);
    copy_tile(input_cb, 1, 1);
    hstencil_init();

    MATH(([] {
        ckernel::math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
        ckernel::sfpu::_lwt_clear_addr_mod_base_();
        TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
        for (uint32_t face = 0; face < 4; ++face) {
            for (uint32_t row = 0; row < 16; row += 4) {
                for (uint32_t parity = 0; parity < 2; ++parity) {
                    constexpr auto a_reg = p_sfpu::LREG0;
                    constexpr auto b_reg = p_sfpu::LREG1;
                    const uint32_t offset = row + 2 * parity;
                    TT_SFPLOAD(
                        a_reg,
                        sfpi::SFPLOAD_MOD0_FMT_FP32,
                        ADDR_MOD_3,
                        ckernel::sfpu::_lwt_dst_base(0, face) + offset);
                    TT_SFPLOAD(
                        b_reg,
                        sfpi::SFPLOAD_MOD0_FMT_FP32,
                        ADDR_MOD_3,
                        ckernel::sfpu::_lwt_dst_base(1, face) + offset);
                    ckernel::sfpu::_horizontal_stencil_rotate_(a_reg, b_reg);
                    TT_SFPSTORE(
                        b_reg,
                        sfpi::SFPSTORE_MOD0_FMT_FP32,
                        ADDR_MOD_3,
                        ckernel::sfpu::_lwt_dst_base(2, face) + offset);
                }
            }
        }
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    }()));

    tile_regs_commit();
    tile_regs_wait();
    pack_tile(2, output_cb, 0);
    tile_regs_release();
    output_buffer.push_back(1);
    input_buffer.pop_front(2);
}
