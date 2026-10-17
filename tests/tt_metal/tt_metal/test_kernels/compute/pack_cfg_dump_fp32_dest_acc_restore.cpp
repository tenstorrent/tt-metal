// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Packer configuration after a dest-acc override is restored.
//
// Built with fp32_dest_acc_en = true. The kernel switches Dest to 16-bit mode with
// set_fp32_dest_acc<false>(), reconfigures the packer from the Float16 output configured at startup to
// the Bfp8 (5-bit exponent) output while in that mode -- which rightly programs no exponent threshold
// for a 16-bit Dest -- and restores the 32-bit Dest with restore_fp32_dest_acc<false>(). The pack thread
// then writes the packer fields the mode decides into the output page, in place of a tile:
//   word 0: THCON_SEC0_REG1 Exp_threshold_en  (fp32 Dest -> BFP-A underflow workaround; must be 1)
//   word 1: THCON_SEC0_REG1 Exp_threshold     (the 8-bit exponent floor of a 5-bit exponent; must be 113)
//   word 2: PCK_DEST_RD_CTRL Read_32b_data    (must be 1 again)
//   word 3: PCK_DEST_RD_CTRL Round_10b_mant   (Bfp8 source: must be 0)

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr auto cb_in = tt::CBIndex::c_0;        // Float32, never consumed
    constexpr auto cb_out = tt::CBIndex::c_16;      // Bfp8, receives the dumped fields
    constexpr auto cb_startup = tt::CBIndex::c_17;  // Float16, only configures the packer at startup

    CircularBuffer out(cb_out);

    compute_kernel_hw_startup(cb_in, cb_startup);
    copy_init(cb_in);

    // A 16-bit Dest section that leaves the packer configured for the Bfp8 output.
    set_fp32_dest_acc<false>(cb_startup);
    pack_reconfig_data_format(cb_startup, cb_out);
    // Back to the 32-bit Dest the kernel was built for; the Bfp8 output now needs its threshold again.
    restore_fp32_dest_acc<false>(cb_out);

    out.reserve_back(1);
    PACK(({
        // fifo_wr_ptr is a 16-byte L1 address (the unit the packer's L1_Dest_addr takes).
        volatile std::uint32_t* words = reinterpret_cast<volatile std::uint32_t*>(out.get_write_ptr() << 4);
        // The restore's threshold / rounding writes are Tensix RMWs queued on this thread; cfg_read is a raw
        // load, so retire them first.
        tensix_sync();
        const std::uint32_t threshold_word = cfg_read(THCON_SEC0_REG1_Exp_threshold_ADDR32);
        const std::uint32_t dest_rd_word = cfg_read(PCK_DEST_RD_CTRL_Read_32b_data_ADDR32);
        words[0] = (threshold_word & THCON_SEC0_REG1_Exp_threshold_en_MASK) >> THCON_SEC0_REG1_Exp_threshold_en_SHAMT;
        words[1] = (threshold_word & THCON_SEC0_REG1_Exp_threshold_MASK) >> THCON_SEC0_REG1_Exp_threshold_SHAMT;
        words[2] = (dest_rd_word & PCK_DEST_RD_CTRL_Read_32b_data_MASK) >> PCK_DEST_RD_CTRL_Read_32b_data_SHAMT;
        words[3] = (dest_rd_word & PCK_DEST_RD_CTRL_Round_10b_mant_MASK) >> PCK_DEST_RD_CTRL_Round_10b_mant_SHAMT;
    }));
    out.push_back(1);
}
