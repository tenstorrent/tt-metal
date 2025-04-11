// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// namespace ckernel{unsigned *instrn_buffer;}
#include "ckernel.h"
#include "compute_kernel_api.h"
#include <sfpi.h>

using namespace sfpi;
namespace NAMESPACE {
void MAIN {
#if 0
    auto cb_out = tt::CBIndex::c_16;

    // init??


    tile_regs_acquire();
#endif
#if COMPILE_FOR_TRISC == 1  // compute
    // do the math
    {
        vUInt c0ffee = vUInt(0x00c0) << 16 | vUInt(0xffee);
        vUInt deadbeef = vUInt(0xdead) << 16 | vUInt(0xbeef);
        vUInt c0edbabe = vUInt(0xc0ed) << 16 | vUInt(0xbabe);

        dst_reg[0] = c0ffee;
        dst_reg[1] = deadbeef;
        dst_reg[2] = c0edbabe;
        //        dbg_halt();
        auto* args = reinterpret_cast<tt_l1_ptr uint32_t*>(get_compile_time_arg_val(0));
        dbg_read_dest_acc_row(0, args);
        dbg_read_dest_acc_row(8, args + 8);
        // maybe +8 needed and then merge?
        ..dbg_unhalt();
    }
#endif
#if 0
    tile_regs_commit();
    tile_regs_wait();
    // FIXME: Copy directly to output area?
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);

    tile_regs_release();
#endif
#if 0

#if !defined(COMPILE_FOR_TRISC)
#error "COMPILE_FOR_TRISC not defined"
#elif COMPILE_FOR_TRISC == 0  // unpacker

#elif COMPILE_FOR_TRISC == 1  // compute
    auto* args = reinterpret_cast<tt_l1_ptr uint32_t*>(get_compile_time_arg_val(0));
    args[0] = args[1] = args[2] = 0;

    {
        vUInt c0ffee   = vUInt(0x00c0) << 16 | vUInt(0xffee);
        vUInt deadbeef = vUInt(0xdead) << 16 | vUInt(0xbeef);
        vUInt c0edbabe = vUInt(0xc0ed) << 16 | vUInt(0xbabe);

        dst_reg[0] = c0ffee;
        dst_reg[1] = deadbeed;
        dst_reg[2] = c0edbabe;
    }

#if 0
    args[0] = 0xc0ffee;
    args[1] = 0xdeadbeef;
    args[2] = 0xc0edbabe;
#endif
#elif COMPILE_FOR_TRISC == 2  // packer

#else
#error "Unknown COMPILE_FOR_TRISC value"
#endif
#endif
}
}  // namespace NAMESPACE
