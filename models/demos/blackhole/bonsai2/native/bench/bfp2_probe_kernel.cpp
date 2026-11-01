// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Single-tile matmul probe with a BF16 control path.
//
// The point is to separate "is my plumbing right" from "does BFP2_b work", which
// an all-zero result cannot distinguish. MODE selects:
//
//   MODE_BF16  both operands are ordinary bf16 tiles. If this is wrong, the bug
//              is in the CB protocol, the output path or the address plumbing and
//              has nothing to do with the format.
//   MODE_BFP2  both operands are raw 320-byte BFP2 tiles. Python cannot declare
//              DataFormat::Bfp2_b (the tt::DataFormat enum is not bound, so
//              CBFormatDescriptor rejects the value), so this path writes the
//              unpacker's tile-descriptor format register directly.
//
// A note on the earlier attempt, which returned exact zeros with no device
// assert: it called reconfig_data_format_srca(cb) expecting that to force a
// format. It does not -- it re-derives the unpacker state from the CB's declared
// format, which was uint8. The unpacker was therefore reading 320-byte pages as
// uint8, and the matmul produced zeros. Forcing a format has to be a register
// write, which is what MODE_BFP2 does.

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#endif

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#ifndef MODE_BFP2
#include "api/compute/reconfig_data_format.h"
#endif
#endif

// tt::DataFormat::Bfp2_b. Not reachable from Python, so it is spelled here.
constexpr uint32_t BFP2_B = 15;

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_b = 1;

    const uint32_t a_addr  = get_arg_val<uint32_t>(0);
    const uint32_t b_addr  = get_arg_val<uint32_t>(1);
    const uint32_t k_tiles = get_arg_val<uint32_t>(2);

#ifdef MODE_BFP2
    constexpr uint32_t a_bytes = 320;
    constexpr uint32_t b_bytes = 320;
#else
    constexpr uint32_t a_bytes = 2048;
    constexpr uint32_t b_bytes = 2048;
#endif

    const InterleavedAddrGenFast<true> a_gen{a_addr, a_bytes};
    const InterleavedAddrGenFast<true> b_gen{b_addr, b_bytes};

    for (uint32_t kt = 0; kt < k_tiles; ++kt) {
        cb_reserve_back(cb_a, 1);
        noc_async_read_page(kt, a_gen, get_write_ptr(cb_a));
        noc_async_read_barrier();
        cb_push_back(cb_a, 1);
        cb_reserve_back(cb_b, 1);
        noc_async_read_page(kt, b_gen, get_write_ptr(cb_b));
        noc_async_read_barrier();
        cb_push_back(cb_b, 1);
    }
#elif defined(COMPILE_FOR_BRISC)
    // The compute side packs straight into the output CB, which is the sharded
    // output tensor's own L1 buffer, so there is nothing to copy.
#elif defined(COMPILE_FOR_TRISC)
    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_b = 1;
    constexpr uint32_t cb_o = 17;
    const uint32_t k_tiles = get_arg_val<uint32_t>(2);

    // compute_kernel_hw_startup is mandatory and must come first; matmul is the
    // one op family that needs SrcOrder::Reverse.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_b, cb_o);
    matmul_init(cb_a, cb_b, cb_o);

#ifdef MODE_BFP2
    // Force the unpackers to read both operands as Bfp2_b (DataFormat 15).
    //
    // This MUST come after matmul_init, not before: matmul_init calls
    // llk_unpack_AB_matmul_init, which reprograms the tile descriptors from each
    // CB's declared format and silently overwrites an earlier register write.
    // That ordering mistake is why an earlier version of this probe returned exact
    // zeros for every encoding while the bf16 control passed.
    //
    // Both THCON contexts are written: SEC0 is the SrcA unpacker and SEC1 the
    // SrcB one, and matmul feeds both operands through the unpacker. The low
    // nibble of the tile descriptor is the input format. Use the named register
    // defines, not literals -- the address is 64 on Blackhole and 52 on Wormhole.
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, 0x0f>(BFP2_B);
    cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32, 0, 0x0f>(BFP2_B);
    // The *destination* format matters just as much: the CB was declared uint8, so
    // the unpacker's output format says uint8 and the math unit would read integer
    // register contents. Bfp2_b on the SrcA/SrcB path expands to Float16_b, which
    // is what these registers have to say for the MVMUL to see floats.
    constexpr uint32_t FLOAT16_B = 5;
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_ADDR32, 0, 0xf>(FLOAT16_B);
    cfg_reg_rmw_tensix<THCON_SEC1_REG2_Out_data_format_ADDR32, 0, 0xf>(FLOAT16_B);
    // The out-of-register format must be a width the BFP2 unpacker can produce;
    // leave it at what the CB declared (bf16), which is the documented target for
    // Bfp2_b on the SrcA/SrcB path.
#else
    reconfig_data_format_srca(cb_a);
    reconfig_data_format_srcb(cb_b);
#endif
    tile_regs_acquire();
    for (uint32_t kt = 0; kt < k_tiles; ++kt) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        matmul_tiles(cb_a, cb_b, 0, 0, 0);
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_o, 1);
    pack_tile(0, cb_o);
    cb_push_back(cb_o, 1);
    tile_regs_release();
#endif
}
