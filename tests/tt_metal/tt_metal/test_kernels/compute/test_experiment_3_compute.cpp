// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/kernel_thread_globals.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#ifdef UCK_CHLKC_ISOLATE_SFPU
#include "llk_sfpu_srcs_api.h"  // TRISC3 only: SrcS helpers (Unpacker2 config, slice shape, buffer descriptors, SFPU init)
#endif

void kernel_main() {
    std::uint32_t tiles_per_neo = get_arg(args::tiles_per_neo);  // rt arg per neo cluster

    const std::uint32_t num_neo_tensix = get_num_threads();
    const std::uint32_t my_neo_id = get_my_thread_id();
    const std::uint32_t tile_count =
        tiles_per_neo / num_neo_tensix + (my_neo_id < (tiles_per_neo % num_neo_tensix) ? 1 : 0);

    // in0 * in1 + in2 = out
    DataflowBuffer dfb_in0(dfb::in_0);
    DataflowBuffer dfb_in1(dfb::in_1);
    DataflowBuffer dfb_in2(dfb::in_2);
    DataflowBuffer dfb_out(dfb::out);

    std::uint32_t dst_tiles = 1;
    std::uint32_t dst_tile_idx = 0;

    compute_kernel_hw_startup(dfb_in0.get_id(), dfb_in1.get_id(), dfb_out.get_id());

#ifdef UCK_CHLKC_ISOLATE_SFPU
    // ---- TRISC3 one-time setup: Unpacker2 -> SrcS -> SFPU -> Dest tile 0 ----
    constexpr DataFormat srcs_format = DataFormat::Float16_b;  // c is bf16 in L1, keep bf16 in SrcS and Dest
    constexpr bool srcs_32bit_mode = false;                    // 16-bit data -> SrcS in 16-bit mode
    const std::uint32_t srcs_ydim =
        ckernel::trisc::srcs_dims::ydim(srcs_32bit_mode);  // rows per SrcS slice (8 for 16-bit)
    const std::uint32_t srcs_slice_count =
        ckernel::trisc::srcs_dims::slice_count(srcs_32bit_mode);  // slices per 32x32 tile (8)
    const ckernel::TensorShape srcs_shape =
        llk_sfpu_srcs_slice_shape_impl(srcs_ydim);    // one slice = ydim rows x 16 cols
    constexpr std::uint32_t l1_addr_unit_bytes = 16;  // buffer descriptors take L1 addresses in 16-byte units
    constexpr std::uint32_t sfpu_rows_per_pass =
        SFP_ROWS;  // each SFPLOAD/SFPSTORE moves 2 rows (ckernel::math::SFP_ROWS)
    const std::uint32_t passes_per_slice = srcs_ydim / sfpu_rows_per_pass;  // 8 rows / 2 = 4 load+store pairs per slice
    _llk_unpack_configure_unary_<p_unpacr::UNP_S>(srcs_format);             // format Unpacker2 writes into SrcS
    cfg[DISABLE_IMPLIED_SRCS_FORMAT_ADDR32 + ckernel::TRISC_ID] =
        1;  // explicit (not implied) SrcS math format, like Metal's SrcA/B
    _llk_unpack_srcs_config_for_tile_<1>(
        srcs_32bit_mode);            // Unpacker2 auto-loop: one unpack call covers all slices of a tile
    _llk_math_eltwise_sfpu_init_();  // SFPU init on this TRISC (address modes etc.)
    _set_dst_write_addr_<ckernel::trisc::DstTileShape::Tile32x32>(
        0);  // TRISC3's Dest section base -> tile 0, the tile math multiply-accumulates into
    const std::uint32_t load_sfpmem = _sfpu_sfpmem_type_(srcs_format);   // SFPLOAD format code (reading bf16 from SrcS)
    const std::uint32_t store_sfpmem = _sfpu_sfpmem_type_(srcs_format);  // SFPSTORE format code (writing bf16 to Dest)
#endif

    for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
        dfb_in0.wait_front(dst_tiles);
        dfb_in1.wait_front(dst_tiles);
        dfb_in2.wait_front(dst_tiles);

        dummy_unpack(dfb_in2.get_id());    // this call is UNPACK only; there because TRISC3 does the actual unpack
        UNPACK((ckernel::tensix_sync()));  // this is needed to not race ahead of data arriving

        const std::uint32_t in2_addr =
            dfb_in2.get_tile_address(0);  // sends in2 tile address from UNPACK thread to others via mailbox

#ifdef UCK_CHLKC_ISOLATE_SFPU
        // ---- TRISC3 per tile: stream c (this DFB slot) through SrcS, relu, into Dest tile 0 ----
        // Point Unpacker2's buffer descriptor at c's slot (address in 16B units); ids cycle round-robin, fine per tile
        const std::uint8_t srcs_bfd = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp2_Slice0>(
            srcs_shape, in2_addr / l1_addr_unit_bytes, static_cast<std::uint32_t>(srcs_format));
        _llk_unpack_srcs_<1>(
            srcs_bfd, 0);  // queue Unpacker2 for the whole tile; it fills SrcS slice by slice as space frees up
        if (tile > 0) {
            ckernel::mailbox_read(
                ckernel::ThreadId::PackThreadId);  // wait until pack has finished reading Dest for the previous tile
        }
        for (std::uint32_t slice = 0; slice < srcs_slice_count; ++slice) {   // 8 slices = 8 chunks of 8 rows x 16 cols
            for (std::uint32_t pass = 0; pass < passes_per_slice; ++pass) {  // 4 passes x 2 rows = the slice's 8 rows
                const std::uint32_t row_in_slice = pass * sfpu_rows_per_pass;  // 0, 2, 4, 6
                const std::uint32_t row_in_tile =
                    slice * srcs_ydim + row_in_slice;  // where these rows live in the 64-row Dest tile
                TT_SFPLOAD(
                    p_sfpu::LREG0, load_sfpmem, ADDR_MOD_7, 0, SFPU_SRCS_BASE_ADDR + row_in_slice);  // SrcS -> LREG0

                TTI_SFPNONLINEAR(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpnonlinear::RELU_MODE);  // LREG1 = max(LREG0, 0)

                TT_SFPSTORE(
                    p_sfpu::LREG1,
                    store_sfpmem,
                    ADDR_MOD_7,
                    0,
                    SFPU_DEST_BASE_ADDR + row_in_tile);  // LREG1 -> Dest tile 0
            }
            _llk_math_eltwise_sfpu_srcs_clear_vlds_<true, false>();  // slice read done -> Unpacker2 may refill; no
                                                                     // write side (no Packer1)
        }
        ckernel::wait_sfpu_idle();  // all SFPU stores to Dest have landed
        ckernel::mailbox_write(
            ckernel::ThreadId::MathThreadId, 1);  // tell math: relu(c) is in Dest tile 0, multiply-accumulate on top
        ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 1);  // tell unpack: done reading c's slot, it may pop
#endif

        tile_regs_acquire();

#ifdef UCK_CHLKC_MATH
        ckernel::mailbox_read(
            ckernel::ThreadId::IsolateSfpuThreadId);  // wait until TRISC3 has written relu(c) into Dest tile 0
#endif
        mul_init(dfb_in0.get_id(), dfb_in1.get_id(), /* acc_to_dest= */ true);  // accumulate: Dest = a*b + Dest
        mul_tiles(
            dfb_in0.get_id(), dfb_in1.get_id(), dst_tile_idx, dst_tile_idx, dst_tile_idx);  // Dest[0] = a*b + relu(c)
        tile_regs_commit();

        dfb_in0.pop_front(dst_tiles);
        dfb_in1.pop_front(dst_tiles);
#ifdef UCK_CHLKC_UNPACK
        ckernel::mailbox_read(
            ckernel::ThreadId::IsolateSfpuThreadId);  // don't free c's slot until TRISC3's Unpacker2 has read it
#endif
        dfb_in2.pop_front(dst_tiles);
        dfb_out.reserve_back(dst_tiles);

        tile_regs_wait();
        pack_tile(dst_tile_idx, dfb_out.get_id());

        tile_regs_release();
#ifdef UCK_CHLKC_PACK
        if (tile + 1 < tile_count) {
            ckernel::tensix_sync();  // packer has actually executed: Dest is no longer being read
            ckernel::mailbox_write(
                ckernel::ThreadId::IsolateSfpuThreadId, 1);  // tell TRISC3: Dest is free, write the next tile
        }
#endif
        dfb_out.push_back(dst_tiles);
    }

    dfb_in0.finish();
    dfb_in1.finish();
    dfb_in2.finish();
    dfb_out.finish();
}
