// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/compute/reduce.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr std::uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr std::uint32_t per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);

    DataflowBuffer dfb_in(dfb::in_data);
    DataflowBuffer dfb_in_scaler(dfb::in_scaler);
    DataflowBuffer dfb_out(dfb::out);

    // Skip compute_kernel_hw_startup: its UNPACK/PACK sweeps walk every DFB and race each other
    // on the shared bd_table[], so PACK can overwrite in_data's strided tilize descriptor with a
    // continuous one. Inline the same startup, but program only the three DFBs this kernel uses.
    // in_data's descriptor is left to tilizeA_B_reduce_init (strided); UNPACK only needs in_scaler
    // continuous + the unpack format registers.
    UNPACK({
        const std::uint32_t unpA_id = get_operand_id(dfb::in_data);
        const std::uint32_t unpB_id = get_operand_id(dfb::in_scaler);

        const tdma_descriptor_t td_b = ckernel::trisc::construct_tdma_desc(
            get_operand_tensor_shape(unpB_id),
            get_local_dfb_interface(unpB_id).tc_slots[0].base_addr,
            static_cast<std::uint32_t>(unpack_src_format[unpB_id]),
            unpB_id,
            static_cast<std::uint32_t>(unpack_dst_format[unpB_id]));
        ckernel::trisc::_configure_buf_desc_table_(unpB_id, td_b.buf_desc);

        tdma_descriptor_t td_val_A, td_val_B;
        td_val_A.reg_data_format = static_cast<std::uint8_t>(unpack_dst_format[unpA_id]);
        td_val_B.reg_data_format = static_cast<std::uint8_t>(unpack_dst_format[unpB_id]);
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val_A, td_val_B);
    });

    MATH((llk_math_pack_sync_init()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(dfb::in_data, dfb::in_scaler)));

    PACK({
        const std::uint32_t out_id = get_output_id(dfb::out);

        const tdma_descriptor_t td = ckernel::trisc::construct_tdma_desc(
            get_output_tensor_shape(out_id),
            get_local_dfb_interface(out_id).tc_slots[0].base_addr,
            pack_dst_format[out_id],
            out_id,
            pack_src_format[out_id]);
        ckernel::trisc::_configure_buf_desc_table_(out_id, td.buf_desc);

        tdma_descriptor_t td_val;
        td_val.reg_data_format = static_cast<std::uint8_t>(pack_src_format[out_id]);
        _llk_pack_hw_configure_<p_pacr::PACK0, DST_ACCUM_MODE>(td_val, ckernel::ReluConfig::none());
    });
    PACK((llk_pack_init(dfb::out)));
    PACK((llk_pack_dest_init()));
    PACK((llk_pack_reduce_mask_config<REDUCE_DIM, PackMode::Default>(dfb::out)));

    tilizeA_B_reduce_init<true /*neginf_srcA*/, false /*zero_srcA_reduce*/>(
        dfb::in_data, dfb::in_scaler, per_core_block_tile_cnt);

    dfb_in_scaler.wait_front(1);

    for (std::uint32_t b = 0; b < per_core_block_cnt; ++b) {
        dfb_in.wait_front(per_core_block_tile_cnt);
        dfb_out.reserve_back(per_core_block_tile_cnt);
        unpack_tilizeA_B_block<true /*neginf_srcA*/, true /*reload_srcB*/, false, false>(
            dfb::in_data,
            dfb::in_scaler,
            per_core_block_tile_cnt,
            0 /*tile idx for Src b is 0 because only 1 scaler tile is loaded*/);
        for (std::uint32_t i = 0; i < per_core_block_tile_cnt; ++i) {
            tile_regs_acquire();
            reduce_tile_math<REDUCE_OP, REDUCE_DIM>(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb::out);
            tile_regs_release();
        }
        dfb_out.push_back(per_core_block_tile_cnt);
        dfb_in.pop_front(per_core_block_tile_cnt);
    }

    reduce_uninit();
}
