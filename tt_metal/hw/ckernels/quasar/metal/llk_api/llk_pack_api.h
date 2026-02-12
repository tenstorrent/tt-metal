// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "chlkc_list.h"
#include "internal/circular_buffer_interface.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_template.h"
#include "llk_defs.h"
#include "cpack_common.h"
#include "llk_io.h"
#include "llk_outputs.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "llk_param_structs.h"

/*************************************************************************
 * LLK PACK
 *************************************************************************/

// template <uint8_t PACK_SEL>
// inline void llk_pack_init(const std::uint32_t pack_output = 16) {
//     const std::uint32_t output_id = get_output_id(pack_output);

//     _llk_pack_init_<PACK_SEL>(output_id, 1 /*num_tiles_per_pack*/);
// }

// template <bool out_of_order_output, bool untilize>
// inline std::uint32_t get_output_tile_index(std::uint8_t output_id, std::uint32_t output_tile_index) {
//     std::uint32_t l1_tile_index;
//     if constexpr (out_of_order_output) {
//         // Use the write tile index to track position within CB
//         l1_tile_index = get_local_cb_interface(output_id).fifo_wr_tile_idx + output_tile_index;
//     } else {
//         if constexpr (untilize) {
//             // TODO: uplift this option from BBE
//         } else {
//             // In-order packing: use fifo_wr_tile_ptr as the incrementing tile offset
//             l1_tile_index =
//                 get_local_cb_interface(output_id).fifo_wr_tile_idx +
//                 get_local_cb_interface(output_id).fifo_wr_tile_ptr;
//             get_local_cb_interface(output_id).fifo_wr_tile_ptr++;
//         }
//     }
//     return l1_tile_index;
// }

// template <bool out_of_order_output = false, uint8_t PACK_SEL>
// inline void llk_pack(std::uint32_t tile_index, std::uint32_t output, std::uint32_t output_tile_index = 0) {
//     std::uint8_t output_id = get_output_id(output);

//     std::uint32_t l1_tile_index = get_output_tile_index<out_of_order_output, false>(output_id, output_tile_index);

//     _llk_pack_<PACK_SEL>(tile_index, l1_tile_index);
// }

// /*************************************************************************
//  * LLK PACK COMMON
//  *************************************************************************/

// template <uint32_t PACK_SEL>
// inline void llk_pack_hw_configure(const llk_pack_params_t* pack_params) {
//     const std::uint32_t output_id = get_output_id(pack_params->pack_output);
//     std::uint32_t base_addr =
//         get_local_cb_interface(output_id).fifo_limit - get_local_cb_interface(output_id).fifo_size;

//     buffer_descriptor_u bd_val = {0};
//     bd_val.f.l1_addr_16B = base_addr;
//     bd_val.f.format = static_cast<uint8_t>(pack_dst_format[output_id]);
//     bd_val.f.x_dim = get_output_face_r_dim(output_id);
//     bd_val.f.y_dim = 16;  // face_c_dim
//     bd_val.f.z_dim = get_output_num_faces(output_id);

//     tdma_descriptor_t td_val;

//     td_val.buf_desc = bd_val;
//     td_val.buf_desc_id = output_id;
//     td_val.reg_data_format = static_cast<uint8_t>(pack_src_format[output_id]);

//     set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>(
//         {dest_dvalid_client::FPU, dest_dvalid_client::PACK});  // incomplete
//     _llk_pack_hw_configure_<PACK_SEL>(td_val);
// }

// template <uint32_t PACK_SEL>
// inline void llk_pack_hw_configure_disaggregated(std::uint32_t pack_output) {
//     llk_pack_params_t llk_pack_params = {
//         .pack_output = pack_output,
//     };
//     llk_pack_hw_configure<PACK_SEL>(&llk_pack_params);
// }

// template <DstSync DST, bool IS_FP32_MATH_DEST_EN>
// inline void llk_pack_dest_dvalid_section_done() {
//     _llk_pack_dest_dvalid_section_done_<DST, IS_FP32_MATH_DEST_EN>();
// }
