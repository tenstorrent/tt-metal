// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "chlkc_list.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_defs.h"
#include "llk_io.h"
#include "llk_operands.h"
#include "llk_param_structs.h"
#include "experimental/llk_unpack_AB_reduce_custom_runtime.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

template <bool is_fp32_dest_acc_en = false>
inline void llk_unpack_AB_reduce_block_max_row_init_runtime(uint32_t block_ct_dim) {
    _llk_unpack_AB_reduce_block_max_row_init_runtime_<is_fp32_dest_acc_en>(block_ct_dim);
}

inline void llk_unpack_AB_reduce_block_max_row_runtime(
    const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t row_start_index) {
    std::uint32_t operandA_id = get_operand_id(operandA);
    std::uint32_t operandB_id = get_operand_id(operandB);
    std::uint32_t base_address_a = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;
    std::uint32_t offset_address_a = get_local_cb_interface(operandA_id).fifo_page_size * row_start_index;
    std::uint32_t address_a = base_address_a + offset_address_a;
    std::uint32_t base_address_b = get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;

    _llk_unpack_AB_reduce_block_max_row_(address_a, base_address_b);
}

inline void llk_unpack_AB_reduce_block_max_row_uninit_runtime() {
    _llk_unpack_AB_reduce_block_max_row_uninit_(FACE_R_DIM, FACE_R_DIM);
}
