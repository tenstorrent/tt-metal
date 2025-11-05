// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "chlkc_list.h"
#include "circular_buffer.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_template.h"
#include "cpack_common.h"
#include "llk_defs.h"
#include "llk_io.h"
#include "llk_outputs.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "llk_pack_reduce_custom.h"
#include "llk_pack_untilize.h"
#include "llk_param_structs.h"

/*************************************************************************
 * LLK PACK REDUCE CUSTOM - Specialized reduce_max_row operations
 *************************************************************************/

/**
 * Configures pack masking for specialized reduce_max_row operations.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole tile operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for native reduce pack configuration.
 * Use the standard llk_pack_reduce_mask_config for general-purpose reduction operations.
 */
inline void llk_pack_reduce_max_row_mask_config() { _llk_pack_reduce_max_row_mask_config_(); }
