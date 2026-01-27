// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_common_api.h"
#include "llk_unpack_mul_reduce_scalar.h"

/*************************************************************************
 * LLK UNPACK MUL REDUCE SCALAR - Unpacker API for fused mul+reduce
 *************************************************************************/

/**
 * @brief Switch UNPACK state for mul_reduce_scalar reduce phase
 *
 * Prepares the unpacker to transition from multiply phase to reduce phase.
 * Resets counters and sets DVALID flags so the math thread can reuse
 * destination registers as source operands.
 */
inline void llk_unpack_mul_reduce_scalar_switch_to_reduce() { _llk_unpack_mul_reduce_scalar_switch_to_reduce_(); }
