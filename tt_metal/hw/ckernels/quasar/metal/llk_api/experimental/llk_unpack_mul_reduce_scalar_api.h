// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "experimental/llk_unpack_mul_reduce_scalar.h"

/**
 * @brief Give math a zeroed SrcA and SrcB bank for the reduce tail.
 */
inline void llk_unpack_mul_reduce_scalar_switch_to_reduce() { _llk_unpack_mul_reduce_scalar_switch_to_reduce_(); }
