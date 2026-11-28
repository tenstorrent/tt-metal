// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_sfpu_common.h"

/**
 * @brief Configures SFPU
 *
 * @tparam EN_32BIT_DEST_FORMAT: Set to true to use 32bit math dest in Float32 or Int32 format
 */
template <bool EN_32BIT_DEST_FORMAT>
inline void llk_sfpu_hw_configure() {
    _llk_sfpu_hw_configure_<EN_32BIT_DEST_FORMAT>();
}
