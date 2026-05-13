// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_pack_common_api.h"

/*************************************************************************
 * LLK PACK ROWS (Quasar)
 *
 * Row-scoped pack is not implemented on Quasar. These stubs match WH/BH entry point names so
 * compute headers can include llk_pack_rows_api.h on every architecture.
 *************************************************************************/

inline void llk_pack_rows_init(const std::uint32_t num_rows) { (void)num_rows; }

inline void llk_pack_rows(
    const std::uint32_t dst_index, const std::uint32_t output, const std::uint32_t output_index = 0) {
    (void)dst_index;
    (void)output;
    (void)output_index;
}

inline void llk_pack_rows_uninit() {}
