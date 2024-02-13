// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>

inline void reader_unary_sharded(const uint32_t num_units, const uint32_t cb_id_in0) {
    cb_reserve_back(cb_id_in0, num_units);
    cb_push_back(cb_id_in0, num_units);
}
