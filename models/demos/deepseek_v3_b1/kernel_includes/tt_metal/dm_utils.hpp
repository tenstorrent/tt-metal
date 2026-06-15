// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"

// Tile is assumed to have 16-bit elements
template <uint32_t num_faces = 4, uint32_t num_cols_per_face = 16>
FORCE_INLINE void generate_reduce_scaler(const uint32_t cb_id, const uint16_t scaler) {
    cb_reserve_back(cb_id, 1);

    uint32_t write_addr = get_write_ptr(cb_id);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(write_addr);

    // Fill tile with zeros
    Noc noc;
    CircularBuffer cb(cb_id);
    noc.async_write_zeros(cb, num_faces * 512);
    noc.write_zeros_l1_barrier();

    if (scaler != 0) {
        for (uint32_t k = 0; k < num_faces; ++k) {
            uint32_t idx = k << 8;
            for (uint32_t j = 0; j < num_cols_per_face; ++j) {
                ptr[idx + j] = scaler;
            }
        }
    }

    cb_push_back(cb_id, 1);
}
