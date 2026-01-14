// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"

// Tile is assumed to have 16-bit elements
template <uint32_t num_faces = 4, uint32_t num_cols_per_face = 16>
FORCE_INLINE void generate_reduce_scaler(const uint32_t cb_id, const uint16_t scaler) {
    cb_reserve_back(cb_id, 1);

    constexpr uint32_t num_zeros_reads = (num_faces * 512) / MEM_ZEROS_SIZE;
    static_assert(num_zeros_reads > 0, "num_zeros_reads must be greater than 0");
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t write_addr = get_write_ptr(cb_id);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(write_addr);

    // Fill tile with zeros
    // TODO: src addr does not need to be rewritten. Update/add api for this
    noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read_one_packet_with_state(zeros_noc_addr, write_addr);
        write_addr += MEM_ZEROS_SIZE;
    }
    noc_async_read_barrier();

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
