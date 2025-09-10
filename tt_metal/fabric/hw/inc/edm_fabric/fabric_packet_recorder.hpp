// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

template <typename HEADER_TYPE>
struct PacketHeaderRecorder {
    volatile uint32_t* buffer_ptr;
    size_t buffer_n_headers;
    size_t buffer_index;

    PacketHeaderRecorder(volatile uint32_t* buffer_ptr, size_t buffer_n_headers) :
        buffer_ptr(buffer_ptr), buffer_n_headers(buffer_n_headers), buffer_index(0) {}

    void record_packet_header(volatile uint32_t* packet_header_ptr) {
        uint32_t dest_l1_addr = (uint32_t)buffer_ptr + buffer_index * sizeof(HEADER_TYPE);
        noc_async_write(
            (uint32_t)packet_header_ptr,
            get_noc_addr(my_x[0], my_y[0], dest_l1_addr),
            sizeof(HEADER_TYPE),
            1 - noc_index  // avoid the contention on main noc
        );
        buffer_index++;
        if (buffer_index == buffer_n_headers) {
            buffer_index = 0;
        }
    }
};
