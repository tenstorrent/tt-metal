/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once


struct CQReadInterface {
    uint32_t fifo_size;
    uint32_t fifo_limit; // range is inclusive of the limit
    uint32_t fifo_rd_ptr;
    uint32_t fifo_rd_toggle;
};

struct CBInterface {
    uint32_t fifo_size;
    uint32_t fifo_limit; // range is inclusive of the limit
    uint32_t fifo_page_size;
    uint32_t fifo_num_pages;

    uint32_t fifo_rd_ptr;
    uint32_t fifo_wr_ptr;

    // Save a cycle during init by writing 0 to the uint32 below
    union {
        uint32_t tiles_acked_received_init;
        struct {
            uint16_t tiles_acked;
            uint16_t tiles_received;
        };
    };

    // used by packer for in-order packing
    uint32_t fifo_wr_tile_ptr;
};
