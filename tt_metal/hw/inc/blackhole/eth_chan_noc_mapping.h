// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

uint16_t eth_chan_to_noc_xy[2][12] __attribute__((used)) = {
    {
        // noc=0
        (((25 << NOC_ADDR_NODE_ID_BITS) | 20) << NOC_COORD_REG_OFFSET),
        (((25 << NOC_ADDR_NODE_ID_BITS) | 21) << NOC_COORD_REG_OFFSET),
        (((25 << NOC_ADDR_NODE_ID_BITS) | 22) << NOC_COORD_REG_OFFSET),
        (((25 << NOC_ADDR_NODE_ID_BITS) | 23) << NOC_COORD_REG_OFFSET),
        (((25 << NOC_ADDR_NODE_ID_BITS) | 24) << NOC_COORD_REG_OFFSET),
        (((25 << NOC_ADDR_NODE_ID_BITS) | 25) << NOC_COORD_REG_OFFSET),
        (((25 << NOC_ADDR_NODE_ID_BITS) | 26) << NOC_COORD_REG_OFFSET),
        (((25 << NOC_ADDR_NODE_ID_BITS) | 27) << NOC_COORD_REG_OFFSET),
        (((25 << NOC_ADDR_NODE_ID_BITS) | 28) << NOC_COORD_REG_OFFSET),
        (((25 << NOC_ADDR_NODE_ID_BITS) | 29) << NOC_COORD_REG_OFFSET),
        (((25 << NOC_ADDR_NODE_ID_BITS) | 30) << NOC_COORD_REG_OFFSET),
        (((25 << NOC_ADDR_NODE_ID_BITS) | 31) << NOC_COORD_REG_OFFSET),
    },
    {
        // noc=1
        (((25 << NOC_ADDR_NODE_ID_BITS) | 20) << NOC_COORD_REG_OFFSET),
        (((25 << NOC_ADDR_NODE_ID_BITS) | 21) << NOC_COORD_REG_OFFSET),
        (((25 << NOC_ADDR_NODE_ID_BITS) | 22) << NOC_COORD_REG_OFFSET),
        (((25 << NOC_ADDR_NODE_ID_BITS) | 23) << NOC_COORD_REG_OFFSET),
        (((25 << NOC_ADDR_NODE_ID_BITS) | 24) << NOC_COORD_REG_OFFSET),
        (((25 << NOC_ADDR_NODE_ID_BITS) | 25) << NOC_COORD_REG_OFFSET),
        (((25 << NOC_ADDR_NODE_ID_BITS) | 26) << NOC_COORD_REG_OFFSET),
        (((25 << NOC_ADDR_NODE_ID_BITS) | 27) << NOC_COORD_REG_OFFSET),
        (((25 << NOC_ADDR_NODE_ID_BITS) | 28) << NOC_COORD_REG_OFFSET),
        (((25 << NOC_ADDR_NODE_ID_BITS) | 29) << NOC_COORD_REG_OFFSET),
        (((25 << NOC_ADDR_NODE_ID_BITS) | 30) << NOC_COORD_REG_OFFSET),
        (((25 << NOC_ADDR_NODE_ID_BITS) | 31) << NOC_COORD_REG_OFFSET),
    },
};
