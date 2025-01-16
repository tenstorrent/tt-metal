// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// TODO: WH only, need to move this to generated code path for BH support
uint16_t eth_chan_to_noc_xy[2][16] __attribute__((used)) = {
    {
        // noc=0
        (((16 << NOC_ADDR_NODE_ID_BITS) | 25) << NOC_COORD_REG_OFFSET),
        (((16 << NOC_ADDR_NODE_ID_BITS) | 18) << NOC_COORD_REG_OFFSET),
        (((16 << NOC_ADDR_NODE_ID_BITS) | 24) << NOC_COORD_REG_OFFSET),
        (((16 << NOC_ADDR_NODE_ID_BITS) | 19) << NOC_COORD_REG_OFFSET),
        (((16 << NOC_ADDR_NODE_ID_BITS) | 23) << NOC_COORD_REG_OFFSET),
        (((16 << NOC_ADDR_NODE_ID_BITS) | 20) << NOC_COORD_REG_OFFSET),
        (((16 << NOC_ADDR_NODE_ID_BITS) | 22) << NOC_COORD_REG_OFFSET),
        (((16 << NOC_ADDR_NODE_ID_BITS) | 21) << NOC_COORD_REG_OFFSET),
        (((17 << NOC_ADDR_NODE_ID_BITS) | 25) << NOC_COORD_REG_OFFSET),
        (((17 << NOC_ADDR_NODE_ID_BITS) | 18) << NOC_COORD_REG_OFFSET),
        (((17 << NOC_ADDR_NODE_ID_BITS) | 24) << NOC_COORD_REG_OFFSET),
        (((17 << NOC_ADDR_NODE_ID_BITS) | 19) << NOC_COORD_REG_OFFSET),
        (((17 << NOC_ADDR_NODE_ID_BITS) | 23) << NOC_COORD_REG_OFFSET),
        (((17 << NOC_ADDR_NODE_ID_BITS) | 20) << NOC_COORD_REG_OFFSET),
        (((17 << NOC_ADDR_NODE_ID_BITS) | 22) << NOC_COORD_REG_OFFSET),
        (((17 << NOC_ADDR_NODE_ID_BITS) | 21) << NOC_COORD_REG_OFFSET),
    },
    {
        // noc=1
        (((16 << NOC_ADDR_NODE_ID_BITS) | 25) << NOC_COORD_REG_OFFSET),
        (((16 << NOC_ADDR_NODE_ID_BITS) | 18) << NOC_COORD_REG_OFFSET),
        (((16 << NOC_ADDR_NODE_ID_BITS) | 24) << NOC_COORD_REG_OFFSET),
        (((16 << NOC_ADDR_NODE_ID_BITS) | 19) << NOC_COORD_REG_OFFSET),
        (((16 << NOC_ADDR_NODE_ID_BITS) | 23) << NOC_COORD_REG_OFFSET),
        (((16 << NOC_ADDR_NODE_ID_BITS) | 20) << NOC_COORD_REG_OFFSET),
        (((16 << NOC_ADDR_NODE_ID_BITS) | 22) << NOC_COORD_REG_OFFSET),
        (((16 << NOC_ADDR_NODE_ID_BITS) | 21) << NOC_COORD_REG_OFFSET),
        (((17 << NOC_ADDR_NODE_ID_BITS) | 25) << NOC_COORD_REG_OFFSET),
        (((17 << NOC_ADDR_NODE_ID_BITS) | 18) << NOC_COORD_REG_OFFSET),
        (((17 << NOC_ADDR_NODE_ID_BITS) | 24) << NOC_COORD_REG_OFFSET),
        (((17 << NOC_ADDR_NODE_ID_BITS) | 19) << NOC_COORD_REG_OFFSET),
        (((17 << NOC_ADDR_NODE_ID_BITS) | 23) << NOC_COORD_REG_OFFSET),
        (((17 << NOC_ADDR_NODE_ID_BITS) | 20) << NOC_COORD_REG_OFFSET),
        (((17 << NOC_ADDR_NODE_ID_BITS) | 22) << NOC_COORD_REG_OFFSET),
        (((17 << NOC_ADDR_NODE_ID_BITS) | 21) << NOC_COORD_REG_OFFSET),
    },
};
