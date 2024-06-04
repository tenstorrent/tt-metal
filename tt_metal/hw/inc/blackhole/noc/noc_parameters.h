// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "third_party/umd/src/firmware/riscv/blackhole/noc/noc_parameters.h"

#ifdef _NOC_PARAMETERS_H_

#define PCIE_NOC_X 11
#define PCIE_NOC_Y 0

// Addres formats

#define NOC_XY_ENCODING(x, y) \
    ((((uint64_t)(y)) << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) | (((uint64_t)(x)) << NOC_ADDR_LOCAL_BITS))

#define NOC_XY_PCIE_ENCODING(x, y, noc_index) \
    NOC_XY_ENCODING(x, y)

#define NOC_MULTICAST_ENCODING(x_start, y_start, x_end, y_end)                      \
    ((((uint64_t)(x_start)) << (NOC_ADDR_LOCAL_BITS + 2 * NOC_ADDR_NODE_ID_BITS)) | \
     (((uint64_t)(y_start)) << (NOC_ADDR_LOCAL_BITS + 3 * NOC_ADDR_NODE_ID_BITS)) | \
     (((uint64_t)(x_end)) << NOC_ADDR_LOCAL_BITS) |                                 \
     (((uint64_t)(y_end)) << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)))

#define NOC_XY_COORD(x, y) ((((uint32_t)(y)) << NOC_ADDR_NODE_ID_BITS) | ((uint32_t)(x)))

// Alignment restrictions
#define NOC_L1_READ_ALIGNMENT_BYTES       16
#define NOC_L1_WRITE_ALIGNMENT_BYTES      16
#define NOC_PCIE_READ_ALIGNMENT_BYTES     32
#define NOC_PCIE_WRITE_ALIGNMENT_BYTES    32
#define NOC_DRAM_READ_ALIGNMENT_BYTES     64
#define NOC_DRAM_WRITE_ALIGNMENT_BYTES    16

#endif
