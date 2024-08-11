// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "third_party/umd/src/firmware/riscv/blackhole/noc/noc_parameters.h"

#ifdef _NOC_PARAMETERS_H_

#define PCIE_NOC_X 11
#define PCIE_NOC_Y 0

#define PCIE_NOC1_X 5
#define PCIE_NOC1_Y 11

// BH has 64 bit address space but pipegen was not updated to support this so WH scheme of encoding addresses is used (36 bits of address followed by coordinates)
// This means that lo and mid registers need to have the address portion while the coordinates go into hi register
#define NOC_COORD_REG_OFFSET 0 // offset (from LSB) in register holding x-y coordinate

// Addres formats

#define NOC_XY_ENCODING(x, y) ((((uint32_t)(y)) << (NOC_ADDR_NODE_ID_BITS)) | (((uint32_t)(x))))

// Base address pulled from tt_SiliconDevice::get_pcie_base_addr_from_device
#define NOC_XY_PCIE_ENCODING(x, y, noc_index)                                        \
   ((uint64_t(NOC_XY_ENCODING(x, y)) << (NOC_ADDR_LOCAL_BITS - NOC_COORD_REG_OFFSET))) |  \
   ((noc_index ? (x == PCIE_NOC1_X and y == PCIE_NOC1_Y) : (x == PCIE_NOC_X and y == PCIE_NOC_Y)) * 0x1000000000000000) \

#define NOC_MULTICAST_ENCODING(x_start, y_start, x_end, y_end)                      \
    ((((uint32_t)(x_start)) << (2 * NOC_ADDR_NODE_ID_BITS)) | \
     (((uint32_t)(y_start)) << (3 * NOC_ADDR_NODE_ID_BITS)) | \
     (((uint32_t)(x_end))) |                                 \
     (((uint32_t)(y_end)) << (NOC_ADDR_NODE_ID_BITS)))

// Because BH uses WH style address encoding (36 bits followed by coordinates) but PCIe transactions require bit 60 to be set, we need to mask out the xy-coordinate
// When NOC_ADDR_LOCAL_BITS is 64 then NOC_LOCAL_ADDR_OFFSET can be used and the below define can be deprecated
#define NOC_LOCAL_ADDR(addr) ((addr) & 0x1000000FFFFFFFFF)

// Alignment restrictions
#define NOC_L1_READ_ALIGNMENT_BYTES       16
#define NOC_L1_WRITE_ALIGNMENT_BYTES      16
#define NOC_PCIE_READ_ALIGNMENT_BYTES     64
#define NOC_PCIE_WRITE_ALIGNMENT_BYTES    16
#define NOC_DRAM_READ_ALIGNMENT_BYTES     64
#define NOC_DRAM_WRITE_ALIGNMENT_BYTES    16

#define L1_ALIGNMENT (static_cast<uint32_t>(NOC_L1_READ_ALIGNMENT_BYTES >= NOC_L1_WRITE_ALIGNMENT_BYTES ? NOC_L1_READ_ALIGNMENT_BYTES : NOC_L1_WRITE_ALIGNMENT_BYTES))

#endif
