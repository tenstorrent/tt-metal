/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "third_party/umd/src/firmware/riscv/wormhole/noc/noc_parameters.h"

#ifdef _NOC_PARAMETERS_H_

// Address formats
#define NOC_XY_ENCODING(x, y) \
   ((((uint32_t)(y)) << (NOC_ADDR_NODE_ID_BITS)) |  \
   (((uint32_t)(x))))

#define NOC_XY_ADDR2(xy, addr)                                         \
   ((((uint64_t)(xy)) << NOC_ADDR_LOCAL_BITS) |                        \
   ((uint64_t)(addr)))

#endif
