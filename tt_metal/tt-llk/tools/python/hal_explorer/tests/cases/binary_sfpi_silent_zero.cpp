// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sfpi.h"
// Binary: sfpload + sfpmul + sfpstore = 3 Tensix instructions.  Tool: count 0, no warning.
void sfpi_one() { sfpi::vFloat x = sfpi::dst_reg[0]; sfpi::dst_reg[1] = x * x; }
