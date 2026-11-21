// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef X280_H
#define X280_H

#include "x280_address_map.h"
#include "x280_hart_protocol.h"

#include <stdint.h>

/* Per-hart stack size (must match entry.S and the linker scripts) */
#define HART_STACK_SIZE 0x8000UL /* 32 KiB */

#endif /* X280_H */
