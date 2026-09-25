// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
// Version: FFN1.3.0

#ifndef OVERLAY_REG_H
#define OVERLAY_REG_H

#include <stdint.h>

// Overlay register map for QUASAR configuration
// Auto-generated file - do not edit manually

#define OVERLAY_REG_MAP_BASE_ADDR  (0x00000000)
#define OVERLAY_REG_MAP_SIZE       (0x08207008)

// Register header includes
#include "memory_port_cacheable_reg.h"
#include "memory_port_noncacheable_reg.h"
#include "tt_cache_controller_reg.h"
#include "tt_cluster_clint_reg.h"
#include "tt_cluster_ctrl_reg.h"
#include "tt_cluster_ctrl_t6_l1_csr_reg.h"
#include "tt_overlay_llk_tile_counters_reg.h"
#include "tt_rocc_accel_reg.h"

#endif // OVERLAY_REG_H
