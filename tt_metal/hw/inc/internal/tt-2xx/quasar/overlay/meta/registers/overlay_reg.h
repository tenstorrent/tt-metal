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
#include "bus_error_unit_0_reg.h"
#include "bus_error_unit_1_reg.h"
#include "bus_error_unit_2_reg.h"
#include "bus_error_unit_3_reg.h"
#include "bus_error_unit_4_reg.h"
#include "bus_error_unit_5_reg.h"
#include "bus_error_unit_6_reg.h"
#include "bus_error_unit_7_reg.h"
#include "edc_biu_map_reg.h"
#include "memory_port_cacheable_reg.h"
#include "memory_port_noncacheable_reg.h"
#include "smn_reg.h"
#include "tt_cache_controller_reg.h"
#include "tt_cluster_clint_reg.h"
#include "tt_cluster_core0_wdt_reg.h"
#include "tt_cluster_core1_wdt_reg.h"
#include "tt_cluster_core2_wdt_reg.h"
#include "tt_cluster_core3_wdt_reg.h"
#include "tt_cluster_core4_wdt_reg.h"
#include "tt_cluster_core5_wdt_reg.h"
#include "tt_cluster_core6_wdt_reg.h"
#include "tt_cluster_core7_wdt_reg.h"
#include "tt_cluster_ctrl_reg.h"
#include "tt_cluster_ctrl_t6_l1_csr_reg.h"
#include "tt_cluster_plic_reg.h"
#include "tt_debug_module_apb_reg.h"
#include "tt_debug_module_sbus_reg.h"
#include "tt_neo_awm_wrap_reg.h"
#include "tt_overlay_llk_tile_counters_reg.h"
#include "tt_rocc_accel_reg.h"
#include "tt_t6l1_slv_reg.h"

#endif // OVERLAY_REG_H
