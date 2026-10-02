// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file
 * @brief Selects the active ATT map configuration at compile time. Exactly one
 * NOC_ATT_CONFIG_* define must be set (the HAL emits it from TT_METAL_NOC_ATT);
 * the selected configuration provides the noc_att_active_config namespace
 * alias, the ACTIVE_ATT_MAP reference, and the NOC_ATT_LOCAL_WINDOW_BASE macro
 * the NOC V3 API requires for implicit local operands.
 */

#if defined(NOC_ATT_CONFIG_GRENDEL_QSR1) && defined(NOC_ATT_CONFIG_QUASAR_AETHER_2X3)
#error "Exactly one ATT configuration may be selected"
#elif defined(NOC_ATT_CONFIG_GRENDEL_QSR1)
#include "internal/tt-2xx/quasar/noc/att/configs/grendel_qsr1_att_config.h"
namespace noc_att_active_config = grendel_qsr1_att_config;
#define NOC_ATT_LOCAL_WINDOW_BASE 0x1800000000ULL
#elif defined(NOC_ATT_CONFIG_QUASAR_AETHER_2X3)
#include "internal/tt-2xx/quasar/noc/att/configs/quasar_aether_2x3_att_config.h"
namespace noc_att_active_config = quasar_aether_2x3_att_config;
#define NOC_ATT_LOCAL_WINDOW_BASE 0x1800000000ULL
#else
#error "The ATT address backend requires an explicit configuration (NOC_ATT_CONFIG_*, from TT_METAL_NOC_ATT)"
#endif

// The macro is a preprocess-time restatement of the configuration's typed
// constant (the V3 header #errors without it); they cannot drift apart.
static_assert(noc_att_active_config::LOCAL_WINDOW_BASE == NOC_ATT_LOCAL_WINDOW_BASE);

inline constexpr const noc_att::MapData& ACTIVE_ATT_MAP = noc_att_active_config::MAP;
