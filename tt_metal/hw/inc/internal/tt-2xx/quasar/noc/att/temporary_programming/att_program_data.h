// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file
 * @brief Selects the bring-up programming data matching the compile-time ATT
 * configuration. Production boot/UMD will own the same operation before DM
 * firmware starts.
 */
#if defined(NOC_ATT_CONFIG_QUASAR_AETHER_2X3)
#include "internal/tt-2xx/quasar/noc/att/temporary_programming/quasar_aether_2x3_att_data.h"
namespace active_att_program = quasar_aether_2x3_att_program;
#elif defined(NOC_ATT_CONFIG_GRENDEL_QSR1)
#include "internal/tt-2xx/quasar/noc/att/temporary_programming/grendel_qsr1_att_data.h"
namespace active_att_program = grendel_qsr1_att_program;
#else
#error "Temporary ATT programming requires an explicit NOC ATT configuration"
#endif
