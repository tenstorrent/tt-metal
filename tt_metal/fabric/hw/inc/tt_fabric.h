// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "risc_attribs.h"
#include <hostdevcommon/common_values.hpp>
#include "dataflow_api.h"
#include "noc_overlay_parameters.h"
#include "ethernet/dataflow_api.h"
#include "eth_chan_noc_mapping.h"
#include "hostdevcommon/fabric_common.h"
#include "tt_metal/hw/inc/risc_common.h"

using namespace tt::tt_fabric;

#ifndef DISABLE_LOW_LATENCY_ROUTING
#ifndef LOW_LATENCY_ROUTING
#define LOW_LATENCY_ROUTING
#endif
#endif
