// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// API
#include <tt-metalium/assert.hpp>
#include <tt-metalium/cluster.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/metal_soc_descriptor.h>
#include <tt-metalium/utils.hpp>

// umd::device
#include <umd/device/chip_helpers/tlb_manager.h>
#include <umd/device/device_api_metal.h>
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/tt_io.hpp>
#include <umd/device/tt_soc_descriptor.h>
#include <umd/device/tt_xy_pair.h>
#include <umd/device/types/arch.h>
#include <umd/device/types/cluster_descriptor_types.h>
#include <umd/device/types/xy_pair.h>

// Metalium::Metal::Hardware
