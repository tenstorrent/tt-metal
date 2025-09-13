// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "std_pch.hpp"

// nlohmann_json::nlohmann_json
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

// enchantum::enchantum
#include <enchantum/enchantum.hpp>

// fmt::fmt-header-only
#include <fmt/base.h>

// TT::STL
#include <tt_stl/overloaded.hpp>
#include <tt_stl/reflection.hpp>
#include <tt_stl/small_vector.hpp>
#include <tt_stl/span.hpp>

// umd::Firmware

// umd::device
#include <umd/device/types/arch.h>
#include "umd/device/types/cluster_descriptor_types.h"
#include <umd/device/types/xy_pair.h>

// simde::simde
#include <simde/x86/avx2.h>

// Taskflow::Taskflow
#include <taskflow/taskflow.hpp>
