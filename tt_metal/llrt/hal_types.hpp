// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/hal_types.hpp>

namespace tt::tt_metal {

static constexpr uint32_t NumHalProgrammableCoreTypes = static_cast<uint32_t>(HalProgrammableCoreType::COUNT);

}  // namespace tt::tt_metal
