// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt_stl/strong_type.hpp>

namespace tt::tt_metal {

using SubDeviceId = tt::stl::StrongType<uint8_t, struct SubDeviceIdTag>;
using SubDeviceManagerId = tt::stl::StrongType<uint64_t, struct SubDeviceManagerIdTag>;

}  // namespace tt::tt_metal
