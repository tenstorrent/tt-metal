// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt_stl/strong_type.hpp>

namespace tt::tt_metal {

using SubDeviceId = ttsl::StrongType<uint8_t, struct SubDeviceIdTag>;
using SubDeviceManagerId = ttsl::StrongType<uint64_t, struct SubDeviceManagerIdTag>;

}  // namespace tt::tt_metal
