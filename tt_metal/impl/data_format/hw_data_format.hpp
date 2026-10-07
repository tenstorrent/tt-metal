// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt::tt_metal {

using hw_format_t = std::underlying_type_t<DataFormat>;

// The HW DataFormat code device compilation needs for a host DataFormat (see hw_data_format.cpp).
hw_format_t host_data_format_to_hw(DataFormat f);

}  // namespace tt::tt_metal
