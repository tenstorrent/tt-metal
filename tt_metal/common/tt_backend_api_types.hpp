// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <umd/device/types/arch.hpp>

#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt {

std::string get_string(ARCH arch);
std::string get_string_lowercase(ARCH arch);
std::string get_alias(ARCH arch);
ARCH get_arch_from_string(const std::string& arch_str);

bool is_fp8_format(DataFormat format);
bool is_data_format_supported(DataFormat format, ARCH arch);

}  // namespace tt
