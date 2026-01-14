// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <umd/device/types/arch.hpp>

namespace tt {

std::string get_string(ARCH arch);
std::string get_string_lowercase(ARCH arch);
std::string get_alias(ARCH arch);
ARCH get_arch_from_string(const std::string& arch_str);

}  // namespace tt
