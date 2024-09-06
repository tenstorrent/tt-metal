// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "common/utils.hpp"

namespace tt {
namespace test_utils {
inline std::string get_env_arch_name() {
    constexpr std::string_view ARCH_NAME_ENV_VAR = "ARCH_NAME";
    std::string arch_name;

    if (const char* arch_name_ptr = std::getenv(ARCH_NAME_ENV_VAR.data())) {
        arch_name = arch_name_ptr;
    } else {
        TT_THROW("Env var {} is not set.", ARCH_NAME_ENV_VAR);
    }
    return arch_name;
}
}  // namespace test_utils
}  // namespace tt
