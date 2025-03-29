// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "system_utils.hpp"

#include <cxxabi.h>

#include <memory>
#include <string>

namespace ttml::core {
std::string demangle(const char* name) {
    int status = -4;

    std::unique_ptr<char, decltype(&free)> res(abi::__cxa_demangle(name, nullptr, nullptr, &status), &free);

    const char* const demangled_name = (status == 0) ? res.get() : name;

    std::string ret_val(demangled_name);

    return ret_val;
}
}  // namespace ttml::core
