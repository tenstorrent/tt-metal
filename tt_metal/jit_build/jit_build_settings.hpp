// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <functional>

namespace tt::tt_metal {

// Abstract base class for kernel specialization
// Higher levels of the SW derive from this and fill in build details not known to the build system
// (eg, API specified settings)
class JitBuildSettings {
public:
    // Returns the full kernel name
    virtual const std::string& get_full_kernel_name() const = 0;
    // Returns the compiler optimization level
    virtual std::string_view get_compiler_opt_level() const = 0;
    // Returns the linker optimization level
    virtual std::string_view get_linker_opt_level() const = 0;

    // Called to process the user defines
    virtual void process_defines(
        const std::function<void(const std::string& define, const std::string& value)>) const = 0;
    // Called to process the user compile time args
    virtual void process_compile_time_args(const std::function<void(const std::vector<uint32_t>& values)>) const = 0;

    virtual ~JitBuildSettings() = default;
};

}  // namespace tt::tt_metal
