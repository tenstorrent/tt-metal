// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TT_METAL_RISCV_DEBUG_INFO=lines: kernels compile and link with -g1, firmware keeps -g.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <utility>
#include <string>
#include <vector>

#include "jit_build/build.hpp"
#include "jit_build/jit_build_utils.hpp"
#include "jit_build/jit_device_config.hpp"
#include "llrt/hal.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal {
namespace {

// GCC uses the last of -g / -g1, so that one is the debug-info level the build gets.
std::string debug_info_level(const std::string& flags) {
    const auto tokens = jit_build::utils::tokenize_flags(flags);
    const auto last =
        std::find_if(tokens.rbegin(), tokens.rend(), [](const std::string& t) { return t == "-g" || t == "-g1"; });
    return last == tokens.rend() ? "" : *last;
}

TEST(RiscvDebugInfo, LinesValueSelectsKernelLineTables) {
    const char* previous = std::getenv("TT_METAL_RISCV_DEBUG_INFO");
    const std::string saved = previous ? previous : "";
    std::vector<std::string> compile_hashes;
    for (const auto& [value, lines_only] : {std::pair{"lines", true}, std::pair{"1", false}}) {
        setenv("TT_METAL_RISCV_DEBUG_INFO", value, 1);
        const llrt::RunTimeOptions options;
        EXPECT_TRUE(options.get_riscv_debug_info_enabled()) << value;
        EXPECT_EQ(options.get_riscv_kernel_debug_info_lines_only(), lines_only) << value;
        compile_hashes.push_back(options.get_compile_hash_string());
    }
    // Line-table-only kernels must not share a JIT build directory with full-debug kernels.
    EXPECT_NE(compile_hashes[0], compile_hashes[1]);
    if (previous) {
        setenv("TT_METAL_RISCV_DEBUG_INFO", saved.c_str(), 1);
    } else {
        unsetenv("TT_METAL_RISCV_DEBUG_INFO");
    }
}

TEST(RiscvDebugInfo, LinesOnlyAppliesToKernelsNotFirmware) {
    llrt::RunTimeOptions options;
    options.set_riscv_debug_info_enabled(true);
    const Hal hal(tt::ARCH::BLACKHOLE, false, false, 0, false);
    const JitDeviceConfig config{.hal = &hal, .arch = tt::ARCH::BLACKHOLE, .max_dfbs = 32};

    for (const bool lines_only : {false, true}) {
        options.set_riscv_kernel_debug_info_lines_only(lines_only);
        JitBuildEnv env;
        env.init(1, config, options, {});
        for (const bool is_fw : {false, true}) {
            const JitBuildState state(
                env,
                {.core_type = HalProgrammableCoreType::TENSIX,
                 .processor_class = HalProcessorClassType::DM,
                 .processor_id = 0,
                 .is_fw = is_fw},
                hal);
            const auto recipe = state.export_target_recipe(nullptr);
            const std::string expected = lines_only && !is_fw ? "-g1" : "-g";
            for (const auto& flags : {recipe.cflags, recipe.lflags}) {
                EXPECT_EQ(debug_info_level(flags), expected)
                    << "lines_only=" << lines_only << " is_fw=" << is_fw << ": " << flags;
            }
        }
    }
}

}  // namespace
}  // namespace tt::tt_metal
