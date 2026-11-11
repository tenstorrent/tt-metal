// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <unistd.h>

#include "jit_build/build_env_manager.hpp"
#include "jit_build/jit_device_config.hpp"
#include "llrt/hal.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal {
namespace {

// A Blackhole context without DRAM programmable cores (a mock, or silicon on firmware < 19.12) shares its
// firmware build key with one that has them. When the context without them builds first, the second context
// must still get a drisc ELF, and must leave the firmware the first context built untouched.
TEST(DramCoreFirmwareBuildDeathTest, BuildsDriscAfterContextWithoutDramCores) {
    namespace fs = std::filesystem;
    auto pattern = (fs::temp_directory_path() / "fw_build_dedup_XXXXXX").string();
    const char* directory = mkdtemp(pattern.data());
    ASSERT_NE(directory, nullptr);
    const fs::path root = directory;
    // The JIT cache directory and the process-wide build dedup state are confined to the child.
    const auto build_both = [&] {
        setenv("TT_METAL_CACHE", (root.string() + "/").c_str(), 1);
        llrt::RunTimeOptions rtoptions;
        // Otherwise a pre-compiled bundle for this build key becomes the firmware binary root: the checks below
        // would read the bundle's ELFs, and the build would write weakened ELFs into it.
        rtoptions.set_disable_precompiled_fw(true);
        bool built = false;
        enumerate_offline_compile_device_configs(rtoptions, [&](const JitDeviceConfig& with_dram_config) {
            if (built || with_dram_config.arch != tt::ARCH::BLACKHOLE) {
                return;
            }
            built = true;
            // The offline configs come with DRAM programmable cores on; match the rest of that HAL.
            const Hal without_dram_hal(
                tt::ARCH::BLACKHOLE,
                with_dram_config.routing_fw_enabled,
                /*enable_2_erisc_mode=*/true,
                with_dram_config.profiler_dram_bank_size_per_risc_bytes,
                /*enable_dram_backed_cq=*/false,
                /*is_simulator=*/false,
                /*enable_blackhole_dram_programmable_cores=*/false,
                /*enable_aerisc_ptp_trace=*/true);
            JitDeviceConfig without_dram_config = with_dram_config;
            without_dram_config.hal = &without_dram_hal;

            BuildEnvManager without_dram(without_dram_hal);
            without_dram.add_build_env(0, without_dram_config, rtoptions);
            BuildEnvManager with_dram(*with_dram_config.hal);
            with_dram.add_build_env(0, with_dram_config, rtoptions);
            if (without_dram.get_device_build_env(0).build_key() != with_dram.get_device_build_env(0).build_key()) {
                std::fputs("DRAM programmable cores changed the build key\n", stderr);
                std::_Exit(1);
            }

            without_dram.build_firmware(0, /*ignore_precompiled=*/true);
            const fs::path brisc = without_dram.get_firmware_binary_path(
                0,
                without_dram_hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX),
                /*processor_class=*/0,
                /*processor_id=*/0);
            const auto brisc_write_time = fs::last_write_time(brisc);

            with_dram.build_firmware(0, /*ignore_precompiled=*/true);
            const fs::path drisc = with_dram.get_firmware_binary_path(
                0,
                with_dram_config.hal->get_programmable_core_type_index(HalProgrammableCoreType::DRAM),
                /*processor_class=*/0,
                /*processor_id=*/0);
            if (!fs::exists(drisc)) {
                std::fprintf(stderr, "missing %s\n", drisc.c_str());
                std::_Exit(2);
            }
            if (fs::last_write_time(brisc) != brisc_write_time) {
                std::fprintf(stderr, "%s was rebuilt\n", brisc.c_str());
                std::_Exit(3);
            }
        });
        std::_Exit(built ? 0 : 4);
    };
    EXPECT_EXIT(build_both(), ::testing::ExitedWithCode(0), "");
    fs::remove_all(root);
}

}  // namespace
}  // namespace tt::tt_metal
