// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <filesystem>
#include <string>

#include "jit_build/build.hpp"
#include "jit_build/build_env_manager.hpp"
#include "jit_build/jit_device_config.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal {

namespace {

void copy_firmware_to_precompiled_dir(
    const std::string& firmware_out_path, const std::string& precompiled_firmware_dir) {
    namespace fs = std::filesystem;
    for (const auto& entry : fs::recursive_directory_iterator(firmware_out_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".elf") {
            auto relative = fs::relative(entry.path(), firmware_out_path);
            auto dest = fs::path(precompiled_firmware_dir) / relative;
            fs::create_directories(dest.parent_path());
            fs::copy_file(entry.path(), dest, fs::copy_options::overwrite_existing);
        }
    }
}

}  // namespace

void precompile_for_config(
    const tt::tt_metal::JitDeviceConfig& jit_device_config, const tt::llrt::RunTimeOptions& rtoptions) {
    BuildEnvManager build_env_manager(*jit_device_config.hal);
    build_env_manager.add_build_env(0, jit_device_config, rtoptions);
    build_env_manager.build_firmware(0, /*ignore_precompiled=*/true);

    auto dev_build_env = build_env_manager.get_device_build_env(0);
    auto build_key = dev_build_env.build_key();
    auto firmware_out_path = dev_build_env.build_env.get_out_firmware_root_path();
    auto precompiled_firmware_dir =
        rtoptions.get_root_dir() + "tt_metal/pre-compiled/" + std::to_string(build_key) + "/";

    copy_firmware_to_precompiled_dir(firmware_out_path, precompiled_firmware_dir);
}

}  // namespace tt::tt_metal

namespace {

struct PrecompileConfig {
    tt::ARCH arch;
    std::string core_descriptor_name;
    std::string soc_descriptor_name;
};

const auto supported_configs = std::to_array<PrecompileConfig>({
    {tt::ARCH::WORMHOLE_B0, "wormhole_b0_80_arch.yaml", "wormhole_b0_80_arch.yaml"},
    {tt::ARCH::WORMHOLE_B0, "wormhole_b0_80_arch_eth_dispatch.yaml", "wormhole_b0_80_arch.yaml"},
    {tt::ARCH::WORMHOLE_B0, "wormhole_b0_80_arch_fabric_mux.yaml", "wormhole_b0_80_arch.yaml"},
    {tt::ARCH::BLACKHOLE, "blackhole_140_arch.yaml", "blackhole_140_arch.yaml"},
    {tt::ARCH::BLACKHOLE, "blackhole_140_arch_eth_dispatch.yaml", "blackhole_140_arch.yaml"},
    {tt::ARCH::BLACKHOLE, "blackhole_140_arch_fabric_mux.yaml", "blackhole_140_arch.yaml"},
});

}  // namespace

int main() {
    tt::llrt::RunTimeOptions rtoptions;
    // Regenerate firmware into the JIT cache (out_firmware_root_), not an existing
    // pre-compiled bundle. Otherwise add_build_env() redirects firmware_binary_root_ to
    // the existing bundle, and since weaken() writes the weakened ELF to
    // firmware_binary_root_ while link() writes the .elf to out_firmware_root_, the
    // freshly-weakened ELF lands in the bundle but is then clobbered when
    // copy_firmware_to_precompiled_dir() copies the JIT cache's (never-refreshed,
    // stale) weakened ELF over it. Disabling precompiled FW keeps both in the JIT
    // cache so the .elf and its weakened ELF stay consistent before the copy.
    rtoptions.set_disable_precompiled_fw(true);
    const std::string core_descriptors_dir = rtoptions.get_root_dir() + "tt_metal/core_descriptors/";
    const std::string soc_descriptors_dir = rtoptions.get_root_dir() + "tt_metal/soc_descriptors/";
    for (const auto& [arch, core_descriptor_name, soc_descriptor_name] : supported_configs) {
        std::string full_core_descriptor_path = core_descriptors_dir + core_descriptor_name;
        std::string full_soc_descriptor_path = soc_descriptors_dir + soc_descriptor_name;
        tt::tt_metal::enumerate_jit_device_configs(
            arch,
            full_core_descriptor_path,
            full_soc_descriptor_path,
            [&rtoptions](const tt::tt_metal::JitDeviceConfig& jit_device_config) {
                tt::tt_metal::precompile_for_config(jit_device_config, rtoptions);
            });
    }
    return 0;
}
