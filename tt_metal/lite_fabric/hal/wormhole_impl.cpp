// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "wormhole_impl.hpp"

namespace lite_fabric {

void WormholeLiteFabricHal::set_reset_state(tt_cxy_pair /*virtual_core*/, bool /*assert_reset*/) {}

void WormholeLiteFabricHal::set_pc(tt_cxy_pair /*virtual_core*/, uint32_t /*pc_val*/) {}

tt::umd::tt_version WormholeLiteFabricHal::get_binary_version() {
    return tt::umd::tt_version{0, 0, 0};
}

void WormholeLiteFabricHal::launch(const std::filesystem::path& /*bin_path*/) {}

void WormholeLiteFabricHal::terminate() {}

void WormholeLiteFabricHal::wait_for_state(tt_cxy_pair /*virtual_core*/, lite_fabric::InitState /*state*/) {}

std::vector<std::filesystem::path> WormholeLiteFabricHal::build_includes(const std::filesystem::path& root_dir) {
    return {
        root_dir / "tt_metal/hw/inc/tt-1xx/wormhole",
        root_dir / "tt_metal/hw/inc/tt-1xx/wormhole/wormhole_b0_defines",
        root_dir / "tt_metal/hw/inc/tt-1xx/wormhole/noc",
        root_dir / "tt_metal/hw/ckernels/wormhole/metal/common",
        root_dir / "tt_metal/hw/ckernels/wormhole/metal/llk_io",
        root_dir / "tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc",
        root_dir / "tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib",
        root_dir / "tt_metal/lite_fabric/hw/inc",
        root_dir / "tt_metal/lite_fabric/hw/inc/wormhole",
    };
}

std::vector<std::string> WormholeLiteFabricHal::build_defines() {
    return {
        "ARCH_WORMHOLE",
        "TENSIX_FIRMWARE",
        "LOCAL_MEM_EN=0",
        "COMPILE_FOR_ERISC",
        "ERISC",
        "RISC_B0_HW",
        "FW_BUILD",
        "NOC_INDEX=0",
        "DISPATCH_MESSAGE_ADDR=0",
        "COMPILE_FOR_LITE_FABRIC=1",
        "ROUTING_FW_ENABLED",
        "NUM_DRAM_BANKS=1",
        "NUM_L1_BANKS=1",
        "LOG_BASE_2_OF_NUM_DRAM_BANKS=0",
        "LOG_BASE_2_OF_NUM_L1_BANKS=0",
        "PCIE_NOC_X=0",
        "PCIE_NOC_Y=0",
        "PROCESSOR_INDEX=0",
    };
}

std::vector<std::filesystem::path> WormholeLiteFabricHal::build_linker(const std::filesystem::path& root_dir) {
    return {
        root_dir / "runtime/hw/lib/blackhole/tmu-crt0.o",
        root_dir / "runtime/hw/lib/blackhole/substitutes.o",
    };
}

}  // namespace lite_fabric
