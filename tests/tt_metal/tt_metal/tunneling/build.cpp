
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "build.hpp"

#include <filesystem>
#include <tt-logger/tt-logger.hpp>
#include <vector>
#include <string>
#include <fmt/format.h>
#include "tt_cluster.hpp"
#include "tt_metal/fabric/hw/inc/fabric_routing_mode.h"

namespace {
std::string GetCommonOptions() {
    std::vector<std::string> options{
        "Os",
        "mcpu=tt-bh",
        "fno-rvtt-sfpu-replay",
        "std=c++17",
        "flto=auto",
        "ffast-math",
        "g",
        "fno-use-cxa-atexit",
        "fno-exceptions",
        "Wall",
        "Werror",
        "Wno-unknown-pragmas",
        "Wno-deprecated-declarations",
        "Wno-error=multistatement-macros",
        "Wno-error=parentheses",
        "Wno-error=unused-but-set-variable",
        "Wno-unused-variable",
        "Wno-unused-function",
        "fno-tree-loop-distribute-patterns",
    };

    std::ostringstream oss;

    for (size_t i = 0; i < options.size(); ++i) {
        oss << "-" << options[i] << " ";
    }

    return oss.str();
}
}  // namespace

namespace lite_fabric {

void CompileLiteFabric(
    std::shared_ptr<tt::Cluster> cluster,
    const std::string& root_dir,
    const std::string& out_dir,
    const std::vector<std::string>& extra_defines) {
    const std::string lite_fabric_src = fmt::format("{}/tests/tt_metal/tt_metal/tunneling/lite_fabric.cpp", root_dir);

    std::vector<std::string> includes = {
        ".",
        "..",
        root_dir,
        root_dir + "/ttnn",
        root_dir + "/ttnn/cpp",
        root_dir + "/tt_metal",
        root_dir + "/tt_metal/include",
        root_dir + "/tt_metal/hw/inc",
        root_dir + "/tt_metal/hw/inc/ethernet",
        root_dir + "/tt_metal/hostdevcommon/api",
        root_dir + "/tt_metal/hw/inc/debug",
        root_dir + "/tt_metal/hw/inc/blackhole",
        root_dir + "/tt_metal/hw/inc/blackhole/blackhole_defines",
        root_dir + "/tt_metal/hw/inc/blackhole/noc",
        root_dir + "/tt_metal/hw/ckernels/blackhole/metal/common",
        root_dir + "/tt_metal/hw/ckernels/blackhole/metal/llk_io",
        root_dir + "/tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc",
        root_dir + "/tt_metal/api/",
        root_dir + "/tt_metal/api/tt-metalium/",
        root_dir + "/tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib"};

    std::vector<std::string> defines{
        "ARCH_BLACKHOLE",
        "IS_NOT_POW2_NUM_L1_BANKS=1",
        "LOG_BASE_2_OF_NUM_DRAM_BANKS=3",
        "NUM_DRAM_BANKS=8 -DNUM_L1_BANKS=130",
        "TENSIX_FIRMWARE",
        "LOCAL_MEM_EN=0",
        "COMPILE_FOR_ERISC",
        "ERISC",
        "RISC_B0_HW",
        "FW_BUILD",
        "NOC_INDEX=0",
        "DISPATCH_MESSAGE_ADDR=0",
        // Fabric
        fmt::format("ROUTING_MODE={}", ROUTING_MODE_1D),
    };

    auto soc_d = cluster->get_soc_desc(0);
    auto pcie_cores = soc_d.get_cores(CoreType::PCIE, CoordSystem::TRANSLATED);
    CoreCoord pcie_core = pcie_cores.empty() ? soc_d.grid_size : pcie_cores[0];

    defines.push_back(fmt::format("PCIE_NOC_X={}", std::to_string(pcie_core.x)));
    defines.push_back(fmt::format("PCIE_NOC_Y={}", std::to_string(pcie_core.y)));

    defines.insert(defines.end(), extra_defines.begin(), extra_defines.end());

    std::ostringstream oss;
    oss << "riscv32-tt-elf-g++ ";
    oss << GetCommonOptions() << " ";

    for (size_t i = 0; i < includes.size(); ++i) {
        oss << "-I" << includes[i] << " ";
    }

    for (size_t i = 0; i < defines.size(); ++i) {
        oss << "-D" << defines[i] << " ";
    }

    oss << lite_fabric_src << " ";

    oss << "-c -o lite_fabric/lite_fabric.o";

    std::string compile_cmd = oss.str();
    log_info(tt::LogMetal, "Compile LiteFabric command:\n{}", compile_cmd);

    // Ensure the output dir exists
    std::filesystem::create_directories(out_dir);
    log_info(tt::LogMetal, "Compile LiteFabric output directory: {}", out_dir);

    system(compile_cmd.c_str());
}

void LinkLiteFabric(const std::string& root_dir, const std::string& out_dir) {
    std::ostringstream oss;
    oss << "riscv32-tt-elf-g++ ";
    oss << GetCommonOptions() << " ";
    oss << "-Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles ";
    oss << fmt::format("-T{}/runtime/hw/toolchain/blackhole/firmware_aerisc.ld ", root_dir);
    oss << fmt::format("-Wl,-Map={}/lite_fabric.map ", out_dir);
    oss << fmt::format("-save-temps {}/lite_fabric.o ", out_dir);
    oss << fmt::format("{}/runtime/hw/lib/blackhole/tmu-crt0.o ", root_dir);
    oss << fmt::format("{}/runtime/hw/lib/blackhole/substitutes.o ", root_dir);
    oss << fmt::format("-o {}/lite_fabric.elf ", out_dir);

    std::string link_cmd = oss.str();
    log_info(tt::LogMetal, "Link LiteFabric command:\n{}", link_cmd);

    system(link_cmd.c_str());
}

}  // namespace lite_fabric
