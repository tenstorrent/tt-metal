
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

int CompileLiteFabric(
    tt::Cluster& cluster,
    const std::filesystem::path& root_dir,
    const std::filesystem::path& out_dir,
    const std::vector<std::string>& extra_defines) {
    const auto root_dir_str = root_dir.string();
    const auto out_dir_str = out_dir.string();

    const std::string lite_fabric_src =
        fmt::format("{}/tests/tt_metal/tt_metal/tunneling/lite_fabric.cpp", root_dir_str);

    std::vector<std::string> includes = {
        ".",
        "..",
        root_dir_str,
        root_dir_str + "/ttnn",
        root_dir_str + "/ttnn/cpp",
        root_dir_str + "/tt_metal",
        root_dir_str + "/tt_metal/include",
        root_dir_str + "/tt_metal/hw/inc",
        root_dir_str + "/tt_metal/hw/inc/ethernet",
        root_dir_str + "/tt_metal/hostdevcommon/api",
        root_dir_str + "/tt_metal/hw/inc/debug",
        root_dir_str + "/tt_metal/hw/inc/blackhole",
        root_dir_str + "/tt_metal/hw/inc/blackhole/blackhole_defines",
        root_dir_str + "/tt_metal/hw/inc/blackhole/noc",
        root_dir_str + "/tt_metal/hw/ckernels/blackhole/metal/common",
        root_dir_str + "/tt_metal/hw/ckernels/blackhole/metal/llk_io",
        root_dir_str + "/tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc",
        root_dir_str + "/tt_metal/api/",
        root_dir_str + "/tt_metal/api/tt-metalium/",
        root_dir_str + "/tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib"};

    // TODO remove hardcoded defines
    std::vector<std::string> defines{
        "ARCH_BLACKHOLE",
        "IS_NOT_POW2_NUM_L1_BANKS=1",
        "LOG_BASE_2_OF_NUM_DRAM_BANKS=3",
        "NUM_DRAM_BANKS=8 -DNUM_L1_BANKS=130",
        "TENSIX_FIRMWARE",
        "LOCAL_MEM_EN=0",
        "COMPILE_FOR_ERISC",  // This is needed to enable the ethernet APIs
        "ERISC",
        "RISC_B0_HW",
        "FW_BUILD",
        "NOC_INDEX=0",
        "DISPATCH_MESSAGE_ADDR=0",
        "COMPILE_FOR_LITE_FABRIC=1",
        // Fabric
        fmt::format("ROUTING_MODE={}", ROUTING_MODE_1D | ROUTING_MODE_LOW_LATENCY),
    };

    // This assumes both chips are the same
    auto soc_d = cluster.get_soc_desc(0);
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

    oss << "-c -o " << out_dir_str << "/lite_fabric.o";

    std::string compile_cmd = oss.str();
    log_info(tt::LogMetal, "Compile LiteFabric command:\n{}", compile_cmd);

    // Ensure the output dir exists
    std::filesystem::create_directories(out_dir);
    log_info(tt::LogMetal, "Compile LiteFabric output directory: {}", out_dir);

    return system(compile_cmd.c_str());
}

int LinkLiteFabric(
    const std::filesystem::path& root_dir, const std::filesystem::path& out_dir, const std::filesystem::path& elf_out) {
    std::ostringstream oss;
    oss << "riscv32-tt-elf-g++ ";
    oss << GetCommonOptions() << " ";
    oss << "-Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles ";
    oss << fmt::format("-T{}/runtime/hw/toolchain/blackhole/firmware_lite_fabric.ld ", root_dir.string());
    oss << fmt::format("-Wl,-Map={}/lite_fabric.map ", out_dir.string());
    oss << fmt::format("-save-temps {}/lite_fabric.o ", out_dir.string());
    oss << fmt::format("{}/runtime/hw/lib/blackhole/tmu-crt0.o ", root_dir.string());
    oss << fmt::format("{}/runtime/hw/lib/blackhole/substitutes.o ", root_dir.string());
    oss << fmt::format("-o {}", elf_out.string());

    std::string link_cmd = oss.str();
    log_info(tt::LogMetal, "Link LiteFabric command:\n{}", link_cmd);

    return system(link_cmd.c_str());
}

}  // namespace lite_fabric
