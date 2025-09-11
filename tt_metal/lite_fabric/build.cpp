
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/lite_fabric/build.hpp"

#include <filesystem>
#include <sstream>
#include <tt-logger/tt-logger.hpp>
#include <vector>
#include <string>
#include <fmt/format.h>
#include "tt_cluster.hpp"

// TODO: Cleanup this file
// https://github.com/tenstorrent/tt-metal/issues/28324

namespace {
std::string GetToolchainPath(const std::string& root_dir) {
    const std::array<std::string, 2> sfpi_roots = {
        root_dir + "/runtime/sfpi/compiler/bin/", "/opt/tenstorrent/sfpi/compiler/bin/"};

    for (unsigned i = 0; i < 2; ++i) {
        auto gxx = sfpi_roots[i];
        if (std::filesystem::exists(gxx)) {
            return gxx;
        }
    }
    TT_THROW("sfpi not found at {} or {}", sfpi_roots[0], sfpi_roots[1]);
}

std::string GetCommonOptions() {
    std::vector<std::string> options{
        "O3",
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

int CompileFabricLite(
    tt::Cluster& cluster,
    const std::filesystem::path& root_dir,
    const std::filesystem::path& out_dir,
    const std::vector<std::string>& extra_defines) {
    const std::filesystem::path lite_fabric_src = root_dir / "tt_metal/lite_fabric/hw/src/lite_fabric.cpp";

    std::vector<std::filesystem::path> includes = {
        root_dir,
        root_dir.parent_path(),
        root_dir / "ttnn",
        root_dir / "ttnn/cpp",
        root_dir / "tt_metal",
        root_dir / "tt_metal/include",
        root_dir / "tt_metal/hw/inc",
        root_dir / "tt_metal/hw/inc/ethernet",
        root_dir / "tt_metal/hostdevcommon/api",
        root_dir / "tt_metal/hw/inc/debug",
        root_dir / "tt_metal/hw/inc/blackhole",
        root_dir / "tt_metal/hw/inc/blackhole/blackhole_defines",
        root_dir / "tt_metal/hw/inc/blackhole/noc",
        root_dir / "tt_metal/hw/ckernels/blackhole/metal/common",
        root_dir / "tt_metal/hw/ckernels/blackhole/metal/llk_io",
        root_dir / "tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc",
        root_dir / "tt_metal/api/",
        root_dir / "tt_metal/api/tt-metalium/",
        root_dir / "tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib",
        root_dir / "tt_metal/lite_fabric/hw/inc"};  // For memory configuration headers

    std::vector<std::string> defines{
        "ARCH_BLACKHOLE",
        "TENSIX_FIRMWARE",
        "LOCAL_MEM_EN=0",
        "COMPILE_FOR_ERISC",  // This is needed to enable the ethernet APIs
        "ERISC",
        "RISC_B0_HW",
        "FW_BUILD",
        "NOC_INDEX=0",
        "DISPATCH_MESSAGE_ADDR=0",
        "COMPILE_FOR_LITE_FABRIC=1",
        "ROUTING_FW_ENABLED",
        // This is needed to get things to compile
        "NUM_DRAM_BANKS=1",
        "NUM_L1_BANKS=1",
        "LOG_BASE_2_OF_NUM_DRAM_BANKS=0",
        "LOG_BASE_2_OF_NUM_L1_BANKS=0",
        // We do not access the PCIe cores
        "PCIE_NOC_X=0",
        "PCIE_NOC_Y=0",
        // Lite Fabric is intended to run on risc1
        "PROCESSOR_INDEX=1",
    };

    defines.insert(defines.end(), extra_defines.begin(), extra_defines.end());

    std::ostringstream oss;
    oss << GetToolchainPath(root_dir.string()) << "riscv32-tt-elf-g++ ";
    oss << GetCommonOptions() << " ";

    for (const auto& include : includes) {
        oss << "-I" << include.string() << " ";
    }

    for (size_t i = 0; i < defines.size(); ++i) {
        oss << "-D" << defines[i] << " ";
    }

    oss << lite_fabric_src << " ";

    // Disable gp relaxations. We store various data in L1. We need functions to work when called directly from
    // the L1 address. The caller may have already setup their own gp, thus the gp relative instructions we have
    // are invalid
    oss << "-mno-relax -c -o " << out_dir / "lite_fabric.o";

    std::string compile_cmd = oss.str();
    log_info(tt::LogMetal, "compile:\n{}", compile_cmd);

    // Ensure the output dir exists
    std::filesystem::create_directories(out_dir);
    log_info(tt::LogMetal, "output directory: {}", out_dir);

    return system(compile_cmd.c_str());
}

int LinkFabricLite(
    const std::filesystem::path& root_dir, const std::filesystem::path& out_dir, const std::filesystem::path& elf_out) {
    // First, preprocess the linker script to resolve #include directives
    const std::filesystem::path tunneling_dir = root_dir / "tt_metal/lite_fabric/hw/inc";
    const std::filesystem::path input_ld = root_dir / "tt_metal/lite_fabric/toolchain/lite_fabric.ld";
    const std::filesystem::path output_ld = out_dir / "lite_fabric_preprocessed.ld";

    std::ostringstream preprocess_oss;
    preprocess_oss << GetToolchainPath(root_dir.string()) << "riscv32-tt-elf-g++ ";
    preprocess_oss << fmt::format("-I{} ", tunneling_dir.string());  // Include path for the memory defs header
    preprocess_oss << "-E -P -x c ";                                 // Preprocess only, no line markers, treat as C
    preprocess_oss << fmt::format("-o {} ", output_ld.string());
    preprocess_oss << input_ld;

    std::string preprocess_cmd = preprocess_oss.str();
    log_info(tt::LogMetal, "Preprocess linker script command:\n{}", preprocess_cmd);

    int preprocess_result = system(preprocess_cmd.c_str());
    if (preprocess_result != 0) {
        log_error(tt::LogMetal, "Failed to preprocess linker script");
        return preprocess_result;
    }

    // Now link using the preprocessed linker script
    std::ostringstream link_oss;
    link_oss << GetToolchainPath(root_dir.string()) << "riscv32-tt-elf-g++ ";
    link_oss << GetCommonOptions() << " ";
    link_oss << "-Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles ";
    link_oss << fmt::format("-T{} ", output_ld.string());  // Use preprocessed linker script
    link_oss << fmt::format("-Wl,-Map={}/lite_fabric.map ", out_dir.string());
    link_oss << fmt::format("-save-temps {}/lite_fabric.o ", out_dir.string());
    link_oss << fmt::format("{}/runtime/hw/lib/blackhole/tmu-crt0.o ", root_dir.string());     // FIXME: hardcoded path
    link_oss << fmt::format("{}/runtime/hw/lib/blackhole/substitutes.o ", root_dir.string());  // FIXME: hardcoded path
    link_oss << fmt::format("-o {}", elf_out.string());

    std::string link_cmd = link_oss.str();
    log_info(tt::LogMetal, "link:\n{}", link_cmd);

    int link_result = system(link_cmd.c_str());
    if (link_result != 0) {
        return link_result;
    }

    // Create flat binary from ELF for direct device loading
    std::filesystem::path bin_path = elf_out;
    bin_path.replace_extension(".bin");

    std::ostringstream objcopy_oss;
    objcopy_oss << GetToolchainPath(root_dir.string()) << "riscv32-tt-elf-objcopy -O binary ";
    objcopy_oss << elf_out << " " << bin_path;

    std::string objcopy_cmd = objcopy_oss.str();
    log_info(tt::LogMetal, "copy binary:\n{}", objcopy_cmd);

    int objcopy_result = system(objcopy_cmd.c_str());
    if (objcopy_result != 0) {
        log_error(tt::LogMetal, "Failed to create flat binary from ELF");
        return objcopy_result;
    }

    log_info(tt::LogMetal, "binary: {}", bin_path);
    return 0;
}

}  // namespace lite_fabric
