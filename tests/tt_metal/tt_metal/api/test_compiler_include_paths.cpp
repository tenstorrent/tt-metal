// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "device_fixture.hpp"

namespace tt::tt_metal {

// Verifies that DataMovementConfig::compiler_include_paths is honored by the JIT build:
// the user-supplied directory is added as an `-I` flag, so a header placed there is
// resolvable from the kernel source.
TEST_F(MeshDeviceFixture, TensixCompilerIncludePathsAreHonoredByJitBuild) {
    namespace fs = std::filesystem;

    // Stage a unique temp dir with a header that defines a sentinel macro.
    const fs::path include_dir =
        fs::temp_directory_path() / fs::path(
                                        "tt_metal_compiler_include_paths_test_" +
                                        std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
    fs::create_directories(include_dir);
    const fs::path header_path = include_dir / "user_header.h";
    {
        std::ofstream header(header_path);
        header << "#pragma once\n#define USER_HEADER_SENTINEL 0xC0DEu\n";
    }

    // Inline kernel source that only compiles if user_header.h is found via -I.
    // Read the sentinel into L1 to give the optimizer a reason to keep it.
    const std::string kernel_src = R"(
#include "api/dataflow/dataflow_api.h"
#include "user_header.h"

void kernel_main() {
    volatile uint32_t tt_l1_ptr* l1_ptr = (volatile uint32_t tt_l1_ptr*)0x100000;
    *l1_ptr = USER_HEADER_SENTINEL;
}
)";

    for (const auto& mesh_device : this->devices_) {
        auto* device = mesh_device->get_devices()[0];
        Program program = CreateProgram();

        DataMovementConfig config{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compiler_include_paths = {include_dir},
        };

        EXPECT_NO_THROW({
            CreateKernelFromString(program, kernel_src, CoreCoord{0, 0}, config);
            detail::CompileProgram(device, program);
        });
    }

    fs::remove_all(include_dir);
}

}  // namespace tt::tt_metal
