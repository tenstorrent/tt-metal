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

#include "multi_device_fixture.hpp"

namespace tt::tt_metal {

using CompilerIncludePathsTest = GenericMeshDeviceFixture;

// Verifies that DataMovementConfig::compiler_include_paths is honored by the JIT build:
// the user-supplied directory is added as an `-I` flag, so a header placed there is
// resolvable from the kernel source.
TEST_F(CompilerIncludePathsTest, TensixCompilerIncludePathsAreHonoredByJitBuild) {
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

    auto mesh_device = get_mesh_device();
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

    fs::remove_all(include_dir);
}

// Relative paths must be resolved against the current working directory. Construct a
// path that exists relative to CWD and pass it through, verifying that the build
// succeeds because the resolver normalizes it against CWD.
TEST_F(CompilerIncludePathsTest, TensixCompilerIncludePathsResolveRelativeViaCwd) {
    namespace fs = std::filesystem;

    const fs::path absolute_include_dir =
        fs::temp_directory_path() / fs::path(
                                        "tt_metal_compiler_include_paths_relative_test_" +
                                        std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
    fs::create_directories(absolute_include_dir);
    {
        std::ofstream header(absolute_include_dir / "user_header.h");
        header << "#pragma once\n#define USER_HEADER_SENTINEL 0xC0DEu\n";
    }

    // Express the same directory as a relative path from CWD. The resolver should
    // normalize this against CWD and accept it.
    const fs::path relative_include_dir = fs::relative(absolute_include_dir, fs::current_path());
    ASSERT_TRUE(relative_include_dir.is_relative()) << "Test setup error: expected a relative path";

    const std::string kernel_src = R"(
#include "api/dataflow/dataflow_api.h"
#include "user_header.h"

void kernel_main() {
    volatile uint32_t tt_l1_ptr* l1_ptr = (volatile uint32_t tt_l1_ptr*)0x100000;
    *l1_ptr = USER_HEADER_SENTINEL;
}
)";

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_devices()[0];
    Program program = CreateProgram();

    DataMovementConfig config{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compiler_include_paths = {relative_include_dir},
    };

    EXPECT_NO_THROW({
        CreateKernelFromString(program, kernel_src, CoreCoord{0, 0}, config);
        detail::CompileProgram(device, program);
    });

    fs::remove_all(absolute_include_dir);
}

// Relative paths that don't resolve to an existing directory must fail-fast at kernel
// construction (rather than silently degrading at compile time).
TEST_F(CompilerIncludePathsTest, TensixCompilerIncludePathsUnresolvedRelativePathThrows) {
    Program program = CreateProgram();

    DataMovementConfig config{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compiler_include_paths = {"this_directory_does_not_exist_anywhere_xyz123"},
    };

    EXPECT_THROW(
        { CreateKernelFromString(program, "void kernel_main() {}", CoreCoord{0, 0}, config); }, std::exception);
}

}  // namespace tt::tt_metal
