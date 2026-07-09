// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <enchantum/enchantum.hpp>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "jit_build/build.hpp"
#include "jit_build/build_env_manager.hpp"
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

// Two kernels identical in every respect except their compiler_include_paths must
// produce different compute_hash() values. This ensures they don't collide on the
// same kernel-binary cache slot — different `-I` lists can resolve different headers
// at the same `#include` line, so the resulting binaries can semantically differ.
TEST_F(CompilerIncludePathsTest, TensixCompilerIncludePathsAffectKernelHash) {
    // Absolute paths that don't need to exist (resolver passes them through unmodified).
    Program program_a = CreateProgram();
    DataMovementConfig config_a{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compiler_include_paths = {"/tmp/compiler_include_paths_test_path_a"},
    };
    auto handle_a = CreateKernelFromString(program_a, "void kernel_main() {}", CoreCoord{0, 0}, config_a);

    Program program_b = CreateProgram();
    DataMovementConfig config_b{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compiler_include_paths = {"/tmp/compiler_include_paths_test_path_b"},
    };
    auto handle_b = CreateKernelFromString(program_b, "void kernel_main() {}", CoreCoord{0, 0}, config_b);

    auto kernel_a = program_a.impl().get_kernel(handle_a);
    auto kernel_b = program_b.impl().get_kernel(handle_b);

    EXPECT_NE(kernel_a->compute_hash(), kernel_b->compute_hash())
        << "Kernels with different compiler_include_paths must produce different compute_hash() values";
}

// Editing a header that's reachable via compiler_include_paths must trigger a rebuild
// on the next compile, even though the kernel source and the include-path list are
// unchanged. This proves the two cache layers cooperate correctly:
//   - compute_hash() depends on the path *list*, not header contents — so the same
//     source + same path list maps to the same kernel cache slot
//   - the .dephash mechanism (gcc -MF) records every header gcc opened with content
//     hashes, so editing a header invalidates the slot and triggers a rebuild
// If either layer were broken, this test would fail in a specific, diagnostic way.
TEST_F(CompilerIncludePathsTest, TensixHeaderEditTriggersRebuild) {
    namespace fs = std::filesystem;

    const fs::path include_dir =
        fs::temp_directory_path() / fs::path(
                                        "tt_metal_compiler_include_paths_rebuild_test_" +
                                        std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
    fs::create_directories(include_dir);
    const fs::path header_path = include_dir / "user_header.h";

    auto write_header = [&](uint32_t value) {
        std::ofstream header(header_path);
        header << "#pragma once\n#define USER_HEADER_SENTINEL 0x" << std::hex << value << "u\n";
    };

    auto read_file_bytes = [](const fs::path& p) {
        std::ifstream f(p, std::ios::binary);
        return std::vector<char>{std::istreambuf_iterator<char>(f), {}};
    };

    // Volatile L1 store keeps the constant materialized in the binary, so different
    // sentinel values produce different compiled output.
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

    auto compile_and_get_elf_path = [&]() -> fs::path {
        Program program = CreateProgram();
        DataMovementConfig config{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compiler_include_paths = {include_dir},
        };
        auto kernel_handle = CreateKernelFromString(program, kernel_src, CoreCoord{0, 0}, config);
        detail::CompileProgram(device, program);

        const uint32_t tensix_core_type =
            MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
        const uint32_t dm_class_idx = enchantum::to_underlying(HalProcessorClassType::DM);
        const int riscv_id = static_cast<std::underlying_type_t<DataMovementProcessor>>(DataMovementProcessor::RISCV_0);
        const JitBuildState& build_state =
            BuildEnvManager::get_instance(extract_context_id(device))
                .get_kernel_build_state(device->build_id(), tensix_core_type, dm_class_idx, riscv_id);
        const auto& kernels = program.impl().get_kernels(static_cast<uint32_t>(HalProgrammableCoreType::TENSIX));
        const std::string full_kernel_name = kernels.at(kernel_handle)->get_full_kernel_name();
        return build_state.get_target_out_path(full_kernel_name);
    };

    // First compile: header has sentinel 0xC0DE.
    write_header(0xC0DE);
    fs::path elf_path_v1 = compile_and_get_elf_path();
    ASSERT_TRUE(fs::exists(elf_path_v1));
    auto bytes_v1 = read_file_bytes(elf_path_v1);
    ASSERT_FALSE(bytes_v1.empty());

    // Edit the header to use a different sentinel value.
    write_header(0xBEEF);

    // Clear the per-process dedup cache (JitBuildCache) before recompiling. That
    // cache is keyed on compute_hash and short-circuits the build path entirely
    // when a kernel with the same hash has already been built in this process.
    // We want to exercise the *disk-level* dephash machinery (which detects header
    // content changes), so we have to bypass the in-process layer first. In real
    // workflows this isn't an issue: header edits happen between process runs,
    // and a fresh process starts with an empty in-process cache.
    jit_build_cache_clear();

    // Second compile. Same source + same include-path list, so we map to the same
    // cache slot — but the dephash mechanism should detect the header content change
    // and rebuild.
    fs::path elf_path_v2 = compile_and_get_elf_path();

    EXPECT_EQ(elf_path_v1, elf_path_v2)
        << "Same source + same include-path list should map to the same kernel cache slot";

    auto bytes_v2 = read_file_bytes(elf_path_v2);
    EXPECT_NE(bytes_v1, bytes_v2) << "Editing a header in compiler_include_paths should trigger a rebuild "
                                     "(if this fails, the .dephash mechanism is not catching the header edit)";

    fs::remove_all(include_dir);
}

}  // namespace tt::tt_metal
