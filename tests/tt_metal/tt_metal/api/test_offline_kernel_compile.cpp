// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <string>

#include <tt-metalium/experimental/offline_kernel_compile.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "device_fixture.hpp"
#include "jit_build/build.hpp"
#include "tt_metal/jit_build/build_cache_telemetry.hpp"
#include "tt_metal/jit_build/build_env_manager.hpp"

namespace tt::tt_metal {

namespace {

namespace fs = std::filesystem;

using BinaryPolicy = experimental::PrecompiledKernelConfig::FallbackPolicy;

struct ScopedCopiedPrecompiledRoot {
    explicit ScopedCopiedPrecompiledRoot(fs::path root) : root_(std::move(root)) {}
    ~ScopedCopiedPrecompiledRoot() {
        std::error_code ec;
        fs::remove_all(root_, ec);
    }

    fs::path root_;
};

constexpr const char* kReaderKernelPath = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp";
constexpr const char* kReaderKernelName = "reader_unary_push_4";
constexpr const char* kMissingPrecompiledRoot = "/tmp/tt_metal_nonexistent_precompiled_dir";
const DataMovementConfig kReaderDmConfig{
    .processor = DataMovementProcessor::RISCV_0,
    .noc = NOC::RISCV_0_default,
};

experimental::PrecompiledKernelConfig make_precompiled_config(const std::string& root, BinaryPolicy policy) {
    return experimental::PrecompiledKernelConfig{.precompiled_dir = root, .fallback_policy = policy};
}

Program create_precompiled_program(
    const experimental::PrecompiledKernelConfig& precompiled_config,
    const std::string& kernel_path = kReaderKernelPath) {
    Program program = CreateProgram();
    experimental::CreateKernelFromPrecompiled(
        program, kernel_path, CoreCoord{0, 0}, kReaderDmConfig, precompiled_config);
    return program;
}

// Snapshot of the process-wide srcs counter, which advances on every JitBuildState::compile()
// call (the shared hot path of every jit_build* entry point). delta() > 0 after a
// CompileProgram call means the JIT pipeline ran. Snapshotting (instead of resetting the
// telemetry singleton) keeps other tests sharing the same process unaffected.
struct JitSrcsBaseline {
    uint32_t baseline = BuildCacheTelemetry::inst().get_srcs_count();
    uint32_t delta() const { return BuildCacheTelemetry::inst().get_srcs_count() - baseline; }
};

Program create_regular_program(const std::string& kernel_path = kReaderKernelPath) {
    Program program = CreateProgram();
    CreateKernel(program, kernel_path, CoreCoord{0, 0}, kReaderDmConfig);
    return program;
}

ScopedCopiedPrecompiledRoot precompiled_root_from_live_compile(IDevice* device) {
    // AOT kernel compilation is not implemented yet, so we seed precompiled artifacts
    // by compiling once on a live device, then copying the resulting kernel subtree into
    // a temporary directory used only as the "precompiled" source.
    Program jit_program = create_regular_program();
    jit_build_cache_clear();
    JitSrcsBaseline jit_srcs;
    detail::CompileProgram(device, jit_program);
    TT_FATAL(jit_srcs.delta() > 0, "Expected seed JIT compile to invoke jit_build");
    const fs::path jit_kernel_root =
        BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_env.get_out_kernel_root_path();
    const fs::path jit_kernel_subdir = jit_kernel_root / kReaderKernelName;
    TT_FATAL(fs::exists(jit_kernel_subdir), "Expected JIT kernel artifacts at {}", jit_kernel_subdir.string());

    const auto timestamp_ns = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path copied_precompiled_root =
        fs::temp_directory_path() / ("tt_metal_precompiled_copy_" + std::to_string(timestamp_ns));
    fs::create_directories(copied_precompiled_root);
    fs::copy(
        jit_kernel_subdir,
        copied_precompiled_root / kReaderKernelName,
        fs::copy_options::recursive | fs::copy_options::overwrite_existing);
    return ScopedCopiedPrecompiledRoot(copied_precompiled_root);
}

}  // namespace

TEST_F(MeshDeviceFixture, RuntimePrecompiledHitLoadsWithoutJit) {
    auto* device = this->devices_.at(0)->get_devices().at(0);
    auto copied_precompiled_root = precompiled_root_from_live_compile(device);

    const auto precompiled_config =
        make_precompiled_config(copied_precompiled_root.root_.string(), BinaryPolicy::Error);
    Program program = create_precompiled_program(precompiled_config);

    jit_build_cache_clear();
    JitSrcsBaseline jit_srcs;
    EXPECT_NO_THROW(detail::CompileProgram(device, program));
    EXPECT_EQ(jit_srcs.delta(), 0u);
}

TEST_F(MeshDeviceFixture, RuntimeMissingPrecompiledFallsBackToJit) {
    const auto precompiled_config = make_precompiled_config(kMissingPrecompiledRoot, BinaryPolicy::JitCompile);
    Program program = create_precompiled_program(precompiled_config);
    auto* device = this->devices_.at(0)->get_devices().at(0);

    jit_build_cache_clear();
    JitSrcsBaseline jit_srcs;
    EXPECT_NO_THROW(detail::CompileProgram(device, program));
    EXPECT_GT(jit_srcs.delta(), 0u);
}

TEST_F(MeshDeviceFixture, RuntimeMissingPrecompiledErrorsOnPolicyError) {
    const auto precompiled_config = make_precompiled_config(kMissingPrecompiledRoot, BinaryPolicy::Error);
    Program program = create_precompiled_program(precompiled_config);
    auto* device = this->devices_.at(0)->get_devices().at(0);

    jit_build_cache_clear();
    JitSrcsBaseline jit_srcs;
    try {
        detail::CompileProgram(device, program);
        FAIL() << "Expected PrecompiledKernelNotFoundError";
    } catch (const experimental::PrecompiledKernelNotFoundError& ex) {
        EXPECT_EQ(ex.kernel_name(), kReaderKernelName);
        EXPECT_EQ(ex.precompiled_dir(), precompiled_config.precompiled_dir);
        EXPECT_EQ(ex.fallback_policy(), precompiled_config.fallback_policy);
    } catch (const std::exception& ex) {
        FAIL() << "Unexpected exception type: " << ex.what();
    }
    EXPECT_EQ(jit_srcs.delta(), 0u);
}

}  // namespace tt::tt_metal
