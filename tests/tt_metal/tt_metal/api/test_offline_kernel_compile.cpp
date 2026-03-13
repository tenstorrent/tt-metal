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
#include "tt_metal/jit_build/build_env_manager.hpp"

namespace tt::tt_metal {

namespace {

namespace fs = std::filesystem;

using BinaryPolicy = experimental::PrecompiledKernelConfig::FallbackPolicy;

struct ScopedCopiedPrecompiledRoot {
    explicit ScopedCopiedPrecompiledRoot(fs::path root) : root_(std::move(root)) {}
    ~ScopedCopiedPrecompiledRoot() { fs::remove_all(root_); }

    fs::path root_;
};

constexpr const char* kReaderKernelPath = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp";
constexpr const char* kComputeKernelPath = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp";
constexpr const char* kReaderKernelName = "reader_unary_push_4";
constexpr const char* kMissingPrecompiledRoot = "/tmp/tt_metal_nonexistent_precompiled_dir";
constexpr const char* kApiOnlyPrecompiledRoot = "/tmp/tt_metal_precompiled";
const CoreCoord kSingleCore{0, 0};
const DataMovementConfig kReaderDmConfig{
    .processor = DataMovementProcessor::RISCV_0,
    .noc = NOC::RISCV_0_default,
};
const ComputeConfig kBasicComputeConfig{
    .math_fidelity = MathFidelity::HiFi4,
    .compile_args = {1},
};

experimental::PrecompiledKernelConfig make_precompiled_config(const std::string& root, BinaryPolicy policy) {
    return experimental::PrecompiledKernelConfig{.precompiled_dir = root, .fallback_policy = policy};
}

Program create_precompiled_program(
    const experimental::PrecompiledKernelConfig& precompiled_config,
    const std::string& kernel_path = kReaderKernelPath) {
    Program program = CreateProgram();
    experimental::CreateKernelFromPrecompiled(program, kernel_path, kSingleCore, kReaderDmConfig, precompiled_config);
    return program;
}

void clear_jit_observability_state() {
    // Keep tests independent when run together in a single process.
    jit_build_cache_clear();
    jit_build_reset_invocation_count();
}

Program create_regular_program(const std::string& kernel_path = kReaderKernelPath) {
    Program program = CreateProgram();
    CreateKernel(program, kernel_path, kSingleCore, kReaderDmConfig);
    return program;
}

ScopedCopiedPrecompiledRoot precompiled_root_from_live_compile(IDevice* device) {
    // AOT kernel compilation is not implemented yet, so we seed precompiled artifacts
    // by compiling once on a live device, then copying the resulting kernel subtree into
    // a temporary directory used only as the "precompiled" source.
    Program jit_program = create_regular_program();
    clear_jit_observability_state();
    detail::CompileProgram(device, jit_program);
    TT_FATAL(jit_build_get_invocation_count() > 0, "Expected seed JIT compile to invoke jit_build");
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

    clear_jit_observability_state();
    EXPECT_NO_THROW(detail::CompileProgram(device, program));
    EXPECT_EQ(jit_build_get_invocation_count(), 0);
}

TEST_F(MeshDeviceFixture, RuntimeMissingPrecompiledFallsBackToJit) {
    const auto precompiled_config = make_precompiled_config(kMissingPrecompiledRoot, BinaryPolicy::JitCompile);
    Program program = create_precompiled_program(precompiled_config);
    auto* device = this->devices_.at(0)->get_devices().at(0);

    clear_jit_observability_state();
    EXPECT_NO_THROW(detail::CompileProgram(device, program));
    EXPECT_GT(jit_build_get_invocation_count(), 0);
}

TEST_F(MeshDeviceFixture, RuntimeMissingPrecompiledErrorsOnPolicyError) {
    const auto precompiled_config = make_precompiled_config(kMissingPrecompiledRoot, BinaryPolicy::Error);
    Program program = create_precompiled_program(precompiled_config);
    auto* device = this->devices_.at(0)->get_devices().at(0);

    clear_jit_observability_state();
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
    EXPECT_EQ(jit_build_get_invocation_count(), 0);
}

TEST_F(MeshDeviceFixture, PrecompiledAPIAcceptsConfigs) {
    Program program = CreateProgram();

    const auto precompiled_config = make_precompiled_config(kApiOnlyPrecompiledRoot, BinaryPolicy::JitCompile);

    EXPECT_NO_THROW(experimental::CreateKernelFromPrecompiled(
        program, kReaderKernelPath, kSingleCore, kReaderDmConfig, precompiled_config));

    EXPECT_NO_THROW(experimental::CreateKernelFromPrecompiled(
        program, kComputeKernelPath, kSingleCore, kBasicComputeConfig, precompiled_config));
}

}  // namespace tt::tt_metal
