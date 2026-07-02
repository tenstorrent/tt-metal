// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Real-hardware tests for the Metal 2.0 offline (AOT) compile + load APIs:
//   experimental::CompileProgramSpecOffline  (produce)
//   experimental::SetProgramPrecompiledConfig (load)
//
// These require a Wormhole B0 or Blackhole device. They prove the round trip:
//   produce ELFs from a ProgramSpec -> load them into a fresh Program built from the same
//   spec -> compile finds the AOT binaries (no JIT) -> and the negative paths (missing dir,
//   hash-bucket mismatch, late call) all fail loudly.
//
// Requires: TT_METAL_SLOW_DISPATCH_MODE=1 (mirrors test_program_spec_hw.cpp).

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>  // detail::CompileProgram
#include <tt-metalium/experimental/offline_kernel_compile.hpp>
#include <tt-metalium/experimental/metal2_host_api/offline_compile.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "device_fixture.hpp"
#include "test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

namespace fs = std::filesystem;

using BinaryPolicy = PrecompiledKernelConfig::FallbackPolicy;
using test_helpers::MakeMinimalGen1ValidProgramSpec;

// RAII temp directory, auto-removed on destruction.
struct ScopedTempDir {
    explicit ScopedTempDir(const std::string& tag) {
        const auto timestamp_ns = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = fs::temp_directory_path() / (tag + "_" + std::to_string(timestamp_ns));
        fs::create_directories(path_);
    }
    ~ScopedTempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    ScopedTempDir(const ScopedTempDir&) = delete;
    ScopedTempDir& operator=(const ScopedTempDir&) = delete;

    std::string string() const { return path_.string(); }

    fs::path path_;
};

// Count regular files anywhere under `root` (the emitted ELFs live in nested per-kernel dirs).
size_t count_regular_files(const fs::path& root) {
    size_t count = 0;
    if (!fs::exists(root)) {
        return 0;
    }
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.is_regular_file()) {
            ++count;
        }
    }
    return count;
}

PrecompiledKernelConfig make_config(const std::string& dir, BinaryPolicy policy) {
    return PrecompiledKernelConfig{.precompiled_dir = dir, .fallback_policy = policy};
}

class OfflineCompileHWTest : public tt::tt_metal::MeshDeviceFixture {
protected:
    void SetUp() override {
        MeshDeviceFixture::SetUp();
        if (this->IsSkipped()) {
            return;
        }
        if (devices_.at(0)->arch() != tt::ARCH::WORMHOLE_B0 && devices_.at(0)->arch() != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Skipping: test requires Wormhole B0 or Blackhole hardware";
        }
    }
};

// Produce: CompileProgramSpecOffline emits AOT artifacts under output_dir.
TEST_F(OfflineCompileHWTest, ProduceEmitsBinaries) {
    auto mesh_device = devices_.at(0);
    ScopedTempDir out_dir("metal2_offline_produce");

    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    CompileProgramSpecOffline(*mesh_device, spec, out_dir.string());

    EXPECT_GT(count_regular_files(out_dir.path_), 0u)
        << "CompileProgramSpecOffline emitted no files under " << out_dir.string();
}

// Load hit: a fresh program from the same spec, pointed at the produced dir with the strict
// Error policy, compiles without throwing -> the AOT binaries were found at exactly the path
// the runtime loader searches (the production code itself is the exact-path assertion).
TEST_F(OfflineCompileHWTest, LoadHitUsesPrecompiledBinaries) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];
    ScopedTempDir out_dir("metal2_offline_loadhit");

    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    CompileProgramSpecOffline(*mesh_device, spec, out_dir.string());

    Program program = MakeProgramFromSpec(*mesh_device, spec);
    SetProgramPrecompiledConfig(program, make_config(out_dir.string(), BinaryPolicy::Error));
    EXPECT_NO_THROW(detail::CompileProgram(device, program));
}

// Load miss (empty dir): strict Error policy against a dir with no binaries throws.
TEST_F(OfflineCompileHWTest, LoadMissEmptyDirThrows) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];
    ScopedTempDir empty_dir("metal2_offline_empty");

    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    Program program = MakeProgramFromSpec(*mesh_device, spec);
    SetProgramPrecompiledConfig(program, make_config(empty_dir.string(), BinaryPolicy::Error));

    EXPECT_THROW(detail::CompileProgram(device, program), PrecompiledKernelNotFoundError);
}

// Load miss (hash-bucket mismatch): produce from spec A, then load a structurally different
// spec B (different kernel source -> different compile hash) from the same dir. The kernel's
// recomputed hash bucket is absent, so strict Error throws. This is the same failure mode as a
// produce-time/load-time build_key mismatch (watcher/dprint/opt-level divergence).
TEST_F(OfflineCompileHWTest, LoadMissOnHashMismatchThrows) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];
    ScopedTempDir out_dir("metal2_offline_mismatch");

    ProgramSpec produced = MakeMinimalGen1ValidProgramSpec();
    CompileProgramSpecOffline(*mesh_device, produced, out_dir.string());

    // Same structure, but mutate one kernel's source so its compile hash changes.
    ProgramSpec divergent = MakeMinimalGen1ValidProgramSpec();
    divergent.kernels.at(0).source = KernelSpec::SourceCode{"void kernel_main() { volatile int x = 7; (void)x; }"};

    Program program = MakeProgramFromSpec(*mesh_device, divergent);
    SetProgramPrecompiledConfig(program, make_config(out_dir.string(), BinaryPolicy::Error));

    EXPECT_THROW(detail::CompileProgram(device, program), PrecompiledKernelNotFoundError);
}

// Load fallback: JitCompile policy against an empty dir silently JIT-compiles (no throw).
TEST_F(OfflineCompileHWTest, LoadMissJitFallbackCompiles) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];
    ScopedTempDir empty_dir("metal2_offline_jitfallback");

    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    Program program = MakeProgramFromSpec(*mesh_device, spec);
    SetProgramPrecompiledConfig(program, make_config(empty_dir.string(), BinaryPolicy::JitCompile));

    EXPECT_NO_THROW(detail::CompileProgram(device, program));
}

// Guard: empty output_dir is rejected.
TEST_F(OfflineCompileHWTest, ProduceEmptyOutputDirThrows) {
    auto mesh_device = devices_.at(0);
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    EXPECT_THROW(CompileProgramSpecOffline(*mesh_device, spec, ""), std::runtime_error);
}

// Guard: SetProgramPrecompiledConfig must be called before the program is compiled.
TEST_F(OfflineCompileHWTest, SetConfigAfterCompileThrows) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];
    ScopedTempDir out_dir("metal2_offline_late");

    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    Program program = MakeProgramFromSpec(*mesh_device, spec);
    detail::CompileProgram(device, program);  // compiles (JIT) -> program is now compiled

    EXPECT_THROW(
        SetProgramPrecompiledConfig(program, make_config(out_dir.string(), BinaryPolicy::Error)), std::runtime_error);
}

}  // namespace
}  // namespace tt::tt_metal::experimental
