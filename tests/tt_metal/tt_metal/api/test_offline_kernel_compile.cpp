// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <string>
#include <stdexcept>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/offline_kernel_compile.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "device_fixture.hpp"
#include "jit_build/build.hpp"
#include "llrt/rtoptions.hpp"
#include "tt_metal/jit_build/build_cache_telemetry.hpp"

namespace tt::tt_metal {

namespace {

namespace fs = std::filesystem;

using BinaryPolicy = experimental::PrecompiledKernelConfig::FallbackPolicy;
using CBCompileConfig = experimental::OfflineKernelCompileParams::CBCompileConfig;

// CompileKernelOffline builds its own RunTimeOptions from the environment; for a non-Silicon target
// (simulator/emulation) that disables multi-erisc mode, which shifts the firmware build_key away
// from the precompiled-firmware bundle. The offline path does not build firmware itself, so for the
// non-simulated arch (e.g. Wormhole) there is no weakened firmware ELF to link kernels against and
// the build fails. Mirror that same fresh RunTimeOptions here (rather than the live MetalContext,
// which a mock fixture forces to Mock) and skip the offline-compile tests until that path can build
// (or locate) firmware for the simulator build_key.
bool offline_compile_unsupported_under_simulator() { return llrt::RunTimeOptions{}.is_simulator_or_emulated(); }

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

    fs::path path_;
};

class OfflineKernelCompileMockFixture : public ::testing::Test {
protected:
    void SetUp() override { experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 1); }

    void TearDown() override { experimental::disable_mock_mode(); }
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

// Seed `output_dir` with offline-compiled kernel artifacts for every supported product, so
// runtime tests can point `PrecompiledKernelConfig::precompiled_dir` at it. Driving this
// through `experimental::CompileKernelOffline(AllSupportedProducts)` (instead of a live
// JIT compile + tree copy) is the contract this slice asserts: the offline-emitted hash
// buckets must match what the runtime precompiled-loader path searches for.
void seed_precompiled_root(
    const fs::path& output_dir,
    const std::string& kernel_path,
    const std::variant<DataMovementConfig, ComputeConfig>& kernel_config,
    const std::vector<CBCompileConfig>& cb_compile_configs = {}) {
    using Params = experimental::OfflineKernelCompileParams;
    Params params{
        .mode = Params::AllSupportedProducts{},
        .output_dir = output_dir,
        .cb_compile_configs = cb_compile_configs,
    };
    experimental::CompileKernelOffline(kernel_path, kernel_config, params);
}

TEST_F(OfflineKernelCompileMockFixture, MetadataFromProgramDerivesConfiguredCbMetadata) {
    Program program = CreateProgram();
    const Tile tile({16, 32});
    const auto page_size = tile.get_tile_size(DataFormat::Float16_b);
    CircularBufferConfig cb_config(page_size, {{CBIndex::c_0, DataFormat::Float16_b}});
    cb_config.set_page_size(CBIndex::c_0, page_size).set_tile_dims(CBIndex::c_0, tile);
    CreateCircularBuffer(program, CoreCoord{0, 0}, cb_config);
    const KernelHandle kernel = CreateKernel(program, kReaderKernelPath, CoreCoord{0, 0}, kReaderDmConfig);

    const auto cb_compile_configs = experimental::CBCompileConfigsFromProgram(program, kernel);
    ASSERT_EQ(cb_compile_configs.size(), 1);
    EXPECT_EQ(cb_compile_configs[0].cb_index, 0);
    EXPECT_EQ(cb_compile_configs[0].data_format, DataFormat::Float16_b);
    ASSERT_TRUE(cb_compile_configs[0].tile.has_value());
    EXPECT_EQ(*cb_compile_configs[0].tile, tile);
}

TEST_F(OfflineKernelCompileMockFixture, CBCompileConfigsFromProgramDeduplicatesOverlappingCbIndex) {
    Program program = CreateProgram();
    const CoreRange left_core(CoreCoord{0, 0}, CoreCoord{0, 0});
    const CoreRange right_core(CoreCoord{1, 0}, CoreCoord{1, 0});
    const CoreRangeSet kernel_cores(std::vector<CoreRange>{left_core, right_core});
    const KernelHandle kernel = CreateKernel(program, kReaderKernelPath, kernel_cores, kReaderDmConfig);

    constexpr uint32_t kPageSize = 2048;
    CreateCircularBuffer(
        program,
        left_core,
        CircularBufferConfig(kPageSize, {{CBIndex::c_0, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_0, kPageSize));
    CreateCircularBuffer(
        program,
        right_core,
        CircularBufferConfig(kPageSize, {{CBIndex::c_0, DataFormat::Bfp8_b}})
            .set_page_size(CBIndex::c_0, kPageSize));

    const auto cb_compile_configs = experimental::CBCompileConfigsFromProgram(program, kernel);
    ASSERT_EQ(cb_compile_configs.size(), 1);
    EXPECT_EQ(cb_compile_configs[0].cb_index, 0);
}

TEST_F(OfflineKernelCompileMockFixture, CompileKernelOfflineRejectsInvalidExplicitCbMetadata) {
    using Params = experimental::OfflineKernelCompileParams;
    Params params{
        .mode = Params::AllSupportedProducts{},
        .output_dir = fs::path("/tmp/unused"),
        .cb_compile_configs =
            {
                Params::CBCompileConfig{.cb_index = 0, .data_format = DataFormat::Float16_b},
                Params::CBCompileConfig{.cb_index = 0, .data_format = DataFormat::Float16_b},
            },
    };

    EXPECT_THROW(experimental::CompileKernelOffline(kReaderKernelPath, kReaderDmConfig, params), std::invalid_argument);
}

TEST_F(OfflineKernelCompileMockFixture, CompileKernelOfflineRejectsEmptyOutputDir) {
    using Params = experimental::OfflineKernelCompileParams;
    Params params{
        .mode = Params::AllSupportedProducts{},
        .output_dir = fs::path{},
        .cb_compile_configs = {},
    };
    EXPECT_THROW(experimental::CompileKernelOffline(kReaderKernelPath, kReaderDmConfig, params), std::invalid_argument);
}

// Returns the number of subdirectories directly under `dir` whose names parse as decimal digits
// (i.e. compile-hash buckets). Returns 0 if `dir` does not exist.
size_t count_compile_hash_subdirs(const fs::path& dir) {
    if (!fs::exists(dir)) {
        return 0;
    }
    size_t count = 0;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_directory()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (!name.empty() &&
            std::all_of(name.begin(), name.end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)); })) {
            ++count;
        }
    }
    return count;
}

// Returns true if `dir` (recursively) contains at least one .elf file with size > 0.
bool contains_nonempty_elf(const fs::path& dir) {
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".elf" && fs::file_size(entry.path()) > 0) {
            return true;
        }
    }
    return false;
}

TEST_F(OfflineKernelCompileMockFixture, CompileKernelOfflineEmitsExpectedSubtreeForReaderKernel) {
    if (offline_compile_unsupported_under_simulator()) {
        GTEST_SKIP() << "CompileKernelOffline has no precompiled firmware for the simulator build_key "
                        "(multi-erisc disabled); skipping under TT_METAL_SIMULATOR.";
    }
    ScopedTempDir output_dir("tt_metal_offline_compile_smoke");

    using Params = experimental::OfflineKernelCompileParams;
    Params params{
        .mode = Params::AllSupportedProducts{},
        .output_dir = output_dir.path_,
        .cb_compile_configs = {},
    };

    ASSERT_NO_THROW(experimental::CompileKernelOffline(kReaderKernelPath, kReaderDmConfig, params));

    const fs::path kernel_subdir = output_dir.path_ / kReaderKernelName;
    ASSERT_TRUE(fs::exists(kernel_subdir)) << "Expected kernel subdir at " << kernel_subdir;

    // AllSupportedProducts enumerates every (arch, core_descriptor, soc_descriptor) tuple in the
    // jit_build offline-compile table; each yields one or more JitDeviceConfig values, so the
    // subtree must contain multiple distinct compile-hash buckets.
    const size_t hash_subdir_count = count_compile_hash_subdirs(kernel_subdir);
    EXPECT_GT(hash_subdir_count, 1u) << "Expected >1 compile-hash buckets under " << kernel_subdir;

    EXPECT_TRUE(contains_nonempty_elf(kernel_subdir)) << "Expected at least one non-empty .elf under " << kernel_subdir;
}

}  // namespace

TEST_F(MeshDeviceFixture, RuntimePrecompiledHitLoadsWithoutJit) {
    if (offline_compile_unsupported_under_simulator()) {
        GTEST_SKIP() << "CompileKernelOffline has no precompiled firmware for the simulator build_key "
                        "(multi-erisc disabled); skipping under TT_METAL_SIMULATOR.";
    }
    auto* device = this->devices_.at(0)->get_devices().at(0);

    ScopedTempDir precompiled_root("tt_metal_precompiled_seed_hit");
    seed_precompiled_root(precompiled_root.path_, kReaderKernelPath, kReaderDmConfig);

    const auto precompiled_config = make_precompiled_config(precompiled_root.path_.string(), BinaryPolicy::Error);
    Program program = create_precompiled_program(precompiled_config);

    jit_build_cache_clear();
    JitSrcsBaseline jit_srcs;
    EXPECT_NO_THROW(detail::CompileProgram(device, program));
    EXPECT_EQ(jit_srcs.delta(), 0u);
}

TEST_F(MeshDeviceFixture, RuntimePrecompiledHitWithCbMetadataLoadsWithoutJit) {
    if (offline_compile_unsupported_under_simulator()) {
        GTEST_SKIP() << "CompileKernelOffline has no precompiled firmware for the simulator build_key "
                        "(multi-erisc disabled); skipping under TT_METAL_SIMULATOR.";
    }
    // Verifies the CBCompileConfigsFromProgram + CompileKernelOffline path produces a
    // bucket whose hash inputs (build_key + hlk_desc CB metadata + kernel compute hash)
    // match the runtime-computed hash for an equivalently-configured program. If the
    // hlk_desc contributions diverge, this test fails as `jit_srcs.delta() > 0` (runtime
    // falls through to JIT) rather than as a layout assertion, which is exactly the
    // failure mode that justifies surfacing CBCompileConfigsFromProgram in the public API.
    auto* device = this->devices_.at(0)->get_devices().at(0);

    constexpr uint32_t kPageSize = 2048;
    constexpr DataFormat kCbFormat = DataFormat::Float16_b;

    // Reference program: built only to derive CB compile configs that mirror the runtime
    // CB layout. CBCompileConfigsFromProgram does not require the program to be compiled.
    Program metadata_program = CreateProgram();
    CircularBufferConfig metadata_cb_config(kPageSize, {{CBIndex::c_0, kCbFormat}});
    metadata_cb_config.set_page_size(CBIndex::c_0, kPageSize);
    CreateCircularBuffer(metadata_program, CoreCoord{0, 0}, metadata_cb_config);
    const KernelHandle metadata_kernel =
        CreateKernel(metadata_program, kReaderKernelPath, CoreCoord{0, 0}, kReaderDmConfig);
    const auto cb_compile_configs = experimental::CBCompileConfigsFromProgram(metadata_program, metadata_kernel);
    ASSERT_EQ(cb_compile_configs.size(), 1);
    EXPECT_EQ(cb_compile_configs[0].cb_index, 0);
    EXPECT_EQ(cb_compile_configs[0].data_format, kCbFormat);

    ScopedTempDir precompiled_root("tt_metal_precompiled_seed_cb_hit");
    seed_precompiled_root(precompiled_root.path_, kReaderKernelPath, kReaderDmConfig, cb_compile_configs);

    // Runtime program: same CB layout + precompiled kernel. Hash inputs must match offline
    // emission for the load-without-JIT contract to hold.
    const auto precompiled_config = make_precompiled_config(precompiled_root.path_.string(), BinaryPolicy::Error);
    Program runtime_program = CreateProgram();
    CircularBufferConfig runtime_cb_config(kPageSize, {{CBIndex::c_0, kCbFormat}});
    runtime_cb_config.set_page_size(CBIndex::c_0, kPageSize);
    CreateCircularBuffer(runtime_program, CoreCoord{0, 0}, runtime_cb_config);
    experimental::CreateKernelFromPrecompiled(
        runtime_program, kReaderKernelPath, CoreCoord{0, 0}, kReaderDmConfig, precompiled_config);

    jit_build_cache_clear();
    JitSrcsBaseline jit_srcs;
    EXPECT_NO_THROW(detail::CompileProgram(device, runtime_program));
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
