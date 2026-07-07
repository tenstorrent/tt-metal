// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <tt-metalium/experimental/offline_kernel_compile.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "device_fixture.hpp"
#include "impl/program/kernel_prewarm.hpp"
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
    const fs::path jit_kernel_root = BuildEnvManager::get_instance(extract_context_id(device))
                                         .get_device_build_env(device->build_id())
                                         .build_env.get_out_kernel_root_path();
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

// A single-core data-movement kernel whose only observable content is |tag|: a file-scope char array
// forced into .rodata by a volatile store of its address. Editing |tag| changes the compiled binary
// but not the kernel hash (source *content* is not hashed -- only path/args/defines), so the two
// versions share a cache dir and manifest key. That collision is exactly the case where a prewarm
// that replayed a stale source snapshot would serve the old binary.
void write_probe_kernel(const fs::path& path, const std::string& tag) {
    std::ofstream f(path, std::ios::trunc | std::ios::binary);
    f << "#include <cstdint>\n"
         "namespace {\n"
         "const char kProbe[] = \""
      << tag
      << "\";\n"
         "}\n"
         "void kernel_main() {\n"
         "    *reinterpret_cast<volatile uintptr_t*>(0x10000) = reinterpret_cast<uintptr_t>(kProbe);\n"
         "}\n";
    TT_FATAL(!f.fail(), "Failed to write probe kernel to {}", path.string());
}

// The loadable kernel ELFs under |dir| (recursive), ordered by path for stable comparison. Excludes
// the "<name>.elf.xip.elf" sidecar: that is a debug disassembly dump written at load time for
// tt-triage (tt_memory.cpp), not a binary the device loads, and the prewarm/JIT path does not
// regenerate it -- so it can lag the source and must not be read as ground truth.
std::vector<fs::path> list_kernel_elfs(const fs::path& dir) {
    std::vector<fs::path> elfs;
    if (!fs::exists(dir)) {
        return elfs;
    }
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".elf" &&
            !entry.path().filename().string().ends_with(".xip.elf")) {
            elfs.push_back(entry.path());
        }
    }
    std::sort(elfs.begin(), elfs.end());
    return elfs;
}

// Concatenated bytes of every loadable kernel ELF under |dir| -- the ground truth for "what the device
// would run".
std::string read_kernel_elf_bytes(const fs::path& dir) {
    std::string blob;
    for (const auto& elf : list_kernel_elfs(dir)) {
        std::ifstream f(elf, std::ios::binary);
        std::stringstream ss;
        ss << f.rdbuf();
        blob += ss.str();
    }
    return blob;
}

bool blob_contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

std::string with_trailing_slash(std::string s) {
    if (!s.empty() && s.back() != '/') {
        s.push_back('/');
    }
    return s;
}

constexpr const char* kProbeTagV1 = "TTPREWARM_PROBE_AAAAAAAAAAAA";
constexpr const char* kProbeTagV2 = "TTPREWARM_PROBE_BBBBBBBBBBBB";

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

// The prewarm invariant: never serve a stale binary. After an edit to a kernel body, the off-device
// prewarm must produce a loaded binary (brisc.elf) that reflects the edit -- not the source snapshot
// captured with the manifest entry.
TEST_F(MeshDeviceFixture, OfflinePrewarmReflectsEditedKernelBody) {
    auto* device = this->devices_.at(0)->get_devices().at(0);
    const auto& build_env =
        BuildEnvManager::get_instance(extract_context_id(device)).get_device_build_env(device->build_id()).build_env;

    const auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path kernel_path = fs::temp_directory_path() / ("ttprewarm_probe_" + std::to_string(ts) + ".cpp");
    const fs::path kernel_subdir = fs::path(build_env.get_out_kernel_root_path()) / kernel_path.stem().string();

    auto compile_probe = [&]() {
        Program program = CreateProgram();
        CreateKernel(program, kernel_path.string(), CoreCoord{0, 0}, kReaderDmConfig);
        detail::CompileProgram(device, program);
    };

    // v1 through the normal device path. Capture is default-on, so this records the recipe (including
    // the generated-file snapshot) into the process manifest that the offline prewarm reads back.
    write_probe_kernel(kernel_path, kProbeTagV1);
    compile_probe();
    const std::string elf_v1 = read_kernel_elf_bytes(kernel_subdir);
    ASSERT_FALSE(elf_v1.empty()) << "no kernel .elf produced under " << kernel_subdir;
    ASSERT_TRUE(blob_contains(elf_v1, kProbeTagV1));
    ASSERT_FALSE(blob_contains(elf_v1, kProbeTagV2));

    // Edit the body only: same path/args => same kernel hash => same cache dir and manifest key. This
    // is the collision case where a prewarm that replayed the captured source snapshot would rebuild
    // the old binary.
    write_probe_kernel(kernel_path, kProbeTagV2);

    const std::size_t built = kernel_prewarm::prewarm_manifest_offline(
        build_env.get_out_root_path(), with_trailing_slash(build_env.get_root_path()));
    ASSERT_GT(built, 0u) << "offline prewarm built nothing (empty/missing manifest)";

    const std::string elf_prewarm = read_kernel_elf_bytes(kernel_subdir);
    EXPECT_TRUE(blob_contains(elf_prewarm, kProbeTagV2)) << "prewarm did not reflect the edited body";
    EXPECT_FALSE(blob_contains(elf_prewarm, kProbeTagV1)) << "prewarm served the STALE kernel body";

    fs::remove(kernel_path);
}

// The dephash backstop through the op-by-op compile path: with the old binary already on disk, a
// body edit must yield a compiled binary that reflects the edit, never a stale cache hit. The cache
// clear models the real workflow -- edit, then a fresh process re-runs -- where the in-memory
// build_once dedup (keyed on kernel_hash, which a body edit does NOT change) is cold and the on-disk
// .dephash is the gate. The gate's own change-detection is unit-tested in jit_build/test_depend.cpp;
// this asserts the end-to-end result: the device binary is v2, not v1.
TEST_F(MeshDeviceFixture, EditedKernelBodyForcesRecompileNotStaleCacheHit) {
    auto* device = this->devices_.at(0)->get_devices().at(0);
    const auto& build_env =
        BuildEnvManager::get_instance(extract_context_id(device)).get_device_build_env(device->build_id()).build_env;

    const auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path kernel_path = fs::temp_directory_path() / ("ttdephash_probe_" + std::to_string(ts) + ".cpp");
    const fs::path kernel_subdir = fs::path(build_env.get_out_kernel_root_path()) / kernel_path.stem().string();

    auto compile_probe = [&]() {
        Program program = CreateProgram();
        CreateKernel(program, kernel_path.string(), CoreCoord{0, 0}, kReaderDmConfig);
        detail::CompileProgram(device, program);
    };

    write_probe_kernel(kernel_path, kProbeTagV1);
    jit_build_cache_clear();
    compile_probe();
    const std::string elf_v1 = read_kernel_elf_bytes(kernel_subdir);
    ASSERT_TRUE(blob_contains(elf_v1, kProbeTagV1));
    ASSERT_FALSE(blob_contains(elf_v1, kProbeTagV2));

    // Edit the body only (same path/args => same kernel_hash => same cache dir). A fresh process would
    // see the on-disk v1 artifacts; model that by clearing the in-memory caches before recompiling.
    write_probe_kernel(kernel_path, kProbeTagV2);
    jit_build_cache_clear();
    compile_probe();

    const std::string elf_v2 = read_kernel_elf_bytes(kernel_subdir);
    EXPECT_TRUE(blob_contains(elf_v2, kProbeTagV2)) << "recompiled binary does not reflect the edit";
    EXPECT_FALSE(blob_contains(elf_v2, kProbeTagV1)) << "stale kernel body survived the edit";

    fs::remove(kernel_path);
}

}  // namespace tt::tt_metal
