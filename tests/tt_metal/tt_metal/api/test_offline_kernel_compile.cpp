// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <tt-metalium/experimental/offline_kernel_compile.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "common/executor.hpp"
#include "device_fixture.hpp"
#include "impl/program/kernel_prewarm.hpp"
#include "jit_build/build.hpp"
#include "llrt/rtoptions.hpp"
#include "tt_metal/jit_build/build_cache_telemetry.hpp"
#include "tt_metal/jit_build/build_env_manager.hpp"

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

constexpr const char* kSubprocessModeEnv = "TT_METAL_PREWARM_TEST_SUBPROCESS";
constexpr const char* kShutdownActiveEnv = "TT_METAL_PREWARM_TEST_ACTIVE";
constexpr const char* kShutdownDoneEnv = "TT_METAL_PREWARM_TEST_DONE";
constexpr const char* kShutdownReleaseEnv = "TT_METAL_PREWARM_TEST_RELEASE";
constexpr const char* kShutdownTeardownEnv = "TT_METAL_PREWARM_TEST_TEARDOWN";

int run_test_subprocess(
    const std::string& filter, const std::vector<std::pair<std::string, std::string>>& environment) {
    const pid_t pid = ::fork();
    if (pid == 0) {
        for (const auto& [name, value] : environment) {
            if (::setenv(name.c_str(), value.c_str(), /*overwrite=*/1) != 0) {
                std::_Exit(120);
            }
        }
        ::unsetenv("TT_METAL_KERNEL_MANIFEST_WRITE");
        ::unsetenv("TT_METAL_KERNEL_PREWARM_MANIFEST");

        const std::string filter_arg = "--gtest_filter=" + filter;
        char* const args[] = {const_cast<char*>("/proc/self/exe"), const_cast<char*>(filter_arg.c_str()), nullptr};
        ::execv("/proc/self/exe", args);
        std::_Exit(121);
    }
    if (pid < 0) {
        return -1;
    }

    int status = 0;
    if (::waitpid(pid, &status, 0) != pid) {
        return -1;
    }
    return status;
}

void expect_subprocess_success(int status) {
    ASSERT_NE(status, -1) << "subprocess launch or wait failed";
    ASSERT_TRUE(WIFEXITED(status)) << "subprocess terminated abnormally with status " << status;
    EXPECT_EQ(WEXITSTATUS(status), 0) << "subprocess exited with code " << WEXITSTATUS(status);
}

// Proof that the child compiled into its isolated TT_METAL_CACHE rather than an inherited one. An
// explicitly-set TT_METAL_CACHE of "<X>" normalizes to a cache root of "<X>/tt-metal-cache" (no
// trailing slash), and JitBuildEnv concatenates the build_key as a *suffix*: "<X>/tt-metal-cache<bk>/".
// So the isolated tree is a "tt-metal-cache"-prefixed entry, never a bare "tt-metal-cache" directory.
bool isolated_cache_populated(const fs::path& cache_dir) {
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(cache_dir, ec)) {
        if (entry.is_directory(ec) && entry.path().filename().string().rfind("tt-metal-cache", 0) == 0) {
            return true;
        }
    }
    return false;
}

class IsolatedKernelCacheMeshDeviceFixture : public MeshDeviceFixture {
protected:
    void SetUp() override {
        if (std::getenv(kSubprocessModeEnv) == nullptr) {
            GTEST_SKIP() << "behavior body runs only in a cache-isolated subprocess";
        }
        MeshDeviceFixture::SetUp();
    }
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

void write_probe_kernel(const fs::path& path, const std::string& tag) {
    std::ofstream file(path, std::ios::trunc | std::ios::binary);
    file << "#include <cstdint>\n"
            "namespace {\n"
            "const char kProbe[] = \""
         << tag
         << "\";\n"
            "}\n"
            "void kernel_main() {\n"
            "    *reinterpret_cast<volatile uintptr_t*>(0x10000) = reinterpret_cast<uintptr_t>(kProbe);\n"
            "}\n";
    TT_FATAL(!file.fail(), "Failed to write probe kernel to {}", path.string());
}

// Loadable ELFs are the ground truth for the code that the device runs. XIP sidecars are debug
// disassembly dumps and can lag the compiled kernel.
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

std::string read_kernel_elf_bytes(const fs::path& dir) {
    std::string bytes;
    for (const auto& elf : list_kernel_elfs(dir)) {
        std::ifstream file(elf, std::ios::binary);
        std::stringstream stream;
        stream << file.rdbuf();
        bytes += stream.str();
    }
    return bytes;
}

bool blob_contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

std::string with_trailing_slash(std::string path) {
    if (!path.empty() && path.back() != '/') {
        path.push_back('/');
    }
    return path;
}

constexpr const char* kProbeTagV1 = "TTPREWARM_PROBE_AAAAAAAAAAAA";
constexpr const char* kProbeTagV2 = "TTPREWARM_PROBE_BBBBBBBBBBBB";

jit_server::CompileRequest make_shutdown_request(const fs::path& root) {
    constexpr std::uint64_t build_key = 24680;
    jit_build::TargetRecipe target{
        .target_name = "shutdown_probe",
        .compiler_opt_level = "O2",
        .srcs = {(root / "shutdown_probe.cpp").string()},
        .objs = {"shutdown_probe.o"},
        .linker_script = (root / "dependency.hpp").string(),
        .linker_opt_level = "O2",
    };
    return {
        .build_key = build_key,
        .kernel_name = "shutdown_probe/hash",
        .gpp = (root / "blocking-compiler.sh").string(),
        .targets = {std::move(target)},
    };
}

void fail_if_prewarm_reaches_dependency_teardown() {
    const char* active = std::getenv(kShutdownActiveEnv);
    const char* done = std::getenv(kShutdownDoneEnv);
    const char* teardown = std::getenv(kShutdownTeardownEnv);
    if (teardown != nullptr) {
        const int fd = ::open(teardown, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) {
            ::close(fd);
        }
    }
    if (active != nullptr && done != nullptr && ::access(active, F_OK) == 0 && ::access(done, F_OK) != 0) {
        std::_Exit(91);
    }
}

void wait_for_file(const fs::path& path) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!fs::exists(path) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(fs::exists(path)) << "timed out waiting for " << path;
}

TEST(KernelPrewarmShutdownChild, WritesManifest) {
    if (std::getenv(kSubprocessModeEnv) == nullptr) {
        GTEST_SKIP() << "behavior body runs only in the manifest-writer subprocess";
    }

    const fs::path root = std::getenv("TT_METAL_CACHE");
    const fs::path out_root = root / "cache";
    constexpr std::uint64_t build_key = 24680;
    fs::create_directories(out_root);
    kernel_prewarm::maybe_launch_prewarm(
        (out_root / std::to_string(build_key) / "kernels").string() + "/", (root / "firmware").string(), build_key, "");
    kernel_prewarm::append_manifest_entry(make_shutdown_request(root));
}

TEST(KernelPrewarmShutdownChild, StartsRealBatchAndReturnsWithoutBarrier) {
    if (std::getenv(kSubprocessModeEnv) == nullptr) {
        GTEST_SKIP() << "behavior body runs only in the shutdown subprocess";
    }

    const fs::path root = std::getenv("TT_METAL_CACHE");
    const fs::path active = std::getenv(kShutdownActiveEnv);
    const fs::path out_root = root / "cache";
    constexpr std::uint64_t build_key = 24680;
    const fs::path kernel_dir = out_root / std::to_string(build_key) / "kernels" / "shutdown_probe" / "hash";
    const fs::path source = root / "shutdown_probe.cpp";
    const fs::path object = kernel_dir / "shutdown_probe.o";
    const fs::path dependency = root / "dependency.hpp";
    const fs::path compiler = root / "blocking-compiler.sh";

    fs::create_directories(kernel_dir);
    fs::create_directories(out_root);
    std::ofstream(source) << "void kernel_main() {}\n";
    std::ofstream(dependency) << "#pragma once\n";
    std::ofstream(object) << "old object\n";
    std::ofstream(object.string() + ".dephash") << dependency << "\t0\n";
    std::ofstream script(compiler);
    script << "#!/usr/bin/env bash\n"
              "set -eu\n"
              "if mkdir \"${TT_METAL_PREWARM_TEST_ACTIVE}.lock\" 2>/dev/null; then\n"
              "  : > \"${TT_METAL_PREWARM_TEST_ACTIVE}\"\n"
              "  IFS= read -r _ < \"${TT_METAL_PREWARM_TEST_RELEASE}\"\n"
              "  : > \"${TT_METAL_PREWARM_TEST_DONE}\"\n"
              "fi\n"
              "out=''\n"
              "dep=''\n"
              "while (($#)); do\n"
              "  case \"$1\" in\n"
              "    -o) shift; out=\"$1\" ;;\n"
              "    -MF) shift; dep=\"$1\" ;;\n"
              "  esac\n"
              "  shift\n"
              "done\n"
              "[[ -z \"$out\" ]] || : > \"$out\"\n"
              "[[ -z \"$dep\" ]] || printf '%s: %s\\n' \"$out\" \""
           << dependency.string() << "\" > \"$dep\"\n";
    script.close();
    ASSERT_EQ(::chmod(compiler.c_str(), 0755), 0);

    // Register this boundary after the executor but before the dependency cache. On the defective
    // ordering, it observes the real batch after that cache has already been destroyed.
    (void)detail::GetExecutor();
    (void)detail::GetExecutorMutex();
    ASSERT_EQ(std::atexit(fail_if_prewarm_reaches_dependency_teardown), 0);

    kernel_prewarm::maybe_launch_prewarm(
        (out_root / std::to_string(build_key) / "kernels").string() + "/", (root / "firmware").string(), build_key, "");
    wait_for_file(active);
}

TEST(KernelPrewarmShutdownTest, ActiveBatchJoinsBeforeLazyBuildDependenciesTeardown) {
    ScopedTempDir tree("ttprewarm_shutdown");
    const fs::path active = tree.path_ / "active";
    const fs::path done = tree.path_ / "done";
    const fs::path release = tree.path_ / "release";
    const fs::path teardown = tree.path_ / "teardown";
    ASSERT_EQ(::mkfifo(release.c_str(), 0600), 0);

    expect_subprocess_success(run_test_subprocess(
        "KernelPrewarmShutdownChild.WritesManifest",
        {
            {kSubprocessModeEnv, "manifest"},
            {"TT_METAL_CACHE", tree.path_.string()},
        }));

    const pid_t pid = ::fork();
    ASSERT_GE(pid, 0) << "fork failed";
    if (pid == 0) {
        const std::vector<std::pair<std::string, std::string>> environment = {
            {kSubprocessModeEnv, "shutdown"},
            {"TT_METAL_CACHE", tree.path_.string()},
            {kShutdownActiveEnv, active.string()},
            {kShutdownDoneEnv, done.string()},
            {kShutdownReleaseEnv, release.string()},
            {kShutdownTeardownEnv, teardown.string()},
        };
        for (const auto& [name, value] : environment) {
            if (::setenv(name.c_str(), value.c_str(), /*overwrite=*/1) != 0) {
                std::_Exit(120);
            }
        }
        const std::string filter_arg =
            "--gtest_filter=KernelPrewarmShutdownChild.StartsRealBatchAndReturnsWithoutBarrier";
        char* const args[] = {const_cast<char*>("/proc/self/exe"), const_cast<char*>(filter_arg.c_str()), nullptr};
        ::execv("/proc/self/exe", args);
        std::_Exit(121);
    }

    wait_for_file(active);
    int status = 0;
    bool child_exited = false;
    const auto teardown_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!fs::exists(teardown) && std::chrono::steady_clock::now() < teardown_deadline) {
        const pid_t result = ::waitpid(pid, &status, WNOHANG);
        ASSERT_NE(result, -1) << "waitpid failed";
        if (result == pid) {
            child_exited = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    {
        std::ofstream unblock(release);
        unblock << "continue\n";
    }
    if (!child_exited) {
        ASSERT_EQ(::waitpid(pid, &status, 0), pid);
    }

    ASSERT_TRUE(WIFEXITED(status)) << "shutdown subprocess terminated abnormally";
    EXPECT_EQ(WEXITSTATUS(status), 0) << "active prewarm crossed the lazy-build dependency teardown boundary";
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

TEST_F(IsolatedKernelCacheMeshDeviceFixture, OfflinePrewarmReflectsEditedKernelBodyChild) {
    auto* device = this->devices_.at(0)->get_devices().at(0);
    const auto& build_env =
        BuildEnvManager::get_instance(extract_context_id(device)).get_device_build_env(device->build_id()).build_env;

    ScopedTempDir source_dir("ttprewarm_probe");
    const fs::path kernel_path = source_dir.path_ / (source_dir.path_.filename().string() + ".cpp");
    const fs::path kernel_subdir = fs::path(build_env.get_out_kernel_root_path()) / kernel_path.stem().string();

    auto compile_probe = [&]() {
        Program program = CreateProgram();
        CreateKernel(program, kernel_path.string(), CoreCoord{0, 0}, kReaderDmConfig);
        detail::CompileProgram(device, program);
    };

    write_probe_kernel(kernel_path, kProbeTagV1);
    compile_probe();
    const std::string elf_v1 = read_kernel_elf_bytes(kernel_subdir);
    ASSERT_FALSE(elf_v1.empty()) << "no kernel .elf produced under " << kernel_subdir;
    ASSERT_TRUE(blob_contains(elf_v1, kProbeTagV1));
    ASSERT_FALSE(blob_contains(elf_v1, kProbeTagV2));

    // The body edit keeps the path and compile arguments stable, so the manifest key and kernel hash
    // are unchanged. Offline prewarm must compile the current source instead of a captured snapshot.
    write_probe_kernel(kernel_path, kProbeTagV2);
    kernel_prewarm::wait_for_prewarm();
    const std::size_t built = kernel_prewarm::prewarm_manifest_offline(
        build_env.get_out_root_path(), with_trailing_slash(build_env.get_root_path()));
    ASSERT_GT(built, 0u) << "offline prewarm built nothing";

    const std::string elf_prewarm = read_kernel_elf_bytes(kernel_subdir);
    EXPECT_TRUE(blob_contains(elf_prewarm, kProbeTagV2)) << "prewarm did not reflect the edited body";
    EXPECT_FALSE(blob_contains(elf_prewarm, kProbeTagV1)) << "prewarm served the stale kernel body";
}

TEST(KernelPrewarmIsolationTest, OfflinePrewarmReflectsEditedKernelBody) {
    fs::path tree_path;
    int status = -1;
    {
        ScopedTempDir tree("ttprewarm_offline_isolated");
        tree_path = tree.path_;
        const fs::path cache = tree.path_ / "cache";
        fs::create_directories(cache);
        status = run_test_subprocess(
            "IsolatedKernelCacheMeshDeviceFixture.OfflinePrewarmReflectsEditedKernelBodyChild",
            {
                {kSubprocessModeEnv, "offline"},
                {"TT_METAL_CACHE", cache.string()},
                {"TT_METAL_KERNEL_PREWARM", "1"},
                {"TT_METAL_SLOW_DISPATCH_MODE", "1"},
            });
        EXPECT_TRUE(isolated_cache_populated(tree_path / "cache"))
            << "child did not compile into the isolated cache under " << (tree_path / "cache");
    }
    EXPECT_FALSE(fs::exists(tree_path));
    expect_subprocess_success(status);
}

TEST_F(IsolatedKernelCacheMeshDeviceFixture, EditedKernelBodyForcesRecompileNotStaleCacheHitChild) {
    auto* device = this->devices_.at(0)->get_devices().at(0);
    const auto& build_env =
        BuildEnvManager::get_instance(extract_context_id(device)).get_device_build_env(device->build_id()).build_env;

    ScopedTempDir source_dir("ttdephash_probe");
    const fs::path kernel_path = source_dir.path_ / (source_dir.path_.filename().string() + ".cpp");
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

    // Clearing in-memory dedup models a fresh process. The dependency hash must reject the v1
    // artifacts on disk after the source body changes.
    write_probe_kernel(kernel_path, kProbeTagV2);
    jit_build_cache_clear();
    compile_probe();

    const std::string elf_v2 = read_kernel_elf_bytes(kernel_subdir);
    EXPECT_TRUE(blob_contains(elf_v2, kProbeTagV2)) << "recompiled binary does not reflect the edit";
    EXPECT_FALSE(blob_contains(elf_v2, kProbeTagV1)) << "stale kernel body survived the edit";
}

TEST(KernelPrewarmIsolationTest, EditedKernelBodyForcesRecompileNotStaleCacheHit) {
    fs::path tree_path;
    int status = -1;
    {
        ScopedTempDir tree("ttdephash_isolated");
        tree_path = tree.path_;
        const fs::path cache = tree.path_ / "cache";
        fs::create_directories(cache);
        status = run_test_subprocess(
            "IsolatedKernelCacheMeshDeviceFixture.EditedKernelBodyForcesRecompileNotStaleCacheHitChild",
            {
                {kSubprocessModeEnv, "dephash"},
                {"TT_METAL_CACHE", cache.string()},
                {"TT_METAL_KERNEL_PREWARM", "1"},
                {"TT_METAL_SLOW_DISPATCH_MODE", "1"},
            });
        EXPECT_TRUE(isolated_cache_populated(tree_path / "cache"))
            << "child did not compile into the isolated cache under " << (tree_path / "cache");
    }
    EXPECT_FALSE(fs::exists(tree_path));
    expect_subprocess_success(status);
}

}  // namespace tt::tt_metal
