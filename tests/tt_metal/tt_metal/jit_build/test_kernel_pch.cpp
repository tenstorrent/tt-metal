// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A kernel's precompiled header (DataMovementConfig::precompiled_header): kernels that share every input
// of the PCH share one, kernels that differ in an input get their own, and each kernel reads its own CB
// tables.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

#include <fmt/format.h>
#include <hostdevcommon/kernel_structs.h>
#include <tt-metalium/base_types.hpp>

#include "jit_build/build.hpp"
#include "jit_build/genfiles.hpp"
#include "jit_build/jit_build_options.hpp"
#include "jit_build/jit_build_settings.hpp"
#include "jit_build/jit_build_utils.hpp"
#include "jit_build/jit_device_config.hpp"
#include "jit_build/pch.hpp"
#include "llrt/hal.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal {
namespace {

namespace fs = std::filesystem;

std::string read_file_text(const fs::path& path) {
    std::ifstream file(path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

class ProbeSettings : public JitBuildSettings {
public:
    std::string name = "probe";
    std::string opt_level = "Os";
    std::map<std::string, std::string> defines;
    std::unordered_map<std::string, uint32_t> named_args;
    std::vector<std::string> include_paths;
    std::string header;

    const std::string& get_full_kernel_name() const override { return name; }
    std::string_view get_compiler_opt_level() const override { return opt_level; }
    std::string_view get_linker_opt_level() const override { return opt_level; }
    void process_defines(std::function<void(const std::string&, const std::string&)> f) const override {
        for (const auto& [define, value] : defines) {
            f(define, value);
        }
    }
    void process_compile_time_args(std::function<void(const std::vector<uint32_t>&)>) const override {}
    void process_named_compile_time_args(
        std::function<void(const std::unordered_map<std::string, uint32_t>&)> f) const override {
        f(named_args);
    }
    void process_include_paths(const std::function<void(const std::string&)>& f) const override {
        for (const auto& path : include_paths) {
            f(path);
        }
    }
    std::string get_precompiled_header() const override { return header; }
};

// Compiles a Tensix processor's kernel entry source (brisck.cc, trisck.cc) for a kernel whose generated
// files are in a directory.
class KernelCompileProbe : public JitBuildState {
public:
    KernelCompileProbe(const JitBuildEnv& env, const Hal& hal, HalProcessorClassType processor_class, int processor_id) :
        JitBuildState(
            env,
            {.core_type = HalProgrammableCoreType::TENSIX,
             .processor_class = processor_class,
             .processor_id = processor_id,
             .is_fw = false},
            hal) {
        cflags_ += " -H";
    }

    // Returns the PCH the compile loaded; GCC's -H output marks it with "! ".
    std::string compile(const fs::path& kernel_dir, const ProbeSettings& settings) const {
        const fs::path out_dir = kernel_dir / get_target_name();
        fs::create_directories(out_dir);
        compile_one(out_dir.string() + "/", &settings, 0);
        std::istringstream lines(read_file_text(out_dir / (objs_[0] + ".log")));
        for (std::string line; std::getline(lines, line);) {
            if (line.starts_with("! ") && line.ends_with(".gch")) {
                return line.substr(2);
            }
        }
        return "";
    }
};

bool is_kernel_pch(const std::string& pch) { return fs::path(pch).filename().string().starts_with("kernel_pch.h"); }

// Kernel build environments for one device, caching under their own directory, and a precompiled
// header that reads the CB tables the way the contract allows: inside a template.
// Build it in a death-test child: it sets TT_METAL_CACHE, and the JIT memoizes PCHs per process.
struct KernelPchHarness {
    fs::path root;
    llrt::RunTimeOptions options;
    Hal hal{tt::ARCH::BLACKHOLE, false, false, 0, false};
    JitBuildEnv env;
    ProbeSettings settings;

    explicit KernelPchHarness(const fs::path& root) : root(root) {
        init_env(env, options, 1);
        settings.header = (root / "probe_pch.h").string();
        std::ofstream(settings.header) << "#pragma once\n"
                                          "template <typename = void>\n"
                                          "constexpr DataFormat probe_cb0_format() {\n"
                                          "    return static_cast<DataFormat>(unpack_src_format[0]);\n"
                                          "}\n";
    }

    void init_env(JitBuildEnv& build_env, const llrt::RunTimeOptions& build_options, uint64_t build_key) const {
        const JitDeviceConfig config{.hal = &hal, .arch = tt::ARCH::BLACKHOLE, .max_dfbs = 32};
        // The defines a device gives every kernel (initialize_device_kernel_defines).
        build_env.init(
            build_key,
            config,
            build_options,
            {{"NUM_DRAM_BANKS", "8"},
             {"NUM_L1_BANKS", "128"},
             {"LOG_BASE_2_OF_NUM_DRAM_BANKS", "3"},
             {"LOG_BASE_2_OF_NUM_L1_BANKS", "7"},
             {"PCIE_NOC_X", "0"},
             {"PCIE_NOC_Y", "0"}});
    }

    KernelCompileProbe probe(HalProcessorClassType processor_class, int processor_id = 0) const {
        return KernelCompileProbe(env, hal, processor_class, processor_id);
    }

    // Writes a kernel's generated files. The kernel includes the precompiled header first and asserts,
    // at compile time, the CB 0 format that its own table holds.
    fs::path make_kernel(
        const std::string& name,
        DataFormat format,
        const std::string& format_name,
        MathFidelity fidelity = MathFidelity::HiFi4) const {
        const fs::path dir = root / name;
        fs::create_directories(dir);
        JitBuildOptions build_options(env);
        build_options.path = dir.string() + "/";
        build_options.cb_tables_for_pch = true;
        build_options.set_cb_dataformat_all_cores(tt::CBIndex::c_0, format);
        build_options.set_hlk_math_fidelity_all_cores(fidelity);
        jit_build_genfiles_descriptors(env, build_options);
        const std::string kernel = "#include \"" + settings.header +
                                   "\"\n"
                                   "void kernel_main() { static_assert(probe_cb0_format() == DataFormat::" +
                                   format_name + "); }\n";
        std::ofstream(dir / "kernel_includes.hpp") << kernel;
        std::ofstream(dir / "defines_generated.h") << "#pragma once\n";
        for (const auto& [thread, define] :
             {std::pair{"unpack", "TRISC_UNPACK"}, std::pair{"math", "TRISC_MATH"}, std::pair{"pack", "TRISC_PACK"}}) {
            std::ofstream(dir / fmt::format("chlkc_{}.cpp", thread))
                << "#define " << define << "\n#include \"defines_generated.h\"\n"
                << kernel;
        }
        return dir;
    }
};

void run_in_child(const std::function<void(KernelPchHarness&)>& body) {
    auto pattern = (fs::temp_directory_path() / "kernel_pch_XXXXXX").string();
    const auto* directory = mkdtemp(pattern.data());
    ASSERT_NE(directory, nullptr);
    const fs::path root = directory;
    const auto child = [&] {
        setenv("TT_METAL_CACHE", (root.string() + "/").c_str(), 1);
        KernelPchHarness harness(root);
        body(harness);
        // An ASSERT failure in body returns early; exit with a code the parent reports as a failure.
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    fs::remove_all(root);
}

// Every data-movement kernel gets its NoC index and mode.
void set_dm_defines(ProbeSettings& settings) { settings.defines = {{"NOC_INDEX", "0"}, {"NOC_MODE", "0"}}; }

TEST(KernelPchDeathTest, SecondKernelBuildsPchSharedAcrossCbFormats) {
    run_in_child([](KernelPchHarness& h) {
        set_dm_defines(h.settings);
        const auto brisc = h.probe(HalProcessorClassType::DM);
        const auto bf16 = h.make_kernel("bf16", DataFormat::Float16_b, "Float16_b");
        const auto first = brisc.compile(bf16, h.settings);
        ASSERT_FALSE(first.empty());
        ASSERT_FALSE(is_kernel_pch(first)) << "the first kernel of a PCH builds none: " << first;

        const auto fp32 = h.make_kernel("fp32", DataFormat::Float32, "Float32");
        const auto second = brisc.compile(fp32, h.settings);
        ASSERT_FALSE(is_kernel_pch(second)) << "the second kernel starts the build and parses: " << second;
        jit_build::wait_for_kernel_pch_builds();
        const auto shared = brisc.compile(fp32, h.settings);
        ASSERT_TRUE(is_kernel_pch(shared)) << shared;
        ASSERT_EQ(brisc.compile(bf16, h.settings), shared);

        // The PCH's headers are in the kernel's dependency hash, so editing one rebuilds the kernel.
        bool lists_header = false;
        for (const auto& entry : fs::directory_iterator(h.root / "fp32" / "brisc")) {
            if (entry.path().extension() == ".dephash") {
                lists_header |= read_file_text(entry.path()).find("probe_pch.h") != std::string::npos;
            }
        }
        ASSERT_TRUE(lists_header);
    });
}

TEST(KernelPchDeathTest, DefinesOptLevelIncludesAndFlagsSeparatePch) {
    run_in_child([](KernelPchHarness& h) {
        set_dm_defines(h.settings);
        const auto kernel = h.make_kernel("kernel", DataFormat::Float32, "Float32");
        const auto with_pch = [&](const KernelCompileProbe& probe) {
            probe.compile(kernel, h.settings);
            probe.compile(kernel, h.settings);
            jit_build::wait_for_kernel_pch_builds();
            return probe.compile(kernel, h.settings);
        };
        const auto brisc = h.probe(HalProcessorClassType::DM);
        const auto plain = with_pch(brisc);
        h.settings.defines["PROBE_DEFINE"] = "1";
        const auto defined = with_pch(brisc);
        h.settings.defines.erase("PROBE_DEFINE");
        h.settings.opt_level = "O2";
        const auto o2 = with_pch(brisc);
        h.settings.opt_level = "Os";
        h.settings.include_paths = {h.root.string()};
        const auto included = with_pch(brisc);
        h.settings.include_paths.clear();
        // Compiler flags without -g.
        llrt::RunTimeOptions no_debug_info;
        no_debug_info.set_riscv_debug_info_enabled(false);
        JitBuildEnv no_debug_env;
        h.init_env(no_debug_env, no_debug_info, 2);
        const auto no_g = with_pch(KernelCompileProbe(no_debug_env, h.hal, HalProcessorClassType::DM, 0));

        const std::set<std::string> pchs = {plain, defined, o2, included, no_g};
        ASSERT_EQ(pchs.size(), 5u);
        for (const auto& pch : pchs) {
            ASSERT_TRUE(is_kernel_pch(pch)) << pch;
        }
    });
}

TEST(KernelPchDeathTest, TriscThreadsAndComputeConfigSeparatePch) {
    run_in_child([](KernelPchHarness& h) {
        const auto bf16 = h.make_kernel("bf16", DataFormat::Float16_b, "Float16_b");
        const auto fp32 = h.make_kernel("fp32", DataFormat::Float32, "Float32");
        const auto lofi = h.make_kernel("lofi", DataFormat::Float32, "Float32", MathFidelity::LoFi);
        std::set<std::string> thread_pchs;
        for (int thread = 0; thread < 3; ++thread) {
            const auto trisc = h.probe(HalProcessorClassType::COMPUTE, thread);
            trisc.compile(bf16, h.settings);
            trisc.compile(fp32, h.settings);
            jit_build::wait_for_kernel_pch_builds();
            const auto pch = trisc.compile(fp32, h.settings);
            ASSERT_TRUE(is_kernel_pch(pch)) << pch;
            ASSERT_EQ(trisc.compile(bf16, h.settings), pch) << "CB formats must not split a thread's PCH";
            thread_pchs.insert(pch);
            if (thread == 1) {  // The math thread reads the math fidelity.
                trisc.compile(lofi, h.settings);
                trisc.compile(lofi, h.settings);
                jit_build::wait_for_kernel_pch_builds();
                const auto lofi_pch = trisc.compile(lofi, h.settings);
                ASSERT_TRUE(is_kernel_pch(lofi_pch)) << lofi_pch;
                ASSERT_NE(lofi_pch, pch);
            }
        }
        ASSERT_EQ(thread_pchs.size(), 3u);
    });
}

// A named map is force-included after the PCH, too late for compile_time_args.h inside it; an
// unreadable header compiles as if none were named.
TEST(KernelPchDeathTest, NoKernelPchWithoutReadableHeaderOrWithNamedArgMapOrProfiler) {
    run_in_child([](KernelPchHarness& h) {
        set_dm_defines(h.settings);
        const auto brisc = h.probe(HalProcessorClassType::DM);
        const auto kernel = h.make_kernel("kernel", DataFormat::Float32, "Float32");
        const auto twice = [&] {
            brisc.compile(kernel, h.settings);
            return brisc.compile(kernel, h.settings);
        };
        h.settings.named_args = {{"probe_arg", 1}};
        std::ofstream(kernel / std::string(jit_build::utils::NAMED_CT_ARG_MAP_HEADER))
            << jit_build::utils::format_named_ct_arg_map_header(h.settings.named_args);
        const auto named = twice();
        h.settings.named_args.clear();
        std::ofstream(kernel / "kernel_includes.hpp") << "void kernel_main() {}\n";
        // A header this machine cannot read, as on a compile server on another host.
        h.settings.header = (h.root / "absent.h").string();
        const auto unreadable = twice();
        h.settings.header.clear();
        const auto no_header = twice();
        // The device profiler takes zone records from each compile's own log.
        h.settings.header = (h.root / "probe_pch.h").string();
        std::ofstream(kernel / "kernel_includes.hpp")
            << "#include \"" << h.settings.header << "\"\nvoid kernel_main() {}\n";
        llrt::RunTimeOptions profiling;
        profiling.set_profiler_enabled(true);
        JitBuildEnv profiling_env;
        h.init_env(profiling_env, profiling, 2);
        const KernelCompileProbe profiled(profiling_env, h.hal, HalProcessorClassType::DM, 0);
        profiled.compile(kernel, h.settings);
        const auto profiler = profiled.compile(kernel, h.settings);
        for (const auto& pch : {named, unreadable, no_header, profiler}) {
            ASSERT_FALSE(pch.empty());
            ASSERT_FALSE(is_kernel_pch(pch)) << pch;
        }
        for (const auto& entry : fs::recursive_directory_iterator(h.root)) {
            ASSERT_FALSE(entry.path().filename().string().starts_with("kernel-")) << "PCH attempted: " << entry.path();
        }
    });
}

// When GCC rejects a PCH, the compile parses its headers instead, with the kernel's own CB tables.
TEST(KernelPchDeathTest, RejectedPchFallsBackToParsing) {
    run_in_child([](KernelPchHarness& h) {
        set_dm_defines(h.settings);
        const auto brisc = h.probe(HalProcessorClassType::DM);
        const auto bf16 = h.make_kernel("bf16", DataFormat::Float16_b, "Float16_b");
        brisc.compile(bf16, h.settings);
        brisc.compile(bf16, h.settings);
        jit_build::wait_for_kernel_pch_builds();
        const auto pch = brisc.compile(bf16, h.settings);
        ASSERT_TRUE(is_kernel_pch(pch)) << pch;
        std::ofstream(pch, std::ios::trunc) << "not a precompiled header";
        const auto loaded = brisc.compile(h.make_kernel("fp32", DataFormat::Float32, "Float32"), h.settings);
        ASSERT_TRUE(loaded.empty()) << loaded;
    });
}

// Compiles run in parallel, as on a compile server: the second request starts the one PCH build, no
// compile waits for it, and compiles that start once it is ready load it.
TEST(KernelPchDeathTest, ConcurrentCompilesBuildOnePch) {
    run_in_child([](KernelPchHarness& h) {
        set_dm_defines(h.settings);
        const auto brisc = h.probe(HalProcessorClassType::DM);
        constexpr int kernels = 8;
        std::vector<fs::path> dirs;
        for (int i = 0; i < kernels; ++i) {
            dirs.push_back(h.make_kernel(fmt::format("k{}", i), DataFormat::Float32, "Float32"));
        }
        std::vector<std::string> loaded(kernels);
        std::vector<std::thread> threads;
        for (int i = 0; i < kernels; ++i) {
            threads.emplace_back([&, i] { loaded[i] = brisc.compile(dirs[i], h.settings); });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        ASSERT_LE(std::ranges::count_if(loaded, is_kernel_pch), kernels - 2) << "the first two requests load none";
        jit_build::wait_for_kernel_pch_builds();
        const auto later = brisc.compile(dirs[0], h.settings);
        ASSERT_TRUE(is_kernel_pch(later)) << later;
        std::set<std::string> pchs = {later};
        std::ranges::copy_if(loaded, std::inserter(pchs, pchs.end()), is_kernel_pch);
        ASSERT_EQ(pchs.size(), 1u);
        int built = 0;
        for (const auto& entry : fs::recursive_directory_iterator(h.root)) {
            built += entry.path().filename() == "kernel_pch.h.gch";
        }
        ASSERT_EQ(built, 1);
    });
}

// A compile server gets the exported recipe and the kernel's generated files in its own cache, and
// must reach the PCH key a local compile does.
TEST(KernelPchDeathTest, ServerCacheReachesLocalPchKey) {
    run_in_child([](KernelPchHarness& h) {
        set_dm_defines(h.settings);
        const auto brisc = h.probe(HalProcessorClassType::DM);
        const auto kernel = h.make_kernel("kernel", DataFormat::Float32, "Float32");
        brisc.compile(kernel, h.settings);
        brisc.compile(kernel, h.settings);
        jit_build::wait_for_kernel_pch_builds();
        const auto local = brisc.compile(kernel, h.settings);
        ASSERT_TRUE(is_kernel_pch(local)) << local;

        const fs::path server_kernel = h.root / "server" / "kernel";
        fs::create_directories(server_kernel);
        for (const auto& entry : fs::directory_iterator(kernel)) {
            if (entry.is_regular_file()) {
                fs::copy_file(entry.path(), server_kernel / entry.path().filename());
            }
        }
        const auto recipe = brisc.export_target_recipe(&h.settings);
        const auto server_pch_root = h.root / "server" / "pch";
        jit_build::select_pch(h.env.get_gpp(), recipe, server_kernel, server_pch_root);
        jit_build::select_pch(h.env.get_gpp(), recipe, server_kernel, server_pch_root);
        jit_build::wait_for_kernel_pch_builds();
        const auto server = jit_build::select_pch(h.env.get_gpp(), recipe, server_kernel, server_pch_root);
        ASSERT_EQ(server.includes.size(), 4u);
        ASSERT_EQ(fs::path(server.includes[1]).parent_path().filename(), fs::path(local).parent_path().filename());
        ASSERT_EQ(fs::path(server.includes[1]).parent_path().parent_path(), server_pch_root);
    });
}

}  // namespace
}  // namespace tt::tt_metal
