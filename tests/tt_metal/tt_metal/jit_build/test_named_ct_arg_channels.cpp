// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/host_api.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "jit_build/build.hpp"
#include "jit_build/build_env_manager.hpp"
#include "jit_build/depend.hpp"
#include "jit_build/jit_build_utils.hpp"
#include "mock_blackhole_fixture.hpp"

namespace tt::tt_metal {

// This fixture has external linkage because gtest's generated TEST_F classes derive from it.
class NamedCtArgChannelsMockBlackholeFixture : public MockBlackholeMeshDispatchFixture {
protected:
    struct NamedCtArtifacts {
        bool legacy_header;
        bool blaze_header;
        bool legacy_force_include;
    };

    // The PCH an object was built with, read from the object's dependency hashes.
    static std::filesystem::path pch_of(const std::filesystem::path& object) {
        std::ifstream hashes(object.string() + ".dephash");
        std::filesystem::path dependency;
        for (uint64_t hash; hashes >> dependency >> hash;) {
            if (dependency.filename() == "prefix.hpp") {
                return dependency.string() + ".gch";
            }
        }
        return {};
    }

    std::vector<std::filesystem::path> compile_pch_case(
        uint32_t tag,
        tt::DataFormat format,
        const std::filesystem::path& dependency = {},
        int dependency_value = 0,
        MathFidelity fidelity = MathFidelity::HiFi4) {
        auto* device = devices_.at(0).get();
        ProgramDescriptor descriptor;
        descriptor.cbs.push_back(CBDescriptor{
            .total_size = 4096,
            .core_ranges = CoreRange(CoreCoord{0, 0}),
            .format_descriptors = {{.buffer_index = 0, .data_format = format, .page_size = 2048}},
        });
        KernelDescriptor kernel = {
            .kernel_source =
                "static_assert(sizeof(FULL_KERNEL_NAME) > 10); "
                "void kernel_main() { asm volatile(\"\" :: \"r\"(blaze_ct_args::typed::value)); }",
            .source_type = KernelDescriptor::SourceType::SOURCE_CODE,
            .core_ranges = CoreRange(CoreCoord{0, 0}),
            .blaze_named_args = {.named_compile_time_args = {{"typed.value", tag}}},
            .config = DataMovementConfigDescriptor{},
        };
        {
            const auto& env = BuildEnvManager::get_instance(extract_context_id(device))
                                  .get_device_build_env(device->build_id())
                                  .build_env;
            const auto source = std::filesystem::path(env.get_out_kernel_root_path()) / "pch-operation-probe.cpp";
            std::filesystem::create_directories(source.parent_path());
            std::ofstream file(source);
            file << "#ifndef BLAZE_OPERATION_PCH_PREFIX_INCLUDED\n#define BLAZE_OPERATION_PCH_PREFIX_INCLUDED\n";
            // Deliberately unguarded: consuming the PCH must not declare this twice. Its source line must survive.
            file << "struct PchOperationPrefix { static constexpr int value = __LINE__; };\n";
            file << "constexpr int pch_probe_define = PCH_PROBE_DEFINE;\n";
            // The DeepSeek startup helpers read CB formats that the prefix only declares.
            file << "#ifdef COMPILE_FOR_TRISC\n#include \"api/compute/common.h\"\n"
                    "#include \"api/compute/experimental/deepseek_compute_kernel_hw_startup.h\"\n#endif\n";
            if (!dependency.empty()) {
                file << "#include \"" << dependency.string() << "\"\n";
            }
            if (env.get_rtoptions().get_profiler_enabled()) {
                file << "#include \"tools/profiler/kernel_profiler.hpp\"\n";
                file << "inline void pch_profile_probe() { DeviceZoneScopedN(\"pch-operation-zone\"); }\n";
            }
            file << "#endif\n// BLAZE_OPERATION_PCH_END\nstatic_assert(PchOperationPrefix::value == 3 && "
                    "pch_probe_define == PCH_PROBE_DEFINE);\n";
            file << "static_assert(unpack_src_format[0] == " << static_cast<uint32_t>(format) << ");\n";
            file << "#ifndef COMPILE_FOR_TRISC\nstatic_assert(get_tile_size(0) > 0 && (uint32_t)get_dataformat(0) == "
                 << static_cast<uint32_t>(format) << ");\n#endif\n";
            file << "#ifdef UCK_CHLKC_MATH\nstatic_assert(MATH_FIDELITY == static_cast<ckernel::MathFidelity>("
                 << static_cast<uint32_t>(fidelity) << "));\n#endif\n";
            if (dependency_value != 0) {
                file << "static_assert(pch_dependency_value == " << dependency_value << ");\n";
            }
            file << kernel.kernel_source;
            file.close();
            kernel.kernel_source = source.string();
            kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
            kernel.defines.emplace_back("BLAZE_OPERATION_PCH_SOURCE", source.string());
            kernel.defines.emplace_back("PCH_PROBE_DEFINE", std::to_string(tag / 100));  // Read by the prefix.
        }
        descriptor.kernels.push_back(kernel);
        kernel.config =
            DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default};
        descriptor.kernels.push_back(kernel);
        kernel.config = ComputeConfigDescriptor{.math_fidelity = fidelity};
        descriptor.kernels.push_back(kernel);
        Program program(descriptor);
        program.impl().compile(device);
        const auto first = program.impl().get_kernel(0);
        const auto& env =
            BuildEnvManager::get_instance(first->get_context_id()).get_device_build_env(device->build_id()).build_env;
        std::vector<std::filesystem::path> kernel_dirs;
        for (size_t i = 0; i < descriptor.kernels.size(); ++i) {
            kernel_dirs.push_back((std::filesystem::path(env.get_out_kernel_root_path()) /
                                   program.impl().get_kernel(i)->get_full_kernel_name())
                                      .parent_path());
        }
        return kernel_dirs;
    }

    NamedCtArtifacts compile_and_inspect(
        KernelDescriptor::NamedCompileTimeArgs legacy_args,
        experimental::blaze::NamedCompileTimeArgs blaze_args,
        const std::string& source) {
        auto* device = devices_.at(0).get();
        KernelDescriptor kernel_descriptor = {
            .kernel_source = source,
            .source_type = KernelDescriptor::SourceType::SOURCE_CODE,
            .core_ranges = CoreRange(CoreCoord{0, 0}),
            .named_compile_time_args = std::move(legacy_args),
            .blaze_named_args = {.named_compile_time_args = std::move(blaze_args)},
            .config = DataMovementConfigDescriptor{},
        };
        Program program(ProgramDescriptor{.kernels = {kernel_descriptor}});
        const auto kernel = program.impl().get_kernel(0);
        program.impl().compile(device);

        auto& build_env_manager = BuildEnvManager::get_instance(kernel->get_context_id());
        const auto recipe =
            kernel_build_state(*kernel, kernel->get_kernel_processor_type(0)).export_target_recipe(kernel.get());

        bool legacy_force_include = false;
        for (std::size_t i = 1; i < recipe.defines.size(); ++i) {
            legacy_force_include |=
                recipe.defines[i - 1] == "-include" && recipe.defines[i] == jit_build::utils::NAMED_CT_ARG_MAP_HEADER;
        }

        const std::filesystem::path kernel_dir =
            std::filesystem::path(
                build_env_manager.get_device_build_env(device->build_id()).build_env.get_out_kernel_root_path()) /
            kernel->get_full_kernel_name();
        return {
            .legacy_header = std::filesystem::exists(kernel_dir / jit_build::utils::NAMED_CT_ARG_MAP_HEADER),
            .blaze_header = std::filesystem::exists(kernel_dir / "named_args_generated.h"),
            .legacy_force_include = legacy_force_include,
        };
    }
};

TEST_F(NamedCtArgChannelsMockBlackholeFixture, BlazeOnlyOmitsLegacyMap) {
    const auto artifacts = compile_and_inspect({}, {{"typed.value", 1}}, R"(
#include "api/dataflow/dataflow_api.h"
#ifdef KERNEL_COMPILE_TIME_ARG_MAP
#error "Blaze-only kernels must not receive the legacy named CT map"
#endif
static_assert(blaze_ct_args::typed::value == 1);
void kernel_main() {}
)");

    EXPECT_FALSE(artifacts.legacy_header);
    EXPECT_TRUE(artifacts.blaze_header);
    EXPECT_FALSE(artifacts.legacy_force_include);
}

TEST_F(NamedCtArgChannelsMockBlackholeFixture, LegacyFieldPreservesBothApis) {
    const auto artifacts = compile_and_inspect({{"legacy_value", 2}, {"legacy_value", 2}, {"legacy.scoped", 3}}, {}, R"(
#include "api/dataflow/dataflow_api.h"
static_assert(get_named_compile_time_arg_val("legacy_value") == 2);
static_assert(get_named_compile_time_arg_val("legacy.scoped") == 3);
static_assert(blaze_ct_args::legacy_value == 2);
static_assert(blaze_ct_args::legacy::scoped == 3);
void kernel_main() {}
)");

    EXPECT_TRUE(artifacts.legacy_header);
    EXPECT_TRUE(artifacts.blaze_header);
    EXPECT_TRUE(artifacts.legacy_force_include);
}

TEST_F(NamedCtArgChannelsMockBlackholeFixture, MixedEmitsBothRepresentations) {
    const auto artifacts = compile_and_inspect(
        {{"legacy.value", 3}, {"legacy.value", 3}, {"legacy.only", 6}},
        {{"typed.only", 5}},
        R"(
#include "api/dataflow/dataflow_api.h"
static_assert(get_named_compile_time_arg_val("legacy.value") == 3);
static_assert(get_named_compile_time_arg_val("legacy.only") == 6);
static_assert(blaze_ct_args::typed::only == 5);
static_assert(blaze_ct_args::legacy::value == 3);
static_assert(blaze_ct_args::legacy::only == 6);
static_assert(sizeof(named_args_map) / sizeof(named_args_map[0]) == 2);
void kernel_main() {}
)");

    EXPECT_TRUE(artifacts.legacy_header);
    EXPECT_TRUE(artifacts.blaze_header);
    EXPECT_TRUE(artifacts.legacy_force_include);
}

TEST_F(NamedCtArgChannelsMockBlackholeFixture, MixedDataMovementKernelCompiles) {
    const CoreCoord core{0, 0};
    KernelDescriptor kernel = {
        .kernel_source = "tests/tt_metal/tt_metal/test_kernels/misc/blaze_named_runtime_args_kernel.cpp",
        .core_ranges = CoreRange(core),
        .named_compile_time_args = {{"legacy_param", 0xCAFE}},
        .defines = {{"WRITE_ADDRESS", "0"}, {"TEST_LEGACY_NAMED_CT_ARGS", "1"}},
        .blaze_named_args =
            {
                .named_compile_time_args = {{"my_kernel.param_a", 42}, {"my_kernel.param_b", 0xBEEF}},
                .named_common_runtime_args = {{"my_kernel.marker", 0}},
                .named_per_core_runtime_args = {{"my_kernel.core_idx", {{core, 0}}}},
            },
        .config = DataMovementConfigDescriptor{},
    };
    Program program(ProgramDescriptor{.kernels = {kernel}});
    program.impl().compile(devices_.at(0).get());
}

TEST_F(NamedCtArgChannelsMockBlackholeFixture, LegacyBlazeKernelsCompile) {
    const CoreCoord core{0, 0};
    KernelDescriptor data_movement = {
        .kernel_source = "tests/tt_metal/tt_metal/test_kernels/misc/blaze_named_runtime_args_kernel.cpp",
        .core_ranges = CoreRange(core),
        .named_compile_time_args = {{"my_kernel.param_a", 42}, {"my_kernel.param_b", 0xBEEF}},
        .defines = {{"WRITE_ADDRESS", "0"}},
        .blaze_named_args =
            {
                .named_common_runtime_args = {{"my_kernel.marker", 0}},
                .named_per_core_runtime_args = {{"my_kernel.core_idx", {{core, 0}}}},
            },
        .config = DataMovementConfigDescriptor{},
    };
    KernelDescriptor compute = {
        .kernel_source =
            "tests/tt_metal/tt_metal/test_kernels/compute/blaze_named_compile_time_args_compute_kernel.cpp",
        .core_ranges = CoreRange(core),
        .named_compile_time_args = {{"my_kernel.param_a", 42}, {"my_kernel.param_b", 0xBEEF}, {"legacy_param", 0xCAFE}},
        .defines = {{"WRITE_ADDRESS", "0"}},
        .config = ComputeConfigDescriptor{},
    };
    Program program(ProgramDescriptor{.kernels = {data_movement, compute}});
    program.impl().compile(devices_.at(0).get());
}

// No API includes: the kernel wrapper must supply the accessors to all three TRISCs.
TEST_F(NamedCtArgChannelsMockBlackholeFixture, LegacyComputeArgsWithoutApiIncludes) {
    Program program = CreateProgram();
    CreateKernelFromString(
        program,
        R"(
static_assert(get_compile_time_arg_val(0) == 17);
static_assert(get_compile_time_arg_val(1) == 23);
static_assert(get_named_compile_time_arg_val("value") == 42);
void kernel_main() {}
)",
        CoreCoord{0, 0},
        ComputeConfig{.compile_args = {17, 23}, .named_compile_args = {{"value", 42}}});
    program.impl().compile(devices_.at(0).get());
}

TEST_F(NamedCtArgChannelsMockBlackholeFixture, Metal2ComputeArgsWithoutApiIncludes) {
    KernelDescriptor kernel = {
        .kernel_source = R"(
static_assert(get_compile_time_arg_val(0) == 17);
static_assert(get_compile_time_arg_val(1) == 23);
static_assert(get_named_compile_time_arg_val("value") == 42);
void kernel_main() {}
)",
        .source_type = KernelDescriptor::SourceType::SOURCE_CODE,
        .core_ranges = CoreRange(CoreCoord{0, 0}),
        .compile_time_args = {17, 23},
        .named_compile_time_args = {{"value", 42}},
        .config = ComputeConfigDescriptor{},
    };
    Program program(ProgramDescriptor{.kernels = {kernel}});
    program.impl().compile(devices_.at(0).get());
}

TEST_F(NamedCtArgChannelsMockBlackholeFixture, LegacyConflictingValuesFail) {
    KernelDescriptor kernel = {
        .kernel_source = "void kernel_main() {}",
        .source_type = KernelDescriptor::SourceType::SOURCE_CODE,
        .core_ranges = CoreRange(CoreCoord{0, 0}),
        .named_compile_time_args = {{"legacy_value", 1}, {"legacy_value", 2}},
        .config = DataMovementConfigDescriptor{},
    };
    EXPECT_THROW(Program program(ProgramDescriptor{.kernels = {kernel}}), std::runtime_error);
}

TEST_F(NamedCtArgChannelsMockBlackholeFixture, BlazeDuplicateNamesFail) {
    KernelDescriptor kernel = {
        .kernel_source = "void kernel_main() {}",
        .source_type = KernelDescriptor::SourceType::SOURCE_CODE,
        .core_ranges = CoreRange(CoreCoord{0, 0}),
        .blaze_named_args = {.named_compile_time_args = {{"typed.value", 1}, {"typed.value", 1}}},
        .config = DataMovementConfigDescriptor{},
    };
    EXPECT_THROW(Program program(ProgramDescriptor{.kernels = {kernel}}), std::runtime_error);
    kernel.blaze_named_args.named_compile_time_args = {{"typed.value", 1}, {"typed.value", 2}};
    EXPECT_THROW(Program program(ProgramDescriptor{.kernels = {kernel}}), std::runtime_error);
}

TEST_F(NamedCtArgChannelsMockBlackholeFixture, CrossFieldDuplicateFails) {
    KernelDescriptor kernel = {
        .kernel_source = "void kernel_main() {}",
        .source_type = KernelDescriptor::SourceType::SOURCE_CODE,
        .core_ranges = CoreRange(CoreCoord{0, 0}),
        .named_compile_time_args = {{"typed.value", 1}},
        .blaze_named_args = {.named_compile_time_args = {{"typed.value", 1}}},
        .config = DataMovementConfigDescriptor{},
    };
    EXPECT_THROW(Program program(ProgramDescriptor{.kernels = {kernel}}), std::runtime_error);
}

// Turns on operation PCH for this suite; the tests compile with the real RISC-V compiler on a mock device.
class OperationPchMockBlackholeFixture : public NamedCtArgChannelsMockBlackholeFixture {
protected:
    static void SetUpTestSuite() {
        setenv("TT_METAL_BLAZE_OPERATION_PCH", "1", 1);
        NamedCtArgChannelsMockBlackholeFixture::SetUpTestSuite();
    }
    static void TearDownTestSuite() {
        NamedCtArgChannelsMockBlackholeFixture::TearDownTestSuite();
        unsetenv("TT_METAL_BLAZE_OPERATION_PCH");
    }
};

TEST_F(OperationPchMockBlackholeFixture, PchIsSharedAcrossArgsAndCbFormatsButNotDefines) {
    namespace fs = std::filesystem;
    auto root = compile_pch_case(101, tt::DataFormat::Float16_b).front();
    const auto pch_root = root.parent_path().parent_path().parent_path() / "prefix-pch";
    auto pch_files = [&] {
        std::map<fs::path, fs::file_time_type> files;
        for (const auto& entry : fs::recursive_directory_iterator(pch_root)) {
            if (entry.path().extension() == ".gch") {
                files.emplace(entry.path(), fs::last_write_time(entry.path()));
            }
        }
        return files;
    };
    const auto first = pch_files();
    ASSERT_FALSE(first.empty());
    compile_pch_case(102, tt::DataFormat::Float16_b);
    EXPECT_EQ(first, pch_files());  // Different named values and kernel names reuse the prefix.
    const auto kernel_dirs = compile_pch_case(103, tt::DataFormat::Bfp8_b);
    root = kernel_dirs.front();
    EXPECT_EQ(first, pch_files());  // Only constexpr CB definitions change; the declarations stay cached.
    for (const auto& built :
         {kernel_dirs[1] / "ncrisc" / "ncrisck.o",
          kernel_dirs[2] / "trisc0" / "trisck.o",
          kernel_dirs[2] / "trisc1" / "trisck.o",
          kernel_dirs[2] / "trisc2" / "trisck.o"}) {
        EXPECT_FALSE(pch_of(built).empty()) << built;  // Every RISC, not just BRISC, uses a PCH.
    }

    const auto object = root / "brisc" / "brisck.o";
    std::ifstream hashes(object.string() + ".dephash");
    const std::string text{std::istreambuf_iterator<char>(hashes), std::istreambuf_iterator<char>()};
    EXPECT_NE(text.find(pch_root.string()), std::string::npos);  // The object was built with the PCH.
    ASSERT_TRUE(jit_build::dependencies_up_to_date(object.parent_path().string(), object.string()));
    std::ofstream(root / "chlkc_descriptors_declarations.h", std::ios::app) << "\n// changed declarations\n";
    EXPECT_FALSE(jit_build::dependencies_up_to_date(object.parent_path().string(), object.string()));

    const auto other = compile_pch_case(201, tt::DataFormat::Float16_b).front() / "brisc" / "brisck.o";
    EXPECT_NE(pch_of(other), pch_of(object));  // A define the prefix reads gets its own PCH.
}

TEST_F(OperationPchMockBlackholeFixture, PchIsNotSharedAcrossMathFidelities) {
    // MATH_FIDELITY is declared in the PCH rather than passed as a flag, so reusing a PCH would compile the wrong one.
    EXPECT_NO_THROW(compile_pch_case(801, tt::DataFormat::Float16_b, {}, 0, MathFidelity::HiFi4));
    EXPECT_NO_THROW(compile_pch_case(802, tt::DataFormat::Float16_b, {}, 0, MathFidelity::LoFi));
}

TEST_F(OperationPchMockBlackholeFixture, EditedPrefixHeaderRebuildsThePchAndDeletedOneFails) {
    namespace fs = std::filesystem;
    const auto& env = BuildEnvManager::get_instance(extract_context_id(devices_.at(0).get()))
                          .get_device_build_env(devices_.at(0)->build_id())
                          .build_env;
    const auto dependency = fs::path(env.get_out_kernel_root_path()) / "pch-mutable-dependency.hpp";
    fs::create_directories(dependency.parent_path());
    std::ofstream(dependency) << "constexpr int pch_dependency_value = 1;\n";
    const auto root = compile_pch_case(501, tt::DataFormat::Float16_b, dependency, 1).front();
    const auto object = root / "brisc" / "brisck.o";
    ASSERT_TRUE(jit_build::dependencies_up_to_date(object.parent_path().string(), object.string()));
    std::ofstream(dependency) << "constexpr int pch_dependency_value = 2;\n";
    // GCC's depfile omits headers inside the PCH, so the object must carry the PCH's dependencies.
    jit_build::clear_file_hash_cache();
    EXPECT_FALSE(jit_build::dependencies_up_to_date(object.parent_path().string(), object.string()));
    jit_build_cache_clear();
    EXPECT_NO_THROW(compile_pch_case(502, tt::DataFormat::Float16_b, dependency, 2));
    fs::remove(dependency);
    jit_build_cache_clear();
    EXPECT_THROW(compile_pch_case(503, tt::DataFormat::Float16_b, dependency, 3), std::runtime_error);
    std::ofstream(dependency) << "constexpr int pch_dependency_value = 3;\n";
    EXPECT_NO_THROW(compile_pch_case(503, tt::DataFormat::Float16_b, dependency, 3));
    fs::remove(dependency);
}

TEST_F(OperationPchMockBlackholeFixture, PrefixIncludingAGeneratedKernelHeaderFails) {
    EXPECT_THROW(compile_pch_case(701, tt::DataFormat::Float16_b, "named_args_generated.h"), std::runtime_error);
    const auto& env = BuildEnvManager::get_instance(extract_context_id(devices_.at(0).get()))
                          .get_device_build_env(devices_.at(0)->build_id())
                          .build_env;
    // The rejected PCH is not left in the cache under its temporary name.
    for (const auto& entry : std::filesystem::recursive_directory_iterator(
             std::filesystem::path(env.get_out_kernel_root_path()).parent_path().parent_path() / "prefix-pch")) {
        if (entry.path().extension() == ".gch") {
            EXPECT_EQ(entry.path().filename(), "prefix.hpp.gch");
        }
    }
}

TEST_F(OperationPchMockBlackholeFixture, RejectedPchFailsInsteadOfParsingThePrefix) {
    namespace fs = std::filesystem;
    const auto pch = pch_of(compile_pch_case(901, tt::DataFormat::Float16_b).front() / "brisc" / "brisck.o");
    ASSERT_TRUE(fs::exists(pch));
    fs::copy_file(pch, pch.string() + ".orig", fs::copy_options::overwrite_existing);
    std::ofstream(pch) << "not a PCH";
    jit_build_cache_clear();
    // GCC would otherwise parse the prefix as text and silently lose the speedup.
    EXPECT_THROW(compile_pch_case(902, tt::DataFormat::Float16_b), std::runtime_error);
    fs::rename(pch.string() + ".orig", pch);
}

TEST_F(OperationPchMockBlackholeFixture, ForceJitRebuildsThePch) {
    namespace fs = std::filesystem;
    const auto pch = pch_of(compile_pch_case(1001, tt::DataFormat::Float16_b).front() / "brisc" / "brisck.o");
    ASSERT_TRUE(fs::exists(pch));
    const auto built = fs::last_write_time(pch);
    auto& options = MetalContext::instance(extract_context_id(devices_.at(0).get())).rtoptions();
    options.set_force_jit_compile(true);
    jit_build_cache_clear();
    EXPECT_NO_THROW(compile_pch_case(1001, tt::DataFormat::Float16_b));
    options.set_force_jit_compile(false);
    EXPECT_NE(built, fs::last_write_time(pch));
}

TEST_F(OperationPchMockBlackholeFixture, ProfilerZonesRetainSourceIdentityAcrossPrefixReuse) {
    if (!MetalContext::instance(extract_context_id(devices_.at(0).get())).rtoptions().get_profiler_enabled()) {
        GTEST_SKIP() << "Requires TT_METAL_DEVICE_PROFILER=1";
    }
    namespace fs = std::filesystem;
    auto expect_zones_in_logs = [](const std::vector<fs::path>& kernel_dirs) {
        for (const auto& path : {
                 kernel_dirs[0] / "brisc/brisck.o.log",
                 kernel_dirs[1] / "ncrisc/ncrisck.o.log",
                 kernel_dirs[2] / "trisc0/trisck.o.log",
                 kernel_dirs[2] / "trisc1/trisck.o.log",
                 kernel_dirs[2] / "trisc2/trisck.o.log",
             }) {
            SCOPED_TRACE(path.string());
            ASSERT_TRUE(fs::exists(path));
            std::ifstream log(path);
            std::string text{std::istreambuf_iterator<char>(log), std::istreambuf_iterator<char>()};
            EXPECT_NE(text.find("pch-operation-zone"), std::string::npos);
            EXPECT_NE(text.find("pch-operation-probe.cpp"), std::string::npos);
        }
    };
    const auto first = compile_pch_case(601, tt::DataFormat::Float16_b);
    expect_zones_in_logs(compile_pch_case(602, tt::DataFormat::Float16_b));

    // A PCH whose build log is gone is rebuilt, so the zones still reach the kernel logs.
    std::vector<fs::path> pch_logs;
    for (const auto& entry :
         fs::recursive_directory_iterator(first.front().parent_path().parent_path().parent_path() / "prefix-pch")) {
        if (entry.path().filename() == "build.log") {
            pch_logs.push_back(entry.path());
        }
    }
    ASSERT_FALSE(pch_logs.empty());
    for (const auto& log : pch_logs) {
        fs::remove(log);
    }
    jit_build_cache_clear();
    expect_zones_in_logs(compile_pch_case(603, tt::DataFormat::Float16_b));
}

TEST_F(OperationPchMockBlackholeFixture, SanitizerBypassesOperationPrefix) {
    const auto& options = MetalContext::instance(extract_context_id(devices_.at(0).get())).rtoptions();
    if (!options.get_sanitizer_settings().enabled) {
        GTEST_SKIP() << "Requires TT_METAL_LLK_SANITIZER=1, TT_METAL_LLK_ASSERTS=1 and a fresh TT_METAL_CACHE";
    }
    // Sanitizer report helpers expand FULL_KERNEL_NAME while parsing operation headers.
    // That name must not be frozen into a prefix shared by different kernels.
    const auto root = compile_pch_case(401, tt::DataFormat::Float16_b).front();
    EXPECT_FALSE(std::filesystem::exists(root.parent_path().parent_path().parent_path() / "prefix-pch"));
}

}  // namespace tt::tt_metal
