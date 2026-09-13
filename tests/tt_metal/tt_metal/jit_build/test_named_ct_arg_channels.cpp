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

#include "impl/kernels/kernel.hpp"
#include "impl/dispatch/dispatch_query_manager.hpp"
#include "impl/program/program_impl.hpp"
#include "jit_build/build.hpp"
#include "jit_build/build_env_manager.hpp"
#include "jit_build/depend.hpp"
#include "jit_build/jit_build_utils.hpp"
#include "llrt/rtoptions.hpp"
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

    std::vector<std::filesystem::path> compile_pch_case(
        uint32_t tag,
        tt::DataFormat format,
        bool runtime_only = false,
        const std::filesystem::path& dependency = {},
        int dependency_value = 0) {
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
        if (runtime_only) {
            kernel.blaze_named_args.named_compile_time_args.clear();
            kernel.blaze_named_args.named_common_runtime_args = {{"typed.value", tag}};
            kernel.kernel_source =
                "#ifdef COMPILE_FOR_TRISC\n#include \"api/compute/common.h\"\n#endif\n"
                "void kernel_main() { asm volatile(\"\" :: "
                "\"r\"(get_common_arg_val<uint32_t>(blaze_ct_args::typed::value.index))); "
                "}";
        }
        if (std::getenv("TT_METAL_BLAZE_OPERATION_PCH") != nullptr &&
            std::string(std::getenv("TT_METAL_BLAZE_OPERATION_PCH")) == "1") {
            const auto& env = BuildEnvManager::get_instance(extract_context_id(device))
                                  .get_device_build_env(device->build_id())
                                  .build_env;
            const auto source = std::filesystem::path(env.get_out_kernel_root_path()) / "pch-operation-probe.cpp";
            std::filesystem::create_directories(source.parent_path());
            std::ofstream file(source);
            file << "#ifndef BLAZE_OPERATION_PCH_PREFIX_INCLUDED\n#define BLAZE_OPERATION_PCH_PREFIX_INCLUDED\n";
            // Deliberately unguarded: consuming the PCH must not declare this twice.
            file << "struct PchOperationPrefix { static constexpr int value = 7; };\n";
            file << "#ifdef COMPILE_FOR_TRISC\n#include \"api/compute/common.h\"\n#endif\n";
            if (!dependency.empty()) {
                file << "#include \"" << dependency.string() << "\"\n";
            } else if (const auto* replay_header = std::getenv("BLAZE_PCH_TEST_HEADER")) {
                file << "#include \"" << replay_header << "\"\n";
            }
            if (env.get_rtoptions().get_profiler_enabled()) {
                file << "#include \"tools/profiler/kernel_profiler.hpp\"\n";
                file << "inline void pch_profile_probe() { DeviceZoneScopedN(\"pch-operation-zone\"); }\n";
            }
            file << "#endif\n// BLAZE_OPERATION_PCH_END\nstatic_assert(PchOperationPrefix::value == 7);\n";
            file << "static_assert(unpack_src_format[0] == " << static_cast<uint32_t>(format) << ");\n";
            if (!dependency.empty()) {
                file << "static_assert(pch_dependency_value == " << dependency_value << ");\n";
            }
            if (const auto* expected = std::getenv("BLAZE_PCH_TEST_EXPECT")) {
                file << "static_assert(PCH_TEST_VALUE == " << expected << ");\n";
            }
            file << kernel.kernel_source;
            file.close();
            kernel.kernel_source = source.string();
            kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
            kernel.defines.emplace_back("BLAZE_OPERATION_PCH_SOURCE", source.string());
        }
        descriptor.kernels.push_back(kernel);
        kernel.config =
            DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default};
        descriptor.kernels.push_back(kernel);
        kernel.config = ComputeConfigDescriptor{};
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

// These opt-in tests invoke the real RISC-V compiler on a mock device, using a fresh
// TT_METAL_CACHE. No device execution or timing claim is made by this host fixture.
TEST_F(NamedCtArgChannelsMockBlackholeFixture, PrefixReusePreservesArgumentAndResourceBoundaries) {
    if (std::getenv("TT_METAL_BLAZE_PREFIX_PCH") == nullptr ||
        std::string(std::getenv("TT_METAL_BLAZE_PREFIX_PCH")) != "1") {
        GTEST_SKIP() << "Enable TT_METAL_BLAZE_PREFIX_PCH=1 with a fresh cache";
    }
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
    compile_pch_case(102, tt::DataFormat::Float16_b);
    EXPECT_EQ(first, pch_files());  // Different named values and kernel names reuse the prefix.
    root = compile_pch_case(103, tt::DataFormat::Bfp8_b).front();
    const auto changed_cb = pch_files();
    const bool operations = std::getenv("TT_METAL_BLAZE_OPERATION_PCH") != nullptr &&
                            std::string(std::getenv("TT_METAL_BLAZE_OPERATION_PCH")) == "1";
    EXPECT_EQ(changed_cb.size(), first.size() + (operations ? 0 : 2));
    if (operations) {
        EXPECT_EQ(first, changed_cb);  // Only constexpr definitions change; the declarations remain cached.
    }

    bool checked_original_cb_dependency = false;
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.path().filename() != "brisck.o") {
            continue;
        }
        const auto object = entry.path();
        const auto cb = object.parent_path().parent_path() / "chlkc_descriptors.h";
        ASSERT_TRUE(jit_build::dependencies_up_to_date(object.parent_path().string(), object.string()));
        {
            std::ofstream changed(cb, std::ios::app);
            changed << "\n// changed original generated descriptor\n";
        }
        EXPECT_FALSE(jit_build::dependencies_up_to_date(object.parent_path().string(), object.string()));
        checked_original_cb_dependency = true;
        break;
    }
    EXPECT_TRUE(checked_original_cb_dependency);
}

TEST_F(NamedCtArgChannelsMockBlackholeFixture, ChangedAndDeletedPrefixDependencyAfterCacheClear) {
    const auto* operation = std::getenv("TT_METAL_BLAZE_OPERATION_PCH");
    if (operation == nullptr || std::string(operation) != "1") {
        GTEST_SKIP() << "Requires operation PCH and a fresh cache";
    }
    namespace fs = std::filesystem;
    const auto& env = BuildEnvManager::get_instance(extract_context_id(devices_.at(0).get()))
                          .get_device_build_env(devices_.at(0)->build_id())
                          .build_env;
    const auto dependency = fs::path(env.get_out_kernel_root_path()) / "pch-mutable-dependency.hpp";
    fs::create_directories(dependency.parent_path());
    std::ofstream(dependency) << "constexpr int pch_dependency_value = 1;\n";
    EXPECT_NO_THROW(compile_pch_case(501, tt::DataFormat::Float16_b, false, dependency, 1));
    std::ofstream(dependency) << "constexpr int pch_dependency_value = 2;\n";
    jit_build_cache_clear();
    EXPECT_NO_THROW(compile_pch_case(502, tt::DataFormat::Float16_b, false, dependency, 2));
    fs::remove(dependency);
    jit_build_cache_clear();
    EXPECT_THROW(compile_pch_case(503, tt::DataFormat::Float16_b, false, dependency, 3), std::runtime_error);
    std::ofstream(dependency) << "constexpr int pch_dependency_value = 3;\n";
    EXPECT_NO_THROW(compile_pch_case(503, tt::DataFormat::Float16_b, false, dependency, 3));
    fs::remove(dependency);
}

TEST_F(NamedCtArgChannelsMockBlackholeFixture, ProfilerZonesRetainSourceIdentityAcrossPrefixReuse) {
    const auto& options = MetalContext::instance(extract_context_id(devices_.at(0).get())).rtoptions();
    const auto* operation = std::getenv("TT_METAL_BLAZE_OPERATION_PCH");
    if (!options.get_profiler_enabled() || operation == nullptr || std::string(operation) != "1") {
        GTEST_SKIP() << "Requires profiling and operation PCH with a fresh cache";
    }
    namespace fs = std::filesystem;
    const auto first = compile_pch_case(601, tt::DataFormat::Float16_b);
    const auto second = compile_pch_case(602, tt::DataFormat::Float16_b);
    for (const auto& path : {
             second[0] / "brisc/brisck.o.log",
             second[1] / "ncrisc/ncrisck.o.log",
             second[2] / "trisc0/trisck.o.log",
             second[2] / "trisc1/trisck.o.log",
             second[2] / "trisc2/trisck.o.log",
         }) {
        SCOPED_TRACE(path.string());
        ASSERT_TRUE(fs::exists(path));
        std::ifstream log(path);
        std::string text{std::istreambuf_iterator<char>(log), std::istreambuf_iterator<char>()};
        EXPECT_NE(text.find("pch-operation-zone"), std::string::npos);
        EXPECT_NE(text.find("pch-operation-probe.cpp"), std::string::npos);
    }
    const auto pch_root = first.front().parent_path().parent_path().parent_path() / "prefix-pch";
    bool replayed_missing_log = false;
    for (const auto& entry : fs::recursive_directory_iterator(pch_root)) {
        if (entry.path().filename() == "build.log") {
            const auto log = entry.path();
            fs::remove(log);
            jit_build_cache_clear();
            EXPECT_NO_THROW(compile_pch_case(603, tt::DataFormat::Float16_b));
            EXPECT_TRUE(fs::exists(log));
            replayed_missing_log = true;
            break;
        }
    }
    EXPECT_TRUE(replayed_missing_log);
}

// A subprocess harness changes/deletes this temporary dependency between invocations.
TEST_F(NamedCtArgChannelsMockBlackholeFixture, PrefixDependencyReplay) {
    const auto* tag = std::getenv("BLAZE_PCH_TEST_TAG");
    if (tag == nullptr) {
        GTEST_SKIP() << "Campaign subprocess replay requires BLAZE_PCH_TEST_TAG";
    }
    compile_pch_case(std::stoul(tag), tt::DataFormat::Float16_b);
}

TEST_F(NamedCtArgChannelsMockBlackholeFixture, SanitizerBypassesOperationPrefix) {
    const auto& options = MetalContext::instance(extract_context_id(devices_.at(0).get())).rtoptions();
    const auto* operation_pch = std::getenv("TT_METAL_BLAZE_OPERATION_PCH");
    if (!options.get_sanitizer_settings().enabled || operation_pch == nullptr || std::string(operation_pch) != "1") {
        GTEST_SKIP() << "Enable TT_METAL_LLK_SANITIZER=1 and TT_METAL_BLAZE_OPERATION_PCH=1 with a fresh cache";
    }
    // Sanitizer report helpers expand FULL_KERNEL_NAME while parsing operation headers.
    // That name must not be frozen into a prefix shared by different products.
    const auto root = compile_pch_case(401, tt::DataFormat::Float16_b).front();
    EXPECT_FALSE(std::filesystem::exists(root.parent_path().parent_path().parent_path() / "prefix-pch"));
}

TEST_F(NamedCtArgChannelsMockBlackholeFixture, RuntimeOnlyPrefix) {
    if (std::getenv("TT_METAL_BLAZE_PREFIX_PCH") == nullptr ||
        std::string(std::getenv("TT_METAL_BLAZE_PREFIX_PCH")) != "1") {
        GTEST_SKIP() << "Enable PCH for this native mock probe";
    }
    const auto root = compile_pch_case(301, tt::DataFormat::Float16_b, true).front();
    const auto pch_root = root.parent_path().parent_path().parent_path() / "prefix-pch";
    EXPECT_TRUE(std::filesystem::exists(pch_root));
}

TEST_F(NamedCtArgChannelsMockBlackholeFixture, GeneratedDescriptorsCompileWithDuplicateBuilds) {
    for (const bool runtime_only : {false, true}) {
        KernelDescriptor kernel = {
            .kernel_source = "void kernel_main() {}",
            .source_type = KernelDescriptor::SourceType::SOURCE_CODE,
            .core_ranges = CoreRange(CoreCoord{0, 0}),
            .defines = {{"BLAZE_GENERATED_KERNEL", "1"}},
            .config = DataMovementConfigDescriptor{},
        };
        if (runtime_only) {
            kernel.blaze_named_args.named_common_runtime_args = {{"typed.value", 7}};
        } else {
            kernel.blaze_named_args.named_compile_time_args = {{"typed.value", 7}};
        }
        ProgramDescriptor descriptor{.kernels = {kernel}};
        kernel.core_ranges = CoreRange(CoreCoord{1, 0});
        descriptor.kernels.push_back(kernel);
        Program program(descriptor);
        EXPECT_NO_THROW(program.impl().compile(devices_.at(0).get(), false));
    }
}

TEST_F(NamedCtArgChannelsMockBlackholeFixture, PlacementFailureDoesNotStartLocalBuilds) {
    auto* device = devices_.at(0).get();
    auto& context = MetalContext::instance(extract_context_id(device));
    if (!context.rtoptions().get_fast_dispatch() ||
        context.get_dispatch_core_manager().get_dispatch_core_type() != CoreType::WORKER) {
        GTEST_SKIP() << "Requires a fast-dispatch mock with worker dispatch cores";
    }
    const auto& dispatch_cores = context.get_dispatch_query_manager().get_logical_dispatch_cores_on_user_chips();
    ASSERT_FALSE(dispatch_cores.empty());
    const auto invalid_core = dispatch_cores.front();
    const CoreCoord valid_core = invalid_core == CoreCoord{0, 0} ? CoreCoord{1, 0} : CoreCoord{0, 0};
    for (const bool invalid_first : {false, true}) {
        KernelDescriptor descriptor{
            .kernel_source = "void kernel_main() {}",
            .source_type = KernelDescriptor::SourceType::SOURCE_CODE,
            .core_ranges = CoreRange(invalid_first ? invalid_core : valid_core),
            .defines = {{"BLAZE_GENERATED_KERNEL", "1"}},
            .blaze_named_args = {.named_compile_time_args = {{"typed.value", 8}}},
            .config = DataMovementConfigDescriptor{},
        };
        ProgramDescriptor descriptors{.kernels = {descriptor}};
        descriptor.core_ranges = CoreRange(invalid_first ? valid_core : invalid_core);
        descriptors.kernels.push_back(descriptor);
        Program program(descriptors);
        EXPECT_THROW(program.impl().compile(device, false), std::runtime_error);
        EXPECT_TRUE(program.impl().get_kernel(0)->get_full_kernel_name().empty());
        EXPECT_TRUE(program.impl().get_kernel(1)->get_full_kernel_name().empty());
        EXPECT_NO_THROW(program.impl().compile(device, true));
    }
}

}  // namespace tt::tt_metal
