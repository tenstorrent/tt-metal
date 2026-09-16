// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "jit_build/build_env_manager.hpp"
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

TEST_F(NamedCtArgChannelsMockBlackholeFixture, PositionalValuesAndKernelDefinesShareFirmwarePch) {
    const auto& options = MetalContext::instance().rtoptions();
    if (!options.get_jit_pch_strict()) {
        GTEST_SKIP() << "Run with TT_METAL_JIT_PCH=1 TT_METAL_JIT_PCH_STRICT=1 TT_METAL_FORCE_JIT_COMPILE=1";
    }
    ASSERT_TRUE(options.get_force_jit_compile());
    std::map<std::filesystem::path, std::filesystem::file_time_type> first_pch;
    for (const std::vector<uint32_t>& values : {std::vector<uint32_t>{7}, {21, 42}, {}}) {
        const std::string source = R"(
#include "api/compile_time_args.h"
static_assert(kernel_compile_time_args.size() == EXPECTED_COUNT);
#if EXPECTED_COUNT > 0
static_assert(get_compile_time_arg_val(0) == EXPECTED_VALUE);
#endif
void kernel_main() {}
)";
        KernelDescriptor descriptor = {
            .kernel_source = source,
            .source_type = KernelDescriptor::SourceType::SOURCE_CODE,
            .core_ranges = CoreRange(CoreCoord{0, 0}),
            .compile_time_args = values,
            .defines =
                {{"EXPECTED_COUNT", std::to_string(values.size())},
                 {"EXPECTED_VALUE", std::to_string(values.empty() ? 0 : values.front())}},
            // Appended kernel include paths must not create new PCH profiles.
            .compiler_include_paths =
                {std::filesystem::path(options.get_root_dir()) / (values.empty() ? "tests" : "tt_metal")},
        };
        ProgramDescriptor::KernelDescriptors descriptors;
        descriptor.config = DataMovementConfigDescriptor{};
        descriptors.push_back(descriptor);
        descriptor.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1};
        descriptors.push_back(descriptor);
        descriptor.config = ComputeConfigDescriptor{};
        descriptors.push_back(descriptor);
        Program program(ProgramDescriptor{.kernels = descriptors});
        auto* device = devices_.at(0).get();
        program.impl().compile(device);
        const auto& env = BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_env;
        std::map<std::filesystem::path, std::filesystem::file_time_type> accepted_pch;
        for (size_t i = 0; i < descriptors.size(); ++i) {
            const auto dir = std::filesystem::path(env.get_out_kernel_root_path()) /
                             program.impl().get_kernel(i)->get_full_kernel_name();
            for (const auto& file : std::filesystem::recursive_directory_iterator(dir)) {
                if (!file.path().string().ends_with(".o.log")) {
                    continue;
                }
                std::ifstream log(file.path());
                for (std::string line; std::getline(log, line);) {
                    if (line.starts_with("! ") && line.ends_with(".gch")) {
                        const std::filesystem::path pch = line.substr(2);
                        accepted_pch.emplace(pch, std::filesystem::last_write_time(pch));
                    }
                }
            }
        }
        ASSERT_EQ(accepted_pch.size(), 5u) << "BRISC, NCRISC and all three TRISCs must consume a PCH";
        if (first_pch.empty()) {
            first_pch = accepted_pch;
        } else {
            EXPECT_EQ(accepted_pch, first_pch) << "Kernel values or include paths rebuilt a PCH";
        }
    }
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

}  // namespace tt::tt_metal
