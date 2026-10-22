// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/host_api.hpp>

#include "common/executor.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/dispatch/dispatch_query_manager.hpp"
#include "impl/program/kernel_compile_utils.hpp"
#include "impl/program/program_impl.hpp"
#include "jit_build/build_env_manager.hpp"
#include "jit_build/jit_build_cache.hpp"
#include "jit_build/jit_build_options.hpp"
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

TEST_F(NamedCtArgChannelsMockBlackholeFixture, GeneratedDescriptorsCompileWithDuplicateBuilds) {
    KernelDescriptor kernel = {
        .kernel_source = "void kernel_main() {}",
        .source_type = KernelDescriptor::SourceType::SOURCE_CODE,
        .core_ranges = CoreRange(CoreCoord{0, 0}),
        .blaze_named_args = {.named_compile_time_args = {{"typed.value", 7}}},
        .config = DataMovementConfigDescriptor{},
    };
    ProgramDescriptor descriptor{.kernels = {kernel}};
    kernel.core_ranges = CoreRange(CoreCoord{1, 0});
    descriptor.kernels.push_back(kernel);
    auto* device = devices_.at(0).get();

    // Build once to put the binary on disk. Same source and args on different cores share one build.
    Program warm(descriptor);
    warm.impl().compile(device, false);
    EXPECT_EQ(warm.impl().get_kernel(0)->get_full_kernel_name(), warm.impl().get_kernel(1)->get_full_kernel_name());

    // The kernel uses no CBs, so its build options need nothing beyond the kernel's own.
    Program program(descriptor);
    const auto& build_env =
        BuildEnvManager::get_instance(extract_context_id(device)).get_device_build_env(device->build_id());
    JitBuildOptions build_options(build_env.build_env);
    program.impl().get_kernel(0)->set_build_options(build_options);
    const size_t kernel_hash =
        detail::KernelCompileHash(program.impl().get_kernel(0), build_options, build_env.build_key());

    // Hold that hash in progress so both kernels of the fresh Program defer and join it after the sync.
    auto& cache = JitBuildCache::inst();
    cache.clear();
    std::promise<void> started;
    std::promise<void> release;
    auto release_future = release.get_future();
    std::thread owner([&] {
        cache.build_once(kernel_hash, [&] {
            started.set_value();
            release_future.wait();
        });
    });
    started.get_future().wait();
    auto compiled = std::async(std::launch::async, [&] { program.impl().compile(device, false); });
    EXPECT_EQ(compiled.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    // The duplicates wait on the compiling thread, not in compile workers.
    for (int i = 0; i < 1000 && detail::GetExecutor().num_topologies() != 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_EQ(detail::GetExecutor().num_topologies(), 0u);
    release.set_value();
    owner.join();
    EXPECT_NO_THROW(compiled.get());
    EXPECT_NO_THROW(program.impl().get_kernel(0)->binaries(build_env.build_key()));
    EXPECT_NO_THROW(program.impl().get_kernel(1)->binaries(build_env.build_key()));
}

TEST_F(NamedCtArgChannelsMockBlackholeFixture, SameProgramDuplicateDoesNotRetryFailedBuild) {
    // Each compile of this source blocks reading the FIFO until the test opens it for writing, then fails.
    const auto fifo = std::filesystem::temp_directory_path() / ("jit_failed_build_" + std::to_string(::getpid()));
    std::filesystem::remove(fifo);
    ASSERT_EQ(::mkfifo(fifo.c_str(), 0600), 0);
    KernelDescriptor kernel = {
        .kernel_source = "#include \"" + fifo.string() + "\"\n#error expected build failure\n",
        .source_type = KernelDescriptor::SourceType::SOURCE_CODE,
        .core_ranges = CoreRange(CoreCoord{0, 0}),
        .config = DataMovementConfigDescriptor{},
    };
    ProgramDescriptor descriptor{.kernels = {kernel}};
    kernel.core_ranges = CoreRange(CoreCoord{1, 0});
    descriptor.kernels.push_back(kernel);
    Program program(descriptor);
    JitBuildCache::inst().clear();
    auto compiled = std::async(std::launch::async, [&] { program.impl().compile(devices_.at(0).get(), false); });

    // A nonblocking open succeeds only while a compiler has the FIFO open, so each success is one build.
    const auto open_build = [&] { return ::open(fifo.c_str(), O_WRONLY | O_NONBLOCK); };
    int fd = -1;
    while ((fd = open_build()) < 0) {
        ASSERT_EQ(compiled.wait_for(std::chrono::milliseconds(1)), std::future_status::timeout);
    }
    // Once the duplicate defers, only the owner's task and its compile step remain.
    for (int i = 0; i < 1000 && detail::GetExecutor().num_topologies() != 2; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_EQ(detail::GetExecutor().num_topologies(), 2u);
    ::close(fd);
    int retries = 0;
    while (compiled.wait_for(std::chrono::milliseconds(10)) == std::future_status::timeout) {
        fd = open_build();
        if (fd >= 0) {
            ++retries;
            ::close(fd);
        }
    }
    try {
        compiled.get();
        ADD_FAILURE() << "compile succeeded";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("Failed to generate binaries"), std::string::npos) << e.what();
    }
    EXPECT_EQ(retries, 0);
    std::filesystem::remove(fifo);
}

TEST_F(NamedCtArgChannelsMockBlackholeFixture, PlacementFailureDoesNotStartBuilds) {
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
        KernelDescriptor kernel{
            .kernel_source = "void kernel_main() {}",
            .source_type = KernelDescriptor::SourceType::SOURCE_CODE,
            .core_ranges = CoreRange(invalid_first ? invalid_core : valid_core),
            .blaze_named_args = {.named_compile_time_args = {{"typed.value", 8}}},
            .config = DataMovementConfigDescriptor{},
        };
        ProgramDescriptor descriptor{.kernels = {kernel}};
        kernel.core_ranges = CoreRange(invalid_first ? valid_core : invalid_core);
        descriptor.kernels.push_back(kernel);
        Program program(descriptor);
        EXPECT_THROW(program.impl().compile(device, false), std::runtime_error);
        EXPECT_TRUE(program.impl().get_kernel(0)->get_full_kernel_name().empty());
        EXPECT_TRUE(program.impl().get_kernel(1)->get_full_kernel_name().empty());
        EXPECT_NO_THROW(program.impl().compile(device, true));
    }
}

}  // namespace tt::tt_metal
