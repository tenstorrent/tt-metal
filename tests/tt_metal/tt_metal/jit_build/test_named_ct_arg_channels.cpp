// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <utility>

#include <enchantum/enchantum.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "common/mesh_dispatch_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "jit_build/build_env_manager.hpp"
#include "jit_build/jit_build_utils.hpp"

namespace tt::tt_metal {

// This fixture has external linkage because gtest's generated TEST_F classes derive from it.
class NamedCtArgChannelsMockBlackholeFixture : public MeshDispatchFixture {
protected:
    struct NamedCtArtifacts {
        bool legacy_header;
        bool blaze_header;
        bool legacy_force_include;
    };

    static void SetUpTestSuite() {
        experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1);
        MeshDispatchFixture::SetUpTestSuite();
    }

    static void TearDownTestSuite() {
        MeshDispatchFixture::TearDownTestSuite();
        experimental::disable_mock_mode();
    }

    NamedCtArtifacts compile_and_inspect(
        KernelDescriptor::NamedCompileTimeArgs legacy_args, experimental::blaze::NamedCompileTimeArgs blaze_args) {
        auto* device = devices_.at(0).get();
        KernelDescriptor kernel_descriptor = {
            .kernel_source = "void kernel_main() {}",
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
        const auto& hal = MetalContext::instance(kernel->get_context_id()).hal();
        const uint32_t core_idx = hal.get_programmable_core_type_index(kernel->get_kernel_programmable_core_type());
        const uint32_t class_idx = enchantum::to_underlying(kernel->get_kernel_processor_class());
        const uint32_t processor_id = kernel->get_kernel_processor_type(0);
        const auto recipe =
            build_env_manager.get_kernel_build_state(device->build_id(), core_idx, class_idx, processor_id)
                .export_target_recipe(kernel.get());

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
    const auto artifacts = compile_and_inspect({}, {{"typed.value", 1}});

    EXPECT_FALSE(artifacts.legacy_header);
    EXPECT_TRUE(artifacts.blaze_header);
    EXPECT_FALSE(artifacts.legacy_force_include);
}

TEST_F(NamedCtArgChannelsMockBlackholeFixture, LegacyOnlyEmitsLegacyMap) {
    const auto artifacts = compile_and_inspect({{"legacy_value", 2}}, {});

    EXPECT_TRUE(artifacts.legacy_header);
    EXPECT_FALSE(artifacts.blaze_header);
    EXPECT_TRUE(artifacts.legacy_force_include);
}

TEST_F(NamedCtArgChannelsMockBlackholeFixture, MixedEmitsBothRepresentations) {
    const auto artifacts = compile_and_inspect({{"legacy_value", 3}}, {{"typed.value", 4}});

    EXPECT_TRUE(artifacts.legacy_header);
    EXPECT_TRUE(artifacts.blaze_header);
    EXPECT_TRUE(artifacts.legacy_force_include);
}

}  // namespace tt::tt_metal
