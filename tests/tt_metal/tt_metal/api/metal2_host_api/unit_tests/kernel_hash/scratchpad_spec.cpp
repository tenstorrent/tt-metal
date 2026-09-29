// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ScratchpadSpec fields that must change the binding kernel's JIT cache key (compute_hash).

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers.hpp"
#include "metal2_host_api/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, CPU_DifferentScratchpadSizeProducesDifferentKernelHash) {
    // Same kernel source and accessor name; the two scratchpads differ only in size_per_node, which
    // flows into the ScratchpadBindingHandle's size (and the generated scratch:: token), so the
    // hashes must differ.
    auto make_spec = [](uint32_t size_per_node) {
        ProgramSpec spec;
        spec.name = "scratchpad_hash_size";
        auto dm_kernel = MakeMinimalGen2DMKernel("dm_kernel");
        dm_kernel.scratchpad_bindings.push_back(KernelSpec::ScratchpadBinding{
            .scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "scratch"});
        spec.kernels = {dm_kernel};
        spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = size_per_node}};
        spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {"dm_kernel"})};
        return spec;
    };

    Program prog_small = MakeProgramFromSpec(*mesh_device_, make_spec(/*size_per_node=*/1024));
    Program prog_large = MakeProgramFromSpec(*mesh_device_, make_spec(/*size_per_node=*/2048));

    auto hash_small = prog_small.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash();
    auto hash_large = prog_large.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash();
    EXPECT_NE(hash_small, hash_large) << "Scratchpads of different sizes must produce different kernel hashes.";
}

TEST_F(ProgramSpecTestQuasar, ScratchpadLlkFormatAffectsKernelHash) {
    auto make_spec = [](std::optional<tt::DataFormat> format) {
        ProgramSpec spec = MakeMinimalValidProgramSpec();
        spec.scratchpads = {ScratchpadSpec{
            .unique_id = ScratchpadSpecName{"pad"},
            .size_per_node = 1024,
            .data_format_metadata = format,
        }};
        spec.kernels[1].scratchpad_bindings.push_back(
            KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"});
        return spec;
    };

    Program prog_none = MakeProgramFromSpec(*mesh_device_, make_spec(std::nullopt));
    Program prog_f16 = MakeProgramFromSpec(*mesh_device_, make_spec(tt::DataFormat::Float16_b));
    Program prog_f32 = MakeProgramFromSpec(*mesh_device_, make_spec(tt::DataFormat::Float32));

    auto hash_none = prog_none.impl().get_kernel_by_spec_name("compute_kernel")->compute_hash();
    auto hash_f16 = prog_f16.impl().get_kernel_by_spec_name("compute_kernel")->compute_hash();
    auto hash_f32 = prog_f32.impl().get_kernel_by_spec_name("compute_kernel")->compute_hash();
    EXPECT_NE(hash_none, hash_f16);
    EXPECT_NE(hash_f16, hash_f32);
}

}  // namespace
}  // namespace tt::tt_metal::experimental
