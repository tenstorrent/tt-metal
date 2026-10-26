// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// KernelDescriptor::cb_table_core_ranges: a kernel's compile-time CB tables can list the CBs of more cores than its
// own, so kernels of one source share a build when a CB is on some of their cores and not others.

#include <gtest/gtest.h>

#include <exception>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/face_geometry.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "mock_blackhole_fixture.hpp"

namespace tt::tt_metal {

// This fixture has external linkage because gtest's generated TEST_F classes derive from it.
class CbTableCoreRangesMockBlackholeFixture : public MockBlackholeMeshDispatchFixture {
protected:
    static constexpr CoreCoord kCbCore{0, 0};
    static constexpr CoreCoord kOtherCore{1, 0};

    // The same compute kernel on two cores, with a CB at index 0 on the first core. other_cb, when given, puts a CB at
    // index 0 on the second core too.
    static ProgramDescriptor two_core_program(
        const CoreRangeSet& cb_table_core_ranges, const std::optional<CBFormatDescriptor>& other_cb = std::nullopt) {
        ProgramDescriptor descriptor;
        descriptor.cbs.push_back(CBDescriptor{
            .total_size = 2048,
            .core_ranges = CoreRangeSet(CoreRange(kCbCore)),
            .format_descriptors = {{.buffer_index = 0, .data_format = tt::DataFormat::Float16_b, .page_size = 2048}},
        });
        if (other_cb) {
            descriptor.cbs.push_back(CBDescriptor{
                .total_size = other_cb->page_size,
                .core_ranges = CoreRangeSet(CoreRange(kOtherCore)),
                .format_descriptors = {*other_cb},
            });
        }
        for (const auto& core : {kCbCore, kOtherCore}) {
            descriptor.kernels.push_back(KernelDescriptor{
                .kernel_source = "void kernel_main() {}",
                .source_type = KernelDescriptor::SourceType::SOURCE_CODE,
                .core_ranges = CoreRangeSet(CoreRange(core)),
                .config = ComputeConfigDescriptor{},
                .cb_table_core_ranges = cb_table_core_ranges,
            });
        }
        return descriptor;
    }

    static CoreRangeSet both_cores() { return CoreRangeSet(std::vector{CoreRange(kCbCore), CoreRange(kOtherCore)}); }

    void expect_compile_rejects_cb_index_0(Program& program) {
        try {
            program.impl().compile(devices_.at(0).get());
            ADD_FAILURE() << "compile succeeded";
        } catch (const std::exception& e) {
            EXPECT_NE(
                std::string(e.what()).find(
                    "CB index 0 has more than one data format, tile or face geometry on the cores of a kernel's CB "
                    "table"),
                std::string::npos)
                << e.what();
        }
    }
};

TEST_F(CbTableCoreRangesMockBlackholeFixture, KernelsWithAndWithoutACbShareABuild) {
    auto* device = devices_.at(0).get();
    Program own_tables(two_core_program({}));
    own_tables.impl().compile(device);
    EXPECT_NE(
        own_tables.impl().get_kernel(0)->get_full_kernel_name(),
        own_tables.impl().get_kernel(1)->get_full_kernel_name());

    Program shared_tables(two_core_program(both_cores()));
    shared_tables.impl().compile(device);
    EXPECT_EQ(
        shared_tables.impl().get_kernel(0)->get_full_kernel_name(),
        shared_tables.impl().get_kernel(1)->get_full_kernel_name());
}

TEST_F(CbTableCoreRangesMockBlackholeFixture, CbIndexWithTwoDataFormatsFails) {
    Program program(two_core_program(
        both_cores(),
        CBFormatDescriptor{.buffer_index = 0, .data_format = tt::DataFormat::Float32, .page_size = 4096}));
    expect_compile_rejects_cb_index_0(program);
}

TEST_F(CbTableCoreRangesMockBlackholeFixture, CbIndexWithTwoFaceGeometriesFails) {
    Program program(two_core_program(
        both_cores(),
        CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = tt::DataFormat::Float16_b,
            .page_size = 2048,
            .face_geometry = FaceGeometry{.face_r_dim = 8},
        }));
    expect_compile_rejects_cb_index_0(program);
}

}  // namespace tt::tt_metal
