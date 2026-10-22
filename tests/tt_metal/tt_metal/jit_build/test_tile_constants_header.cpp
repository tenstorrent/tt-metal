// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Device code reads the default tile dimensions from api/tile_constants.h (tt_metal/hw/inc), while
// tt-metalium/constants.hpp is compiled by the host compiler only. These tests push both through the
// real JIT pipeline on a mock device: the device header must build for data-movement and compute
// kernels, and the host header must be rejected.

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "impl/program/program_impl.hpp"
#include "mock_blackhole_fixture.hpp"

namespace tt::tt_metal {

// External linkage: gtest's generated TEST_F classes derive from it.
class TileConstantsHeaderMockBlackholeFixture : public MockBlackholeMeshDispatchFixture {
protected:
    KernelDescriptor kernel_from_source(std::string source, KernelDescriptor::ConfigDescriptor config) {
        return KernelDescriptor{
            .kernel_source = std::move(source),
            .source_type = KernelDescriptor::SourceType::SOURCE_CODE,
            .core_ranges = CoreRange(CoreCoord{0, 0}),
            .config = std::move(config),
        };
    }

    void compile(const KernelDescriptor& kernel) {
        Program program(ProgramDescriptor{.kernels = {kernel}});
        program.impl().compile(devices_.at(0).get());
    }
};

}  // namespace tt::tt_metal

using namespace tt::tt_metal;

namespace {

constexpr const char* kTileConstantsKernel = R"(
#include "api/tile_constants.h"
static_assert(tt::constants::TILE_HEIGHT == 32 && tt::constants::TILE_WIDTH == 32);
static_assert(tt::constants::TILE_HW == tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH);
static_assert(tt::constants::FACE_HEIGHT == 16 && tt::constants::FACE_WIDTH == 16);
static_assert(tt::constants::FACE_HW == tt::constants::FACE_HEIGHT * tt::constants::FACE_WIDTH);
void kernel_main() {}
)";

constexpr const char* kHostConstantsKernel = R"(
#include <tt-metalium/constants.hpp>
void kernel_main() {}
)";

}  // namespace

TEST_F(TileConstantsHeaderMockBlackholeFixture, DataMovementKernelIncludesTileConstants) {
    compile(kernel_from_source(kTileConstantsKernel, DataMovementConfigDescriptor{}));
}

TEST_F(TileConstantsHeaderMockBlackholeFixture, ComputeKernelIncludesTileConstants) {
    compile(kernel_from_source(kTileConstantsKernel, ComputeConfigDescriptor{}));
}

TEST_F(TileConstantsHeaderMockBlackholeFixture, HostConstantsHeaderIsRejectedInKernels) {
    try {
        compile(kernel_from_source(kHostConstantsKernel, DataMovementConfigDescriptor{}));
        ADD_FAILURE() << "kernel including tt-metalium/constants.hpp compiled";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("constants.hpp is host-only"), std::string::npos) << e.what();
    }
}
