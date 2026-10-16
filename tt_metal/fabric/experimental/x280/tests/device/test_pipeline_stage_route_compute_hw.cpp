// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "harness/artifacts/artifacts.hpp"
#include "harness/gtest_dispatch.hpp"
#include "harness/runner.hpp"
#include "suites/pipeline_stage/pinned_input.hpp"
#include "suites/pipeline_stage/route_compute.hpp"

#include "fabric_shadow.h"
#include "fabric_shadow_header.hpp"
#include "pipeline_stage_common.h"
#include "route_compute_args.h"
#include "rt/x280_pinned_input.h"

#include <hostdevcommon/fabric_common.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

namespace suite = x280::suites::pipeline_stage::route_compute;

class PipelineStageRouteCompute : public x280::suites::pipeline_stage::pinned_input::PreparedInputRunner<suite::Axis> {
protected:
    x280::harness::RunSpec make_spec() override {
        return x280::harness::RunSpec{
            .open =
                {
                    .image_env_var = "X280_PIPELINE_STAGE_ROUTE_COMPUTE_FW_IMAGE",
                    .image_file = "firmware-pipeline-stage-route-compute.bin",
                    .kernel_dir = x280::harness::artifacts::kernel_dir_of(__FILE__),
                },
            .suite_label = "pipeline_stage_route_compute",
        };
    }
};

}  // namespace

TEST_P(PipelineStageRouteCompute, BatchedStageCostPerItem) { X280_HARNESS_DISPATCH(suite); }

INSTANTIATE_TEST_SUITE_P(
    Sweep,
    PipelineStageRouteCompute,
    ::testing::Values(
        suite::SweepPoint{
            PIPELINE_STAGE_INSTRUMENTATION_OFF,
            "direction_table_plain",
        },
        suite::SweepPoint{
            PIPELINE_STAGE_INSTRUMENTATION_PER_SAMPLE,
            "direction_table_instrumented",
        }),
    [](const ::testing::TestParamInfo<suite::SweepPoint>& info) { return std::string(info.param.label); });

// ===========================================================================
// Offline verification tests for the route oracle and flat argument section.
// These run without hardware, device initialization, or fixtures.
// ===========================================================================

TEST(PipelineStageRouteComputeOffline, SectionPacking16WordsAndPadding) {
    const uint32_t warmup = 128u;
    const uint32_t timed = 1024u;
    const uint32_t instrumentation = PIPELINE_STAGE_INSTRUMENTATION_OFF;
    const uint32_t run_nonce = 0xC01u;
    const uint32_t fixture_id = PIPELINE_STAGE_LOOKUP_FIXTURE_V1;
    const uint64_t staged_input_addr = 0x123456780A001000ull;
    const uint64_t result_addr = 0x9ABCDEF00A003000ull;
    const uint64_t sample_region_addr = 0xFEDCBA980A004000ull;

    UBENCH_SECTION_STORAGE(pipeline_stage_route_compute_section_t, sec) store{};
    fill_section(
        store.sec,
        warmup,
        timed,
        instrumentation,
        run_nonce,
        staged_input_addr,
        result_addr,
        sample_region_addr,
        fixture_id);

    const std::vector<uint8_t> arg_bytes(
        reinterpret_cast<const uint8_t*>(&store.sec), reinterpret_cast<const uint8_t*>(&store.sec) + sizeof(store.sec));

    ASSERT_EQ(arg_bytes.size(), 64u);
    EXPECT_EQ(arg_bytes.size() / sizeof(uint32_t), 16u);

    // Independent expected wire representation.
    std::vector<uint8_t> expected_wire(64u, 0u);
    auto write_u32 = [&expected_wire](size_t offset, uint32_t val) {
        expected_wire[offset + 0] = static_cast<uint8_t>(val);
        expected_wire[offset + 1] = static_cast<uint8_t>(val >> 8u);
        expected_wire[offset + 2] = static_cast<uint8_t>(val >> 16u);
        expected_wire[offset + 3] = static_cast<uint8_t>(val >> 24u);
    };
    auto write_u64 = [&write_u32](size_t offset, uint64_t val) {
        write_u32(offset, static_cast<uint32_t>(val));
        write_u32(offset + 4u, static_cast<uint32_t>(val >> 32u));
    };

    write_u32(0u, pipeline_stage_cookie_for(UBENCH_SECTION_COMMON));
    write_u32(4u, warmup);
    write_u32(8u, timed);
    write_u32(12u, instrumentation);
    write_u32(16u, run_nonce);
    write_u32(20u, pipeline_stage_cookie_for(UBENCH_SECTION_BACKEND));
    write_u64(24u, staged_input_addr);
    write_u64(32u, result_addr);
    write_u64(40u, sample_region_addr);
    write_u32(48u, fixture_id);
    // Bytes 52..63 remain 0u padding.

    EXPECT_EQ(arg_bytes, expected_wire);
}

TEST(PipelineStageRouteComputeOffline, BuildInputImageSpanAndHeaderPlacement) {
    const std::vector<uint8_t> image = suite::build_input_image();

    // Verify complete image is exactly X280_PINNED_INPUT_SPAN (4288 bytes).
    ASSERT_EQ(image.size(), static_cast<size_t>(X280_PINNED_INPUT_SPAN));

    // Headers begin at X280_PINNED_INPUT_HEADERS_OFFSET = 2752 (0xAC0), leaving 48 padding bytes.
    for (size_t offset = suite::kRoutingImageBytes; offset < X280_PINNED_INPUT_HEADERS_OFFSET; ++offset) {
        EXPECT_EQ(image[offset], 0u) << "padding byte non-zero at offset " << offset;
    }

    // Verify routing info content at offset 0.
    const auto* routing = reinterpret_cast<const tt::tt_fabric::routing_l1_info_t*>(image.data());
    EXPECT_EQ(routing->my_mesh_id, suite::kFixtureMesh);
    EXPECT_EQ(routing->my_device_id, 0u);

    for (uint32_t slot = 0u; slot < 8u; ++slot) {
        uint8_t direction = FS_INVALID_ROUTING_TABLE_ENTRY;
        const int ok = fs_original_direction(
            routing->intra_mesh_direction_table.packed_directions,
            FS_MAX_MESH_SIZE,
            suite::kFixture[slot].chip,
            &direction);
        EXPECT_EQ(ok, 1);
        EXPECT_EQ(direction, suite::expected_direction(slot)) << "mismatch at intra slot " << slot;
    }

    for (uint32_t slot = 8u; slot < PIPELINE_STAGE_LOOKUP_HEADER_COUNT; ++slot) {
        uint8_t direction = FS_INVALID_ROUTING_TABLE_ENTRY;
        const int ok = fs_original_direction(
            routing->inter_mesh_direction_table.packed_directions,
            FS_MAX_NUM_MESHES,
            suite::kFixture[slot].mesh,
            &direction);
        EXPECT_EQ(ok, 1);
        EXPECT_EQ(direction, suite::expected_direction(slot)) << "mismatch at inter slot " << slot;
    }

    // Verify seeded headers land at X280_PINNED_INPUT_HEADERS_OFFSET.
    const size_t expected_header_span =
        static_cast<size_t>(PIPELINE_STAGE_LOOKUP_HEADER_COUNT) * FS_PACKET_HEADER_BYTES;
    EXPECT_EQ(X280_PINNED_INPUT_HEADERS_OFFSET + expected_header_span, static_cast<size_t>(X280_PINNED_INPUT_SPAN));

    for (uint32_t slot = 0u; slot < PIPELINE_STAGE_LOOKUP_HEADER_COUNT; ++slot) {
        const size_t header_offset =
            static_cast<size_t>(X280_PINNED_INPUT_HEADERS_OFFSET) + static_cast<size_t>(slot) * FS_PACKET_HEADER_BYTES;
        ASSERT_LE(header_offset + sizeof(fs_hybrid_mesh_packet_header_t), image.size());

        const auto* header = reinterpret_cast<const fs_hybrid_mesh_packet_header_t*>(image.data() + header_offset);

        EXPECT_EQ(header->dst_start_chip_id, suite::kFixture[slot].chip);
        EXPECT_EQ(header->dst_start_mesh_id, suite::kFixture[slot].mesh);
        EXPECT_EQ(header->routing_fields.value, 0x10000u + slot);
        EXPECT_EQ(header->mcast_params_64, 0xA500000000000000ull | static_cast<uint64_t>(slot));
        EXPECT_EQ(header->is_mcast_active, 0u);
    }
}
