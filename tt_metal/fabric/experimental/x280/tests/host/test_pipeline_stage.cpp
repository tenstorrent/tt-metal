// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_shadow.h"
#include "fabric_shadow_header.hpp"
#include "backends/x280_route_compute.hpp"
#include "backends/x280_route_forward.hpp"
#include "pipeline_stage_common.h"
#include "pipeline_stage_frontend_cxx.hpp"
#include "route_compute_args.h"
#include "route_forward_args.h"
#include "rt/x280_pinned_input.h"
#include "tests/device/suites/pipeline_stage/suite_common.hpp"
#include "tests/host/scoped_env.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

namespace {

using DirectionDecoder = int (*)(const volatile uint8_t*, uint16_t, uint16_t, uint8_t*);

static_assert(std::is_same_v<decltype(&fs_packed_direction), DirectionDecoder>);
static_assert(std::is_same_v<decltype(&fs_original_direction), DirectionDecoder>);

struct Mapping {
    void* address = MAP_FAILED;
    size_t size = 0;

    ~Mapping() {
        if (address != MAP_FAILED) {
            munmap(address, size);
        }
    }
};

template <size_t Size>
void encode_direction(std::array<uint8_t, Size>& bytes, uint16_t index, uint8_t value) {
    const uint32_t first_bit = static_cast<uint32_t>(index) * FS_BITS_PER_COMPRESSED_ENTRY;
    for (uint32_t bit = 0; bit < FS_BITS_PER_COMPRESSED_ENTRY; ++bit) {
        const uint32_t position = first_bit + bit;
        const uint32_t byte_index = position / FS_BITS_PER_BYTE;
        const uint32_t bit_in_byte = position % FS_BITS_PER_BYTE;
        const uint8_t mask = static_cast<uint8_t>(1u << bit_in_byte);
        bytes[byte_index] = static_cast<uint8_t>(
            (bytes[byte_index] & static_cast<uint8_t>(~mask)) | (((value >> bit) & 1u) != 0u ? mask : 0u));
    }
}

uint8_t expected_original_direction(uint8_t compressed) {
    constexpr std::array<uint8_t, 8> expected = {
        FS_DIRECTION_EAST,
        FS_DIRECTION_WEST,
        FS_DIRECTION_NORTH,
        FS_DIRECTION_SOUTH,
        FS_DIRECTION_Z,
        FS_INVALID_DIRECTION,
        FS_INVALID_ROUTING_TABLE_ENTRY,
        FS_INVALID_ROUTING_TABLE_ENTRY,
    };
    return expected[compressed];
}

struct FixtureDestination {
    uint16_t chip;
    uint16_t mesh;
    uint8_t direction;
};

constexpr uint16_t kFixtureMesh = 513u;
constexpr std::array<FixtureDestination, 16> kFixtureDestinations = {{
    {0u, kFixtureMesh, 0u},
    {1u, kFixtureMesh, 1u},
    {2u, kFixtureMesh, 2u},
    {5u, kFixtureMesh, 1u},
    {7u, kFixtureMesh, 3u},
    {8u, kFixtureMesh, 0u},
    {254u, kFixtureMesh, 2u},
    {255u, kFixtureMesh, 3u},
    {255u, 0u, 2u},
    {254u, 1u, 3u},
    {8u, 2u, 0u},
    {7u, 5u, 3u},
    {5u, 8u, 2u},
    {2u, 1021u, 3u},
    {1u, 1022u, 0u},
    {0u, 1023u, 1u},
}};

using ForwardWords = std::array<uint32_t, PIPELINE_STAGE_ROUTE_FORWARD_ARGS_BYTES / 4u>;

ForwardWords forward_words() {
    return {{
        pipeline_stage_cookie_for(UBENCH_SECTION_COMMON),
        17u,
        29u,
        PIPELINE_STAGE_INSTRUMENTATION_OFF,
        0xA1B2C3D4u,
        pipeline_stage_cookie_for(UBENCH_SECTION_BACKEND),
        0x55667780u,
        0x11223344u,
        0x66778880u,
        0x22334455u,
        PIPELINE_STAGE_DIR_COUNT,
        0u,
        0u,
        0u,
        0u,
        0u,
        static_cast<uint32_t>(PIPELINE_STAGE_PREPARED_LIM_BEGIN),
        static_cast<uint32_t>(PIPELINE_STAGE_PREPARED_LIM_BEGIN >> 32u),
        0x0006C000u,
        0u,
        static_cast<uint32_t>(X280_PINNED_INPUT_FIRST_HEADER_BASE),
        static_cast<uint32_t>(X280_PINNED_INPUT_FIRST_HEADER_BASE >> 32u),
    }};
}

TEST(PipelineStageRouteForwardLoad, AcceptsWireOrderAndRetainsValues) {
    alignas(uint64_t) auto words = forward_words();
    pipeline_stage::backend_route_forward::state_t state{};
    ASSERT_TRUE(load_section<pipeline_stage_route_forward_args_block_t>(words.data(), sizeof(words), state.as_uint32));
    EXPECT_TRUE(validate_section(state.args.section));
    EXPECT_TRUE(layout_is_sane(state.args.layout));
    const pipeline_stage_common_args_t expected_common{17u, 29u, PIPELINE_STAGE_INSTRUMENTATION_OFF, 0xA1B2C3D4u};
    EXPECT_EQ(std::memcmp(&state.args.section.common, &expected_common, sizeof(expected_common)), 0);
    EXPECT_EQ(state.args.section.hop_count, PIPELINE_STAGE_DIR_COUNT);
    EXPECT_EQ(state.args.section.backend.result_base, UINT64_C(0x1122334455667780));
    EXPECT_EQ(state.args.section.backend.sample_region_base, UINT64_C(0x2233445566778880));
    EXPECT_EQ(state.args.layout.window.addr, PIPELINE_STAGE_PREPARED_LIM_BEGIN);
    EXPECT_EQ(state.args.layout.window.bytes, 0x0006C000u);
    EXPECT_EQ(state.args.layout.source_base, X280_PINNED_INPUT_FIRST_HEADER_BASE);
}

TEST(PipelineStageRouteForwardValidate, RejectsIncompatibleCookiesAndNonzeroTail) {
    alignas(uint64_t) auto words = forward_words();
    pipeline_stage::backend_route_forward::state_t state{};
    ASSERT_TRUE(load_section<pipeline_stage_route_forward_args_block_t>(words.data(), sizeof(words), state.as_uint32));
    EXPECT_TRUE(validate_section(state.args.section));

    auto s = state.args.section;
    s.cookie_common ^= 1u;
    EXPECT_FALSE(validate_section(s));

    s = state.args.section;
    s.cookie_backend ^= 1u;
    EXPECT_FALSE(validate_section(s));

    for (uint32_t i = 0u; i < 5u; ++i) {
        s = state.args.section;
        s.pad[i] = 1u;
        EXPECT_FALSE(validate_section(s)) << "pad[" << i << "]";
    }
}

TEST(PipelineStageRouteForwardSanity, RejectsUnsafeConfigurations) {
    alignas(uint64_t) auto words = forward_words();
    pipeline_stage::backend_route_forward::state_t valid{};
    ASSERT_TRUE(load_section<pipeline_stage_route_forward_args_block_t>(words.data(), sizeof(words), valid.as_uint32));

    valid.args.section.backend.result_base = PIPELINE_STAGE_PREPARED_LIM_BEGIN + 64u;
    valid.args.section.backend.sample_region_base = 0u;
    EXPECT_TRUE(pipeline_stage::backend_route_forward::sanity_check(valid));

    auto invalid = valid;
    invalid.args.section.common.instrumentation = PIPELINE_STAGE_INSTRUMENTATION_PER_SAMPLE;
    invalid.args.section.backend.sample_region_base = PIPELINE_STAGE_PREPARED_LIM_BEGIN + 128u;
    EXPECT_FALSE(pipeline_stage::backend_route_forward::sanity_check(invalid)) << "per-sample instrumentation";

    invalid = valid;
    invalid.args.section.backend.sample_region_base = PIPELINE_STAGE_PREPARED_LIM_BEGIN + 128u;
    EXPECT_FALSE(pipeline_stage::backend_route_forward::sanity_check(invalid))
        << "a sample region without per-sample instrumentation";

    invalid = valid;
    invalid.args.section.backend.result_base = PIPELINE_STAGE_PREPARED_LIM_END;
    EXPECT_FALSE(pipeline_stage::backend_route_forward::sanity_check(invalid));

    invalid = valid;
    invalid.args.section.backend.result_base = PIPELINE_STAGE_PREPARED_LIM_BEGIN + 65u;
    EXPECT_FALSE(pipeline_stage::backend_route_forward::sanity_check(invalid));
}

TEST(PipelineStageRouteForwardDecode, InvalidCommandHasNoDirection) {
    for (const uint8_t command : {uint8_t{FS_HOP_NOOP}, uint8_t{0xFFu}}) {
        EXPECT_EQ(pipeline_stage::decode_hop(command), PIPELINE_STAGE_DIR_COUNT);
    }
}

TEST(PipelineStageDirections, DecodesEveryValueAtEveryIntraMeshIndex) {
    std::array<uint8_t, sizeof(fs_intra_mesh_direction_table_t)> packed{};

    for (uint8_t value = 0; value <= FS_COMPRESSED_ENTRY_MASK; ++value) {
        packed.fill(0u);
        for (uint16_t index = 0; index < FS_MAX_MESH_SIZE; ++index) {
            encode_direction(packed, index, value);
        }

        for (uint16_t index = 0; index < FS_MAX_MESH_SIZE; ++index) {
            uint8_t compressed = 0xA5u;
            ASSERT_NE(fs_packed_direction(packed.data(), FS_MAX_MESH_SIZE, index, &compressed), 0);
            EXPECT_EQ(compressed, value) << "index=" << index << " compressed=" << static_cast<uint32_t>(value);

            uint8_t original = 0xA5u;
            ASSERT_NE(fs_original_direction(packed.data(), FS_MAX_MESH_SIZE, index, &original), 0);
            EXPECT_EQ(original, expected_original_direction(value))
                << "index=" << index << " compressed=" << static_cast<uint32_t>(value);
        }
    }
}

TEST(PipelineStageDirections, DecodesEveryValueAtEveryInterMeshIndex) {
    std::array<uint8_t, sizeof(fs_inter_mesh_direction_table_t)> packed{};

    for (uint8_t value = 0; value <= FS_COMPRESSED_ENTRY_MASK; ++value) {
        packed.fill(0u);
        for (uint16_t index = 0; index < FS_MAX_NUM_MESHES; ++index) {
            encode_direction(packed, index, value);
        }

        for (uint16_t index = 0; index < FS_MAX_NUM_MESHES; ++index) {
            uint8_t compressed = 0xA5u;
            ASSERT_NE(fs_packed_direction(packed.data(), FS_MAX_NUM_MESHES, index, &compressed), 0);
            EXPECT_EQ(compressed, value) << "index=" << index << " compressed=" << static_cast<uint32_t>(value);
            uint8_t original = 0xA5u;
            ASSERT_NE(fs_original_direction(packed.data(), FS_MAX_NUM_MESHES, index, &original), 0);
            EXPECT_EQ(original, expected_original_direction(value))
                << "index=" << index << " compressed=" << static_cast<uint32_t>(value);
        }
    }
}

TEST(PipelineStageDirections, MixedEntriesPreserveCrossByteNeighborsAndTerminalIndex) {
    std::array<uint8_t, sizeof(fs_inter_mesh_direction_table_t)> packed{};
    for (uint16_t index = 0; index < FS_MAX_NUM_MESHES; ++index) {
        encode_direction(packed, index, static_cast<uint8_t>(index & FS_COMPRESSED_ENTRY_MASK));
    }

    for (uint16_t index = 0; index < FS_MAX_NUM_MESHES; ++index) {
        uint8_t compressed = 0xA5u;
        ASSERT_NE(fs_packed_direction(packed.data(), FS_MAX_NUM_MESHES, index, &compressed), 0);
        EXPECT_EQ(compressed, static_cast<uint8_t>(index & FS_COMPRESSED_ENTRY_MASK)) << "index=" << index;
    }

    for (const uint16_t index : {2u, 5u, 255u, 1023u}) {
        uint8_t compressed = 0xA5u;
        ASSERT_NE(fs_packed_direction(packed.data(), FS_MAX_NUM_MESHES, index, &compressed), 0);
        EXPECT_EQ(compressed, static_cast<uint8_t>(index & FS_COMPRESSED_ENTRY_MASK));
    }
}

TEST(PipelineStageDirections, InvalidIndicesFailBeforeReadingTheTable) {
    const long page_size_result = sysconf(_SC_PAGESIZE);
    ASSERT_GT(page_size_result, 0);
    Mapping unreadable{
        mmap(nullptr, static_cast<size_t>(page_size_result), PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0),
        static_cast<size_t>(page_size_result)};
    ASSERT_NE(unreadable.address, MAP_FAILED);
    const auto* const table = static_cast<const volatile uint8_t*>(unreadable.address);

    struct InvalidIndex {
        uint16_t count;
        uint16_t index;
    };
    constexpr std::array<InvalidIndex, 5> invalid = {{
        {0u, 0u},
        {FS_MAX_MESH_SIZE, FS_MAX_MESH_SIZE},
        {FS_MAX_MESH_SIZE, FS_MAX_MESH_SIZE + 1u},
        {FS_MAX_NUM_MESHES, FS_MAX_NUM_MESHES},
        {FS_MAX_NUM_MESHES, std::numeric_limits<uint16_t>::max()},
    }};

    for (const auto& input : invalid) {
        uint8_t output = 0xA5u;
        EXPECT_EQ(fs_packed_direction(table, input.count, input.index, &output), 0);
        EXPECT_EQ(output, FS_INVALID_ROUTING_TABLE_ENTRY);

        output = 0xA5u;
        EXPECT_EQ(fs_original_direction(table, input.count, input.index, &output), 0);
        EXPECT_EQ(output, FS_INVALID_ROUTING_TABLE_ENTRY);
    }
}

TEST(PipelineStageDirections, NullPointersFailWithoutReadingOrLeavingSuccess) {
    const long page_size_result = sysconf(_SC_PAGESIZE);
    ASSERT_GT(page_size_result, 0);
    Mapping unreadable{
        mmap(nullptr, static_cast<size_t>(page_size_result), PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0),
        static_cast<size_t>(page_size_result)};
    ASSERT_NE(unreadable.address, MAP_FAILED);
    const auto* const table = static_cast<const volatile uint8_t*>(unreadable.address);

    EXPECT_EQ(fs_packed_direction(table, FS_MAX_MESH_SIZE, 0u, nullptr), 0);
    EXPECT_EQ(fs_original_direction(table, FS_MAX_MESH_SIZE, 0u, nullptr), 0);

    uint8_t output = 0xA5u;
    EXPECT_EQ(fs_packed_direction(nullptr, FS_MAX_MESH_SIZE, 0u, &output), 0);
    EXPECT_EQ(output, FS_INVALID_ROUTING_TABLE_ENTRY);

    output = 0xA5u;
    EXPECT_EQ(fs_original_direction(nullptr, FS_MAX_MESH_SIZE, 0u, &output), 0);
    EXPECT_EQ(output, FS_INVALID_ROUTING_TABLE_ENTRY);
}

// The route_compute access pattern: the device and the host oracle share
// next_slot, so its sequence is pinned here. From run_nonce 0xC01 the xorshift32
// generator selects these slots first and reaches all 16 within 43 steps.
TEST(PipelineStageRouteComputeGenerator, SelectsAPinnedSequenceCoveringEverySlot) {
    uint32_t generator = 0xC01u;
    constexpr std::array<uint32_t, 8> kFirst = {1u, 13u, 0u, 3u, 8u, 8u, 2u, 11u};
    for (size_t i = 0; i < kFirst.size(); ++i) {
        EXPECT_EQ(pipeline_stage::next_slot(generator), kFirst[i]) << "step " << i;
    }
    generator = 0xC01u;
    uint32_t seen = 0u;
    for (uint32_t step = 0; step < 43u; ++step) {
        seen |= 1u << pipeline_stage::next_slot(generator);
    }
    EXPECT_EQ(seen, 0xFFFFu) << "slots selected in 43 steps: 0x" << std::hex << seen;
}

TEST(PipelineStageDirections, FixtureDirectionsDecodeAndWrongTablesDiffer) {
    std::array<uint8_t, sizeof(fs_intra_mesh_direction_table_t)> intra{};
    std::array<uint8_t, sizeof(fs_inter_mesh_direction_table_t)> inter{};
    std::array<bool, FS_MAX_MESH_SIZE> intra_selected{};
    std::array<bool, FS_MAX_NUM_MESHES> inter_selected{};

    for (uint16_t index = 0; index < FS_MAX_MESH_SIZE; ++index) {
        encode_direction(intra, index, FS_COMPRESSED_INVALID_ROUTING_TABLE_ENTRY);
    }
    for (uint16_t index = 0; index < FS_MAX_NUM_MESHES; ++index) {
        encode_direction(inter, index, FS_COMPRESSED_INVALID_ROUTING_TABLE_ENTRY);
    }

    for (const auto& destination : kFixtureDestinations) {
        if (destination.mesh == kFixtureMesh) {
            encode_direction(intra, destination.chip, destination.direction);
            intra_selected[destination.chip] = true;
        } else {
            encode_direction(inter, destination.mesh, destination.direction);
            inter_selected[destination.mesh] = true;
        }
    }

    for (const auto& destination : kFixtureDestinations) {
        uint8_t direction = 0xA5u;
        if (destination.mesh == kFixtureMesh) {
            ASSERT_NE(fs_original_direction(intra.data(), FS_MAX_MESH_SIZE, destination.chip, &direction), 0);
            EXPECT_EQ(direction, destination.direction);
            ASSERT_NE(fs_original_direction(inter.data(), FS_MAX_NUM_MESHES, destination.chip, &direction), 0);
            EXPECT_NE(direction, destination.direction);
        } else {
            ASSERT_NE(fs_original_direction(inter.data(), FS_MAX_NUM_MESHES, destination.mesh, &direction), 0);
            EXPECT_EQ(direction, destination.direction);
            if (destination.mesh < FS_MAX_MESH_SIZE) {
                ASSERT_NE(fs_original_direction(intra.data(), FS_MAX_MESH_SIZE, destination.mesh, &direction), 0);
                EXPECT_NE(direction, destination.direction);
            }
        }
    }

    for (uint16_t index = 0; index < FS_MAX_MESH_SIZE; ++index) {
        if (!intra_selected[index]) {
            uint8_t direction = 0xA5u;
            ASSERT_NE(fs_original_direction(intra.data(), FS_MAX_MESH_SIZE, index, &direction), 0);
            EXPECT_EQ(direction, FS_INVALID_ROUTING_TABLE_ENTRY);
        }
    }
    for (uint16_t index = 0; index < FS_MAX_NUM_MESHES; ++index) {
        if (!inter_selected[index]) {
            uint8_t direction = 0xA5u;
            ASSERT_NE(fs_original_direction(inter.data(), FS_MAX_NUM_MESHES, index, &direction), 0);
            EXPECT_EQ(direction, FS_INVALID_ROUTING_TABLE_ENTRY);
        }
    }
}

constexpr host_device_layout_t kLimLayout{
    {PIPELINE_STAGE_PREPARED_LIM_BEGIN, PIPELINE_STAGE_PREPARED_LIM_END - PIPELINE_STAGE_PREPARED_LIM_BEGIN},
    X280_PINNED_INPUT_FIRST_HEADER_BASE,
};

TEST(PipelineStageLimSanity, PreparedLimBoundsRetainBuildTimeSanity) {
    EXPECT_EQ(PIPELINE_STAGE_PREPARED_LIM_BEGIN, UINT64_C(0x08154000));
    EXPECT_EQ(PIPELINE_STAGE_PREPARED_LIM_END, UINT64_C(0x081C0000));
    EXPECT_EQ(PIPELINE_STAGE_SAMPLES_REGION_BYTES, 32832u);
    EXPECT_EQ(sizeof(host_device_layout_t), 24u);
    EXPECT_EQ(offsetof(host_device_layout_t, window.addr), 0u);
    EXPECT_EQ(offsetof(host_device_layout_t, window.bytes), 8u);
    EXPECT_EQ(offsetof(host_device_layout_t, source_base), 16u);
}

TEST(PipelineStageLocalSpans, AcceptsCompleteAlignedHalfOpenRanges) {
    const std::array<std::pair<uint64_t, uint64_t>, 4> valid = {{
        {PIPELINE_STAGE_PREPARED_LIM_BEGIN, 1u},
        {PIPELINE_STAGE_PREPARED_LIM_BEGIN, 64u},
        {PIPELINE_STAGE_PREPARED_LIM_END - 64u, 64u},
        {PIPELINE_STAGE_PREPARED_LIM_BEGIN, PIPELINE_STAGE_PREPARED_LIM_END - PIPELINE_STAGE_PREPARED_LIM_BEGIN},
    }};

    for (const auto& [addr, bytes] : valid) {
        EXPECT_TRUE(is_span_contained(addr, bytes, kLimLayout)) << "address=" << addr << " bytes=" << bytes;
    }
}

TEST(PipelineStageLocalSpans, AcceptsDeclaredComputeAndForwardSpanSets) {
    const uint64_t begin = PIPELINE_STAGE_PREPARED_LIM_BEGIN;
    const std::array<std::pair<uint64_t, uint64_t>, 4> compute = {{
        {begin, 88u},
        {begin + 128u, X280_PINNED_INPUT_SPAN},
        {begin + 4416u, 64u},
        {begin + 4480u, PIPELINE_STAGE_SAMPLES_REGION_BYTES},
    }};
    for (const auto& [addr, bytes] : compute) {
        EXPECT_TRUE(is_span_contained(addr, bytes, kLimLayout));
    }
    for (size_t i = 0; i < compute.size(); ++i) {
        for (size_t j = i + 1; j < compute.size(); ++j) {
            EXPECT_TRUE(spans_disjoint(compute[i].first, compute[i].second, compute[j].first, compute[j].second));
        }
    }

    const std::array<std::pair<uint64_t, uint64_t>, 2> forward = {{
        {begin, PIPELINE_STAGE_ROUTE_FORWARD_ARGS_BYTES},
        {begin + 128u, 64u},
    }};
    for (const auto& [addr, bytes] : forward) {
        EXPECT_TRUE(is_span_contained(addr, bytes, kLimLayout));
    }
    for (size_t i = 0; i < forward.size(); ++i) {
        for (size_t j = i + 1; j < forward.size(); ++j) {
            EXPECT_TRUE(spans_disjoint(forward[i].first, forward[i].second, forward[j].first, forward[j].second));
        }
    }
}

TEST(PipelineStageLocalSpans, RejectsEmptyOutOfBoundsAndOverflowingRanges) {
    constexpr std::array<std::pair<uint64_t, uint64_t>, 8> invalid = {{
        {PIPELINE_STAGE_PREPARED_LIM_BEGIN, 0u},
        {PIPELINE_STAGE_PREPARED_LIM_BEGIN - 1u, 1u},
        {PIPELINE_STAGE_PREPARED_LIM_BEGIN, PIPELINE_STAGE_PREPARED_LIM_END - PIPELINE_STAGE_PREPARED_LIM_BEGIN + 1u},
        {PIPELINE_STAGE_PREPARED_LIM_END - 63u, 64u},
        {PIPELINE_STAGE_PREPARED_LIM_END, 1u},
        {PIPELINE_STAGE_PREPARED_LIM_END + 1u, 1u},
        {UINT64_C(0x0A000000), 2704u},
        {std::numeric_limits<uint64_t>::max() - 63u, 64u},
    }};

    for (const auto& [addr, bytes] : invalid) {
        EXPECT_FALSE(is_span_contained(addr, bytes, kLimLayout)) << "address=" << addr << " bytes=" << bytes;
    }
}

TEST(PipelineStageLocalSpans, RejectsCanonicalInputOrResultAsALocalSpan) {
    EXPECT_FALSE(is_span_contained(UINT64_C(0x0A000000), 4288u, kLimLayout));
    EXPECT_FALSE(is_span_contained(UINT64_C(0x0A000000), 64u, kLimLayout));
}

TEST(PipelineStageLocalSpans, AcceptsAdjacentRegionsAndRejectsEveryOverlapOrder) {
    const uint64_t begin = PIPELINE_STAGE_PREPARED_LIM_BEGIN;
    EXPECT_TRUE(spans_disjoint(begin, 64u, begin + 64u, 128u));
    EXPECT_TRUE(spans_disjoint(begin + 64u, 128u, begin, 64u));

    EXPECT_FALSE(spans_disjoint(begin, 64u, begin, 64u));
    EXPECT_FALSE(spans_disjoint(begin, 128u, begin + 64u, 64u));
    EXPECT_FALSE(spans_disjoint(begin + 64u, 64u, begin, 128u));
    EXPECT_FALSE(spans_disjoint(begin, 192u, begin + 64u, 64u));
}

TEST(PipelineStageLayout, PackedTablesAndRoutingImageRetainFixedOffsets) {
    EXPECT_EQ(FS_MAX_MESH_SIZE, 256u);
    EXPECT_EQ(FS_MAX_NUM_MESHES, 1024u);
    EXPECT_EQ(sizeof(fs_intra_mesh_direction_table_t), 96u);
    EXPECT_EQ(sizeof(fs_inter_mesh_direction_table_t), 384u);
    EXPECT_EQ(sizeof(fs_routing_l1_info_t), 2704u);
    EXPECT_EQ(alignof(fs_routing_l1_info_t), 16u);
    EXPECT_EQ(offsetof(fs_routing_l1_info_t, my_mesh_id), 32u);
    EXPECT_EQ(offsetof(fs_routing_l1_info_t, intra_mesh_direction_table), 36u);
    EXPECT_EQ(offsetof(fs_routing_l1_info_t, inter_mesh_direction_table), 132u);
}

TEST(PipelineStageLayout, PacketHeaderRetainsTheConcreteNinetySixByteAbi) {
    EXPECT_EQ(FS_PACKET_HEADER_BYTES, 96u);
    EXPECT_EQ(FS_ROUTE_BUFFER_CAPACITY, 35u);
    EXPECT_EQ(sizeof(fs_hybrid_mesh_packet_header_t), 96u);
    EXPECT_EQ(alignof(fs_hybrid_mesh_packet_header_t), 16u);
    EXPECT_EQ(offsetof(fs_hybrid_mesh_packet_header_t, routing_fields), 44u);
    EXPECT_EQ(offsetof(fs_hybrid_mesh_packet_header_t, route_buffer), 48u);
    EXPECT_EQ(offsetof(fs_hybrid_mesh_packet_header_t, dst_start_chip_id), 83u);
    EXPECT_EQ(offsetof(fs_hybrid_mesh_packet_header_t, dst_start_mesh_id), 85u);
}

TEST(PipelineStageLayout, SeedForwardHeaderOverwritesTheCompleteByteImage) {
    fs_hybrid_mesh_packet_header_t actual;
    std::memset(&actual, 0xA5, sizeof(actual));
    fs_seed_forward_header(&actual);
    fs_hybrid_mesh_packet_header_t expected{};
    constexpr std::array<uint8_t, 4> commands = {
        FS_HOP_FORWARD_SOUTH, FS_HOP_FORWARD_EAST, FS_HOP_FORWARD_NORTH, FS_HOP_FORWARD_WEST};
    std::memcpy(expected.route_buffer, commands.data(), commands.size());
    for (size_t i = 4u; i < FS_ROUTE_BUFFER_CAPACITY; ++i) {
        expected.route_buffer[i] = FS_HOP_NOOP;
    }
    EXPECT_EQ(std::memcmp(&actual, &expected, sizeof(actual)), 0);
}

TEST(PipelineStageLayout, ResultRecordRetainsItsPublishedCacheLineAbi) {
    EXPECT_EQ(PIPELINE_STAGE_STATE_WINDOW_CLEANUP_FAILED, 12u);
    EXPECT_EQ(PIPELINE_STAGE_STATE_COUNTER_ERROR, 13u);
    EXPECT_EQ(PIPELINE_STAGE_BACKEND_ID_X280_ROUTE_COMPUTE, 0x10200001u);
    EXPECT_EQ(PIPELINE_STAGE_BACKEND_ID_X280_ROUTE_FORWARD, 0x10200002u);
    EXPECT_EQ(PIPELINE_STAGE_LOOKUP_HEADER_COUNT, 16u);
    EXPECT_EQ(PIPELINE_STAGE_LOOKUP_FIXTURE_V1, 1u);
    EXPECT_EQ(sizeof(pipeline_stage_result_t), 64u);
    EXPECT_EQ(offsetof(pipeline_stage_result_t, backend_id), 0u);
    EXPECT_EQ(offsetof(pipeline_stage_result_t, terminal_state), 4u);
    EXPECT_EQ(offsetof(pipeline_stage_result_t, packet_header_bytes), 8u);
    EXPECT_EQ(offsetof(pipeline_stage_result_t, instrumentation), 12u);
    EXPECT_EQ(offsetof(pipeline_stage_result_t, warmup_completed), 16u);
    EXPECT_EQ(offsetof(pipeline_stage_result_t, timed_count), 20u);
    EXPECT_EQ(offsetof(pipeline_stage_result_t, raw_batch_delta), 24u);
    EXPECT_EQ(offsetof(pipeline_stage_result_t, secondary_batch_delta), 32u);
    EXPECT_EQ(offsetof(pipeline_stage_result_t, items_completed), 40u);
    EXPECT_EQ(offsetof(pipeline_stage_result_t, verify_sentinel), 48u);
    EXPECT_EQ(offsetof(pipeline_stage_result_t, reserved), 52u);
    EXPECT_EQ(offsetof(pipeline_stage_result_t, workload_size), 56u);
    EXPECT_EQ(offsetof(pipeline_stage_result_t, stage_witness), 60u);
}

TEST(PipelineStageLayout, LookupWireMatchesLayout) {
    EXPECT_EQ(sizeof(pipeline_stage_route_compute_args_t), 24u);
    EXPECT_EQ(offsetof(pipeline_stage_route_compute_args_t, staged_input_base), 0u);
    EXPECT_EQ(offsetof(pipeline_stage_route_compute_args_t, result_base), 8u);
    EXPECT_EQ(offsetof(pipeline_stage_route_compute_args_t, sample_region_base), 16u);

    EXPECT_EQ(sizeof(pipeline_stage_route_compute_section_t), 64u);
    EXPECT_EQ(offsetof(pipeline_stage_route_compute_section_t, cookie_common), 0u);
    EXPECT_EQ(offsetof(pipeline_stage_route_compute_section_t, common), 4u);
    EXPECT_EQ(offsetof(pipeline_stage_route_compute_section_t, cookie_backend), 20u);
    EXPECT_EQ(offsetof(pipeline_stage_route_compute_section_t, backend), 24u);
    EXPECT_EQ(offsetof(pipeline_stage_route_compute_section_t, backend.staged_input_base), 24u);
    EXPECT_EQ(offsetof(pipeline_stage_route_compute_section_t, backend.result_base), 32u);
    EXPECT_EQ(offsetof(pipeline_stage_route_compute_section_t, backend.sample_region_base), 40u);
    EXPECT_EQ(offsetof(pipeline_stage_route_compute_section_t, fixture_id), 48u);
    EXPECT_EQ(offsetof(pipeline_stage_route_compute_section_t, pad), 52u);

    // Published extent is the block rounded to a whole cache line, because that
    // is what Arena allocates and therefore what the firmware is told.
    EXPECT_EQ(PIPELINE_STAGE_ROUTE_COMPUTE_ARGS_BYTES, 128u);
    EXPECT_EQ(sizeof(pipeline_stage_route_compute_args_block_t), 88u);
    EXPECT_EQ(offsetof(pipeline_stage_route_compute_args_block_t, section), 0u);
    EXPECT_EQ(offsetof(pipeline_stage_route_compute_args_block_t, layout), 64u);
    EXPECT_EQ(offsetof(pipeline_stage_route_compute_args_block_t, layout.window.addr), 64u);
    EXPECT_EQ(offsetof(pipeline_stage_route_compute_args_block_t, layout.window.bytes), 72u);
    EXPECT_EQ(offsetof(pipeline_stage_route_compute_args_block_t, layout.source_base), 80u);
}

TEST(PipelineStageLayout, ForwardWireMatchesLayout) {
    EXPECT_EQ(PIPELINE_STAGE_ROUTE_FORWARD_ARGS_USED_BYTES, 44u);
    EXPECT_EQ(PIPELINE_STAGE_ROUTE_FORWARD_ARGS_BYTES, 128u);
    EXPECT_EQ(PIPELINE_STAGE_DIR_COUNT, 4u);

    EXPECT_EQ(sizeof(pipeline_stage_route_forward_args_t), 16u);
    EXPECT_EQ(offsetof(pipeline_stage_route_forward_args_t, result_base), 0u);
    EXPECT_EQ(offsetof(pipeline_stage_route_forward_args_t, sample_region_base), 8u);

    EXPECT_EQ(sizeof(pipeline_stage_route_forward_section_t), 64u);
    EXPECT_EQ(offsetof(pipeline_stage_route_forward_section_t, cookie_common), 0u);
    EXPECT_EQ(offsetof(pipeline_stage_route_forward_section_t, common), 4u);
    EXPECT_EQ(offsetof(pipeline_stage_route_forward_section_t, common.warmup_count), 4u);
    EXPECT_EQ(offsetof(pipeline_stage_route_forward_section_t, common.timed_count), 8u);
    EXPECT_EQ(offsetof(pipeline_stage_route_forward_section_t, common.instrumentation), 12u);
    EXPECT_EQ(offsetof(pipeline_stage_route_forward_section_t, common.run_nonce), 16u);
    EXPECT_EQ(offsetof(pipeline_stage_route_forward_section_t, cookie_backend), 20u);
    EXPECT_EQ(offsetof(pipeline_stage_route_forward_section_t, backend), 24u);
    EXPECT_EQ(offsetof(pipeline_stage_route_forward_section_t, backend.result_base), 24u);
    EXPECT_EQ(offsetof(pipeline_stage_route_forward_section_t, backend.sample_region_base), 32u);
    EXPECT_EQ(offsetof(pipeline_stage_route_forward_section_t, hop_count), 40u);
    EXPECT_EQ(offsetof(pipeline_stage_route_forward_section_t, pad), 44u);

    EXPECT_EQ(sizeof(pipeline_stage_route_forward_args_block_t), 88u);
    EXPECT_EQ(offsetof(pipeline_stage_route_forward_args_block_t, section), 0u);
    EXPECT_EQ(offsetof(pipeline_stage_route_forward_args_block_t, layout), 64u);
    EXPECT_EQ(offsetof(pipeline_stage_route_forward_args_block_t, layout.window.addr), 64u);
    EXPECT_EQ(offsetof(pipeline_stage_route_forward_args_block_t, layout.window.bytes), 72u);
    EXPECT_EQ(offsetof(pipeline_stage_route_forward_args_block_t, layout.source_base), 80u);
}

TEST(PipelineStageBatchDelta, RefusesABackwardPairAndAcceptsAForwardWrap) {
    constexpr uint64_t kWrapStart = UINT64_MAX - 4u;  // forward across 2^64: +8
    struct Case {
        const char* name;
        uint64_t wall_start, wall_end, cpu_start, cpu_end;
        bool counter_error;
        uint64_t wall, cpu;
    };
    const Case cases[] = {
        {"forward", 1000u, 1500u, kWrapStart, 3u, false, 500u, 8u},
        {"wall backward", 2000u, 1000u, 10u, 30u, true, 0u, 20u},
        {"cpu backward", 10u, 30u, 2000u, 1000u, true, 20u, 0u},
    };
    for (const Case& c : cases) {
        pipeline_stage_result_t result{};
        result.terminal_state = static_cast<uint32_t>(PIPELINE_STAGE_STATE_TIMED_PLAIN);
        pipeline_stage_publish_batch_delta(&result, c.wall_start, c.wall_end, c.cpu_start, c.cpu_end);
        EXPECT_EQ(
            result.terminal_state,
            static_cast<uint32_t>(
                c.counter_error ? PIPELINE_STAGE_STATE_COUNTER_ERROR : PIPELINE_STAGE_STATE_TIMED_PLAIN))
            << c.name;
        EXPECT_EQ(result.raw_batch_delta, c.wall) << c.name;
        EXPECT_EQ(result.secondary_batch_delta, c.cpu) << c.name;
    }
}

// ---------------------------------------------------------------------------
// pipeline_stage timing contract, pinned from docs/benchmarks/pipeline_stage.md
// ("Timing and result contract", "Bracket overhead control and counter
// monotonicity") and the shared frontend in
// kernels/pipeline_stage/pipeline_stage_frontend_cxx.hpp. A scripted fake
// backend stands in for a member's: it exercises the frontend's sequencing and
// the observables a host reads back, not the backend's own clock guard.
// ---------------------------------------------------------------------------

namespace frontend_probe {

// Stands in for the counter-error terminal state a backend publishes on a
// decreasing timestamp pair. It is the first value after the last published
// enumerator so it collides with no existing state.
constexpr uint32_t kCounterErrorState = static_cast<uint32_t>(PIPELINE_STAGE_STATE_WINDOW_CLEANUP_FAILED) + 1u;

struct state_t {
    pipeline_stage_common_args_t common{};
    pipeline_stage_result_t result{};
    std::unique_ptr<pipeline_stage_samples_t> samples = std::make_unique<pipeline_stage_samples_t>();
    std::vector<uint64_t> bracket_readings;
    size_t bracket_reads = 0;
    std::vector<uint64_t> sample_readings;
    size_t sample_reads = 0;
    uint32_t stage_calls = 0;
    bool delta_published = false;
    uint64_t delta_begin = 0;
    uint64_t delta_end = 0;
    bool stage_ok = true;
    bool after_ok = true;
    bool footer_ok = true;
    // The terminal_state each footer call saw, and the one each barrier published.
    std::vector<uint32_t> footer_saw;
    std::vector<uint32_t> barrier_published;
};

struct Backend {
    using state_t = frontend_probe::state_t;
    using timestamp_t = uint64_t;

    static const pipeline_stage_common_args_t& common(state_t& s) { return s.common; }
    static pipeline_stage_result_t& result(state_t& s) { return s.result; }
    static pipeline_stage_samples_t* samples(state_t& s) { return s.samples.get(); }
    static uint32_t gen_seed(state_t&) { return 0x9E3779B9u; }
    static uint32_t next_index(state_t&, uint32_t& generator, uint32_t i) {
        generator = generator * 1664525u + 1013904223u;
        return (generator >> 28u) + i;
    }
    static bool verify_stage(state_t& s) { return s.stage_ok; }
    static bool verify_after(state_t& s) { return s.after_ok; }
    static timestamp_t measurement_now(state_t& s) {
        if (s.bracket_reads >= s.bracket_readings.size()) {
            ++s.bracket_reads;
            return 0u;
        }
        return s.bracket_readings[s.bracket_reads++];
    }
    static uint64_t sample_now(state_t& s) {
        if (s.sample_reads >= s.sample_readings.size()) {
            ++s.sample_reads;
            return 0u;
        }
        return s.sample_readings[s.sample_reads++];
    }
    static void publish_stage_witness(state_t& s, uint32_t witness) { s.result.stage_witness = witness; }
    // "Bracket overhead control and counter monotonicity": a decreasing pair
    // publishes no interval and latches the counter-error state; otherwise the
    // raw difference, with nothing subtracted.
    static void publish_delta(state_t& s, const timestamp_t& begin, const timestamp_t& end) {
        s.delta_published = true;
        s.delta_begin = begin;
        s.delta_end = end;
        if (end < begin) {
            s.result.terminal_state = kCounterErrorState;
            return;
        }
        s.result.raw_batch_delta = end - begin;
    }
    static bool publish_footer(state_t& s) {
        s.footer_saw.push_back(s.result.terminal_state);
        return s.footer_ok;
    }
    static void publish_barrier(state_t& s) { s.barrier_published.push_back(s.result.terminal_state); }
    static void stage_once(state_t& s, uint32_t index, uint32_t& witness) {
        ++s.stage_calls;
        witness = witness * 31u + index;
    }
};

state_t make_state(uint32_t warmup, uint32_t timed, uint32_t instrumentation) {
    state_t s;
    s.common = {warmup, timed, instrumentation, 0x5EED0001u};
    s.result.backend_id = PIPELINE_STAGE_BACKEND_ID_X280_ROUTE_COMPUTE;
    return s;
}

}  // namespace frontend_probe

// Spec (ubench_finish, kernels/shared/ubench_finish.hpp): each exit runs the
// footer once before its own terminal state is set, then publishes one
// terminal state behind one barrier -- WINDOW_CLEANUP_FAILED when the footer
// failed. A state the timed phase latched stands while the footer runs.
TEST(PipelineStageFrontend, FooterRunsBeforeTheTerminalStateAndItsFailureIsTheTerminalState) {
    using frontend_probe::kCounterErrorState;
    constexpr uint32_t kFailed = PIPELINE_STAGE_STATE_WINDOW_CLEANUP_FAILED;
    struct Case {
        const char* name;
        bool stage_ok, after_ok, footer_ok, counter_error;
        uint32_t footer_saw, terminal;
    } cases[] = {
        {"done", true, true, true, false, PIPELINE_STAGE_STATE_PUBLISH, PIPELINE_STAGE_STATE_DONE},
        {"done, footer fails", true, true, false, false, PIPELINE_STAGE_STATE_PUBLISH, kFailed},
        {"stage mismatch",
         false,
         true,
         true,
         false,
         PIPELINE_STAGE_STATE_VERIFY_STAGE,
         PIPELINE_STAGE_STATE_VERIFY_MISMATCH},
        {"stage mismatch, footer fails", false, true, false, false, PIPELINE_STAGE_STATE_VERIFY_STAGE, kFailed},
        {"after mismatch",
         true,
         false,
         true,
         false,
         PIPELINE_STAGE_STATE_VERIFY_AFTER,
         PIPELINE_STAGE_STATE_VERIFY_MISMATCH},
        {"after mismatch, footer fails", true, false, false, false, PIPELINE_STAGE_STATE_VERIFY_AFTER, kFailed},
        {"latched", true, true, true, true, kCounterErrorState, kCounterErrorState},
        {"latched, footer fails", true, true, false, true, kCounterErrorState, kFailed},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        auto state = frontend_probe::make_state(2u, 5u, PIPELINE_STAGE_INSTRUMENTATION_OFF);
        state.stage_ok = c.stage_ok;
        state.after_ok = c.after_ok;
        state.footer_ok = c.footer_ok;
        state.bracket_readings = c.counter_error ? std::vector<uint64_t>{UINT64_C(0x20), UINT64_C(0x10)}
                                                 : std::vector<uint64_t>{UINT64_C(0x10), UINT64_C(0x20)};
        pipeline_stage::run_member<frontend_probe::Backend>(state);
        EXPECT_EQ(state.footer_saw, std::vector<uint32_t>{c.footer_saw});
        EXPECT_EQ(state.barrier_published, std::vector<uint32_t>{c.terminal});
        EXPECT_EQ(state.result.terminal_state, c.terminal);
    }
}

// "Counter monotonicity" and the result table's terminal_state ("completion or
// explicit failure"): a counter error and a zero raw_batch_delta are what the
// host reads out of the result record after the whole member ran, so the
// frontend must leave a refusal latched by publish_delta in place rather than
// advancing to DONE. The decreasing pair is the fake's, so this covers the
// frontend half of the contract only.
TEST(PipelineStageFrontend, RefusalLatchedByPublishDeltaIsTerminal) {
    auto state = frontend_probe::make_state(2u, 5u, PIPELINE_STAGE_INSTRUMENTATION_OFF);
    state.bracket_readings = {UINT64_C(0x0000000200000010), UINT64_C(0x00000001FFFFFFF0)};

    pipeline_stage::run_member<frontend_probe::Backend>(state);

    ASSERT_TRUE(state.delta_published);
    EXPECT_EQ(state.result.raw_batch_delta, 0u) << "a decreasing pair published a plausible-looking interval";
    EXPECT_EQ(state.result.terminal_state, frontend_probe::kCounterErrorState)
        << "frontend advanced past the counter error latched by publish_delta";
    EXPECT_NE(state.result.terminal_state, static_cast<uint32_t>(PIPELINE_STAGE_STATE_DONE));
    EXPECT_FALSE(ubench_outcome_is_measurement(
        pipeline_stage_outcome(static_cast<pipeline_stage_state_t>(state.result.terminal_state))));
}

// "Timing and result contract" ("Plain timing brackets the batch") and
// "Overhead is not subtracted from measured deltas": the plain bracket is
// exactly two readings with the timed loop between them, handed to
// publish_delta unmodified and in order, and items_completed is the timed
// count; so ticks/item is (end - begin) / N with no overhead figure entering
// from the frontend on either side of the ratio.
TEST(PipelineStageFrontend, PlainBracketHandsRawReadingsToPublishDeltaAndCountsEveryItem) {
    constexpr uint32_t kWarmup = 3u;
    constexpr uint32_t kTimed = 7u;
    constexpr uint64_t kBegin = UINT64_C(0x00000001FFFFFFF0);
    constexpr uint64_t kEnd = UINT64_C(0x0000000200000F30);
    auto state = frontend_probe::make_state(kWarmup, kTimed, PIPELINE_STAGE_INSTRUMENTATION_OFF);
    state.bracket_readings = {kBegin, kEnd};

    pipeline_stage::run_member<frontend_probe::Backend>(state);

    EXPECT_EQ(state.result.terminal_state, static_cast<uint32_t>(PIPELINE_STAGE_STATE_DONE));
    EXPECT_EQ(state.bracket_reads, 2u) << "the plain bracket must read the measurement clock exactly twice";
    ASSERT_TRUE(state.delta_published);
    EXPECT_EQ(state.delta_begin, kBegin);
    EXPECT_EQ(state.delta_end, kEnd);
    EXPECT_EQ(state.result.raw_batch_delta, kEnd - kBegin);
    EXPECT_EQ(state.stage_calls, kWarmup + kTimed);
    EXPECT_EQ(state.result.warmup_completed, kWarmup);
    EXPECT_EQ(state.result.timed_count, kTimed);
    EXPECT_EQ(state.result.items_completed, static_cast<uint64_t>(kTimed));
    EXPECT_EQ(state.result.instrumentation, static_cast<uint32_t>(PIPELINE_STAGE_INSTRUMENTATION_OFF));
    EXPECT_EQ(state.result.packet_header_bytes, FS_PACKET_HEADER_BYTES);
    EXPECT_EQ(state.sample_reads, 0u);

    // The tabulator divides the batch by the record's items, the timed count
    // ("ns/item", "cpu/item"); items_completed must be the same divisor.
    const double ticks_per_item =
        static_cast<double>(state.result.raw_batch_delta) / static_cast<double>(state.result.items_completed);
    EXPECT_DOUBLE_EQ(ticks_per_item, static_cast<double>(kEnd - kBegin) / kTimed);
    EXPECT_DOUBLE_EQ(ticks_per_item, static_cast<double>(state.result.raw_batch_delta) / state.result.timed_count);
}

// "Instrumented timing also records per-item deltas": the batch bracket is
// still two raw readings, every item publishes its own raw difference, and
// samples_completed equals items_completed, so per-sample and batch figures
// share one divisor.
TEST(PipelineStageFrontend, InstrumentedBracketPublishesEverySampleAndTheBatchPair) {
    constexpr uint32_t kWarmup = 1u;
    constexpr uint32_t kTimed = 5u;
    constexpr uint64_t kBegin = 1000u;
    constexpr uint64_t kEnd = 1000u + 5u * 40u;
    auto state = frontend_probe::make_state(kWarmup, kTimed, PIPELINE_STAGE_INSTRUMENTATION_PER_SAMPLE);
    state.bracket_readings = {kBegin, kEnd};
    for (uint32_t i = 0u; i < kTimed; ++i) {
        state.sample_readings.push_back(kBegin + 40u * i + 3u);
        state.sample_readings.push_back(kBegin + 40u * i + 3u + 11u + i);
    }

    pipeline_stage::run_member<frontend_probe::Backend>(state);

    EXPECT_EQ(state.result.terminal_state, static_cast<uint32_t>(PIPELINE_STAGE_STATE_DONE));
    EXPECT_EQ(state.bracket_reads, 2u);
    EXPECT_EQ(state.sample_reads, 2u * kTimed);
    ASSERT_TRUE(state.delta_published);
    EXPECT_EQ(state.delta_begin, kBegin);
    EXPECT_EQ(state.delta_end, kEnd);
    EXPECT_EQ(state.result.raw_batch_delta, kEnd - kBegin);
    EXPECT_EQ(state.result.instrumentation, static_cast<uint32_t>(PIPELINE_STAGE_INSTRUMENTATION_PER_SAMPLE));
    EXPECT_EQ(state.result.items_completed, static_cast<uint64_t>(kTimed));
    EXPECT_EQ(state.samples->samples_completed, kTimed);
    for (uint32_t i = 0u; i < kTimed; ++i) {
        EXPECT_EQ(state.samples->raw_deltas[i], 11u + i) << "sample " << i;
    }
    EXPECT_EQ(state.samples->raw_deltas[kTimed], 0u) << "frontend wrote past samples_completed";
}

// "Only OK admits a measurement row" (ubench_outcome.h) and the result table's
// terminal_state is "completion or explicit failure": every published state
// maps as documented, and a state value the mapping does not know is never
// admitted as a measurement.
TEST(PipelineStageOutcome, OnlyDoneAdmitsAMeasurementAndUnmappedStatesDoNot) {
    struct Case {
        pipeline_stage_state_t state;
        ubench_outcome_t outcome;
    };
    constexpr std::array<Case, 12> cases = {{
        {PIPELINE_STAGE_STATE_INIT, UBENCH_OUTCOME_IN_PROGRESS},
        {PIPELINE_STAGE_STATE_VERIFY_STAGE, UBENCH_OUTCOME_IN_PROGRESS},
        {PIPELINE_STAGE_STATE_WARMUP, UBENCH_OUTCOME_IN_PROGRESS},
        {PIPELINE_STAGE_STATE_TIMED_PLAIN, UBENCH_OUTCOME_IN_PROGRESS},
        {PIPELINE_STAGE_STATE_TIMED_INSTRUMENTED, UBENCH_OUTCOME_IN_PROGRESS},
        {PIPELINE_STAGE_STATE_PUBLISH, UBENCH_OUTCOME_IN_PROGRESS},
        {PIPELINE_STAGE_STATE_DONE, UBENCH_OUTCOME_OK},
        {PIPELINE_STAGE_STATE_VERIFY_MISMATCH, UBENCH_OUTCOME_PATH_MISMATCH},
        {PIPELINE_STAGE_STATE_ARG_LAYOUT_MISMATCH, UBENCH_OUTCOME_ARG_LAYOUT_MISMATCH},
        {PIPELINE_STAGE_STATE_VERIFY_AFTER, UBENCH_OUTCOME_IN_PROGRESS},
        {PIPELINE_STAGE_STATE_WINDOW_CLEANUP_FAILED, UBENCH_OUTCOME_TRAP},
        {PIPELINE_STAGE_STATE_COUNTER_ERROR, UBENCH_OUTCOME_TRAP},
    }};
    for (const Case& c : cases) {
        EXPECT_EQ(pipeline_stage_outcome(c.state), c.outcome) << "state " << c.state;
        EXPECT_EQ(ubench_outcome_is_measurement(pipeline_stage_outcome(c.state)), c.state == PIPELINE_STAGE_STATE_DONE);
    }
    // 8 and 9 are the enumeration's gap and 15 lies past it: none names a state.
    for (const uint32_t unmapped : {8u, 9u, 15u}) {
        const auto outcome = pipeline_stage_outcome(static_cast<pipeline_stage_state_t>(unmapped));
        EXPECT_FALSE(ubench_outcome_is_measurement(outcome)) << "state " << unmapped;
    }
}

using ComputeWords = std::array<uint32_t, PIPELINE_STAGE_ROUTE_COMPUTE_ARGS_BYTES / 4u>;

ComputeWords compute_words() {
    return {{
        pipeline_stage_cookie_for(UBENCH_SECTION_COMMON),
        128u,
        16384u,
        PIPELINE_STAGE_INSTRUMENTATION_OFF,
        0x0BADF00Du,
        pipeline_stage_cookie_for(UBENCH_SECTION_BACKEND),
        0x08154000u,
        0x00000000u,
        0x08155040u,
        0x00000000u,
        0x00000000u,
        0x00000000u,
        PIPELINE_STAGE_LOOKUP_FIXTURE_V1,
        0u,
        0u,
        0u,
        static_cast<uint32_t>(PIPELINE_STAGE_PREPARED_LIM_BEGIN),
        static_cast<uint32_t>(PIPELINE_STAGE_PREPARED_LIM_BEGIN >> 32u),
        0x0006C000u,
        0u,
        0x08154000u,
        0u,
    }};
}

TEST(PipelineStageRouteComputeLoad, AcceptsWireAndRetainsValues) {
    alignas(uint64_t) const auto words = compute_words();
    pipeline_stage::backend_route_compute::state_t state{};
    ASSERT_TRUE(load_section<pipeline_stage_route_compute_args_block_t>(words.data(), sizeof(words), state.as_uint32));
    EXPECT_TRUE(validate_section(state.args.section));
    EXPECT_TRUE(layout_is_sane(state.args.layout));
    const pipeline_stage_common_args_t expected_common{128u, 16384u, PIPELINE_STAGE_INSTRUMENTATION_OFF, 0x0BADF00Du};
    EXPECT_EQ(std::memcmp(&state.args.section.common, &expected_common, sizeof(expected_common)), 0);
    EXPECT_EQ(state.args.section.fixture_id, PIPELINE_STAGE_LOOKUP_FIXTURE_V1);
    EXPECT_EQ(state.args.section.backend.staged_input_base, UINT64_C(0x08154000));
    EXPECT_EQ(state.args.section.backend.result_base, UINT64_C(0x08155040));
    EXPECT_EQ(state.args.section.backend.sample_region_base, 0u);
    EXPECT_EQ(state.args.layout.window.addr, PIPELINE_STAGE_PREPARED_LIM_BEGIN);
    EXPECT_EQ(state.args.layout.window.bytes, 0x0006C000u);
    EXPECT_EQ(state.args.layout.source_base, UINT64_C(0x08154000));
}

TEST(PipelineStageRouteComputeValidate, RejectsIncompatibleCookiesAndPad) {
    alignas(uint64_t) const auto words = compute_words();
    pipeline_stage::backend_route_compute::state_t state{};
    ASSERT_TRUE(load_section<pipeline_stage_route_compute_args_block_t>(words.data(), sizeof(words), state.as_uint32));
    EXPECT_TRUE(validate_section(state.args.section));

    auto s = state.args.section;
    s.cookie_common ^= 1u;
    EXPECT_FALSE(validate_section(s));

    s = state.args.section;
    s.cookie_backend ^= 1u;
    EXPECT_FALSE(validate_section(s));

    s = state.args.section;
    s.pad[0] = 1u;
    EXPECT_FALSE(validate_section(s));

    s = state.args.section;
    s.pad[1] = 1u;
    EXPECT_FALSE(validate_section(s));

    s = state.args.section;
    s.pad[2] = 1u;
    EXPECT_FALSE(validate_section(s));
}

TEST(PipelineStageLoadSection, EnforcesExtentAndAlignmentGuards) {
    using Block = pipeline_stage_route_compute_args_block_t;
    // The host publishes a whole-cache-line extent, so the source is larger
    // than the block it carries. The destination is sized by the block.
    alignas(uint64_t) const auto words = compute_words();
    static_assert(sizeof(words) > sizeof(Block));
    uint32_t dest[sizeof(Block) / sizeof(uint32_t)];
    std::memset(dest, 0xA5, sizeof(dest));

    EXPECT_FALSE((load_section<Block>(nullptr, sizeof(words), dest)));
    for (uint32_t val : dest) {
        EXPECT_EQ(val, 0xA5A5A5A5u);
    }

    EXPECT_FALSE((load_section<Block>(words.data(), sizeof(Block) - 4u, dest)));
    for (uint32_t val : dest) {
        EXPECT_EQ(val, 0xA5A5A5A5u);
    }

    const uint8_t* raw_bytes = reinterpret_cast<const uint8_t*>(words.data());
    EXPECT_FALSE((load_section<Block>(raw_bytes + 4u, sizeof(words), dest)));
    for (uint32_t val : dest) {
        EXPECT_EQ(val, 0xA5A5A5A5u);
    }

    EXPECT_TRUE((load_section<Block>(words.data(), sizeof(words), dest)));
    EXPECT_EQ(std::memcmp(dest, words.data(), sizeof(Block)), 0);
}

TEST(PipelineStageLayoutIsSane, EnforcesWindowAndSourceConstraints) {
    const host_device_layout_t valid{{0x1000u, 0x10000u}, 0x2000u};
    EXPECT_TRUE(layout_is_sane(valid));

    host_device_layout_t invalid = valid;
    invalid.window.bytes = 0u;
    EXPECT_FALSE(layout_is_sane(invalid));

    invalid = valid;
    invalid.window.addr = UINT64_MAX - 100u;
    invalid.window.bytes = 200u;
    EXPECT_FALSE(layout_is_sane(invalid));

    invalid = valid;
    invalid.window.addr = 0x100000000ull;
    invalid.window.bytes = 0x1000u;
    EXPECT_FALSE(layout_is_sane(invalid));

    invalid = valid;
    invalid.window.addr = 0xFFFF0000ull;
    invalid.window.bytes = 0x00020000ull;
    EXPECT_FALSE(layout_is_sane(invalid));

    invalid = valid;
    invalid.source_base = 0u;
    EXPECT_FALSE(layout_is_sane(invalid));

    invalid = valid;
    invalid.source_base = 0x2001ull;
    EXPECT_FALSE(layout_is_sane(invalid));

    invalid = valid;
    invalid.source_base = 0x2004ull;
    EXPECT_FALSE(layout_is_sane(invalid));

    invalid = valid;
    invalid.source_base = 0x100000000ull;
    EXPECT_FALSE(layout_is_sane(invalid));
}

TEST(PipelineStageRefusalPath, BadCookieProducesArgLayoutMismatch) {
    alignas(uint64_t) auto words = forward_words();
    words[0] ^= 0xDEADBEEFu;

    const uint64_t res_base = PIPELINE_STAGE_PREPARED_LIM_BEGIN + 64u;
    words[6] = static_cast<uint32_t>(res_base);
    words[7] = static_cast<uint32_t>(res_base >> 32u);

    pipeline_stage::backend_route_forward::state_t state{};
    state.header = nullptr;

    const bool loaded =
        load_section<pipeline_stage_route_forward_args_block_t>(words.data(), sizeof(words), state.as_uint32);
    EXPECT_TRUE(loaded);

    const bool ok = loaded && validate_section(state.args.section) && layout_is_sane(state.args.layout) &&
                    pipeline_stage::backend_route_forward::sanity_check(state);
    EXPECT_FALSE(ok);
}

// A failure record, pipeline_stage_write_result(result, backend_id,
// terminal_state, reason, 0): the id passed in is the id published,
// terminal_state is carried through, verify_sentinel is the reason (0xA0 for
// an argument-layout mismatch), and every remaining field is zero.
TEST(PipelineStageFailureRecord, CarriesIdStateAndReasonAndZeroesEveryOtherField) {
    constexpr uint32_t kTerminalState = PIPELINE_STAGE_STATE_ARG_LAYOUT_MISMATCH;
    constexpr uint32_t kReason = 0xA0u;
    static_assert(kTerminalState != kReason, "swapped fields must be detectable");

    pipeline_stage_result_t result;
    std::memset(&result, 0xA5, sizeof(result));
    pipeline_stage_write_result(&result, PIPELINE_STAGE_BACKEND_ID_X280_ROUTE_FORWARD, kTerminalState, kReason, 0u);

    // The struct has no padding (64 bytes, offsets pinned by static_assert),
    // so a byte comparison covers every field.
    pipeline_stage_result_t expected{};
    expected.backend_id = PIPELINE_STAGE_BACKEND_ID_X280_ROUTE_FORWARD;
    expected.terminal_state = kTerminalState;
    expected.verify_sentinel = kReason;
    EXPECT_EQ(std::memcmp(&result, &expected, sizeof(expected)), 0);
}

}  // namespace

TEST(PipelineStageTimedCount, PerSampleRefusesASetCountAboveCapacityAndCapsTheDefault) {
    namespace common = x280::suites::pipeline_stage::common;
    constexpr uint32_t kDefault = 16384u;
    constexpr uint32_t kOff = PIPELINE_STAGE_INSTRUMENTATION_OFF;
    constexpr uint32_t kPerSample = PIPELINE_STAGE_INSTRUMENTATION_PER_SAMPLE;
    {
        x280::testing::ScopedEnv env("PIPELINE_STAGE_N", nullptr);
        EXPECT_EQ(common::timed_count_from_env(kOff, kDefault), kDefault);
        EXPECT_EQ(common::timed_count_from_env(kPerSample, kDefault), PIPELINE_STAGE_MAX_SAMPLES);
    }
    {
        x280::testing::ScopedEnv env("PIPELINE_STAGE_N", "4096");
        EXPECT_EQ(common::timed_count_from_env(kPerSample, kDefault), 4096u);
    }
    {
        x280::testing::ScopedEnv env("PIPELINE_STAGE_N", "4097");
        EXPECT_EQ(common::timed_count_from_env(kOff, kDefault), 4097u);
        EXPECT_THROW((void)common::timed_count_from_env(kPerSample, kDefault), std::invalid_argument);
    }
}
