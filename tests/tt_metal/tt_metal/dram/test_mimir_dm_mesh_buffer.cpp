// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MM-8 host DRAM MeshBuffer port (test_quasar_mesh_buffers DRAM cases, plan D3). InterleavedPlacement writes one page
// per bank through the MeshBuffer host API and checks each page with a raw view read at
// get_address_offset(view) + address, which does not use the buffer's page mapping, then reads the buffer back.
// WriteRead and MultipleWriteReadRounds port MeshBufferWriteReadDRAM and MeshBufferMultipleWriteReadRoundsDRAM.

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <cstdint>
#include <exception>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_device.hpp>
#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/dram/mimir_dm_common.hpp"
#include "tt_metal/tt_metal/dram/mimir_hw_state.hpp"
#include "tt_metal/tt_metal/dram/mimir_patterns.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"

namespace tt::tt_metal::mimir_test::dm_mesh_buffer {

// InterleavedPlacement: one page per bank, a few KiB each, since the emulator's host writes are slow.
inline constexpr uint32_t kPlacementPageBytes = 4 * 1024;
inline constexpr uint32_t kPlacementPageWords = kPlacementPageBytes / sizeof(uint32_t);
// Upstream sizes (test_quasar_mesh_buffers.cpp): 1024 words in 8 pages, so two pages per bank on four banks.
inline constexpr uint32_t kUpstreamWords = 1024;
inline constexpr uint32_t kUpstreamBytes = kUpstreamWords * sizeof(uint32_t);
inline constexpr uint32_t kUpstreamPageBytes = kUpstreamBytes / 8;
inline constexpr uint32_t kUpstreamRounds = 10;

std::string descriptor_param_name(const ::testing::TestParamInfo<std::string>& info) { return info.param; }

std::shared_ptr<distributed::MeshBuffer> make_interleaved_dram_buffer(
    distributed::MeshDevice& mesh, uint32_t size_bytes, uint32_t page_bytes) {
    return distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = size_bytes},
        distributed::DeviceLocalBufferConfig{.page_size = page_bytes, .buffer_type = BufferType::DRAM},
        &mesh);
}

// Bank -> DRAM view, from the allocator's DRAM bank list of each view's logical DRAM core. The interleaved host write
// passes the bank index straight to write_dram_vec as the view, so this mapping is not shared with it.
std::vector<std::optional<uint32_t>> view_of_each_bank(
    const Allocator& allocator, const metal_SocDescriptor& soc, uint32_t num_banks) {
    std::vector<std::optional<uint32_t>> bank_views(num_banks);
    for (uint32_t view = 0; view < soc.get_num_dram_views(); ++view) {
        const CoreCoord logical_core = soc.get_logical_core_for_dram_view(static_cast<int>(view));
        for (uint32_t bank_id : allocator.get_bank_ids_from_logical_core(BufferType::DRAM, logical_core)) {
            if (bank_id < num_banks) {
                bank_views[bank_id] = view;
            }
        }
    }
    return bank_views;
}

// MM0-style raw view read at get_address_offset(view) + view_offset, guarded against leaving the view.
::testing::AssertionResult raw_view_page_matches(
    distributed::MeshDevice& unit_mesh,
    const metal_SocDescriptor& soc,
    std::optional<uint32_t> view,
    uint64_t view_offset,
    const std::vector<uint32_t>& expected,
    const std::string& label) {
    const uint32_t size_bytes = static_cast<uint32_t>(expected.size() * sizeof(uint32_t));
    if (!view.has_value()) {
        return ::testing::AssertionFailure() << label << ": no DRAM view lists this bank";
    }
    if (view.value() >= soc.get_num_dram_views() || view_offset + size_bytes > soc.dram_view_size) {
        return ::testing::AssertionFailure() << fmt::format(
                   "{}: view {} offset 0x{:x} +{} B is outside the {} views of 0x{:x} B",
                   label,
                   view.value(),
                   view_offset,
                   size_bytes,
                   soc.get_num_dram_views(),
                   soc.dram_view_size);
    }
    std::vector<uint32_t> actual;
    try {
        mimir_contract::read_gddr_view(unit_mesh, view.value(), view_offset, size_bytes, actual);
    } catch (const std::exception& error) {
        return ::testing::AssertionFailure() << label << ": read threw: " << error.what();
    }
    return compare_words(label, expected, actual);
}

void record_result(const std::string& name, const ::testing::AssertionResult& result) {
    ::testing::Test::RecordProperty(name + "_result", result ? "pass" : "fail");
    ::testing::Test::RecordProperty(name + "_detail", result ? "" : result.message());
}

class MimirDmMeshBufferDram : public ::testing::TestWithParam<std::string> {};

TEST_P(MimirDmMeshBufferDram, InterleavedPlacement) {
    skip_unless_programmable_dram_package();
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    std::shared_ptr<distributed::MeshDevice> mesh = distributed::MeshDevice::create_unit_mesh(0);
    ScopedMesh mesh_closer(mesh);
    IDevice* device = mesh->get_devices().front();
    const metal_SocDescriptor& soc = soc_desc_of(device);
    const Allocator& allocator = *device->allocator();
    const uint32_t num_banks = allocator.get_num_banks(BufferType::DRAM);
    ASSERT_GT(num_banks, 0u);

    std::shared_ptr<distributed::MeshBuffer> mesh_buffer =
        make_interleaved_dram_buffer(*mesh, num_banks * kPlacementPageBytes, kPlacementPageBytes);
    const Buffer& buffer = *mesh_buffer->get_reference_buffer();
    ASSERT_EQ(buffer.num_pages(), num_banks);
    ASSERT_EQ(buffer.aligned_page_size(), kPlacementPageBytes);

    std::vector<std::vector<uint32_t>> bank_pages;
    std::vector<uint32_t> source;
    for (uint32_t bank_id = 0; bank_id < num_banks; ++bank_id) {
        bank_pages.push_back(
            make_pattern(MimirPattern::Prng, kPlacementPageWords, 0xB0000000u + bank_id, mimir_test_seed()));
        source.insert(source.end(), bank_pages.back().begin(), bank_pages.back().end());
    }
    distributed::MeshCommandQueue& command_queue = mesh->mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(command_queue, mesh_buffer, source, /*blocking=*/true);

    const std::vector<std::optional<uint32_t>> bank_views = view_of_each_bank(allocator, soc, num_banks);
    for (uint32_t bank_id = 0; bank_id < num_banks; ++bank_id) {
        const std::string label = fmt::format(
            "bank {} at view {} address 0x{:x}",
            bank_id,
            bank_views[bank_id].has_value() ? fmt::format("{}", bank_views[bank_id].value()) : "none",
            buffer.address());
        const ::testing::AssertionResult placement =
            raw_view_page_matches(*mesh, soc, bank_views[bank_id], buffer.address(), bank_pages[bank_id], label);
        record_result(fmt::format("bank{}", bank_id), placement);
        EXPECT_TRUE(placement);
    }

    std::vector<uint32_t> round_trip;
    distributed::EnqueueReadMeshBuffer(command_queue, round_trip, mesh_buffer, /*blocking=*/true);
    const ::testing::AssertionResult round_trip_match = compare_words("MeshBuffer round trip", source, round_trip);
    record_result("round_trip", round_trip_match);
    EXPECT_TRUE(round_trip_match);
}

TEST_P(MimirDmMeshBufferDram, WriteRead) {
    skip_unless_programmable_dram_package();
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    std::shared_ptr<distributed::MeshDevice> mesh = distributed::MeshDevice::create_unit_mesh(0);
    ScopedMesh mesh_closer(mesh);
    distributed::MeshCommandQueue& command_queue = mesh->mesh_command_queue();
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer =
        make_interleaved_dram_buffer(*mesh, kUpstreamBytes, kUpstreamPageBytes);

    std::vector<uint32_t> source(kUpstreamWords);
    // Seeded per run: the emulator's DRAM keeps the last run's data at the same address.
    std::iota(source.begin(), source.end(), 0xabcd0000u ^ (mimir_test_seed() * 0x9E3779B9u));
    distributed::EnqueueWriteMeshBuffer(command_queue, mesh_buffer, source);

    std::vector<uint32_t> destination;
    distributed::EnqueueReadMeshBuffer(command_queue, destination, mesh_buffer, /*blocking=*/true);
    const ::testing::AssertionResult match = compare_words("MeshBuffer write/read", source, destination);
    record_result("round_trip", match);
    EXPECT_TRUE(match);
}

TEST_P(MimirDmMeshBufferDram, MultipleWriteReadRounds) {
    skip_unless_programmable_dram_package();
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    std::shared_ptr<distributed::MeshDevice> mesh = distributed::MeshDevice::create_unit_mesh(0);
    ScopedMesh mesh_closer(mesh);
    distributed::MeshCommandQueue& command_queue = mesh->mesh_command_queue();
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer =
        make_interleaved_dram_buffer(*mesh, kUpstreamBytes, kUpstreamPageBytes);

    for (uint32_t round = 0; round < kUpstreamRounds; ++round) {
        std::vector<uint32_t> source(kUpstreamWords);
        std::iota(source.begin(), source.end(), round * 1000u);
        distributed::EnqueueWriteMeshBuffer(command_queue, mesh_buffer, source);

        std::vector<uint32_t> destination;
        distributed::EnqueueReadMeshBuffer(command_queue, destination, mesh_buffer, /*blocking=*/true);
        const ::testing::AssertionResult match =
            compare_words(fmt::format("MeshBuffer round {}", round), source, destination);
        record_result(fmt::format("round{}", round), match);
        ASSERT_TRUE(match);
    }
}

INSTANTIATE_TEST_SUITE_P(
    MM8, MimirDmMeshBufferDram, ::testing::Values(configured_descriptor_name()), descriptor_param_name);

}  // namespace tt::tt_metal::mimir_test::dm_mesh_buffer
