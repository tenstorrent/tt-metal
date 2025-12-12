// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/buffers/dispatch.hpp"
#include "gtest/gtest.h"
#include "tt_metal/api/tt-metalium/buffer_page_mapping.hpp"
#include <vector>

namespace tt::tt_metal {

constexpr uint32_t PADDING = UncompressedBufferPageMapping::PADDING;

// Converts a simple page mapping vector to BufferCorePageMapping.
//
// The input 'page_mapping' is a vector where each index represents a device page,
// and the value at that index is either a host page number or PADDING.
// Contiguous device pages with non-PADDING values are assumed to map to sequential host pages.
//
// Example:
//   {1, 2, 3, PADDING, 4, 5}
// means device pages 0-2 map to host pages 1-3,
// device page 3 is padding,
// and device pages 4-5 map to host pages 4-5.
BufferCorePageMapping core_page_mapping_from_page_mapping(const std::vector<uint32_t>& page_mapping) {
    BufferCorePageMapping core_page_mapping;
    core_page_mapping.device_start_page = 0;
    core_page_mapping.num_pages = page_mapping.size();

    uint32_t device_page_offset = 0;

    for (uint32_t i = 0; i < page_mapping.size(); i++) {
        if (page_mapping[i] != PADDING and (i == page_mapping.size() - 1 or page_mapping[i + 1] == PADDING)) {
            core_page_mapping.host_ranges.push_back(BufferCorePageMapping::ContiguousHostPages{
                .device_page_offset = device_page_offset,
                .host_page_start = page_mapping[device_page_offset],
                .num_pages = i - device_page_offset + 1,
            });
        } else if (i == page_mapping.size() - 1) {
            break;
        } else if (page_mapping[i] == PADDING and page_mapping[i + 1] != PADDING) {
            device_page_offset = i + 1;
        }
    }
    return core_page_mapping;
}

/**
 * Tests the BufferCorePageMapping::Iterator by simulating the dispatch write path.
 * Compares the iterator-based approach against a naive golden reference implementation.
 * Verifies that both approaches produce identical src/dst offset pairs when writing pages from host to device.
 * The pages_per_txn parameter simulates different transaction sizes to test edge cases around padding handling.
 */
void test_BufferCorePageMapping_Iterator(const std::vector<uint32_t>& page_mapping, uint32_t pages_per_txn) {
    const auto core_page_mapping = core_page_mapping_from_page_mapping(page_mapping);

    auto core_it = BufferCorePageMapping::Iterator(&core_page_mapping, 0, 0);
    uint32_t page_size_to_write = 1024;
    uint32_t end_device_page_offset = 0;
    std::vector<uint32_t> src_offsets_buffer_core_mapping;
    std::vector<uint32_t> dst_offsets_buffer_core_mapping;
    std::vector<uint32_t> src_offsets_golden;
    std::vector<uint32_t> dst_offsets_golden;
    uint32_t dst_offset = 0;

    // Path for BufferCorePageMapping::Iterator
    while (core_it.range_index() < core_page_mapping.host_ranges.size()) {
        pages_per_txn = std::min(pages_per_txn, core_page_mapping.num_pages - end_device_page_offset);
        uint32_t start_device_page_offset = core_it.device_page_offset();
        end_device_page_offset = start_device_page_offset + pages_per_txn;

        while (true) {
            auto range = core_it.next_range(end_device_page_offset);
            if (range.num_pages == 0) {
                break;
            }
            uint64_t src_offset = (uint64_t)(range.host_page_start) * page_size_to_write;
            auto cmd_region_offset = page_size_to_write * (range.device_page_offset - start_device_page_offset);
            for (uint32_t i = 0; i < range.num_pages; i++) {
                src_offsets_buffer_core_mapping.push_back(src_offset + (i * page_size_to_write));
                dst_offsets_buffer_core_mapping.push_back(dst_offset + cmd_region_offset + i * page_size_to_write);
            }
        }
        dst_offset = end_device_page_offset * page_size_to_write;  // update for the next transaction
    }

    // Naive Golden Path
    for (uint32_t i = 0; i < page_mapping.size(); i++) {
        if (page_mapping[i] != PADDING) {
            src_offsets_golden.push_back(page_mapping[i] * page_size_to_write);
            dst_offsets_golden.push_back(i * page_size_to_write);
        }
    }
    EXPECT_EQ(src_offsets_buffer_core_mapping, src_offsets_golden);
    EXPECT_EQ(dst_offsets_buffer_core_mapping, dst_offsets_golden);
}

struct Test_BufferCorePageMapping_Iterator_params {
    std::vector<uint32_t> page_mapping;
    uint32_t pages_per_txn;
};

class Test_BufferCorePageMapping_Iterator
    : public ::testing::TestWithParam<Test_BufferCorePageMapping_Iterator_params> {};

TEST_P(Test_BufferCorePageMapping_Iterator, Runs) {
    const auto& p = GetParam();
    test_BufferCorePageMapping_Iterator(p.page_mapping, p.pages_per_txn);
}

INSTANTIATE_TEST_SUITE_P(
    Variants,
    Test_BufferCorePageMapping_Iterator,
    ::testing::Values(
        Test_BufferCorePageMapping_Iterator_params{{1, 2, 3, 4, 5}, 3},
        Test_BufferCorePageMapping_Iterator_params{{1, 2, 3, 4, 5}, 5},
        Test_BufferCorePageMapping_Iterator_params{{1, 2, 3, PADDING, PADDING, 4, 5, 6, PADDING, PADDING}, 2},
        Test_BufferCorePageMapping_Iterator_params{{1, 2, 3, PADDING, PADDING, 4, 5, 6, PADDING, PADDING}, 3},
        Test_BufferCorePageMapping_Iterator_params{{1, 2, 3, PADDING, PADDING, 4, 5, 6, PADDING, PADDING}, 5},
        Test_BufferCorePageMapping_Iterator_params{{1, 2, 3, PADDING, PADDING, 4, 5, 6, PADDING, PADDING}, 7},
        Test_BufferCorePageMapping_Iterator_params{{1, 2, 3, PADDING, PADDING, 4, 5, 6, PADDING, PADDING}, 8},
        Test_BufferCorePageMapping_Iterator_params{
            {1,
             2,
             3,
             PADDING,
             PADDING,
             PADDING,
             PADDING,
             PADDING,
             4,
             5,
             6,
             PADDING,
             PADDING,
             PADDING,
             PADDING,
             PADDING},
            2},
        Test_BufferCorePageMapping_Iterator_params{
            {1,
             2,
             3,
             PADDING,
             PADDING,
             PADDING,
             PADDING,
             PADDING,
             4,
             5,
             6,
             PADDING,
             PADDING,
             PADDING,
             PADDING,
             PADDING},
            3},
        Test_BufferCorePageMapping_Iterator_params{
            {1,
             2,
             3,
             PADDING,
             PADDING,
             PADDING,
             PADDING,
             PADDING,
             4,
             5,
             6,
             PADDING,
             PADDING,
             PADDING,
             PADDING,
             PADDING},
            5},
        Test_BufferCorePageMapping_Iterator_params{
            {1,
             2,
             3,
             PADDING,
             PADDING,
             PADDING,
             PADDING,
             PADDING,
             4,
             5,
             6,
             PADDING,
             PADDING,
             PADDING,
             PADDING,
             PADDING},
            8},
        Test_BufferCorePageMapping_Iterator_params{
            {1,
             2,
             3,
             PADDING,
             PADDING,
             PADDING,
             PADDING,
             PADDING,
             4,
             5,
             6,
             PADDING,
             PADDING,
             PADDING,
             PADDING,
             PADDING},
            9},
        Test_BufferCorePageMapping_Iterator_params{{PADDING, PADDING, 1, 2, PADDING, PADDING, 3, 4}, 1},
        Test_BufferCorePageMapping_Iterator_params{{PADDING, PADDING, 1, 2, PADDING, PADDING, 3, 4}, 2},
        Test_BufferCorePageMapping_Iterator_params{{PADDING, PADDING, 1, 2, PADDING, PADDING, 3, 4}, 4},
        Test_BufferCorePageMapping_Iterator_params{{PADDING, PADDING, 1, 2, PADDING, PADDING, 3, 4}, 5},
        Test_BufferCorePageMapping_Iterator_params{{PADDING, PADDING, 1, 2, PADDING, PADDING, 3, 4}, 8},
        Test_BufferCorePageMapping_Iterator_params{{1}, 1},
        Test_BufferCorePageMapping_Iterator_params{{PADDING, 1, PADDING}, 1},
        Test_BufferCorePageMapping_Iterator_params{{1, 2, 3, PADDING, PADDING}, 1},
        Test_BufferCorePageMapping_Iterator_params{{1, 2, 3, PADDING, PADDING}, 2},
        Test_BufferCorePageMapping_Iterator_params{{1, 2, 3, PADDING, PADDING}, 3},
        Test_BufferCorePageMapping_Iterator_params{{1, 2, 3, PADDING, PADDING}, 4},
        Test_BufferCorePageMapping_Iterator_params{{1, 2, 3, PADDING, PADDING}, 5},
        Test_BufferCorePageMapping_Iterator_params{{PADDING, PADDING, PADDING}, 1},
        Test_BufferCorePageMapping_Iterator_params{{PADDING, PADDING, PADDING}, 2},
        Test_BufferCorePageMapping_Iterator_params{{PADDING, PADDING, PADDING}, 3}));

}  // namespace tt::tt_metal
