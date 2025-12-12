// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/buffers/dispatch.hpp"
#include "gtest/gtest.h"
#include <iostream>
#include "tt_metal/api/tt-metalium/buffer_page_mapping.hpp"
#include <vector>

namespace tt::tt_metal {
// uint32_t start_device_page_offset = dispatch_params.core_page_mapping_it.device_page_offset();
// uint32_t end_device_page_offset = start_device_page_offset + dispatch_params.pages_per_txn;
// while (true) {
//     auto range = dispatch_params.core_page_mapping_it.next_range(end_device_page_offset);
//     if (range.num_pages == 0) {
//         return;
//     }
//     uint64_t src_offset = (uint64_t)(range.host_page_start) * dispatch_params.page_size_to_write;
//     auto cmd_region_offset =
//         dispatch_params.page_size_to_write * (range.device_page_offset - start_device_page_offset);
//     command_sequence.update_cmd_sequence(
//         dst_offset + cmd_region_offset,
//         (char*)(src) + src_offset,
//         range.num_pages * dispatch_params.page_size_to_write);
// }
BufferCorePageMapping core_page_mapping_from_page_mapping(std::vector<uint32_t>& page_mapping) {
    BufferCorePageMapping core_page_mapping;
    core_page_mapping.device_start_page = 0;
    core_page_mapping.num_pages = page_mapping.size();
    core_page_mapping.host_ranges = {
        BufferCorePageMapping::ContiguousHostPages{
            .device_page_offset = 0,
            .host_page_start = 1,
            .num_pages = 3,
        },
        BufferCorePageMapping::ContiguousHostPages{
            .device_page_offset = 8,
            .host_page_start = 4,
            .num_pages = 3,
        },
    };
    return core_page_mapping;
}
void test_write_pages_to_sharded_buffer() {
    uint32_t PADDING = UncompressedBufferPageMapping::PADDING;
    std::vector<uint32_t> page_mapping = {
        1, 2, 3, PADDING, PADDING, PADDING, PADDING, PADDING, 4, 5, 6, PADDING, PADDING, PADDING, PADDING, PADDING};
    uint32_t pages_per_txn = 2;
    const auto core_page_mapping = core_page_mapping_from_page_mapping(page_mapping);
    // BufferCorePageMapping core_page_mapping{
    //     .device_start_page = 0,
    //     .num_pages = 16,
    //     .host_ranges =
    //         {
    //             BufferCorePageMapping::ContiguousHostPages{
    //                 .device_page_offset = 0,
    //                 .host_page_start = 1,
    //                 .num_pages = 3,
    //             },
    //             BufferCorePageMapping::ContiguousHostPages{
    //                 .device_page_offset = 8,
    //                 .host_page_start = 4,
    //                 .num_pages = 3,
    //             },
    //         },
    // };

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
        std::cout << "start_device_page_offset=" << start_device_page_offset << std::endl;
        std::cout << "end_device_page_offset=" << end_device_page_offset << std::endl;

        while (true) {
            auto range = core_it.next_range(end_device_page_offset);
            std::cout << "range.num_pages=" << range.num_pages << std::endl;
            if (range.num_pages == 0) {
                dst_offset = end_device_page_offset * page_size_to_write;
                break;
            }
            uint64_t src_offset = (uint64_t)(range.host_page_start) * page_size_to_write;
            auto cmd_region_offset = page_size_to_write * (range.device_page_offset - start_device_page_offset);
            // uint32_t num_pages_written = range.device_page_offset - start_device_page_offset;
            // auto cmd_region_offset = page_size_to_write * num_pages_written;
            std::cout << "cmd_region_offset=" << cmd_region_offset << std::endl;
            std::cout << "src_offset=" << src_offset << std::endl;
            std::cout << "dst_offset=" << dst_offset << std::endl;
            for (uint32_t i = 0; i < range.num_pages; i++) {
                src_offsets_buffer_core_mapping.push_back(src_offset + i * page_size_to_write);
                dst_offsets_buffer_core_mapping.push_back(dst_offset + cmd_region_offset + i * page_size_to_write);
            }

            // command_sequence.update_cmd_sequence(
            //     dst_offset + cmd_region_offset,
            //     (char*)(src) + src_offset,
            //     range.num_pages * dispatch_params.page_size_to_write);
            dst_offset = end_device_page_offset * page_size_to_write;  // update for the next transaction
        }
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

    // std::cout << "foo" << std::endl;
    // EXPECT_TRUE(1 == 0) << "failed assertion 1 == 0";
}

TEST(DispatchBuffer, WritePagesToShardedBuffer) { test_write_pages_to_sharded_buffer(); }

}  // namespace tt::tt_metal
