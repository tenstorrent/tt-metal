// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0



#include <cstddef>
#include "gtest/gtest.h"
#include "tt_metal/hw/inc/wormhole/noc/noc_parameters.h"

#if !(defined(KERNEL_BUILD) || defined(FW_BUILD))

#define FORCE_INLINE inline __attribute__((always_inline))
#define noc_index 0
#define NOC_XY_ADDR(x, y, addr)                                                                                      \
    ((((uint64_t)(y)) << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) | (((uint64_t)(x)) << NOC_ADDR_LOCAL_BITS) | \
     ((uint64_t)(addr)))
#define NOC_0_X(noc_index, noc_size_x, x) x
#define NOC_0_Y(noc_index, noc_size_y, y) y
#define DYNAMIC_NOC_X(noc, x) NOC_0_X(noc, noc_size_x, (x))
#define DYNAMIC_NOC_Y(noc, y) NOC_0_Y(noc, noc_size_y, (y))

#endif

#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
mapping_table_t map[9] = {0x00000001, 0x00020003, 0x00040200, 0x02010202, 0x02030204, 0x03000301, 0x03020303, 0x04000401, 0x04020403};
uint64_t real_core_x_vals [18] = {0x0, 0x0, 0x0, 0x0, 0x0, 0x2, 0x2, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3, 0x4, 0x4, 0x4, 0x4};
uint64_t real_core_y_vals [18] = {0x0, 0x1, 0x2, 0x3, 0x4, 0x0, 0x1, 0x2, 0x3, 0x4, 0x0, 0x1, 0x2, 0x3, 0x0, 0x1, 0x2, 0x3};
uint32_t test_pages [12] = {0, 1, 17, 32, 49, 100, 122, 134, 155, 160, 2, 170};
namespace tt {
namespace tt_metal {

template <typename ADDRgen, typename ADDRgenInfo>
static void run_sharded_addrgen_test(
    ADDRgen addrgen, ADDRgenInfo constants, uint32_t bank_base_address, uint32_t page) {
    // Calculate the right address and number of contiguous pages manually
    uint64_t x_core;
    uint64_t y_core;
    uint32_t page_num;
    uint32_t number_of_cont_pages = 1;
    uint32_t l1_address;
    if (constants.shard_type == ttnn::ccl::common::shard_addr_gen_utils::ShardingLayout::WIDTH_SHARDED) {
        uint32_t page_row = page / constants.pages_per_tensor_row;
        uint32_t page_col = page % constants.pages_per_tensor_row;
        uint32_t w_core_id = page_col / constants.pages_per_shard_width;
        uint32_t w_offset = page_col % constants.pages_per_shard_width;
        x_core = real_core_x_vals[w_core_id];
        y_core = real_core_y_vals[w_core_id];
        page_num = page_row * constants.pages_per_shard_width + w_offset;
        if (constants.contiguity != ttnn::ccl::common::shard_addr_gen_utils::Contiguity_types::PADDING_BETWEEN_PAGES) {
            uint32_t space_left_in_shard = constants.pages_per_shard_width - w_offset;
            uint32_t space_left_in_tensor = constants.pages_per_tensor_row - page_col;
            number_of_cont_pages =
                space_left_in_shard < space_left_in_tensor ? space_left_in_shard : space_left_in_tensor;
        }
    } else if (constants.shard_type == ttnn::ccl::common::shard_addr_gen_utils::ShardingLayout::HEIGHT_SHARDED) {
        uint32_t num_pages_per_core = constants.pages_per_tensor_row * constants.rows_per_shard_height;
        uint32_t core_num = page / num_pages_per_core;
        page_num = page % num_pages_per_core;
        x_core = real_core_x_vals[core_num];
        y_core = real_core_y_vals[core_num];
        if (constants.contiguity == ttnn::ccl::common::shard_addr_gen_utils::Contiguity_types::PADDING_BETWEEN_PAGES) {
            number_of_cont_pages = 1;
        } else if (
            constants.contiguity ==
            ttnn::ccl::common::shard_addr_gen_utils::Contiguity_types::PADDING_IN_RIGHTMOST_SHARD) {
            number_of_cont_pages = constants.pages_per_tensor_row - page_num % constants.pages_per_tensor_row;
        } else {
            number_of_cont_pages = num_pages_per_core - page_num;
        }
    } else {
        constexpr uint32_t cores_per_block_row =
            (constants.pages_per_tensor_row - 1) / constants.pages_per_shard_width + 1;
        experimental::shard_addr_gen_utils::shard_coord_info coord_info;
        // Get row and column ID of this page
        uint32_t page_row = page / constants.pages_per_tensor_row;
        uint32_t page_col = page % constants.pages_per_tensor_row;
        // Find the w direction core and the offset within it
        uint32_t w_core_id = page_col / constants.pages_per_shard_width;
        uint32_t w_offset = page_col % constants.pages_per_shard_width;
        // Find the h direction core and the offset within it
        uint32_t h_core_id = page_row / constants.pages_per_tensor_row;
        uint32_t h_offset = page_row % constants.pages_per_tensor_row;
        // Find the coord_info
        uint32_t full_core_id = w_core_id + h_core_id * cores_per_block_row;
        uint32_t full_offset = w_offset + h_offset * constants.pages_per_shard_width;
        if (constants.contiguity != ttnn::ccl::common::shard_addr_gen_utils::Contiguity_types::PADDING_BETWEEN_PAGES) {
            uint32_t space_left_in_shard = constants.pages_per_shard_width - w_offset;
            uint32_t space_left_in_tensor = constants.pages_per_tensor_row - page_col;
            number_of_cont_pages =
                space_left_in_shard < space_left_in_tensor ? space_left_in_shard : space_left_in_tensor;
        }
    }
    l1_address = bank_base_address + (page_num * constants.page_size_jump);
    auto calculated_address = NOC_XY_ADDR(DYNAMIC_NOC_X(noc, x_core), DYNAMIC_NOC_Y(noc, y_core), l1_address);
    // Get the locations using the addrgen
    auto address = addrgen.get_noc_addr(page);
    auto cont_address = addrgen.get_contiguous_noc_addr(page);
    ASSERT_EQ(cont_address.first, address);
    ASSERT_EQ(calculated_address, address);
    ASSERT_EQ(number_of_cont_pages, cont_address.second);
}

TEST(CclBlockShardedTensorSliceIndexer_Wormhole, width_sharded_test) {
    static constexpr std::size_t shard_type = static_cast<uint32_t>(ttnn::ccl::common::shard_addr_gen_utils::ShardingLayout::WIDTH_SHARDED);
    static constexpr std::size_t number_of_cores = 4;
    static constexpr std::size_t page_size_jump = 1024;
    static constexpr std::size_t pages_per_tensor_row = 32;
    static constexpr std::size_t contiguity = static_cast<uint32_t>(ttnn::ccl::common::shard_addr_gen_utils::Contiguity_types::NO_SHARD_PADDING);
    static constexpr std::size_t pages_per_shard_width = 8;
    static constexpr std::size_t rows_per_shard_height = 1;
    static constexpr std::size_t tensor_address = 0x100000;
    typedef Sharded_Info<
        shard_type,
        number_of_cores,
        page_size_jump,
        pages_per_tensor_row,
        contiguity,
        pages_per_shard_width,
        rows_per_shard_height> ct_shard_info;
    auto info_var = ct_shard_info{};
    experimental::ShardedAddrGen<ct_shard_info> addrgen = {
            .bank_base_address = tensor_address, .shard_array=map};
    for(int i = 0; i < 12; i++)
    {
        run_sharded_addrgen_test(addrgen, info_var, tensor_address, test_pages[i]);
    }
}

TEST(CclBlockShardedTensorSliceIndexer_Wormhole, height_sharded_test) {
    static constexpr std::size_t shard_type = static_cast<uint32_t>(ttnn::ccl::common::shard_addr_gen_utils::ShardingLayout::HEIGHT_SHARDED);
    static constexpr std::size_t number_of_cores = 4;
    static constexpr std::size_t page_size_jump = 1024;
    static constexpr std::size_t pages_per_tensor_row = 32;
    static constexpr std::size_t contiguity = static_cast<uint32_t>(ttnn::ccl::common::shard_addr_gen_utils::Contiguity_types::NO_SHARD_PADDING);
    static constexpr std::size_t pages_per_shard_width = 1;
    static constexpr std::size_t rows_per_shard_height = 8;
    static constexpr std::size_t tensor_address = 0x100000;
    typedef Sharded_Info<
        shard_type,
        number_of_cores,
        page_size_jump,
        pages_per_tensor_row,
        contiguity,
        pages_per_shard_width,
        rows_per_shard_height> ct_shard_info;
    auto info_var = ct_shard_info{};
    experimental::ShardedAddrGen<ct_shard_info> addrgen = {
            .bank_base_address = tensor_address, .shard_array=map};
    for(int i = 0; i < 12; i++)
    {
        run_sharded_addrgen_test(addrgen, info_var, tensor_address, test_pages[i]);
    }
}


TEST(CclBlockShardedTensorSliceIndexer_Wormhole, block_sharded_test) {
    static constexpr std::size_t shard_type = static_cast<uint32_t>(ttnn::ccl::common::shard_addr_gen_utils::ShardingLayout::HEIGHT_SHARDED);
    static constexpr std::size_t number_of_cores = 4;
    static constexpr std::size_t page_size_jump = 1024;
    static constexpr std::size_t pages_per_tensor_row = 32;
    static constexpr std::size_t contiguity = static_cast<uint32_t>(ttnn::ccl::common::shard_addr_gen_utils::Contiguity_types::NO_SHARD_PADDING);
    static constexpr std::size_t pages_per_shard_width = 8;
    static constexpr std::size_t rows_per_shard_height = 8;
    static constexpr std::size_t tensor_address = 0x100000;
    typedef Sharded_Info<
        shard_type,
        number_of_cores,
        page_size_jump,
        pages_per_tensor_row,
        contiguity,
        pages_per_shard_width,
        rows_per_shard_height> ct_shard_info;
    auto info_var = ct_shard_info{};
    experimental::ShardedAddrGen<ct_shard_info> addrgen = {
            .bank_base_address = tensor_address, .shard_array=map};
    for(int i = 0; i < 12; i++)
    {
        run_sharded_addrgen_test(addrgen, info_var, tensor_address, test_pages[i]);
    }
}


}  // namespace tt_metal
}  // namespace tt
