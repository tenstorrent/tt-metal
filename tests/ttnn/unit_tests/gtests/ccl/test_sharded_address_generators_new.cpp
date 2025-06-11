// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstddef>
#include <iterator>
#include <memory>

#include <tt-metalium/buffer_types.hpp>
#include "gtest/gtest.h"
#include "ttnn/operations/ccl/common/types/sharding_common.hpp"
#if !(defined(KERNEL_BUILD) || defined(FW_BUILD))

#define NOC_ADDR_LOCAL_BITS 36
#define NOC_ADDR_NODE_ID_BITS 6

#define FORCE_INLINE inline __attribute__((always_inline))
#define noc_index 0
#define NOC_XY_ADDR(x, y, addr)                                                                                      \
    ((((uint64_t)(y)) << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) | (((uint64_t)(x)) << NOC_ADDR_LOCAL_BITS) | \
     ((uint64_t)(addr)))
#define NOC_0_X(noc_index, noc_size_x, x) x
#define NOC_0_Y(noc_index, noc_size_y, y) y
#define DYNAMIC_NOC_X(noc, x) NOC_0_X(noc, noc_size_x, (x))
#define DYNAMIC_NOC_Y(noc, y) NOC_0_Y(noc, noc_size_y, (y))

#define NOC_ADDR_COORD_SHIFT 36
#define NUM_DRAM_BANKS 6
#define NUM_NOCS 2
int32_t bank_to_dram_offset[NUM_DRAM_BANKS];
uint16_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS];

#endif
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

namespace sharding_testing_parameters {
mapping_table_t map[9] = {0x00000001, 0x00020003, 0x00040200, 0x02010202, 0x02030204, 0x03000301, 0x03020303, 0x04000401, 0x04020403};
uint64_t real_core_x_vals [18] = {0x0, 0x0, 0x0, 0x0, 0x0, 0x2, 0x2, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3, 0x4, 0x4, 0x4, 0x4};
uint64_t real_core_y_vals [18] = {0x0, 0x1, 0x2, 0x3, 0x4, 0x0, 0x1, 0x2, 0x3, 0x4, 0x0, 0x1, 0x2, 0x3, 0x0, 0x1, 0x2, 0x3};
}  // namespace sharding_testing_parameters
namespace tt {
namespace tt_metal {

template <typename ADDRgen, typename ADDRgenInfo>
void run_full_width_test(ADDRgen addrgen, ADDRgenInfo constants, uint32_t bank_base_address) {
    uint32_t rows[7] = {0, 1, 31, 32, 33, 66, 10000};
    for (int i = 0; i < std::size(rows); i++) {
        uint32_t page = constants.pages_per_tensor_row * rows[i];
        uint32_t base_address = constants.pages_per_shard_width * rows[i] * constants.page_size_jump;
        for (int j = 0; j < constants.number_of_cores; j++) {
            uint64_t l1_address = base_address;
            for (int k = 0; k < constants.pages_per_shard_width; k++) {
                if (j * constants.pages_per_shard_width + k < constants.pages_per_tensor_row) {
                    uint64_t calculated_address =
                        bank_base_address + NOC_XY_ADDR(
                                                DYNAMIC_NOC_X(noc, sharding_testing_parameters::real_core_x_vals[j]),
                                                DYNAMIC_NOC_Y(noc, sharding_testing_parameters::real_core_y_vals[j]),
                                                l1_address);
                    uint64_t retrieved_address = addrgen.get_noc_addr(page);
                    ASSERT_EQ(calculated_address, retrieved_address);
                    l1_address += constants.page_size_jump;
                }
                page++;
            }
        }
    }
}

template <typename ADDRgen, typename ADDRgenInfo>
void run_full_height_test(ADDRgen addrgen, ADDRgenInfo constants, uint32_t bank_base_address) {
    uint32_t width_pages[5] = {0, 1, 31, 30, 14};
    for (int i = 0; i < std::size(width_pages); i++) {
        uint32_t page = width_pages[i];
        uint32_t base_address = page * constants.page_size_jump;
        for (int j = 0; j < constants.number_of_cores; j++) {
            uint32_t l1_address = base_address;
            for (int k = 0; k < constants.rows_per_shard_height; k++) {
                uint64_t calculated_address =
                    bank_base_address + NOC_XY_ADDR(
                                            DYNAMIC_NOC_X(noc, sharding_testing_parameters::real_core_x_vals[j]),
                                            DYNAMIC_NOC_Y(noc, sharding_testing_parameters::real_core_y_vals[j]),
                                            l1_address);
                uint64_t retrieved_address = addrgen.get_noc_addr(page);
                ASSERT_EQ(calculated_address, retrieved_address);
                l1_address += constants.page_size_jump * constants.pages_per_tensor_row;
                page = page + constants.pages_per_tensor_row;
            }
        }
    }
}

template <typename ADDRgen, typename ADDRgenInfo>
void run_full_block_test(ADDRgen addrgen, ADDRgenInfo constants, uint32_t bank_base_address) {
    uint32_t random_width_offsets[4] = {0, 1, 5, 7};
    uint32_t random_height_offsets[4] = {0, 1, 5, 7};
    uint32_t cores_per_block_row = (constants.pages_per_tensor_row - 1) / constants.pages_per_shard_width + 1;
    uint32_t cores_height = constants.number_of_cores / cores_per_block_row;
    for (int i = 0; i < std::size(random_width_offsets); i++) {
        for (int j = 0; j < std::size(random_height_offsets); j++) {
            uint64_t outer_page = random_width_offsets[i] + random_height_offsets[j] * constants.pages_per_tensor_row;
            uint64_t l1_address =
                (random_width_offsets[i] + random_height_offsets[j] * constants.pages_per_shard_width) *
                constants.page_size_jump;
            for (int h = 0; h < cores_height; h++) {
                uint64_t page = outer_page;
                for (int w = 0; w < cores_per_block_row; w++) {
                    uint32_t core_number = w + h * cores_per_block_row;
                    uint64_t calculated_address =
                        bank_base_address +
                        NOC_XY_ADDR(
                            DYNAMIC_NOC_X(noc, sharding_testing_parameters::real_core_x_vals[core_number]),
                            DYNAMIC_NOC_Y(noc, sharding_testing_parameters::real_core_y_vals[core_number]),
                            l1_address);
                    uint64_t retrieved_address = addrgen.get_noc_addr(page);
                    ASSERT_EQ(calculated_address, retrieved_address);
                    page += constants.pages_per_shard_width;
                }
                outer_page += constants.pages_per_tensor_row * constants.rows_per_shard_height;
            }
        }
    }
}

TEST(CclnewWidthShardedTensorSliceIndexer_Wormhole, width_sharded_test) {
    constexpr std::size_t shard_type = static_cast<std::size_t>(tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED);
    constexpr std::size_t number_of_cores = 8;
    constexpr std::size_t page_size_jump = 1024;
    constexpr std::size_t pages_per_tensor_row = 32;
    constexpr std::size_t contiguity =
        static_cast<std::size_t>(shard_addr_gen_consts::ContiguityType::L1_NO_SHARD_PADDING);
    constexpr std::size_t pages_per_shard_width = 6;
    constexpr std::size_t rows_per_shard_height = 1;
    constexpr std::size_t tensor_address = 0x100000;
    using ct_shard_info = ShardedInfo<
        shard_type,
        number_of_cores,
        page_size_jump,
        pages_per_tensor_row,
        contiguity,
        pages_per_shard_width,
        rows_per_shard_height>;
    auto info_var = ct_shard_info{};
    ::experimental::ShardedAddrGen<ct_shard_info> addrgen = {
        .bank_base_address = tensor_address, .shard_array = sharding_testing_parameters::map};
    run_full_width_test(addrgen, info_var, tensor_address);
}

TEST(CclnewHeightShardedTensorSliceIndexer_Wormhole, height_sharded_test) {
    static constexpr std::size_t shard_type =
        static_cast<std::size_t>(tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED);
    static constexpr std::size_t number_of_cores = 4;
    static constexpr std::size_t page_size_jump = 1024;
    static constexpr std::size_t pages_per_tensor_row = 32;
    static constexpr std::size_t contiguity =
        static_cast<std::size_t>(shard_addr_gen_consts::ContiguityType::L1_NO_SHARD_PADDING);
    static constexpr std::size_t pages_per_shard_width = 1;
    static constexpr std::size_t rows_per_shard_height = 8;
    static constexpr std::size_t tensor_address = 0x100000;
    using ct_shard_info = ShardedInfo<
        shard_type,
        number_of_cores,
        page_size_jump,
        pages_per_tensor_row,
        contiguity,
        pages_per_shard_width,
        rows_per_shard_height>;
    auto info_var = ct_shard_info{};
    ::experimental::ShardedAddrGen<ct_shard_info> addrgen = {
        .bank_base_address = tensor_address, .shard_array = sharding_testing_parameters::map};
    run_full_height_test(addrgen, info_var, tensor_address);
}

TEST(CclnewBlockShardedTensorSliceIndexer_Wormhole, block_sharded_test) {
    static constexpr std::size_t shard_type = static_cast<std::size_t>(tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED);
    static constexpr std::size_t number_of_cores = 16;
    static constexpr std::size_t page_size_jump = 1024;
    static constexpr std::size_t pages_per_tensor_row = 32;
    static constexpr std::size_t contiguity =
        static_cast<std::size_t>(shard_addr_gen_consts::ContiguityType::L1_NO_SHARD_PADDING);
    static constexpr std::size_t pages_per_shard_width = 8;
    static constexpr std::size_t rows_per_shard_height = 8;
    static constexpr std::size_t tensor_address = 0x1000000;
    using ct_shard_info = ShardedInfo<
        shard_type,
        number_of_cores,
        page_size_jump,
        pages_per_tensor_row,
        contiguity,
        pages_per_shard_width,
        rows_per_shard_height>;
    auto info_var = ct_shard_info{};
    ::experimental::ShardedAddrGen<ct_shard_info> addrgen = {
        .bank_base_address = tensor_address, .shard_array = sharding_testing_parameters::map};
    run_full_block_test(addrgen, info_var, tensor_address);
}

}  // namespace tt_metal
}  // namespace tt
