// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "multi_device_fixture.hpp"
#include <tt-metalium/mesh_device.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace tt::tt_metal::unit_tests::dm::matmul {

constexpr uint32_t L1_DEBUG_PADDING_BYTES = 0x10;

struct MatmulTestConfig {
    uint32_t test_id = 0;
    CoreCoord start_logical_core;
    CoreCoord end_logical_core;
    uint32_t num_subblocks_r_dim = 2;
    uint32_t num_subblocks_c_dim = 2;
    uint32_t num_subblocks_k_dim = 1;
    uint32_t subblock_r_dim = 1;
    uint32_t subblock_c_dim = 1;
    uint32_t subblock_k_dim = 1;
    uint32_t page_size_bytes = 1;
    DataFormat l1_data_format = DataFormat::Float16_b;
    uint32_t dram_bank_id = 0;

    std::vector<uint32_t> num_subblocks_r_dim_sweep;
    std::vector<uint32_t> num_subblocks_c_dim_sweep;
    std::vector<uint32_t> num_subblocks_k_dim_sweep;
    std::vector<uint32_t> subblock_r_dim_sweep;
    std::vector<uint32_t> subblock_c_dim_sweep;
    std::vector<uint32_t> subblock_k_dim_sweep;
};

// Hardcoded test configurations shared by all matmul test variants.
// Each config is an explicit, representative test point.
inline std::vector<MatmulTestConfig> get_matmul_test_configs() {
    std::vector<MatmulTestConfig> configs = {
        // ---- Grid shape tests ----
        // ID 1000: 2x2 default grid
        {.test_id = 1000},
        // ID 1001: 1x1 single core — sender multicasts to itself
        {.test_id = 1001, .num_subblocks_r_dim = 1, .num_subblocks_c_dim = 1},
        // ID 1002: 1x3 single row — one sender, 3 receivers
        {.test_id = 1002, .num_subblocks_r_dim = 1, .num_subblocks_c_dim = 3},
        // ID 1003: 3x1 single column — 3 independent senders
        {.test_id = 1003, .num_subblocks_r_dim = 3, .num_subblocks_c_dim = 1},
        // ID 1004: 2x3 non-square grid (R=3, C=2)
        {.test_id = 1004, .num_subblocks_r_dim = 3, .num_subblocks_c_dim = 2},
        // ID 1005: 3x2 non-square grid (R=2, C=3)
        {.test_id = 1005, .num_subblocks_r_dim = 2, .num_subblocks_c_dim = 3},
        // ID 1006: 4x4 large square grid
        {.test_id = 1006, .num_subblocks_r_dim = 4, .num_subblocks_c_dim = 4},
        // ID 1007: 6x6 large grid with K=2
        {.test_id = 1007, .num_subblocks_r_dim = 6, .num_subblocks_c_dim = 6, .num_subblocks_k_dim = 2},
        // ID 1008: 1x8 wide row — max multicast fan-out
        {.test_id = 1008, .num_subblocks_r_dim = 1, .num_subblocks_c_dim = 8},
        // ID 1009: 8x1 tall column with K=4
        {.test_id = 1009, .num_subblocks_r_dim = 8, .num_subblocks_c_dim = 1, .num_subblocks_k_dim = 4},

        // ---- Non-origin start tests ----
        // ID 1010: 2x2 grid starting at logical core (2,2)
        {.test_id = 1010, .start_logical_core = CoreCoord(2, 2)},
        // ID 1011: 5x3 grid starting at (2,3) with K=2
        {.test_id = 1011,
         .start_logical_core = CoreCoord(2, 3),
         .num_subblocks_r_dim = 3,
         .num_subblocks_c_dim = 5,
         .num_subblocks_k_dim = 2},

        // ---- K dimension tests ----
        // ID 1012: K=2 on 2x2 grid, sweep subblock_k_dim
        {.test_id = 1012, .num_subblocks_k_dim = 2, .subblock_k_dim_sweep = {1u, 2u}},
        // ID 1013: K=3 on 3x2 grid, sweep subblock_k_dim
        {.test_id = 1013,
         .num_subblocks_r_dim = 3,
         .num_subblocks_c_dim = 2,
         .num_subblocks_k_dim = 3,
         .subblock_k_dim_sweep = {1u, 2u}},

        // ---- Subblock dimension tests ----
        // ID 1014: subblock_r=2
        {.test_id = 1014, .subblock_r_dim = 2},
        // ID 1015: subblock_c=2
        {.test_id = 1015, .subblock_c_dim = 2},
        // ID 1016: subblock_k=2
        {.test_id = 1016, .subblock_k_dim = 2},
        // ID 1017: all subblocks=2, K=2
        {.test_id = 1017, .num_subblocks_k_dim = 2, .subblock_r_dim = 2, .subblock_c_dim = 2, .subblock_k_dim = 2},
        // ID 1018: all subblocks=4, K=2
        {.test_id = 1018, .num_subblocks_k_dim = 2, .subblock_r_dim = 4, .subblock_c_dim = 4, .subblock_k_dim = 4},
        // ID 1019: asymmetric subblocks (R=2, C=6, K=3, sub_r=3, sub_c=2, sub_k=2)
        {.test_id = 1019,
         .num_subblocks_r_dim = 2,
         .num_subblocks_c_dim = 6,
         .num_subblocks_k_dim = 3,
         .subblock_r_dim = 3,
         .subblock_c_dim = 2,
         .subblock_k_dim = 2},

        // ---- Stress tests ----
        // ID 1020: 4x4 grid, K=4, all subblocks=2
        {.test_id = 1020,
         .num_subblocks_r_dim = 4,
         .num_subblocks_c_dim = 4,
         .num_subblocks_k_dim = 4,
         .subblock_r_dim = 2,
         .subblock_c_dim = 2,
         .subblock_k_dim = 2},
        // ID 1021: 1x6 wide row, K=3, sub=(3,2,2)
        {.test_id = 1021,
         .num_subblocks_r_dim = 1,
         .num_subblocks_c_dim = 6,
         .num_subblocks_k_dim = 3,
         .subblock_r_dim = 3,
         .subblock_c_dim = 2,
         .subblock_k_dim = 2},

        // ---- DRAM bank tests ----
        // ID 1022: 2x2 grid reading in1 from DRAM bank 1
        {.test_id = 1022, .dram_bank_id = 1},

        // ---- K-vs-C edge case tests ----
        // ID 1023: K < C — only some columns ever send (R=2, C=4, K=2), sweep subblock_c_dim
        {.test_id = 1023,
         .num_subblocks_r_dim = 2,
         .num_subblocks_c_dim = 4,
         .num_subblocks_k_dim = 2,
         .subblock_c_dim_sweep = {1u, 2u}},
        // ID 1024: K == C — each column sends exactly once (R=2, C=3, K=3), sweep subblock_r_dim
        {.test_id = 1024,
         .num_subblocks_r_dim = 2,
         .num_subblocks_c_dim = 3,
         .num_subblocks_k_dim = 3,
         .subblock_r_dim_sweep = {1u, 2u}},
        // ID 1025: K not divisible by C — uneven round-robin (R=2, C=3, K=5), sweep subblock_k_dim
        {.test_id = 1025,
         .num_subblocks_r_dim = 2,
         .num_subblocks_c_dim = 3,
         .num_subblocks_k_dim = 5,
         .subblock_k_dim_sweep = {1u, 2u}},

        // ---- K subblock size sweep ----
        {.test_id = 1026,
         .num_subblocks_r_dim = 4,
         .num_subblocks_c_dim = 4,
         .num_subblocks_k_dim = 4,
         .subblock_r_dim = 4,
         .subblock_c_dim = 4,
         .subblock_k_dim_sweep = {1u, 2u, 4u, 8u}},

        // ---- R subblock size sweep ----
        // ID 1027: 4x4 grid, K=4, subblock_k=4, subblock_c=1, sweep subblock_r_dim.
        {.test_id = 1027,
         .num_subblocks_r_dim = 4,
         .num_subblocks_c_dim = 4,
         .num_subblocks_k_dim = 8,
         .subblock_c_dim = 1,
         .subblock_k_dim = 1,
         .subblock_r_dim_sweep = {1u, 2u, 4u, 8u, 16u, 32u}},

        // ---- C subblock size sweep ----
        // ID 1028: 4x4 grid, K=4, subblock_k=4, subblock_r=1, sweep subblock_c_dim.
        {.test_id = 1028,
         .num_subblocks_r_dim = 4,
         .num_subblocks_c_dim = 4,
         .num_subblocks_k_dim = 8,
         .subblock_r_dim = 1,
         .subblock_k_dim = 1,
         .subblock_c_dim_sweep = {1u, 2u, 4u, 8u, 16u, 32u}},
    };

    return configs;
}

struct MatmulTestNameGenerator {
    std::string operator()(const ::testing::TestParamInfo<MatmulTestConfig>& info) const {
        const auto& c = info.param;
        return "ID" + std::to_string(c.test_id) + "_R" + std::to_string(c.num_subblocks_r_dim) + "_C" +
               std::to_string(c.num_subblocks_c_dim) + "_K" + std::to_string(c.num_subblocks_k_dim) + "_sr" +
               std::to_string(c.subblock_r_dim) + "_sc" + std::to_string(c.subblock_c_dim) + "_sk" +
               std::to_string(c.subblock_k_dim) + "_X" + std::to_string(c.start_logical_core.x) + "Y" +
               std::to_string(c.start_logical_core.y) + "_bank" + std::to_string(c.dram_bank_id);
    }
};

}  // namespace tt::tt_metal::unit_tests::dm::matmul

namespace tt::tt_metal {

class Matmul1DParamFixture : public GenericMeshDeviceFixture,
                             public ::testing::WithParamInterface<unit_tests::dm::matmul::MatmulTestConfig> {};

class Matmul1DV2ParamFixture : public GenericMeshDeviceFixture,
                               public ::testing::WithParamInterface<unit_tests::dm::matmul::MatmulTestConfig> {};

class Matmul2DParamFixture : public GenericMeshDeviceFixture,
                             public ::testing::WithParamInterface<unit_tests::dm::matmul::MatmulTestConfig> {};

}  // namespace tt::tt_metal
