/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <optional>
#include <array>
//#include "tt_dnn/op_library/run_operation.hpp"
using namespace std;

#include "third_party/umd/device/tt_xy_pair.h"
using std::pair;
using CoreCoord = tt_xy_pair;

namespace bmm_op_utils {
//using namespace tt::tt_metal;

constexpr std::array<tuple<uint32_t, uint32_t>, 20> SUBBLOCK_HW_CHOICES = {{
    {4, 2}, {2, 4}, {8, 1}, {1, 8},
    {7, 1}, {1, 7},
    {3, 2}, {2, 3}, {6, 1}, {1, 6},
    {5, 1}, {1, 5},
    {2, 2}, {4, 1}, {1, 4},
    {3, 1}, {1, 3},
    {2, 1}, {1, 2},
    {1, 1},
}};

tuple<uint32_t, uint32_t, uint32_t, uint32_t> get_large_matmul_params(uint32_t Mt, uint32_t Nt, uint32_t num_cores_y, uint32_t num_cores_x, uint32_t in0_block_w);

CoreCoord get_core_range(uint32_t num_blocks_rows, uint32_t num_blocks_cols, uint32_t max_num_rows, uint32_t max_num_cols);

}  // namespace bmm_op_utils
