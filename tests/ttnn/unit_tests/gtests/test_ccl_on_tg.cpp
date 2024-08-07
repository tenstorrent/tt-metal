// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/bfloat16.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/line_all_gather/device/line_all_gather_op.hpp"
#include "ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_op.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_types.hpp"
#include "ttnn/cpp/ttnn/multi_device.hpp"
#include "ttnn_multi_command_queue_fixture.hpp"

using namespace tt;
using namespace tt_metal;

TEST(TGTests, TestAllGatherDeadlock) {
}

TEST(TGTests, TestReduceScatterDeadlock) {
}
