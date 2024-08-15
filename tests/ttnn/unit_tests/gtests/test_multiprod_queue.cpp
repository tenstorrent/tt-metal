// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor.hpp"
#include "ttnn_multi_command_queue_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_sum/moreh_sum_op.hpp"
#include "common/bfloat16.hpp"
#include "tt_numpy/functions.hpp"
#include <cmath>
#include <thread>

using namespace tt;
using namespace tt_metal;
using MultiCommandQueueSingleDeviceFixture = ttnn::MultiCommandQueueSingleDeviceFixture;
using namespace constants;

TEST_F(MultiCommandQueueSingleDeviceFixture, TestMultiProducerLockBasedQueue) {
}
