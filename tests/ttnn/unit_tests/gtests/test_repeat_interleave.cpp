// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "tt_metal/common/bfloat16.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/repeat_interleave.hpp"
#include "tt_numpy/functions.hpp"
#include "tt_metal/common/logger.hpp"

#include "ttnn_test_fixtures.hpp"

#include <memory>

namespace ttnn {
namespace operations {
namespace data_movement {
namespace test {

struct RepeatInterleaveParams {
    int repeats = 0;
    int dim = 0;
};

class RepeatInterleaveTest : public ttnn::TTNNFixtureWithDevice, public ::testing::WithParamInterface<RepeatInterleaveParams> {};

TEST_P(RepeatInterleaveTest, RunsCorrectly) {
    RepeatInterleaveParams params = GetParam();
}

INSTANTIATE_TEST_SUITE_P(
    RepeatInterleaveWithDim0,
    RepeatInterleaveTest,
    ::testing::Values(
        RepeatInterleaveParams{1, 0},
        RepeatInterleaveParams{2, 0},
        RepeatInterleaveParams{3, 0}
    )
);

// tests/ttnn/unit_tests/operations/test_repeat_interleave.py proves that it should work over dim 1 too
// likely need to fix the comparison in the test
INSTANTIATE_TEST_SUITE_P(
    DISABLED_RepeatInterleaveWithDim1,
    RepeatInterleaveTest,
    ::testing::Values(
        RepeatInterleaveParams{1, 1},
        RepeatInterleaveParams{2, 1},
        RepeatInterleaveParams{3, 1}
    )
);


}  // namespace test
}  // namespace binary
}  // namespace operations
}  // namespace ttnn
