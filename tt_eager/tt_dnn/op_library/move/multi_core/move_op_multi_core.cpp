// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "tt_dnn/op_library/move/move_op.hpp"
#include "tt_dnn/op_library/copy/copy_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks move_multi_core(const Tensor &input, Tensor &output) {
    bool src_and_dst_in_l1 = input.memory_config().is_l1() && output.memory_config().is_l1();
    return copy_multi_core(input, output, src_and_dst_in_l1);
}

}  // namespace tt_metal

}  // namespace tt
