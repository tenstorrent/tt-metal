// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "tt_dnn/op_library/move/move_op.hpp"
#include "tt_dnn/op_library/clone/clone_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks move_multi_core(const Tensor &input, Tensor &output) {
    return clone_multi_core(input, output);
}

}  // namespace tt_metal

}  // namespace tt
