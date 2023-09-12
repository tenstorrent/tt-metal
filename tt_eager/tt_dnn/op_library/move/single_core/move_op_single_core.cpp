// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/move/move_op.hpp"
#include "tt_dnn/op_library/clone/clone_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks move_single_core(const Tensor &input, Tensor &output) {
    return clone_single_core(input, output);
}

}  // namespace tt_metal

}  // namespace tt
