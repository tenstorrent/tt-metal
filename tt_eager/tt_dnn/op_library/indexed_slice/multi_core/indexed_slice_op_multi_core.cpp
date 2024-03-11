// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "tt_dnn/op_library/indexed_slice/indexed_slice_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks indexed_slice_multi_core(const Tensor &batch_ids, const Tensor &input, const Tensor &output) {
    tt_metal::Program program{};



    return {std::move(program), };
}

}  // namespace tt_metal

}  // namespace tt
