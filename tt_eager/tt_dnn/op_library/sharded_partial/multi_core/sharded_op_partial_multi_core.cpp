// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "tt_metal/common/assert.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/sharded/sharded_op.hpp"
#include "tt_dnn/op_library/sharded_partial/sharded_op_partial.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks interleaved_to_sharded_partial_multi_core(const Tensor& input, const Tensor& output, int num_slices, int slice_index) {
    return interleaved_to_sharded_multi_core(input, output, num_slices, slice_index);
}

operation::ProgramWithCallbacks sharded_to_interleaved_partial_multi_core(const Tensor& input, const Tensor& output, int num_slices, int slice_index) {
    return sharded_to_interleaved_multi_core(input, output, num_slices, slice_index);
}

}  // namespace tt_metal

}  // namespace tt
