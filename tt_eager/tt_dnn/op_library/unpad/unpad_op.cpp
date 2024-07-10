// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/unpad/unpad_op.hpp"

#include "tensor/tensor_utils.hpp"
#include "tt_dnn/op_library/copy/copy_op.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

void UnpadOnHost::validate(const std::vector<Tensor> &input_tensors) const {
    const auto &input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() != StorageType::DEVICE);
    TT_FATAL(input_tensor.get_layout() == Layout::ROW_MAJOR);

    TT_FATAL(this->output_tensor_start[0] < input_tensor.get_legacy_shape()[0]);
    TT_FATAL(this->output_tensor_end[0] < input_tensor.get_legacy_shape()[0]);
    TT_FATAL(this->output_tensor_start[1] < input_tensor.get_legacy_shape()[1]);
    TT_FATAL(this->output_tensor_end[1] < input_tensor.get_legacy_shape()[1]);
    TT_FATAL(this->output_tensor_start[2] < input_tensor.get_legacy_shape()[2]);
    TT_FATAL(this->output_tensor_end[2] < input_tensor.get_legacy_shape()[2]);
    TT_FATAL(this->output_tensor_start[3] < input_tensor.get_legacy_shape()[3]);
    TT_FATAL(this->output_tensor_end[3] < input_tensor.get_legacy_shape()[3]);

    // Check if start shape is <= end shape
    TT_FATAL(this->output_tensor_start[0] <= this->output_tensor_end[0]);
    TT_FATAL(this->output_tensor_start[1] <= this->output_tensor_end[1]);
    TT_FATAL(this->output_tensor_start[2] <= this->output_tensor_end[2]);
    TT_FATAL(this->output_tensor_start[3] <= this->output_tensor_end[3]);
}
std::vector<Shape> UnpadOnHost::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    Shape output_tensor_shape = {
        this->output_tensor_end[0] - this->output_tensor_start[0] + 1,
        this->output_tensor_end[1] - this->output_tensor_start[1] + 1,
        this->output_tensor_end[2] - this->output_tensor_start[2] + 1,
        this->output_tensor_end[3] - this->output_tensor_start[3] + 1,
    };
    return {output_tensor_shape};
}
std::vector<Tensor> UnpadOnHost::compute_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto &input_tensor = input_tensors.at(0);
    if (input_tensor.get_legacy_shape() == UnpadOnHost::compute_output_shapes(input_tensors).at(0)) {
        return {input_tensor};
    } else {
        return {input_tensor.unpad(this->output_tensor_start, this->output_tensor_end)};
    }
}
Tensor unpad_on_host(
    const Tensor &input_tensor,
    const Shape &output_tensor_start,
    const Shape &output_tensor_end,
    const MemoryConfig &mem_config) {
    return operation::run(UnpadOnHost{output_tensor_start, output_tensor_end}, {input_tensor}).at(0);
}

}  // namespace tt_metal

}  // namespace tt
