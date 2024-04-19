// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/layout_conversion/layout_conversion_op.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

#include <fmt/ranges.h>

using namespace tt::constants;

namespace tt {

namespace tt_metal {

void LayoutConversionOnHost::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (this->target_layout == Layout::TILE) {
        TT_FATAL(input_tensor.get_legacy_shape()[2] % TILE_HEIGHT == 0 && input_tensor.get_legacy_shape()[3] % TILE_WIDTH == 0);
    }
}
std::vector<Shape> LayoutConversionOnHost::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_legacy_shape()};
}
std::vector<Tensor> LayoutConversionOnHost::compute_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (input_tensor.get_layout() == this->target_layout){
        return {input_tensor};
    } else {
        return {input_tensor.to(target_layout)};
    }
}

tt::stl::reflection::Attributes LayoutConversionOnHost::attributes() const {
    return {
        {"target_layout", this->target_layout},
    };
}

Tensor layout_conversion_on_host(const Tensor &input_tensor, const Layout target_layout) {
    return operation::run(LayoutConversionOnHost{target_layout}, {input_tensor}).at(0);
}

}  // namespace tt_metal

}  // namespace tt
