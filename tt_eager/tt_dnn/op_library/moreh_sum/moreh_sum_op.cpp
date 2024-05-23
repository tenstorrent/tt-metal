// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_sum/moreh_sum_op.hpp"

#include <numeric>

#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
using namespace constants;
namespace operations {
namespace primary {

////////////////////////////////////////////////////////////////////////////
//                         MorehSum
////////////////////////////////////////////////////////////////////////////
namespace {
    // TODO: move these check functions to a common header.
    inline void check_tensor(
        const Tensor& tensor,
        const std::string& op_name,
        DataType data_type = DataType::BFLOAT16,
        Layout layout = Layout::TILE) {
        TT_FATAL(tensor.get_layout() == layout, "{} only supports tiled layout.", op_name);
        TT_FATAL(tensor.get_dtype() == data_type, "{} only supports data type {}.", op_name, data_type);
        TT_FATAL(
            tensor.storage_type() == StorageType::DEVICE, "Operands to {} need to be on device!", op_name);
        TT_FATAL(
            tensor.buffer() != nullptr, "Operands to {} need to be allocated in buffers on device!", op_name);
    }

inline void check_tensor(
    std::optional<Tensor> tensor,
    const std::string& op_name,
    tt_metal::DataType data_type = DataType::BFLOAT16,
    Layout layout = Layout::TILE) {
    if (!tensor.has_value()) {
        return;
    }
    check_tensor(tensor.value(), op_name, data_type, layout);
}

Tensor _moreh_sum(
    const Tensor& input,
    const int64_t& dim,
    const std::optional<const Tensor>& output,
    const MemoryConfig& output_mem_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input}))};

    TT_FATAL(input.storage_type() == StorageType::DEVICE);
    auto kernel_config_val = init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4);

    operation::launch_op(
        [dim, output_mem_config, kernel_config_val](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehSum{.dim = dim, .output_mem_config = output_mem_config, .compute_kernel_config = kernel_config_val},
                input_tensors,
                optional_input_tensors,
                optional_output_tensors);
        },
        {input},
        output_tensors,
        {},
        {output});

    return output_tensors.at(0);
}
}  // namespace

void MorehSum::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input = input_tensors.at(0);
    auto& output = output_tensors.at(0);

    // validate tensor
    check_tensor(input, "input");
    check_tensor(output, "output");

    // validate input dim
    auto input_shape = input.get_legacy_shape();
    auto input_shape_wo_padding = input.get_legacy_shape().without_padding();
    const auto input_rank = input_shape.rank();
    log_debug(LogOp, "{}:{} input_rank {}", __func__, __LINE__, input_rank);
    TT_FATAL(
        (this->dim >= 0 && this->dim <= tt::tt_metal::MAX_NUM_DIMENSIONS),
        "dim must be between 0 and {}.",
        tt::tt_metal::MAX_NUM_DIMENSIONS);
    TT_FATAL((this->dim < input_rank), "dim must be smaller than input tensor rank {}.", input_rank);

// validate shape
// keepdim=true
// TODO: fix when input rank and output rank are different.
#if 0
    if (output.has_value()) {
        const auto& output_shape = output.value().get_legacy_shape();
        const auto& output_shape_wo_padding = output.value().get_legacy_shape().without_padding();

        // last 2-dim
        if (this->dim == input_rank - 1 || this->dim == input_rank - 2) {
            input_shape[this->dim] = TILE_HEIGHT;
            input_shape_wo_padding[this->dim] = 1;
        } else {
            input_shape[this->dim] = 1;
            input_shape_wo_padding[this->dim] = 1;
        }

        log_debug(LogOp, "{}:{} input_shape {}", __func__, __LINE__, input_shape);
        log_debug(LogOp, "{}:{} output_shape {}", __func__, __LINE__, output_shape);
        log_debug(LogOp, "{}:{} input_shape_wo_padding {}", __func__, __LINE__, input_shape_wo_padding);
        log_debug(LogOp, "{}:{} output_shape_wo_padding {}", __func__, __LINE__, output_shape_wo_padding);
        for (int i = 0; i < input_shape.rank(); ++i) {
            TT_FATAL(input_shape[i] == output_shape[i]);
            TT_FATAL(input_shape_wo_padding[i] == output_shape_wo_padding[i]);
        }
    }
#endif
}

std::vector<Shape> MorehSum::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input = input_tensors.at(0);
    const auto& input_shape = input.get_legacy_shape();
    const auto input_rank = input_shape.rank();

    // keepdim=true
    auto output_shape = input_shape;
    auto padding = output_shape.padding();

    // last 2-dim
    if (this->dim == input_rank - 1 || this->dim == input_rank - 2) {
        output_shape[this->dim] = TILE_HEIGHT;
        padding[this->dim] = Padding::PadDimension{0, 31};
    } else {
        output_shape[this->dim] = 1;
    }

    output_shape = Shape(output_shape, padding);
    log_debug(LogOp, "{}:{} dim {}", __func__, __LINE__, dim);
    log_debug(LogOp, "{}:{} output_shape {}", __func__, __LINE__, output_shape);
    return {output_shape};
}

std::vector<Tensor> MorehSum::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        log_debug(LogOp, "{}:{} use output tensor", __func__, __LINE__);
        return {output_tensors.at(0).value()};
    }

    log_debug(LogOp, "{}:{} create output tensor", __func__, __LINE__);
    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensors.at(0).get_dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks MorehSum::create_program(
    const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) const {
    auto& input = inputs.at(0);
    auto& output = outputs.at(0);

    const auto input_rank = input.get_legacy_shape().rank();
    if (this->dim == input_rank - 1) {
        return moreh_sum_w_impl(input, output, this->compute_kernel_config);
    } else if(this->dim == input_rank - 2) {
        return moreh_sum_h_impl(input, output, this->compute_kernel_config);
    } else {
        return moreh_sum_nc_impl(input, output, dim, this->compute_kernel_config);
    }
}

Tensor moreh_sum(
    const Tensor& input,
    std::vector<int64_t>& dims,
    const std::optional<const Tensor> output,
    const MemoryConfig& output_mem_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    // reduce for all dims
    if (dims.empty()) {
        const auto input_rank = input.get_legacy_shape().rank();
        dims.resize(input_rank);
        std::iota(dims.begin(), dims.end(), 0);
    }

    std::vector<int64_t> sorted_dims = dims;
    std::sort(sorted_dims.begin(), sorted_dims.end());

    auto temp_input = input;
    for (uint32_t i = dims.size() - 1; i > 0; i--) {
        log_debug(LogOp, "{}:{} dim {}", __func__, __LINE__, sorted_dims[i]);
        auto temp_output = _moreh_sum(temp_input, sorted_dims[i], std::nullopt, output_mem_config, compute_kernel_config);
        temp_input = temp_output;
    }
    log_debug(LogOp, "{}:{} dim {}", __func__, __LINE__, sorted_dims.front());
    return _moreh_sum(temp_input, sorted_dims.front(), output, output_mem_config, compute_kernel_config);
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
