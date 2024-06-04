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
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "Operands to {} need to be on device!", op_name);
    TT_FATAL(tensor.buffer() != nullptr, "Operands to {} need to be allocated in buffers on device!", op_name);
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

inline void validate_input_tensor_with_dim(const Tensor& input, const int64_t &dim) {
    auto input_shape = input.get_legacy_shape();
    auto input_shape_wo_padding = input.get_legacy_shape().without_padding();
    const auto input_rank = input_shape.rank();
    log_debug(LogOp, "{}:{} input_rank {}", __func__, __LINE__, input_rank);
    TT_FATAL(
        (dim >= 0 && dim <= tt::tt_metal::MAX_NUM_DIMENSIONS),
        "dim must be between 0 and {}.",
        tt::tt_metal::MAX_NUM_DIMENSIONS);
    TT_FATAL((dim < input_rank), "dim must be smaller than input tensor rank {}.", input_rank);
}

inline void validate_output_tensor_with_keepdim(const Tensor& input, const Tensor& output, const int64_t &dim, const bool &keepdim) {
    auto input_shape = input.get_legacy_shape();
    auto input_shape_wo_padding = input_shape.without_padding();
    const auto input_rank = input_shape.rank();

    const auto& output_shape = output.get_legacy_shape();
    const auto& output_shape_wo_padding = output_shape.without_padding();

    const bool is_tile_dim = (dim == input_rank - 1 || dim == input_rank - 2);

    log_debug(LogOp, "{}:{} input_shape {}", __func__, __LINE__, input_shape);
    log_debug(LogOp, "{}:{} output_shape {}", __func__, __LINE__, output_shape);
    log_debug(LogOp, "{}:{} input_shape_wo_padding {}", __func__, __LINE__, input_shape_wo_padding);
    log_debug(LogOp, "{}:{} output_shape_wo_padding {}", __func__, __LINE__, output_shape_wo_padding);

    if (keepdim) {
        input_shape[dim] = (is_tile_dim) ? (TILE_HEIGHT) : (1);
        input_shape_wo_padding[dim] = 1;

        for (int i = 0; i < input_rank; ++i) {
            TT_FATAL(input_shape[i] == output_shape[i]);
            TT_FATAL(input_shape_wo_padding[i] == output_shape_wo_padding[i]);
        }
        } else {
            std::vector<uint32_t> expected_output_shape;
            std::vector<uint32_t> expected_output_shape_wo_padding;
            for (int i = 0; i < output_shape.rank(); ++i) {
                if (i == dim && !is_tile_dim) {
                    expected_output_shape.push_back(1);
                    expected_output_shape_wo_padding.push_back(1);
                }
                expected_output_shape.push_back(output_shape[i]);
                expected_output_shape_wo_padding.push_back(output_shape_wo_padding[i]);
            }

            log_debug(LogOp, "{}:{} expected_output_shape {}", __func__, __LINE__, expected_output_shape);
            log_debug(
                LogOp,
                "{}:{} expected_output_shape_wo_padding {}",
                __func__,
                __LINE__,
                expected_output_shape_wo_padding);
            for (int i = 0; i < input_rank; ++i) {
                if (i == dim)
                    continue;
                TT_FATAL(input_shape[i] == expected_output_shape[i]);
                TT_FATAL(input_shape_wo_padding[i] == expected_output_shape_wo_padding[i]);
            }
        }
}

Tensor _moreh_sum(
    const Tensor& input,
    const int64_t& dim,
    const bool& keepdim,
    const std::optional<const Tensor>& output,
    const MemoryConfig& output_mem_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input}))};

    TT_FATAL(input.storage_type() == StorageType::DEVICE || input.storage_type() == StorageType::MULTI_DEVICE);
    auto kernel_config_val = init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4);

    operation::launch_op(
        [dim, keepdim, output_mem_config, kernel_config_val](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehSum{
                    .dim = dim,
                    .keepdim = keepdim,
                    .output_mem_config = output_mem_config,
                    .compute_kernel_config = kernel_config_val},
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

    check_tensor(input, "input");
    check_tensor(output, "output");

    validate_input_tensor_with_dim(input, this->dim);

    if (output.has_value()) {
        validate_output_tensor_with_keepdim(input, output.value(), this->dim, this->keepdim);
    }
}

std::vector<Shape> MorehSum::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input = input_tensors.at(0);
    const auto& input_shape = input.get_legacy_shape();
    const auto input_rank = input_shape.rank();
    const bool is_tile_dim = (this->dim == input_rank - 1 || this->dim == input_rank - 2);
    log_debug(LogOp, "{}:{} dim {}, keepdim {}", __func__, __LINE__, this->dim, this->keepdim);

    Shape output_shape = input_shape;
    if (this->keepdim) {
        auto shape = input_shape;
        auto padding = shape.padding();

        if (is_tile_dim) {
            // e.g. (2, 64, 64) with dim 1 to be (2, 1[32], 64)
            shape[this->dim] = TILE_HEIGHT;
            padding[this->dim] = Padding::PadDimension{0, 31};
        } else {
            // e.g. (2, 64, 64) with dim 0 to be (1, 64, 64)
            shape[this->dim] = 1;
        }

        output_shape = Shape(shape, padding);
    } else {
        std::vector<uint32_t> shape;
        std::vector<Padding::PadDimension> pad_dimensions;
        const std::size_t output_rank = (is_tile_dim) ? (input_rank) : (input_rank - 1);
        auto input_padding = input_shape.padding();

        // e.g. (2, 64, 64) with dim 1 to be (2, 1[32], 64)
        // e.g. (2, 64, 64) with dim 0 to be (64, 64)
        for (int i = 0; i < input_rank; ++i) {
            bool is_reduced_dim = (i == this->dim);
            if (is_reduced_dim && !is_tile_dim)
                continue;

            shape.push_back((is_reduced_dim && is_tile_dim) ? (TILE_HEIGHT) : (input_shape[i]));
            pad_dimensions.push_back(
                (is_reduced_dim && is_tile_dim) ? (Padding::PadDimension{0, 31}) : (input_padding[i]));
        }

        auto padding = Padding(pad_dimensions, input_padding.pad_value());
        output_shape = Shape(shape, padding);
    }

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
    } else if (this->dim == input_rank - 2) {
        return moreh_sum_h_impl(input, output, this->compute_kernel_config);
    } else {
        return moreh_sum_nc_impl(input, output, dim, this->compute_kernel_config);
    }
}

Tensor moreh_sum(
    const Tensor& input,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
    const bool keepdim,
    const std::optional<const Tensor> output,
    const MemoryConfig& output_mem_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    std::vector<int64_t> dims = get_dim(dim, input.get_legacy_shape().rank());
    std::sort(dims.begin(), dims.end());

    auto temp_input = input;
    for (uint32_t i = dims.size() - 1; i > 0; i--) {
        log_debug(LogOp, "{}:{} dim {} keepdim {}", __func__, __LINE__, dims[i], keepdim);
        auto temp_output =
            _moreh_sum(temp_input, dims[i], keepdim, std::nullopt, output_mem_config, compute_kernel_config);
        temp_input = temp_output;
    }
    log_debug(LogOp, "{}:{} dim {} keepdim {}", __func__, __LINE__, dims.front(), keepdim);
    return _moreh_sum(temp_input, dims.front(), keepdim, output, output_mem_config, compute_kernel_config);
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
