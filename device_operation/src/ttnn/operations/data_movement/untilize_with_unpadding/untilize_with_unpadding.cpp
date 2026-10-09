#include "untilize_with_unpadding.hpp"
#include "device_operation/src/ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding_device_operation.hpp"
#include "device_operation/src/ttnn/operations/ccl/helper/flexible_queue_untilize_with_unpadding.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement {
    using namespace ttnn::shape_utils;

    void untilize_with_unpadding_op::validate(
        const std::vector<tensor::Tensor>& input_tensors,
        const std::vector<std::vector<std::optional<int64_t>>>& optional_input_shapes,
        const std::optional<ttnn::operations::unpadding::UnpaddingSpec>& output_tensor_end_opt,
        const std::optional<MemoryConfig>& memory_config_opt,
        const std::optional<const std::vector<tensor::Tensor>>& input_tensors_transformed,
        const std::optional<QueueId>& queue_id) const {
        auto input_shape = input_tensors[0].get_padded_shape();
        TT_FATAL(
            input_shape.rank() >= 4,
            "Input tensor must have rank >= 4, but got rank: {}",
            input_shape.rank());

        auto output_tensor_end = output_tensor_end_opt->value();
        if (output_tensor_end.has_value()) {
            auto end = output_tensor_end->value();
            TT_FATAL(
                end.rank() == input_shape.rank(),
                "output_tensor_end must have the same rank as the input tensor, but got rank: {}, expected rank: {}",
                end.rank(),
                input_shape.rank());

            for (int d = 0; d < input_shape.rank(); d++) {
                TT_FATAL(
                    end[d] >= 0 && end[d] < input_shape[d],
                    "output_tensor_end[{}] must be in range [0, {}), but got: {}",
                    d,
                    input_shape[d] - 1,
                    end[d]);
            }
        }

        // For rank > 4, fold leading dimensions into one before the op.
        // This is only valid when the crop is a prefix of the folded dimension,
        // i.e., when all non-folded dimensions are cropped to their full extent.
        // If we're cropping a leading dimension that's part of the fold, we must
        // let the op handle it directly.
        if (input_shape.rank() > 4) {
            bool crop_is_prefix = true;
            for (int d = 4; d < input_shape.rank(); d++) {
                if (end[d] != input_shape[d] - 1) {
                    crop_is_prefix = false;
                    break;
                }
            }
            if (crop_is_prefix) {
                // Fold leading dimensions into one
                input_shape = Shape({prod(input_shape.get_rect(0, 4))}, input_shape.tensor_layout().get_grid_config());
            }
        }

        Operation::validate(
            input_tensors,
            optional_input_shapes,
            {input_shape},
            memory_config_opt,
            input_tensors_transformed,
            queue_id);
    }

    std::vector<tensor::Tensor> untilize_with_unpadding_op::create_output_tensors(
        const std::vector<tensor::Tensor>& input_tensors,
        const std::vector<std::vector<std::optional<int64_t>>>>& optional_input_shapes,
        const std::optional<ttnn::operations::unpadding::UnpaddingSpec>& output_tensor_end_opt,
        const std::optional<MemoryConfig>& memory_config_opt,
        const std::optional<const std::vector<tensor::Tensor>>& input_tensors_transformed) const {
        auto input_shape = input_tensors[0].get_padded_shape();
        auto output_tensor_end = output_tensor_end_opt->value();

        // Determine the effective input shape after potential folding
        auto effective_input_shape = input_shape;
        if (input_shape.rank() > 4 && output_tensor_end.has_value()) {
            auto end = output_tensor_end->value();
            bool crop_is_prefix = true;
            for (int d = 4; d < input_shape.rank(); d++) {
                if (end[d] != input_shape[d] - 1) {
                    crop_is_prefix = false;
                    break;
                }
            }
            if (crop_is_prefix) {
                effective_input_shape = Shape({prod(input_shape.get_rect(0, 4))}, input_shape.tensor_layout().get_grid_config());
            }
        }

        auto output_shape = compute_untilize_with_unpadding_output_shape(effective_input_shape, output_tensor_end);

        return {tensor::Tensor(output_shape, input_tensors[0].get_dtype(), input_tensors[0].get_layout(), memory_config_opt.value_or(input_tensors[0].get_memory_config()))};
    }

    std::vector<tensor::Tensor> untilize_with_unpadding_op::invoke(
        const std::vector<tensor::Tensor>& input_tensors,
        const std::optional<QueueId>& queue_id) {
        auto input = input_tensors[0];
        auto output = create_output_tensors({input}, {}, this->get_output_tensor_end(), this->get_memory_config());

        run_operation(input, output, this->get_device_operation(), queue_id);

        return {output};
    }
}  // namespace ttnn::operations::data_movement
