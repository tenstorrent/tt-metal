#include "untilize_with_unpadding_device_operation.hpp"
#include "program_factory/src/untilize_with_unpadding_multi_core_block_interleaved_program_factory.hpp"
#include "program_factory/src/untilize_with_unpadding_multi_core_row_program_factory.hpp"
#include "program_factory/src/untilize_with_unpadding_single_core_program_factory.hpp"
#include "program_factory/src/untilize_with_unpadding_program_factory.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement {
    using namespace ttnn::shape_utils;

    UntilizeWithUnpaddingDeviceOperation::UntilizeWithUnpaddingDeviceOperation(
        const tt::application_model::Device& device,
        const std::optional<ttnn::operations::unpadding::UnpaddingSpec>& output_tensor_end_opt,
        const std::optional<MemoryConfig>& memory_config_opt)
        : DeviceOperation(device) {
        this->set_output_tensor_end(output_tensor_end_opt);
        this->set_memory_config(memory_config_opt);

        auto input_shape = this->get_input_shape();
        auto output_tensor_end = this->get_output_tensor_end();

        // Determine the effective input shape for program selection
        // If rank > 4 and crop is a prefix, the folding was done in validate
        // and we should use the folded shape
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

        auto input_dims = effective_input_shape.get_rect(0, 4);
        auto h_in = input_dims[2];
        auto w_in = input_dims[3];

        // Compute output dimensions
        std::vector<int> output_dims(4, 0);
        if (output_tensor_end.has_value()) {
            auto end = output_tensor_end->value();
            for (int d = 0; d < 4; d++) {
                output_dims[d] = end[d] + 1;
            }
        } else {
            for (int d = 0; d < 4; d++) {
                output_dims[d] = input_dims[d];
            }
        }

        auto h_out = output_dims[2];
        auto w_out = output_dims[3];

        // Select program factory based on dimensions
        bool use_multicore = this->get_memory_config().has_value() && this->get_memory_config()->memory_layout == TensorMemoryLayout::BLOCK_INTERLEAVED;

        if (use_multicore) {
            // Check if block-interleaved factory should be used
            // Chosen when Wt > 32 and (Ht > 32 or Wt > Ht)
            auto wt = w_out / TILE_DIM;
            auto ht = h_out / TILE_DIM;

            if (wt > 32 && (ht > 32 || wt > ht)) {
                // Block-interleaved factory
                // FIX: Pass input dimensions to the factory, not output dimensions
                // The reader walks input slabs contiguously, so we need input shape
                this->set_program_factory(std::make_unique<untilize_with_unpadding_multi_core_block_interleaved_program_factory>(
                    device,
                    effective_input_shape,  // Use effective input shape
                    Shape(output_dims),      // Output shape
                    this->get_memory_config()));
            } else {
                // Row factory or single-core
                this->set_program_factory(std::make_unique<untilize_with_unpadding_multi_core_row_program_factory>(
                    device,
                    effective_input_shape,
                    Shape(output_dims),
                    this->get_memory_config()));
            }
        } else {
            // Single-core factory
            this->set_program_factory(std::make_unique<untilize_with_unpadding_single_core_program_factory>(
                device,
                effective_input_shape,
                Shape(output_dims),
                this->get_memory_config()));
        }
    }

    void UntilizeWithUnpaddingDeviceOperation::prepare(
        const std::vector<tensor::Tensor>& input_tensors,
        std::vector<tensor::Tensor>& output_tensors) {
        // Implementation
    }

    void UntilizeWithUnpaddingDeviceOperation::compute(
        const std::vector<tensor::Tensor>& input_tensors,
        std::vector<tensor::Tensor>& output_tensors) {
        // Implementation
    }
}  // namespace ttnn::operations::data_movement
