#include "untilize_with_unpadding_multi_core_block_interleaved_program_factory.hpp"
#include "reader_unary_interleaved_wh_multicore_metal2.hpp"
#include "writer_unary_metal2.hpp"
#include "command_sequence.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement {
    using namespace ttnn::shape_utils;

    UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory::UntilizeWithUnpaddingProgramFactory(
        const tt::application_model::Device& device,
        const Shape& input_shape,
        const Shape& output_shape,
        const std::optional<MemoryConfig>& memory_config)
        : ProgramFactory(device, memory_config)
        , input_shape_(input_shape)
        , output_shape_(output_shape) {
        // input_shape is now the effective shape (possibly folded for rank > 4)
    }

    std::unique_ptr<OperationSequence> UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory::create(
        const std::vector<tensor::Tensor>& input_tensors,
        tensor::Tensor& output_tensor) const {
        auto input = input_tensors[0];
        auto input_dims = input_shape_.get_rect(0, 4);
        auto output_dims = output_shape_.get_rect(0, 4);

        auto h_in = input_dims[2];
        auto w_in = input_dims[3];
        auto h_out = output_dims[2];
        auto w_out = output_dims[3];

        // Compute tile dimensions
        auto ht = h_out / TILE_DIM;
        auto wt = w_out / TILE_DIM;

        // For block-interleaved, we need to know the number of tiles per 2D slab
        // FIX: Use input dimensions to compute num_tiles_per_2d, not output dimensions
        // The reader walks input slabs contiguously
        auto input_ht = h_in / TILE_DIM;
        auto input_wt = w_in / TILE_DIM;
        int num_tiles_per_2d = input_ht * input_wt;

        // Compute third_dim for the reader
        // third_dim should be based on input shape, not output shape
        // This ensures the reader offset calculation is correct
        int third_dim = input_wt;  // Was: output_wt

        // Create reader with corrected third_dim
        auto reader = std::make_unique<ReaderUnaryInterleavedWHMultiCoreMetal2>(
            input,
            num_tiles_per_2d,
            third_dim,
            output_shape_);

        // Create writer
        auto writer = std::make_unique<WriterUnaryMetal2>(
            output_tensor,
            output_dims,
            TILE_DIM,
            TILE_DIM);

        // Build command sequence
        auto sequence = std::make_unique<OperationSequence>();
        sequence->add_reader(std::move(reader));
        sequence->add_writer(std::move(writer));

        return sequence;
    }

    const Shape& UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory::get_input_shape() const {
        return input_shape_;
    }

    const Shape& UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory::get_output_shape() const {
        return output_shape_;
    }
}  // namespace ttnn::operations::data_movement
