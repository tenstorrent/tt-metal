#pragma once

#include "program_factory/include/ttnn/program_factory/program_factory.hpp"
#include "tt-metal/tensor/shape.hpp"
#include <memory>

namespace ttnn::operations::data_movement {
    class UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory : public ProgramFactory {
    public:
        UntilizeWithUnpaddingProgramFactory(
            const tt::application_model::Device& device,
            const Shape& input_shape,
            const Shape& output_shape,
            const std::optional<MemoryConfig>& memory_config);

        std::unique_ptr<OperationSequence> create(
            const std::vector<tensor::Tensor>& input_tensors,
            tensor::Tensor& output_tensor) const override;

        const Shape& get_input_shape() const;
        const Shape& get_output_shape() const;

    private:
        Shape input_shape_;
        Shape output_shape_;
    };
}  // namespace ttnn::operations::data_movement
