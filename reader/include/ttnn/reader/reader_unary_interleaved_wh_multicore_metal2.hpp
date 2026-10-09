#pragma once

#include "reader/include/ttnn/reader/reader_base.hpp"
#include "tt-metal/tensor/shape.hpp"

namespace ttnn::operations::data_movement {
    class ReaderUnaryInterleavedWHMultiCoreMetal2 : public ReaderBase {
    public:
        ReaderUnaryInterleavedWHMultiCoreMetal2(
            const tensor::Tensor& input,
            int num_tiles_per_2d,
            int third_dim,
            const Shape& output_shape);

        void compute_offset(int slab_idx, std::vector<int>& offset) const override;

    private:
    };
}  // namespace ttnn::operations::data_movement
