#include "reader_unary_interleaved_wh_multicore_metal2.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement {
    using namespace ttnn::shape_utils;

    ReaderUnaryInterleavedWHMultiCoreMetal2::ReaderUnaryInterleavedWHMultiCoreMetal2(
        const tensor::Tensor& input,
        int num_tiles_per_2d,
        int third_dim,
        const Shape& output_shape)
        : ReaderBase(input, num_tiles_per_2d, third_dim, output_shape) {
        // third_dim is now correctly set based on input shape, not output shape
    }

    void ReaderUnaryInterleavedWHMultiCoreMetal2::compute_offset(
        int slab_idx,
        std::vector<int>& offset) const {
        // offset = dim * num_tiles_per_2d + ...
        // With third_dim based on input shape, this is now correct
        auto dims = this->get_output_shape().get_rect(0, 4);
        int d0 = slab_idx / (this->third_dim_ * this->num_tiles_per_2d_);
        int remainder = slab_idx % (this->third_dim_ * this->num_tiles_per_2d_);
        int d1 = remainder / this->num_tiles_per_2d_;
        int d2 = remainder % this->num_tiles_per_2d_;

        offset = {d0, d1, d2, 0, 0};
    }
}  // namespace ttnn::operations::data_movement
