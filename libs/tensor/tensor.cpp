#include "tensor/tensor.hpp"

#include "tensor/tensor_impl.hpp"
#include "tensor/tensor_impl_wrapper.hpp"
#include "tensor/tensor_utils.hpp"
#include "common/bfloat16.hpp"
#include "llrt/llrt.hpp"
#include "tt_metal/common/constants.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

Tensor::Tensor(const HostBuffer& host_buffer, const std::array<uint32_t, 4> &shape, DataType dtype, Layout layout)
    : data_(host_buffer), shape_(shape), strides_(compute_strides()), dtype_(dtype), layout_(layout) {
}

Tensor::Tensor(const HostBuffer& host_buffer, const std::array<uint32_t, 4> &shape, DataType dtype, Layout layout, Device *device, const MemoryConfig &mem_config)
    : shape_(shape), strides_(compute_strides()), dtype_(dtype), layout_(layout), device_(device), mem_config_(mem_config) {
    TT_ASSERT(device != nullptr);
    tensor_impl::validate_on_device_dtype_and_layout(device, dtype, layout);
    switch (dtype) {
        case DataType::UINT32: {
            tensor_impl::convert_and_write_data_wrapper<uint32_t>(*this, host_buffer);
            break;
        }
        case DataType::FLOAT32: {
            tensor_impl::convert_and_write_data_wrapper<float>(*this, host_buffer);
            break;
        }
        case DataType::BFLOAT16: {
            tensor_impl::convert_and_write_data_wrapper<bfloat16>(*this, host_buffer);
            break;
        }
        case DataType::BFLOAT8_B: {
            tensor_impl::convert_and_write_data_wrapper<float>(*this, host_buffer);
            break;
        }
    }
}

Tensor::Tensor(const std::array<uint32_t, 4> &shape, DataType dtype, Layout layout, Device *device, const MemoryConfig &mem_config)
    : shape_(shape), strides_(compute_strides()), dtype_(dtype), layout_(layout), device_(device), mem_config_(mem_config) {
    TT_ASSERT(device != nullptr);
    tensor_impl::validate_on_device_dtype_and_layout(device, dtype, layout);
    uint32_t packed_size_in_bytes = tensor_impl::packed_buffer_size_bytes_wrapper(dtype, volume());
    tensor_impl::allocate_buffer_on_device(*this, packed_size_in_bytes);
}

Tensor::~Tensor() {
    this->deallocate();
}

void Tensor::deallocate() {
    if (not on_host() and this->is_allocated_on_device() and this->buffer_.use_count() == 1) {
        DeallocateBuffer(*this->buffer_);
    }

    this->data_.reset();
    this->buffer_.reset();
}

Tensor Tensor::to(Device *target_device, const MemoryConfig &mem_config) const {
    if (on_host()) {
        tensor_impl::validate_on_device_dtype_and_layout(target_device, this->dtype(), this->layout());
        return tensor_impl::to_device_wrapper(*this, target_device, mem_config);
    }
    TT_ASSERT(this->device_ == target_device && "Currently do not support moving between devices");
    return Tensor(*this);
}

Tensor Tensor::to(Host *host) const {
    if (on_host()) {
        return *this;
    }
    return tensor_impl::to_host_wrapper(*this);
}

Tensor Tensor::to(Layout target_layout) const {
    TT_ASSERT(on_host() && "Bring tensor to host before converting to target layout");
    return tensor_impl::to_layout_wrapper(*this, target_layout);
}

void Tensor::print(Layout print_layout, bool pretty_print) const {
    tensor_impl::print_wrapper(*this, print_layout, pretty_print);
}

Tensor Tensor::pad(const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value) const {
    TT_ASSERT(on_host() && "Tensor must be on host for padding");
    TT_ASSERT(this->layout() == Layout::ROW_MAJOR && "Tensor layout must be ROW_MAJOR for padding");
    return tensor_impl::pad_wrapper(*this, output_tensor_shape, input_tensor_start, pad_value);
}

Tensor Tensor::unpad(const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end) const {
    TT_ASSERT(on_host() && "Tensor must be on host for unpadding");
    TT_ASSERT(this->layout() == Layout::ROW_MAJOR && "Tensor layout must be ROW_MAJOR for unpadding");
    return tensor_impl::unpad_wrapper(*this, output_tensor_start, output_tensor_end);
}

Tensor Tensor::pad_to_tile(float pad_value) const {
    uint32_t padded_h = roundup(this->shape()[2], TILE_HEIGHT);
    uint32_t padded_w = roundup(this->shape()[3], TILE_WIDTH);
    std::array<uint32_t, 4> output_tensor_shape = {this->shape()[0], this->shape()[1], padded_h, padded_w};
    std::array<uint32_t, 4> input_tensor_start = {0, 0, 0, 0};

    return this->pad(output_tensor_shape, input_tensor_start, pad_value);
}

Tensor Tensor::unpad_from_tile(const std::array<uint32_t, 4> &output_tensor_shape) const {
    TT_ASSERT(this->shape()[0] == output_tensor_shape[0] && this->shape()[1] == output_tensor_shape[1], "Input shape must match output shape apart from last 2 dims");
    TT_ASSERT(this->shape()[2] % TILE_HEIGHT == 0 && this->shape()[3] % TILE_WIDTH==0, "Last 2 dims of input shape must be multiples of 32");
    TT_ASSERT(this->shape()[2] - TILE_HEIGHT < output_tensor_shape[2] && this->shape()[3] - TILE_WIDTH < output_tensor_shape[3], "Last 2 dims of output must be within range to have been padded to input");
    std::array<uint32_t, 4> output_tensor_start = {0, 0, 0, 0};
    std::array<uint32_t, 4> output_tensor_end = {output_tensor_shape[0] - 1, output_tensor_shape[1] - 1, output_tensor_shape[2] - 1, output_tensor_shape[3] - 1};
    return this->unpad(output_tensor_start, output_tensor_end);
}

// Prints like numpy print function to make it more readable. Only supports row major layout.
void Tensor::pretty_print() const {
    print(Layout::ROW_MAJOR, /*pretty_print=*/true);
}

bool Tensor::interleaved() const {
    if (this->on_host()) {
        return false;
    }
    return mem_config_.interleaved;
}

uint32_t Tensor::element_size() const {
    return tensor_impl::element_size_bytes_wrapper(this->dtype_);
}

const std::array<uint32_t, 4>& Tensor::reshape(int N, int C, int H, int W) {
    auto new_shape = infer_dims_for_reshape(N, C, H, W, this->volume());

    if (this->layout() == Layout::TILE) {
        TT_ASSERT(new_shape[2] % TILE_HEIGHT == 0 && new_shape[3] % TILE_WIDTH == 0 && "Expected a multiple of 32 for H, W (or -1 evaluating to such) in Tensor::reshape()!");
    }

    shape_ = new_shape;
    strides_ = compute_strides();

    return shape_;
}

Tensor Tensor::reshape(const std::array<uint32_t, 4>& new_shape) const {
    auto new_tensor = *this;
    new_tensor.shape_ = new_shape;
    new_tensor.strides_ = new_tensor.compute_strides();
    return new_tensor;
}

}  // namespace tt_metal

}  // namespace tt
