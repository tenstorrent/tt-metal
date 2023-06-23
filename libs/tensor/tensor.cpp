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

Tensor::Tensor(std::vector<bfloat16> &data, const std::array<uint32_t, 4> &shape, DataType dtype, Layout layout)
    : shape_(shape), strides_(compute_strides()), dtype_(dtype), layout_(layout) {
    tensor_impl::convert_and_write_data_wrapper(*this, data);
}

Tensor::Tensor(std::vector<bfloat16> &data, const std::array<uint32_t, 4> &shape, DataType dtype, Layout layout, Device *device, const MemoryConfig &mem_config)
    : shape_(shape), strides_(compute_strides()), dtype_(dtype), layout_(layout), device_(device), mem_config_(mem_config) {
    TT_ASSERT(device != nullptr);
    tensor_impl::validate_on_device_dtype_and_layout(device, dtype, layout);
    tensor_impl::convert_and_write_data_wrapper(*this, data);
}

Tensor::Tensor(std::vector<uint32_t> &data, const std::array<uint32_t, 4> &shape, DataType dtype, Layout layout)
    : shape_(shape), strides_(compute_strides()), dtype_(dtype), layout_(layout) {
    tensor_impl::convert_and_write_data_wrapper(*this, data);
}

Tensor::Tensor(std::vector<uint32_t> &data, const std::array<uint32_t, 4> &shape, DataType dtype, Layout layout, Device *device, const MemoryConfig &mem_config)
    : shape_(shape), strides_(compute_strides()), dtype_(dtype), layout_(layout), device_(device), mem_config_(mem_config) {
    TT_ASSERT(device != nullptr);
    tensor_impl::validate_on_device_dtype_and_layout(device, dtype, layout);
    tensor_impl::convert_and_write_data_wrapper(*this, data);
}

Tensor::Tensor(std::vector<float> &data, const std::array<uint32_t, 4> &shape, DataType dtype, Layout layout)
    : shape_(shape), strides_(compute_strides()), dtype_(dtype), layout_(layout) {
    tensor_impl::convert_and_write_data_wrapper(*this, data);
}

Tensor::Tensor(std::vector<float> &data, const std::array<uint32_t, 4> &shape, DataType dtype, Layout layout, Device *device, const MemoryConfig &mem_config)
    : shape_(shape), strides_(compute_strides()), dtype_(dtype), layout_(layout), device_(device), mem_config_(mem_config) {
    TT_ASSERT(device != nullptr);
    tensor_impl::validate_on_device_dtype_and_layout(device, dtype, layout);
    tensor_impl::convert_and_write_data_wrapper(*this, data);
}

Tensor::Tensor(const std::array<uint32_t, 4> &shape, Initialize init_type, DataType dtype, Layout layout)
    : shape_(shape), strides_(compute_strides()), dtype_(dtype), layout_(layout) {
    tensor_impl::initialize_data_wrapper(*this, init_type);
}

Tensor::Tensor(const std::array<uint32_t, 4> &shape, Initialize init_type, DataType dtype, Layout layout, Device *device, const MemoryConfig &mem_config)
    : shape_(shape), strides_(compute_strides()), dtype_(dtype), layout_(layout), device_(device), mem_config_(mem_config) {
    TT_ASSERT(device != nullptr);
    tensor_impl::validate_on_device_dtype_and_layout(device, dtype, layout);
    tensor_impl::initialize_data_wrapper(*this, init_type);
}

Tensor::Tensor(const std::array<uint32_t, 4> &shape, DataType dtype, Layout layout, Device *device, const MemoryConfig &mem_config)
    : shape_(shape), strides_(compute_strides()), dtype_(dtype), layout_(layout), device_(device), mem_config_(mem_config) {
    TT_ASSERT(device != nullptr);
    tensor_impl::validate_on_device_dtype_and_layout(device, dtype, layout);
    uint32_t packed_size_in_bytes = tensor_impl::packed_buffer_size_bytes_wrapper(dtype, volume());
    tensor_impl::allocate_buffer_on_device(*this, packed_size_in_bytes);
}

Tensor::Tensor(const Tensor &other)
    : shape_(other.shape_), strides_(other.strides_), dtype_(other.dtype_), layout_(other.layout_), device_(other.device_), mem_config_(other.mem_config_) {
    if (other.on_host()) {
        // deep copy other.data_ into this->data_
        tensor_impl::deepcopy_host_data_wrapper(other, *this);
    } else {
        // allocate buffer of same size and on same device as other.buffer_ and copy data from other.buffer_ into it
        tensor_impl::deepcopy_device_data_wrapper(other, *this);
    }
}

Tensor &Tensor::operator=(const Tensor &other) {
    if (this != &other) {
        bool originally_on_host = this->on_host();
        if (not originally_on_host) {
            // Always deallocate in this case because `this` is either updated to be host tensor or gets new buffer
            // free the buffer before `this` members get updated
            this->free_buffer();
        } else {
            tensor_impl::free_data_wrapper(*this);
        }
        this->shape_ = other.shape_;
        this->strides_ = other.strides_;
        this->dtype_ = other.dtype_;
        this->layout_ = other.layout_;
        this->device_ = other.device_;
        this->mem_config_ = other.mem_config_;

        if (other.on_host()) {
            // deep copy other.data_ into this->data_
            tensor_impl::deepcopy_host_data_wrapper(other, *this);
        } else {
            // allocate new buffer of same size and on same device as other.buffer_ and copy data from other.buffer_ into it
            tensor_impl::deepcopy_device_data_wrapper(other, *this);
        }
    }
    return *this;
}

Tensor::Tensor(Tensor &&other)
    : shape_(other.shape_), strides_(other.strides_), dtype_(other.dtype_), layout_(other.layout_), device_(other.device_), mem_config_(other.mem_config_) {
    if (other.on_host()) {
        // move other.data_ into this->data_
        tensor_impl::move_host_data_wrapper(std::move(other), *this);
    } else {
        // this owns buffer, does not need to be deallocated from device
        tensor_impl::move_device_data_wrapper(std::move(other), *this);
    }
}

Tensor &Tensor::operator=(Tensor &&other) {
    if (this != &other) {
        bool originally_on_host = this->on_host();
        if (not originally_on_host) {
            // Always deallocate in this case because `this` is either updated to be host tensor or gets new buffer
            // free the buffer before `this` members get updated
            this->free_buffer();
        } else {
            tensor_impl::free_data_wrapper(*this);
        }
        this->shape_ = other.shape_;
        this->strides_ = other.strides_;
        this->dtype_ = other.dtype_;
        this->layout_ = other.layout_;
        this->device_ = other.device_;
        this->mem_config_ = other.mem_config_;

        if (other.on_host()) {
            // move other.data_ into this->data_
            tensor_impl::move_host_data_wrapper(std::move(other), *this);
        } else {
            // move other.buffer_ into this->buffer_
            tensor_impl::move_device_data_wrapper(std::move(other), *this);
        }
    }
    return *this;
}

Tensor::~Tensor() {
    this->deallocate();
}

void Tensor::deallocate() {
    if (not on_host()) {
        this->free_buffer();
    }
    tensor_impl::free_data_wrapper(*this);
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
        return Tensor(*this);
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

void Tensor::free_buffer() {
    TT_ASSERT(not on_host() && "Tensor needs to have a buffer on device to free it!");
    if (this->buffer_ == nullptr) {
        return;
    }
    DeallocateBuffer(*this->buffer_);
    delete this->buffer_;
    this->buffer_ = nullptr;
}

}  // namespace tt_metal

}  // namespace tt
