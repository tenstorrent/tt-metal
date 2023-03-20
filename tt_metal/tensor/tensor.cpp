#include "tt_metal/tensor/tensor.hpp"

#include "tt_metal/tensor/tensor_impl.hpp"
#include "tt_metal/tensor/tensor_impl_wrapper.hpp"
#include "common/bfloat16.hpp"
#include "llrt/llrt.hpp"
#include "constants.hpp"

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
        }
        this->shape_ = other.shape_;
        this->strides_ = other.strides_;
        this->dtype_ = other.dtype_;
        this->layout_ = other.layout_;
        this->device_ = other.device_;
        this->mem_config_ = other.mem_config_;
        if (originally_on_host) {
            if (other.on_host()) {
                // deep copy other.data_ into this->data_
                tensor_impl::deepcopy_host_data_wrapper(other, *this);
            } else {
                // `this` is updated to be a device tensor,
                // allocate buffer of same size and on same device as other.buffer_ and copy data from other.buffer_ into it
                tensor_impl::deepcopy_device_data_wrapper(other, *this);
            }
        } else {
            // deallocate this->buffer_ from device
            // if `this` is updated to be a host tensor no buffer is allocated
            // otherwise a new buffer is allocated holding same data as other.buffer_
            if (other.on_host()) {
                // `this` is updated to be a host tensor, copy data from other into `this`
                tensor_impl::deepcopy_host_data_wrapper(other, *this);
            } else {
                // allocate new buffer with same size and on same device as other.buffer_ and copy data from other.buffer_ into it
                tensor_impl::deepcopy_device_data_wrapper(other, *this);
            }
        }
    }
    return *this;
}

Tensor::Tensor(Tensor &&other)
    : shape_(other.shape_), strides_(other.strides_), dtype_(other.dtype_), layout_(other.layout_), device_(other.device_), mem_config_(other.mem_config_) {
    if (other.on_host()) {
        // deep copy other.data_ into this->data_ and delete other.data_
        tensor_impl::move_host_data_wrapper(std::move(other), *this);
    } else {
        // this owns buffer, does not need to be deallocated from device
        this->buffer_ = std::move(other.buffer_);
        other.buffer_ = nullptr;
    }
}

Tensor &Tensor::operator=(Tensor &&other) {
    if (this != &other) {
        bool originally_on_host = this->on_host();
        if (not originally_on_host) {
            // Always deallocate in this case because `this` is either updated to be host tensor or gets new buffer
            // free the buffer before `this` members get updated
            this->free_buffer();
        }
        this->shape_ = other.shape_;
        this->strides_ = other.strides_;
        this->dtype_ = other.dtype_;
        this->layout_ = other.layout_;
        this->device_ = other.device_;
        this->mem_config_ = other.mem_config_;
        if (originally_on_host) {
            if (other.on_host()) {
                // move other.data_ into this->data_ and free other.data_
                tensor_impl::move_host_data_wrapper(std::move(other), *this);
            } else {
                // `this` is updated to be a device tensor,
                // allocate buffer of same size and on same device as other.buffer_ move data from other.buffer_ into it then deallocate other.buffer_
                tensor_impl::move_device_data_wrapper(std::move(other), *this);
            }
        } else {
            // if `this` is updated to be a host tensor no buffer is allocated
            // otherwise a new buffer is allocated holding same data as other.buffer_
            if (other.on_host()) {
                // `this` is updated to be a host tensor, move data from other into `this`
                tensor_impl::move_host_data_wrapper(std::move(other), *this);
            } else {
                // allocate new buffer with same size and on same device as other.buffer_ and move data from other.buffer_ into it then deallocate other.buffer_
                tensor_impl::move_device_data_wrapper(std::move(other), *this);
            }
        }
    }
    return *this;
}

Tensor::~Tensor() {
    if (not on_host()) {
        this->free_buffer();
    }
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
    vector<int> ns{N, C, H, W};
    int neg_idx = -1;
    for (int i = 0; i < ns.size(); i++) {
        if (ns[i] == -1) {
            TT_ASSERT(neg_idx == -1, "Only one -1 is allowed in Tensor::reshape");
            neg_idx = i;
        } else {
            TT_ASSERT(ns[i] > 0, "New shape entries can only have -1 or positive values");
        }
    }

    uint32_t old_volume = this->volume();

    switch (neg_idx) {
        case 0:
            TT_ASSERT(old_volume % C*H*W == 0);
            N = old_volume/(C*H*W);
            break;
        case 1:
            TT_ASSERT(old_volume % N*H*W == 0);
            C = old_volume/(N*H*W);
            break;
        case 2:
            TT_ASSERT(old_volume % N*C*W == 0);
            H = old_volume/(N*C*W);
            TT_ASSERT(H%32 == 0);
            break;
        case 3:
            TT_ASSERT(old_volume % N*C*H == 0);
            W = old_volume/(N*C*H);
            TT_ASSERT(W%32 == 0);
            break;
        case -1: // In case where there is no negative value in ns
            TT_ASSERT(N*C*H*W == old_volume);
            break;
        default:
            TT_ASSERT(false && "Unexpected neg_idx in Tensor::reshape!");
    }

    if (this->layout() == Layout::TILE) {
        TT_ASSERT(H % 32 == 0 && W % 32 == 0 && "Expected a multiple of 32 for H, W (or -1 evaluating to such) in Tensor::reshape()!");
    }

    shape_[0] = N;
    shape_[1] = C;
    shape_[2] = H;
    shape_[3] = W;
    strides_ = compute_strides();

    return shape_;
}

void Tensor::free_buffer() {
    TT_ASSERT(not on_host() && "Tensor needs to have a buffer on device to free it!");
    if (this->buffer_ == nullptr) {
        return;
    }
    DeallocateBuffer(this->buffer_);
    delete this->buffer_;
    this->buffer_ = nullptr;
}

}  // namespace tt_metal

}  // namespace tt
