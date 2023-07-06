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

namespace detail {
template<class>
inline constexpr bool always_false_v = false;
}

Tensor::Tensor(const HostStorage& storage, const Shape& shape, DataType dtype, Layout layout)
    : storage_(storage), shape_(shape), dtype_(dtype), layout_(layout) {}

Tensor::Tensor(const DeviceStorage& storage, const Shape& shape, DataType dtype, Layout layout)
    : storage_(storage), shape_(shape), dtype_(dtype), layout_(layout) {
    TT_ASSERT(storage.device != nullptr);
    tensor_impl::validate_on_device_dtype_and_layout(storage.device, dtype, layout);
}

Tensor::~Tensor() {
    this->deallocate();
}

void Tensor::deallocate() {
    std::visit(
        [](auto&& storage)
        {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, HostStorage>) {
                storage.buffer.reset();
            }
            else if constexpr (std::is_same_v<T, DeviceStorage>) {
                if (storage.buffer.use_count() == 1) {
                     DeallocateBuffer(*storage.buffer);
                }
                storage.buffer.reset();
            }
            else {
                static_assert(detail::always_false_v<T>, "non-exhaustive visitor!");
            }
        },
        this->storage_
    );
}

Tensor Tensor::to(Device *target_device, const MemoryConfig &mem_config) const {
    if (storage_type() == StorageType::HOST) {
        tensor_impl::validate_on_device_dtype_and_layout(target_device, this->dtype(), this->layout());
        return tensor_impl::to_device_wrapper(*this, target_device, mem_config);
    }
    TT_ASSERT(this->device_storage().value().device == target_device && "Currently do not support moving between devices");
    return Tensor(*this);
}

Tensor Tensor::to(Host *host) const {
    if (storage_type() == StorageType::HOST) {
        return *this;
    }
    return tensor_impl::to_host_wrapper(*this);
}

Tensor Tensor::to(Layout target_layout) const {
    TT_ASSERT(this->storage_type() == StorageType::HOST && "Bring tensor to host before converting to target layout");
    return tensor_impl::to_layout_wrapper(*this, target_layout);
}

void Tensor::print(Layout print_layout, bool pretty_print) const {
    tensor_impl::print_wrapper(*this, print_layout, pretty_print);
}

Tensor Tensor::pad(const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value) const {
    TT_ASSERT(this->storage_type() == StorageType::HOST && "Tensor must be on host for padding");
    TT_ASSERT(this->layout() == Layout::ROW_MAJOR && "Tensor layout must be ROW_MAJOR for padding");
    return tensor_impl::pad_wrapper(*this, output_tensor_shape, input_tensor_start, pad_value);
}

Tensor Tensor::unpad(const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end) const {
    TT_ASSERT(this->storage_type() == StorageType::HOST && "Tensor must be on host for unpadding");
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

uint32_t Tensor::element_size() const {
    return tensor_impl::element_size_bytes_wrapper(this->dtype_);
}

Tensor Tensor::reshape(int N, int C, int H, int W) {
    auto new_shape = infer_dims_for_reshape(N, C, H, W, this->volume());
    return this->reshape(new_shape);
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    if (this->layout() == Layout::TILE) {
        TT_ASSERT(new_shape[2] % TILE_HEIGHT == 0 && new_shape[3] % TILE_WIDTH == 0 && "Expected a multiple of 32 for H, W (or -1 evaluating to such) in Tensor::reshape()!");
    }

    auto new_tensor = *this;
    new_tensor.shape_ = new_shape;
    return new_tensor;
}

bool Tensor::is_allocated() const {
    return std::visit(
        [] (auto&& storage) -> bool
        {
            return bool(storage.buffer);
        },
        this->storage_
    );
}


StorageType Tensor::storage_type() const {
    return std::visit(
        [] (auto&& storage) -> StorageType
        {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, HostStorage>) {
                return StorageType::HOST;
            }
            else if constexpr (std::is_same_v<T, DeviceStorage>) {
                return StorageType::DEVICE;
            }
            else {
                static_assert(detail::always_false_v<T>, "non-exhaustive visitor!");
            }
        },
        this->storage_
    );
}

const std::optional<HostStorage> Tensor::host_storage() const {
    if (std::holds_alternative<HostStorage>(this->storage_)) {
        return std::get<HostStorage>(this->storage_);
    }
    return std::nullopt;
}

const std::optional<DeviceStorage> Tensor::device_storage() const {
    if (std::holds_alternative<DeviceStorage>(this->storage_)) {
        return std::get<DeviceStorage>(this->storage_);
    }
    return std::nullopt;
}

namespace detail {
const std::array<uint32_t, 4> compute_strides(const Shape& shape) {
    return {shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1};
}
}

const std::array<uint32_t, 4> Tensor::strides() const {
    return detail::compute_strides(this->shape_);
}

uint32_t Tensor::volume() const {
    return tt::tt_metal::volume(this->shape_);
}

Tensor create_device_tensor(const Shape& shape, DataType data_type, Layout layout, Device *device, const MemoryConfig& memory_config) {
    uint32_t packed_size_in_bytes = tensor_impl::packed_buffer_size_bytes_wrapper(data_type, volume(shape));
    auto device_buffer = tensor_impl::allocate_buffer_on_device(packed_size_in_bytes, device, shape, data_type, layout, memory_config);
    return Tensor(DeviceStorage{device_buffer, device, memory_config}, shape, data_type, layout);
}

}  // namespace tt_metal

}  // namespace tt
