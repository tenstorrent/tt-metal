// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/borrowed_buffer_functions.hpp"
#include "tensor/owned_buffer_functions.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_utils.hpp"
#include "tensor/types.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_stl/concepts.hpp"

#include <optional>
#include "tensor/tensor_impl_wrapper.hpp"

namespace tt {

namespace tt_metal {

namespace tensor_impl {

std::array<uint32_t, 2> get_sharded_page_shape(Layout layout,  DataType dtype, std::array<uint32_t, 2> shard_shape);

// -----------------------------------------------------------------------------------------------------------------------------------------------
// ===============================================================================================================================================
//                                                              Low Level APIs
// ===============================================================================================================================================
// -----------------------------------------------------------------------------------------------------------------------------------------------

// ======================================================================================
//                        Data type converters, packers, and unpackers
// ======================================================================================
// TODO(arakhmati): Should cast_vec be a generator?

template <typename OutputDataType, template<typename> typename BufferType, typename InputDataType>
std::vector<OutputDataType> cast_vec(const BufferType<InputDataType>& data_to_convert) {
    std::vector<OutputDataType> converted_data;
    for (auto datum : data_to_convert) {
        if constexpr (std::is_same_v<OutputDataType, float> and std::is_same_v<InputDataType, bfloat16>) {
            converted_data.push_back(datum.to_float());
        }
        else if constexpr (std::is_same_v<OutputDataType, uint32_t> and std::is_same_v<InputDataType, bfloat16>) {
            converted_data.push_back((uint32_t)datum.to_uint16());
        }
        else {
            converted_data.push_back(static_cast<OutputDataType>(datum));
        }
    }
    return converted_data;
}

// TODO(arakhmati): Should pack_vec_into_uint32_vec be a generator?
template <typename DataType, template <typename> typename BufferType>
std::vector<uint32_t> pack_vec_into_uint32_vec(const BufferType<DataType>& data_to_pack) {
    if constexpr (std::is_same_v<DataType, uint32_t>) {
        return std::vector(std::begin(data_to_pack), std::end(data_to_pack));
    } else if constexpr (std::is_same_v<DataType, uint16_t>) {
        std::vector<uint32_t> output;
        for (auto index = 0; index < data_to_pack.size(); index += 2) {
            auto value = data_to_pack[index + 1] << 16 | data_to_pack[index];
            output.push_back(value);
        }
        return output;
    } else if constexpr (std::is_same_v<DataType, bfloat16>) {
        auto bfloat16_vec = std::vector(std::begin(data_to_pack), std::end(data_to_pack));
        return pack_bfloat16_vec_into_uint32_vec(bfloat16_vec);
    } else if constexpr (std::is_same_v<DataType, float>) {
        std::vector<uint32_t> uint32_data;
        union float_uint32_convert {
            uint32_t u;
            float f;
            float_uint32_convert() : u(0) {}
        };
        for (auto i = 0; i < data_to_pack.size(); i ++) {
            float_uint32_convert a;
            a.f = data_to_pack[i];
            uint32_data.push_back(a.u);
        }
        return uint32_data;
    } else {
        static_assert(tt::stl::concepts::always_false_v<DataType>, "Don't know how to unpack uint32 data generically!");
    }
}

template <typename DataType>
std::vector<DataType> unpack_uint32_vec(std::vector<uint32_t>& data_to_unpack) {
    if constexpr (std::is_same_v<DataType, uint32_t>) {
        return data_to_unpack;
    } else if constexpr (std::is_same_v<DataType, uint16_t>) {
        std::vector<DataType> output;
        for (auto index = 0; index < data_to_unpack.size(); index++) {
            output.push_back(data_to_unpack[index] & 0xFFFF);
            output.push_back(data_to_unpack[index] >> 16);
        }
        return output;
    } else if constexpr (std::is_same_v<DataType, bfloat16>) {
        return unpack_uint32_vec_into_bfloat16_vec(data_to_unpack);
    } else if constexpr (std::is_same_v<DataType, float>) {
        union float_uint32_convert {
            uint32_t u;
            float f;
            float_uint32_convert() : u(0) {}
        };
        std::vector<float> float_data;
        for (auto i = 0; i < data_to_unpack.size(); i++) {
            float_uint32_convert a;
            a.u = data_to_unpack[i];
            float_data.push_back(a.f);
        }
        return float_data;
    } else {
        static_assert(tt::stl::concepts::always_false_v<DataType>, "Don't know how to unpack uint32 data generically!");
    }
}

template <typename T>
constexpr inline uint32_t element_size_bytes() {
    return sizeof(T);
}

template <typename T>
constexpr inline uint32_t packed_buffer_size_bytes(uint32_t volume_unpacked_data) {
    auto num_type_in_u32 = sizeof(uint32_t) / sizeof(T);
    return (volume_unpacked_data/num_type_in_u32) * sizeof(uint32_t);
}

// Specialization for float because it gets converted to bfloat16 before being packed
template <>
constexpr inline uint32_t packed_buffer_size_bytes<float>(uint32_t volume_unpacked_data) {
    auto num_type_in_u32 = sizeof(uint32_t) / sizeof(float);
    return (volume_unpacked_data / num_type_in_u32) * sizeof(uint32_t);
}

// ======================================================================================
//                                  Layout converters
// ======================================================================================
namespace detail {
static std::vector<uint32_t> to_4D_shape(const Shape& shape) {
    if (shape.rank() == 1) {
        return {1, 1, 1, shape[-1]};
    } else if (shape.rank() == 2) {
        return {1, 1, shape[-2], shape[-1]};
    } else if (shape.rank() == 3) {
        return {1, shape[-3], shape[-2], shape[-1]};
    } else if (shape.rank() == 4) {
        return {shape[-4], shape[-3], shape[-2], shape[-1]};
    } else {
        TT_THROW("Rank {} is not supported!", shape.rank());
    }
}
}  // namespace detail

template <typename T, template<typename> typename BufferType>
inline std::vector<T> convert_layout_row_major_to_tile(const Shape& shape, const BufferType<T>& data_to_convert) {
    TT_ASSERT(
        (shape[-2] % tt::constants::TILE_HEIGHT == 0 && shape[-1] % tt::constants::TILE_WIDTH == 0),
        "Unsupported shape for tensor conversion");
    auto shape_vec = detail::to_4D_shape(shape);
    return convert_layout(data_to_convert, shape_vec, TensorLayout::LIN_ROW_MAJOR, TensorLayout::TILED32_4FACES);
}

template <typename T, template<typename> typename BufferType>
inline std::vector<T> convert_layout_tile_to_row_major(const Shape& shape, const BufferType<T>& data_to_convert) {
    auto shape_vec = detail::to_4D_shape(shape);
    return convert_layout(data_to_convert, shape_vec, TensorLayout::TILED32_4FACES, TensorLayout::LIN_ROW_MAJOR);
}

// ======================================================================================
//                                      Validators
// ======================================================================================
void validate_on_device_dtype_and_layout(Device* device, DataType dtype, Layout layout);

// -----------------------------------------------------------------------------------------------------------------------------------------------
// ===============================================================================================================================================
//                                                              High Level APIs
// ===============================================================================================================================================
// -----------------------------------------------------------------------------------------------------------------------------------------------

// ======================================================================================
//                           Data reader, writer, and initializers
// ======================================================================================
DeviceBuffer allocate_buffer_on_device(
    uint32_t buffer_size_bytes,
    Device* device,
    const Shape& shape,
    DataType data_type,
    Layout layout,
    const MemoryConfig& memory_config,
    std::optional<ShardSpecBuffer> shard_spec = std::nullopt);

template <typename T>
inline void read_data_from_device_buffer(
    CommandQueue& cq, DeviceBuffer device_buffer, void* host_buffer_data, bool blocking) {
    EnqueueReadBuffer(cq, device_buffer, host_buffer_data, blocking);
}

template <typename T>
inline void read_data_from_device_buffer(DeviceBuffer device_buffer, vector<T>& host_buffer) {
    std::vector<uint32_t> host_buffer_uint32;
    ::detail::ReadFromBuffer(device_buffer, host_buffer_uint32);
    host_buffer = unpack_uint32_vec<T>(host_buffer_uint32);
}

template <typename T, template <typename> typename BufferType>
inline void write_data_to_device_buffer(
    CommandQueue & cq, const BufferType<T>& host_buffer, DeviceBuffer device_buffer) {
    ZoneScoped;
    // TODO(arakhmati): can we use generators in this function to go from `data_to_write` to `uint32_data`?
    // And effectively get rid of any additional allocation
    if (CommandQueue::default_mode() == CommandQueue::CommandQueueMode::ASYNC) {
        if constexpr (std::is_same_v<BufferType<T>, borrowed_buffer::Buffer<T>>) {
            // When writing borrowed storage asynchronously, we have no control over when host memory is deallocated by the main thread.
            // To ensure that worker threads enqueues the correct buffer, make a copy and caputre it in an owned buffer.
            uint32_t borrowed_buf_size_words = device_buffer->num_pages() * device_buffer->page_size() / sizeof(uint32_t);
            const uint32_t* borrowed_buf_base = static_cast<const uint32_t*>(host_buffer.data());
            std::vector<uint32_t> owned_copy_vec(borrowed_buf_base, borrowed_buf_base + borrowed_buf_size_words);
            owned_buffer::Buffer<uint32_t> owned_copy(std::make_shared<std::vector<uint32_t>>(owned_copy_vec));
            EnqueueWriteBuffer( cq, device_buffer, owned_copy.get_ptr(), false);
        }
        else if constexpr (std::is_same_v<BufferType<T>, owned_buffer::Buffer<T>>) {
            EnqueueWriteBuffer( cq, device_buffer, host_buffer.get_ptr(), false);
        }
    }
    else {
        EnqueueWriteBuffer(cq, device_buffer, host_buffer.data(), false);
    }
}

template <typename T, template <typename> typename BufferType>
inline void write_data_to_device_buffer(const BufferType<T>& host_buffer, Buffer& device_buffer) {
    ZoneScoped;
    // TODO(arakhmati): can we use generators in this function to go from `data_to_write` to `uint32_data`?
    // And effectively get rid of any additional allocation

    auto uint32_data = pack_vec_into_uint32_vec<T>(host_buffer);
    ::detail::WriteToBuffer(device_buffer, uint32_data);
}


template <typename T, template <typename> typename BufferType>
inline DeviceBuffer initialize_data_on_device(
    BufferType<T>& data_to_write,
    Device* device,
    const Shape& shape,
    DataType data_type,
    Layout layout,
    const MemoryConfig& memory_config,
    std::optional<ShardSpecBuffer> shard_spec,
    std::optional<std::reference_wrapper<CommandQueue>> queue = std::nullopt) {
    ZoneScoped;
    TT_ASSERT(device != nullptr);
    auto packed_size_in_bytes = packed_buffer_size_bytes<T>(data_to_write.size());

    auto device_buffer =
        allocate_buffer_on_device(packed_size_in_bytes, device, shape, data_type, layout, memory_config, shard_spec);
    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        write_data_to_device_buffer<T>(
            queue.has_value() ? queue.value().get() : device->command_queue(), data_to_write, device_buffer);
    } else {
        write_data_to_device_buffer<T>(data_to_write, *device_buffer);
    }
    return device_buffer;
}

template <typename T>
inline DeviceBuffer to_device_buffer(
    const Storage& storage,
    Device* device,
    const Shape& shape,
    DataType data_type,
    Layout layout,
    const MemoryConfig& memory_config,
    std::optional<ShardSpecBuffer> shard_spec,
    std::optional<std::reference_wrapper<CommandQueue>> queue) {
    return std::visit(
        [&device, &shape, &data_type, &layout, memory_config, shard_spec](auto&& storage) -> DeviceBuffer {
            using StorageType = std::decay_t<decltype(storage)>;
            if (memory_config.is_sharded()) {
                TT_ASSERT(shard_spec.has_value(), "If sharded must provide shard_spec");
            }
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                auto data_to_write = owned_buffer::get_as<T>(storage.buffer);
                TT_ASSERT(
                    compute_buffer_size(shape, data_type) == data_to_write.size(),
                    fmt::format(
                        "Tensor buffer size and number of data elements does not match: {} != {}",
                        compute_buffer_size(shape, data_type),
                        data_to_write.size()));
                if (layout == Layout::TILE) {
                    TT_ASSERT(
                        (shape[-2] % tt::constants::TILE_HEIGHT == 0 && shape[-1] % tt::constants::TILE_WIDTH == 0),
                        "Tensor shape incompatible for specified layout");
                }
                return initialize_data_on_device<T>(
                    data_to_write, device, shape, data_type, layout, memory_config, shard_spec);
            }
            else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Device storage doesn't support to_device_buffer");
            } else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                if constexpr (
                    std::is_same_v<T, float> or std::is_same_v<T, bfloat16> or std::is_same_v<T, std::uint32_t> or
                    std::is_same_v<T, std::uint16_t>) {
                    auto data_to_write = borrowed_buffer::get_as<T>(storage.buffer);
                    TT_ASSERT(
                        compute_buffer_size(shape, data_type) == data_to_write.size(),
                        fmt::format(
                            "Tensor buffer size and number of data elements does not match: {} != {}",
                            compute_buffer_size(shape, data_type),
                            data_to_write.size()));
                    if (layout == Layout::TILE) {
                        TT_ASSERT(
                            (shape[-2] % tt::constants::TILE_HEIGHT == 0 && shape[-1] % tt::constants::TILE_WIDTH == 0),
                            "Tensor shape incompatible for specified layout");
                    }
                    return initialize_data_on_device<T>(
                        data_to_write, device, shape, data_type, layout, memory_config, shard_spec);

                } else {
                    TT_THROW("Borrowed storage doesn't support this data type");
                }
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
                TT_THROW("MultiHostStorage storage doesn't support to_device_buffer");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                TT_THROW("MultiDeviceStorage doesn't support to_device_buffer");
            } else {
                raise_unsupported_storage<StorageType>();
            }
        },
        storage);
}

// ======================================================================================
//                                         .to()
// ======================================================================================
template <typename T>
inline Tensor to_host(const Tensor& tensor, bool blocking = true) {
    if (tensor.storage_type() != StorageType::DEVICE) {
        return tensor;
    }
    TT_ASSERT(tensor.is_allocated(), "Buffer must be allocated on device!");
    auto device_buffer = tensor.device_buffer();
    auto device = tensor.device();
    TT_ASSERT(device != nullptr && "Need device to be set copy data from device to host!");
    uint32_t size_in_bytes = device_buffer->size();
    vector<T> data_vec;
    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        data_vec.resize(size_in_bytes / sizeof(T));
        read_data_from_device_buffer<T>(device->command_queue(), device_buffer, data_vec.data(), blocking);
    } else {
        read_data_from_device_buffer<T>(device_buffer, data_vec);
    }
    auto output_buffer = owned_buffer::create<T>(std::move(data_vec));
    return Tensor(OwnedStorage{output_buffer}, tensor.get_legacy_shape(), tensor.get_dtype(), tensor.get_layout());
}

template <typename T>
inline Tensor to_host_sharded(const Tensor& tensor) {
    TT_ASSERT(tensor.is_allocated(), "Buffer must be allocated on device!");
    auto device_buffer = tensor.buffer();
    auto device = tensor.device();
    TT_ASSERT(device != nullptr && "Need device to be set copy data from device to host!");
    uint32_t size_in_bytes = device_buffer->size();
    std::vector<uint32_t> device_data;
    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        TT_THROW("FAST_DISPATCH is not supported for to_host_sharded!");
    }
    ::detail::ReadFromBuffer(*device_buffer, device_data, true);
    auto data_vec = unpack_uint32_vec<T>(device_data);
    auto output_buffer = owned_buffer::create<T>(std::move(data_vec));
    return Tensor(OwnedStorage{output_buffer}, tensor.get_legacy_shape(), tensor.get_dtype(), tensor.get_layout());
}

template <typename T>
inline Tensor to_device(
    const Tensor& tensor,
    Device* target_device,
    const MemoryConfig& memory_config,
    std::optional<std::reference_wrapper<CommandQueue>> queue) {
    TT_ASSERT(tensor.storage_type() != StorageType::DEVICE);
    if (tensor.storage_type() == StorageType::OWNED) {
        TT_ASSERT(tensor.is_allocated(), "Need host buffer on device to exist to copy data to device!");
    }
    TT_ASSERT(target_device != nullptr && "Need target device in order to move tensor to device!");
    TT_ASSERT(tensor.is_allocated() && "Need data to exist in order to move it to device");

    auto shape = tensor.get_legacy_shape();
    auto data_type = tensor.get_dtype();
    auto layout = tensor.get_layout();

    std::optional<ShardSpecBuffer> shard_spec_buffer_opt = std::nullopt;
    if (memory_config.is_sharded()) {
        auto page_shape = get_sharded_page_shape(layout, data_type, memory_config.shard_spec.value().shape);
        std::array<uint32_t, 2> tensor2d_size = {
            shape[0] * shape[1] * shape[2] / page_shape[0], shape[3] / page_shape[1]};
        shard_spec_buffer_opt = ShardSpecBuffer(memory_config.shard_spec.value(), page_shape, tensor2d_size);
    }

    auto device_buffer = tensor_impl::to_device_buffer<T>(
        tensor.get_storage(), target_device, shape, data_type, layout, memory_config, shard_spec_buffer_opt, queue);
    return Tensor(DeviceStorage{device_buffer}, shape, data_type, layout);
}

template <typename T>
inline Tensor to_layout(const Tensor& tensor, Layout target_layout) {
    if (tensor.get_layout() == target_layout) {
        return tensor;
    }

    auto shape = tensor.get_legacy_shape();
    auto source_layout = tensor.get_layout();
    auto convert = [&shape, source_layout, target_layout](const auto& input_data) -> std::vector<T> {
        switch (source_layout) {
            case Layout::ROW_MAJOR:
                if (target_layout == Layout::TILE) {
                    return convert_layout_row_major_to_tile(shape, input_data);
                } else {
                    TT_THROW("Unsupported layout conversion");
                }
                break;
            case Layout::TILE:
                if (target_layout == Layout::ROW_MAJOR) {
                    return convert_layout_tile_to_row_major(shape, input_data);
                } else {
                    TT_THROW("Unsupported layout conversion");
                }
                break;
            default: TT_THROW("Unsupported layout conversion");
        }
    };

    auto output_storage = std::visit(
        [&convert](auto&& storage) -> std::variant<OwnedStorage, MultiDeviceHostStorage> {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                const auto input_data = owned_buffer::get_as<T>(storage.buffer);
                auto output_buffer = owned_buffer::create<T>(std::move(convert(input_data)));
                return OwnedStorage{output_buffer};
            } else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                const auto input_data = borrowed_buffer::get_as<T>(storage.buffer);
                auto output_buffer = owned_buffer::create<T>(std::move(convert(input_data)));
                return OwnedStorage{output_buffer};
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                std::vector<OwnedBuffer> output_buffers;
                std::vector<Shape> output_shapes;
                for (int i = 0; i < storage.buffers.size(); i++) {
                    const auto input_data = owned_buffer::get_as<T>(storage.buffers[i]);
                    auto output_buffer = owned_buffer::create<T>(std::move(convert(input_data)));
                    output_buffers.push_back(output_buffer);
                    output_shapes.push_back(storage.shapes[i]);
                }
                return MultiDeviceHostStorage{output_buffers, output_shapes};
            } else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
                TT_THROW("On-device layout conversion for tensor with MultiDeviceStorage is not supported.");
            } else {
                raise_unsupported_storage<StorageType>();
            }
        },
        tensor.get_storage());


    return std::visit(
        [&tensor, &target_layout](auto&& storage) -> Tensor {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                return Tensor(storage, tensor.get_legacy_shape(), tensor.get_dtype(), target_layout);
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                return Tensor(storage, tensor.get_legacy_shape(), tensor.get_dtype(), target_layout);
            } else {
                raise_unsupported_storage<StorageType>();
            }
        }, output_storage);
}

Tensor to_layout_bfloat8_b(const Tensor& tensor, Layout target_layout);
Tensor to_layout_bfloat4_b(const Tensor& tensor, Layout target_layout);

// ======================================================================================
//                                  .pad() and .unpad()
// ======================================================================================
template <typename T>
inline Tensor pad(
    const Tensor& tensor, const Shape& output_tensor_shape, const Shape& input_tensor_start, float pad_value) {
    auto pad_value_ = static_cast<T>(pad_value);
    const auto input_tensor_shape = tensor.get_legacy_shape();
    const auto input_tensor_strides = tensor.strides();
    const auto input_tensor_data_type = tensor.get_dtype();

    auto pad = [&input_tensor_shape,
                &input_tensor_strides,
                &input_tensor_data_type,
                &output_tensor_shape,
                &input_tensor_start,
                &pad_value_](const auto& input_buffer) {
        // Check if input tensor fits in output tensor given the input tensor start indices
        TT_ASSERT(input_tensor_shape[0] + input_tensor_start[0] <= output_tensor_shape[0]);
        TT_ASSERT(input_tensor_shape[1] + input_tensor_start[1] <= output_tensor_shape[1]);
        TT_ASSERT(input_tensor_shape[2] + input_tensor_start[2] <= output_tensor_shape[2]);
        TT_ASSERT(input_tensor_shape[3] + input_tensor_start[3] <= output_tensor_shape[3]);

        // Figure out pad size on each dim
        uint32_t pad_size[4][2] = {
            {input_tensor_start[0], output_tensor_shape[0] - input_tensor_shape[0] - input_tensor_start[0]},
            {input_tensor_start[1], output_tensor_shape[1] - input_tensor_shape[1] - input_tensor_start[1]},
            {input_tensor_start[2], output_tensor_shape[2] - input_tensor_shape[2] - input_tensor_start[2]},
            {input_tensor_start[3], output_tensor_shape[3] - input_tensor_shape[3] - input_tensor_start[3]}};

        const std::array<uint32_t, 4> output_tensor_strides = {
            output_tensor_shape[1] * output_tensor_shape[2] * output_tensor_shape[3],
            output_tensor_shape[2] * output_tensor_shape[3],
            output_tensor_shape[3],
            1};

        auto output_buffer = owned_buffer::create<T>(compute_volume(output_tensor_shape));
        auto output_index = 0;
        for (auto i = 0; i < pad_size[0][0] * output_tensor_strides[0]; i++) {
            output_buffer[output_index++] = pad_value_;
        }
        for (auto dim0 = 0; dim0 < input_tensor_shape[0]; dim0++) {
            for (auto i = 0; i < pad_size[1][0] * output_tensor_strides[1]; i++) {
                output_buffer[output_index++] = pad_value_;
            }
            for (auto dim1 = 0; dim1 < input_tensor_shape[1]; dim1++) {
                for (auto i = 0; i < pad_size[2][0] * output_tensor_strides[2]; i++) {
                    output_buffer[output_index++] = pad_value_;
                }
                for (auto dim2 = 0; dim2 < input_tensor_shape[2]; dim2++) {
                    for (auto i = 0; i < pad_size[3][0] * output_tensor_strides[3]; i++) {
                        output_buffer[output_index++] = pad_value_;
                    }
                    for (auto dim3 = 0; dim3 < input_tensor_shape[3]; dim3++) {
                        auto input_index = dim3 + input_tensor_strides[2] * dim2 + input_tensor_strides[1] * dim1 +
                                           input_tensor_strides[0] * dim0;
                        output_buffer[output_index++] = input_buffer[input_index];
                    }
                    for (auto i = 0; i < pad_size[3][1] * output_tensor_strides[3]; i++) {
                        output_buffer[output_index++] = pad_value_;
                    }
                }
                for (auto i = 0; i < pad_size[2][1] * output_tensor_strides[2]; i++) {
                    output_buffer[output_index++] = pad_value_;
                }
            }
            for (auto i = 0; i < pad_size[1][1] * output_tensor_strides[1]; i++) {
                output_buffer[output_index++] = pad_value_;
            }
        }
        for (auto i = 0; i < pad_size[0][1] * output_tensor_strides[0]; i++) {
            output_buffer[output_index++] = pad_value_;
        }
        return output_buffer;
    };

    auto output_buffer = std::visit(
        [&pad](auto&& storage) -> owned_buffer::Buffer<T> {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                const auto input_data = owned_buffer::get_as<T>(storage.buffer);
                return pad(input_data);
            } else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                const auto input_data = borrowed_buffer::get_as<T>(storage.buffer);
                return pad(input_data);
            } else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                TT_THROW("Device storage isn't supported");
            } else {
                raise_unsupported_storage<StorageType>();
            }
        },
        tensor.get_storage());
    return Tensor(OwnedStorage{output_buffer}, output_tensor_shape, tensor.get_dtype(), tensor.get_layout());
}

Tensor pad_bfloat8_b(
    const Tensor& tensor, const Shape& output_tensor_shape, const Shape& input_tensor_start, float pad_value);
Tensor pad_bfloat4_b(
    const Tensor& tensor, const Shape& output_tensor_shape, const Shape& input_tensor_start, float pad_value);

template <typename T>
inline Tensor unpad(const Tensor& tensor, const Shape& output_tensor_start, const Shape& output_tensor_end) {
    const auto input_tensor_shape = tensor.get_legacy_shape();
    const auto input_tensor_strides = tensor.strides();

    // Check if tensor start and end indices are within input tensor shape
    TT_ASSERT(output_tensor_start[0] < input_tensor_shape[0]);
    TT_ASSERT(output_tensor_end[0] < input_tensor_shape[0]);
    TT_ASSERT(output_tensor_start[1] < input_tensor_shape[1]);
    TT_ASSERT(output_tensor_end[1] < input_tensor_shape[1]);
    TT_ASSERT(output_tensor_start[2] < input_tensor_shape[2]);
    TT_ASSERT(output_tensor_end[2] < input_tensor_shape[2]);
    TT_ASSERT(output_tensor_start[3] < input_tensor_shape[3]);
    TT_ASSERT(output_tensor_end[3] < input_tensor_shape[3]);

    // Check if start shape is <= end shape
    TT_ASSERT(output_tensor_start[0] <= output_tensor_end[0]);
    TT_ASSERT(output_tensor_start[1] <= output_tensor_end[1]);
    TT_ASSERT(output_tensor_start[2] <= output_tensor_end[2]);
    TT_ASSERT(output_tensor_start[3] <= output_tensor_end[3]);

    // Figure out output tensor shape
    const Shape output_tensor_shape = {
        output_tensor_end[0] - output_tensor_start[0] + 1,
        output_tensor_end[1] - output_tensor_start[1] + 1,
        output_tensor_end[2] - output_tensor_start[2] + 1,
        output_tensor_end[3] - output_tensor_start[3] + 1,
    };

    auto unpad =
        [&input_tensor_shape, &input_tensor_strides, &output_tensor_shape, &output_tensor_start, &output_tensor_end](
            const auto& input_buffer) {
            auto output_buffer = owned_buffer::create<T>(compute_volume(output_tensor_shape));
            auto output_index = 0;
            for (auto dim0 = output_tensor_start[0]; dim0 <= output_tensor_end[0]; dim0++) {
                for (auto dim1 = output_tensor_start[1]; dim1 <= output_tensor_end[1]; dim1++) {
                    for (auto dim2 = output_tensor_start[2]; dim2 <= output_tensor_end[2]; dim2++) {
                        for (auto dim3 = output_tensor_start[3]; dim3 <= output_tensor_end[3]; dim3++) {
                            auto input_index = dim3 + input_tensor_strides[2] * dim2 + input_tensor_strides[1] * dim1 +
                                               input_tensor_strides[0] * dim0;
                            output_buffer[output_index++] = input_buffer[input_index];
                        }
                    }
                }
            }
            return output_buffer;
        };

    auto output_buffer = std::visit(
        [&unpad](auto&& storage) -> owned_buffer::Buffer<T> {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                const auto input_data = owned_buffer::get_as<T>(storage.buffer);
                return unpad(input_data);
            } else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                const auto input_data = borrowed_buffer::get_as<T>(storage.buffer);
                return unpad(input_data);
            } else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                TT_THROW("Device storage isn't supported");
            } else {
                raise_unsupported_storage<StorageType>();
            }
        },
        tensor.get_storage());
    return Tensor(OwnedStorage{output_buffer}, output_tensor_shape, tensor.get_dtype(), tensor.get_layout());
}

Tensor unpad_bfloat8_b(const Tensor& tensor, const Shape& output_tensor_start, const Shape& output_tensor_end);
Tensor unpad_bfloat4_b(const Tensor& tensor, const Shape& output_tensor_start, const Shape& output_tensor_end);

// ======================================================================================
//                                         Print
// ======================================================================================

std::ostream& operator<<(std::ostream& os, const DataType& dtype);

enum class TensorPrintProfile {
    Empty,
    Short,
    Full,
};

inline TensorPrintProfile TTNN_TENSOR_PRINT_PROFILE = TensorPrintProfile::Short;

namespace detail {

struct DimensionShortener {
    size_t size;
    std::optional<std::size_t> max;

    bool print_parenthesis_and_advance_index_if_reached_half_of_max_and_check_if_loop_is_done(
        std::ostream& ss, std::size_t& index, const std::string& before, const std::string& after) const {
        if (this->max.has_value() and this->size >= this->max.value() and index == this->max.value() / 2) {
            ss << before << "...," << after;
            index = this->size - (this->max.value() / 2);
        }
        return index < this->size;
    }
};

inline DimensionShortener get_dimension_shortener(std::size_t size) {
    switch (TTNN_TENSOR_PRINT_PROFILE) {
        case TensorPrintProfile::Empty: return DimensionShortener{size, 0};
        case TensorPrintProfile::Short: return DimensionShortener{size, 4};
        case TensorPrintProfile::Full: return DimensionShortener{size, std::nullopt};
        default: TT_THROW("Unrecognized TTNN_TENSOR_PRINT_PROFILE {}", TTNN_TENSOR_PRINT_PROFILE);
    }
}

inline void print_trailing_comma(std::ostream& ss, std::size_t index, std::size_t size, const std::string& after) {
    if (index < size - 1) {
        ss << "," << after;
    }
}

template <typename T>
inline void print_datum(std::ostream& ss, T datum) {
    if (std::is_integral<T>::value) {
        ss << fmt::format("{:5}", datum);
    } else {
        ss << fmt::format("{:8.5f}", datum);
    }
}

template <>
inline void print_datum(std::ostream& ss, bfloat16 datum) {
    print_datum(ss, datum.to_float());
}

inline constexpr int constexpr_strlen(const char* str) { return *str ? 1 + constexpr_strlen(str + 1) : 0; }

constexpr auto TENSOR_TYPE_STRING = "ttnn.Tensor";
constexpr auto TENSOR_TYPE_STRING_PLUS_OPEN_PARENTHESIS_LENGTH = constexpr_strlen(TENSOR_TYPE_STRING) + 1;

static constexpr auto TAB = "    ";
static constexpr auto TAB_MINUS_1 = "   ";

template <typename BufferType, std::int64_t Rank, std::int64_t Dim = 0>
void to_string_row_major(
    std::stringstream& ss,
    const BufferType& buffer,
    const Shape& shape,
    std::size_t outer_index,
    const std::size_t buffer_offset) {
    auto stride = 1;
    for (auto index = Dim + 1; index < shape.rank(); index++) {
        stride *= shape[index];
    }

    std::string spaces = std::string(TENSOR_TYPE_STRING_PLUS_OPEN_PARENTHESIS_LENGTH + Dim, ' ');
    std::string before;
    std::string after;
    if constexpr (Rank == 1) {
        before = " ";
        after = " ";
    } else if constexpr (Rank == 2) {
        before = spaces + " ";
        after = "\n";
    } else {
        before = spaces + " ";
        after = "\n\n";
    }

    if (Dim > 0 and outer_index > 0) {
        ss << spaces;
    }
    ss << "[";
    auto dimension_shortener = get_dimension_shortener(shape[-Rank]);
    for (std::size_t index = 0;
         dimension_shortener.print_parenthesis_and_advance_index_if_reached_half_of_max_and_check_if_loop_is_done(
             ss, index, before, after);
         index++) {
        std::string after_comma;
        if constexpr (Rank == 1) {
            after_comma = " ";
        } else if constexpr (Rank == 2) {
            after_comma = "\n";
        } else {
            after_comma = after;
        }

        if constexpr (Rank > 1) {
            to_string_row_major<BufferType, Rank - 1, Dim + 1>(
                ss, buffer, shape, index, buffer_offset + index * stride);
        } else {
            print_datum(ss, buffer[buffer_offset + index]);
        }
        print_trailing_comma(ss, index, shape[-Rank], after_comma);
    }
    ss << "]";
}

template <typename BufferType, std::int64_t Rank, std::int64_t Dim = 0>
void to_string_tile(
    std::stringstream& ss,
    const BufferType& buffer,
    const Shape& shape,
    std::size_t outer_index,
    const std::size_t buffer_offset) {
    // For now, print it the same way as row-major
    return to_string_row_major<BufferType, Rank, Dim>(ss, buffer, shape, outer_index, buffer_offset);
}

template <typename BufferType>
std::string to_string(const BufferType& buffer, const Shape& shape, DataType dtype, Layout layout) {
    std::stringstream ss;
    ss << TENSOR_TYPE_STRING << "(";

    if (TTNN_TENSOR_PRINT_PROFILE == TensorPrintProfile::Empty) {
        ss << "...";
    } else if (layout == Layout::ROW_MAJOR) {
        switch (shape.rank()) {
            case 0: to_string_row_major<BufferType, 0>(ss, buffer, shape, 0, 0); break;
            case 1: to_string_row_major<BufferType, 1>(ss, buffer, shape, 0, 0); break;
            case 2: to_string_row_major<BufferType, 2>(ss, buffer, shape, 0, 0); break;
            case 3: to_string_row_major<BufferType, 3>(ss, buffer, shape, 0, 0); break;
            case 4: to_string_row_major<BufferType, 4>(ss, buffer, shape, 0, 0); break;
            case 5: to_string_row_major<BufferType, 5>(ss, buffer, shape, 0, 0); break;
            case 6: to_string_row_major<BufferType, 6>(ss, buffer, shape, 0, 0); break;
            case 7: to_string_row_major<BufferType, 7>(ss, buffer, shape, 0, 0); break;
            case 8: to_string_row_major<BufferType, 8>(ss, buffer, shape, 0, 0); break;
            default: TT_THROW("Unsupported Rank for printing tensor with ROW_MAJOR_LAYOUT!"); break;
        }
    } else if (layout == Layout::TILE) {
        switch (shape.rank()) {
            case 2: to_string_tile<BufferType, 2>(ss, buffer, shape, 0, 0); break;
            case 3: to_string_tile<BufferType, 3>(ss, buffer, shape, 0, 0); break;
            case 4: to_string_tile<BufferType, 4>(ss, buffer, shape, 0, 0); break;
            case 5: to_string_tile<BufferType, 5>(ss, buffer, shape, 0, 0); break;
            case 6: to_string_tile<BufferType, 6>(ss, buffer, shape, 0, 0); break;
            case 7: to_string_tile<BufferType, 7>(ss, buffer, shape, 0, 0); break;
            case 8: to_string_tile<BufferType, 8>(ss, buffer, shape, 0, 0); break;
            default: TT_THROW("Unsupported Rank for printing tensor with TILE_LAYOUT!"); break;
        }
    } else {
        TT_THROW("Unsupported Layout for printing tensor!");
    }
    ss << ", shape=" << fmt::format("{}", shape) << ", dtype=" << fmt::format("{}", dtype)
       << ", layout=" << fmt::format("{}", layout) << ")";
    return ss.str();
}

}  // namespace detail

template <typename T>
inline std::string to_string(const Tensor& tensor, std::optional<DataType> original_dtype = std::nullopt) {
    const auto shape = tensor.get_legacy_shape();
    const auto dtype = original_dtype.value_or(tensor.get_dtype());
    const auto layout = tensor.get_layout();

    if (not tensor.is_allocated()) {
        return fmt::format(
            "{}(<buffer is not allocated>, shape={}, dtype={}, layout={})",
            detail::TENSOR_TYPE_STRING,
            shape,
            dtype,
            layout);
    }

    if (tensor.storage_type() == StorageType::DEVICE) {
        return to_string<T>(to_host<T>(tensor));
    }

    if (dtype == DataType::BFLOAT8_B and original_dtype == std::nullopt) {
        // Convert to FLOAT32 tensor before printing
        auto input_packed_data = owned_buffer::get_as<uint32_t>(tensor).get();
        auto input_float_data = unpack_bfp8_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false);
        auto input_float_buffer = owned_buffer::create<float>(std::move(input_float_data));
        auto float_tensor =
            Tensor(OwnedStorage{input_float_buffer}, tensor.get_legacy_shape(), DataType::FLOAT32, tensor.get_layout());
        return to_string<float>(float_tensor, tensor.get_dtype());
    }

    if (dtype == DataType::BFLOAT4_B and original_dtype == std::nullopt) {
        // Convert to FLOAT32 tensor before printing
        auto input_packed_data = owned_buffer::get_as<uint32_t>(tensor).get();
        auto input_float_data = unpack_bfp4_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false);
        auto input_float_buffer = owned_buffer::create<float>(std::move(input_float_data));
        auto float_tensor =
            Tensor(OwnedStorage{input_float_buffer}, tensor.get_legacy_shape(), DataType::FLOAT32, tensor.get_layout());
        return to_string<float>(float_tensor, tensor.get_dtype());
    }

    return std::visit(
        [&](auto&& storage) -> std::string {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                const auto buffer = owned_buffer::get_as<T>(storage.buffer);
                return detail::to_string(buffer, shape, dtype, layout);
            }
            else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                const auto buffer = borrowed_buffer::get_as<T>(storage.buffer);
                return detail::to_string(buffer, shape, dtype, layout);
            }
            else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Cannot print a device tensor!");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                TT_THROW("Device storage isn't supported");
            } else {
                raise_unsupported_storage<StorageType>();
            }
        },
        tensor.get_storage());
}

template <typename T>
Tensor extract_shard(const Tensor & tensor, const uint32_t & core_id){

    auto buffer= tensor.buffer();
    auto buffer_shard_shape = buffer->shard_spec().shape();
    std::array <uint32_t, 4> shard_shape_array = {1,1,buffer_shard_shape[0],buffer_shard_shape[1]};
    Shape shard_shape(shard_shape_array);
    std::vector<uint32_t> device_data;
    ::detail::ReadShard(*buffer, device_data, core_id);


    auto unpacked_data = tensor_impl::unpack_uint32_vec<T>(device_data);
    auto output_buffer = owned_buffer::create<T>(std::move(unpacked_data));
    return Tensor(OwnedStorage{output_buffer}, shard_shape, tensor.get_dtype(), tensor.get_layout());

}

template <typename DataType>
void* get_raw_host_data_ptr(const Tensor& tensor) {
    return std::visit(
        [](auto&& storage) -> void* {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                auto buffer = owned_buffer::get_as<DataType>(storage.buffer);
                return buffer.data();
            } else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                if constexpr (
                    std::is_same_v<DataType, float> or std::is_same_v<DataType, bfloat16> or
                    std::is_same_v<DataType, std::uint32_t> or std::is_same_v<DataType, std::uint16_t>) {
                    auto buffer = borrowed_buffer::get_as<DataType>(storage.buffer);
                    return buffer.data();
                } else {
                    TT_THROW("Borrowed storage doesn't support this data type");
                }
            } else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                TT_THROW("Device storage isn't supported");
            } else {
                raise_unsupported_storage<StorageType>();
            }
        },
        tensor.get_storage());
}

}  // namespace tensor_impl

}  // namespace tt_metal

}  // namespace tt
