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
namespace tt {

namespace tt_metal {

namespace tensor_impl {

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
template <typename DataType, template<typename> typename BufferType>
constexpr inline std::vector<uint32_t> pack_vec_into_uint32_vec(const BufferType<DataType>& data_to_pack) {
    if constexpr (std::is_same_v<DataType, uint32_t>) {
        return std::vector(std::begin(data_to_pack), std::end(data_to_pack));
    } else if constexpr (std::is_same_v<DataType, uint16_t>) {
        std::vector<uint32_t> output;
        for (auto index = 0; index < data_to_pack.size(); index += 2) {
            auto value = data_to_pack[index] << 16 | data_to_pack[index + 1];
            output.push_back(value);
        }
        return output;
    } else if constexpr (std::is_same_v<DataType, bfloat16>) {
        auto bfloat16_vec = std::vector(std::begin(data_to_pack), std::end(data_to_pack));
        return pack_bfloat16_vec_into_uint32_vec(bfloat16_vec);
    } else if constexpr (std::is_same_v<DataType, float>) {
        std::vector<uint32_t> uint32_data;
        assert(data_to_pack.size() % 2 == 0);
        for (auto i = 0; i < data_to_pack.size(); i += 2) {
            auto float_val1 = data_to_pack[i];
            auto float_val2 = data_to_pack[i + 1];
            auto bfloat_val1 = bfloat16(float_val1);
            auto bfloat_val2 = bfloat16(float_val2);
            auto uint32_val = pack_two_bfloat16_into_uint32({bfloat_val1, bfloat_val2});
            uint32_data.push_back(uint32_val);
        }
        return uint32_data;
    } else {
        static_assert(tt::stl::concepts::always_false_v<DataType>, "Don't know how to unpack uint32 data generically!");
    }
}

template <typename DataType>
constexpr inline std::vector<DataType> unpack_uint32_vec(std::vector<uint32_t> &data_to_unpack) {
    if constexpr (std::is_same_v<DataType, uint32_t>) {
        return data_to_unpack;
    } else if constexpr (std::is_same_v<DataType, uint16_t>) {
        std::vector<DataType> output;
        for (auto index = 0; index < data_to_unpack.size(); index++) {
            output.push_back(data_to_unpack[index] >> 16);
            output.push_back(data_to_unpack[index] & 0xFFFF);
        }
        return output;
    } else if constexpr (std::is_same_v<DataType, bfloat16>) {
        return unpack_uint32_vec_into_bfloat16_vec(data_to_unpack);
    } else if constexpr (std::is_same_v<DataType, float>) {
        std::vector<float> float_data;
        for (auto i = 0; i < data_to_unpack.size(); i++) {
            auto unpacked = unpack_two_bfloat16_from_uint32(data_to_unpack[i]);
            auto float_val1 = unpacked.first.to_float();
            auto float_val2 = unpacked.second.to_float();
            float_data.push_back(float_val1);
            float_data.push_back(float_val2);
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
    auto num_type_in_u32 = sizeof(uint32_t) / sizeof(bfloat16);
    return (volume_unpacked_data/num_type_in_u32) * sizeof(uint32_t);
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
//                                         Print
// ======================================================================================
std::ostream& operator<<(std::ostream& os, const DataType& dtype);

namespace detail {

template <typename T>
inline void print_datum(std::ostream& ss, T datum) {
    ss << datum;
}

template <>
inline void print_datum(std::ostream& ss, bfloat16 datum) {
    ss << datum.to_float();
}

template <typename BufferType>
std::string to_string(const BufferType& buffer, DataType dtype) {
    std::stringstream ss;
    ss << "[ ";
    for (int i = 0; i < buffer.size(); i++) {
        print_datum(ss, buffer[i]);
        if (i < buffer.size() - 1) {
            ss << ", ";
        }
    }
    ss << " dtype=" <<  dtype << " ]" << std::endl;
    return ss.str();
}

template <typename BufferType>
std::string to_string_row_major_0D(const BufferType& buffer, const Shape& shape, DataType dtype) {

    std::stringstream ss;
    ss << "Tensor( ";
    print_datum(ss, buffer[0]);
    ss << ", dtype=" <<  dtype << " )" << std::endl;
    return ss.str();
}

template <typename BufferType>
std::string to_string_row_major_1D(const BufferType& buffer, const Shape& shape, DataType dtype) {

    std::stringstream ss;
    ss << "Tensor([ ";
    for(auto x = 0; x < shape[0]; x++) {
        // data in row major order
        auto index = x;
        print_datum(ss, buffer[index]);
        if (x < shape[0] - 1) {
            ss << ", ";
        }
    }
    ss << "], dtype=" <<  dtype << " )" << std::endl;
    return ss.str();
}

template <typename BufferType>
std::string to_string_row_major_2D(const BufferType& buffer, const Shape& shape, DataType dtype) {

    std::stringstream ss;
    ss << "Tensor([ ";
    for(auto y = 0; y < shape[0]; y++) {
        if (y == 0)
            ss << "[";
        else
            ss << "    [";
        for(auto x = 0; x < shape[1]; x++) {
            // data in row major order
            auto index = x + y*shape[1];
            print_datum(ss, buffer[index]);
            if (x < shape[1] - 1) {
                ss << ", ";
            }
        }
        if(y < shape[0] - 1)
            ss << "]," << std::endl;
        else
            ss << "]";
    }
    ss << "], dtype=" <<  dtype << " )" << std::endl;
    return ss.str();
}

template <typename BufferType>
std::string to_string_row_major_3D(const BufferType& buffer, const Shape& shape, DataType dtype) {

    std::stringstream ss;
    ss << "Tensor([ ";
    for(auto z = 0; z < shape[0]; z++) {
        if (z == 0)
            ss << "[";
        else
            ss << "   [";
        for(auto y = 0; y < shape[1]; y++) {
            if (y == 0)
                ss << "[";
            else
                ss << "    [";
            for(auto x = 0; x < shape[2]; x++) {
                // data in row major order
                auto index = x + y*shape[2] + z*shape[1]*shape[2];
                print_datum(ss, buffer[index]);
                if (x < shape[2] - 1) {
                    ss << ", ";
                }
            }
            if(y < shape[1] - 1)
                ss << "]," << std::endl;
            else
                ss << "]";
        }
        if(z < shape[0] - 1)
            ss << "]," << std::endl << std::endl;
        else
            ss << "]";
    }
    ss << "], dtype=" <<  dtype << " )" << std::endl;
    return ss.str();
}

template <typename BufferType>
std::string to_string_row_major_4D(const BufferType& buffer, const Shape& shape, DataType dtype) {

    std::stringstream ss;
    ss << "Tensor([ ";
    for(auto w = 0; w < shape[0]; w++) {
        if(w == 0)
            ss << "[";
        else
            ss << "  [";
        for(auto z = 0; z < shape[1]; z++) {
            if (z == 0)
                ss << "[";
            else
                ss << "   [";
            for(auto y = 0; y < shape[2]; y++) {
                if (y == 0)
                    ss << "[";
                else
                    ss << "    [";
                for(auto x = 0; x < shape[3]; x++) {
                    // data in row major order
                    auto index = x + y*shape[3] + z*shape[2]*shape[3] + w*shape[1]*shape[2]*shape[3];
                    print_datum(ss, buffer[index]);
                    if (x < shape[3] - 1) {
                        ss << ", ";
                    }
                }
                if(y < shape[2] - 1)
                    ss << "]," << std::endl;
                else
                    ss << "]";
            }
            if(z < shape[1] - 1)
                ss << "]," << std::endl << std::endl;
            else
                ss << "]";
        }
        if(w < shape[0] - 1)
            ss << "]," << std::endl << std::endl << std::endl;
        else
            ss << "]";
    }
    ss << "], dtype=" <<  dtype << " )" << std::endl;
    return ss.str();
}


template <typename BufferType>
std::string to_string_row_major(const BufferType& buffer, const Shape& shape, DataType dtype) {
    if (shape.rank() == 0) {
        return to_string_row_major_0D(buffer, shape, dtype);
    }
    if (shape.rank() == 1) {
        return to_string_row_major_1D(buffer, shape, dtype);
    }
    else if (shape.rank() == 2) {
        return to_string_row_major_2D(buffer, shape, dtype);
    }
    else if (shape.rank() == 3) {
        return to_string_row_major_3D(buffer, shape, dtype);
    }
    else if (shape.rank() == 4) {
        return to_string_row_major_4D(buffer, shape, dtype);
    }
    else {
        TT_THROW("Cannot print tensor of rank ", shape.rank());
    }
}

} // namespace detail


// ======================================================================================
//                                      Validators
// ======================================================================================
void validate_on_device_dtype_and_layout(Device *device, DataType dtype, Layout layout);

// -----------------------------------------------------------------------------------------------------------------------------------------------
// ===============================================================================================================================================
//                                                              High Level APIs
// ===============================================================================================================================================
// -----------------------------------------------------------------------------------------------------------------------------------------------

// ======================================================================================
//                           Data reader, writer, and initializers
// ======================================================================================
DeviceBuffer allocate_sharded_buffer_on_device(uint32_t buffer_size_bytes, Device *device, uint32_t shard_size, const MemoryConfig& memory_config);

DeviceBuffer allocate_buffer_on_device(
    uint32_t buffer_size_bytes,
    Device *device,
    const Shape& shape,
    DataType data_type,
    Layout layout,
    const MemoryConfig& memory_config
);

template <typename T>
std::vector<T> read_data_from_device(const Tensor &tensor, uint32_t size_in_bytes) {
    auto device_buffer = tensor.buffer();
    TT_ASSERT(device_buffer->size() == size_in_bytes);

    const char *TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        std::vector<T> device_data;
        device_data.resize(size_in_bytes / sizeof(T));
        EnqueueReadBuffer(*tt::tt_metal::detail::GLOBAL_CQ, *device_buffer, device_data.data(), true);
        return device_data;
    } else {
        std::vector<uint32_t> device_data;
        ::detail::ReadFromBuffer(*device_buffer, device_data);
        return unpack_uint32_vec<T>(device_data);
    }
}

template <typename T, template<typename> typename BufferType>
inline void write_data_to_device_buffer(const BufferType<T>& data_to_write, DeviceBuffer buffer, const Shape& shape, DataType data_type, Layout layout, const MemoryConfig& memory_config) {
    ZoneScoped;
    // TODO(arakhmati): can we use generators in this function to go from `data_to_write` to `uint32_data`?
    // And effectively get rid of any additional allocation

    const char *TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        EnqueueWriteBuffer(*tt::tt_metal::detail::GLOBAL_CQ, *buffer, std::begin(data_to_write), false);
    } else {
        auto uint32_data = pack_vec_into_uint32_vec<T>(data_to_write);
        ::detail::WriteToBuffer(*buffer, uint32_data);
    }
}

template <typename T, template<typename> typename BufferType>
inline DeviceBuffer initialize_data_on_device(const BufferType<T>& data_to_write, Device* device, const Shape& shape, DataType data_type, Layout layout, const MemoryConfig& memory_config) {
    ZoneScoped;
    TT_ASSERT(device != nullptr);
    auto packed_size_in_bytes = packed_buffer_size_bytes<T>(data_to_write.size());
    auto device_buffer = allocate_buffer_on_device(packed_size_in_bytes, device, shape, data_type, layout, memory_config);
    write_data_to_device_buffer<T>(data_to_write, device_buffer, shape, data_type, layout, memory_config);
    return device_buffer;
}

template <typename T>
inline DeviceBuffer to_device_buffer(const Storage& storage, Device* device, const Shape& shape, DataType data_type, Layout layout, const MemoryConfig& memory_config) {
    return std::visit(
        [&device, &shape, &data_type, &layout, memory_config] (auto&& storage) -> DeviceBuffer {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                auto data_to_write = owned_buffer::get_as<T>(storage.buffer);
                TT_ASSERT(
                    compute_buffer_size(shape, data_type) == data_to_write.size(),
                    fmt::format("Tensor buffer size and number of data elements does not match: {} != {}", compute_buffer_size(shape, data_type), data_to_write.size())
                );
                if (layout == Layout::TILE) {
                    TT_ASSERT(
                        (shape[-2] % tt::constants::TILE_HEIGHT == 0 && shape[-1] % tt::constants::TILE_WIDTH == 0),
                        "Tensor shape incompatible for specified layout");
                }
                return initialize_data_on_device<T>(data_to_write, device, shape, data_type, layout, memory_config);
            }
            else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Device storage doesn't support to_device_buffer");
            }
            else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                if constexpr (std::is_same_v<T, float> or std::is_same_v<T, bfloat16> or std::is_same_v<T,uint32_t>) {
                    auto data_to_write = borrowed_buffer::get_as<T>(storage.buffer);
                    TT_ASSERT(
                        compute_buffer_size(shape, data_type) == data_to_write.size(),
                        fmt::format("Tensor buffer size and number of data elements does not match: {} != {}", compute_buffer_size(shape, data_type), data_to_write.size())
                    );
                    if (layout == Layout::TILE) {
                        TT_ASSERT(
                            (shape[-2] % tt::constants::TILE_HEIGHT == 0 && shape[-1] % tt::constants::TILE_WIDTH == 0),
                            "Tensor shape incompatible for specified layout");
                    }
                    return initialize_data_on_device<T>(data_to_write, device, shape, data_type, layout, memory_config);
                }
                else {
                    TT_THROW("Borrowed storage doesn't support this data type");
                }
            }
            else {
                raise_unsupported_storage<StorageType>();
            }
        },
        storage
    );
}

// ======================================================================================
//                                         .to()
// ======================================================================================
template <typename T>
inline Tensor to_host(const Tensor &tensor) {
    if (tensor.storage_type() != StorageType::DEVICE) {
        return tensor;
    }
    TT_ASSERT(tensor.is_allocated(), "Buffer must be allocated on device!");
    auto device_buffer = tensor.buffer();
    auto device = tensor.device();
    TT_ASSERT(device != nullptr && "Need device to be set copy data from device to host!");
    TT_ASSERT(!tensor.memory_config().is_sharded(), "Sharded tensors cannot be directly read from device");
    uint32_t size_in_bytes = device_buffer->size();
    auto data_vec = read_data_from_device<T>(tensor, size_in_bytes);
    auto output_buffer = owned_buffer::create<T>(std::move(data_vec));
    return Tensor(OwnedStorage{output_buffer}, tensor.shape(), tensor.dtype(), tensor.layout());
}

template <typename T>
inline Tensor to_device(const Tensor &tensor, Device *target_device, const MemoryConfig &memory_config) {
    TT_ASSERT(tensor.storage_type() != StorageType::DEVICE);
    if (tensor.storage_type() ==  StorageType::OWNED) {
        TT_ASSERT(tensor.is_allocated(), "Need host buffer on device to exist to copy data to device!");
    }
    TT_ASSERT(target_device != nullptr && "Need target device in order to move tensor to device!");
    TT_ASSERT(tensor.is_allocated() && "Need data to exist in order to move it to device");
    TT_ASSERT(!memory_config.is_sharded(), "Sharded tensors cannot be directly written from device");

    auto shape = tensor.shape();
    auto data_type = tensor.dtype();
    auto layout = tensor.layout();

    auto device_buffer = tensor_impl::to_device_buffer<T>(
        tensor.storage(), target_device, shape, data_type, layout, memory_config
    );
    return Tensor(DeviceStorage{device_buffer, target_device, memory_config}, shape, data_type, layout);
}

template <typename T>
inline Tensor to_layout(const Tensor &tensor, Layout target_layout) {
    if(tensor.layout() == target_layout) {
        return tensor;
    }

    auto shape = tensor.shape();
    auto source_layout = tensor.layout();
    auto convert = [&shape, source_layout, target_layout](const auto& input_data) -> std::vector<T> {
        switch (source_layout) {
            case Layout::ROW_MAJOR:
                if (target_layout == Layout::TILE) {
                    return convert_layout_row_major_to_tile(shape, input_data);
                }
                else {
                    TT_THROW("Unsupported layout conversion");
                }
                break;
            case Layout::TILE:
                if (target_layout == Layout::ROW_MAJOR) {
                    return convert_layout_tile_to_row_major(shape, input_data);
                }
                else {
                    TT_THROW("Unsupported layout conversion");
                }
                break;
            default:
                TT_THROW("Unsupported layout conversion");
        }
    };

    auto output_data = std::visit(
        [&convert] (auto&& storage) -> std::vector<T> {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                const auto input_data = owned_buffer::get_as<T>(storage.buffer);
                return convert(input_data);
            }
            else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                const auto input_data = borrowed_buffer::get_as<T>(storage.buffer);
                return convert(input_data);
            }
            else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            }
            else {
                raise_unsupported_storage<StorageType>();
            }
        },
        tensor.storage()
    );

    auto output_buffer = owned_buffer::create<T>(std::move(output_data));
    return Tensor(OwnedStorage{output_buffer}, tensor.shape(), tensor.dtype(), target_layout);
}

Tensor to_layout_bfloat8_b(const Tensor &tensor, Layout target_layout);

// ======================================================================================
//                                  .pad() and .unpad()
// ======================================================================================
template <typename T>
inline Tensor pad(const Tensor &tensor, const Shape& output_tensor_shape, const Shape& input_tensor_start, float pad_value) {
    auto pad_value_ = static_cast<T>(pad_value);
    const auto input_tensor_shape = tensor.shape();
    const auto input_tensor_strides = tensor.strides();
    const auto input_tensor_data_type = tensor.dtype();

    auto pad =
        [&input_tensor_shape, &input_tensor_strides, &input_tensor_data_type, &output_tensor_shape, &input_tensor_start, &pad_value_]
        (const auto& input_buffer) {
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
            {input_tensor_start[3], output_tensor_shape[3] - input_tensor_shape[3] - input_tensor_start[3]}
        };

        const std::array<uint32_t, 4> output_tensor_strides = {
            output_tensor_shape[1] * output_tensor_shape[2] * output_tensor_shape[3],
            output_tensor_shape[2] * output_tensor_shape[3],
            output_tensor_shape[3],
            1
        };

        auto output_buffer = owned_buffer::create<T>(compute_volume(output_tensor_shape));
        auto output_index = 0;
        for(auto i = 0; i < pad_size[0][0] * output_tensor_strides[0]; i++) {
            output_buffer[output_index++] = pad_value_;
        }
        for(auto dim0 = 0; dim0 < input_tensor_shape[0]; dim0++) {
            for(auto i = 0; i < pad_size[1][0] * output_tensor_strides[1]; i++) {
                output_buffer[output_index++] = pad_value_;
            }
            for(auto dim1 = 0; dim1 < input_tensor_shape[1]; dim1++) {
                for(auto i = 0; i < pad_size[2][0] * output_tensor_strides[2]; i++) {
                    output_buffer[output_index++] = pad_value_;
                }
                for(auto dim2 = 0; dim2 < input_tensor_shape[2]; dim2++) {
                    for(auto i = 0; i < pad_size[3][0] * output_tensor_strides[3]; i++) {
                        output_buffer[output_index++] = pad_value_;
                    }
                    for(auto dim3 = 0; dim3 < input_tensor_shape[3]; dim3++) {
                        auto input_index = dim3 + input_tensor_strides[2] * dim2 + input_tensor_strides[1] * dim1 + input_tensor_strides[0] * dim0;
                        output_buffer[output_index++] = input_buffer[input_index];
                    }
                    for(auto i = 0; i < pad_size[3][1] * output_tensor_strides[3]; i++) {
                        output_buffer[output_index++] = pad_value_;
                    }
                }
                for(auto i = 0; i < pad_size[2][1] * output_tensor_strides[2]; i++) {
                    output_buffer[output_index++] = pad_value_;
                }
            }
            for(auto i = 0; i < pad_size[1][1] * output_tensor_strides[1]; i++) {
                output_buffer[output_index++] = pad_value_;
            }
        }
        for(auto i = 0; i < pad_size[0][1] * output_tensor_strides[0]; i++) {
            output_buffer[output_index++] = pad_value_;
        }
        return output_buffer;
    };

    auto output_buffer = std::visit(
        [&pad] (auto&& storage) -> owned_buffer::Buffer<T> {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                const auto input_data = owned_buffer::get_as<T>(storage.buffer);
                return pad(input_data);
            }
            else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                const auto input_data = borrowed_buffer::get_as<T>(storage.buffer);
                return pad(input_data);
            }
            else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            }
            else {
                raise_unsupported_storage<StorageType>();
            }
        },
        tensor.storage()
    );
    return Tensor(OwnedStorage{output_buffer}, output_tensor_shape, tensor.dtype(), tensor.layout());
}

Tensor pad_bfloat8_b(const Tensor &tensor, const Shape& output_tensor_shape, const Shape& input_tensor_start, float pad_value);

template <typename T>
inline Tensor unpad(const Tensor &tensor, const Shape& output_tensor_start, const Shape& output_tensor_end) {

    auto input_buffer = owned_buffer::get_as<T>(tensor);

    // Check if tensor start and end indices are within input tensor shape
    const Shape& input_tensor_shape = tensor.shape();
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

    const Shape input_tensor_strides = tensor.strides();

    auto output_buffer = owned_buffer::create<T>(compute_volume(output_tensor_shape));
    auto output_index = 0;
    for(auto dim0 = output_tensor_start[0]; dim0 <= output_tensor_end[0]; dim0++) {
        for(auto dim1 = output_tensor_start[1]; dim1 <= output_tensor_end[1]; dim1++) {
            for(auto dim2 = output_tensor_start[2]; dim2 <= output_tensor_end[2]; dim2++) {
                for(auto dim3 = output_tensor_start[3]; dim3 <= output_tensor_end[3]; dim3++) {
                    auto input_index = dim3 + input_tensor_strides[2] * dim2 + input_tensor_strides[1] * dim1 + input_tensor_strides[0] * dim0;
                    output_buffer[output_index++] = input_buffer[input_index];
                }
            }
        }
    }

    return Tensor(OwnedStorage{output_buffer}, output_tensor_shape, tensor.dtype(), tensor.layout());
}


Tensor unpad_bfloat8_b(const Tensor &tensor, const Shape& output_tensor_start, const Shape& output_tensor_end);

// ======================================================================================
//                                         Print
// ======================================================================================
template <typename T>
inline std::string to_string(const Tensor &tensor, Layout print_layout, bool pretty_print = false) {

    const auto shape = tensor.shape();
    const auto dtype = tensor.dtype();
    const auto layout = tensor.layout();

    auto to_string_impl = [&print_layout, &pretty_print, &shape, &dtype, &layout](const auto& buffer) -> std::string {
        switch (layout) {
            case Layout::ROW_MAJOR:
                if (print_layout == Layout::ROW_MAJOR) {
                    return pretty_print ? detail::to_string_row_major(buffer, shape, dtype) : detail::to_string(buffer, dtype);
                } else if (print_layout == Layout::TILE) {
                    TT_ASSERT(pretty_print == false && "Can only pretty print in Row Major layout!");
                    auto converted_data = convert_layout_row_major_to_tile(shape, buffer);
                    return detail::to_string(converted_data, dtype);
                }
                else {
                    TT_THROW("Unsupported print layout");
                }
                break;
            case Layout::TILE:
                if (print_layout == Layout::ROW_MAJOR) {
                    auto converted_data = convert_layout_tile_to_row_major(shape, buffer);
                    return pretty_print ? detail::to_string_row_major(converted_data, shape, dtype) : detail::to_string(converted_data, dtype);
                } else if (print_layout == Layout::TILE) {
                    TT_ASSERT(pretty_print == false && "Can only pretty print in Row Major layout!");
                    return detail::to_string(buffer, dtype);
                } else {
                    TT_THROW("Unsupported print layout");
                }
                break;
            default:
                TT_THROW("Unsupported print layout");
        }
    };

    return std::visit(
        [&] (auto&& storage) -> std::string {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                const auto input_data = owned_buffer::get_as<T>(storage.buffer);
                return to_string_impl(input_data);
            }
            else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                const auto input_data = borrowed_buffer::get_as<T>(storage.buffer);
                return to_string_impl(input_data);
            }
            else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                return to_string<T>(to_host<T>(tensor), print_layout, pretty_print);
            }
            else {
                raise_unsupported_storage<StorageType>();
            }
        },
        tensor.storage()
    );
}

}  // namespace tensor_impl

}  // namespace tt_metal

}  // namespace tt
