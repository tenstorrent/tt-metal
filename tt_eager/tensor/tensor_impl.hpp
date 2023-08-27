#pragma once

#include "tensor/types.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_utils.hpp"
#include "tensor/borrowed_buffer_functions.hpp"
#include "tensor/owned_buffer_functions.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
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

namespace detail{
template<class>
inline constexpr bool always_false_v = false;
}

// TODO(arakhmati): Should pack_vec_into_uint32_vec be a generator?
template <typename DataType, template<typename> typename BufferType>
constexpr inline std::vector<uint32_t> pack_vec_into_uint32_vec(const BufferType<DataType>& data_to_pack) {
    if constexpr (std::is_same_v<DataType, uint32_t>) {
        return std::vector(std::begin(data_to_pack), std::end(data_to_pack));
    }
    else if constexpr (std::is_same_v<DataType, bfloat16>) {
        auto bfloat16_vec = std::vector(std::begin(data_to_pack), std::end(data_to_pack));
        return pack_bfloat16_vec_into_uint32_vec(bfloat16_vec);
    }
    else if constexpr (std::is_same_v<DataType, float>) {
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
        static_assert(detail::always_false_v<DataType>, "Don't know how to unpack uint32 data generically!");
    }
}

template <typename DataType>
constexpr inline std::vector<DataType> unpack_uint32_vec(std::vector<uint32_t> &data_to_unpack) {
    if constexpr (std::is_same_v<DataType, uint32_t>) {
        return data_to_unpack;
    }
    else if constexpr (std::is_same_v<DataType, bfloat16>) {
        return unpack_uint32_vec_into_bfloat16_vec(data_to_unpack);
    }
    else if constexpr (std::is_same_v<DataType, float>) {
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
        static_assert(detail::always_false_v<DataType>, "Don't know how to unpack uint32 data generically!");
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
template <typename T, template<typename> typename BufferType>
inline std::vector<T> convert_layout_row_major_to_tile(const Shape& shape, const BufferType<T>& data_to_convert) {
    TT_ASSERT((shape[2] % tt::constants::TILE_HEIGHT == 0 && shape[3] % tt::constants::TILE_WIDTH == 0), "Unsupported shape for tensor conversion");
    std::vector<uint32_t> shape_vec = {shape[0], shape[1], shape[2], shape[3]};
    return convert_layout(data_to_convert, shape_vec, TensorLayout::LIN_ROW_MAJOR, TensorLayout::TILED32_4FACES);
}

template <typename T, template<typename> typename BufferType>
inline std::vector<T> convert_layout_tile_to_row_major(const Shape& shape, const BufferType<T>& data_to_convert) {
    std::vector<uint32_t> shape_vec = {shape[0], shape[1], shape[2], shape[3]};
    return convert_layout(data_to_convert, shape_vec, TensorLayout::TILED32_4FACES, TensorLayout::LIN_ROW_MAJOR);
}

// ======================================================================================
//                                         Print
// ======================================================================================
std::ostream& operator<<(std::ostream& os, const DataType& dtype);

template <typename T>
inline void print_datum(T datum) {
    std::cout << datum;
}

template <>
inline void print_datum(bfloat16 datum) {
    std::cout << datum.to_float();
}

template <typename T>
void print_data(const std::vector<T> &data, DataType dtype) {
    std::cout << "[ ";
    for (int i = 0; i < data.size(); i++) {
        print_datum(data[i]);
        if (i < data.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << " dtype=" <<  dtype << " ]" << std::endl;
}

template <typename T>
void print_row_major_data(const std::vector<T> &data, const Shape& shape, DataType dtype) {
    std::cout << "[ ";
    for(auto w = 0; w < shape[0]; w++) {
        if(w == 0)
            std::cout << "[";
        else
            std::cout << "  [";
        for(auto z = 0; z < shape[1]; z++) {
            if (z == 0)
                std::cout << "[";
            else
                std::cout << "   [";
            for(auto y = 0; y < shape[2]; y++) {
                if (y == 0)
                    std::cout << "[";
                else
                    std::cout << "    [";
                for(auto x = 0; x < shape[3]; x++) {
                    // data in row major order
                    auto index = x + y*shape[3] + z*shape[2]*shape[3] + w*shape[1]*shape[2]*shape[3];
                    print_datum(data[index]);
                    if (x < shape[3] - 1) {
                        std::cout << ", ";
                    }
                }
                if(y < shape[2] - 1)
                    std::cout << "]," << std::endl;
                else
                    std::cout << "]";
            }
            if(z < shape[1] - 1)
                std::cout << "]," << std::endl << std::endl;
            else
                std::cout << "]";
        }
        if(w < shape[0] - 1)
            std::cout << "]," << std::endl << std::endl << std::endl;
        else
            std::cout << "]";
    }
    std::cout << " dtype=" <<  dtype << " ]" << std::endl;
}


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

    std::vector<uint32_t> device_data;
    const char *TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        EnqueueReadBuffer(*tt::tt_metal::detail::GLOBAL_CQ, *device_buffer, device_data, true);
    } else {
        ReadFromBuffer(*device_buffer, device_data);
    }

    return unpack_uint32_vec<T>(device_data);
}

template <typename T, template<typename> typename BufferType>
inline void write_data_to_device_buffer(const BufferType<T>& data_to_write, DeviceBuffer buffer, const Shape& shape, DataType data_type, Layout layout, const MemoryConfig& memory_config) {
    ZoneScoped;
    // TODO(arakhmati): can we use generators in this function to go from `data_to_write` to `uint32_data`?
    // And effectively get rid of any additional allocation

    if (data_type == DataType::BFLOAT16) {
        if (memory_config.interleaved) {
            TT_ASSERT(shape[3] % 2 == 0, "Input tensor width must be a multiple of 2 to pack interleaved row major data");
        } else {
            TT_ASSERT(compute_volume(shape) % 2 == 0, "Input tensor volume must be a multiple of 2 to pack contiguous data");
        }
    }
    auto uint32_data = pack_vec_into_uint32_vec<T>(data_to_write);

    const char *TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        EnqueueWriteBuffer(*tt::tt_metal::detail::GLOBAL_CQ, *buffer, uint32_data, false);
    } else {
        WriteToBuffer(*buffer, uint32_data);
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
                    TT_ASSERT((shape[2] % tt::constants::TILE_HEIGHT == 0 && shape[3] % tt::constants::TILE_WIDTH == 0), "Tensor shape incompatible for specified layout");
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
                        TT_ASSERT((shape[2] % tt::constants::TILE_HEIGHT == 0 && shape[3] % tt::constants::TILE_WIDTH == 0), "Tensor shape incompatible for specified layout");
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
    TT_ASSERT(tensor.is_allocated(), "Need DRAM buffers to be allocated!");
    auto device_buffer = tensor.buffer();
    auto device = tensor.device();
    TT_ASSERT(device != nullptr && "Need device to be set copy data from device to host!");
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
    auto convert = [&shape, source_layout, target_layout] (const auto& input_data) -> std::vector<T> {
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
inline void print(const Tensor &tensor, Layout print_layout, bool pretty_print) {
    if (tensor.storage_type() == StorageType::DEVICE) {
        print<T>(to_host<T>(tensor), print_layout, pretty_print);
        return;
    }

    auto data_vec = owned_buffer::get_as<T>(tensor).get();

    switch (tensor.layout()) {
        case Layout::ROW_MAJOR:
            if (print_layout == Layout::ROW_MAJOR) {
                pretty_print ? print_row_major_data(data_vec, tensor.shape(), tensor.dtype()) : print_data(data_vec, tensor.dtype());
            } else if (print_layout == Layout::TILE) {
                TT_ASSERT(pretty_print == false && "Can only pretty print in Row Major layout!");
                auto converted_data = convert_layout_row_major_to_tile(tensor.shape(), data_vec);
                print_data(converted_data, tensor.dtype());
            }
            else {
                TT_THROW("Unsupported print layout");
            }
            break;
        case Layout::TILE:
            if (print_layout == Layout::ROW_MAJOR) {
                auto converted_data = convert_layout_tile_to_row_major(tensor.shape(), data_vec);
                pretty_print ? print_row_major_data(converted_data, tensor.shape(), tensor.dtype()) : print_data(converted_data, tensor.dtype());
            } else if (print_layout == Layout::TILE) {
                TT_ASSERT(pretty_print == false && "Can only pretty print in Row Major layout!");
                print_data(data_vec, tensor.dtype());
            } else {
                TT_THROW("Unsupported print layout");
            }
            break;
        default:
            TT_THROW("Unsupported print layout");
    }
}

}  // namespace tensor_impl

}  // namespace tt_metal

}  // namespace tt
