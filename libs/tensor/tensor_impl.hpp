#pragma once

#include "tensor/types.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_utils.hpp"
#include "tensor/host_buffer.hpp"
#include "tt_metal/host_api.hpp"

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
template <typename T1, typename T2>
inline std::vector<T1> cast_vec(const span_t<T2>& data_to_convert) {
    std::vector<T1> converted_data;
    for (auto datum : data_to_convert) {
        converted_data.push_back(static_cast<T1>(datum));
    }
    return converted_data;
}

template <>
inline std::vector<float> cast_vec(const span_t<bfloat16>& data_to_convert) {
    std::vector<float> converted_data;
    for (auto datum : data_to_convert) {
        converted_data.push_back(datum.to_float());
    }
    return converted_data;
}

template <>
inline std::vector<uint32_t> cast_vec(const span_t<bfloat16>& data_to_convert) {
    std::vector<uint32_t> converted_data;
    for (auto datum : data_to_convert) {
        converted_data.push_back((uint32_t)datum.to_uint16());
    }
    return converted_data;
}

// TODO(arakhmati): Should pack_vec_into_uint32_vec be a generator?
template <typename T>
constexpr inline std::vector<uint32_t> pack_vec_into_uint32_vec(const span_t<T>& data_to_pack) {
    TT_THROW("Don't know how to pack data into uint32 vector generically!");
}

template <>
inline std::vector<uint32_t> pack_vec_into_uint32_vec(const span_t<uint32_t>& data_to_pack) {
    return std::vector(std::begin(data_to_pack), std::end(data_to_pack));
}

template <>
inline std::vector<uint32_t> pack_vec_into_uint32_vec(const span_t<bfloat16>& data_to_pack) {
    auto bfloat16_vec = std::vector(std::begin(data_to_pack), std::end(data_to_pack));
    return pack_bfloat16_vec_into_uint32_vec(bfloat16_vec);
}

template <>
inline std::vector<uint32_t> pack_vec_into_uint32_vec(const span_t<float>& data_to_pack) {
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
}

template <typename T>
constexpr inline std::vector<T> unpack_uint32_vec(std::vector<uint32_t> &data_to_unpack) {
    std::vector<uint32_t> unpacked_data;
    TT_ASSERT(false && "Don't know how to unpack uint32 data generically!");
    return unpacked_data;
}

template <>
inline std::vector<uint32_t> unpack_uint32_vec(std::vector<uint32_t> &data_to_unpack) {
    return data_to_unpack;
}

template <>
inline std::vector<bfloat16> unpack_uint32_vec(std::vector<uint32_t> &data_to_unpack) {
    return unpack_uint32_vec_into_bfloat16_vec(data_to_unpack);
}

template <>
inline std::vector<float> unpack_uint32_vec(std::vector<uint32_t> &data_to_unpack) {
    std::vector<float> float_data;
    for (auto i = 0; i < data_to_unpack.size(); i++) {
        auto unpacked = unpack_two_bfloat16_from_uint32(data_to_unpack[i]);
        auto float_val1 = unpacked.first.to_float();
        auto float_val2 = unpacked.second.to_float();
        float_data.push_back(float_val1);
        float_data.push_back(float_val2);
    }
    return float_data;
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

// Specialization for bfloat8_b because tile size is 1088 after being packed
template <>
constexpr inline uint32_t packed_buffer_size_bytes<bfloat8_b>(uint32_t volume_unpacked_data) {
    auto num_tiles = volume_unpacked_data / (32 * 32);
    return num_tiles * 1088; // TODO: Update to get size from tt_metal::GetTileSize (issue 462)
}

// ======================================================================================
//                                  Layout converters
// ======================================================================================
template <typename T>
inline std::vector<T> convert_layout_row_major_to_tile(const std::array<uint32_t, 4> &shape, const std::vector<T>& data_to_convert) {
    TT_ASSERT((shape[2] % tt::constants::TILE_HEIGHT == 0 && shape[3] % tt::constants::TILE_WIDTH == 0), "Unsupported shape for tensor conversion");
    std::vector<uint32_t> shape_vec = {shape[0], shape[1], shape[2], shape[3]};
    return convert_layout(data_to_convert, shape_vec, TensorLayout::LIN_ROW_MAJOR, TensorLayout::TILED32_4FACES);
}

template <typename T>
inline std::vector<T> convert_layout_tile_to_row_major(const std::array<uint32_t, 4> &shape, const std::vector<T>& data_to_convert) {
    std::vector<uint32_t> shape_vec = {shape[0], shape[1], shape[2], shape[3]};
    return convert_layout(data_to_convert, shape_vec, TensorLayout::TILED32_4FACES, TensorLayout::LIN_ROW_MAJOR);
}

template <typename T>
inline std::vector<T> convert_layout_row_major_to_channels_last(const std::array<uint32_t, 4> &shape, const std::vector<T>& data_to_convert) {
    std::vector<uint32_t> shape_vec = {shape[0], shape[1], shape[2], shape[3]};
    return convert_layout(data_to_convert, shape_vec, TensorLayout::LIN_ROW_MAJOR, TensorLayout::CHANNELS_LAST);
}

template <typename T>
inline std::vector<T> convert_layout_channels_last_to_row_major(const std::array<uint32_t, 4> &shape, const std::vector<T>& data_to_convert) {
    std::vector<uint32_t> shape_vec = {shape[0], shape[1], shape[2], shape[3]};
    return convert_layout(data_to_convert, shape_vec, TensorLayout::CHANNELS_LAST, TensorLayout::LIN_ROW_MAJOR);
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
void print_row_major_data(const std::vector<T> &data, std::array<uint32_t, 4> shape, DataType dtype) {
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

// ======================================================================================
//                           Data reader, writer, and initializers
// ======================================================================================
std::tuple<int, int, int> get_interleaved_read_write_unit_metadata(DataType dtype, Layout layout, uint32_t total_size_bytes, const std::array<uint32_t, 4>& shape);

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
    auto device_buffer = tensor.device_storage().value().buffer;
    TT_ASSERT(device_buffer->size() == size_in_bytes);

    std::vector<uint32_t> device_data;
    const char *TT_METAL_DEVICE_DISPATCH_MODE = std::getenv("TT_METAL_DEVICE_DISPATCH_MODE");
    if (TT_METAL_DEVICE_DISPATCH_MODE != nullptr) {
        EnqueueReadBuffer(*HACK_CQ, *device_buffer, device_data, true);
    } else {
        ReadFromBuffer(*device_buffer, device_data);
    }

    std::vector<T> unpacked_data;
    if (tensor.dtype() == DataType::BFLOAT8_B) {
        std::vector<float> float_unpacked_data = unpack_bfp8_tiles_into_float_vec(device_data, /*row_major_output=*/false, /*is_exp_a=*/false);
        auto float_unpacked_data_view = span_t(float_unpacked_data);
        unpacked_data = cast_vec<T>(float_unpacked_data_view);
    } else {
        unpacked_data = unpack_uint32_vec<T>(device_data);
    }
    return unpacked_data;
}

template <typename T>
inline void write_data_to_device_buffer(const span_t<T>& data_to_write, DeviceBuffer buffer, const Shape& shape, DataType data_type, Layout layout, const MemoryConfig& memory_config) {
    // TODO(arakhmati): can we use generators in this function to go from `data_to_write` to `uint32_data`?
    // And effectively get rid of any additional allocation

    std::vector<uint32_t> uint32_data;
    if (data_type == DataType::BFLOAT8_B) {
        std::vector<float> float_data = cast_vec<float>(data_to_write);
        uint32_data = pack_fp32_vec_as_bfp8_tiles(float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
    } else if (data_type == DataType::BFLOAT16) {
        if (memory_config.interleaved) {
            if (layout == Layout::ROW_MAJOR) {
                TT_ASSERT(shape[3] % 2 == 0, "Input tensor width must be a multiple of 2 to pack interleaved row major data");
            } else if (layout == Layout::CHANNELS_LAST) {
                TT_ASSERT(shape[1] % 2 == 0, "Input tensor channel must be a multiple of 2 to pack interleaved channels last data");
            }
        } else {
            TT_ASSERT(volume(shape) % 2 == 0, "Input tensor volume must be a multiple of 2 to pack contiguous data");
        }
        uint32_data = pack_vec_into_uint32_vec<T>(data_to_write);
    } else {
        uint32_data = pack_vec_into_uint32_vec<T>(data_to_write);
    }

    const char *TT_METAL_DEVICE_DISPATCH_MODE = std::getenv("TT_METAL_DEVICE_DISPATCH_MODE");
    if (TT_METAL_DEVICE_DISPATCH_MODE != nullptr) {
        EnqueueWriteBuffer(*HACK_CQ, *buffer, uint32_data, false);
    } else {
        WriteToBuffer(*buffer, uint32_data);
    }
}

template <typename T>
inline DeviceBuffer initialize_data_on_device(const span_t<T>& data_to_write, Device* device, const Shape& shape, DataType data_type, Layout layout, const MemoryConfig& memory_config) {
    TT_ASSERT(device != nullptr);
    uint32_t packed_size_in_bytes;
    if (data_type == DataType::BFLOAT8_B) {
        packed_size_in_bytes = packed_buffer_size_bytes<bfloat8_b>(data_to_write.size());
    } else {
        packed_size_in_bytes = packed_buffer_size_bytes<T>(data_to_write.size());
    }
    auto device_buffer = allocate_buffer_on_device(packed_size_in_bytes, device, shape, data_type, layout, memory_config);
    write_data_to_device_buffer<T>(data_to_write, device_buffer, shape, data_type, layout, memory_config);
    return device_buffer;
}

template <typename T>
inline DeviceBuffer device_buffer_from_host_buffer(const HostBuffer& host_buffer, Device* device, const Shape& shape, DataType data_type, Layout layout, const MemoryConfig& memory_config) {
    auto data_to_write = host_buffer::view_as<T>(host_buffer);
    TT_ASSERT(volume(shape) == data_to_write.size(), "Tensor shape and number of data elements does not match");
    if (layout == Layout::TILE) {
        TT_ASSERT((shape[2] % tt::constants::TILE_HEIGHT == 0 && shape[3] % tt::constants::TILE_WIDTH == 0), "Tensor shape incompatible for specified layout");
    }
    return initialize_data_on_device<T>(data_to_write, device, shape, data_type, layout, memory_config);
}

// ======================================================================================
//                                         .to()
// ======================================================================================
template <typename T>
inline Tensor to_host(const Tensor &tensor) {
    auto device_storage = tensor.device_storage().value();
    auto device_buffer = device_storage.buffer;
    auto device = device_storage.device;
    TT_ASSERT(tensor.storage_type() == StorageType::DEVICE and tensor.is_allocated(), "Need DRAM buffers on device to exist to copy data to host!");
    TT_ASSERT(device != nullptr && "Need device to be set copy data from device to host!");
    uint32_t size_in_bytes = device_buffer->size();
    auto data_vec = read_data_from_device<T>(tensor, size_in_bytes);

    // TODO(arakhmati): remove copying
    auto output_buffer = host_buffer::create<T>(data_vec.size());
    auto output_view = host_buffer::view_as<T>(output_buffer);
    for (auto index = 0; index < data_vec.size(); index++) {
        output_view[index] = data_vec[index];
    }

    return Tensor(HostStorage{output_buffer}, tensor.shape(), tensor.dtype(), tensor.layout());
}

template <typename T>
inline Tensor to_device(const Tensor &tensor, Device *target_device, const MemoryConfig &memory_config) {
    auto host_storage = tensor.host_storage().value();
    TT_ASSERT(target_device != nullptr && "Need target device in order to move tensor to device!");
    TT_ASSERT(tensor.is_allocated() && "Need data to exist in order to move it to device");

    auto shape = tensor.shape();
    auto data_type = tensor.dtype();
    auto layout = tensor.layout();

    auto device_buffer = tensor_impl::device_buffer_from_host_buffer<T>(
        host_storage.buffer, target_device, shape, data_type, layout, memory_config
    );
    return Tensor(DeviceStorage{device_buffer, target_device, memory_config}, shape, data_type, layout);
}

template <typename T>
inline Tensor to_layout(const Tensor &tensor, Layout target_layout) {
    if(tensor.layout() == target_layout) {
        return tensor;
    }

    // TODO(arakhmati): remove copying
    auto tensor_view = host_buffer::view_as<T>(tensor);
    auto data = std::vector<T>(tensor_view.begin(), tensor_view.end());

    switch (tensor.layout()) {
        case Layout::ROW_MAJOR:
            if (target_layout == Layout::TILE) {
                data = convert_layout_row_major_to_tile(tensor.shape(), data);
            }
            else if (target_layout == Layout::CHANNELS_LAST) {
                data = convert_layout_row_major_to_channels_last(tensor.shape(), data);
            }
            else {
                TT_ASSERT(false && "Unsupported layout conversion");
            }
        break;
        case Layout::TILE:
            if (target_layout == Layout::ROW_MAJOR) {
                data = convert_layout_tile_to_row_major(tensor.shape(), data);
            }
            else {
                TT_ASSERT(false && "Unsupported layout conversion");
            }
        break;
        case Layout::CHANNELS_LAST:
            if (target_layout == Layout::ROW_MAJOR) {
                data = convert_layout_channels_last_to_row_major(tensor.shape(), data);
            }
            else {
                TT_ASSERT(false && "Unsupported layout conversion");
            }
        break;
        default:
            TT_ASSERT(false && "Unsupported layout conversion");
    }

    // TODO(arakhmati): remove copying
    auto output_buffer = host_buffer::create<T>(data.size());
    auto output_view = host_buffer::view_as<T>(output_buffer);
    for (auto index = 0; index < data.size(); index++) {
        output_view[index] = data[index];
    }
    return Tensor(HostStorage{output_buffer}, tensor.shape(), tensor.dtype(), target_layout);
}

// ======================================================================================
//                                  .pad() and .unpad()
// ======================================================================================
template <typename T>
inline Tensor pad(const Tensor &tensor, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value) {
    auto pad_value_ = static_cast<T>(pad_value);

    auto tensor_view = host_buffer::view_as<T>(tensor);

    // Check if input tensor fits in output tensor given the input tensor start indices
    const std::array<uint32_t, 4> input_tensor_shape = tensor.shape();
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

    const std::array<uint32_t, 4> input_tensor_strides = tensor.strides();
    const std::array<uint32_t, 4> output_tensor_strides = {
        output_tensor_shape[1] * output_tensor_shape[2] * output_tensor_shape[3],
        output_tensor_shape[2] * output_tensor_shape[3],
        output_tensor_shape[3],
        1
    };

    auto output_buffer = host_buffer::create<T>(volume(output_tensor_shape));
    auto output_view = host_buffer::view_as<T>(output_buffer);
    auto output_index = 0;
    for(auto i = 0; i < pad_size[0][0] * output_tensor_strides[0]; i++) {
        output_view[output_index++] = pad_value_;
    }
    for(auto dim0 = 0; dim0 < input_tensor_shape[0]; dim0++) {
        for(auto i = 0; i < pad_size[1][0] * output_tensor_strides[1]; i++) {
            output_view[output_index++] = pad_value_;
        }
        for(auto dim1 = 0; dim1 < input_tensor_shape[1]; dim1++) {
            for(auto i = 0; i < pad_size[2][0] * output_tensor_strides[2]; i++) {
                output_view[output_index++] = pad_value_;
            }
            for(auto dim2 = 0; dim2 < input_tensor_shape[2]; dim2++) {
                for(auto i = 0; i < pad_size[3][0] * output_tensor_strides[3]; i++) {
                    output_view[output_index++] = pad_value_;
                }
                for(auto dim3 = 0; dim3 < input_tensor_shape[3]; dim3++) {
                    auto input_index = dim3 + input_tensor_strides[2] * dim2 + input_tensor_strides[1] * dim1 + input_tensor_strides[0] * dim0;
                    output_view[output_index++] = tensor_view[input_index];
                }
                for(auto i = 0; i < pad_size[3][1] * output_tensor_strides[3]; i++) {
                    output_view[output_index++] = pad_value_;
                }
            }
            for(auto i = 0; i < pad_size[2][1] * output_tensor_strides[2]; i++) {
                output_view[output_index++] = pad_value_;
            }
        }
        for(auto i = 0; i < pad_size[1][1] * output_tensor_strides[1]; i++) {
            output_view[output_index++] = pad_value_;
        }
    }
    for(auto i = 0; i < pad_size[0][1] * output_tensor_strides[0]; i++) {
        output_view[output_index++] = pad_value_;
    }

    return Tensor(HostStorage{output_buffer}, output_tensor_shape, tensor.dtype(), tensor.layout());
}

template <typename T>
inline Tensor unpad(const Tensor &tensor, const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end) {

    auto tensor_view = host_buffer::view_as<T>(tensor);

    // Check if tensor start and end indices are within input tensor shape
    const std::array<uint32_t, 4> input_tensor_shape = tensor.shape();
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
    const std::array<uint32_t, 4> output_tensor_shape = {
        output_tensor_end[0] - output_tensor_start[0] + 1,
        output_tensor_end[1] - output_tensor_start[1] + 1,
        output_tensor_end[2] - output_tensor_start[2] + 1,
        output_tensor_end[3] - output_tensor_start[3] + 1,
    };

    const std::array<uint32_t, 4> input_tensor_strides = tensor.strides();

    auto output_buffer = host_buffer::create<T>(volume(output_tensor_shape));
    auto output_view = host_buffer::view_as<T>(output_buffer);
    auto output_index = 0;
    for(auto dim0 = output_tensor_start[0]; dim0 <= output_tensor_end[0]; dim0++) {
        for(auto dim1 = output_tensor_start[1]; dim1 <= output_tensor_end[1]; dim1++) {
            for(auto dim2 = output_tensor_start[2]; dim2 <= output_tensor_end[2]; dim2++) {
                for(auto dim3 = output_tensor_start[3]; dim3 <= output_tensor_end[3]; dim3++) {
                    auto input_index = dim3 + input_tensor_strides[2] * dim2 + input_tensor_strides[1] * dim1 + input_tensor_strides[0] * dim0;
                    output_view[output_index++] = tensor_view[input_index];
                }
            }
        }
    }

    return Tensor(HostStorage{output_buffer}, output_tensor_shape, tensor.dtype(), tensor.layout());
}

// ======================================================================================
//                                         Print
// ======================================================================================
template <typename T>
inline void print(const Tensor &tensor, Layout print_layout, bool pretty_print) {
    if (tensor.storage_type() == StorageType::DEVICE ) {
        auto temp_tensor = to_host<T>(tensor);
        print<T>(temp_tensor, print_layout, pretty_print);
        return;
    }

    // TODO(arakhmati): remove copying
    auto tensor_view = host_buffer::view_as<T>(tensor);
    auto data_vec = std::vector<T>(tensor_view.begin(), tensor_view.end());

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
                TT_ASSERT(false && "Unsupported print layout");
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
                TT_ASSERT(false && "Unsupported print layout");
            }
        break;
        case Layout::CHANNELS_LAST:
            if (print_layout == Layout::ROW_MAJOR) {
                auto converted_data = convert_layout_channels_last_to_row_major(tensor.shape(), data_vec);
                pretty_print ? print_row_major_data(converted_data, tensor.shape(), tensor.dtype()) : print_data(converted_data, tensor.dtype());
            }
            else if (print_layout == Layout::CHANNELS_LAST) {
                auto cl_shape = tensor.shape();
                cl_shape[3] = tensor.shape()[1];
                cl_shape[2] = tensor.shape()[3];
                cl_shape[1] = tensor.shape()[2];
                pretty_print ? print_row_major_data(data_vec, cl_shape, tensor.dtype()) : print_data(data_vec, tensor.dtype());
            }
            else {
                TT_ASSERT(false && "Unsupported print layout");
            }
        break;
        default:
            TT_ASSERT(false && "Unsupported print layout");
    }
}

}  // namespace tensor_impl

}  // namespace tt_metal

}  // namespace tt
