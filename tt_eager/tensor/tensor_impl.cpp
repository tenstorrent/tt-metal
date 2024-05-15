// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/tensor_impl.hpp"
#include "tensor/tensor_impl_wrapper.hpp"

namespace tt {

namespace tt_metal {

namespace tensor_impl {

TensorPrintProfile TTNN_TENSOR_PRINT_PROFILE = TensorPrintProfile::Short;

std::ostream& operator<<(std::ostream& os, const DataType& dtype) {
    switch (dtype) {
        case DataType::BFLOAT8_B: os << "bfloat8_b"; break;
        case DataType::BFLOAT4_B: os << "bfloat4_b"; break;
        case DataType::BFLOAT16: os << "bfloat16"; break;
        case DataType::FLOAT32: os << "float32"; break;
        case DataType::UINT16: os << "uint16"; break;
        case DataType::UINT32: os << "uint32"; break;
        case DataType::INT32: os << "int32"; break;
        default: throw std::invalid_argument("Unknown data type");
    }
    return os;
}



uint32_t get_page_size(DataType dtype, Layout layout, uint32_t total_size_bytes, const Shape& shape) {
    uint32_t W = shape[-1];
    uint32_t page_size = 0;
    switch (layout) {
        case Layout::ROW_MAJOR: {
            uint32_t size_of_element = element_size_bytes_wrapper(dtype);
            page_size = W * size_of_element;
        }
        break;
        case Layout::TILE: {
            // TODO: Update to be generic for data type (issue 462)
            switch (dtype) {
                case DataType::BFLOAT16: {
                    // Float is converted to bfloat16 before being written to device
                    uint32_t size_of_element = element_size_bytes_wrapper(DataType::BFLOAT16);
                    page_size = constants::TILE_HW * size_of_element;
                }
                break;
                case DataType::FLOAT32: {
                    uint32_t size_of_element = element_size_bytes_wrapper(DataType::FLOAT32);
                    page_size = constants::TILE_HW * size_of_element;
                }
                break;
                case DataType::UINT32:
                case DataType::INT32:
                case DataType::UINT16: {
                    uint32_t size_of_element = element_size_bytes_wrapper(dtype);
                    page_size = constants::TILE_HW * size_of_element;
                }
                break;
                case DataType::BFLOAT4_B: {
                    page_size = constants::BFLOAT4_B_TILE_HW;
                }
                break;
                case DataType::BFLOAT8_B:  {
                    page_size = constants::BFLOAT8_B_TILE_HW;
                }
                break;
                default:
                    TT_ASSERT(false && "Unsupported data type!");
            }
            TT_ASSERT(total_size_bytes % page_size == 0);
        }
        break;
        default:
            TT_ASSERT(false && "Unsupported layout to write to device");
    }
    TT_ASSERT(page_size != 0);
    return page_size;
}



std::array<uint32_t, 2> get_sharded_page_shape(Layout layout,  DataType dtype, std::array<uint32_t, 2> shard_shape) {
    uint32_t page_size = 0;

    std::array<uint32_t, 2> page_shape = {constants::TILE_HEIGHT, constants::TILE_WIDTH};

    //Physical limitation in FD for now
    switch (layout) {
        case Layout::ROW_MAJOR: {
            //TODO: Explore valid page shapes other than 1,W
            page_shape = {1, shard_shape[1]};
        }
        break;
        case Layout::TILE: {;}
        break;
        default:
            TT_ASSERT(false && "Unsupported layout to write to device");
    }

    return page_shape;
}

void validate_sharded_buffer_allocation(const Shape& shape, Layout layout, std::optional<ShardSpecBuffer> shard_params, const MemoryConfig& memory_config) {
    TT_ASSERT(shard_params.has_value(), "Shard params are required for sharded buffer and they were not initialized");

    auto shard_spec = memory_config.shard_spec.value();
    auto& shard_shape = shard_spec.shape;

    uint32_t num_cores = shard_spec.num_cores();

    uint32_t total_height = tt_metal::compute_volume(shape) / shape[-1];
    uint32_t total_width = shape[-1];
    if (memory_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        TT_ASSERT(total_width == shard_shape[1], fmt::format("Shard shape {} does not divide tensor shape {} correctly according to sharding scheme", shard_shape[1], total_width));
        uint32_t num_shards = div_up(total_height, shard_shape[0]);
        TT_ASSERT(num_shards <= num_cores, fmt::format("Number of shards {} must match number of cores {}", num_shards, num_cores));
    } else if (memory_config.memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        TT_ASSERT(total_height == shard_shape[0], "Shard shape does not divide tensor shape correctly according to sharding scheme");
        uint32_t num_shards = div_up(total_width, shard_shape[1]);
        TT_ASSERT(num_shards <= num_cores, fmt::format("Number of shards {} must match number of cores {}", num_shards, num_cores));
    } else if (memory_config.memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        TT_ASSERT(shard_spec.grid.ranges().size() == 1, "Shard grid must be one full rectangular grid for block sharded!");
        uint32_t num_shards_along_height = div_up(total_height, shard_shape[0]);
        uint32_t num_shards_along_width = div_up(total_width, shard_shape[1]);

        // Additionally check that number of cores along height and width matches shard grid
        const CoreCoord shard_grid = shard_spec.grid.bounding_box().grid_size();
        if (shard_spec.orientation == ShardOrientation::ROW_MAJOR) {
            TT_ASSERT(num_shards_along_height <= shard_grid.y, fmt::format("Number of shards along height {} must match number of rows {} for row major orientation!", num_shards_along_height, shard_grid.y));
            TT_ASSERT(num_shards_along_width <= shard_grid.x, fmt::format("Number of shards along width {} must match number of columns {} for row major orientation!", num_shards_along_width, shard_grid.x));
        } else {
            TT_ASSERT(num_shards_along_height <= shard_grid.x, fmt::format("Number of shards along height {} must match number of columns {} for column major orientation!", num_shards_along_height, shard_grid.x));
            TT_ASSERT(num_shards_along_width <= shard_grid.y, fmt::format("Number of shards along width {} must match number of rows {} for column major orientation!", num_shards_along_width, shard_grid.y));
        }
    } else {
        TT_FATAL(false, "Unsupported sharding scheme");
    }

    if (layout == Layout::TILE) {
        TT_ASSERT((shard_shape[0] % constants::TILE_HEIGHT == 0 && shard_shape[1] % constants::TILE_WIDTH == 0), "Shard shape must be tile sized");
    } else if (layout == Layout::ROW_MAJOR) {
        // Require alignment for now
        // TT_ASSERT(shard_shape[1] * tensor_impl::element_size_bytes_wrapper(data_type) % ADDRESS_ALIGNMENT == 0);
    }
}

namespace detail {

DeviceBuffer allocate_interleaved_buffer_on_device(uint32_t buffer_size_bytes, Device *device, const Shape& shape, DataType data_type, Layout layout, const MemoryConfig& memory_config) {
    uint32_t page_size = get_page_size(data_type, layout, buffer_size_bytes, shape);
    return std::make_shared<Buffer>(device, buffer_size_bytes, page_size, memory_config.buffer_type);
}

DeviceBuffer allocate_contiguous_buffer_on_device(uint32_t buffer_size_bytes, Device *device, const MemoryConfig& memory_config) {
    return std::make_shared<Buffer>(device, buffer_size_bytes, buffer_size_bytes, memory_config.buffer_type);
}


DeviceBuffer allocate_sharded_buffer_on_device(uint32_t buffer_size_bytes, Device *device,
                                            const Shape& shape, DataType data_type, Layout layout,
                                            std::optional<ShardSpecBuffer> shard_params,
                                            const MemoryConfig& memory_config) {
    validate_sharded_buffer_allocation(shape, layout, shard_params, memory_config);
    auto page_shape = shard_params.value().page_shape;
    uint32_t size_of_element = element_size_bytes_wrapper(data_type);
    uint32_t page_size = page_shape[0] * page_shape[1] * size_of_element;
    if(layout == Layout::TILE){
        page_size = get_page_size(data_type, layout, buffer_size_bytes, shape);
    }

    return std::make_shared<Buffer>(device, buffer_size_bytes, page_size,
                                 memory_config.buffer_type,
                                 memory_config.memory_layout,
                                 shard_params);
}


}




DeviceBuffer allocate_buffer_on_device(uint32_t buffer_size_bytes, Device *device, const Shape& shape, DataType data_type, Layout layout, const MemoryConfig& memory_config, std::optional<ShardSpecBuffer> shard_spec) {
    if (memory_config.memory_layout == tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        return detail::allocate_interleaved_buffer_on_device(buffer_size_bytes, device, shape, data_type, layout, memory_config);
    }
    else if(memory_config.memory_layout == tt::tt_metal::TensorMemoryLayout::SINGLE_BANK){
        return detail::allocate_contiguous_buffer_on_device(buffer_size_bytes, device, memory_config);
    }
    else {
        TT_ASSERT( memory_config.is_sharded() && "Incorrect Memory Layout");
        return detail::allocate_sharded_buffer_on_device(buffer_size_bytes, device, shape, data_type, layout, shard_spec, memory_config);
    }
}

void validate_on_device_dtype_and_layout(Device *device, const Shape& shape, DataType dtype, Layout layout) {
    // TODO: Get supported layout and dtypes from device
    auto supported_dtype = [&dtype]() {
        TT_ASSERT(
            (
                dtype == DataType::UINT32 ||
                dtype == DataType::INT32 ||
                dtype == DataType::FLOAT32 ||
                dtype == DataType::UINT16 ||
                dtype == DataType::BFLOAT16 ||
                dtype == DataType::BFLOAT8_B ||
                dtype == DataType::BFLOAT4_B
            ),
            "Only UINT32, INT32, FLOAT32, UINT16, BFLOAT16, BFLOAT8_B, or BFLOAT4_B dtypes are supported on device!"
        );
    };
    auto supported_layout = [&shape, &dtype, &layout]() {
        switch (dtype) {
            case DataType::UINT32:
            case DataType::INT32:
            case DataType::FLOAT32:
                break;
            case DataType::UINT16:
            case DataType::BFLOAT16:
                if (layout == Layout::ROW_MAJOR) {
                    TT_ASSERT(shape[-1] % 2 == 0, "For ROW_MAJOR layout tensors with dtype BFLOAT16 or UINT16, tensor width must be divisible by 2 since data is packed as uint32_t when creating buffers on device!");
                }
                break;
            case DataType::BFLOAT8_B:
            case DataType::BFLOAT4_B:
                TT_ASSERT(layout == Layout::TILE, "Only TILE layout is supported for BFLOAT8_B dtype!");
                break;
            default:
                TT_ASSERT(false, "Only UINT32, INT32, FLOAT32, UINT16, BFLOAT16, BFLOAT8_B, or BFLOAT4_B dtypes are supported on device!");
                break;
            }
    };
    supported_dtype();
    supported_layout();
}

Tensor pad_bfloat8_b(const Tensor &tensor, const Shape& output_tensor_shape, const Shape& input_tensor_start, float pad_value) {
    // TODO(arakhmati): do not convert to FLOAT32

    // Convert to FLOAT32 tensor and pad
    auto input_packed_data = owned_buffer::get_as<uint32_t>(tensor).get();
    auto input_float_data = unpack_bfp8_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false);
    auto input_float_buffer = owned_buffer::create<float>(std::move(input_float_data));
    auto float_tensor = Tensor(OwnedStorage{input_float_buffer}, tensor.get_legacy_shape(), DataType::FLOAT32, tensor.get_layout()).pad(output_tensor_shape, input_tensor_start, pad_value);

    // Convert back to BFLOAT8_B
    auto output_float_data = owned_buffer::get_as<float>(float_tensor).get();
    auto output_packed_data = pack_fp32_vec_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
    auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
    return Tensor(std::move(OwnedStorage{std::move(output_uint32_buffer)}), float_tensor.get_legacy_shape(), DataType::BFLOAT8_B, tensor.get_layout());
}

Tensor unpad_bfloat8_b(const Tensor &tensor, const Shape& output_tensor_start, const Shape& output_tensor_end) {
    // TODO(arakhmati): do not convert to FLOAT32

    // Convert to FLOAT32 tensor and unpad
    auto input_packed_data = owned_buffer::get_as<uint32_t>(tensor).get();
    auto input_float_data = unpack_bfp8_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false);
    auto input_float_buffer = owned_buffer::create<float>(std::move(input_float_data));
    auto float_tensor = Tensor(OwnedStorage{input_float_buffer}, tensor.get_legacy_shape(), DataType::FLOAT32, tensor.get_layout()).unpad(output_tensor_start, output_tensor_end);

    // Convert back to BFLOAT8_B
    auto output_float_data = owned_buffer::get_as<float>(float_tensor).get();
    auto output_packed_data = pack_fp32_vec_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
    auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
    return Tensor(std::move(OwnedStorage{std::move(output_uint32_buffer)}), float_tensor.get_legacy_shape(), DataType::BFLOAT8_B, tensor.get_layout());
}

Tensor pad_bfloat4_b(const Tensor &tensor, const Shape& output_tensor_shape, const Shape& input_tensor_start, float pad_value) {
    // TODO(arakhmati): do not convert to FLOAT32

    // Convert to FLOAT32 tensor and pad
    auto input_packed_data = owned_buffer::get_as<uint32_t>(tensor).get();
    auto input_float_data = unpack_bfp4_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false);
    auto input_float_buffer = owned_buffer::create<float>(std::move(input_float_data));
    auto float_tensor = Tensor(OwnedStorage{input_float_buffer}, tensor.get_legacy_shape(), DataType::FLOAT32, tensor.get_layout()).pad(output_tensor_shape, input_tensor_start, pad_value);

    // Convert back to BFLOAT4_B
    auto output_float_data = owned_buffer::get_as<float>(float_tensor).get();
    auto output_packed_data = pack_fp32_vec_as_bfp4_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
    auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
    return Tensor(std::move(OwnedStorage{std::move(output_uint32_buffer)}), float_tensor.get_legacy_shape(), DataType::BFLOAT4_B, tensor.get_layout());
}

Tensor unpad_bfloat4_b(const Tensor &tensor, const Shape& output_tensor_start, const Shape& output_tensor_end) {
    // TODO(arakhmati): do not convert to FLOAT32

    // Convert to FLOAT32 tensor and unpad
    auto input_packed_data = owned_buffer::get_as<uint32_t>(tensor).get();
    auto input_float_data = unpack_bfp4_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false);
    auto input_float_buffer = owned_buffer::create<float>(std::move(input_float_data));
    auto float_tensor = Tensor(OwnedStorage{input_float_buffer}, tensor.get_legacy_shape(), DataType::FLOAT32, tensor.get_layout()).unpad(output_tensor_start, output_tensor_end);

    // Convert back to BFLOAT4_B
    auto output_float_data = owned_buffer::get_as<float>(float_tensor).get();
    auto output_packed_data = pack_fp32_vec_as_bfp4_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
    auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
    return Tensor(std::move(OwnedStorage{std::move(output_uint32_buffer)}), float_tensor.get_legacy_shape(), DataType::BFLOAT4_B, tensor.get_layout());
}


}  // namespace tensor_impl

}  // namespace tt_metal

}  // namespace tt
