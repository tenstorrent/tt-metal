#include "tensor/tensor_impl.hpp"
#include "tensor/tensor_impl_wrapper.hpp"

namespace tt {

namespace tt_metal {

namespace tensor_impl {

std::ostream& operator<<(std::ostream& os, const DataType& dtype) {
    switch (dtype) {
        case DataType::BFLOAT16: os << "bfloat16"; break;
        case DataType::FLOAT32: os << "float32"; break;
        case DataType::UINT32: os << "uint32"; break;
        case DataType::BFLOAT8_B: os << "bfloat8_b"; break;
        default: throw std::invalid_argument("Unknown data type");
    }
    return os;
}

uint32_t get_page_size(DataType dtype, Layout layout, uint32_t total_size_bytes, const std::array<uint32_t, 4>& shape) {
    uint32_t W = shape[3];
    uint32_t C = shape[1];
    uint32_t page_size = 0;
    switch (layout) {
        case Layout::ROW_MAJOR: {
            page_size = W * 2;
        }
        break;
        case Layout::TILE: {
            // TODO: Update to be generic for data type (issue 462)
            switch (dtype) {
                case DataType::BFLOAT16:
                case DataType::FLOAT32: {
                    // Float is converted to bfloat16 before being written to device
                    uint32_t size_of_element = element_size_bytes_wrapper(DataType::BFLOAT16);
                    page_size = 32 * 32 * size_of_element;
                }
                break;
                case DataType::UINT32: {
                    uint32_t size_of_element = element_size_bytes_wrapper(dtype);
                    page_size = 32 * 32 * size_of_element;
                }
                break;
                case DataType::BFLOAT8_B:  {
                    page_size = 1088; // (256 * 4) + (16 *4)
                }
                break;
                default:
                    TT_ASSERT(false && "Unsupported data type!");
            }
            TT_ASSERT(total_size_bytes % page_size == 0);
        }
        break;
        case Layout::CHANNELS_LAST:
            page_size = C * 2;
        break;
        default:
            TT_ASSERT(false && "Unsupported layout to write to device");
    }
    TT_ASSERT(page_size != 0);
    return page_size;
}

namespace detail {

DeviceBuffer allocate_interleaved_buffer_on_device(uint32_t buffer_size_bytes, Device *device, const Shape& shape, DataType data_type, Layout layout, const MemoryConfig& memory_config) {
    uint32_t page_size = get_page_size(data_type, layout, buffer_size_bytes, shape);
    return std::make_shared<Buffer>(device, buffer_size_bytes, page_size, memory_config.buffer_type);
}

DeviceBuffer allocate_contiguous_buffer_on_device(uint32_t buffer_size_bytes, Device *device, const MemoryConfig& memory_config) {
    return std::make_shared<Buffer>(device, buffer_size_bytes, buffer_size_bytes, memory_config.buffer_type);
}

}

DeviceBuffer allocate_buffer_on_device(uint32_t buffer_size_bytes, Device *device, const Shape& shape, DataType data_type, Layout layout, const MemoryConfig& memory_config) {
    if (memory_config.interleaved) {
        return detail::allocate_interleaved_buffer_on_device(buffer_size_bytes, device, shape, data_type, layout, memory_config);
    } else {
        return detail::allocate_contiguous_buffer_on_device(buffer_size_bytes, device, memory_config);
    }
}

void validate_on_device_dtype_and_layout(Device *device, DataType dtype, Layout layout) {
    // TODO: Get supported layout and dtypes from device
    auto supported_dtype = [&dtype]() {
        TT_ASSERT(
            (dtype == DataType::BFLOAT16 || dtype == DataType::BFLOAT8_B) &&
            "Only BFLOAT16 or BFLOAT8_B is supported on device!"
        );
    };
    auto supported_layout = [&dtype, &layout]() {
        switch (dtype) {
            case DataType::BFLOAT16:
                break;
            case DataType::BFLOAT8_B:
                TT_ASSERT(layout == Layout::TILE && "Only TILE layout is supported for BFLOAT8_B dtype!");
                break;
            default:
                TT_ASSERT(false && "Only BFLOAT16 or BFLOAT8_B is supported on device!");
                break;
            }
    };
    supported_dtype();
    supported_layout();
}


}  // namespace tensor_impl

}  // namespace tt_metal

}  // namespace tt
