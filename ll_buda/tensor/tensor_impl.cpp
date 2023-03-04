#include "ll_buda/tensor/tensor_impl.hpp"
#include "ll_buda/tensor/tensor_impl_wrapper.hpp"

namespace tt {

namespace ll_buda {

namespace tensor_impl {

std::tuple<int, int, int> get_interleaved_read_write_unit_metadata(
    DataType dtype, Layout layout, uint32_t total_size_bytes, const std::array<uint32_t, 4>& shape) {
    uint32_t W = shape[3];
    int num_bank_units;
    int num_entries_per_bank_unit;
    int num_bytes_per_entry;
    switch (layout) {
        case Layout::ROW_MAJOR: {
            num_bank_units = total_size_bytes / (W*2);
            num_entries_per_bank_unit = W/2; // num elements in tile packed as uint32
            num_bytes_per_entry = 4;
        }
        break;
        case Layout::TILE: {
            uint32_t size_of_element;
            int num_elements_packed_as_uint32;
            switch (dtype) {
                case DataType::BFLOAT16:
                case DataType::FLOAT32: {
                    // Float is converted to bfloat16 before being written to device
                    size_of_element = element_size_bytes_wrapper(DataType::BFLOAT16);
                    num_elements_packed_as_uint32 = 2;
                }
                break;
                case DataType::UINT32: {
                    size_of_element = element_size_bytes_wrapper(dtype);
                    num_elements_packed_as_uint32 = 1;
                }
                break;
                default:
                    TT_ASSERT(false && "Unsupported data type!");
            }
            int tile_size = 32 * 32 * size_of_element; // TODO: Update to be generic for data type
            TT_ASSERT(total_size_bytes % tile_size == 0);
            num_bank_units = total_size_bytes / tile_size;
            num_entries_per_bank_unit = (32 * 32) / num_elements_packed_as_uint32;
            num_bytes_per_entry = 4;
        }
        break;
        case Layout::CHANNELS_LAST:
            TT_ASSERT(false && "Writing in CHANNELS_LAST layout to device is currently unsupported");
        break;
        default:
            TT_ASSERT(false && "Unsupported layout to write to device");
    }
    return {num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry};
}

void allocate_interleaved_buffer_on_device(Tensor &tensor, uint32_t buffer_size_bytes) {
    auto [num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry] = get_interleaved_read_write_unit_metadata(tensor.dtype(), tensor.layout(), buffer_size_bytes, tensor.shape());
    tensor.interleaved_buffer_ = CreateInterleavedDramBuffer(tensor.device(), num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry);
}

void validate_on_device_dtype_and_layout(Device *device, DataType dtype, Layout layout) {
    // TODO: Get supported layout and dtypes from device
    auto supported_dtype = [&dtype]() { return dtype == DataType::BFLOAT16; };
    auto supported_layout = [&layout]() { return layout == Layout::ROW_MAJOR or layout == Layout::TILE; };
    TT_ASSERT(supported_dtype() && "Only BFLOAT16 is supported on device!");
    TT_ASSERT(supported_layout() && "Only ROW_MAJOR and TILE layouts are supported on device!");
}


}  // namespace tensor_impl

}  // namespace ll_buda

}  // namespace tt
