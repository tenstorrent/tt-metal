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
        default: throw std::invalid_argument("Unknown data type");
    }
    return os;
}

std::tuple<int, int, int> get_interleaved_read_write_unit_metadata(
    DataType dtype, Layout layout, uint32_t total_size_bytes, const std::array<uint32_t, 4>& shape) {
    uint32_t W = shape[3];
    uint32_t C = shape[1];
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
            num_bank_units = total_size_bytes / (C*2);
            num_entries_per_bank_unit = C/2; // num elements in tile packed as uint32
            num_bytes_per_entry = 4;
        break;
        default:
            TT_ASSERT(false && "Unsupported layout to write to device");
    }
    return {num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry};
}

void allocate_interleaved_buffer_on_device(Tensor &tensor, uint32_t buffer_size_bytes) {
    auto [num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry] = get_interleaved_read_write_unit_metadata(tensor.dtype(), tensor.layout(), buffer_size_bytes, tensor.shape());
    tensor.buffer_ = CreateInterleavedDramBuffer(tensor.device(), num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry);
}

void allocate_dram_buffer_on_device(Tensor &tensor, uint32_t buffer_size_bytes) {
    TT_ASSERT(tensor.mem_config_.dram_channel != -1);
    tensor.buffer_ = CreateDramBuffer(tensor.device(), tensor.mem_config_.dram_channel, buffer_size_bytes);
}

void allocate_buffer_on_device(Tensor &tensor, uint32_t buffer_size_bytes) {
    if (tensor.interleaved()) {
        allocate_interleaved_buffer_on_device(tensor, buffer_size_bytes);
    } else {
        allocate_dram_buffer_on_device(tensor, buffer_size_bytes);
    }
}

void read_interleaved_data_from_device(const Tensor &tensor, uint32_t size_in_bytes, std::vector<uint32_t> &host_buffer) {
    auto [num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry] = get_interleaved_read_write_unit_metadata(tensor.dtype(), tensor.layout(), size_in_bytes, tensor.shape());
    ReadFromDeviceDRAMChannelsInterleaved(tensor.device(), host_buffer, tensor.buffer()->address(), num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry);
}

void read_contiguous_data_from_device(const Tensor &tensor, uint32_t size_in_bytes, std::vector<uint32_t> &host_buffer) {
    TT_ASSERT(tensor.buffer()->size() == size_in_bytes);
    auto dram_buffer = dynamic_cast<DramBuffer *>(tensor.buffer());
    TT_ASSERT(dram_buffer != nullptr);
    ReadFromDeviceDRAM(dram_buffer, host_buffer);
}

void write_interleaved_data_to_device(const Tensor &tensor, std::vector<uint32_t> &host_buffer) {
    uint32_t packed_size_in_bytes = host_buffer.size() * sizeof(uint32_t);
    auto [num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry] = get_interleaved_read_write_unit_metadata(tensor.dtype(), tensor.layout(), packed_size_in_bytes, tensor.shape());
    WriteToDeviceDRAMChannelsInterleaved(
        tensor.device(), host_buffer, tensor.buffer()->address(), num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry);
}

void write_contiguous_data_to_device(const Tensor &tensor, std::vector<uint32_t> &host_buffer) {
    auto dram_buffer = dynamic_cast<DramBuffer *>(tensor.buffer());
    TT_ASSERT(dram_buffer != nullptr);
    WriteToDeviceDRAM(dram_buffer, host_buffer);
}

void validate_on_device_dtype_and_layout(Device *device, DataType dtype, Layout layout) {
    // TODO: Get supported layout and dtypes from device
    auto supported_dtype = [&dtype]() { return dtype == DataType::BFLOAT16; };
    auto supported_layout = [&layout]() { return true; };
    TT_ASSERT(supported_dtype() && "Only BFLOAT16 is supported on device!");
    TT_ASSERT(supported_layout());
}


}  // namespace tensor_impl

}  // namespace tt_metal

}  // namespace tt
