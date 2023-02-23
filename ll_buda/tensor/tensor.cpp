#include "ll_buda/tensor/tensor.hpp"

#include "common/bfloat16.hpp"
#include "llrt/llrt.hpp"
#include "ll_buda/host_api.hpp"
#include "constants.hpp"

using namespace tt::constants;

namespace tt {

namespace ll_buda {

// Copied from root/tensor.hpp. TODO: Consolidate
std::vector<bfloat16> initialize_row_major_tensor_data(const std::array<uint32_t, 4> &shape, Initialize init_type, int rand_max_val = 100, int seed = 0) {
    std::vector<bfloat16> values;
    auto rand_float = std::bind(std::uniform_real_distribution<float>(0, rand_max_val), std::mt19937(seed));
    for(auto w = 0; w < shape[0]; w++) {
        for(auto z = 0; z < shape[1]; z++) {
            for(auto y = 0; y < shape[2]; y++) {
                for(auto x = 0; x < shape[3]; x++) {
                    float val;
                    switch (init_type)
                    {
                        case Initialize::ZEROS:
                            val = 0;
                            break;
                        case Initialize::ONES:
                            val = 1;
                            break;
                        case Initialize::INCREMENT:
                            val = x + shape[3] * y + shape[3] * shape[2] * z + shape[3] * shape[2] * shape[1] * w;
                            break;
                        case Initialize::RANDOM:
                            val = rand_float();
                            break;
                        default:
                            val = 0;
                            break;
                    }
                    values.push_back(static_cast<bfloat16>(val));
                }
            }
        }
    }

    return values;
}

// Checks that there is only one DRAM buffer per DRAM bank and all buffers share the same address
void validate_interleaved_buffers_address(const std::vector<DramBuffer *> buffers) {
    if (buffers.empty()) {
        return;
    }
    std::unordered_set<int> dram_channels;
    for (auto buffer : buffers) {
        if (buffer->address() != buffers.at(0)->address()) {
            TT_THROW("Expected interleaved buffers to share the same base address!");
        }
        if (dram_channels.find(buffer->dram_channel()) != dram_channels.end()) {
            TT_THROW("Expected only one DRAM buffer per DRAM bank!");
        }
    }
}

std::tuple<int, int, int> get_interleaved_read_write_unit_metadata(
    DataFormat data_type, Layout layout, uint32_t total_size_bytes, const std::array<uint32_t, 4>& shape) {

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
            int tile_size = 32 * 32 * sizeof(bfloat16); // TODO: Update to be generic for data type
            TT_ASSERT(total_size_bytes % tile_size == 0);
            num_bank_units = total_size_bytes / tile_size;
            num_entries_per_bank_unit = 1024 / 2; // num elements in tile packed as uint32
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

void Tensor::allocate_buffers_on_device(uint32_t buffer_size_bytes) {

    auto [num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry] = get_interleaved_read_write_unit_metadata(data_type_, layout_, buffer_size_bytes, shape());
    buffers_ = CreateInterleavedDramBuffers(device_, num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry);
    if (buffers_.empty()) {
        TT_THROW("Not enough memory to create buffers with total size " + std::to_string(buffer_size_bytes) + " bytes on device!");
    }
    validate_interleaved_buffers_address(buffers_);
}

std::vector<bfloat16> Tensor::initialize_data(const std::array<uint32_t, 4> &shape, Initialize init_type, Layout layout) {
    TT_ASSERT(layout == Layout::TILE or layout == Layout::ROW_MAJOR, "Only ROW_MAJOR or TILE layout is supported!");
    std::vector<bfloat16> data = initialize_row_major_tensor_data(shape, init_type);
    if (layout == Layout::TILE) {
        data = convert_layout_row_major_to_tile(data);
    }
    return data;
}

Tensor::Tensor(std::vector<uint32_t> data, const std::array<uint32_t, 4> &shape, DataFormat data_type, Layout layout)
    : data_(unpack_uint32_vec_into_bfloat16_vec(data)), shape_(shape), strides_(compute_strides()), data_type_(data_type), layout_(layout) {
}

Tensor::Tensor(std::vector<uint32_t> data, const std::array<uint32_t, 4> &shape, DataFormat data_type, Layout layout, Device *device)
    : shape_(shape), strides_(compute_strides()), data_type_(data_type), layout_(layout), device_(device) {
    TT_ASSERT(device != nullptr);
    uint32_t size_in_bytes = data.size() * sizeof(uint32_t); // TODO: Update to be generic for data type
    allocate_buffers_on_device(size_in_bytes);
    auto [num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry] = get_interleaved_read_write_unit_metadata(data_type, layout, size_in_bytes, shape_);
    WriteToDeviceDRAMChannelsInterleaved(device, data, buffers_.at(0)->address(), num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry);
}

/*
    TODO: Generalize to other data formats
*/
std::vector<uint32_t> convert_float_data_to_uint32_data(std::vector<float> &float_data) {
    std::vector<uint32_t> uint32_data;
    assert(float_data.size() % 2 == 0);
    for (auto i = 0; i < float_data.size(); i += 2) {
        auto float_val1 = float_data[i];
        auto float_val2 = float_data[i + 1];
        auto bfloat_val1 = bfloat16(float_val1);
        auto bfloat_val2 = bfloat16(float_val2);
        auto uint32_val = pack_two_bfloat16_into_uint32({bfloat_val1, bfloat_val2});
        uint32_data.push_back(uint32_val);
    }
    return uint32_data;
}

Tensor::Tensor(std::vector<float> &data, const std::array<uint32_t, 4> &shape, DataFormat data_type, Layout layout)
    : Tensor(convert_float_data_to_uint32_data(data), shape, data_type, layout) {}

Tensor::Tensor(std::vector<float> &data, const std::array<uint32_t, 4> &shape, DataFormat data_type, Layout layout, Device *device)
    : Tensor(convert_float_data_to_uint32_data(data), shape, data_type, layout, device) {}

Tensor::Tensor(const std::array<uint32_t, 4> &shape, Initialize init_type, DataFormat data_type, Layout layout)
    : shape_(shape), strides_(compute_strides()), data_type_(data_type), layout_(layout) {
    TT_ASSERT(data_type == DataFormat::Float16_b);
    data_ = initialize_data(shape, init_type, layout);
}

Tensor::Tensor(const std::array<uint32_t, 4> &shape, Initialize init_type, DataFormat data_type, Layout layout, Device *device)
    : shape_(shape), strides_(compute_strides()), data_type_(data_type), layout_(layout), device_(device) {
    TT_ASSERT(device != nullptr);
    TT_ASSERT(data_type == DataFormat::Float16_b);
    auto bfloat16_data = initialize_data(shape, init_type, layout);
    uint32_t size_in_bytes = bfloat16_data.size() * sizeof(bfloat16); // TODO: Update to be generic for data type
    allocate_buffers_on_device(size_in_bytes);
    auto [num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry] = get_interleaved_read_write_unit_metadata(data_type, layout, size_in_bytes, shape_);
    auto uint32_data = pack_bfloat16_vec_into_uint32_vec(bfloat16_data);
    WriteToDeviceDRAMChannelsInterleaved(device, uint32_data, buffers_.at(0)->address(), num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry);
}

Tensor::Tensor(const std::array<uint32_t, 4> &shape, DataFormat data_type, Layout layout, Device *device)
    : shape_(shape), strides_(compute_strides()), data_type_(data_type), layout_(layout), device_(device) {
    TT_ASSERT(device != nullptr);
    TT_ASSERT(layout == Layout::ROW_MAJOR or layout == Layout::TILE, "Only ROW_MAJOR or TILE layout is supported!");
    uint32_t size_in_bytes = volume() * sizeof(bfloat16); // TODO: Update to be generic for data type
    allocate_buffers_on_device(size_in_bytes);
}

Tensor Tensor::copy_to_host() const {
    TT_ASSERT(device_ != nullptr, "Need device to be set copy data from device to host!");
    TT_ASSERT(not buffers_.empty(), "Need DRAM buffers on device to exist to copy data to host!");
    std::vector<uint32_t> device_data;
    uint32_t size_in_bytes = 0;
    for (auto buffer : buffers_) {
        size_in_bytes += buffer->size();
    }
    auto [num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry] = get_interleaved_read_write_unit_metadata(data_type_, layout_, size_in_bytes, shape());
    ReadFromDeviceDRAMChannelsInterleaved(device_, device_data, buffers_.at(0)->address(), num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry);
    return Tensor(device_data, shape_, data_type_, layout_);
}

Tensor Tensor::copy_to_device(Device *device) const {
    std::vector<bfloat16> data_copy = data_;
    return Tensor(pack_bfloat16_vec_into_uint32_vec(data_), shape_, data_type_, layout_, device);
}

Tensor Tensor::to(Device *target_device) const {
    if (on_host()) {
        return copy_to_device(target_device);
    }
    TT_ASSERT(device_ == target_device && "Currently do not support moving between devices");
    return Tensor(*this);
}

Tensor Tensor::to(Host *host) const {
    if (on_host()) {
        return *this;
    }
    return copy_to_host();
}

void print_data(const std::vector<bfloat16> &data, DataFormat data_type) {
    TT_ASSERT(data_type == DataFormat::Float32 or data_type == DataFormat::Float16_b);
    std::cout << "[ ";
    for (int i = 0; i < data.size(); i++) {
        bfloat16 datum = data[i];
        if (data_type == DataFormat::Float32) {
            std::cout << datum.to_float() << ", ";
        } else {
            std::cout << datum << ", ";
        }
    }
    std::cout << " dtype=" <<  data_type << " ]" << std::endl;
}

void Tensor::print(Layout print_layout) const {
    std::vector<bfloat16> data_to_print = data_;
    if (not on_host()) {
        auto temp_tensor = copy_to_host();
        data_to_print = temp_tensor.to_vec();
    }

    switch (layout_) {
        case Layout::ROW_MAJOR:
            if (print_layout == Layout::ROW_MAJOR) {
                print_data(data_to_print, data_type_);
            } else if (print_layout == Layout::TILE) {
                data_to_print = convert_layout_row_major_to_tile(data_to_print);
                print_data(data_to_print, data_type_);
            } else {
                TT_ASSERT(false && "Unsupported print layout");
            }
        break;
        case Layout::TILE:
            if (print_layout == Layout::ROW_MAJOR) {
                data_to_print = convert_layout_tile_to_row_major(data_to_print);
                print_data(data_to_print, data_type_);
            } else if (print_layout == Layout::TILE) {
                print_data(data_to_print, data_type_);
            } else {
                TT_ASSERT(false && "Unsupported print layout");
            }
        break;
        case Layout::CHANNELS_LAST:
            TT_ASSERT(false && "Unsupported print layout");
        break;
        default:
            TT_ASSERT(false && "Unsupported print layout");
    }
}

void print_row_major_data(const std::vector<bfloat16> &data, std::array<uint32_t, 4> shape) {
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
                    std::cout << data[index];
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
    std::cout << " ]" << std::endl;
}

void Tensor::pretty_print(Layout print_layout) const {
    std::vector<bfloat16> data_to_print = data_;
    if (not on_host()) {
        auto temp_tensor = copy_to_host();
        data_to_print = temp_tensor.to_vec();
    }

    switch (layout_) {
        case Layout::ROW_MAJOR:
            if (print_layout == Layout::ROW_MAJOR) {
                print_row_major_data(data_to_print, shape_);
            } else if (print_layout == Layout::TILE) {
                data_to_print = convert_layout_row_major_to_tile(data_to_print);
                print_row_major_data(data_to_print, shape_);
            } else {
                TT_ASSERT(false && "Unsupported print layout");
            }
        break;
        default:
            TT_ASSERT(false && "Unsupported print layout");
    }
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

}  // namespace ll_buda

}  // namespace tt
