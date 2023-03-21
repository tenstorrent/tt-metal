#include <chrono>
#include <fstream>
#include <errno.h>
#include <random>
#include <string>

#include "tt_cluster.hpp"
#include "utils.hpp"
#include "common/logger.hpp"
#include "tensix.h"

#include "llrt.hpp"
#include "test_libs/tiles.hpp"

enum INITIALIZE
{
    Zeros = 0,
    Ones = 1,
    Increment = 2
};

class Tensor {
    public:
        Tensor(std::vector<uint32_t> &values, std::array<uint32_t, 4> &shape) {
            this->shape = shape;
            this->values = values;
            this->strides = {shape[1]*shape[2]*shape[3], shape[2]*shape[3], shape[3], 1};
        }
        Tensor(std::array<uint32_t, 4> &shape) {
            this->shape = shape;
            auto volume = shape[0] * shape[1] * shape[2] * shape[3];
            this->values.resize(volume, 0);
            this->strides = {shape[1]*shape[2]*shape[3], shape[2]*shape[3], shape[3], 1};
        }
        vector<uint32_t> get_values() {
            return this->values;
        }
        void print() {
            std::cout<<"Shape = ["<<shape[0]<<","<<shape[1]<<","<<shape[2]<<","<<shape[3]<<"]"<<std::endl;
            std::cout<<"Strides = ["<<strides[0]<<","<<strides[1]<<","<<strides[2]<<","<<strides[3]<<"]"<<std::endl;
            std::cout<<"Values = [";
            for(auto w = 0; w < shape[0]; w++) {
                std::cout<<"[";
                for(auto z = 0; z < shape[1]; z++) {
                    std::cout<<"[";
                    for(auto y = 0; y < shape[2]; y++) {
                        std::cout<<"[";
                        for(auto x = 0; x < shape[3]; x++) {
                            auto idx = x + shape[3] * y + shape[3] * shape[2] * z + shape[3] * shape[2] * shape[1] * w;
                            std::cout<<values[idx]<<",";
                        }
                        std::cout<<"]"<<std::endl;
                    }
                    std::cout<<"]"<<std::endl;
                }
                std::cout<<"]"<<std::endl;
            }
            std::cout<<"]"<<std::endl;
        }
    // private:
        vector<uint32_t> values;
        std::array<uint32_t, 4> shape; // outer-most dimension first
        std::array<uint32_t, 4> strides; // outer-most dimension first
};

Tensor pad(Tensor &input) {
    std::array<uint32_t, 4> in_shape = input.shape;
    std::array<uint32_t, 4> out_shape = {in_shape[0], in_shape[1], in_shape[2] + 2, in_shape[3] + 2};

    std::vector<uint32_t> out;
    for(auto w = 0; w < in_shape[0]; w++) {
        for(auto z = 0; z < in_shape[1]; z++) {
            for(auto i = 0; i < out_shape[3]; i++) {
                out.push_back(0);
            }
            for(auto y = 0; y < in_shape[2]; y++) {
                out.push_back(0);
                for(auto x = 0; x < in_shape[3]; x++) {
                    auto idx = x + in_shape[3] * y + in_shape[3] * in_shape[2] * z + in_shape[3] * in_shape[2] * in_shape[1] * w;
                    out.push_back(input.values[idx]);
                }
                out.push_back(0);
            }
            for(auto i = 0; i < out_shape[3]; i++) {
                out.push_back(0);
            }
        }
    }

    return Tensor(out, out_shape);
}

Tensor get_tensor(std::array<uint32_t, 4> &shape, INITIALIZE init_type) {
    std::vector<uint32_t> values;
    for(auto w = 0; w < shape[0]; w++) {
        for(auto z = 0; z < shape[1]; z++) {
            for(auto y = 0; y < shape[2]; y++) {
                for(auto x = 0; x < shape[3]; x++) {
                    uint32_t val;
                    switch (init_type)
                    {
                        case INITIALIZE::Zeros:
                            val = 0;
                            break;
                        case INITIALIZE::Ones:
                            val = 1;
                            break;
                        case INITIALIZE::Increment:
                            val = x + shape[3] * y + shape[3] * shape[2] * z + shape[3] * shape[2] * shape[1] * w;
                            break;
                        default:
                            break;
                    }
                    values.push_back(val);
                }
            }
        }
    }

    return Tensor(values, shape);
}

Tensor convert_XYZW_to_ZXYW(Tensor &input) {
    std::array<uint32_t, 4> in_shape = input.shape;
    std::array<uint32_t, 4> out_shape = {in_shape[0], in_shape[2], in_shape[3], in_shape[1]};
    auto output_volume = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3];
    std::vector<uint32_t> out = std::vector<uint32_t>(output_volume, 0);

    for(auto w = 0; w < in_shape[0]; w++) { // N
        for(auto z = 0; z < in_shape[1]; z++) { // Z
            for(auto y = 0; y < in_shape[2]; y++) { // Y
                for(auto x = 0; x < in_shape[3]; x++) { // X
                    auto in_idx = x + y * in_shape[3] + z * in_shape[3] * in_shape[2] + w * in_shape[3] * in_shape[2] * in_shape[1];
                    auto out_idx = z + x * out_shape[3] + y * out_shape[3] * out_shape[2] + w * out_shape[3] * out_shape[2] * out_shape[1];
                    out[out_idx] = input.values[in_idx];
                }
            }
        }
    }

    return Tensor(out, out_shape);
}

vector<vector<uint32_t>> move_act_dram_to_l1(Tensor &input_zxyw) {
    vector<vector<uint32_t>> output;
    std::array<uint32_t, 4> in_shape = input_zxyw.shape;


    std::vector<std::pair<int, int>> increments;
    for (auto j=0; j<3; j++) {
        for(auto i =0; i<3; i++) {
            increments.push_back(std::make_pair(i - 1, j - 1));
        }
    }

    for(int w = 0; w < in_shape[0]; w++) {
        for(int y = 1; y < in_shape[1] - 1; y++) {
            for(int x = 1; x < in_shape[2] - 1; x++) {
                std::vector<uint32_t> row;
                // cout<<"X = "<<x<<" Y = "<<y<<endl;
                for(auto increment: increments) {
                    auto x_new = x + increment.first;
                    auto y_new = y + increment.second;
                    // cout<<"\tXnew = "<<x_new<<" Ynew = "<<y_new<<endl;
                    for(int z = 0; z < in_shape[3]; z++) {
                        auto idx = z + x_new * in_shape[3] + y_new * in_shape[3] * in_shape[2] + w * in_shape[3] * in_shape[2] * in_shape[1];
                        assert(idx >= 0 and idx < input_zxyw.values.size());
                        row.push_back(input_zxyw.values[idx]);
                    }
                }
                output.push_back(row);
            }
        }

    }

    return output;
}

vector<uint32_t> flatten(vector<vector<uint32_t>> &act_matrix) {
    vector<uint32_t> output;
    for(auto i = 0 ; i < act_matrix.size(); i++) {
        for(auto j = 0; j < act_matrix[i].size(); j++) {
            output.push_back(act_matrix[i][j]);
        }
    }
    return output;
}

/*
Generate all dram to l1 move specs. The input is given as ZXYW layout
and we need to move it as a 2D matrix layout. Of course, both dram and
L1 have 1D address range, but the stored data should reflect the proper layout
(i.e. ZXYW vs 2D).
*/
std::vector<tt::llrt::DramToL1CopySpec> get_dram_to_l1_specs(
    const tt_xy_pair core,
    int dram_channel_id,
    Tensor &activation_zxyw,
    std::uint32_t starting_dram_address,
    std::uint32_t starting_l1_address
) {
    std::vector<tt::llrt::DramToL1CopySpec> specs = {};
    tt::llrt::LoadFirmwareFlag load_firmware_flag = true;
    vector<uint32_t> activation_vector = activation_zxyw.values;
    std::uint32_t total_buffer_size = activation_vector.size() * sizeof(std::uint32_t);
    unsigned total_vec_size = activation_vector.size();
    std::vector<std::pair<int, int>> increments;
    for (auto j=0; j<3; j++) {
        for(auto i =0; i<3; i++) {
            increments.push_back(std::make_pair(i - 1, j - 1));
        }
    }
    std::uint32_t l1_address = starting_l1_address;
    std::uint32_t chunk_size = activation_zxyw.shape[3] * sizeof(uint32_t); // input channels * 4B
    for(int w = 0; w < activation_zxyw.shape[0]; w++) { // w = 0
        for(int y = 1; y < activation_zxyw.shape[1] - 1; y++) { // y = 1 .. 33
            for(int x = 1; x < activation_zxyw.shape[2] - 1; x++) { // x = 1 .. 33
                for(auto increment: increments) { // go over all 3x3 window
                    auto x_new = x + increment.first;
                    auto y_new = y + increment.second;
                    auto idx = x_new * activation_zxyw.shape[3] + y_new * activation_zxyw.shape[3] * activation_zxyw.shape[2] + w * activation_zxyw.shape[3] * activation_zxyw.shape[2] * activation_zxyw.shape[1];
                    TT_ASSERT(idx >= 0 and idx + 15 < activation_zxyw.values.size());
                    auto dram_address = starting_dram_address + idx * sizeof(uint32_t);
                    TT_ASSERT(dram_address >= 0 and dram_address <=total_buffer_size);
                    // move the data at xnew/ynew to L1
                    specs.push_back(tt::llrt::create_dram_to_l1_copy_spec(
                        core,
                        dram_channel_id,
                        chunk_size/*buffer size*/,
                        dram_address/*dram address*/,
                        l1_address,
                        load_firmware_flag
                    ));
                    load_firmware_flag = false;
                    l1_address += chunk_size;
                }
            }
        }
    }
    return specs;
}

int main(int argc, char** argv)
{
    bool pass = true;

    const TargetDevice target_type = TargetDevice::Silicon;
    const tt::ARCH arch = tt::ARCH::GRAYSKULL;
    const std::string sdesc_file = get_soc_description_file(arch, target_type);

    const int chip_id = 0;
    int dram_channel_id = 0;
    const tt_xy_pair core = {11, 3};

    try {
        tt_device_params default_params;
        tt_cluster *cluster = new tt_cluster;
        cluster->open_device(arch, target_type, {0}, sdesc_file);
        cluster->start_device(default_params); // use default params
        tt::llrt::utils::log_current_ai_clk(cluster);

        std::uint32_t starting_l1_address = 250 * 1024;
        std::uint32_t starting_dram_address = 0;

        std::array<uint32_t, 4> act_shape = {1, 16, 32, 32};
        Tensor activation = get_tensor(act_shape, INITIALIZE::Increment);
        // Tensor becomes 34x34 with zero padding
        Tensor padded_activation = pad(activation);
        // Tranform the XYZW to ZXYW layout
        Tensor activation_zxyw = convert_XYZW_to_ZXYW(padded_activation);
        // This will create the 2D matrix by modeling what dram to l1 read patterns are
        vector<vector<uint32_t>> golden_matrix = move_act_dram_to_l1(activation_zxyw);
        // This would be the actual golden that we compare the L1 data against
        vector<uint32_t> golden_vector = flatten(golden_matrix);
        vector<uint32_t> activation_vector = activation_zxyw.values;
        std::uint32_t chunk_size = act_shape[1] * sizeof(uint32_t);

        log_info(tt::LogVerif, "Running dram to l1 copy for dram channel {} -> core {}", dram_channel_id, core.str());
        cluster->write_dram_vec(activation_vector, tt_target_dram{chip_id, dram_channel_id, 0}, starting_dram_address); // write to address

        auto specs = get_dram_to_l1_specs(core, dram_channel_id, activation_zxyw, starting_dram_address, starting_l1_address);
        bool load_blanks = true;
        for(auto &spec: specs) {
            std::vector<tt::llrt::DramToL1CopySpec> new_specs = {spec};
            tt::llrt::run_dram_to_l1_copy_kernel_with_specs(cluster, chip_id, new_specs, load_blanks);
            load_blanks = false;
        }
        // read only the first 9 chuncks from L1
        std::uint32_t total_l1_buffer_size = chunk_size * 9 * act_shape[2] * act_shape[3];
        vector<std::uint32_t> dst_vec = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, starting_l1_address, total_l1_buffer_size); // read size is in bytes
        pass &= golden_vector == dst_vec;

        cluster->close_device();
        delete cluster;

    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        log_error(tt::LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(tt::LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(tt::LogTest, "Test Passed");
    } else {
        log_fatal(tt::LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
