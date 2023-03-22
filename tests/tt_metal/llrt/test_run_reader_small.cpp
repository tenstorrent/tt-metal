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

void print(std::vector<uint32_t> &vec, std::string name) {
    std::cout<<name <<" = [";
    for(auto i = 0; i < vec.size(); i++) {
        std::cout<<vec[i]<<",";
    }
    std::cout<<"]"<<std::endl;
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

        /*
            Smaller test case:
            - activation initially is (1, 16, 1, 1) with conv3x3
            - after padding it becomes (1, 16, 3, 3)
            - transform to DRAM layout (1, 3, 3, 16)
            - move to L1 2D matrix layout: 1 row, 16 x (3x3) columns
            - essentially dram layout would be the same as the L1 layout in this case
        */
        std::array<uint32_t, 4> shape = {1, 3, 3, 16}; // shape in DRAM layout (W, Y, X, Z)
        std::vector<uint32_t> src_vec = {
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, //x=0, y=0
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, //x=1, y=0
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, //x=2, y=0
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, //x=0, y=1
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, //x=1, y=1
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, //x=2, y=1
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, //x=0, y=2
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, //x=1, y=2
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0  //x=2, y=2
        };
        std::vector<uint32_t> golden_vector = src_vec; // special case when golden vec is the same as src vec
        // print(golden_vector, "Golden");
        std::uint32_t chunck_size = 16 * sizeof(uint32_t); // Z * 4B

        cluster->write_dram_vec(src_vec, tt_target_dram{chip_id, dram_channel_id, 0}, starting_dram_address); // write to address

        std::vector<tt::llrt::DramToL1CopySpec> specs = {
            tt::llrt::create_dram_to_l1_copy_spec(core, dram_channel_id, chunck_size/*buffer size*/, 0*chunck_size/*dram address*/, starting_l1_address+0*chunck_size, true  /*load_firmware_flag*/),
            tt::llrt::create_dram_to_l1_copy_spec(core, dram_channel_id, chunck_size/*buffer size*/, 1*chunck_size/*dram address*/, starting_l1_address+1*chunck_size, false /*load_firmware_flag*/),
            tt::llrt::create_dram_to_l1_copy_spec(core, dram_channel_id, chunck_size/*buffer size*/, 2*chunck_size/*dram address*/, starting_l1_address+2*chunck_size, false /*load_firmware_flag*/),
            tt::llrt::create_dram_to_l1_copy_spec(core, dram_channel_id, chunck_size/*buffer size*/, 3*chunck_size/*dram address*/, starting_l1_address+3*chunck_size, false /*load_firmware_flag*/),
            tt::llrt::create_dram_to_l1_copy_spec(core, dram_channel_id, chunck_size/*buffer size*/, 4*chunck_size/*dram address*/, starting_l1_address+4*chunck_size, false /*load_firmware_flag*/),
            tt::llrt::create_dram_to_l1_copy_spec(core, dram_channel_id, chunck_size/*buffer size*/, 5*chunck_size/*dram address*/, starting_l1_address+5*chunck_size, false /*load_firmware_flag*/),
            tt::llrt::create_dram_to_l1_copy_spec(core, dram_channel_id, chunck_size/*buffer size*/, 6*chunck_size/*dram address*/, starting_l1_address+6*chunck_size, false /*load_firmware_flag*/),
            tt::llrt::create_dram_to_l1_copy_spec(core, dram_channel_id, chunck_size/*buffer size*/, 7*chunck_size/*dram address*/, starting_l1_address+7*chunck_size, false /*load_firmware_flag*/),
            tt::llrt::create_dram_to_l1_copy_spec(core, dram_channel_id, chunck_size/*buffer size*/, 8*chunck_size/*dram address*/, starting_l1_address+8*chunck_size, false /*load_firmware_flag*/)
        };
        bool load_blanks = true;
        for(auto &spec: specs) {
            std::vector<tt::llrt::DramToL1CopySpec> new_specs = {spec};
            tt::llrt::run_dram_to_l1_copy_kernel_with_specs(cluster, chip_id, new_specs, load_blanks);
            load_blanks = false;
        }
        // read only the first 9 chuncks from L1
        vector<std::uint32_t> dst_vec = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, starting_l1_address, chunck_size*9); // read size is in bytes
        // print(dst_vec, "Result");

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
