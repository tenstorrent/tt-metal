#include <chrono>
#include <fstream>
#include <errno.h>
#include <random>
#include <algorithm>

#include "llrt/llrt.hpp"
#include "utils.hpp"
#include "common/logger.hpp"
#include "test_libs/tiles.hpp"

using RAMData = std::vector<uint32_t>;

bool dram_rdwr_check(tt_cluster *cluster, unsigned start_address, std::size_t data_size) {
    RAMData actual_vec;
    std::size_t vec_size = data_size / sizeof(uint32_t);
    RAMData expected_vec = tt::tiles_test::create_random_vec<RAMData>(vec_size, tt::tiles_test::get_seed_from_systime());

    // tt_target_dram = (device_id, channel, subchannel)
    std::vector<tt_target_dram> drams {
        {0, 0, 0},
        {0, 1, 0},
        {0, 2, 0},
        {0, 3, 0},
        {0, 4, 0},
        {0, 5, 0},
        {0, 6, 0},
        {0, 7, 0},
    };

    bool all_are_equal = true;
    for (tt_target_dram dram : drams) {
        int device_id, channel, subchannel;
        std::tie(device_id, channel, subchannel) = dram;
        log_info(tt::LogTest, "Writing to channel {}...", channel);
        cluster->write_dram_vec(expected_vec, dram, start_address); // write to address
        log_info(tt::LogTest, "Wrote to channel {}", channel);
    }

    for (tt_target_dram dram : drams) {
        int device_id, channel, subchannel;
        std::tie(device_id, channel, subchannel) = dram;
        cluster->read_dram_vec(actual_vec, dram, start_address, data_size); // read size is in bytes
        log_info(tt::LogVerif, "expected vec size = {}", expected_vec.size());
        log_info(tt::LogVerif, "actual vec size   = {}", actual_vec.size());
        bool are_equal = actual_vec == expected_vec;

        all_are_equal &= are_equal;
        if (are_equal){
            log_info(tt::LogVerif, "Channel {} has passed", channel);
        }
        else {
            log_error(tt::LogVerif, "Channel {} has not passed", channel);
        }

        std::fill(actual_vec.begin(), actual_vec.end(), 0);
    }

    //log_info(tt::LogTest, "dram_rdwr_check wrote {} and read {}, passed in {} attempts\n", vec, tmp, attempt);

    return all_are_equal;
}

int main(int argc, char** argv)
{
    bool pass = true;

    const std::string output_dir = ".";

    const TargetDevice target_type = TargetDevice::Silicon;
    const tt::ARCH arch = tt::ARCH::GRAYSKULL;
    const std::string sdesc_file = get_soc_description_file(arch, target_type);


    try {
        tt_device_params default_params;
        tt_cluster *cluster = new tt_cluster;
        cluster->open_device(arch, target_type, {0}, sdesc_file);

        //cluster->start_device({.init_device = false}); // works on 2/3 machines
        cluster->start_device(default_params); // use default params
        tt::llrt::utils::log_current_ai_clk(cluster);

        const std::size_t chunk_size = 1024 * 1024 * 1024;
        const unsigned total_chunks = 1024 * 1024 * 1024 / chunk_size;
        for (int chunk_num = 0; chunk_num < total_chunks; chunk_num++) {
            int start_address = chunk_size * chunk_num;
            log_info(tt::LogTest, "Testing chunk #{}/{}", chunk_num + 1, total_chunks);
            TT_ASSERT(dram_rdwr_check(cluster, start_address, chunk_size));
        }

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
