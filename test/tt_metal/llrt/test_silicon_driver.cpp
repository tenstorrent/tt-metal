#include <chrono>
#include <fstream>
#include <errno.h>
#include <random>

#include "llrt/llrt.hpp"
#include "utils.hpp"
#include "common/logger.hpp"

bool dram_rdwr_check(tt_cluster *cluster) {
    std::vector<uint32_t> tmp;
    //std::vector<uint32_t> vec = {(uint32_t)tt_rnd_int(std::numeric_limits<int>::min(), std::numeric_limits<int>::max())};
    std::vector<uint32_t> vec = {10}; // single int for now

    // tt_target_dram = (device_id, channel, subchannel)
    tt_target_dram dram = {0, 0, 0};
    int address = 1024;

    cluster->write_dram_vec(vec, dram, address); // write to address
    cluster->read_dram_vec(tmp, dram, address, vec.size() * sizeof(uint32_t)); // read size is in bytes

    log_info(tt::LogVerif, "vec size = {}", vec.size());
    log_info(tt::LogVerif, "tmp size = {}", tmp.size());
    log_info(tt::LogVerif, "Wrote vec[0] = {}", vec[0]);
    log_info(tt::LogVerif, "Read tmp[0] = {}", tmp[0]);

    // FIXME: this doesn't work?
    //log_info(tt::LogTest, "dram_rdwr_check wrote {} and read {}, passed in {} attempts\n", vec, tmp, attempt);

    return vec[0] == tmp[0];
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

        TT_ASSERT(dram_rdwr_check(cluster));

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
