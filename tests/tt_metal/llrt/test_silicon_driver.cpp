#include <chrono>
#include <fstream>
#include <errno.h>
#include <random>

#include "llrt/llrt.hpp"
#include "utils.hpp"
#include "common/logger.hpp"

bool dram_rdwr_check(tt_cluster *cluster) {
    std::vector<uint32_t> tmp;
    std::vector<uint32_t> vec = {10}; // single int for now

    tt_target_dram dram = {0, 0, 0};
    int address = 1024;

    cluster->write_dram_vec(vec, dram, address); // write to address
    cluster->read_dram_vec(tmp, dram, address, vec.size() * sizeof(uint32_t)); // read size is in bytes

    log_info(tt::LogTest, "vec size = {}", vec.size());
    log_info(tt::LogTest, "tmp size = {}", tmp.size());
    log_info(tt::LogTest, "Wrote vec[0] = {}", vec[0]);
    log_info(tt::LogTest, "Read tmp[0] = {}", tmp[0]);

    return vec[0] == tmp[0];
}

int main(int argc, char** argv)
{
    bool pass = true;
    const std::string output_dir = ".";

    std::vector<std::string> input_args(argv, argv + argc);
    string arch_name = "";
    try {
        std::tie(arch_name, input_args) =
            test_args::get_command_option_and_remaining_args(input_args, "--arch", "grayskull");
    } catch (const std::exception& e) {
        log_fatal(tt::LogTest, "Command line arguments found exception", e.what());
    }

    const TargetDevice target_type = TargetDevice::Silicon;
    const tt::ARCH arch = tt::get_arch_from_string(arch_name);
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
