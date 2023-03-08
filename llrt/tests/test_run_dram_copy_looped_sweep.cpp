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

std::mt19937 rand_gen(0);

int tt_rnd_int(int min, int max) {
    tt::log_assert((min <= max), "min is greater than max!");
    std::uniform_int_distribution<int> distrib(min, max);
    return distrib(rand_gen);
}

void tt_rnd_set_seed(int seed) {
    log_info(tt::LogVerif, "Setting Test Seed = {}", (uint32_t)seed);
    rand_gen.seed(seed);
}

std::uint32_t parse_out_dram_buffer_size(int argc, char** argv) {
    TT_ASSERT(argc == 2 || argc == 1);

    if (argc == 1) {
        return 4 * 1024 * 1024;
    }
    else {
        std::uint32_t dram_buffer_size = std::stoi(argv[1]);

        TT_ASSERT(dram_buffer_size >= 512 * 1024);
        TT_ASSERT(dram_buffer_size <= 32 * 1024 * 1024);

        return dram_buffer_size;
    }
}

int main(int argc, char** argv)
{
    bool pass = true;

    const TargetDevice target_type = TargetDevice::Silicon;
    const tt::ARCH arch = tt::ARCH::GRAYSKULL;
    const std::string sdesc_file = get_soc_description_file(arch, target_type);


    tt_rnd_set_seed(std::random_device()());

    try {
        tt_device_params default_params;
        tt_cluster *cluster = new tt_cluster;
        cluster->open_device(arch, target_type, {0}, sdesc_file);
        cluster->start_device(default_params); // use default params
        tt::llrt::utils::log_current_ai_clk(cluster);

        // tt::llrt::print_worker_cores(cluster);

        std::uint32_t dram_buffer_size = parse_out_dram_buffer_size(argc, argv);


        tt::llrt::LoadFirmwareFlag load_firmware_flag = true;
        for (int dram_src_channel_id = 0; dram_src_channel_id < 8; dram_src_channel_id++) {
            for (int dram_dst_channel_id = 0; dram_dst_channel_id < 8; dram_dst_channel_id++) {
                // the first worker core starts at (1,1)
                log_info(tt::LogVerif, "Running dram copy for dram channel pair {} -> {}", dram_src_channel_id, dram_dst_channel_id);

                int init_value_for_buffer = tt_rnd_int(0,100000);
                log_info(tt::LogVerif, "init_value_for_buffer = {}", init_value_for_buffer);
                std::vector<std::uint32_t> src_vec(dram_buffer_size/sizeof(std::uint32_t), init_value_for_buffer);
                cluster->write_dram_vec(src_vec, tt_target_dram{0, dram_src_channel_id, 0}, 0); // write to address

                std::vector<tt::llrt::DramCopySpec> specs = {tt::llrt::create_dram_copy_spec({11,3}, dram_src_channel_id, dram_dst_channel_id, dram_buffer_size, 0, 0, load_firmware_flag)};
                tt::llrt::run_dram_copy_kernel_with_specs(cluster, 0, specs, load_firmware_flag);
                load_firmware_flag = false;

                vector<std::uint32_t> dst_vec;
                cluster->read_dram_vec(dst_vec, tt_target_dram{0, dram_dst_channel_id, 0}, 0, dram_buffer_size); // read size is in bytes

                pass &= src_vec == dst_vec;

                log_info(tt::LogVerif, "Finished dram copy for dram channel pair {} -> {} with pass status: {}", dram_src_channel_id, dram_dst_channel_id, pass);
            }
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
