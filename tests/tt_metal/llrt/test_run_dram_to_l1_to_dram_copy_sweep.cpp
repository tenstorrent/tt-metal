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

int main(int argc, char** argv)
{
    bool pass = true;

    const TargetDevice target_type = TargetDevice::Silicon;
    const tt::ARCH arch = tt::ARCH::GRAYSKULL;
    const std::string sdesc_file = get_soc_description_file(arch, target_type);


    try {
        tt_device_params default_params;
        tt_cluster *cluster = new tt_cluster;
        const int chip_id = 0;
        cluster->open_device(arch, target_type, {chip_id}, sdesc_file);
        cluster->start_device(default_params); // use default params
        tt::llrt::utils::log_current_ai_clk(cluster);

        const std::uint32_t total_buffer_size = 512* 1024;

        tt::llrt::LoadFirmwareFlag load_firmware_flag = true;

        std::uint32_t index_ = 0;
        std::function<std::function<bool(void)>(std::uint32_t)> create_is_nth_index_fn = [&index_](std::uint32_t nth_value) {
            index_ = 0;
            return [&index_, nth_value]() {
                return !static_cast<bool>(index_ % nth_value);
            };
        };

        std::vector<CoreCoord> cores = {{1, 1}, {2, 4}, {7, 3}, {10, 9}};

        for (int dram_src_channel_id = 0; dram_src_channel_id < 4; dram_src_channel_id++) {
            for (int dram_dst_channel_id = 4; dram_dst_channel_id < 8; dram_dst_channel_id++) {
                // for (const CoreCoord core : tt::llrt::get_worker_cores_from_cluster(cluster, chip_id)) {
                for (const CoreCoord core : cores) {
                    log_info(tt::LogVerif, "Running dram to l1 copy for dram channel {} -> core {}", dram_src_channel_id, core.str());

                    unsigned total_vec_size = total_buffer_size / sizeof(std::uint32_t);
                    std::vector<std::uint32_t> src_vec = tt::tiles_test::create_random_vec<std::vector<std::uint32_t>>(total_vec_size, tt::tiles_test::get_seed_from_systime());
                    std::uint32_t starting_dram_address = 0;
                    cluster->write_dram_vec(src_vec, tt_target_dram{chip_id, dram_src_channel_id, 0}, starting_dram_address); // write to address

                    std::uint32_t starting_l1_address = 250 * 1024;

                    std::vector<tt::llrt::DramToL1CopySpec> specs = {tt::llrt::create_dram_to_l1_copy_spec(core, dram_src_channel_id, total_buffer_size, starting_dram_address, starting_l1_address, load_firmware_flag)};
                    bool load_blanks = load_firmware_flag;
                    tt::llrt::run_dram_to_l1_copy_kernel_with_specs(cluster, 0, specs, load_blanks);

                    vector<std::uint32_t> dst_vec = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, starting_l1_address, total_buffer_size); // read size is in bytes

                    pass &= src_vec == dst_vec;

                    log_info(tt::LogVerif, "Finished dram to l1 copy for dram channel {} -> core {} with pass status: {}", dram_src_channel_id, core.str(), pass);

                    log_info(tt::LogVerif, "Running l1 to dram copy for core {} -> dram channel {}", core.str(), dram_dst_channel_id);

                    std::uint32_t dst_dram_address = 50 * 1024 * 1024;
                    std::vector<tt::llrt::L1ToDramCopySpec> l1_to_dram_specs = {tt::llrt::create_l1_to_dram_copy_spec(core, dram_dst_channel_id, total_buffer_size, dst_dram_address, starting_l1_address, load_firmware_flag)};
                    tt::llrt::run_l1_to_dram_copy_kernel_with_specs(cluster, 0, l1_to_dram_specs, load_blanks);

                    std::vector<std::uint32_t> dst_vec_dram;
                    cluster->read_dram_vec(dst_vec_dram, tt_target_dram{chip_id, dram_dst_channel_id, 0}, dst_dram_address, total_buffer_size); // read size is in bytes

                    pass &= src_vec == dst_vec_dram;
                    log_info(tt::LogVerif, "Finished l1 to dram copy for core {} -> dram channel {} with pass status: {}", core.str(), dram_src_channel_id, pass);
                }
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
