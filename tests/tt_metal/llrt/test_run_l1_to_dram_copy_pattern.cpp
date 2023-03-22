#include <cmath>
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
#include "test_libs/conv_pattern.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"

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
        tt::llrt::LoadFirmwareFlag load_firmware_flag = true;
        tt_xy_pair core = {11, 3};
        int dram_channel_id = 0;
        std::uint32_t starting_dram_address = 0;


        SHAPE shape = {1, 16, 32, 32};
        tt::deprecated::Tensor<std::uint32_t> tensor = tt::deprecated::initialize_tensor<std::uint32_t>(shape, tt::deprecated::Initialize::INCREMENT);
        auto tensor_p = tt::deprecated::permute(tensor, {0, 2, 3, 1}); // NHWC

        auto src_vec = tensor_p.get_values();
        cluster->write_dram_vec(src_vec, tt_target_dram{chip_id, dram_channel_id, 0}, starting_dram_address); // write to address

        std::uint32_t total_buffer_size = src_vec.size() * sizeof(std::uint32_t);
        std::uint32_t starting_l1_address = 250 * 1024;
        std::uint32_t dst_dram_address = total_buffer_size;

        std::vector<tt::llrt::DramToL1CopySpec> specs = {tt::llrt::create_dram_to_l1_copy_spec(core, dram_channel_id, total_buffer_size, starting_dram_address, starting_l1_address, load_firmware_flag)};
        bool load_blanks = load_firmware_flag;
        tt::llrt::run_dram_to_l1_copy_kernel_with_specs(cluster, chip_id, specs, load_blanks);

        std::vector<tt::llrt::L1ToDramCopySpec> l1_to_dram_specs = {tt::llrt::create_l1_to_dram_copy_spec(core, dram_channel_id, total_buffer_size, dst_dram_address, starting_l1_address, load_firmware_flag)};
        tt::llrt::run_l1_to_dram_copy_kernel_with_specs(cluster, chip_id, l1_to_dram_specs, load_blanks);

        std::vector<std::uint32_t> dst_vec_dram;
        cluster->read_dram_vec(dst_vec_dram, tt_target_dram{chip_id, dram_channel_id, 0}, dst_dram_address, total_buffer_size); // read size is in bytes
        pass &= src_vec == dst_vec_dram;

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
