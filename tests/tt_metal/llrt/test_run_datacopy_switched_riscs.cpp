#include <chrono>
#include <fstream>
#include <errno.h>
#include <random>
#include <iomanip>

#include "tt_cluster.hpp"
#include "utils.hpp"
#include "common/logger.hpp"
#include "tensix.h"

#include "llrt.hpp"
#include "common/bfloat16.hpp"
#include "test_libs/tiles.hpp"

using tt::llrt::CircularBufferConfigVec;

bool run_data_copy_multi_tile(tt_cluster* cluster, int chip_id, const CoreCoord& core, int num_tiles) {

    std::uint32_t single_tile_size = 2 * 1024;
    std::uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    std::uint32_t dram_buffer_src_addr = 0;
    std::uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
    int dram_src_channel_id = 0;
    int dram_dst_channel_id = 0;
    log_info(tt::LogVerif, "num_tiles = {}", num_tiles);
    log_info(tt::LogVerif, "single_tile_size = {} B", single_tile_size);
    log_info(tt::LogVerif, "dram_bufer_size = {} B", dram_buffer_size);


    tt::llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, {core});

    CoreCoord dram_src_noc_xy = tt::llrt::get_core_for_dram_channel(cluster, dram_src_channel_id);
    log_info(tt::LogVerif, "dram_src_noc_xy = {}", dram_src_noc_xy.str());
    CoreCoord dram_dst_noc_xy = tt::llrt::get_core_for_dram_channel(cluster, dram_dst_channel_id);
    log_info(tt::LogVerif, "dram_dst_noc_xy = {}", dram_dst_noc_xy.str());

    // BufferConfigVec -- common across all kernels, so written once to the core
    CircularBufferConfigVec circular_buffer_config_vec = tt::llrt::create_circular_buffer_config_vector();
    tt::llrt::set_config_for_circular_buffer(circular_buffer_config_vec, 0, 200*1024, 2*single_tile_size, 2); // input buf
    tt::llrt::set_config_for_circular_buffer(circular_buffer_config_vec, 16, 300*1024, 2*single_tile_size, 2); // output buf (starts at index 16)
    // buffer_config_vec written in one-shot
    tt::llrt::write_circular_buffer_config_vector_to_core(cluster, chip_id, core, circular_buffer_config_vec);

    // NCRISC kernel arguments to L1 in one-shot
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, core,
        { dram_buffer_src_addr, (std::uint32_t)dram_src_noc_xy.x, (std::uint32_t)dram_src_noc_xy.y, (std::uint32_t)num_tiles },
        NCRISC_L1_ARG_BASE);

    // BRISC kernel arguments to L1 in one-shot
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, core,
        { dram_buffer_dst_addr, (std::uint32_t)dram_dst_noc_xy.x, (std::uint32_t)dram_dst_noc_xy.y, (std::uint32_t)num_tiles},
        BRISC_L1_ARG_BASE);

    // Note: TRISC 0/1/2 kernel args are hard-coded

    TT_ASSERT(dram_buffer_size % sizeof(std::uint32_t) == 0);

    // Write tiles sequentially to DRAM
    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(dram_buffer_size, 100, tt::tiles_test::get_seed_from_systime());
    cluster->write_dram_vec(src_vec, tt_target_dram{chip_id, dram_src_channel_id, 0}, dram_buffer_src_addr); // write to address

    tt::llrt::internal_::setup_riscs_on_specified_cores(cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, {core});
    tt::llrt::internal_::run_riscs_on_specified_cores(cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, {core});

    std::vector<std::uint32_t> dst_vec;
    cluster->read_dram_vec(dst_vec, tt_target_dram{chip_id, dram_dst_channel_id, 0}, dram_buffer_dst_addr, dram_buffer_size);

    bool pass = (dst_vec == src_vec);

    return pass;
}

int main(int argc, char** argv)
{
    bool pass = true;

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
        cluster->start_device(default_params); // use default params
        tt::llrt::utils::log_current_ai_clk(cluster);

        // tt::llrt::print_worker_cores(cluster);

        string op_path = "built_kernels/datacopy_op_switched_riscs";

        pass = tt::llrt::test_load_write_read_risc_binary(cluster, op_path + "/brisc/brisc.hex", 0, {1,1}, 0); // brisc
        pass = pass & tt::llrt::test_load_write_read_risc_binary(cluster, op_path + "/ncrisc/ncrisc.hex", 0, {1,1}, 1); // ncrisc

        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread0/tensix_thread0.hex", 0, {1,1}, 0); // trisc0
        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread1/tensix_thread1.hex", 0, {1,1}, 1); // trisc1
        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread2/tensix_thread2.hex", 0, {1,1}, 2); // trisc2

        if (pass) {
            pass &= run_data_copy_multi_tile(cluster, 0, {1, 1}, 2048); // must match the value in test_compile_datacopy!
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
