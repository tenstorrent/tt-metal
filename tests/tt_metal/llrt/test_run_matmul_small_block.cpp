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
#include "test_libs/debug_mailbox.hpp"


using tt::llrt::CircularBufferConfigVec;

bool run_matmul(tt_cluster* cluster, int chip_id, const CoreCoord& core, int num_tiles) {

    std::uint32_t single_tile_size = 2 * 1024;
    std::uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    std::uint32_t dram_buffer_src0_addr = 0;
    int dram_src0_channel_id = 0;
    std::uint32_t dram_buffer_src1_addr = 0;
    int dram_src1_channel_id = 1;

    std::uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
    int dram_dst_channel_id = 0;

    log_info(tt::LogVerif, "num_tiles = {}", num_tiles);
    log_info(tt::LogVerif, "single_tile_size = {} B", single_tile_size);
    log_info(tt::LogVerif, "dram_bufer_size = {} B", dram_buffer_size);

    tt::llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, {core});

    CoreCoord dram_src0_noc_xy = tt::llrt::get_core_for_dram_channel(cluster, dram_src0_channel_id);
    log_info(tt::LogVerif, "dram_src0_noc_xy = {}", dram_src0_noc_xy.str());
    CoreCoord dram_src1_noc_xy = tt::llrt::get_core_for_dram_channel(cluster, dram_src1_channel_id);
    log_info(tt::LogVerif, "dram_src1_noc_xy = {}", dram_src1_noc_xy.str());

    CoreCoord dram_dst_noc_xy = tt::llrt::get_core_for_dram_channel(cluster, dram_dst_channel_id);
    log_info(tt::LogVerif, "dram_dst_noc_xy = {}", dram_dst_noc_xy.str());

    // BufferConfigVec -- common across all kernels, so written once to the core
    CircularBufferConfigVec circular_buffer_config_vec = tt::llrt::create_circular_buffer_config_vector();

    int input_buffer_size_tiles = 2;
    int output_buffer_size_tiles = 2;

    tt::llrt::set_config_for_circular_buffer(circular_buffer_config_vec, 0,  200*1024,  input_buffer_size_tiles  * single_tile_size, input_buffer_size_tiles); // input0 buf
    tt::llrt::set_config_for_circular_buffer(circular_buffer_config_vec, 1,  300*1024,  input_buffer_size_tiles  * single_tile_size, input_buffer_size_tiles); // input1 buf
    tt::llrt::set_config_for_circular_buffer(circular_buffer_config_vec, 16, 400*1024,  output_buffer_size_tiles * single_tile_size, output_buffer_size_tiles); // output buf (output operands start at index 16)
    // buffer_config_vec written in one-shot
    tt::llrt::write_circular_buffer_config_vector_to_core(cluster, chip_id, core, circular_buffer_config_vec);

    // NCRISC kernel arguments to L1 in one-shot
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, core,
        { dram_buffer_src0_addr, (std::uint32_t)dram_src0_noc_xy.x, (std::uint32_t)dram_src0_noc_xy.y,
          dram_buffer_src1_addr, (std::uint32_t)dram_src1_noc_xy.x, (std::uint32_t)dram_src1_noc_xy.y,
        (std::uint32_t)num_tiles },
        NCRISC_L1_ARG_BASE);

    // BRISC kernel arguments to L1 in one-shot
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, core,
        { dram_buffer_dst_addr, (std::uint32_t)dram_dst_noc_xy.x, (std::uint32_t)dram_dst_noc_xy.y, (std::uint32_t)num_tiles},
        BRISC_L1_ARG_BASE);

    // Note: TRISC 0/1/2 kernel args are hard-coded

    TT_ASSERT(dram_buffer_size % sizeof(std::uint32_t) == 0);

    // Write tiles sequentially to DRAM
    std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(dram_buffer_size, 100, tt::tiles_test::get_seed_from_systime());
    cluster->write_dram_vec(src0_vec, tt_target_dram{chip_id, dram_src0_channel_id, 0}, dram_buffer_src0_addr); // write to address
    std::vector<uint32_t> src1_vec = create_random_vector_of_bfloat16(dram_buffer_size, 0, tt::tiles_test::get_seed_from_systime());
    cluster->write_dram_vec(src1_vec, tt_target_dram{chip_id, dram_src1_channel_id, 0}, dram_buffer_src1_addr); // write to address

    tt::llrt::internal_::setup_riscs_on_specified_cores(cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, {core});
    tt::llrt::internal_::run_riscs_on_specified_cores(cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, {core});

    std::vector<std::uint32_t> dst_vec;
    cluster->read_dram_vec(dst_vec, tt_target_dram{chip_id, dram_dst_channel_id, 0}, dram_buffer_dst_addr, dram_buffer_size);

    // sanity checks
    // check that src data has been copied to DRAM correctly
    //cluster->read_dram_vec(dst_vec, tt_target_dram{chip_id, dram_src_channel_id, 0}, dram_buffer_src_addr, dram_buffer_size);
    // check that data has been copied to core's src l1 buffers (l1_buffer_addr_ncrisc)
    //dst_vec = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, l1_buffer_addr_ncrisc, dram_buffer_size);  // read a single uint32_t
    // check that data has been copied to core's dst l1 buffers (l1_buffer_addr_brisc)
    //dst_vec = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, l1_buffer_addr_ncrisc, dram_buffer_size);  // read a single uint32_t

    bool pass = (dst_vec == src1_vec); // src1 is all 0's --> thus the result should be all 0's

#define PRINT_VECTORS
#ifdef PRINT_VECTORS
    print_vec_of_uint32_as_packed_bfloat16(src0_vec, num_tiles);
    print_vec_of_uint32_as_packed_bfloat16(src1_vec, num_tiles);
    print_vec_of_uint32_as_packed_bfloat16(dst_vec, num_tiles);
#endif

    return pass;
}

int main(int argc, char** argv)
{
    bool pass = true;

    const TargetDevice target_type = TargetDevice::Silicon;
    const tt::ARCH arch = tt::ARCH::GRAYSKULL;
    const std::string sdesc_file = get_soc_description_file(arch, target_type);

    try {
        tt_device_params default_params;
        tt_cluster *cluster = new tt_cluster;
        cluster->open_device(arch, target_type, {0}, sdesc_file);
        cluster->start_device(default_params); // use default params
        tt::llrt::utils::log_current_ai_clk(cluster);

        // tt::llrt::print_worker_cores(cluster);

        string op_path = "built_kernels/matmul_small_block";

        pass = tt::llrt::test_load_write_read_risc_binary(cluster, op_path + "/brisc/brisc.hex", 0, {1,1}, 0); // brisc
        pass = pass & tt::llrt::test_load_write_read_risc_binary(cluster, op_path + "/ncrisc/ncrisc.hex", 0, {1,1}, 1); // ncrisc

        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread0/tensix_thread0.hex", 0, {1,1}, 0); // trisc0
        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread1/tensix_thread1.hex", 0, {1,1}, 1); // trisc1
        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread2/tensix_thread2.hex", 0, {1,1}, 2); // trisc2

        int num_tiles = 1;
        if (pass) {
            pass &= run_matmul(cluster, 0, {1, 1}, num_tiles); // must match the value in test_compile_matmul
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
