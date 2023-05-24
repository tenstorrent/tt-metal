#include <chrono>
#include <fstream>
#include <errno.h>
#include <random>
#include <iomanip>

#include "tt_cluster.hpp"
#include "utils.hpp"
#include "common/logger.hpp"
#include "tensix.h"
// #include "tt_gdb/tt_gdb.hpp"

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

    // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
    tt::llrt::set_config_for_circular_buffer(circular_buffer_config_vec, 0, 200*1024, 8*single_tile_size, 8);
    // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
    tt::llrt::set_config_for_circular_buffer(circular_buffer_config_vec, 16, 300*1024, 1*single_tile_size, 1);

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

    // sanity checks
    // check that src data has been copied to DRAM correctly
    //cluster->read_dram_vec(dst_vec, tt_target_dram{chip_id, dram_src_channel_id, 0}, dram_buffer_src_addr, dram_buffer_size);
    // check that data has been copied to core's src l1 buffers (l1_buffer_addr_ncrisc)
    //dst_vec = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, l1_buffer_addr_ncrisc, dram_buffer_size);  // read a single uint32_t
    // check that data has been copied to core's dst l1 buffers (l1_buffer_addr_brisc)
    //dst_vec = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, l1_buffer_addr_ncrisc, dram_buffer_size);  // read a single uint32_t
    // for(int i = 0; i <src_vec.size(); i++) {
    //     std::cout<<src_vec[i]<<", "<<op_vec[i]<<std::endl;
    // }

    // debug
    //auto op_vec = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, 200*1024, dram_buffer_size);
    //std::cout<<"Operand buffer correct = "<<(src_vec == op_vec)<<std::endl;
    //for(int i = 0; i <src_vec.size(); i++) {
   //     std::cout<<i<<": "<<src_vec[i]<<", "<<dst_vec[i]<<std::endl;
    //}

    //print_vec_of_uint32_as_packed_bfloat16(src_vec, num_tiles);
    //print_vec_of_uint32_as_packed_bfloat16(dst_vec, num_tiles);

    /*
    size_t tile_size = 512; // a tile is 1024 elements: this is 512 uint32_t, and 1024 bfloat16

    // copy each tile (with size of tile_size) to a separate vector
    vector<vector<uint32_t>> src_tiles(num_tiles);
    vector<vector<uint32_t>> dst_tiles(num_tiles);
    for (int i = 0; i < num_tiles; i++) {
        src_tiles[i].insert(src_tiles[i].begin(), src_vec.begin() + i*tile_size, src_vec.begin() + (i+1)*tile_size);
        dst_tiles[i].insert(dst_tiles[i].begin(), dst_vec.begin() + i*tile_size, dst_vec.begin() + (i+1)*tile_size);
    }

    // compare each src and dst tile and print the tiles
    for (int i = 0; i < num_tiles; i++) {
      //  print_vec_of_uint32_as_packed_bfloat16(src_tiles[i], 1, "src", i);
      //  print_vec_of_uint32_as_packed_bfloat16(dst_tiles[i], 1, "dst", i);
        std::cout<<"Tile "<<i<<", correct = "<<(src_tiles[i] == dst_tiles[i])<<std::endl;
    }
    */

    bool pass = (dst_vec == src_vec);

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

        string op = "datacopy_op";

        int chip_id = 0;
        const CoreCoord core = {1, 1};

        pass = tt::llrt::test_load_write_read_risc_binary(cluster, op + "/brisc/brisc.hex", chip_id, core, 0); // brisc
        pass = pass & tt::llrt::test_load_write_read_risc_binary(cluster, op + "/ncrisc/ncrisc.hex", chip_id, core, 1); // ncrisc

        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op + "/tensix_thread0/tensix_thread0.hex", chip_id, core, 0); // trisc0
        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op + "/tensix_thread1/tensix_thread1.hex", chip_id, core, 1); // trisc1
        pass = pass & tt::llrt::test_load_write_read_trisc_binary(cluster, op + "/tensix_thread2/tensix_thread2.hex", chip_id, core, 2); // trisc2

        if (pass) {
            const vector<CoreCoord> cores = {core};
            const vector<string> ops = {op};

            // tt_gdb::tt_gdb(cluster, chip_id, cores, ops);
            pass &= run_data_copy_multi_tile(cluster, chip_id, core, 2048); // must match the value in test_compile_datacopy!
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
