#include <chrono>
#include <fstream>
#include <errno.h>
#include <random>

#include "tt_cluster.hpp"
#include "utils.hpp"
#include "common/logger.hpp"
#include "tensix.h"

#include "llrt.hpp"
#include "test_libs/tiles.hpp"

bool run_risc_read_speed(tt_cluster *cluster, int chip_id, const tt_xy_pair& core, std::uint32_t buffer_size, std::uint32_t num_repetitions, std::uint32_t transaction_size, const tt_xy_pair& src_xy) {
    // Only need to load blanks onto BRISCs, not worry about other RISCs
    // because this test only deals with BRISC
    const tt::llrt::TensixRiscsOptions riscs_options = tt::llrt::TensixRiscsOptions::BRISC_ONLY;
    tt::llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(cluster, chip_id, riscs_options, {core});

    const int brisc_id = 0;
    if (!tt::llrt::test_load_write_read_risc_binary(cluster, "built_kernels/risc_read_speed/brisc/brisc.hex", chip_id, core, brisc_id)) {
        return false;
    }

    // Kernel arguments
    std::uint32_t buffer_dst_addr = 200 * 1024;
    std::uint32_t buffer_src_addr = 200 * 1024;
    log_info(tt::LogVerif, "src_xy = {}", src_xy.str());

    TT_ASSERT(buffer_size <= 512*1024);
    //TT_ASSERT(transaction_size <= 8192);
    TT_ASSERT(buffer_size % transaction_size == 0);

    //std::uint32_t num_transactions = 1; // debug
    std::uint32_t num_transactions = buffer_size / transaction_size;

    //std::uint32_t num_repetitions = 10000;
    //std::uint32_t num_repetitions = 1; // debug

    // blast kernel arguments to L1 in one-shot
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, core,
        {buffer_dst_addr,

        buffer_src_addr,
        (std::uint32_t)src_xy.x,
        (std::uint32_t)src_xy.y,

        transaction_size,
        num_transactions,
        num_repetitions

        }, BRISC_L1_ARG_BASE);

    // blast the src buffer to DRAM
    std::vector<std::uint32_t> src_vec = tt::tiles_test::create_random_vec<std::vector<std::uint32_t>>(buffer_size/sizeof(std::uint32_t), tt::tiles_test::get_seed_from_systime());
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, src_xy, src_vec, buffer_src_addr);
    //cluster->write_dram_vec(src_vec, tt_target_dram{chip_id, dram_src_channel_id, 0}, buffer_src_addr); // write to address

    // TIMER START
    auto start = std::chrono::steady_clock::now();
    tt::llrt::internal_::setup_riscs_on_specified_cores(cluster, chip_id, riscs_options, {core});
    tt::llrt::internal_::run_riscs_on_specified_cores(cluster, chip_id, riscs_options, {core});
    auto end = std::chrono::steady_clock::now();
    // TIMER END

    log_info(tt::LogVerif, "brisc on core {} finished", core.str());

    std::chrono::duration<double> elapsed_seconds = end-start;
    log_info(tt::LogVerif, "BRISC time: {}s", elapsed_seconds.count());
    uint64_t total_bytes = (uint64_t)buffer_size * num_repetitions;
    double total_GB = (double)total_bytes / (1024*1024*1024);
    log_info(tt::LogVerif, "Bytes read: {}, GB read: {}", total_bytes, total_GB);
    log_info(tt::LogVerif, "Read speed GB/s: {}", total_GB/elapsed_seconds.count());

    // read the result from the dst L1
    vector<std::uint32_t> dst_vec = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, buffer_dst_addr, buffer_size);  // read L1

    return src_vec == dst_vec;
}

int main(int argc, char** argv)
{
    log_info(tt::LogTest, "num cmd line args: {}",argc);
    TT_ASSERT(argc == 1 || argc == 6);

    std::uint32_t buffer_size;
    std::uint32_t num_repetitions;
    std::uint32_t transaction_size;
    std::uint32_t src_noc_x;
    std::uint32_t src_noc_y;

    if (argc == 1) {
        // defaults
        log_info(tt::LogTest, "Using default test arguments");
        buffer_size = 400 * 1024;
        num_repetitions = 10000;
        transaction_size = 512;
        // remote L1
        src_noc_x = 1;
        src_noc_y = 1;
        // DRAM
        //src_noc_x = 1;
        //src_noc_y = 0;

    } else {
        buffer_size = std::stoi(argv[1]);
        num_repetitions = std::stoi(argv[2]);
        transaction_size = std::stoi(argv[3]);
        src_noc_x = std::stoi(argv[4]);
        src_noc_y = std::stoi(argv[5]);
    }
    log_info(tt::LogTest, "Test arguments: buffer_size = {}, num_repetitions = {}, transaction_size = {}, src_noc_x = {}, src_noc_y = {}", buffer_size, num_repetitions, transaction_size, src_noc_x, src_noc_y);

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

        // the first worker core starts at (1,1)
        pass = run_risc_read_speed(cluster, 0, {2,1}, buffer_size, num_repetitions, transaction_size, {src_noc_x,src_noc_y});

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
