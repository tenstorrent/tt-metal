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

bool run_risc_write_speed(tt_cluster *cluster, int chip_id, const tt_xy_pair& core, std::uint32_t buffer_size, std::uint32_t num_repetitions, std::uint32_t transaction_size, const tt_xy_pair& dst_xy) {
    tt::llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(cluster, chip_id, tt::llrt::TensixRiscsOptions::BRISC_NCRISC, {});

    const int ncrisc_id = 1;
    if (!tt::llrt::test_load_write_read_risc_binary(cluster, "built_kernels/risc_write_speed/ncrisc/ncrisc.hex", chip_id, core, ncrisc_id)) {
        return false;
    }

    // Kernel arguments
    std::uint32_t buffer_src_addr = 200 * 1024;
    std::uint32_t buffer_dst_addr = 300 * 1024;
    log_info(tt::LogVerif, "dst_xy = {}", dst_xy.str());

    TT_ASSERT(buffer_size <= 512*1024);
    TT_ASSERT(buffer_size % transaction_size == 0);
   // TT_ASSERT(transaction_size <= 8192);

    std::uint32_t num_transactions = buffer_size / transaction_size;

    // blast kernel arguments to L1 in one-shot
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, core,
        {buffer_src_addr,

        buffer_dst_addr,
        (std::uint32_t)dst_xy.x,
        (std::uint32_t)dst_xy.y,

        transaction_size,
        num_transactions,
        num_repetitions

        }, NCRISC_L1_ARG_BASE);

    std::vector<std::uint32_t> src_vec = tt::tiles_test::create_random_vec<std::vector<std::uint32_t>>(buffer_size/sizeof(std::uint32_t), tt::tiles_test::get_seed_from_systime());
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, core, src_vec, buffer_src_addr);

    const tt::llrt::TensixRiscsOptions riscs_options = tt::llrt::TensixRiscsOptions::BRISC_NCRISC;
    // TIMER START
    auto start = std::chrono::steady_clock::now();
    tt::llrt::internal_::setup_riscs_on_specified_cores(cluster, chip_id, riscs_options, {core});
    tt::llrt::internal_::run_riscs_on_specified_cores(cluster, chip_id, riscs_options, {core});
    auto end = std::chrono::steady_clock::now();
    // TIMER END

    log_info(tt::LogVerif, "ncrisc on core {} finished", core.str());
    std::chrono::duration<double> elapsed_seconds = end-start;
    log_info(tt::LogVerif, "NCRISC time: {}s", elapsed_seconds.count());
    uint64_t total_bytes = (uint64_t)buffer_size * num_repetitions;
    double total_GB = (double)total_bytes / (1024*1024*1024);
    log_info(tt::LogVerif, "Bytes written: {}, GB written: {}", total_bytes, total_GB);
    log_info(tt::LogVerif, "Write speed GB/s: {}", total_GB/elapsed_seconds.count());

    vector<std::uint32_t> dst_vec = tt::llrt::read_hex_vec_from_core(cluster, chip_id, dst_xy, buffer_dst_addr, buffer_size);  // read L1

    return src_vec == dst_vec;
}

int main(int argc, char** argv)
{
    bool pass = true;

    std::vector<std::string> input_args(argv, argv + argc);
    string arch_name = "";
    std::uint32_t buffer_size;
    std::uint32_t num_repetitions;
    std::uint32_t transaction_size;
    std::uint32_t dst_noc_x;
    std::uint32_t dst_noc_y;
    try {
        std::tie(arch_name, input_args) =
            test_args::get_command_option_and_remaining_args(input_args, "--arch", "grayskull");
        std::tie(buffer_size, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--buffer-size", 400*1024);
        std::tie(num_repetitions, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-repetitions", 10000);
        std::tie(transaction_size, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--transaction-size", 512);
        std::tie(dst_noc_x, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--dst-noc-x", 1);
        std::tie(dst_noc_y, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--dst-noc-y", 2);
    } catch (const std::exception& e) {
        log_fatal(tt::LogTest, "Command line arguments found exception", e.what());
    }
    log_info(tt::LogTest, "Test arguments: buffer_size = {}, num_repetitions = {}, transaction_size = {}, dst_noc_x = {}, dst_noc_y = {}", buffer_size, num_repetitions, transaction_size, dst_noc_x, dst_noc_y);

    const TargetDevice target_type = TargetDevice::Silicon;
    const tt::ARCH arch = tt::get_arch_from_string(arch_name);
    const std::string sdesc_file = get_soc_description_file(arch, target_type);

    try {
        tt_device_params default_params;
        tt_cluster *cluster = new tt_cluster;
        cluster->open_device(arch, target_type, {0}, sdesc_file);
        cluster->start_device(default_params); // use default params
        tt::llrt::utils::log_current_ai_clk(cluster);

        pass = run_risc_write_speed(cluster, 0, {1,1}, buffer_size, num_repetitions, transaction_size, {dst_noc_x,dst_noc_y});

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
