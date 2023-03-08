#include <chrono>
#include <fstream>
#include <errno.h>
#include <random>

#include "tt_cluster.hpp"
#include "utils.hpp"
#include "common/logger.hpp"
#include "tensix.h"
#include "common/math.hpp"

#include "llrt.hpp"
#include "test_libs/tiles.hpp"

void concat_vector_to_another_inplace(std::vector<uint32_t>& to_extend, std::vector<uint32_t>& data_src) {
    for (uint32_t src_value : data_src) {
        to_extend.push_back(src_value);
    }
}

std::uint32_t dram_channel_count_from_count_as_bits(std::uint32_t dram_channel_count_as_bits) {
    TT_ASSERT(dram_channel_count_as_bits <= 3);
    return tt::positive_pow_of_2(dram_channel_count_as_bits);
}

std::vector<uint32_t> read_banked_vector(
    tt_cluster* cluster,
    int chip_id,
    std::uint32_t starting_buffer_dst_addr,
    std::uint32_t num_transactions,
    std::uint32_t transaction_size,
    std::uint32_t dram_channel_count
) {
    std::uint32_t buffer_size = num_transactions * transaction_size;
    std::vector<uint32_t> result;
    for (std::uint32_t transaction = 0; transaction < num_transactions; transaction++) {
        std::uint32_t txn_dst_addr = (transaction / dram_channel_count) * transaction_size + starting_buffer_dst_addr;
        std::uint32_t dram_channel_id = (transaction + 1) % dram_channel_count;
        std::vector<uint32_t> txn_data;
        cluster->read_dram_vec(txn_data, tt_target_dram{chip_id, dram_channel_id, 0}, txn_dst_addr, transaction_size);

        concat_vector_to_another_inplace(result, txn_data);
    }

    return result;
}

void write_banked_vector(
    tt_cluster* cluster,
    int chip_id,
    std::uint32_t starting_buffer_dst_addr,
    std::uint32_t num_transactions,
    std::uint32_t transaction_size,
    const std::vector<uint32_t>& data,
    std::uint32_t dram_channel_count
) {
    std::uint32_t buffer_size = num_transactions * transaction_size;
    for (std::uint32_t transaction = 0; transaction < num_transactions; transaction++) {
        std::uint32_t txn_dst_addr = (transaction / dram_channel_count) * transaction_size + starting_buffer_dst_addr;
        std::uint32_t dram_channel_id = transaction % dram_channel_count;
        std::vector<uint32_t> txn_data(data.begin() + transaction * transaction_size / sizeof(uint32_t),
                                       data.begin() + (transaction + 1) * transaction_size / sizeof(uint32_t));
        if (transaction == num_transactions - 1) {
            TT_ASSERT(data.begin() + (transaction + 1) * transaction_size / sizeof(uint32_t) == data.end());
        }
        cluster->write_dram_vec(txn_data, tt_target_dram{chip_id, dram_channel_id, 0}, txn_dst_addr);
    }
}

std::vector<std::vector<uint32_t>> create_read_args_for_cores(
    const std::vector<uint32_t>& dram_channel_cores,
    const std::uint32_t buffer_src_addr,
    const std::uint32_t buffer_dst_addr,
    const std::uint32_t transaction_size,
    const std::uint32_t num_transactions,
    const std::uint32_t num_repetitions,
    const std::vector<tt_xy_pair>& cores,
    const std::uint32_t dram_channel_count_as_bits
) {
    std::vector<std::vector<uint32_t>> read_args_for_cores;

    std::uint32_t txns_per_core = num_transactions / cores.size();

    for (std::uint32_t core_index = 0; core_index < cores.size(); core_index++) {
        std::uint32_t starting_txn_index = core_index * txns_per_core;

        std::vector<std::uint32_t> read_args = {dram_channel_count_as_bits, buffer_src_addr};
        read_args.insert(read_args.end(), dram_channel_cores.begin(), dram_channel_cores.end());
        read_args.insert(read_args.end(), {buffer_dst_addr, transaction_size, starting_txn_index, txns_per_core, num_repetitions});

        read_args_for_cores.push_back(read_args);
    }

    return read_args_for_cores;
}

std::vector<std::vector<uint32_t>> create_write_args_for_cores(
    const std::vector<uint32_t>& dram_channel_cores,
    const std::uint32_t buffer_src_addr,
    const std::uint32_t buffer_dst_addr,
    const std::uint32_t transaction_size,
    const std::uint32_t num_transactions,
    const std::uint32_t num_repetitions,
    const std::vector<tt_xy_pair>& cores,
    const std::uint32_t dram_channel_count_as_bits
) {
    std::vector<std::vector<uint32_t>> write_args_for_cores;

    std::uint32_t txns_per_core = num_transactions / cores.size();

    for (std::uint32_t core_index = 0; core_index < cores.size(); core_index++) {
        std::uint32_t starting_txn_index = core_index * txns_per_core;

        std::vector<std::uint32_t> write_args = {dram_channel_count_as_bits, buffer_src_addr, buffer_dst_addr};
        write_args.insert(write_args.end(), dram_channel_cores.begin(), dram_channel_cores.end());
        write_args.insert(write_args.end(), {transaction_size, starting_txn_index, txns_per_core, num_repetitions});

        write_args_for_cores.push_back(write_args);
    }

    return write_args_for_cores;
}

bool run_risc_rw_speed_dram_banked(tt_cluster *cluster, int chip_id, const std::vector<tt_xy_pair>& cores, std::uint32_t buffer_size, std::uint32_t num_repetitions, std::uint32_t transaction_size, std::uint32_t dram_channel_count_as_bits) {
    TT_ASSERT(buffer_size % cores.size() == 0);
    std::uint32_t buffer_size_per_core = buffer_size / cores.size();

    TT_ASSERT(buffer_size_per_core <= 800*1024);
    TT_ASSERT(transaction_size <= 8192);

    // Kernel arguments
    std::uint32_t buffer_src_addr = 200 * 1024;
    std::uint32_t buffer_dst_addr = buffer_src_addr;

    TT_ASSERT(buffer_size % transaction_size == 0);
    std::uint32_t num_transactions = buffer_size / transaction_size;
    constexpr std::uint32_t starting_txn_index = 0;
    uint64_t total_bytes = (uint64_t)buffer_size * num_repetitions;
    double total_GB = (double)total_bytes / (1024*1024*1024);

    TT_ASSERT(num_transactions % cores.size() == 0);

    std::string cores_desc = "Cores to use: ";
    for (const tt_xy_pair& core : cores) {
        cores_desc += core.str() + " ";
    }
    log_info(tt::LogVerif, "{}", cores_desc);

    // DRAM channel cores in argument form
    std::vector<uint32_t> dram_channel_cores;
    std::uint32_t dram_channel_count = dram_channel_count_from_count_as_bits(dram_channel_count_as_bits);
    for (int channel_id = 0; channel_id < dram_channel_count; channel_id++) {
        tt_xy_pair dram_src_noc_xy = cluster->get_soc_desc(chip_id).get_core_for_dram_channel(channel_id, /*sub*/0);
        dram_channel_cores.push_back(static_cast<uint32_t>(dram_src_noc_xy.x));
        dram_channel_cores.push_back(static_cast<uint32_t>(dram_src_noc_xy.y));
    }

    std::vector<std::uint32_t> src_vec = tt::tiles_test::create_random_vec<std::vector<std::uint32_t>>(buffer_size/sizeof(std::uint32_t), tt::tiles_test::get_seed_from_systime());

    // Read
    tt::llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(cluster, chip_id, tt::llrt::TensixRiscsOptions::BRISC_NCRISC, cores);

    const int brisc_id = 0;
    for (std::uint32_t core_index = 0; core_index < cores.size(); core_index++) {
        if (!tt::llrt::test_load_write_read_risc_binary(cluster, "built_kernels/risc_rw_speed_banked_dram/brisc/brisc.hex", chip_id, cores[core_index], brisc_id)) {
            return false;
        }
    }

    std::vector<std::vector<uint32_t>> read_args_for_cores = create_read_args_for_cores(dram_channel_cores, buffer_src_addr, buffer_dst_addr, transaction_size, num_transactions, num_repetitions, cores, dram_channel_count_as_bits);

    for (std::uint32_t core_index = 0; core_index < cores.size(); core_index++) {
        tt::llrt::write_hex_vec_to_core(cluster, chip_id, cores[core_index], read_args_for_cores[core_index], BRISC_L1_ARG_BASE);
    }

    write_banked_vector(cluster, chip_id, buffer_src_addr, num_transactions, transaction_size, src_vec, dram_channel_count);

    log_info(tt::LogVerif, "");
    log_info(tt::LogVerif, "Starting read speed test");
    const tt::llrt::TensixRiscsOptions read_riscs_options = tt::llrt::TensixRiscsOptions::BRISC_ONLY;
    // TIMER START
    std::chrono::time_point read_start = std::chrono::steady_clock::now();
    tt::llrt::internal_::setup_riscs_on_specified_cores(cluster, chip_id, read_riscs_options, cores);
    tt::llrt::internal_::run_riscs_on_specified_cores(cluster, chip_id, read_riscs_options, cores);
    std::chrono::time_point read_end = std::chrono::steady_clock::now();
    // TIMER END

    const std::chrono::duration<double> read_elapsed_seconds = read_end - read_start;
    log_info(tt::LogVerif, "BRISC time: {}s", read_elapsed_seconds.count());
    log_info(tt::LogVerif, "Bytes read: {}, GB read: {}", total_bytes, total_GB);
    log_info(tt::LogVerif, "Read speed GB/s: {}", total_GB/read_elapsed_seconds.count());
    log_info(tt::LogVerif, "Done read speed test");

    // Write
    // Write blank to all since we're using only NCRISCs
    tt::llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(cluster, chip_id, tt::llrt::TensixRiscsOptions::BRISC_NCRISC, {});

    const int ncrisc_id = 1;
    for (std::uint32_t core_index = 0; core_index < cores.size(); core_index++) {
        if (!tt::llrt::test_load_write_read_risc_binary(cluster, "built_kernels/risc_rw_speed_banked_dram/ncrisc/ncrisc.hex", chip_id, cores[core_index], ncrisc_id)) {
            return false;
        }
    }

    // blast kernel arguments to L1 in one-shot
    std::vector<std::vector<uint32_t>> write_args_for_cores = create_write_args_for_cores(dram_channel_cores, buffer_src_addr, buffer_dst_addr, transaction_size, num_transactions, num_repetitions, cores, dram_channel_count_as_bits);

    for (std::uint32_t core_index = 0; core_index < cores.size(); core_index++) {
        tt::llrt::write_hex_vec_to_core(cluster, chip_id, cores[core_index], write_args_for_cores[core_index], NCRISC_L1_ARG_BASE);
    }

    log_info(tt::LogVerif, "");
    log_info(tt::LogVerif, "Starting write speed test");

    const tt::llrt::TensixRiscsOptions riscs_options = tt::llrt::TensixRiscsOptions::BRISC_NCRISC;
    // TIMER START
    std::chrono::time_point start = std::chrono::steady_clock::now();
    tt::llrt::internal_::setup_riscs_on_specified_cores(cluster, chip_id, riscs_options, cores);
    tt::llrt::internal_::run_riscs_on_specified_cores(cluster, chip_id, riscs_options, cores);
    std::chrono::time_point end = std::chrono::steady_clock::now();
    // TIMER END

    std::chrono::duration<double> elapsed_seconds = end-start;
    log_info(tt::LogVerif, "NCRISC time: {}s", elapsed_seconds.count());
    log_info(tt::LogVerif, "Bytes written: {}, GB written: {}", total_bytes, total_GB);
    log_info(tt::LogVerif, "Write speed GB/s: {}", total_GB/elapsed_seconds.count());

    // Need to write a sharded read from DRAM
    vector<std::uint32_t> dst_vec = read_banked_vector(cluster, chip_id, buffer_dst_addr, num_transactions, transaction_size, dram_channel_count);

    TT_ASSERT(dst_vec.size() == num_transactions * transaction_size / sizeof(uint32_t));

    return src_vec == dst_vec;
}

int main(int argc, char** argv)
{
    log_info(tt::LogTest, "num cmd line args: {}",argc);
    TT_ASSERT(argc == 1 || (argc >= 7 && (argc % 2 == 1)));

    std::uint32_t buffer_size;
    std::uint32_t num_repetitions;
    std::uint32_t transaction_size;
    std::vector<tt_xy_pair> cores;
    std::uint32_t dram_channel_count_as_bits;

    if (argc == 1) {
        // defaults
        log_info(tt::LogTest, "Using default test arguments");
        buffer_size = 400 * 1024;
        num_repetitions = 10000;
        transaction_size = 512;
        cores = {{5, 4}};
        dram_channel_count_as_bits = 3; // 8 dram channels
    } else {
        buffer_size = std::stoi(argv[1]);
        num_repetitions = std::stoi(argv[2]);
        transaction_size = std::stoi(argv[3]);
        dram_channel_count_as_bits = std::stoi(argv[4]);

        for (int arg_index = 5; arg_index < argc; arg_index += 2) {
            int core_x_index = arg_index;
            int core_y_index = arg_index + 1;

            std::size_t core_x = std::stoi(argv[core_x_index]);
            std::size_t core_y = std::stoi(argv[core_y_index]);

            cores.push_back({core_x, core_y});
        }
    }
    log_info(tt::LogTest, "Test arguments: buffer_size = {}, num_repetitions = {}, transaction_size = {}", buffer_size, num_repetitions, transaction_size);

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

        pass = run_risc_rw_speed_dram_banked(cluster, 0, cores, buffer_size, num_repetitions, transaction_size, dram_channel_count_as_bits);

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
