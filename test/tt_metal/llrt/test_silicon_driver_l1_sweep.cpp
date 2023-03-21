#include <chrono>
#include <fstream>
#include <errno.h>
#include <random>
#include <algorithm>

#include "tt_cluster.hpp"
#include "utils.hpp"
#include "common/logger.hpp"

#include "llrt.hpp"
#include "test_libs/tiles.hpp"

using RAMData = std::vector<uint32_t>;

bool l1_rdwr_check(tt_cluster *cluster, unsigned start_address, std::size_t data_size) {
    RAMData actual_vec;
    std::size_t vec_size = data_size / sizeof(uint32_t);
    RAMData expected_vec = tt::tiles_test::create_random_vec<RAMData>(vec_size, tt::tiles_test::get_seed_from_systime());

    bool all_are_equal = true;
    constexpr int chip_id = 0;
    for (const tt::llrt::WorkerCore worker_core : tt::llrt::get_worker_cores_from_cluster(cluster, chip_id)) {
        log_info(tt::LogTest, "Writing to core {}...", worker_core.str());
        cluster->write_dram_vec(expected_vec, worker_core, start_address); // write to address
        log_info(tt::LogTest, "Wrote to core {}.", worker_core.str());
    }

    for (const tt::llrt::WorkerCore worker_core : tt::llrt::get_worker_cores_from_cluster(cluster, chip_id)) {
        cluster->read_dram_vec(actual_vec, worker_core, start_address, data_size); // read size is in bytes
        log_info(tt::LogVerif, "expected vec size = {}", expected_vec.size());
        log_info(tt::LogVerif, "actual vec size   = {}", actual_vec.size());
        bool are_equal = actual_vec == expected_vec;

        all_are_equal &= are_equal;
        if (are_equal){
            log_info(tt::LogVerif, "Core {} has passed", worker_core.str());
        }
        else {
            log_error(tt::LogVerif, "Core {} has not passed", worker_core.str());
        }

        std::fill(actual_vec.begin(), actual_vec.end(), 0);
    }

    return all_are_equal;
}

int main(int argc, char** argv)
{
    bool pass = true;

    const std::string output_dir = ".";

    const TargetDevice target_type = TargetDevice::Silicon;
    const tt::ARCH arch = tt::ARCH::GRAYSKULL;
    const std::string sdesc_file = get_soc_description_file(arch, target_type);


    try {
        tt_device_params default_params;
        tt_cluster *cluster = new tt_cluster;
        cluster->open_device(arch, target_type, {0}, sdesc_file);

        //cluster->start_device({.init_device = false}); // works on 2/3 machines
        cluster->start_device(default_params); // use default params
        tt::llrt::utils::log_current_ai_clk(cluster);

        const std::size_t chunk_size = 1024 * 1024;
        TT_ASSERT(l1_rdwr_check(cluster, 0, chunk_size));

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
