#include <chrono>
#include <fstream>
#include <errno.h>
#include <random>
#include <algorithm>
#include <functional>
#include <boost/range/adaptor/indexed.hpp>

#include "tt_cluster.hpp"
#include "utils.hpp"
#include "common/logger.hpp"
#include "llrt.hpp"
#include "test_libs/tiles.hpp"

template <typename T>
bool dram_rdwr_check(tt_cluster *cluster, unsigned start_address, std::size_t data_size, const int tile_count_height, const int tile_count_width) {
    std::vector<T> actual_vec;
    std::size_t vec_size = data_size / sizeof(T);
    std::vector<T> expected_vec = tt::tiles_test::create_random_vec<std::vector<T>>(vec_size, tt::tiles_test::get_seed_from_systime());
    log_info(tt::LogVerif, "Created vector of count (data size) = {} ({} bytes)", vec_size, data_size);

    int chip_id = 0;

    // tt_target_dram = {device_id, channel, subchannel}

    std::vector<std::vector<uint32_t>> expected_tile_datas = tt::tiles_test::tilize_ram_data(tile_count_height, tile_count_width, expected_vec);

    tt::tiles_test::write_tilized_data_to_dram_with_offset<T>(cluster, chip_id, expected_tile_datas);

    std::vector<std::vector<uint32_t>> actual_tile_datas = tt::tiles_test::read_tilized_data_from_dram_with_offset<T>(cluster, chip_id, expected_tile_datas.size());

    return tt::tiles_test::tile_lists_are_equal(expected_tile_datas, actual_tile_datas);
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

        const std::size_t data_size = 1024 * 1024 * 1024;
        int start_address = 0;
        int tile_count_width = 1024;
        int tile_count_height = 512;
        TT_ASSERT(dram_rdwr_check<tt::tiles_test::tiles_types::TwoBInt>(cluster, start_address, data_size, tile_count_height, tile_count_width));

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

