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
bool dram_rdwr_check(
    tt_cluster *cluster,
    const std::vector<tt_xy_pair> src_data_cores,
    const std::vector<tt_xy_pair> execution_cores,
    const std::vector<tt_xy_pair> dst_data_cores,
    unsigned start_address,
    std::size_t data_size,
    const int tile_count_height,
    const int tile_count_width
) {
    std::vector<T> actual_vec;
    std::size_t vec_size = data_size / sizeof(T);
    std::vector<T> expected_vec = tt::tiles_test::create_random_vec<std::vector<T>>(vec_size, tt::tiles_test::get_seed_from_systime());
    log_info(tt::LogVerif, "Created vector of count (data size) = {} ({} bytes)", vec_size, data_size);

    // tt_target_dram = {device_id, channel, subchannel}
    int chip_id = 0;

    std::vector<std::vector<uint32_t>> expected_tile_datas = tt::tiles_test::tilize_ram_data(tile_count_height, tile_count_width, expected_vec);

    int num_of_src_data_cores = src_data_cores.size();
    int num_of_dst_data_cores = dst_data_cores.size();
    int num_of_working_cores = execution_cores.size();
    tt::llrt::CountOffset src_core_count_offset = 0;
    tt::llrt::CountOffset dst_core_count_offset = 0;

    tt::tiles_test::write_tilized_data_to_l1_of_cores_with_offset<T>(cluster, chip_id, src_data_cores, expected_tile_datas, src_core_count_offset, start_address);

    tt::llrt::assert_reset_for_all_chips(cluster);

    TT_ASSERT(expected_tile_datas.size() % num_of_working_cores == 0);

    for (int tile_idx = 0; tile_idx < expected_tile_datas.size(); tile_idx += num_of_working_cores) {
        unsigned src_address = tt::tiles_test::get_address_no_offset_on_l1_for_tile<T>(tile_idx, num_of_src_data_cores) + start_address;
        unsigned dst_address = tt::tiles_test::get_address_no_offset_on_l1_for_tile<T>(tile_idx, num_of_dst_data_cores) + start_address;

        unsigned src_address_2 = tt::tiles_test::get_address_no_offset_on_l1_for_tile<T>(tile_idx + 1, num_of_src_data_cores) + start_address;
        unsigned dst_address_2 = tt::tiles_test::get_address_no_offset_on_l1_for_tile<T>(tile_idx + 1, num_of_dst_data_cores) + start_address;

        std::vector<uint32_t>& tile = expected_tile_datas[tile_idx];
        uint32_t tile_data_size = tile.size() * sizeof(uint32_t);
        TT_ASSERT(tile_data_size == 32 * 32 * sizeof(T));

        tt::llrt::LoadFirmwareFlag load_firmware_flag = tile_idx == 0;
        std::vector<tt::llrt::RamCopySpec> specs = {
            tt::llrt::create_ram_copy_spec(execution_cores[0], src_data_cores, dst_data_cores, tile_data_size, src_address, dst_address, tile_data_size, tile_idx, src_core_count_offset, dst_core_count_offset, load_firmware_flag),
            tt::llrt::create_ram_copy_spec(execution_cores[1], src_data_cores, dst_data_cores, tile_data_size, src_address_2, dst_address_2, tile_data_size, tile_idx + 1, src_core_count_offset, dst_core_count_offset, load_firmware_flag)
                /*
            tt::llrt::create_dram_copy_spec({7, 11}, dram_src_channel_id_2, dram_dst_channel_id_2, tile_data_size, src_start_address_2, dst_start_address_2, load_firmware_flag),
            tt::llrt::create_dram_copy_spec({8, 4}, dram_src_channel_id_3, dram_dst_channel_id_3, tile_data_size, src_start_address_3, dst_start_address_3, load_firmware_flag),
            tt::llrt::create_dram_copy_spec({9, 9}, dram_src_channel_id_4, dram_dst_channel_id_4, tile_data_size, src_start_address_4, dst_start_address_4, load_firmware_flag)*/
        };
        TT_ASSERT(specs.size() == num_of_working_cores);
        tt::llrt::run_ram_copy_kernel_with_specs(cluster, chip_id, specs, load_firmware_flag);

        if (tile_idx % 1000 == 0) {
            log_info(tt::LogVerif, "Ran kernel for tile {}/{}", tile_idx, expected_tile_datas.size());
        }
    }

    std::vector<std::vector<uint32_t>> actual_tile_datas = tt::tiles_test::read_tilized_data_from_l1_of_cores_with_offset<T>(cluster, chip_id, dst_data_cores, expected_tile_datas.size(), dst_core_count_offset, start_address);

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

        // const std::size_t data_size = 8 * 512 * 1024 / 8;
        const std::size_t tile_size = 32 * 32 * 2;
        int tile_count_width = 24;
        int tile_count_height = 30;
        const std::size_t data_size = tile_size * tile_count_width * tile_count_height;
        int start_address = 300 * 1024;
        tt::llrt::SrcL1Cores src_l1_cores = {{1, 1}, {2, 1}, {2, 2}};
        TT_ASSERT(start_address + data_size < src_l1_cores.size() * 1024 * 1024);
        TT_ASSERT(dram_rdwr_check<tt::tiles_test::tiles_types::TwoBInt>(cluster, src_l1_cores, {{1, 2}, {1, 4}}, {{1, 3}, {3, 1}, {3, 2}, {3, 3}, {3, 4}}, start_address, data_size, tile_count_height, tile_count_width));

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

