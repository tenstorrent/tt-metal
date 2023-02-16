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

    std::uint32_t src_address_offset = 0;
    std::uint32_t dst_address_offset = data_size;
    std::uint32_t channel_offset = 0;

    int NUM_OF_SPECS = 4;

    tt::llrt::assert_reset_for_all_chips(cluster);

    for (int tile_idx = 0; tile_idx < expected_tile_datas.size(); tile_idx += NUM_OF_SPECS) {
        std::vector<uint32_t>& tile = expected_tile_datas[tile_idx];
        uint32_t tile_data_size = tile.size() * sizeof(uint32_t);
        TT_ASSERT(tile_data_size == 32 * 32 * sizeof(T));
        tt::llrt::SrcChannelId dram_src_channel_id = tt::tiles_test::get_src_channel_id_no_offset_from_tile_index(tile_idx);
        tt::llrt::DstChannelId dram_dst_channel_id = tt::tiles_test::get_src_channel_id_no_offset_from_tile_index(tile_idx) + channel_offset;
        tt::llrt::DramSrcAddr src_start_address = tt::tiles_test::get_src_address_no_offset_on_dram_for_tile<T>(tile_idx) + src_address_offset;
        tt::llrt::DramDstAddr dst_start_address = tt::tiles_test::get_src_address_no_offset_on_dram_for_tile<T>(tile_idx) + dst_address_offset;


        // 2nd tile
        tt::llrt::SrcChannelId dram_src_channel_id_2 = tt::tiles_test::get_src_channel_id_no_offset_from_tile_index(tile_idx + 1);
        tt::llrt::DstChannelId dram_dst_channel_id_2 = tt::tiles_test::get_src_channel_id_no_offset_from_tile_index(tile_idx + 1) + channel_offset;
        tt::llrt::DramSrcAddr src_start_address_2 = tt::tiles_test::get_src_address_no_offset_on_dram_for_tile<T>(tile_idx+ 1) + src_address_offset;
        tt::llrt::DramDstAddr dst_start_address_2 = tt::tiles_test::get_src_address_no_offset_on_dram_for_tile<T>(tile_idx + 1) + dst_address_offset;

        // 2nd tile
        tt::llrt::SrcChannelId dram_src_channel_id_3 = tt::tiles_test::get_src_channel_id_no_offset_from_tile_index(tile_idx + 2);
        tt::llrt::DstChannelId dram_dst_channel_id_3 = tt::tiles_test::get_src_channel_id_no_offset_from_tile_index(tile_idx + 2) + channel_offset;
        tt::llrt::DramSrcAddr src_start_address_3 = tt::tiles_test::get_src_address_no_offset_on_dram_for_tile<T>(tile_idx+ 2) + src_address_offset;
        tt::llrt::DramDstAddr dst_start_address_3 = tt::tiles_test::get_src_address_no_offset_on_dram_for_tile<T>(tile_idx + 2) + dst_address_offset;

        // 2nd tile
        tt::llrt::SrcChannelId dram_src_channel_id_4 = tt::tiles_test::get_src_channel_id_no_offset_from_tile_index(tile_idx + 3);
        tt::llrt::DstChannelId dram_dst_channel_id_4 = tt::tiles_test::get_src_channel_id_no_offset_from_tile_index(tile_idx + 3) + channel_offset;
        tt::llrt::DramSrcAddr src_start_address_4 = tt::tiles_test::get_src_address_no_offset_on_dram_for_tile<T>(tile_idx+ 3) + src_address_offset;
        tt::llrt::DramDstAddr dst_start_address_4 = tt::tiles_test::get_src_address_no_offset_on_dram_for_tile<T>(tile_idx + 3) + dst_address_offset;

        tt::llrt::LoadFirmwareFlag load_firmware_flag = tile_idx == 0;
        std::vector<tt::llrt::DramCopySpec> specs = {
            tt::llrt::create_dram_copy_spec({3, 2}, dram_src_channel_id, dram_dst_channel_id, tile_data_size, src_start_address, dst_start_address, load_firmware_flag),
            tt::llrt::create_dram_copy_spec({7, 11}, dram_src_channel_id_2, dram_dst_channel_id_2, tile_data_size, src_start_address_2, dst_start_address_2, load_firmware_flag),
            tt::llrt::create_dram_copy_spec({8, 4}, dram_src_channel_id_3, dram_dst_channel_id_3, tile_data_size, src_start_address_3, dst_start_address_3, load_firmware_flag),
            tt::llrt::create_dram_copy_spec({9, 9}, dram_src_channel_id_4, dram_dst_channel_id_4, tile_data_size, src_start_address_4, dst_start_address_4, load_firmware_flag)
            /*
            tt::llrt::create_dram_copy_spec({1, 1}, dram_src_channel_id, dram_dst_channel_id, tile_data_size, src_start_address, dst_start_address),
            tt::llrt::create_dram_copy_spec({1, 2}, dram_src_channel_id_2, dram_dst_channel_id_2, tile_data_size, src_start_address_2, dst_start_address_2),
            tt::llrt::create_dram_copy_spec({1, 3}, dram_src_channel_id_3, dram_dst_channel_id_3, tile_data_size, src_start_address_3, dst_start_address_3),
            tt::llrt::create_dram_copy_spec({1, 4}, dram_src_channel_id_4, dram_dst_channel_id_4, tile_data_size, src_start_address_4, dst_start_address_4)*/
        };
        tt::llrt::run_dram_copy_kernel_with_specs(cluster, chip_id, specs, load_firmware_flag);

        if (tile_idx % 1000 == 0) {
            log_info(tt::LogVerif, "Values for: {}: {}, {}, {}", tile_idx, tile[0], tile[1], tile[2]);
            log_info(tt::LogVerif, "Ran kernel for tile {}/{}", tile_idx, expected_tile_datas.size());
        }
    }

    std::vector<std::vector<uint32_t>> actual_tile_datas = tt::tiles_test::read_tilized_data_from_dram_with_offset<T>(cluster, chip_id, expected_tile_datas.size(), channel_offset, dst_address_offset);

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

        const std::size_t data_size = 256 * 1024 * 1024;
        int start_address = 0;
        int tile_count_width = 256;
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

