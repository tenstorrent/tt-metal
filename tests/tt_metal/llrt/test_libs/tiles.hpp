#pragma once
#include <random>
#include <algorithm>
#include <functional>
#include <tuple>
#include <iostream>
#include <chrono>

#include "tensix.h"
#include "llrt_common/tiles.hpp"

namespace tt {

namespace tiles_test {

template <typename T>
T create_with_size(std::size_t size) {
    return T(size);
}

std::uint32_t get_seed_from_systime() {
    return std::chrono::system_clock::now().time_since_epoch().count();
}

template <typename T>
T create_random_vec(std::size_t size, std::uint32_t seed) {
    std::mt19937 rand_gen(seed);
    T random_vec = create_with_size<T>(size);

    std::generate(random_vec.begin(), random_vec.end(), rand_gen);

    return random_vec;
}

template <typename T>
void assert_is_full_tile(const std::vector<uint32_t>& tile) {
    TT_ASSERT(tile.size() == 32 * 32 * sizeof(T) / sizeof(uint32_t));
}

template <typename T>
int get_address_no_offset_on_l1_for_tile(int tile_index, int num_of_cores) {
    int increments = tile_index / num_of_cores;
    return 32 * 32 * sizeof(T) * increments;
}

template <typename T>
int get_src_address_no_offset_on_dram_for_tile(int tile_index) {
    return get_address_no_offset_on_l1_for_tile<T>(tile_index, 8);
}

// tile_index is the nth tile given the tiles in an ordered list
// tile_position is the position of a value in a given tile, so 0 <= tile_position < 32 * 32
int get_flattened_index_from_tile_index_and_position(const int tile_count_height, const int tile_count_width, const int tile_index, const int tile_position) {
    TT_ASSERT(tile_position >= 0 && tile_position < 32 * 32);

    int tile_height_coord = tile_index / tile_count_width;
    int tile_width_coord = tile_index % tile_count_width;

    int start_of_tile = tile_height_coord * (tile_count_width * 32 * 32) + tile_width_coord * 32;
    // offset_into_tile = width_offset + height_offset
    int offset_into_tile = tile_position % 32 + (tile_position / 32) * (tile_count_width * 32);

    int flattened_index = start_of_tile + offset_into_tile;

    return flattened_index;
}

// D is datatype of container
template <typename D>
std::vector<std::vector<uint32_t>> tilize_ram_data(const int tile_count_height, const int tile_count_width, const std::vector<D>& ram_data) {
    const int total_tiles = tile_count_width * tile_count_height;
    std::size_t tile_size = (32 * 32 * sizeof(D)) / sizeof(uint32_t);
    std::vector<std::vector<uint32_t>> tiles(total_tiles);

    std::function<void(std::vector<uint32_t>&)> resize_to_tile_size = [&](std::vector<uint32_t>& tile) {
        tile.resize(tile_size);
    };

    std::for_each(tiles.begin(), tiles.end(), resize_to_tile_size);

    const int width = tile_count_width * 32;
    const int height = tile_count_height * 32;

    log_info(tt::LogVerif, "width, height: {}, {}", width, height);

    TT_ASSERT(width * height == ram_data.size());

    std::function<void(int, std::vector<uint32_t>&)> stuff_tile = [&](int tile_index, std::vector<uint32_t>& tile) {
        D* tile_data_buffer = static_cast<D*>(static_cast<void*>(tile.data()));
        for (int i = 0; i < 32 * 32; i++, tile_data_buffer++) {
            int ram_data_idx = get_flattened_index_from_tile_index_and_position(tile_count_height, tile_count_width, tile_index, i);
            TT_ASSERT(ram_data_idx < ram_data.size());
            *tile_data_buffer = ram_data[ram_data_idx];
            TT_ASSERT(*(static_cast<D*>(static_cast<void*>(tile.data())) + i) == ram_data[ram_data_idx]);
        }
    };

    for (int tile_idx = 0; tile_idx < tiles.size(); tile_idx++) {
        stuff_tile(tile_idx, tiles[tile_idx]);
    }

    std::for_each(tiles.begin(), tiles.end(), assert_is_full_tile<D>);


    return tiles;
}

template <typename T>
void write_single_tile_to_dram(tt_cluster *cluster, int chip_id, int channel_id, unsigned start_address, std::vector<uint32_t>& tile_data) {
    tt_target_dram dram = {chip_id, channel_id, 0};
    assert_is_full_tile<T>(tile_data);
    cluster->write_dram_vec(tile_data, dram, start_address);
}

template <typename T>
void write_single_tile_to_l1(tt_cluster *cluster, const tt_cxy_pair& core, unsigned start_address, std::vector<uint32_t>& tile_data) {
    assert_is_full_tile<T>(tile_data);
    cluster->write_dram_vec(tile_data, core, start_address);
}

template <typename T>
void write_tilized_data_to_dram_with_offset(tt_cluster *cluster, int chip_id, std::vector<std::vector<uint32_t>>& tile_datas, int channel_offset = 0, unsigned address_offset = 0) {
    for (int tile_idx = 0; tile_idx < tile_datas.size(); tile_idx++) {
        int channel_id = get_src_channel_id_no_offset_from_tile_index(tile_idx) + channel_offset;
        int start_address = get_src_address_no_offset_on_dram_for_tile<T>(tile_idx) + address_offset;
        write_single_tile_to_dram<T>(cluster, chip_id, channel_id, start_address, tile_datas[tile_idx]);
    }
}

template <typename T>
void write_tilized_data_to_l1_of_cores_with_offset(tt_cluster *cluster, int chip_id, const std::vector<CoreCoord>& cores, std::vector<std::vector<uint32_t>>& tile_datas, int core_count_offset = 0, unsigned address_offset = 0) {
    const int num_of_cores = cores.size();
    for (int tile_idx = 0; tile_idx < tile_datas.size(); tile_idx++) {
        int core_index = get_src_core_index_from_tile_index(tile_idx, num_of_cores, core_count_offset);
        const CoreCoord &core = cores[core_index];

        int start_address = get_address_no_offset_on_l1_for_tile<T>(tile_idx, num_of_cores) + address_offset;
        write_single_tile_to_l1<T>(cluster, tt_cxy_pair(chip_id, core), start_address, tile_datas[tile_idx]);
    }
}

template <typename D>
std::vector<uint32_t>& read_single_tile_from_dram(tt_cluster *cluster, int chip_id, int channel_id, unsigned start_address, std::vector<uint32_t>& tile_data) {
    int read_size = 32 * 32 * sizeof(D);
    tt_target_dram dram = {chip_id, channel_id, 0};
    cluster->read_dram_vec(tile_data, dram, start_address, read_size); // read size is in bytes
    return tile_data;
}

template <typename D>
std::vector<uint32_t>& read_single_tile_from_l1(tt_cluster *cluster, const tt_cxy_pair &core, unsigned start_address, std::vector<uint32_t>& tile_data) {
    int read_size = 32 * 32 * sizeof(D);
    cluster->read_dram_vec(tile_data, core, start_address, read_size); // read size is in bytes
    return tile_data;
}

template <typename D>
std::vector<std::vector<uint32_t>> read_tilized_data_from_dram_with_offset(tt_cluster *cluster, int chip_id, int num_tiles, int channel_offset = 0, unsigned address_offset = 0) {
    std::vector<std::vector<uint32_t>> tile_datas(num_tiles);

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int dram_channel_id = (get_src_channel_id_no_offset_from_tile_index(tile_idx) + channel_offset) % 8;
        unsigned start_address = get_src_address_no_offset_on_dram_for_tile<D>(tile_idx) + address_offset;
        read_single_tile_from_dram<D>(cluster, chip_id, dram_channel_id, start_address, tile_datas[tile_idx]);
    }

    return tile_datas;
}

template <typename D>
std::vector<std::vector<uint32_t>> read_tilized_data_from_l1_of_cores_with_offset(tt_cluster *cluster, int chip_id, const std::vector<CoreCoord> &cores, int num_tiles, int core_count_offset = 0, unsigned address_offset = 0) {
    std::vector<std::vector<uint32_t>> tile_datas(num_tiles);

    const int num_of_cores = cores.size();
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int core_index = get_src_core_index_from_tile_index(tile_idx, num_of_cores, core_count_offset);
        const CoreCoord &core = cores[core_index];


        int start_address = get_address_no_offset_on_l1_for_tile<D>(tile_idx, num_of_cores) + address_offset;
        read_single_tile_from_l1<D>(cluster, tt_cxy_pair(chip_id, core), start_address, tile_datas[tile_idx]);
    }

    return tile_datas;
}

bool tile_lists_are_equal(std::vector<std::vector<uint32_t>> expected_tile_datas, std::vector<std::vector<uint32_t>> actual_tile_datas) {
    bool all_are_equal = actual_tile_datas.size() == expected_tile_datas.size();
    for (int tile_idx = 0; tile_idx < expected_tile_datas.size(); tile_idx++) {
        bool current_are_equal = expected_tile_datas[tile_idx] == actual_tile_datas[tile_idx];
        if (not current_are_equal) {
            log_info(tt::LogVerif, "Index {} MISmatch", tile_idx);
        }
        else if (current_are_equal && tile_idx % 10000 == 0) {
            log_info(tt::LogVerif, "Index {}/{} match", tile_idx, expected_tile_datas.size());
        }
        all_are_equal &= current_are_equal;
    }

    return all_are_equal;
}

vector<std::uint16_t> untilize_nchw(const vector<std::uint16_t>& in, const vector<std::uint32_t>& shape) {

    TT_ASSERT(shape[2] % 32 == 0 && shape[3] % 32 == 0);

    std::vector<std::uint16_t> result;
    // Untilize into row major
    int N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    result.resize(N*C*H*W);
    uint32_t linear = 0;
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
    for (int hs32 = 0; hs32 < H; hs32 += 32) // iterate over h with stride 32
    for (int ws32 = 0; ws32 < W; ws32 += 32) // iterate over w with stride 32
    for (int h32 = 0; h32 < 32; h32++) // hs32 + h32 = h
    for (int w32 = 0; w32 < 32; w32++) { // ws32 + w32 = w
        std::uint16_t packed_bfloat16 = in.at(linear);
        auto w = w32 + ws32;
        auto h = h32 + hs32;
        auto offs = w + h*W + c*H*W + n*C*H*W;
        result[offs] = packed_bfloat16;
        linear ++;
    }

    return result;
}

vector<std::uint16_t> tilize_nchw(const vector<std::uint16_t>& in_rowmajor, const vector<std::uint32_t>& shape) {
    int N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    TT_ASSERT(C % 32 == 0 && H % 32 == 0 && W % 32 == 0); // zero-padding is not yet implemented
    std::vector<std::uint16_t> tilized_result;
    tilized_result.reserve(N*C*H*W);
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
    for (int hs32 = 0; hs32 < H; hs32 += 32)
    for (int ws32 = 0; ws32 < W; ws32 += 32)
    for (int h32 = 0; h32 < 32; h32++)
    for (int w32 = 0; w32 < 32; w32++) {
        auto w = w32+ws32;
        auto h = h32+hs32;
        auto offs = w + h*W + c*H*W + n*C*H*W;
        auto val = (w > W || h > H || c > C) ? 0 : in_rowmajor.at(offs);
        tilized_result.push_back(val);
    }
    TT_ASSERT(tilized_result.size() == N*C*H*W);

    return tilized_result;
}

}  // namespace tiles_test

}  // namespace tt
