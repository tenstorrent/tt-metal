// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
// #include <type_traits>
#include <vector>
#include <limits>

namespace tt {
namespace tt_metal {
namespace ccl {

enum EriscDataMoverBufferSharingMode: uint32_t {
    NOT_SHARED = 0,
    ROUND_ROBIN = 1,
    SHARED = 2,
    ROUND_ROBIN_AND_SHARED = 3
};

// TODO: let the kernel runtime args

enum ShardType : uint8_t { Width = 0, Height = 1, Block = 2 };

/*
 * Worker coordinate, used by EDM and some kernels to know which workers to signal
 */
struct WorkerXY {
    uint16_t x;
    uint16_t y;

    WorkerXY(uint16_t x, uint16_t y) : x(x), y(y) {}

    uint32_t to_uint32() const {
        return (y << 16) | x;
    }

    bool operator==(const WorkerXY& rhs) const {
        return x == rhs.x && y == rhs.y;
    }
    bool operator!=(const WorkerXY& rhs) const {
        return !(*this == rhs);
    }
};

static constexpr uint32_t UNINITIALIZED_VALUE_U32 = std::numeric_limits<uint32_t>::max();
static constexpr uint16_t UNINITIALIZED_VALUE_U16 = std::numeric_limits<uint16_t>::max();

template <bool T>
struct ArchDependentTypes;

template <>
struct ArchDependentTypes<true> {
    using workers_list_t = std::vector<ccl::WorkerXY>;
    static const workers_list_t WORKERS_LIST_UNINITIALIZED_VALUE;
};

template <>
struct ArchDependentTypes<false> {
    using workers_list_t = ccl::WorkerXY*;
    static const workers_list_t WORKERS_LIST_UNINITIALIZED_VALUE;
};


template <bool IS_HOST>
struct FullWorkerGridShardAddrGenArgs final {
    typename ArchDependentTypes<IS_HOST>::workers_list_t dest_cores;
    uint32_t tile_size_in_bytes = UNINITIALIZED_VALUE_U32;
    uint32_t shards_start_address = UNINITIALIZED_VALUE_U32;
    uint16_t curr_core_index = UNINITIALIZED_VALUE_U16;
    uint16_t total_num_cores = UNINITIALIZED_VALUE_U16;
    uint16_t curr_shard_tile_x = UNINITIALIZED_VALUE_U16;
    uint16_t curr_shard_tile_y = UNINITIALIZED_VALUE_U16;
    uint16_t curr_tile_index = UNINITIALIZED_VALUE_U16;
    uint16_t curr_shard = UNINITIALIZED_VALUE_U16;
    uint16_t input_shard_num_tiles_x = UNINITIALIZED_VALUE_U16;
    uint16_t input_shard_num_tiles_y = UNINITIALIZED_VALUE_U16;
    uint16_t total_shards_x = UNINITIALIZED_VALUE_U16;
    bool is_clockwise = false;

    inline uint32_t get_expected_num_args() const {
        if constexpr (IS_HOST) {
            return 12 + this->total_num_cores;
        } else {
            return 12 + this->total_num_cores;
        }
    }
};

template <bool IS_HOST>
struct ShardAddrGenArgs final {

    uint32_t shards_start_address = UNINITIALIZED_VALUE_U32;
    uint32_t shard_size_in_bytes = UNINITIALIZED_VALUE_U32;
    uint16_t total_chunks_per_core = UNINITIALIZED_VALUE_U16;

    uint16_t starting_core_index = UNINITIALIZED_VALUE_U16;
    uint16_t starting_chunk_into_shard = UNINITIALIZED_VALUE_U16;

    uint16_t intra_core_stride_in_shards = UNINITIALIZED_VALUE_U16;
    uint16_t contiguous_chunks_before_stride = UNINITIALIZED_VALUE_U16;

    uint16_t num_dest_cores = UNINITIALIZED_VALUE_U16;

    typename ArchDependentTypes<IS_HOST>::workers_list_t dest_cores;
    bool is_clockwise = false;

    inline uint32_t get_expected_num_args() const {
        if constexpr (IS_HOST) {
            return 9 + dest_cores.size();
        } else {
            return 9 + this->num_dest_cores;
        }
    }
};

// uint16_t &curr_shard_tile_x,
// uint16_t &curr_shard_tile_y,
// uint16_t &curr_tile_index,
// uint16_t &curr_shard,
// uint16_t const input_shard_num_tiles_x,
// uint16_t const input_shard_num_tiles_y,
// uint16_t const total_shards_x,
// bool is_clockwise) {

namespace all_gather {
inline void addr_gen_advance_width_sharded(
    // uint16_t& curr_core_chunk_index,
    // uint16_t& curr_worker_index,
    // uint16_t& contiguous_chunk_count,
    // // uint16_t& current_core_chunks_visited,
    // const uint16_t& total_chunks_per_core,
    // const uint16_t& num_dest_cores,
    // const uint16_t& intra_core_stride_in_shards,
    // const uint16_t& contiguous_chunks_before_stride,
    // bool is_clockwise
    uint16_t& curr_core_tile_index,
    uint16_t& curr_worker_index,
    uint16_t& contiguous_tile_count,
    // uint16_t& current_core_chunks_visited,
    const uint16_t& total_chunks_per_core,
    const uint16_t& num_dest_cores,
    const uint16_t& intra_core_stride_in_shards,
    const uint16_t& contiguous_chunks_before_stride,
    bool is_clockwise
) {
    if (is_clockwise) {
        bool do_stride = contiguous_tile_count == contiguous_chunks_before_stride;
        bool stride_induced_chunk_wraparound = (do_stride && curr_core_tile_index < (intra_core_stride_in_shards + contiguous_chunks_before_stride - 1));
        bool do_chunk_wrap = curr_core_tile_index >= total_chunks_per_core || stride_induced_chunk_wraparound;

        // current_core_chunks_visited++;
        if (do_chunk_wrap) {
            bool do_core_wrap = curr_worker_index == 0;
            uint32_t past_end_index = (total_chunks_per_core + curr_core_tile_index + 1 - contiguous_chunks_before_stride);
            uint32_t backward_step_amount = (intra_core_stride_in_shards + contiguous_chunks_before_stride - 1);
            // ASSERT(past_end_index >= backward_step_amount);
            curr_core_tile_index = past_end_index - backward_step_amount;
            // curr_core_tile_index = (total_chunks_per_core + curr_core_tile_index - contiguous_chunks_before_stride) - (intra_core_stride_in_shards + contiguous_chunks_before_stride);
            contiguous_tile_count = 1;
            if (do_core_wrap) {
                curr_worker_index = num_dest_cores - 1;
                // current_core_chunks_visited=0;
            } else {
                curr_worker_index--;
            }
        } else {

            if (do_stride) {
                contiguous_tile_count = 1;
                curr_core_tile_index -= (intra_core_stride_in_shards + contiguous_chunks_before_stride - 1);
            } else {
                contiguous_tile_count++;
                curr_core_tile_index++;
            }
        }

    } else {
        // current_core_chunks_visited++;
        if (contiguous_tile_count == contiguous_chunks_before_stride) {
            contiguous_tile_count = 1;
            // TT_ASSERT(curr_core_chunk_index >= intra_core_stride_in_shards);
            curr_core_tile_index += intra_core_stride_in_shards;
        } else {
            contiguous_tile_count++;
            curr_core_tile_index++;
        }

        bool do_chunk_wrap = curr_core_tile_index >= total_chunks_per_core;
        if (do_chunk_wrap) {
            // current_core_chunks_visited = 0;
            curr_core_tile_index = curr_core_tile_index - total_chunks_per_core;
            curr_worker_index++;
            bool do_core_wrap = curr_worker_index == num_dest_cores;
            if (do_core_wrap) {
                curr_worker_index = 0;
            }
        }
    }
}

inline void full_worker_grid_addr_gen_width_sharded_advance_shard_impl(
    uint16_t &curr_shard_tile_x,
    uint16_t &curr_shard_tile_y,
    uint16_t &curr_tile_index,
    uint16_t &curr_core_index,
    uint16_t const total_num_cores,
    uint16_t const input_shard_num_tiles_x,
    uint16_t const total_shards_x,
    uint16_t const shard_offset,
    bool is_clockwise
) {
    bool wrap_around = is_clockwise ? curr_core_index == 0 : curr_core_index == total_num_cores - 1;
    curr_core_index = wrap_around ?
        (is_clockwise ? total_num_cores - 1 : 0) :
        (is_clockwise ? curr_core_index - 1 : curr_core_index + 1);
    curr_tile_index = input_shard_num_tiles_x * shard_offset;
    curr_shard_tile_x = 0;
    curr_shard_tile_y = 0;
}

// TODO: support cases where the full shard is contiguous (e.g. height sharded)
inline void full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
    uint16_t &curr_shard_tile_x,
    uint16_t &curr_shard_tile_y,
    uint16_t &curr_tile_index,
    uint16_t &curr_core_index,
    uint16_t const total_num_cores,
    uint16_t const input_shard_num_tiles_x,
    uint16_t const input_shard_num_tiles_y,
    uint16_t const total_shards_x,
    uint16_t const shard_offset,
    bool is_clockwise) {

    // Keep it verbose for now. we can reduce to a flat index later
    bool is_last_row = curr_shard_tile_y == input_shard_num_tiles_y - 1;
    if (is_last_row) {
        full_worker_grid_addr_gen_width_sharded_advance_shard_impl(
            curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_core_index, total_num_cores, input_shard_num_tiles_x, total_shards_x, shard_offset, is_clockwise);

    } else {
        curr_tile_index += total_shards_x * input_shard_num_tiles_x - curr_shard_tile_x;
        curr_shard_tile_x = 0;
        curr_shard_tile_y++;
    }
}

inline void full_worker_grid_addr_gen_width_sharded_advance (
    uint16_t &curr_shard_tile_x,
    uint16_t &curr_shard_tile_y,
    uint16_t &curr_tile_index,
    uint16_t &curr_core_index,
    uint16_t const total_num_cores,
    uint16_t const input_shard_num_tiles_x,
    uint16_t const input_shard_num_tiles_y,
    uint16_t const total_shards_x,
    uint16_t const shard_offset,
    bool is_clockwise) {

    // Keep it verbose for now. we can reduce to a flat index later
    bool last_tile_in_row = curr_shard_tile_x == input_shard_num_tiles_x - 1;
    bool last_tile_in_col = curr_shard_tile_y == input_shard_num_tiles_y - 1;
    if (last_tile_in_row && last_tile_in_col) {
        full_worker_grid_addr_gen_width_sharded_advance_shard_impl(
            curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_core_index, total_num_cores, input_shard_num_tiles_x, total_shards_x, shard_offset, is_clockwise);

    } else if (last_tile_in_row) {
        curr_tile_index += total_shards_x * input_shard_num_tiles_x - curr_shard_tile_x;
        curr_shard_tile_x = 0;
        curr_shard_tile_y++;
    } else {
        curr_shard_tile_x++;
        curr_tile_index++;
    }
}


}; // namespace all_gather

}  // namespace ccl
}  // namespace tt_metal
}  // namespace tt
