// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
// #include <type_traits>
#include <vector>
#include <limits>

// #include "tt_dnn/op_library/ccl/ccl_common.hpp"

namespace ccl {

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

template <bool T>
struct ArchDependentTypes;

template <>
struct ArchDependentTypes<true> {
    using workers_list_t = std::vector<ccl::WorkerXY>;
};

template <>
struct ArchDependentTypes<false> {
    using workers_list_t = ccl::WorkerXY*;
};


template <bool IS_HOST>
struct ShardAddrGenArgs final {
    static constexpr uint32_t UNINITIALIZED_VALUE = std::numeric_limits<uint32_t>::max();

    uint32_t shard_size_in_bytes = UNINITIALIZED_VALUE;
    uint32_t chunks_per_core_before_advance = UNINITIALIZED_VALUE;
    uint32_t shards_start_address = UNINITIALIZED_VALUE;

    uint32_t starting_core_index = UNINITIALIZED_VALUE;
    uint32_t starting_chunk_into_shard = UNINITIALIZED_VALUE;

    uint32_t num_dest_cores = UNINITIALIZED_VALUE;
    typename ArchDependentTypes<IS_HOST>::workers_list_t dest_cores;
    bool is_clockwise = false;

    uint32_t get_expected_num_args() const {
        if constexpr (IS_HOST) {
            return 7 + dest_cores.size();
        } else {
            return 7 + this->num_dest_cores;
        }
    }
};


}  // namespace ccl
