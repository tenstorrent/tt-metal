#pragma once

#include "dataflow_api.h"
#include "hostdevcommon/kernel_structs.h"

namespace views {

template <size_t Accessors, size_t Offset = 0, uint32_t DefaultNumTilesPerCycle = 1>
class AccessorView {
    static constexpr uint32_t cb_index = get_compile_time_arg_val(Offset);
    static constexpr uint32_t crta_offset = get_compile_time_arg_val(Offset + 1);
    static constexpr uint32_t tile_size = get_tile_size(cb_index);

    using args_type = TensorAccessorArgs<Offset + 2, crta_offset>;

    static constexpr args_type args{};

    __attribute__((always_inline)) static auto get_bank_base_address(int arg_idx) {
        return get_arg_val<uint32_t>(arg_idx);
    }

    decltype(TensorAccessor(args, get_bank_base_address(0), tile_size)) head;
    AccessorView<Accessors - 1, args_type::next_compile_time_args_offset(), DefaultNumTilesPerCycle> tail;

public:
    __attribute__((always_inline)) AccessorView(int arg_idx) :
        head(args, get_bank_base_address(arg_idx), tile_size), tail(arg_idx + 1) {}

    // reader members

    __attribute__((always_inline)) auto reserve_back(uint32_t num_tiles_per_cycle) const noexcept {
        cb_reserve_back(cb_index, num_tiles_per_cycle);

        // recurse for remaining accessors
        if constexpr (Accessors > 1) {
            this->tail.reserve_back(num_tiles_per_cycle);
        }
    }

    __attribute__((always_inline)) auto read_tile_block(
        uint32_t start_id, uint32_t num_tiles_per_cycle) const noexcept {
        auto write_ptr = get_write_ptr(cb_index);

        for (auto id = start_id, tiles = num_tiles_per_cycle; tiles > 0; ++id, --tiles) {
            noc_async_read_page(id, head, write_ptr);
            write_ptr += tile_size;
        }

        // recurse for remaining accessors
        if constexpr (Accessors > 1) {
            this->tail.read_tile_block(start_id, num_tiles_per_cycle);
        }
    }

    __attribute__((always_inline)) auto push_back(uint32_t num_tiles_per_cycle) const noexcept {
        // recurse for remaining accessors
        if constexpr (Accessors > 1) {
            this->tail.push_back(num_tiles_per_cycle);
        }

        cb_push_back(cb_index, num_tiles_per_cycle);
    }

    template <uint32_t NumTilesPerCycle = DefaultNumTilesPerCycle>
    auto read_tiles(uint32_t tiles, uint32_t id = 0) const noexcept {
        for (; tiles >= NumTilesPerCycle; tiles -= NumTilesPerCycle, id += NumTilesPerCycle) {
            this->reserve_back(NumTilesPerCycle);
            this->read_tile_block(id, NumTilesPerCycle);
            noc_async_read_barrier();
            this->push_back(NumTilesPerCycle);
        }

        // only instantiate recursion when stride could allow remaining tiles > 0
        if constexpr (NumTilesPerCycle > 1) {
            if (tiles > 0) {
                this->read_tiles<1>(tiles, id);
            }
        }
    }

    // writer members

    __attribute__((always_inline)) auto wait_front(uint32_t num_tiles_per_cycle) const noexcept {
        cb_wait_front(cb_index, num_tiles_per_cycle);

        // recurse for remaining accessors
        if constexpr (Accessors > 1) {
            this->tail.wait_front(num_tiles_per_cycle);
        }
    }

    __attribute__((always_inline)) auto write_tile_block(
        uint32_t start_id, uint32_t num_tiles_per_cycle) const noexcept {
        auto read_ptr = get_read_ptr(cb_index);

        for (auto id = start_id, tiles = num_tiles_per_cycle; tiles > 0; ++id, --tiles) {
            noc_async_write_page(id, head, read_ptr);
            read_ptr += tile_size;
        }

        // recurse for remaining accessors
        if constexpr (Accessors > 1) {
            this->tail.write_tile_block(start_id, num_tiles_per_cycle);
        }
    }

    __attribute__((always_inline)) auto pop_front(uint32_t num_tiles_per_cycle) const noexcept {
        // recurse for remaining accessors
        if constexpr (Accessors > 1) {
            this->tail.pop_front(num_tiles_per_cycle);
        }

        cb_pop_front(cb_index, num_tiles_per_cycle);
    }

    template <uint32_t NumTilesPerCycle = DefaultNumTilesPerCycle>
    auto write_tiles(uint32_t tiles, uint32_t id = 0) const noexcept {
        for (; tiles >= NumTilesPerCycle; tiles -= NumTilesPerCycle, id += NumTilesPerCycle) {
            this->wait_front(NumTilesPerCycle);
            this->write_tile_block(id, NumTilesPerCycle);
            noc_async_write_barrier();
            this->pop_front(NumTilesPerCycle);
        }

        // only instantiate recursion when stride could allow remaining tiles > 0
        if constexpr (NumTilesPerCycle > 1) {
            if (tiles > 0) {
                this->write_tiles<1>(tiles, id);
            }
        }
    }
};

// empty base case
template <size_t Offset, uint32_t DefaultNumTilesPerCycle>
class AccessorView<0, Offset, DefaultNumTilesPerCycle> {
public:
    __attribute__((always_inline)) AccessorView(int arg_idx) {}
};

}  // namespace views
