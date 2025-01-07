// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api_declarations.h"

namespace interleaved_addr_gen {

template <bool DRAM>
FORCE_INLINE uint32_t get_bank_offset_index(uint32_t id) {
    if constexpr (DRAM) {  // DRAM
#ifdef IS_NOT_POW2_NUM_DRAM_BANKS
        return udivsi3_const_divisor<NUM_DRAM_BANKS>(id);
#else
        return id >> LOG_BASE_2_OF_NUM_DRAM_BANKS;
#endif
    } else {  // L1
#ifdef IS_NOT_POW2_NUM_L1_BANKS
        return udivsi3_const_divisor<NUM_L1_BANKS>(id);
#else
        return id >> LOG_BASE_2_OF_NUM_L1_BANKS;
#endif
    }
}

template <bool DRAM>
FORCE_INLINE uint32_t get_bank_index(uint32_t id, uint32_t bank_offset_index) {
    if constexpr (DRAM) {  // DRAM
        return id - bank_offset_index * NUM_DRAM_BANKS;
    } else {  // L1
        return id - bank_offset_index * NUM_L1_BANKS;
    }
}

template <bool DRAM>
FORCE_INLINE uint32_t get_noc_xy(uint32_t bank_index, uint8_t noc = noc_index) {
    if constexpr (DRAM) {  // DRAM
        return dram_bank_to_noc_xy[noc][bank_index];
    } else {  // L1
        return l1_bank_to_noc_xy[noc][bank_index];
    }
}

template <bool DRAM>
FORCE_INLINE uint32_t get_bank_offset(uint32_t bank_index) {
    if constexpr (DRAM) {  // DRAM
        return bank_to_dram_offset[bank_index];
    } else {  // L1
        return bank_to_l1_offset[bank_index];
    }
}

}  // namespace interleaved_addr_gen

template <uint32_t tile_hw = 1024>
FORCE_INLINE constexpr static std::uint32_t MUL_WITH_TILE_SIZE(uint format, uint index) {
    constexpr uint8_t datum_shift = (tile_hw == 1024)  ? 10
                                    : (tile_hw == 512) ? 9
                                    : (tile_hw == 256) ? 8
                                    : (tile_hw == 128) ? 7
                                    : (tile_hw == 64)  ? 6
                                    : (tile_hw == 32)  ? 5
                                    : (tile_hw == 16)  ? 4
                                                       : 10;

    constexpr uint8_t exp_shift = (tile_hw == 1024)  ? 6
                                  : (tile_hw == 512) ? 5
                                  : (tile_hw == 256) ? 4
                                  : (tile_hw == 128) ? 4
                                  : (tile_hw == 64)  ? 4
                                  : (tile_hw == 32)  ? 4
                                  : (tile_hw == 16)  ? 4
                                                     : 6;
    switch (format & 0x1F) {
        case ((uint8_t)DataFormat::UInt8): return (index << datum_shift);
        case ((uint8_t)DataFormat::UInt16):
        case ((uint8_t)DataFormat::Float16):
        case ((uint8_t)DataFormat::Float16_b): return (index << (datum_shift + 1));
        case ((uint8_t)DataFormat::Int32):
        case ((uint8_t)DataFormat::UInt32):
        case ((uint8_t)DataFormat::Float32): return (index << (datum_shift + 2));
        case ((uint8_t)DataFormat::Bfp2):
        case ((uint8_t)DataFormat::Bfp2_b): return ((index << (datum_shift - 2)) + (index << (exp_shift)));
        case ((uint8_t)DataFormat::Bfp4):
        case ((uint8_t)DataFormat::Bfp4_b): return ((index << (datum_shift - 1)) + (index << (exp_shift)));
        case ((uint8_t)DataFormat::Bfp8):
        case ((uint8_t)DataFormat::Bfp8_b):
        // Keep default as Bfp8?
        default: return ((index << datum_shift) + (index << (exp_shift)));
    };
}

/*
    Need an alias to get_noc_addr so that the structs below don't confuse the above get_noc_addr with
    the struct variant
*/
FORCE_INLINE
std::uint64_t get_noc_addr_helper(std::uint32_t noc_xy, std::uint32_t addr) {
    /*
        Get an encoding which contains tensix core and address you want to
        write to via the noc multicast
    */
    return ((uint64_t)(noc_xy) << NOC_ADDR_COORD_SHIFT) | addr;
}

namespace shard_addr_gen_utils {

template <uint32_t start_x, uint32_t start_y, uint32_t width, uint32_t height>
struct shard_grid_info {
    constexpr static uint32_t x = start_x;
    constexpr static uint32_t y = start_x;
    constexpr static uint32_t w = start_x;
    constexpr static uint32_t h = start_x;
    constexpr static uint32_t volume = width * height;
};

template <bool transposed, typename info_struct>
FORCE_INLINE std::pair<uint32_t, uint32_t> resolve_within_grid(uint32_t linear_core_id, const info_struct grid_info) {
    uint32_t outer_offset = linear_core_id / grid_info.w;                   // Unavoidable division to get the offset
    uint32_t inner_offset = linear_core_id - linear_core_id * grid_info.w;  // linear_core_id % width;
    if constexpr (transposed) {
        std::pair<uint32_t, uint32_t> coordinates(outer_offset + grid_info.x, inner_offset + grid_info.y);
        return coordinates;
    } else {
        std::pair<uint32_t, uint32_t> coordinates(inner_offset + grid_info.x, outer_offset + grid_info.y);
        return coordinates;
    }
}

template <
    bool transposed,
    uint32_t num_shard_grids,
    typename is0,
    typename is1,
    typename is2,
    typename is3,
    typename is4,
    typename is5,
    typename is6,
    typename is7>
FORCE_INLINE std::pair<uint32_t, uint32_t> resolve_core_grid(
    uint32_t linear_core_id,
    const is0 grid0,
    const is1 grid1,
    const is2 grid2,
    const is3 grid3,
    const is4 grid4,
    const is5 grid5,
    const is6 grid6,
    const is7 grid7) {
    constexpr struct shard_grid_info<0, 0, 0, 0> grid_null{};
    // Finds the logical core ID given the nth core is wanted
    if constexpr (num_shard_grids == 1) {
        return resolve_within_grid<transposed>(linear_core_id, grid0);
    } else {
        constexpr uint32_t halfway_volume = grid0.volume + ((num_shard_grids > 3) ? grid1.volume : 0) +
                                            ((num_shard_grids > 5) ? grid2.volume : 0) +
                                            ((num_shard_grids > 7) ? grid3.volume : 0);
        if constexpr (num_shard_grids == 8) {
            // 8 grids, halfway covers grids 0-3
            if (linear_core_id >= halfway_volume) {
                return resolve_core_grid<transposed, num_shard_grids - 4>(
                    linear_core_id - halfway_volume,
                    grid4,
                    grid5,
                    grid6,
                    grid7,
                    grid_null,
                    grid_null,
                    grid_null,
                    grid_null);
            } else {
                return resolve_core_grid<transposed, 4>(
                    linear_core_id, grid0, grid1, grid2, grid3, grid_null, grid_null, grid_null, grid_null);
            }
        } else if constexpr (num_shard_grids > 5) {
            // 6 or 7 grids, halfway covers grids 0-2
            if (linear_core_id >= halfway_volume) {
                return resolve_core_grid<transposed, num_shard_grids - 3>(
                    linear_core_id - halfway_volume,
                    grid3,
                    grid4,
                    grid5,
                    grid6,
                    grid_null,
                    grid_null,
                    grid_null,
                    grid_null);
            } else {
                return resolve_core_grid<transposed, 3>(
                    linear_core_id, grid0, grid1, grid2, grid_null, grid_null, grid_null, grid_null, grid_null);
            }
        } else if constexpr (num_shard_grids > 3) {
            // 4 or 5 grids halfway covers grids 0-1
            if (linear_core_id >= halfway_volume) {
                return resolve_core_grid<transposed, num_shard_grids - 2>(
                    linear_core_id - halfway_volume,
                    grid2,
                    grid3,
                    grid4,
                    grid_null,
                    grid_null,
                    grid_null,
                    grid_null,
                    grid_null);
            } else {
                return resolve_core_grid<transposed, 2>(
                    linear_core_id, grid0, grid1, grid_null, grid_null, grid_null, grid_null, grid_null, grid_null);
            }
        } else {
            // 2 or 3 grids halfway covers grid 0
            if (linear_core_id >= halfway_volume) {
                return resolve_core_grid<transposed, num_shard_grids - 1>(
                    linear_core_id - halfway_volume,
                    grid1,
                    grid2,
                    grid_null,
                    grid_null,
                    grid_null,
                    grid_null,
                    grid_null,
                    grid_null);
            } else {
                return resolve_within_grid<transposed>(linear_core_id, grid0);
            }
        }
    }
}

struct shard_coord_info {
    uint32_t core_num;
    uint32_t page_num;
    uint32_t num_contiguous_pages;
};

template <uint32_t page_size, uint32_t columns_per_shard, uint32_t total_pages_last_dim>
struct shard_coord_info get_width_sharded_coordinates(uint32_t page_num) {
    // Returns core index followed by the page number
    struct shard_coord_info coord_info;
    uint32_t page_row = page_num / total_pages_last_dim;
    uint32_t page_col = page_num - page_row * total_pages_last_dim;  // page_num%total_pages_last_dim
    uint32_t w_core_id = page_col / columns_per_shard;
    uint32_t w_offset = page_col - w_core_id * columns_per_shard;
    coord_info.core_num = w_core_id;
    coord_info.page_num = page_row * columns_per_shard + w_offset;
    coord_info.num_contiguous_pages = columns_per_shard - w_offset;
    return coord_info;
}

template <uint32_t page_size, uint32_t rows_per_shard, uint32_t total_pages_last_dim>
struct shard_coord_info get_height_sharded_coordinates(uint32_t page_num) {
    // Returns core index followed by the page number
    struct shard_coord_info coord_info;
    constexpr uint32_t num_pages_per_core = total_pages_last_dim * rows_per_shard;
    coord_info.core_num = page_num / num_pages_per_core;
    coord_info.page_num = page_num - coord_info.core_num * num_pages_per_core;
    coord_info.num_contiguous_pages = num_pages_per_core - coord_info.page_num;
    return coord_info;
}

template <uint32_t page_size, uint32_t columns_per_shard, uint32_t rows_per_shard, uint32_t total_pages_last_dim>
struct shard_coord_info get_block_sharded_coordinates(uint32_t page_num) {
    // Returns core index followed by the page number
    // Calculate how many cores are in the sharding grid
    constexpr uint32_t cores_per_block_row = (total_pages_last_dim - 1) / columns_per_shard + 1;
    struct shard_coord_info coord_info;
    // Get row and column ID of this page
    uint32_t page_row = page_num / total_pages_last_dim;
    uint32_t page_col = page_num - page_row * total_pages_last_dim;  // page_col = page_num%total_pages_last_dim;
    // Find the w direction core and the offset within it
    uint32_t w_core_id = page_col / columns_per_shard;
    uint32_t w_offset = page_col - w_core_id * columns_per_shard;  // w_offset = page_col%columns_per_shard;
    // Find the h direction core and the offset within it
    uint32_t h_core_id = page_row / rows_per_shard;
    uint32_t h_offset = page_row - h_core_id * rows_per_shard;  // h_offset = page_row%rows_per_shard;
    // Find the coord_info
    coord_info.core_num = w_core_id + h_core_id * cores_per_block_row;
    coord_info.page_num = w_offset + h_offset * columns_per_shard;
    coord_info.num_contiguous_pages = columns_per_shard - w_offset;
    return coord_info;
}

std::pair<const uint32_t* const, uint32_t> parse_map(uint32_t rt_address) {
    uint32_t elements = get_arg_val<uint32_t>(rt_address++);
    const uint32_t* const map = reinterpret_cast<const uint32_t* const>(get_arg_addr(rt_address));
    return std::pair<const uint32_t* const, uint32_t>(map, rt_address + elements);
}

}  // namespace shard_addr_gen_utils

template <
    bool transposed,
    uint32_t shard_type,
    uint32_t num_shard_grids,
    uint32_t page_size,
    uint32_t pages_per_shard_x,
    uint32_t pages_per_shard_y,
    uint32_t pages_last_shard_dim,
    typename is0,
    typename is1,
    typename is2,
    typename is3,
    typename is4,
    typename is5,
    typename is6,
    typename is7>
struct ShardedAddrGen {
    uint32_t bank_base_address;
    const static is0 grid0;
    const static is1 grid1;
    const static is2 grid2;
    const static is3 grid3;
    const static is4 grid4;
    const static is5 grid5;
    const static is6 grid6;
    const static is7 grid7;
    constexpr static uint32_t num_cores =
        grid0.volume + ((num_shard_grids > 1) ? grid1.volume : 0) + ((num_shard_grids > 2) ? grid2.volume : 0) +
        ((num_shard_grids > 3) ? grid3.volume : 0) + ((num_shard_grids > 4) ? grid4.volume : 0) +
        ((num_shard_grids > 5) ? grid5.volume : 0) + ((num_shard_grids > 6) ? grid6.volume : 0) +
        ((num_shard_grids > 7) ? grid7.volume : 0);
    FORCE_INLINE
    std::uint64_t get_addr(
        const uint32_t logx,
        const uint32_t logy,
        const uint32_t core_page,
        const uint32_t offset,
        const uint32_t noc = noc_index) {
        const uint32_t address = this->bank_base_address + (core_page * page_size) + offset;

        return get_noc_addr(logx, logy, address, noc);
    }

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        return this->get_contiguous_noc_addr(id, offset, noc).first;
    }
    FORCE_INLINE
    std::pair<uint64_t, uint32_t> get_contiguous_noc_addr(
        const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        // Returns the noc address AND the number of contiguous pages after.
        // Resolve linear core id and the page num within that core
        struct shard_coord_info sharding_coordinates;
        if constexpr (shard_type == 0) {
            sharding_coordinates =
                shard_addr_gen_utils::get_width_sharded_coordinates<page_size, pages_per_shard_x, pages_last_shard_dim>(
                    id);
        } else if constexpr (shard_type == 1) {
            sharding_coordinates = shard_addr_gen_utils::
                get_height_sharded_coordinates<page_size, pages_per_shard_y, pages_last_shard_dim>(id);
        } else {
            sharding_coordinates = shard_addr_gen_utils::
                get_block_sharded_coordinates<page_size, pages_per_shard_x, pages_per_shard_y, pages_last_shard_dim>(
                    id);
        }
        uint32_t linear_core_id = 0;
        uint32_t page_num = 0;
        std::pair<uint32_t, uint32_t> core_index = shard_addr_gen_utils::resolve_core_grid<transposed, num_shard_grids>(
            sharding_coordinates.core_num, grid0, grid1, grid2, grid3, grid4, grid5, grid6, grid7);
        // return get_addr
        std::pair<uint64_t, uint32_t> return_val(
            get_addr(core_index.first, core_index.second, sharding_coordinates.page_num, offset, noc),
            sharding_coordinates.num_contiguous_pages);
        return return_val;
    }

    FORCE_INLINE
    void noc_async_read_page(
        const uint32_t id, const uint32_t dest_addr, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        noc_async_read(this->get_noc_addr(id, offset), dest_addr, page_size, noc);
    }
};

template <bool DRAM>
struct InterleavedAddrGen {
    uint32_t bank_base_address;  // Base address for the whole tensor.
    const uint32_t page_size;    // Num bytes in page.
    const uint32_t aligned_page_size = align(page_size, ALLOCATOR_ALIGNMENT);

    FORCE_INLINE
    uint32_t get_addr(
        const uint32_t id,
        const uint32_t bank_offset_index,
        const uint32_t bank_index,
        const uint32_t offset = 0) const {
        return (bank_offset_index * this->aligned_page_size) + this->bank_base_address + offset +
               interleaved_addr_gen::get_bank_offset<DRAM>(bank_index);
    }

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

        uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
        return noc_addr;
    }

    FORCE_INLINE
    std::pair<uint64_t, uint32_t> get_contiguous_noc_addr(
        const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        std::pair<uint64_t, uint32_t> ret_val(this->get_noc_addr(id, offset, noc), 1);
        return ret_val;
    }

    FORCE_INLINE
    void noc_async_read_page(
        const uint32_t id, const uint32_t dest_addr, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        noc_async_read(this->get_noc_addr(id, offset), dest_addr, page_size, noc);
    }
};

template <bool DRAM>
struct InterleavedPow2AddrGen {
    const uint32_t bank_base_address;
    const uint32_t log_base_2_of_page_size;  // WARNING: This struct is used for optimized get_noc_addr in which case
                                             // you know that bank_unit_size is a power of 2
    const uint32_t aligned_log_base_2_of_page_size = this->log_base_2_of_page_size > LOG_BASE_2_OF_ALLOCATOR_ALIGNMENT
                                                         ? this->log_base_2_of_page_size
                                                         : LOG_BASE_2_OF_ALLOCATOR_ALIGNMENT;

    FORCE_INLINE
    uint32_t get_addr(
        const uint32_t id,
        const uint32_t bank_offset_index,
        const uint32_t bank_index,
        const uint32_t offset = 0) const {
        return (bank_offset_index << this->aligned_log_base_2_of_page_size) + this->bank_base_address + offset +
               interleaved_addr_gen::get_bank_offset<DRAM>(bank_index);
    }

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

        uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
        return noc_addr;
    }

    FORCE_INLINE
    std::pair<uint64_t, uint32_t> get_contiguous_noc_addr(
        const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        std::pair<uint64_t, uint32_t> ret_val(this->get_noc_addr(id, offset, noc), 1);
        return ret_val;
    }
};

template <bool DRAM, uint32_t tile_hw = 1024>
struct InterleavedAddrGenFast {
    uint32_t bank_base_address;  // Base address for the whole tensor.
    // TODO: Remove page_size from argument list. This can be derived from data_format
    uint32_t page_size;      // Num bytes in bank unit.
    DataFormat data_format;  // Data format

    FORCE_INLINE
    uint32_t get_addr(
        const uint32_t id,
        const uint32_t bank_offset_index,
        const uint32_t bank_index,
        const uint32_t offset = 0) const {
        return MUL_WITH_TILE_SIZE<tile_hw>((uint)this->data_format, bank_offset_index) + this->bank_base_address +
               offset + interleaved_addr_gen::get_bank_offset<DRAM>(bank_index);
    }

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

        uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
        return noc_addr;
    }

    FORCE_INLINE
    std::pair<uint64_t, uint32_t> get_contiguous_noc_addr(
        const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        std::pair<uint64_t, uint32_t> ret_val(this->get_noc_addr(id, offset, noc), 1);
        return ret_val;
    }

    FORCE_INLINE
    void noc_async_read_tile(
        const uint32_t id, uint32_t dest_addr, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t src_addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t src_noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

        WAYPOINT("NRTW");
        DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc, get_noc_addr_helper(src_noc_xy, src_addr), dest_addr, this->page_size);
        while (!noc_cmd_buf_ready(noc, read_cmd_buf));
        WAYPOINT("NRTD");

        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_RET_ADDR_LO, dest_addr);
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_LO, src_addr);            // (uint32_t)src_addr
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, src_noc_xy);  // src_addr >> 32
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_AT_LEN_BE, this->page_size);        // len_bytes
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_reads_num_issued[noc] += 1;
    }

    FORCE_INLINE
    void noc_async_write_tile(const uint32_t id, uint32_t src_addr, uint8_t noc = noc_index) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t dest_addr = this->get_addr(id, bank_offset_index, bank_index);
        uint32_t dest_noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

        WAYPOINT("NWTW");
        DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(
            noc, get_noc_addr_helper(dest_noc_xy, dest_addr), src_addr, this->page_size);
        while (!noc_cmd_buf_ready(noc, write_cmd_buf));
        WAYPOINT("NWTD");

        constexpr uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC |
                                           NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) |
                                           0x0 |  // (linked ? NOC_CMD_VC_LINKED : 0x0)
                                           0x0 |  // (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0)
                                           NOC_CMD_RESP_MARKED;

        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CTRL, noc_cmd_field);
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_TARG_ADDR_LO, src_addr);
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_LO, dest_addr);            // (uint32_t)dest_addr
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_COORDINATE, dest_noc_xy);  // dest_addr >> 32
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_AT_LEN_BE, this->page_size);        // len_bytes
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        if constexpr (noc_mode == DM_DYNAMIC_NOC) {
            inc_noc_nonposted_writes_acked<proc_type>(noc);
        } else {
            noc_nonposted_writes_num_issued[noc] += 1;
            noc_nonposted_writes_acked[noc] += 1;  // num_dests
        }
    }
};

// TODO: add noc_async_write_page
// TODO: need static assert + host assert that page size <= 8192, hard constraint
template <bool DRAM>
struct InterleavedPow2AddrGenFast {
    uint32_t bank_base_address;              // Base address for the whole tensor.
    const uint32_t log_base_2_of_page_size;  // Num bytes in bank unit.
    const uint32_t aligned_log_base_2_of_page_size = this->log_base_2_of_page_size > LOG_BASE_2_OF_ALLOCATOR_ALIGNMENT
                                                         ? this->log_base_2_of_page_size
                                                         : LOG_BASE_2_OF_ALLOCATOR_ALIGNMENT;

    FORCE_INLINE
    uint32_t get_addr(
        const uint32_t id,
        const uint32_t bank_offset_index,
        const uint32_t bank_index,
        const uint32_t offset = 0) const {
        return (bank_offset_index << this->aligned_log_base_2_of_page_size) + this->bank_base_address + offset +
               interleaved_addr_gen::get_bank_offset<DRAM>(bank_index);
    }

    FORCE_INLINE
    std::uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

        uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
        return noc_addr;
    }

    FORCE_INLINE
    std::pair<uint64_t, uint32_t> get_contiguous_noc_addr(
        const uint32_t id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        std::pair<uint64_t, uint32_t> ret_val(this->get_noc_addr(id, offset, noc), 1);
        return ret_val;
    }

    FORCE_INLINE
    void noc_async_read_page(
        const uint32_t id, uint32_t dest_addr, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t src_addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t src_noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

        WAYPOINT("NRPW");
        DEBUG_SANITIZE_NOC_READ_TRANSACTION(
            noc, get_noc_addr_helper(src_noc_xy, src_addr), dest_addr, 1 << this->aligned_log_base_2_of_page_size);
        while (!noc_cmd_buf_ready(noc, read_cmd_buf));
        WAYPOINT("NRPD");

        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_RET_ADDR_LO, dest_addr);
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_LO, src_addr);            // (uint32_t)src_addr
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, src_noc_xy);  // src_addr >> 32
        NOC_CMD_BUF_WRITE_REG(
            noc, read_cmd_buf, NOC_AT_LEN_BE, 1 << this->aligned_log_base_2_of_page_size);  // len_bytes
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_reads_num_issued[noc] += 1;
    }

    FORCE_INLINE
    void noc_async_read_partial_page(
        const uint32_t id,
        uint32_t dest_addr,
        const uint32_t size,
        const uint32_t offset,
        uint8_t noc = noc_index) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t src_addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t src_noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

        WAYPOINT("RP1W");
        while (!noc_cmd_buf_ready(noc, read_cmd_buf));
        WAYPOINT("RP1D");
        DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc, get_noc_addr_helper(src_noc_xy, src_addr), dest_addr, size);

        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_RET_ADDR_LO, dest_addr);
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_LO, src_addr);            // (uint32_t)src_addr
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, src_noc_xy);  // src_addr >> 32
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_AT_LEN_BE, size);                   // len_bytes
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_reads_num_issued[noc] += 1;
    }

    FORCE_INLINE
    void noc_async_write_page(
        const uint32_t id,
        uint32_t src_addr,
        const uint32_t write_size_bytes,
        const uint32_t offset = 0,
        uint8_t noc = noc_index) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t dest_addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t dest_noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);

        WAYPOINT("NWPW");
        DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(
            noc, get_noc_addr_helper(dest_noc_xy, dest_addr), src_addr, write_size_bytes);
        while (!noc_cmd_buf_ready(noc, write_cmd_buf));
        WAYPOINT("NWPD");

        constexpr uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC |
                                           NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) |
                                           0x0 |  // (linked ? NOC_CMD_VC_LINKED : 0x0)
                                           0x0 |  // (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0)
                                           NOC_CMD_RESP_MARKED;

        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CTRL, noc_cmd_field);
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_TARG_ADDR_LO, src_addr);
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_LO, dest_addr);            // (uint32_t)dest_addr
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_COORDINATE, dest_noc_xy);  // dest_addr >> 32
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_AT_LEN_BE, write_size_bytes);       // len_bytes
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        if constexpr (noc_mode == DM_DYNAMIC_NOC) {
            inc_noc_nonposted_writes_acked<proc_type>(noc);
        } else {
            noc_nonposted_writes_num_issued[noc] += 1;
            noc_nonposted_writes_acked[noc] += 1;  // num_dests
        }
    }
};

template <
    bool transposed,
    uint32_t shard_type,
    uint32_t num_shard_grids,
    uint32_t page_size,
    uint32_t pages_per_shard_x,
    uint32_t pages_per_shard_y,
    uint32_t pages_last_shard_dim,
    typename is0,
    typename is1,
    typename is2,
    typename is3,
    typename is4,
    typename is5,
    typename is6,
    typename is7>
FORCE_INLINE std::uint64_t get_noc_addr(
    const uint32_t id,
    const ShardedAddrGen<
        transposed,
        shard_type,
        num_shard_grids,
        page_size,
        pages_per_shard_x,
        pages_per_shard_y,
        pages_last_shard_dim,
        is0,
        is1,
        is2,
        is3,
        is4,
        is5,
        is6,
        is7>& s,
    uint32_t offset = 0,
    uint8_t noc = noc_index) {
    return s.get_noc_addr(id, offset, noc);
}

template <bool DRAM>
FORCE_INLINE std::uint64_t get_noc_addr(
    const uint32_t id, const InterleavedAddrGen<DRAM>& s, uint32_t offset = 0, uint8_t noc = noc_index) {
    /*
        Alternative API for getting the noc address when we are reading using a swizzled
        layout. This version assumes bank unit size can be arbitrary size. Use
        get_noc_addr(const uint32_t id, InterleavedPow2AddrGen s) for optimized algorithm in which stick size
        is a power of 2.

        id: Unique id for the bank_unit you want to read, assuming row major order. We use this to compute the
        bank for this unit of data.

        InterleavedAddrGen: Check struct for attribute definitions.
    */
    return s.get_noc_addr(id, offset, noc);
}

template <bool DRAM>
FORCE_INLINE std::uint64_t get_noc_addr(
    const uint32_t id, const InterleavedPow2AddrGen<DRAM>& s, uint32_t offset = 0, uint8_t noc = noc_index) {
    /*
        Alternative API for getting the noc address when we are reading using a swizzled
        layout. This version assumes bank unit size is a power of 2. For arbitrary bank
        unit size, use get_noc_addr(const uint32_t id, const InterleavedOffset s)

        id: Unique id for the bank_unit you want to read, assuming row major order. We use this to compute the
        bank for this unit of data.

        InterleavedPow2AddrGen: Check struct for attribute definitions.
    */

    return s.get_noc_addr(id, offset, noc);
}

template <bool DRAM, uint32_t tile_hw>
FORCE_INLINE std::uint64_t get_noc_addr(
    const uint32_t id, const InterleavedAddrGenFast<DRAM, tile_hw>& s, uint32_t offset = 0, uint8_t noc = noc_index) {
    /*
        Alternative API for getting the noc address when we are reading using a swizzled
        layout. This version assumes bank unit size can be arbitrary size. Use
        get_noc_addr(const uint32_t id, InterleavedPow2AddrGen s) for optimized algorithm in which stick size
        is a power of 2.

        id: Unique id for the bank_unit you want to read, assuming row major order. We use this to compute the
        bank for this unit of data.

        InterleavedAddrGen: Check struct for attribute definitions.
    */
    return s.get_noc_addr(id, offset, noc);
}

template <bool DRAM>
FORCE_INLINE std::uint64_t get_noc_addr(
    const uint32_t id, const InterleavedPow2AddrGenFast<DRAM>& s, uint32_t offset = 0, uint8_t noc = noc_index) {
    /*
        Alternative API for getting the noc address when we are reading using a swizzled
        layout. This version assumes bank unit size is a power of 2 and less than or equal to NOC_MAX_BURST_SIZE.
        For arbitrary bank unit size, use get_noc_addr(const uint32_t id, const InterleavedOffset s)

        id: Unique id for the bank_unit you want to read, assuming row major order. We use this to compute the
        bank for this unit of data.

        InterleavedPow2AddrGenFast: Check struct for attribute definitions.
    */

    return s.get_noc_addr(id, offset, noc);
}

template <
    bool transposed,
    uint32_t shard_type,
    uint32_t num_shard_grids,
    uint32_t page_size,
    uint32_t pages_per_shard_x,
    uint32_t pages_per_shard_y,
    uint32_t pages_last_shard_dim,
    typename is0,
    typename is1,
    typename is2,
    typename is3,
    typename is4,
    typename is5,
    typename is6,
    typename is7>
FORCE_INLINE std::pair<uint64_t, uint32_t> get_contiguous_noc_addr(
    const uint32_t id,
    const ShardedAddrGen<
        transposed,
        shard_type,
        num_shard_grids,
        page_size,
        pages_per_shard_x,
        pages_per_shard_y,
        pages_last_shard_dim,
        is0,
        is1,
        is2,
        is3,
        is4,
        is5,
        is6,
        is7>& s,
    uint32_t offset = 0,
    uint8_t noc = noc_index) {
    return s.get_contiguous_noc_addr(id, offset, noc);
}

template <bool DRAM>
FORCE_INLINE std::pair<uint64_t, uint32_t> get_contiguous_noc_addr(
    const uint32_t id, const InterleavedAddrGen<DRAM>& s, uint32_t offset = 0, uint8_t noc = noc_index) {
    /*
        Alternative API for getting the noc address when we are reading using a swizzled
        layout. This version assumes bank unit size can be arbitrary size. Use
        get_noc_addr(const uint32_t id, InterleavedPow2AddrGen s) for optimized algorithm in which stick size
        is a power of 2.

        id: Unique id for the bank_unit you want to read, assuming row major order. We use this to compute the
        bank for this unit of data.

        InterleavedAddrGen: Check struct for attribute definitions.
    */
    return s.get_contiguous_noc_addr(id, offset, noc);
}

template <bool DRAM>
FORCE_INLINE std::pair<uint64_t, uint32_t> get_contiguous_noc_addr(
    const uint32_t id, const InterleavedPow2AddrGen<DRAM>& s, uint32_t offset = 0, uint8_t noc = noc_index) {
    /*
        Alternative API for getting the noc address when we are reading using a swizzled
        layout. This version assumes bank unit size is a power of 2. For arbitrary bank
        unit size, use get_noc_addr(const uint32_t id, const InterleavedOffset s)

        id: Unique id for the bank_unit you want to read, assuming row major order. We use this to compute the
        bank for this unit of data.

        InterleavedPow2AddrGen: Check struct for attribute definitions.
    */

    return s.get_contiguous_noc_addr(id, offset, noc);
}

template <bool DRAM, uint32_t tile_hw>
FORCE_INLINE std::pair<uint64_t, uint32_t> get_contiguous_noc_addr(
    const uint32_t id, const InterleavedAddrGenFast<DRAM, tile_hw>& s, uint32_t offset = 0, uint8_t noc = noc_index) {
    /*
        Alternative API for getting the noc address when we are reading using a swizzled
        layout. This version assumes bank unit size can be arbitrary size. Use
        get_noc_addr(const uint32_t id, InterleavedPow2AddrGen s) for optimized algorithm in which stick size
        is a power of 2.

        id: Unique id for the bank_unit you want to read, assuming row major order. We use this to compute the
        bank for this unit of data.

        InterleavedAddrGen: Check struct for attribute definitions.
    */
    return s.get_contiguous_noc_addr(id, offset, noc);
}

template <bool DRAM>
FORCE_INLINE std::pair<uint64_t, uint32_t> get_contiguous_noc_addr(
    const uint32_t id, const InterleavedPow2AddrGenFast<DRAM>& s, uint32_t offset = 0, uint8_t noc = noc_index) {
    /*
        Alternative API for getting the noc address when we are reading using a swizzled
        layout. This version assumes bank unit size is a power of 2 and less than or equal to NOC_MAX_BURST_SIZE.
        For arbitrary bank unit size, use get_noc_addr(const uint32_t id, const InterleavedOffset s)

        id: Unique id for the bank_unit you want to read, assuming row major order. We use this to compute the
        bank for this unit of data.

        InterleavedPow2AddrGenFast: Check struct for attribute definitions.
    */

    return s.get_contiguous_noc_addr(id, offset, noc);
}
template <bool DRAM>
FORCE_INLINE void noc_async_read_page(
    const uint32_t id,
    const InterleavedAddrGen<DRAM>& s,
    std::uint32_t dst_local_l1_addr,
    uint32_t offset = 0,
    uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    s.noc_async_read_page(id, dst_local_l1_addr, offset, noc);
}

template <bool DRAM, uint32_t tile_hw>
FORCE_INLINE void noc_async_read_tile(
    const uint32_t id,
    const InterleavedAddrGenFast<DRAM, tile_hw>& s,
    std::uint32_t dst_local_l1_addr,
    uint32_t offset = 0,
    uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    s.noc_async_read_tile(id, dst_local_l1_addr, offset, noc);
}

template <bool DRAM, uint32_t page_size>
FORCE_INLINE auto get_interleaved_addr_gen(uint32_t base_addr) {
    constexpr bool is_pow_2 = is_power_of_2(page_size);
    if constexpr (is_pow_2) {
        constexpr uint32_t log2_page_size = __builtin_ctz(page_size);
        if constexpr (page_size <= NOC_MAX_BURST_SIZE) {
            return InterleavedPow2AddrGenFast<DRAM>{
                .bank_base_address = base_addr, .log_base_2_of_page_size = log2_page_size};
        } else {
            return InterleavedPow2AddrGen<DRAM>{
                .bank_base_address = base_addr, .log_base_2_of_page_size = log2_page_size};
        }
    } else {
        return InterleavedAddrGen<DRAM>{.bank_base_address = base_addr, .page_size = page_size};
    }
}
