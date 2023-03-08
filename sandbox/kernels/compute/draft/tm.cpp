#include <cstdint>

#include "compute_hlk_api.h"

constexpr std::uint32_t TM_MAX_TILES = 200;
constexpr std::uint32_t TM_MAX_OUTPUTS = 8;
constexpr std::uint8_t NULL_BUFFER_INDEX = 0xFF;  // when used as buffer index, tile won't be packed anywhere

struct tile_info_t {
    std::uint8_t tile_index_within_out_buf;
    std::uint8_t buffer_index;
    std::uint8_t tile_offset_within_in_buf; // offset from the last one we read in input buffer
};

struct hlk_args_t {
    tile_info_t tile_info[TM_MAX_TILES];
    std::uint32_t tiles_per_output[TM_MAX_OUTPUTS];
    std::uint32_t num_tiles;
    std::uint32_t num_outputs;
};

#ifdef DEVICE
#if DEVICE == Model
#include <iostream>

std::ostream &operator<<(std::ostream &os, const tile_info_t &tile_info) {
    os << "buf = " << (int)tile_info.buffer_index
       << " out_index=" << (int)tile_info.tile_index_within_out_buf
       << " in_offset=" << (int)tile_info.tile_offset_within_in_buf << std::endl;
    return os;
}

std::ostream &operator<<(std::ostream &os, const hlk_args_t &args) {
    os << std::endl;
    os << "num_tiles = " << args.num_tiles << std::endl;
    for (std::uint32_t i = 0; i < args.num_tiles; i++) {
        os << " [" << i << "] " << args.tile_info[i] << std::endl;
    }
    os << "num_outputs = " << args.num_outputs << std::endl;
    for (std::uint32_t i = 0; i < args.num_outputs; i++) {
        os << " [" << i << "] cnt=" << args.tiles_per_output[i] << std::endl;
    }
    os << std::endl;
    return os;
}
#endif
#endif

void hlk_main(tt_core *core_ptr, const hlk_args_t *args) {

    // Wait until we have output space for all
    for (std::uint32_t i=0; i < args->num_outputs; i++) {
        hlk_wait_for_free_tiles(core_ptr, HlkOperand::out0 + i, args->tiles_per_output[i]);
    }

    // Bring in tiles
    for (std::uint32_t i=0; i < args->num_tiles; i++)
    {
        // Skip any unused tiles. For now, we're constrained to do it one tile at a time
        const tile_info_t &tile_info = args->tile_info[i];
        for (std::uint32_t s=0; s < tile_info.tile_offset_within_in_buf; s++) {
            hlk_wait_tiles(core_ptr, HlkOperand::in0, 1);
            hlk_pop_tiles(core_ptr, HlkOperand::in0, 1);
        }

        if (tile_info.buffer_index == NULL_BUFFER_INDEX)
          continue;  // skip packing, should only be used to read through remaining tiles in a buffer that won't be
                     // packed

        hlk_acquire_dst(core_ptr, DstMode::Half);
        hlk_wait_tiles(core_ptr, HlkOperand::in0, 1);
        hlk_copy_tile_to_dst(core_ptr, HlkOperand::in0, 0, 0);
        hlk_pack_tile_to_stream(core_ptr, 0, HlkOperand::out0 + tile_info.buffer_index, tile_info.tile_index_within_out_buf);
        hlk_pop_tiles(core_ptr, HlkOperand::in0, 1);
        hlk_release_dst(core_ptr, DstMode::Half);
    }

    // Push tiles
    for (std::uint32_t i=0; i < args->num_outputs; i++) {
        hlk_push_tiles(core_ptr, HlkOperand::out0 + i, args->tiles_per_output[i]);
    }
}
