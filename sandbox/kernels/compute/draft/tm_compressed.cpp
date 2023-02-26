#include <cstdint>

#include "compute_hlk_api.h"

constexpr std::uint32_t TM_DYNAMIC_LOCAL_MEMORY_SIZE = 900;
constexpr std::uint32_t TM_MAX_OUTPUTS = 8;

enum Skip {
    Skip_DrainInBuffer = 0,
    Skip_ResetOutBuffer = 0,
};

struct pattern_t {
    using integer = std::uint16_t;

    integer take = 0;
    integer skip = 0;

    bool operator==(pattern_t const &rhs) const { return (take == rhs.take) && (skip == rhs.skip); }
};

struct hlk_args_t {
    std::uint32_t tiles_per_output[TM_MAX_OUTPUTS];
    std::uint32_t num_outputs;
    std::uint32_t num_in_pattern_ids;
    std::uint32_t num_out_pattern_ids;
    // in_pattern_lm_offset implicitly 0
    std::uint32_t out_pattern_lm_offset;
    std::uint32_t patterns_lm_offset;
    std::uint32_t in_buffer_size;
    // Dynamic local memory area, split between input pattern indices, output pattern indices, and patterns
    std::uint8_t local_mem[TM_DYNAMIC_LOCAL_MEMORY_SIZE];
};

void hlk_main(tt_core *core_ptr, const hlk_args_t *args) {
    // Wait until we have output space for all
    for (std::uint32_t i = 0; i < args->num_outputs; i++) {
        hlk_wait_for_free_tiles(core_ptr, HlkOperand::out0 + i, args->tiles_per_output[i]);
    }

    std::uint8_t const *in_patterns = &args->local_mem[0] + 0;
    std::uint8_t const *out_patterns = &args->local_mem[0] + args->out_pattern_lm_offset;
    pattern_t const *patterns = reinterpret_cast<pattern_t const *>(&args->local_mem[0] + args->patterns_lm_offset);

    pattern_t in_pattern;
    pattern_t out_pattern;
    std::uint32_t in_pattern_idx = 0;
    std::uint32_t out_pattern_idx = 0;
    std::uint32_t in_tile_index = 0;
    std::uint32_t out_tile_index = 0;

    while ((in_pattern_idx < args->num_in_pattern_ids) || (out_pattern_idx < args->num_out_pattern_ids)) {
        if (in_pattern.take == 0) {
            for (std::uint32_t s = 0; s < in_pattern.skip; s++) {
                hlk_wait_tiles(core_ptr, HlkOperand::in0, 1);
                hlk_pop_tiles(core_ptr, HlkOperand::in0, 1);
            }
            in_tile_index += in_pattern.skip;
            in_pattern = patterns[in_patterns[in_pattern_idx++]];
            // cout << "in_pattern[" << (in_pattern_idx - 1) << "/" << args->num_in_pattern_ids << "] take "
            //     << (int)in_pattern.take << " skip " << (int)in_pattern.skip << endl;
        }

        if (out_pattern.take == 0) {
            if (out_pattern.skip == Skip_ResetOutBuffer) {
                out_tile_index = 0;
            }
            out_tile_index += out_pattern.skip;
            out_pattern = patterns[out_patterns[out_pattern_idx++]];
            // cout << "out_pattern[" << (out_pattern_idx - 1) << "/" << args->num_out_pattern_ids << "] take "
            //     << (int)out_pattern.take << " skip " << (int)out_pattern.skip << endl;
        }

        while ((in_pattern.take > 0) && (out_pattern.take > 0)) {
            hlk_acquire_dst(core_ptr, DstMode::Half);
            hlk_wait_tiles(core_ptr, HlkOperand::in0, 1);
            hlk_copy_tile_to_dst(core_ptr, HlkOperand::in0, 0, 0);
            hlk_pack_tile_to_stream(core_ptr, 0, HlkOperand::out0 /* TODO + buffer_index*/, out_tile_index);
            hlk_pop_tiles(core_ptr, HlkOperand::in0, 1);
            hlk_release_dst(core_ptr, DstMode::Half);

            --in_pattern.take;
            --out_pattern.take;
            ++in_tile_index;
            ++out_tile_index;
        }

        // skip of 0 means drain the rest of the buffer
        if (in_pattern.take == 0 && in_pattern.skip == Skip_DrainInBuffer) {
            for (; in_tile_index < args->in_buffer_size; ++in_tile_index) {
                hlk_wait_tiles(core_ptr, HlkOperand::in0, 1);
                hlk_pop_tiles(core_ptr, HlkOperand::in0, 1);
            }
            in_tile_index = 0;
        }
    }

    // Push tiles
    for (std::uint32_t i = 0; i < args->num_outputs; i++) {
        hlk_push_tiles(core_ptr, HlkOperand::out0 + i, args->tiles_per_output[i]);
    }
}

