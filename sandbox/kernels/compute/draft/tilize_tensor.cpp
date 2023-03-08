#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    std::int32_t per_core_tile_cnt;
};

void hlk_main(tt_core *core_ptr, const hlk_args_t *args) {
    for(int b=0;b<args->per_core_tile_cnt;++b)
    {
        hlk_acquire_dst(core_ptr, DstMode::Half);

        // Pop tile after tile, copy to DST and pack
        hlk_wait_tiles(core_ptr, HlkOperand::in0, 1);
        hlk_wait_for_free_tiles(core_ptr, HlkOperand::out0, 1);

        hlk_copy_tile_to_dst(core_ptr, HlkOperand::in0, 0, 0);
        hlk_pack_tile_to_stream(core_ptr, 0, HlkOperand::out0);

        hlk_pop_tiles(core_ptr, HlkOperand::in0, 1);
        hlk_push_tiles(core_ptr, HlkOperand::out0, 1);

        hlk_release_dst(core_ptr, DstMode::Half);
    }

}
