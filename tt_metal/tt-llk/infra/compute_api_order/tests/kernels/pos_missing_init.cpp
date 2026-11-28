// POSITIVE: add_tiles with no preceding add_init/binary_tiles_init.
#include "api/compute/eltwise_binary.h"
void kernel_main() {
    tile_regs_acquire();
    add_tiles(0, 1, 0, 0, 0);
    tile_regs_commit();
}
