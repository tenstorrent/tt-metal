// NEGATIVE: binary_tiles_init is an accepted alias for add_tiles.
#include "api/compute/eltwise_binary.h"
void kernel_main() {
    binary_tiles_init(0, 1);
    for (int i = 0; i < 4; ++i) {
        tile_regs_acquire();
        add_tiles(0, 1, i, i, 0);
        tile_regs_commit();
    }
}
