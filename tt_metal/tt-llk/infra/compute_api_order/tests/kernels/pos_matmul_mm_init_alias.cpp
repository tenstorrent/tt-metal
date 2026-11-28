// POSITIVE: matmul_tiles after only add_init (wrong init for this op).
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
void kernel_main() {
    add_init(0, 1);
    tile_regs_acquire();
    matmul_tiles(0, 1, 0, 0, 0);
    tile_regs_commit();
}
