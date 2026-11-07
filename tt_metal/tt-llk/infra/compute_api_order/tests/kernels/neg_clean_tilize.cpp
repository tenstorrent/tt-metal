// NEGATIVE: init, block, uninit, no reconfig in between.
#include "api/compute/tilize.h"
void kernel_main() {
    tilize_init(0, 4, 16);
    tilize_block(0, 4, 16);
    tilize_uninit(0, 16);
}
