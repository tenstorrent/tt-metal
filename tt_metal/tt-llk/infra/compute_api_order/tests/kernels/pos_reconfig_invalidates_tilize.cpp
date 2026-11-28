// POSITIVE: pack_reconfig_data_format between tilize_init and tilize_block.
#include "api/compute/tilize.h"
void kernel_main() {
    tilize_init(0, 4, 16);
    pack_reconfig_data_format(16);
    tilize_block(0, 4, 16);
    tilize_uninit(0, 16);
}
