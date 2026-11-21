// POSITIVE: tilize_init never paired with tilize_uninit.
#include "api/compute/tilize.h"
void kernel_main() {
    tilize_init(0, 4, 16);
    tilize_block(0, 4, 16);
}
