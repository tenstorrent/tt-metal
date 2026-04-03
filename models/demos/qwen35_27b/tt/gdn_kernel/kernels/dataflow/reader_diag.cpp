// Diagnostic reader: writes runtime arg values into the output tile for verification
// Instead of reading from DRAM, fills the tile with a pattern derived from runtime args
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t arg0 = get_arg_val<uint32_t>(0);  // buffer address
    uint32_t arg1 = get_arg_val<uint32_t>(1);  // tile_start
    uint32_t arg2 = get_arg_val<uint32_t>(2);  // num_tiles

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t tile_bytes = 2048;

    // Fill the CB tile with a known pattern based on runtime args
    cb_reserve_back(cb_in, 1);
    uint32_t wp = get_write_ptr(cb_in);

    // Write the address value into first few elements as bfloat16
    // Each bfloat16 is 2 bytes, tile is 32x32 = 1024 elements
    volatile uint16_t* ptr = reinterpret_cast<volatile uint16_t*>(wp);

    // Fill with zeros first
    for (uint32_t i = 0; i < 1024; i++) {
        ptr[i] = 0;
    }

    // Write arg0 (address) as 2x uint16 at positions 0,1
    ptr[0] = static_cast<uint16_t>(arg0 & 0xFFFF);
    ptr[1] = static_cast<uint16_t>((arg0 >> 16) & 0xFFFF);

    // Write arg1 at positions 2,3
    ptr[2] = static_cast<uint16_t>(arg1 & 0xFFFF);
    ptr[3] = static_cast<uint16_t>((arg1 >> 16) & 0xFFFF);

    // Write arg2 at positions 4,5
    ptr[4] = static_cast<uint16_t>(arg2 & 0xFFFF);
    ptr[5] = static_cast<uint16_t>((arg2 >> 16) & 0xFFFF);

    // Write a magic value at position 6 to confirm kernel executed
    ptr[6] = 0x4248;  // bfloat16 for 50.0

    cb_push_back(cb_in, 1);
}
