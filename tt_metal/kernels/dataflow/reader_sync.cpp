#include <cstdint>
#include "hostdevcommon/common_runtime_address_map.h"
/**
 * NOC APIs are prefixed w/ "ncrisc" (legacy name) but there's nothing NCRISC specific, they can be used on BRISC or other RISCs
 * Any two RISC processors cannot use the same CMD_BUF
 * non_blocking APIs shouldn't be mixed with slow noc.h APIs
 * explicit flushes need to be used since the calls are non-blocking
 * */
constexpr static std::uint32_t VALID_VAL = 0x1234;
constexpr static std::uint32_t INVALID_VAL = 0x4321;
void kernel_main() {
    std::uint32_t buffer_src_addr            = get_arg_val<uint32_t>(0);
    std::uint32_t src_noc_x                  = get_arg_val<uint32_t>(1);
    std::uint32_t src_noc_y                  = get_arg_val<uint32_t>(2);
    std::uint32_t stream_register_address    = get_arg_val<uint32_t>(3);
    std::uint32_t num_tiles                  = get_arg_val<uint32_t>(4);
    std::uint32_t transient_buffer_size_tiles= get_arg_val<uint32_t>(5);
    std::uint32_t transient_buffer_size_bytes= get_arg_val<uint32_t>(6);

    // L1 address of the operand buffer
    std::uint32_t l1_address = get_write_ptr(0);

    // Scratch address in L1, two write register value before we copy it to into local/remote registers
    volatile tt_l1_ptr uint32_t* constant_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(CONSTANT_REGISTER_VALUE);
    *(constant_ptr) = INVALID_VAL;

    std::uint32_t counter = 0;
    // src noc address
    std::uint64_t src_noc_addr = get_noc_addr(src_noc_x, src_noc_y, buffer_src_addr);
    // Local and remote register addresses (used for sync)
    std::uint64_t local = get_noc_addr(stream_register_address);
    std::uint64_t remote = get_noc_addr(src_noc_x, src_noc_y, stream_register_address);
    while(counter < num_tiles) {
        // Wait until sync register is VALID_VAL (means its safe to read data from source buffer into operand buffer)
        wait_for_sync_register_value(stream_register_address, VALID_VAL);
        cb_reserve_back(0, transient_buffer_size_tiles);

        noc_async_read(src_noc_addr, l1_address, transient_buffer_size_bytes);
        noc_async_read_barrier();

        // push single tile
        cb_push_back(0, transient_buffer_size_tiles);
        // Write INVALID_VAL into local register
        noc_async_write(CONSTANT_REGISTER_VALUE, local, 4);
        noc_async_write_barrier();

        // Write INVALID_VAL into remote register
        noc_async_write(CONSTANT_REGISTER_VALUE, remote, 4);
        noc_async_write_barrier();

        counter += transient_buffer_size_tiles;
    }
}
