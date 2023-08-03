#pragma once
#include "ckernel_include.h"
#include "ckernel_globals.h"
#include "ckernel.h"
#include "ckernel_gpr_map.h"
#include "stream_interface.h"
#include "stream_io_map.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "llk_pack_common.h"

using namespace ckernel;

// GS RISC-V RTL bug workaround (l1 reads followed by local mem reads causes a hang)
// in ncrisc.cc/brisc.cc: volatile uint32_t local_mem_barrier;
void write_to_local_mem_barrier(uint32_t data) {
    local_mem_barrier = data;
}

inline void llk_setup_cb_write_interface() {

    volatile std::uint32_t* circular_buffer_config_addr = (volatile uint32_t*)(CIRCULAR_BUFFER_CONFIG_BASE);

    for (std::uint32_t cb_id = 0; cb_id < NUM_CIRCULAR_BUFFERS; cb_id++) {

        std::uint32_t fifo_addr = circular_buffer_config_addr[0];
        std::uint32_t fifo_size = circular_buffer_config_addr[1];
        std::uint32_t fifo_num_pages = circular_buffer_config_addr[2];
        write_to_local_mem_barrier(fifo_num_pages);

        cb_write_interface[cb_id].fifo_wr_ptr = fifo_addr;
        cb_write_interface[cb_id].fifo_limit = fifo_addr + fifo_size - 1;  // Check if there is overflow
        cb_write_interface[cb_id].fifo_size = fifo_size;
        cb_write_interface[cb_id].fifo_num_pages = fifo_num_pages;

        // local copy used by the packer
        cb_write_interface[cb_id].tiles_received = 0;
        // this is currently used for in-order packing (the default mode)
        cb_write_interface[cb_id].fifo_wr_tile_ptr = 0;

        circular_buffer_config_addr += UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG; // move by 3 uint32's
    }
}

// "llk_setup_outputs" is the old function name that HLKC emits
inline void llk_setup_outputs() {
    llk_setup_cb_write_interface();
}

// Blocking call to wait for free space needed to pack N tiles
template <bool skip_sync = false, bool wait_for_blocks = false, bool brisc_pack = false>
inline void llk_wait_for_free_tiles(const std::int32_t operand, const std::int32_t num_tiles) {

    std::uint32_t output = operand;

    volatile std::uint32_t* tiles_acked_ptr = get_cb_tiles_acked_ptr(operand);
    volatile std::uint32_t* tiles_received_ptr = get_cb_tiles_received_ptr(operand);

    // while the producer (write-side interface) is waiting for space to free up "tiles_pushed" is not changing
    // "tiles_pushed" is updated by the producer only when the tiles are pushed
    // note: we need to use "tiles_received", because this is updated by RISC-V, and not tiles_received_ptr which is updated by packer
    // here we don't synchronize with packer, so if we use tiles_received_ptr could case a data race
    // alternatively we could sync with packer, but that's slower and more complex code
    // that is, don't do this: uint32_t tiles_received = tiles_received_ptr[0];
    uint32_t tiles_received = cb_write_interface[output].tiles_received;

    std::int32_t free_tiles;
    do {
        std::uint16_t tiles_acked = (std::uint16_t) reg_read_barrier((std::uint32_t)tiles_acked_ptr);
        std::uint16_t free_tiles_wrap = cb_write_interface[output].fifo_num_pages - (tiles_received - tiles_acked);
        free_tiles = (std::int32_t) free_tiles_wrap;
    } while (free_tiles < num_tiles);

    // TODO: check if this one is really necessary
    mem_barrier(free_tiles);
}

inline void llk_push_to_brisc(const std::int32_t operand, const std::int32_t num_tiles, const std::int32_t num_words) {
    std::uint32_t output = operand;

    // Tensix uses 4B addresses (tiles_received_ptr byte address but div-by-4)
    volatile std::uint32_t* tiles_received_ptr_tensix =
        (volatile std::uint32_t*)((((volatile std::uint32_t)get_cb_tiles_received_ptr(operand)) >> 2) & 0x3ffff);

    // cb_write_interface[output].tiles_received is used only by the TRISC2 (the one driving packer)
    // we need it becasue tiles_received_ptr is updated by the packer, and in cb_reserve_back func (see above) we want to avoid synchronization with packer
    // cb_reserve_back must used the most recent value of tiles_received (cannot use stale or delayed), otherwise it would think there's less tiles in the CB than there actually are
    // so we use cb_write_interface[output].tiles_received instead of tiles_received_ptr, because it is updated by TRISC2 and no additional synchronization is needed
    cb_write_interface[output].tiles_received += num_tiles;
    uint16_t tiles_received_new = cb_write_interface[output].tiles_received;

    // Update the value at tiles_received_ptr with tiles_received_new only after the packer has finished packing
    // We need to use a Tensix instruction to do the update, which runs only after STALLWAIT has finished
    // Note that the consumer side of the circular buffer (the one reading from the buffer) is ok to use stale/delayed version of the value at tiles_received_ptr
    // This is because consumer is polling the value at tiles_received_ptr, and it will eventually see the updated value
    TT_SETDMAREG(0,tiles_received_new, 0, LO_16(p_gpr_pack::NUM_MSGS_RECEIVED));
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);  // wait for pack to finish
    TT_STOREREG(p_gpr_pack::NUM_MSGS_RECEIVED, (uint32_t)&tiles_received_ptr_tensix[0]);
}

// Push N tiles to stream buffer (increment write pointer)
template <bool push_blocks = false, bool brisc_pack = false>
inline void llk_push_tiles(const std::int32_t operand, const std::int32_t num_tiles) {

    std::uint32_t output = operand;
    std::uint32_t num_words = num_tiles * GET_L1_TILE_SIZE((uint)pack_dst_format[output]);

    cb_write_interface[output].fifo_wr_ptr += num_words;
    cb_write_interface[output].fifo_wr_tile_ptr = 0;

    if (cb_write_interface[output].fifo_wr_ptr > cb_write_interface[output].fifo_limit) {
        cb_write_interface[output].fifo_wr_ptr -= cb_write_interface[output].fifo_size;
    }

    llk_push_to_brisc(operand, num_tiles, num_words);
}

inline void llk_wait_for_free_blocks(const std::int32_t operand, const std::int32_t num_blocks) {
    llk_wait_for_free_tiles<false, true>(operand, num_blocks);
}

inline void llk_push_blocks(const std::int32_t operand, const std::int32_t num_blocks) {
    llk_push_tiles<true>(operand, num_blocks);
}
