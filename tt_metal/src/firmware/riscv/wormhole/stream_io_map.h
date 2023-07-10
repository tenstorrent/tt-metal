#ifndef _STREAM_IO_MAP_
#define _STREAM_IO_MAP_

#include <stdint.h>

#include "tt_metal/hostdevcommon/common_runtime_address_map.h"

// TODO: in ll-buda we can probably just start at stream 0 and not at stream 8?
/*
   Kernel operand mapping scheme:
   - ID 0-7 (inputs, unpacker-only) => streams 8-15
   - ID 8-15 (params, unpacker-only) => streams 16-23
   - ID 16-23 (outputs, packer-only) => streams 24-31
   - ID 24-31 (intermediates, packer/unpacker) => streams 32-39
*/
const uint32_t OPERAND_START_STREAM = 8;

// Indexed with operand = kernel operand ID (0-31) per the table above
// Used for tile push/pop operations.
inline __attribute__((always_inline)) uint32_t get_operand_stream_id(int operand) {
  return OPERAND_START_STREAM + operand;
}

// Pointers to stream scratch registers (implemented using don't-care functional registers) that are used for CB synchronization

inline __attribute__((always_inline)) volatile uint32_t* get_cb_tiles_received_ptr(int operand) {
  return (volatile uint32_t*)(uintptr_t)(STREAM_REG_ADDR(get_operand_stream_id(operand), STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX));
}

inline __attribute__((always_inline)) volatile uint32_t* get_cb_tiles_acked_ptr(int operand) {
  return (volatile uint32_t*)(uintptr_t)(STREAM_REG_ADDR(get_operand_stream_id(operand), STREAM_REMOTE_DEST_BUF_START_REG_INDEX));
}

inline __attribute__((always_inline)) volatile uint32_t* get_cq_read_ptr() {
    return reinterpret_cast<volatile uint32_t*>(CQ_READ_PTR);
}

inline __attribute__((always_inline)) volatile uint32_t* get_cq_write_ptr() {
    return reinterpret_cast<volatile uint32_t*>(CQ_WRITE_PTR);
}

inline __attribute__((always_inline)) volatile u32* get_cq_read_toggle() {
    return reinterpret_cast<volatile u32*>(CQ_READ_TOGGLE);
}

inline __attribute__((always_inline)) volatile u32* get_cq_write_toggle() {
    return reinterpret_cast<volatile u32*>(CQ_WRITE_TOGGLE);
}

inline __attribute__((always_inline)) volatile uint32_t* get_cq_finish_ptr() {
    return (volatile uint32_t*)(uintptr_t)(STREAM_REG_ADDR(
        get_operand_stream_id(0), STREAM_REMOTE_DEST_BUF_START_REG_INDEX));
}

inline __attribute__((always_inline)) volatile uint32_t* get_sync_register_ptr() {
  return (volatile uint32_t*)(uintptr_t)(STREAM_REG_ADDR(0, STREAM_PHASE_AUTO_CFG_PTR_REG_INDEX));
}
#endif
