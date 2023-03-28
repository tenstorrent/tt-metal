#ifndef _STREAM_IO_MAP_
#define _STREAM_IO_MAP_

#include <stdint.h>
#include <stdbool.h>


/*
   Kernel operand mapping scheme:
   - ID 0-7 (inputs, unpacker-only) => streams 8-15
   - ID 8-15 (params, unpacker-only) => streams 16-23
   - ID 16-23 (outputs, packer-only) => streams 24-31
   - ID 24-31 (intermediates, packer/unpacker) => streams 32-39
*/

const uint32_t OPERAND_START_STREAM = 8;
const uint32_t INPUT_START_STREAM = 8;
const uint32_t INPUT_PARAMS_START_STREAM = 16;
const uint32_t OUTPUT_START_STREAM = 24;
const uint32_t INTERMEDIATES_START_STREAM = 32;
const uint32_t END_IO_STREAM = 39;

const int OPERAND_INPUT_START_INDEX = 0;
const int OPERAND_INPUT_PARAMS_START_INDEX = 8;
const int OPERAND_OUTPUT_START_INDEX = 16;
const int OPERAND_INTERMEDIATES_START_INDEX = 24;
const int OPERAND_RELAY_START_INDEX = 32;
const int MAX_NUM_OPERANDS = 64;

// Indexed with operand = kernel operand ID (0-31) per the table above
// Used for tile push/pop operations.
inline __attribute__((always_inline)) uint32_t get_operand_stream_id(int operand) {
  return OPERAND_START_STREAM + operand;
}

inline __attribute__((always_inline)) int stream_id_to_operand(uint32_t stream_id) {
  return (stream_id - OPERAND_START_STREAM);
}


// Functions below convert between kernel operand indexes (per the above table)
// and input/output indexes that can be used to iterate separately through
// streams that have kernel input (stream->unpacker) or kernel output
// (packer->stream) functionality.
inline __attribute__((always_inline)) int operand_to_input_index(int operand) {
  return (operand >= OPERAND_INTERMEDIATES_START_INDEX) ? (operand - (OPERAND_INTERMEDIATES_START_INDEX - OPERAND_OUTPUT_START_INDEX)) : operand;
}

inline __attribute__((always_inline)) int input_to_operand_index(int input) {
  return (input >= OPERAND_OUTPUT_START_INDEX) ? (input + (OPERAND_INTERMEDIATES_START_INDEX - OPERAND_OUTPUT_START_INDEX)) : input;
}

inline __attribute__((always_inline)) int operand_to_output_index(int operand) {
  return operand - OPERAND_OUTPUT_START_INDEX;
}

inline __attribute__((always_inline)) int output_to_operand_index(int output) {
  return output + OPERAND_OUTPUT_START_INDEX;
}

inline __attribute__((always_inline)) bool operand_is_intermediate(int operand) {
  return (operand>=OPERAND_INTERMEDIATES_START_INDEX);
}


// Pointers to scratch registers (implemented using don't-care functional registers) for input
// stream tile sync operations:

inline __attribute__((always_inline)) volatile uint32_t* get_operand_tiles_received_ptr(int operand) {
  return (volatile uint32_t*)(uintptr_t)(STREAM_REG_ADDR(get_operand_stream_id(operand), STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX));
}

inline __attribute__((always_inline)) volatile uint32_t* get_operand_tiles_acked_ptr(int operand) {
  return (volatile uint32_t*)(uintptr_t)(STREAM_REG_ADDR(get_operand_stream_id(operand), STREAM_REMOTE_DEST_BUF_START_REG_INDEX));
}

inline __attribute__((always_inline)) volatile uint32_t* get_operand_phase_changed_ptr(int operand) {
  return (volatile uint32_t*)(uintptr_t)(STREAM_REG_ADDR(get_operand_stream_id(operand), STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_REG_INDEX));
}

inline __attribute__((always_inline)) volatile uint32_t* get_packer_tiles_received_ptr(int operand) {
  return (volatile uint32_t*)(uintptr_t)(STREAM_REG_ADDR(get_operand_stream_id(operand), STREAM_REMOTE_SRC_PHASE_REG_INDEX));
}

inline __attribute__((always_inline)) volatile uint32_t* get_packer_tiles_acked_ptr(int operand) {
  return (volatile uint32_t*)(uintptr_t)(STREAM_REG_ADDR(get_operand_stream_id(operand), STREAM_REMOTE_SRC_REG_INDEX));
}

#endif
