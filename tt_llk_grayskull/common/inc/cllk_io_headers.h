
#include "ckernel_defs.h"

//
// Receiving from a stream input
//

// Setup pipe receiving data over stream
inline void llk_setup_input_operand(src_op_id_e operand);

// Wait for N tiles available in the incoming stream
inline void llk_wait_tiles(src_op_id_e operand, std::uint32_t num_tiles);

// Pop N tiles from the incoming stream
inline void llk_pop_tiles(src_op_id_e operand, std::uint32_t num_tiles);

// 
// Receiving from a local buffer
//                                      

// Setup pipe for receiving data over local buffer
inline void llk_setup_local_operand(src_op_id_e operand);

// Wait for N tiles available in the local buffer
inline void llk_wait_local_tiles(src_op_id_e operand, std::uint32_t num_tiles);

// Pop N tiles from the incoming stream
inline void llk_pop_local_tiles(src_op_id_e operand, std::uint32_t num_tiles);

//
// Write to stream output 
//

// Setup pipe for writing output data to stream buffer
inline void llk_setup_output(out_op_id_e output);

// Blockig call to wait for free space needed to pack N tiles
inline void llk_wait_for_free_tiles(out_op_id_e output, std::uint32_t num_tiles);

// Push N tiles to stream buffer (increment write pointer)
inline void llk_push_tiles(out_op_id_e output, std::uint32_t num_tiles);

//
// Write to local output 
//

// Setup pipe for writing output data to local output
inline void llk_setup_local_output(out_op_id_e output);

// Blockig call to wait for free space needed to pack N tiles
inline void llk_wait_for_free_tiles(out_op_id_e output, std::uint32_t num_tiles);

// Push N tiles to stream buffer (increment write pointer)
inline void llk_push_tiles(out_op_id_e output, std::uint32_t num_tiles);