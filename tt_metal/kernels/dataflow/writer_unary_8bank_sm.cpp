#define GENERATE_BCAST_SCALER 1
#define TILE_OFFSET get_arg_val<uint32_t>(4)

#ifndef BLOCK_SIZE // can be already defined via add_define
#error "Block size must be defined"
#endif

#include "tt_metal/kernels/dataflow/writer_unary_8bank.cpp"
