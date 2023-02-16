
#include "ckernel_defs.h"
#include "tensix_types.h"

// MT: is it only formats for packer ??
inline void llk_pack_hw_configure(DataFormat unpack_src, DataFormat unpack_dst);

template <DestSyncSections Dst>
inline void llk_pack_wait_for_dest_available<Dst>();

template <DestSyncSections Dst>
inline void llk_pack_set_dest_base<Dst>();

template <DestSyncSections Dst>
inline void llk_pack_dest_section_done<Dst>();

template <DstClearSections Dst>
inline void llk_math_clear_dst<Dst>();

//
// LLK pack tile to output stream - using row tables
// 
inline void llk_pack_stream_row_tables_init();
inline void llk_pack_stream_row_tables(std::uint32_t dst_tile_index);

//
// LLK pack tile to output stream - using tile tables
// 
inline void llk_pack_stream_tile_tables_init();
inline void llk_pack_stream_tile_tables(std::uint32_t dst_tile_index);


//
// LLK pack tile to local L1 buffer - using row tables
// 
inline void llk_pack_local_row_tables_init();
inline void llk_pack_local_row_tables(std::uint32_t dst_tile_index);
