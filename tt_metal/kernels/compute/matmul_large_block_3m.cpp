#include <cstdint>

#include "llk_3c.h"

namespace NAMESPACE {

#ifdef TRISC_MATH
#include <cstdint>
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_matmul.h"

inline void tilize_activation(
    uint32_t in0_subblock_h, uint32_t in0_block_w, uint32_t in0_num_subblocks) {
    llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>();
    for (uint32_t in0_subblock = 0U; in0_subblock < in0_num_subblocks; in0_subblock++) {
        for (uint32_t i = 0U; i < in0_subblock_h; i++) {
            for (uint32_t j = 0U; j < in0_block_w; j++) {
                llk_math_wait_for_dest_available<SyncHalf>();
                llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE, SyncHalf>(0);
                llk_math_dest_section_done<SyncHalf>();
            }
        }
    }
}

inline void reblock_and_untilize_output(uint32_t out_subblock_h, uint32_t out_block_w) {
    llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>();
    volatile uint32_t* mbox = reinterpret_cast<volatile uint32_t*>(l1_mem::address_map::TRISC0_DEBUG_BUFFER_BASE);
    volatile uint32_t* mbox2 = reinterpret_cast<volatile uint32_t*>(l1_mem::address_map::TRISC1_DEBUG_BUFFER_BASE);

    for (uint32_t i = 0; i < out_subblock_h; i++) {
        for (int j = 0; j < 2; j++) {
            for (uint32_t k = 0; k < out_block_w; k++) {
                llk_math_wait_for_dest_available<SyncHalf>();
                llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE, SyncHalf>(0);
                llk_math_dest_section_done<SyncHalf>();
            }
        }
    }
}

void math_main()
{
    uint32_t in0_block_w = get_compile_time_arg_val(0);
    llk_math_pack_sync_init<SyncHalf>();

    // inner block size in tiles
    uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
    // outer row block size (in inner row blocks)
    uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);
    // out_subblock_h*in0_block_w*in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);

    uint32_t in0_subblock_h = get_compile_time_arg_val(4);

    // out_subblock_h*in0_block_w
    uint32_t in1_num_subblocks = get_compile_time_arg_val(5);
    // outer column block size (in inner column blocks)
    uint32_t in1_block_num_tiles = get_compile_time_arg_val(6);
    //out_subblock_w*in0_block_w* in1_num_subblocks;
    uint32_t in1_per_core_w = get_compile_time_arg_val(7);
    // out_subblock_w*in1_num_subblocks
    constexpr uint32_t num_blocks = get_compile_time_arg_val(8);
    // outer inner dim (in inner dim blocks)
    uint32_t out_subblock_h = get_compile_time_arg_val(9);
    // inner row block size in tiles
    uint32_t out_subblock_w = get_compile_time_arg_val(10);
    // inner column block size in tiles
    uint32_t out_subblock_num_tiles = get_compile_time_arg_val(11);

    uint32_t out_block_w = in1_per_core_w;

    // If true, this assumes data coming in RM
    constexpr bool tilize_in = get_compile_time_arg_val(12);

    // If true, this assumes consumer wants data RM
    constexpr bool untilize_out = get_compile_time_arg_val(13);

    constexpr bool spill = num_blocks > 1U;
    bool enable_reload = false;

    for (uint32_t block = 0U; block < num_blocks; block++) {
    bool last_out = block == num_blocks - 1U;

    if constexpr (tilize_in) {
        tilize_activation(in0_subblock_h, in0_block_w, in0_num_subblocks);
    }

    for (uint32_t in0_subblock = 0U; in0_subblock < in0_num_subblocks; in0_subblock++) {
        for (uint32_t in1_subblock = 0U; in1_subblock < in1_num_subblocks; in1_subblock++) {

        llk_math_wait_for_dest_available<SyncHalf>();
        if (enable_reload) {
            llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>();
            for (uint32_t i = 0U; i < out_subblock_num_tiles; i++) {
            llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE, SyncHalf>(i);
            }
        }
        llk_math_matmul_init<MATH_FIDELITY>(0);

        int dst_index = 0;
        for (uint32_t h = 0U; h < out_subblock_h; h++) {
            for (uint32_t w = 0U; w < out_subblock_w; w++) {
            for (uint32_t inner_dim = 0U; inner_dim < in0_block_w; inner_dim++) {
                llk_math_matmul<MATH_FIDELITY>(dst_index);
            }
            dst_index++;
            }
        }

        llk_math_dest_section_done<SyncHalf>();
        }
        if constexpr (untilize_out) {
        if (last_out) {
            reblock_and_untilize_output(out_subblock_h, out_block_w);
        }
        }

    }
    if constexpr (spill) {
        enable_reload = true;
    }
    }
}
#endif // #ifdef TRISC_MATH


#ifdef TRISC_UNPACK
#include <cstdint>
#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"
#include "llk_unpack_untilize.h"
#include "llk_unpack_A.h"
#include "llk_unpack_AB_matmul.h"

inline void tilize_activation(uint32_t in0_subblock_h, uint32_t in0_block_w, uint32_t in0_num_subblocks) {
    // Tilize block code
    llk_unpack_tilize_init(0, in0_block_w);
    for (uint32_t in0_subblock = 0U; in0_subblock < in0_num_subblocks; in0_subblock++) {
        for (uint32_t i = 0U; i < in0_subblock_h; i++) {
            llk_wait_tiles(0, in0_block_w); // These "tiles" are actually not real tiles
            llk_unpack_tilize_(0,in0_block_w);
            llk_pop_tiles(0,in0_block_w); // Pop the original untilized inputs
        }
    }
    llk_unpack_tilize_uninit();
}


inline __attribute__((always_inline))
void reblock_and_untilize(
    uint32_t num_out_subblocks_in_col,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_block_w,
    uint32_t interm_cb_id,
    uint32_t reblock_cb_id) {

    volatile uint32_t* mbox = reinterpret_cast<volatile uint32_t*>(l1_mem::address_map::TRISC0_DEBUG_BUFFER_BASE);

    // Wait for a row of subblocks such that the total width matches
    // the out block width. Must wait for a whole row of subblocks to arrive
    // before we can proceed.
    uint32_t num_tiles_in_row_of_subblocks = mulsi3(out_subblock_num_tiles,  num_out_subblocks_in_col);
    llk_wait_tiles(interm_cb_id, num_tiles_in_row_of_subblocks);

    int within_block_index = 0;
    for (uint32_t h = 0; h < out_subblock_h; h++) {
        int block_offset = 0;

        llk_unpack_A_init<BroadcastType::NONE, false, false>();
        for (uint32_t n = 0; n < num_out_subblocks_in_col; n++) {
            for (uint32_t w = 0; w < out_subblock_w; w++) {
                uint32_t tile_index = block_offset + within_block_index + w;
                llk_unpack_A(interm_cb_id, tile_index);
            }
            block_offset += out_subblock_num_tiles;
        }

        // Since our reblock CB can only fit one row of
        // tiles, we need to immediately untilize to
        // consume this row
        llk_wait_tiles(reblock_cb_id, out_block_w);
        /*
        for (uint32_t i = 0; i < out_block_w; i++) {
            llk_unpack_A(reblock_cb_id, i);
        }
        */

            llk_unpack_untilize_init(reblock_cb_id);
            llk_unpack_untilize_<true>(reblock_cb_id, out_block_w);
            llk_unpack_untilize_<false>(reblock_cb_id, out_block_w);
            llk_unpack_untilize_uninit(reblock_cb_id);

        llk_pop_tiles(reblock_cb_id, out_block_w);

        within_block_index += out_subblock_w;
    }
    llk_pop_tiles(interm_cb_id, num_tiles_in_row_of_subblocks);
}

inline void unpack_for_matmul_output_row(
    uint32_t in1_num_subblocks,
    bool enable_reload,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t in0_block_w,
    uint32_t in0_index_subblock_offset,
    uint32_t in1_per_core_w,
    uint32_t matmul_act_cb_id,
    uint32_t matmul_out_intermediate_cb_id) {

    uint32_t in1_index_subblock_offset = 0;
    for (uint32_t in1_subblock = 0U; in1_subblock < in1_num_subblocks; in1_subblock++) {
      if (enable_reload) {
        llk_unpack_A_init<BroadcastType::NONE, false, false>();
        llk_wait_tiles(matmul_out_intermediate_cb_id, out_subblock_num_tiles);
        for (uint32_t i = 0U; i < out_subblock_num_tiles; i++) {
          llk_unpack_A(matmul_out_intermediate_cb_id, i);
        }
        llk_pop_tiles(matmul_out_intermediate_cb_id, out_subblock_num_tiles);
      }

      llk_unpack_AB_matmul_init(0);
      int dst_index = 0;
      int in0_index_h_offset = 0;
      for (uint32_t h = 0U; h < out_subblock_h; h++) {
        for (uint32_t w = 0U; w < out_subblock_w; w++) {
          int in1_index_inner_dim_offset = 0;
          for (uint32_t inner_dim = 0U; inner_dim < in0_block_w; inner_dim++) {
            int in0_index = ((in0_index_subblock_offset + in0_index_h_offset) + inner_dim);
            int in1_index = ((in1_index_subblock_offset + in1_index_inner_dim_offset) + w);
            llk_unpack_AB_matmul(matmul_act_cb_id, 1, in0_index, in1_index);
            in1_index_inner_dim_offset += in1_per_core_w;
          }
          dst_index++;
        }
        in0_index_h_offset += in0_block_w;
      }
      in1_index_subblock_offset += out_subblock_w;
    }
}

void unpack_main()
{
    uint32_t in0_block_w = get_compile_time_arg_val(0);
    llk_setup_operands();
    llk_unpack_AB_matmul_init(0);
    // inner block size in tiles
    uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
    // outer row block size (in inner row blocks)
    uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);
    // out_subblock_h*in0_block_w*in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);

    uint32_t in0_subblock_h = get_compile_time_arg_val(4);

    // out_subblock_h*in0_block_w
    uint32_t in1_num_subblocks = get_compile_time_arg_val(5);
    // outer column block size (in inner column blocks)
    uint32_t in1_block_num_tiles = get_compile_time_arg_val(6);
    //out_subblock_w*in0_block_w* in1_num_subblocks;
    uint32_t in1_per_core_w = get_compile_time_arg_val(7);
    // out_subblock_w*in1_num_subblocks
    constexpr uint32_t num_blocks = get_compile_time_arg_val(8);
    // outer inner dim (in inner dim blocks)
    uint32_t out_subblock_h = get_compile_time_arg_val(9);
    // inner row block size in tiles
    uint32_t out_subblock_w = get_compile_time_arg_val(10);
    // inner column block size in tiles
    uint32_t out_subblock_num_tiles = get_compile_time_arg_val(11);

    uint32_t out_block_w = in1_per_core_w;

    // If true, this assumes data coming in RM
    constexpr bool tilize_in = get_compile_time_arg_val(12);

    // If true, this assumes consumer wants data RM
    constexpr bool untilize_out = get_compile_time_arg_val(13);


    // These are required depending on tilize/untilize
    uint32_t matmul_act_cb_id = 0;
    uint32_t matmul_out_intermediate_cb_id = 24;
    if constexpr (tilize_in) {
        // If we tilize, matmul doesn't consume original input,
        // it consumes what is produced by tilize
        matmul_act_cb_id = 24;

        matmul_out_intermediate_cb_id = 25; // Given 24 is no longer available, we use 25 instead
    }

    llk_unpack_AB_matmul_hw_configure_disaggregated(0,1,0);

    uint32_t reblock_cb_id = 26;

    constexpr bool spill = num_blocks > 1U;
    bool enable_reload = false;
    for (uint32_t block = 0U; block < num_blocks; block++) {
    bool last_out = block == num_blocks - 1U;

    if constexpr (tilize_in) {
        tilize_activation(in0_subblock_h, in0_block_w, in0_num_subblocks);
    } else {
        llk_wait_tiles(matmul_act_cb_id, in0_block_num_tiles);
    }

    // Wait on weight tiles
    llk_wait_tiles(1, in1_block_num_tiles);
    int in0_index_subblock_offset = 0;
    for (uint32_t in0_subblock = 0U; in0_subblock < in0_num_subblocks; in0_subblock++) {
        unpack_for_matmul_output_row(
            in1_num_subblocks,
            enable_reload,
            out_subblock_num_tiles,
            out_subblock_h,
            out_subblock_w,
            in0_block_w,
            in0_index_subblock_offset,
            in1_per_core_w,
            matmul_act_cb_id,
            matmul_out_intermediate_cb_id);

        if constexpr (untilize_out) {
            if (last_out) {
                reblock_and_untilize(
                    in1_num_subblocks,
                    out_subblock_num_tiles,
                    out_subblock_h,
                    out_subblock_w,
                    out_block_w,
                    matmul_out_intermediate_cb_id,
                    reblock_cb_id);
            }
        }

        in0_index_subblock_offset += in0_subblock_num_tiles;
    }

    // Need to do a reblock datacopy
    if constexpr (spill) {
        enable_reload = true;
    }

    llk_pop_tiles(matmul_act_cb_id, in0_block_num_tiles);
    llk_pop_tiles(1, in1_block_num_tiles);
    }
}


#endif // TRISC_UNPACK


#ifdef TRISC_PACK

#include <cstdint>
#include "llk_pack_common.h"
#include "llk_pack.h"

inline void tilize_activation(
    uint32_t in0_subblock_h, uint32_t in0_block_w, uint32_t in0_num_subblocks, uint32_t in0_block_num_tiles, uint32_t matmul_act_cb_id) {
    llk_wait_for_free_tiles<false,false,false>(matmul_act_cb_id, in0_block_num_tiles);
    for (uint32_t in0_subblock = 0U; in0_subblock < in0_num_subblocks; in0_subblock++) {
        for (uint32_t i = 0U; i < in0_subblock_h; i++) {
            for (uint32_t j = 0U; j < in0_block_w; j++) {
                llk_packer_wait_for_math_done();
                llk_pack<false, SyncHalf, false >(0, matmul_act_cb_id);
                llk_pack_dest_section_done<SyncHalf>();
                llk_push_tiles<false,false>(matmul_act_cb_id, 1);
            }
        }
    }
}

inline void pack_row(uint32_t num_tiles_to_pack, uint32_t cb_id) {
    /*
        Used either for packing reblocked tiles for untilized tiles
    */
    llk_wait_for_free_tiles<false,false,false>(cb_id, num_tiles_to_pack);
    for (uint32_t i = 0; i < num_tiles_to_pack; i++) {
        llk_packer_wait_for_math_done();
        llk_pack<false, SyncHalf, false >(0, cb_id);
        llk_pack_dest_section_done<SyncHalf>();
    }
    llk_push_tiles<false,false>(cb_id, num_tiles_to_pack);
}

inline void reblock_and_untilize_output(uint32_t out_subblock_h, uint32_t out_block_w, uint32_t reblock_cb_id, uint32_t untilize_cb_id) {
    // volatile uint32_t* mbox = reinterpret_cast<volatile uint32_t*>(l1_mem::address_map::TRISC1_DEBUG_BUFFER_BASE);
    // mbox[0] = out_block_w;
    for (uint32_t h = 0; h < out_subblock_h; h++) {
        // Can only push row because the CB can only fit
        // one row
        pack_row(out_block_w, reblock_cb_id);
        pack_row(out_block_w, untilize_cb_id);
    }
}

inline void pack_block_and_untilize(
    uint32_t in0_num_subblocks, uint32_t in1_num_subblocks,
    uint32_t out_subblock_num_tiles, uint32_t out_subblock_h, uint32_t out_block_w,
    uint32_t interm_cb_id, uint32_t reblock_cb_id) {
    // volatile uint32_t* mbox = reinterpret_cast<volatile uint32_t*>(l1_mem::address_map::TRISC1_DEBUG_BUFFER_BASE);
    volatile uint32_t* mbox = reinterpret_cast<volatile uint32_t*>(l1_mem::address_map::TRISC0_DEBUG_BUFFER_BASE);


    for (uint32_t in0_subblock = 0U; in0_subblock < in0_num_subblocks; in0_subblock++) {
        for (uint32_t in1_subblock = 0U; in1_subblock < in1_num_subblocks; in1_subblock++) {
            llk_packer_wait_for_math_done();

            llk_wait_for_free_tiles<false,false,false>(interm_cb_id, out_subblock_num_tiles);
            mbox[0] = 5;
            for (uint32_t i = 0U; i < out_subblock_num_tiles; i++) {
                llk_pack<false, SyncHalf, false >(i, interm_cb_id);
            }
            llk_push_tiles<false,false>(interm_cb_id, out_subblock_num_tiles);
            llk_pack_dest_section_done<SyncHalf>();
        }
        reblock_and_untilize_output(out_subblock_h, out_block_w, reblock_cb_id, 16);
    }
}

inline void pack_block(uint32_t in0_num_subblocks, uint32_t in1_num_subblocks, uint32_t out_subblock_num_tiles, uint32_t cb_id) {

    for (uint32_t in0_subblock = 0U; in0_subblock < in0_num_subblocks; in0_subblock++) {
        for (uint32_t in1_subblock = 0U; in1_subblock < in1_num_subblocks; in1_subblock++) {
            llk_packer_wait_for_math_done();

            llk_wait_for_free_tiles<false,false,false>(cb_id, out_subblock_num_tiles);
            for (uint32_t i = 0U; i < out_subblock_num_tiles; i++) {
                llk_pack<false, SyncHalf, false >(i, cb_id);
            }
            llk_push_tiles<false,false>(cb_id, out_subblock_num_tiles);
            llk_pack_dest_section_done<SyncHalf>();
        }
    }
}

void pack_main()
{
    uint32_t in0_block_w = get_compile_time_arg_val(0);
    llk_pack_init();
    llk_setup_outputs();
    llk_pack_dest_init<SyncHalf, DstTileFaceLayout::RowMajor, false>();
    llk_init_packer_dest_offset_registers<SyncHalf,DstTileFaceLayout::RowMajor,false>();
    llk_pack_hw_configure_disaggregated<false>(16);
    // inner block size in tiles
    uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
    // outer row block size (in inner row blocks)
    uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);
    // out_subblock_h*in0_block_w*in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);
    uint32_t in0_subblock_h = get_compile_time_arg_val(4);
    uint32_t in1_num_subblocks = get_compile_time_arg_val(5);
    uint32_t in1_block_num_tiles = get_compile_time_arg_val(6);
    uint32_t in1_per_core_w = get_compile_time_arg_val(7);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(8);
    uint32_t out_subblock_h = get_compile_time_arg_val(9);
    uint32_t out_subblock_w = get_compile_time_arg_val(10);
    uint32_t out_subblock_num_tiles = get_compile_time_arg_val(11);

    uint32_t out_block_w = in1_per_core_w;

    // If true, this assumes data coming in RM
    constexpr bool tilize_in = get_compile_time_arg_val(12);

    // If true, this assumes consumer wants data RM
    constexpr bool untilize_out = get_compile_time_arg_val(13);

    constexpr bool spill = num_blocks > 1U;
    bool enable_reload = false;

    // These are required depending on tilize/untilize
    uint32_t matmul_act_cb_id = 0;
    uint32_t matmul_out_intermediate_cb_id = 24;
    if constexpr (tilize_in) {
        // If we tilize, matmul doesn't consume original input,
        // it consumes what is produced by tilize
        matmul_act_cb_id = 24;
        matmul_out_intermediate_cb_id = 25; // Given 24 is no longer available, we use 25 instead
    }

    uint32_t reblock_cb_id = 26; // Only used if untilize is required
    uint32_t matmul_out_cb_id = 16;

    for (uint32_t block = 0U; block < num_blocks - 1; block++) {
    if constexpr (tilize_in) {
        tilize_activation(
            in0_subblock_h,
            in0_block_w,
            in0_num_subblocks,
            in0_block_num_tiles,
            matmul_act_cb_id);
    }

    pack_block(
            in0_num_subblocks,
            in1_num_subblocks,
            out_subblock_num_tiles,
            matmul_out_intermediate_cb_id);
    }

    // Last block
    if constexpr (tilize_in) {
        tilize_activation(
            in0_subblock_h,
            in0_block_w,
            in0_num_subblocks,
            in0_block_num_tiles,
            matmul_act_cb_id);
    }

    if constexpr (untilize_out) {
    pack_block_and_untilize(
            in0_num_subblocks,
            in1_num_subblocks,
            out_subblock_num_tiles,
            out_subblock_h,
            out_block_w,
            matmul_out_intermediate_cb_id,
            reblock_cb_id
        );
    } else {
        pack_block(
            in0_num_subblocks,
            in1_num_subblocks,
            out_subblock_num_tiles,
            matmul_out_cb_id);
    }
}
#endif


} // NAMESPACE
