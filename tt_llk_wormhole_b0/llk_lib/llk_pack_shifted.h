#include "llk_io_pack.h"
#include "llk_param_structs.h"

#include "ckernel.h"
#include "ckernel_template.h"
#include "llk_pack_common.h"
#include "ckernel_globals.h"

using namespace ckernel;
using namespace ckernel::packer;

inline void llk_pack_shifted_mop_config(std::uint32_t stride) {
    addr_mod_pack_t{
        .y_src = {.incr = (std::uint8_t) stride},
        .y_dst = {.incr = (std::uint8_t) stride},
    }
        .set(ADDR_MOD_0);

    addr_mod_pack_t{
        .y_src = {.incr = 0, .clr = 1, .cr = 0},
        .y_dst = {.incr = 0, .clr = 1, .cr = 0},
        .z_src = {.incr = 0, .clr = 0},
        .z_dst = {.incr = 0, .clr = 0},
    }
        .set(ADDR_MOD_1);

    addr_mod_pack_t{
        .y_src = {.incr = 0, .clr = 0, .cr = 0},
        .y_dst = {.incr = 0, .clr = 0, .cr = 0},
        .z_src = {.incr = 0, .clr = 0},
        .z_dst = {.incr = 0, .clr = 0},
    }
        .set(ADDR_MOD_2);

    const uint MOP_INNER_LOOP = 16;
    const uint MOP_OUTER_LOOP = 1;
    const uint PACKCNT = 4;
    const uint MEGAROW = 1;
    constexpr uint ZERO_OUTPUT_FLAG = p_pacr::P_ZERO_OUTPUT_DISABLED;

    ckernel::ckernel_template tmp(
        MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));

    tmp.set_last_inner_loop_instr(TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, 0, 0, 0));
    tmp.set_last_outer_loop_instr(TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, 0, 0, 0));

    // Write header to l1
    tmp.set_end_op(TT_OP_STOREIND(1, 0, p_ind::LD_16B, LO_16(0), p_ind::INC_NONE, p_gpr_pack::TILE_HEADER, p_gpr_pack::OUTPUT_ADDR));

    tmp.program(instrn_buffer);
}

template <bool untilize = false, bool is_fp32_dest_acc_en = false>
inline void llk_pack_shifted_hw_configure(const llk_pack_shifted_params_t *pack_params) {
    configure_pack<is_fp32_dest_acc_en>(get_output_id(pack_params->pack_output), pack_params->relu_config.val);

    std::uint32_t output = get_output_id(pack_params->pack_output);
}

template <bool untilize = false, ReluType relu_type=ReluType::NO_RELU, std::uint32_t relu_threshold=0>
inline void llk_pack_shifted_hw_configure_disaggregated(std::uint32_t pack_output) {
    llk_pack_shifted_params_t llk_pack_shifted_params = {
        .pack_output = pack_output, .relu_config = {.f = {.ApplyRelu = (std::uint32_t)relu_type, .Threshold = relu_threshold}}};
    llk_pack_shifted_hw_configure(&llk_pack_shifted_params);
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();
    // Disable auto-last generation
    for (uint i=0; i<4; i++) { cfg[PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32+i]=0; }
    
    // FIXME: configure based on initial padding param value
    //regfile[p_gpr_pack::TMP_DEST_OFFSET]   = 0x0 - 1;
    //regfile[p_gpr_pack::TMP_DEST_OFFSET+1] = 0x0 + 0x20 - 1;
    //regfile[p_gpr_pack::TMP_DEST_OFFSET+2] = 0x0 + 0x10 - 1;
    //regfile[p_gpr_pack::TMP_DEST_OFFSET+3] = 0x0 + 0x30 - 1;
}

inline void llk_pack_shifted_init(const llk_pack_shifted_params_t *params=0) {
    llk_pack_shifted_mop_config(params->stride);
}

inline void llk_pack_shifted(const llk_pack_shifted_params_t *params, llk_pack_shifted_state_t *state, std::uint32_t output, std::uint32_t output_tile_index = 0) {
    std::uint8_t output_id = get_output_id(output);
    constexpr std::uint8_t OUTPUT_BASE_ID = (std::uint8_t) get_output_base_id();

    std::uint16_t pack_tile_base_addr;
    std::uint16_t pack_tile_offset_addr = 0;
    pack_tile_base_addr = outputs[output_id].f.fifo_wr_ptr + MUL_TILE_SIZE_AND_INDEX((std::uint8_t)pack_dst_format[OUTPUT_BASE_ID], output_tile_index);

    int write_row_index = state->current_wr_ptr;

    if (state->partial_tile) {
        pack_tile_offset_addr = ((write_row_index&(FACE_HEIGHT-1))+2*(write_row_index&FACE_HEIGHT))*2; //FIXME: scale row index with format
        state->partial_tile = false;
    }

    program_packer_destination<PACK_01>((pack_tile_base_addr+pack_tile_offset_addr), OUTPUT_BASE_ID);

    if (params->initial_padding>0) {
       if (params->initial_padding <= FACE_HEIGHT) {
           TT_SETADCXX(p_setadc::PAC, ((params->initial_padding*16)-1), 0x0); 
           TTI_PACR(ADDR_MOD_2, 1, 0x3, 0, 0, 0, 0);
           write_row_index+=params->initial_padding;
       } else if (params->initial_padding < TILE_HEIGHT) {
           TTI_SETADCXX(p_setadc::PAC, (16*FACE_HEIGHT)-1, 0x0);
           TTI_PACR(ADDR_MOD_2, 1, 0x3, 0, 0, 0, 1);
           TT_SETADCXX(p_setadc::PAC, (((params->initial_padding-FACE_HEIGHT)*16)-1), 0x0); 
           program_packer_destination<PACK_01>((std::uint16_t)(pack_tile_base_addr+2*(2*FACE_HEIGHT)), OUTPUT_BASE_ID); //FIXME: scale based on the format
           TTI_PACR(ADDR_MOD_2, 1, 0x3, 0, 0, 0, 0);
           write_row_index+=params->initial_padding;
       } else {
           program_packer_destination<PACK_ALL>((std::uint16_t)pack_tile_base_addr, OUTPUT_BASE_ID);
           TTI_SETADCXX(p_setadc::PAC, (256)-1, 0x0); // zero tile detected
           TTI_PACR(ADDR_MOD_2, 1, 0xF, 0, 0, 0, 1);
           write_row_index+=TILE_HEIGHT;
       }
       // Pack single rows
       TTI_SETADCXX(p_setadc::PAC, 16-1, 0x0); 
    } 

    int curr_tile_index=-1;
    while ( (write_row_index < TILE_HEIGHT) &&
            // Keep going until we reached end of valid dest, unless it's final iteration in which case we just pad to the end
            ( (state->current_rd_ptr < params->valid_row_count) || params->final_iteration) )
    {
        bool insert_blank = 
            ((state->current_y) >= params->original_y) ||  // we're past the end
            (((state->current_x) < params->row_shift_x) && (params->row_shift_x > 0)) || // initial postive X-shift
            (((state->current_x) >= (params->original_x + params->row_shift_x)) && (params->row_shift_x < 0)); // final negative X-shift
       
        if (write_row_index == FACE_HEIGHT) {
           TTI_PACR(ADDR_MOD_2, 0, 0x3, 0, 0, 1, 1); //close tile in order to update address
           program_packer_destination<PACK_01>((std::uint16_t)(pack_tile_base_addr+2*(2*FACE_HEIGHT)), OUTPUT_BASE_ID); //FIXME: scale based on the format
        }

   

        if (insert_blank)
        {
            // Insert empty rows 
            TTI_PACR(ADDR_MOD_0, 1, 0x3, 0, 0, 0, 0);
        }
        else
        {
            int tile_index = state->current_rd_ptr / TILE_HEIGHT;
            int pack_zeros = 0;

            if (curr_tile_index != tile_index) {
               curr_tile_index = tile_index;
               if ( (tile_index < 0) || (tile_index >= 16) ) {
                  pack_zeros = 1;
               } else {
                  uint16_t row_index = (state->current_rd_ptr & (TILE_HEIGHT-1));
                  TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Y, row_index);
                  TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, tile_index);
               }
            }

            TT_PACR(ADDR_MOD_0, pack_zeros, 0x3, 0, 0, 0, 0);

            
        }
        write_row_index++;

        // Move read pointers accordingly
        state->current_rd_ptr += params->stride;
        state->current_x += params->stride;
        if (state->current_x >= params->original_x)
        {
            if (state->current_x > params->original_x)
            {
                // Stride got us too far, let's rewind back
                state->current_rd_ptr -= state->current_x - params->original_x;
            }
            state->current_x = params->stride_offset;
            state->current_y += params->stride;
            state->current_rd_ptr += (params->stride - 1) * params->original_x; // stride Y
            if (params->stride > 1) {
               uint16_t row_index = (state->current_rd_ptr & (TILE_HEIGHT-1));
               TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Y, row_index);
            }
        }

    }

    if (write_row_index == TILE_HEIGHT) {
        state->current_wr_ptr = 0;
        state->partial_tile = false;
        // write header
        TT_SETDMAREG(0, pack_tile_base_addr, 0, LO_16(p_gpr_pack::HEADER_ADDR));
        TTI_STOREIND(1, 0, p_ind::LD_16B, LO_16(0), p_ind::INC_NONE, p_gpr_pack::TILE_HEADER, p_gpr_pack::HEADER_ADDR);
    }
    else {
        state->current_wr_ptr = write_row_index;
        state->partial_tile = true;
    }

    TTI_PACR(ADDR_MOD_2, 0, 0x3, 0, 0, 1, 1); //close tile
}
