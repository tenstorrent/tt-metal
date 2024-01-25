#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "fw_debug.h"
#include "cpack_common.h"
#include "llk_param_structs.h"
#ifdef PERF_DUMP
#include "ckernel_perf_api.h"
#endif

using namespace ckernel;
using namespace ckernel::packer;

// wait until math is done and has produced something to pack
inline void llk_packer_wait_for_math_done() {
#ifdef PERF_DUMP
    if constexpr (MATH_PACK_DECOUPLE == 0) {
        TTI_SEMWAIT(p_stall::STALL_TDMA, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO);
    }
#else
    TTI_SEMWAIT(p_stall::STALL_TDMA, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO);
#endif
}

// Tell math that it can write again
inline void llk_packer_set_math_semaphore() {
    t6_semaphore_get(semaphore::MATH_PACK);  // Indicate that packer is done and header is written into L1
}

// Wait for all writes to complete in L1 (header + data)
// Tell math it can write again
// Clear dest
template <DstSync Dst>
inline void llk_pack_dest_section_done() {
#ifdef PERF_DUMP
    if constexpr (MATH_PACK_DECOUPLE) {
        return;
    }
#endif
    if constexpr ((Dst == DstSync::SyncTile16)) {
        llk_packer_set_math_semaphore();
    } else if constexpr (Dst == DstSync::SyncTile2) {
        // Tell math that it can write again
        TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::PACK);  // stall sem update until pack is done
        llk_packer_set_math_semaphore();
    } else {
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK);  // wait for pack to finish

        if constexpr (Dst == DstSync::SyncFull) {
            TT_ZEROACC(p_zeroacc::CLR_ALL, ADDR_MOD_1, 0);
        } else {
            TT_ZEROACC(p_zeroacc::CLR_HALF, ADDR_MOD_1, (dest_offset_id) % 2);
        }

        // Tell math that it can write again
        llk_packer_set_math_semaphore();

        if constexpr (Dst == DstSync::SyncHalf) {
            flip_packer_dest_offset_id();
            select_packer_dest_registers<Dst>();
        }
    }
}

template <DstSync Dst, DstTileFaceLayout FaceLayout = RowMajor, bool untilize = false>
inline void llk_pack_dest_init() {
    tensix_sync();
    reset_dest_offset_id();
    init_packer_dest_offset_registers<FaceLayout, untilize>();
    select_packer_dest_registers<Dst>();
    packer_addr_counter_init();
    pack_sync_tile_dst_ptr = 0;
}

template <DstSync Dst, DstTileFaceLayout FaceLayout, bool untilize = false>
inline void llk_init_packer_dest_offset_registers() {
    TTI_STALLWAIT(p_stall::STALL_TDMA, p_stall::PACK);  // wait for pack to finish
    if constexpr (untilize) {
       if constexpr (FaceLayout == ColMajor) {
          // Packer0 :  0,32,  1,33 ...  7, 39
          // Packer1 :  8,40,  9,41 ... 15, 47
          // Packer2 : 16,48, 17,49 ... 23, 55		  
          // Packer3 : 23,56, 24,57 ... 31, 63		  
          TT_SETDMAREG(0, 0x000 + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
          TT_SETDMAREG(0, 0x000 + 0x08, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 1));
          TT_SETDMAREG(0, 0x000 + 0x10, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 2));
          TT_SETDMAREG(0, 0x000 + 0x18, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 3));
          TT_SETDMAREG(0, 0x200 + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
          TT_SETDMAREG(0, 0x200 + 0x08, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 1));
          TT_SETDMAREG(0, 0x200 + 0x10, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 2));
          TT_SETDMAREG(0, 0x200 + 0x18, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 3));
       } else {		 
          // Packer0 :  0,16,  1,17 ...  7, 23
          // Packer1 :  8,24,  9,25 ... 15, 31
          // Packer2 : 32,48, 33,49 ... 39, 55		  
          // Packer3 : 40,56, 41,57 ... 47, 63		  
          TT_SETDMAREG(0, 0x000 + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
          TT_SETDMAREG(0, 0x000 + 0x08, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 1));
          TT_SETDMAREG(0, 0x000 + 0x20, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 2));
          TT_SETDMAREG(0, 0x000 + 0x28, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 3));
          TT_SETDMAREG(0, 0x200 + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
          TT_SETDMAREG(0, 0x200 + 0x08, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 1));
          TT_SETDMAREG(0, 0x200 + 0x20, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 2));
          TT_SETDMAREG(0, 0x200 + 0x28, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 3));
       }    
    } else { 
       if constexpr (FaceLayout == ColMajor) {
           TT_SETDMAREG(0, 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
           TT_SETDMAREG(0, 0x20, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 1));
           TT_SETDMAREG(0, 0x10, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 2));
           TT_SETDMAREG(0, 0x30, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 3));
           TT_SETDMAREG(0, 0x200 + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
           TT_SETDMAREG(0, 0x200 + 0x20, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 1));
           TT_SETDMAREG(0, 0x200 + 0x10, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 2));
           TT_SETDMAREG(0, 0x200 + 0x30, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 3));
       } else {  // Default to row major layout
           TT_SETDMAREG(0, 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
           TT_SETDMAREG(0, 0x10, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 1));
           TT_SETDMAREG(0, 0x20, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 2));
           TT_SETDMAREG(0, 0x30, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 3));
           TT_SETDMAREG(0, 0x200 + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
           TT_SETDMAREG(0, 0x200 + 0x10, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 1));
           TT_SETDMAREG(0, 0x200 + 0x20, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 2));
           TT_SETDMAREG(0, 0x200 + 0x30, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 3));
       }
    }   
    select_packer_dest_registers<Dst>();
}

template <bool mail2math=true, bool mail2pack=true>
inline void llk_pack_get_tile(std::uint32_t operand, std::uint32_t tile_index, uint32_t *p_tile) {
    if constexpr (mail2pack) {
       *p_tile =  mailbox_read(ThreadId::UnpackThreadId);
    } else {
       *p_tile = 0;
    }

}

template <bool mail2math=true, bool mail2pack=true>
inline void llk_pack_release_tile(std::uint32_t operand) {
    if constexpr (mail2pack) {
       semaphore_get(semaphore::UNPACK_OPERAND_SYNC);
    }   
}

inline void llk_pack_debug_dump(std::uint8_t *data, std::uint32_t byte_size) {
    debug_dump(data, byte_size);
}

inline void llk_pack_debug_dump_seek(std::uint8_t offset) {
    debug_dump_seek(offset);
}

inline void llk_pack_reconfig_data_format(const std::uint32_t old_operand, const std::uint32_t new_operand) {
    std::uint32_t old_operand_id = get_output_id(old_operand);
    std::uint32_t new_operand_id = get_output_id(new_operand);

    if((pack_dst_format[old_operand_id] != pack_dst_format[new_operand_id])
       && (pack_dst_format[old_operand_id] != (uint)DataFormat::Invalid) 
       && (pack_dst_format[new_operand_id] != (uint)DataFormat::Invalid)) {
        reconfig_packer_data_format(new_operand_id);
    }
}

inline void llk_pack_relu_config(std::uint32_t config) {
    ReluType mode = (config&0xf) == 0 ? ReluType::NO_RELU : ((config&0xf) == 3 ? ReluType::MAX_THRESHOLD_RELU : ReluType::MIN_THRESHOLD_RELU);
    uint32_t threshold = (config>>16) << STACC_RELU_ReluThreshold_SHAMT;
    TTI_SETDMAREG(0, 0, 0, LO_16(p_gpr_pack::TMP0));
    TTI_SETDMAREG(0,((uint32_t)mode), 0, HI_16(p_gpr_pack::TMP0));
    TTI_SETDMAREG(0, threshold, 0, LO_16(p_gpr_pack::TMP1));
    TTI_SETDMAREG(0, 0, 0, HI_16(p_gpr_pack::TMP1));
	TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    TTI_WRCFG(p_gpr_pack::TMP0,  p_cfg::WRCFG_32b, STACC_RELU_ApplyRelu_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP1,  p_cfg::WRCFG_32b, STACC_RELU_ReluThreshold_ADDR32);
    TTI_NOP; TTI_NOP;
}

inline void llk_pack_reconfig_l1_acc(const std::uint32_t enable)
{
}    