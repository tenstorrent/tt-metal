# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Blackhole pack family (``tools.llk_state_audit.bh.families.pack``).

``REAL_SNIPPETS`` holds verbatim slices of the pack scope on main (8a7beeeaede): cpack_common.h,
llk_pack_common.h, llk_pack.h, llk_pack_untilize.h, experimental/llk_pack_fast_tilize.h,
llk_io_pack.h and llk_pack_common_api.h, plus the cfg_defines.h fields and ckernel_gpr_map.h
structs they use. ``STUBS`` are minimal stand-ins for the headers those snippets include (they are
not under test). Expected values follow the C++ semantics of each snippet, not a snapshot.
"""

import tempfile
import unittest
from pathlib import Path

from tools.llk_state_audit.bh import effects_db, scan
from tools.llk_state_audit.bh.families import pack

LLK = "tt_metal/tt-llk/tt_llk_blackhole"

#: Minimal stand-ins for the headers the snippets include (not under test themselves).
STUBS = {
    f"{LLK}/common/inc/ckernel_ops.h": """
// instruction encoders: the scanner never expands this file (scan.INSTRUCTION_MACRO_FILES)
#define TTI_ADDDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) 0
#define TTI_CFGSHIFTMASK(disable_mask_on_old_val, operation, mask_width, right_cshift_amt, scratch_sel, CfgReg) 0
#define TTI_DMANOP 0
#define TTI_MOP(mop_type, loop_count, zmask_lo16) 0
#define TTI_NOP 0
#define TTI_PACR(CfgContext, RowPadZero, DstAccessMode, AddrMode, AddrCntContext, ZeroWrite, ReadIntfSel, OvrdThreadId, Concat, CtxtCtrl, Flush, Last) 0
#define TTI_REPLAY(start_idx, len, execute_while_loading, load_mode) 0
#define TTI_SEMWAIT(stall_res, sem_sel, wait_sem_cond) 0
#define TTI_SETADCXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask) 0
#define TTI_SETADCZW(CntSetMask, Ch1_W, Ch1_Z, Ch0_W, Ch0_Z, BitMask) 0
#define TTI_SETDMAREG(Payload_SigSelSize, Payload_SigSel, SetSignalsMode, RegIndex16b) 0
#define TTI_STALLWAIT(stall_res, wait_res) 0
#define TTI_WRCFG(GprAddress, wr128b, CfgReg) 0
#define TTI_ZEROACC(clear_mode, use_32_bit_mode, clear_zero_flags, addr_mode, where) 0
#define TT_INSN(x) 0
#define TT_OP_ADDRCRZW(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask) 0
#define TT_OP_INCADCZW(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X) 0
#define TT_OP_PACR(CfgContext, RowPadZero, DstAccessMode, AddrMode, AddrCntContext, ZeroWrite, ReadIntfSel, OvrdThreadId, Concat, CtxtCtrl, Flush, Last) 0
#define TT_OP_REPLAY(start_idx, len, execute_while_loading, load_mode) 0
#define TT_RMWCIB0(Mask, Data, CfgRegAddr) 0
#define TT_SETADC(CntSetMask, ChannelIndex, DimensionIndex, Value) 0
#define TT_SETDMAREG(Payload_SigSelSize, Payload_SigSel, SetSignalsMode, RegIndex16b) 0
#define TT_STOREREG(TdmaDataRegIndex, RegAddr) 0
#define TT_WRCFG(GprAddress, wr128b, CfgReg) 0
#define TT_ZEROACC(clear_mode, use_32_bit_mode, clear_zero_flags, addr_mode, where) 0
""",
    f"{LLK}/common/inc/ckernel_defs.h": """
namespace ckernel
{
constexpr std::uint32_t FACE_R_DIM            = 16;
constexpr std::uint32_t FACE_C_DIM            = 16;
constexpr std::uint32_t TILE_R_DIM            = 32;
constexpr std::uint32_t TILE_C_DIM            = 32;
constexpr std::uint32_t TILE_NUM_FACES        = 4;
constexpr std::uint32_t DATA_FORMAT_BIT_COUNT = 4;
enum class PackMode { Default, Untilize, Tilize };
enum class DstSync { SyncHalf, SyncFull };
enum class ReduceDim { REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR };
#define LO_16(REG) (2 * (REG))
#define HI_16(REG) (2 * (REG) + 1)
#define LOWER_HALFWORD(x) ((x) & 0xFFFF)
#define UPPER_HALFWORD(x) ((x) >> 16)
}
""",
    f"{LLK}/common/inc/ckernel_globals.h": """
extern std::uint32_t dest_offset_id;
extern std::uint32_t pack_sync_tile_dst_ptr;
""",
    f"{LLK}/common/inc/ckernel.h": """
namespace ckernel
{
template <std::uint32_t CfgAddr32, std::uint32_t Shamt, std::uint32_t Mask>
inline void cfg_reg_rmw_tensix(std::uint32_t val)
{
    TT_RMWCIB0(Mask & 0xff, val << Shamt, CfgAddr32);
}
inline void tensix_sync() { volatile std::uint32_t foo = 0; }
inline void mop_sync() { }
inline volatile std::uint32_t* get_cfg_pointer() { return 0; }
template <typename Callable>
inline void load_replay_buf(std::uint32_t start, std::uint32_t len, Callable&& callable)
{
    lltt::record(start, len);
    callable();
}
}
namespace lltt
{
inline void record(std::uint32_t start, std::uint32_t len) { TTI_REPLAY(start, len, 0, 1); }
inline std::uint32_t replay_insn(std::uint32_t start, std::uint32_t len) { return TT_OP_REPLAY(start, len, 0, 0); }
}
""",
    f"{LLK}/common/inc/ckernel_template.h": """
namespace ckernel
{
class ckernel_template
{
public:
    ckernel_template(std::uint32_t outer, std::uint32_t inner, std::uint32_t op) { }
    void set_last_inner_loop_instr(std::uint32_t op) { }
    void set_last_outer_loop_instr(std::uint32_t op) { }
    void set_start_op(std::uint32_t op) { }
    void set_end_op(std::uint32_t op) { }
    void program() { }
    static void run() { TTI_MOP(1, 0, 0); }
};
}
""",
}

REAL_SNIPPETS = {
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_common_api.h": """
// ---- llk_pack_common_api.h:215-218 (verbatim, main 8a7beeeaede)
TT_ALWAYS_INLINE void llk_pack_relu_config(const ckernel::ReluConfig& relu_config) {
    SAN_HOOK(unsupported());
    _llk_pack_relu_config_(relu_config);
}
// ---- llk_pack_common_api.h:225-225 (verbatim, main 8a7beeeaede)
inline void llk_pack_reconfig_l1_acc(const std::uint32_t enable) { _llk_pack_reconfig_l1_acc_(enable); }
""",
    "tt_metal/hw/ckernels/blackhole/metal/llk_io/llk_io_pack.h": """
// ---- llk_io_pack.h:16-16 (verbatim, main 8a7beeeaede)
using namespace ckernel;
// ---- llk_io_pack.h:48-97 (verbatim, main 8a7beeeaede)
inline void llk_push_to_brisc(const std::int32_t operand, const std::int32_t num_tiles, const std::int32_t num_words) {
    std::uint32_t output = operand;

    // Convert the counter's byte address to a Tensix 4-byte word address, masked to 18 bits.
    const std::uint32_t tiles_received_addr_tensix = static_cast<std::uint32_t>(
        (reinterpret_cast<std::uintptr_t>(get_cb_tiles_received_ptr(operand)) >> 2) & 0x3ffff);

    // get_local_cb_interface(output).tiles_received is used only by the TRISC2 (the one driving packer)
    // we need it because tiles_received_ptr is updated by the packer, and in cb_reserve_back func (see above) we want
    // to avoid synchronization with packer cb_reserve_back must used the most recent value of tiles_received (cannot
    // use stale or delayed), otherwise it would think there's less tiles in the CB than there actually are so we use
    // get_local_cb_interface(output).tiles_received instead of tiles_received_ptr, because it is updated by TRISC2 and
    // no additional synchronization is needed
    get_local_cb_interface(output).tiles_received += num_tiles;
    std::uint16_t tiles_received_new = get_local_cb_interface(output).tiles_received;

    // Update the value at tiles_received_ptr with tiles_received_new only after the packer has finished packing
    // We need to use a Tensix instruction to do the update, which runs only after STALLWAIT has finished
    // Note that the consumer side of the circular buffer (the one reading from the buffer) is ok to use stale/delayed
    // version of the value at tiles_received_ptr This is because consumer is polling the value at tiles_received_ptr,
    // and it will eventually see the updated value
    TT_SETDMAREG(0, tiles_received_new, 0, LO_16(p_gpr_pack::NUM_MSGS_RECEIVED));
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);  // wait for pack to finish
    TT_STOREREG(p_gpr_pack::NUM_MSGS_RECEIVED, tiles_received_addr_tensix);
}

// Push N tiles to stream buffer (increment write pointer)
template <bool push_blocks = false, bool brisc_pack = false>
inline void llk_push_tiles(const std::int32_t operand, const std::int32_t num_tiles) {
    SYNC_SIGNAL("SYNC-CB-PUSH", operand);
    std::uint32_t output = operand;

    auto& cb = get_local_cb_interface(output);
    std::uint32_t num_words = num_tiles * cb.fifo_page_size;

    LLK_ASSERT(cb.fifo_wr_ptr < cb.fifo_limit, "CB push_back: fifo_wr_ptr already at or past fifo_limit");

    std::uint32_t remaining = cb.fifo_limit - cb.fifo_wr_ptr;

    LLK_ASSERT(remaining >= num_words, "CB push_back: fifo_wr_ptr would exceed fifo_limit");

    cb.fifo_wr_ptr += num_words;
    cb.fifo_wr_tile_ptr = 0;

    if (cb.fifo_wr_ptr >= cb.fifo_limit) {
        cb.fifo_wr_ptr -= cb.fifo_size;
    }

    llk_push_to_brisc(operand, num_tiles, num_words);
}
""",
    "tt_metal/hw/inc/internal/tt-1xx/blackhole/cfg_defines.h": """
// Registers for ALU
#define ALU_FORMAT_SPEC_REG_Dstacc_val_ADDR32 0
#define ALU_FORMAT_SPEC_REG_Dstacc_val_SHAMT 10
#define ALU_FORMAT_SPEC_REG_Dstacc_val_MASK 0x3c00
#define ALU_FORMAT_SPEC_REG_Dstacc_override_ADDR32 0
#define ALU_FORMAT_SPEC_REG_Dstacc_override_SHAMT 14
#define ALU_FORMAT_SPEC_REG_Dstacc_override_MASK 0x4000
#define ALU_FORMAT_SPEC_REG2_Dstacc_ADDR32 1
#define ALU_FORMAT_SPEC_REG2_Dstacc_SHAMT 25
#define ALU_FORMAT_SPEC_REG2_Dstacc_MASK 0x1e000000
#define ALU_ACC_CTRL_Zero_Flag_disabled_src_ADDR32 2
#define ALU_ACC_CTRL_Zero_Flag_disabled_src_SHAMT 0
#define ALU_ACC_CTRL_Zero_Flag_disabled_src_MASK 0x1
#define ALU_ACC_CTRL_Zero_Flag_disabled_dst_ADDR32 2
#define ALU_ACC_CTRL_Zero_Flag_disabled_dst_SHAMT 1
#define ALU_ACC_CTRL_Zero_Flag_disabled_dst_MASK 0x2
#define STACC_RELU_ApplyRelu_ADDR32 2
#define STACC_RELU_ApplyRelu_SHAMT 2
#define STACC_RELU_ApplyRelu_MASK 0x3c
#define STACC_RELU_ReluThreshold_ADDR32 2
#define STACC_RELU_ReluThreshold_SHAMT 6
#define STACC_RELU_ReluThreshold_MASK 0x3fffc0
// Registers for PACK0
#define PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32 12
#define PCK0_ADDR_CTRL_XY_REG_0_Xstride_SHAMT 0
#define PCK0_ADDR_CTRL_XY_REG_0_Xstride_MASK 0xffff
#define PCK0_ADDR_CTRL_XY_REG_0_Ystride_ADDR32 12
#define PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT 16
#define PCK0_ADDR_CTRL_XY_REG_0_Ystride_MASK 0xffff0000
#define PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32 13
#define PCK0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT 0
#define PCK0_ADDR_CTRL_ZW_REG_0_Zstride_MASK 0xffff
#define PCK0_ADDR_CTRL_ZW_REG_0_Wstride_ADDR32 13
#define PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT 16
#define PCK0_ADDR_CTRL_ZW_REG_0_Wstride_MASK 0xffff0000
#define PCK0_ADDR_CTRL_XY_REG_1_Xstride_ADDR32 14
#define PCK0_ADDR_CTRL_XY_REG_1_Xstride_SHAMT 0
#define PCK0_ADDR_CTRL_XY_REG_1_Xstride_MASK 0xffff
#define PCK0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32 14
#define PCK0_ADDR_CTRL_XY_REG_1_Ystride_SHAMT 16
#define PCK0_ADDR_CTRL_XY_REG_1_Ystride_MASK 0xffff0000
#define PCK0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32 15
#define PCK0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT 0
#define PCK0_ADDR_CTRL_ZW_REG_1_Zstride_MASK 0xffff
#define PCK0_ADDR_CTRL_ZW_REG_1_Wstride_ADDR32 15
#define PCK0_ADDR_CTRL_ZW_REG_1_Wstride_SHAMT 16
#define PCK0_ADDR_CTRL_ZW_REG_1_Wstride_MASK 0xffff0000
#define PCK0_ADDR_BASE_REG_0_Base_ADDR32 16
#define PCK0_ADDR_BASE_REG_0_Base_SHAMT 0
#define PCK0_ADDR_BASE_REG_0_Base_MASK 0x3ffff
#define PCK_DEST_RD_CTRL_Read_32b_data_ADDR32 18
#define PCK_DEST_RD_CTRL_Read_32b_data_SHAMT 0
#define PCK_DEST_RD_CTRL_Read_32b_data_MASK 0x1
#define PCK_DEST_RD_CTRL_Read_unsigned_ADDR32 18
#define PCK_DEST_RD_CTRL_Read_unsigned_SHAMT 1
#define PCK_DEST_RD_CTRL_Read_unsigned_MASK 0x2
#define PCK_DEST_RD_CTRL_Read_int8_ADDR32 18
#define PCK_DEST_RD_CTRL_Read_int8_SHAMT 2
#define PCK_DEST_RD_CTRL_Read_int8_MASK 0x4
#define PCK_DEST_RD_CTRL_Round_10b_mant_ADDR32 18
#define PCK_DEST_RD_CTRL_Round_10b_mant_SHAMT 3
#define PCK_DEST_RD_CTRL_Round_10b_mant_MASK 0x8
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32 20
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_0_SHAMT 0
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_0_MASK 0x3
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_1_ADDR32 20
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_1_SHAMT 2
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_1_MASK 0xc
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_2_ADDR32 20
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_2_SHAMT 4
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_2_MASK 0x30
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_3_ADDR32 20
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_3_SHAMT 6
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_3_MASK 0xc0
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_4_ADDR32 20
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_4_SHAMT 8
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_4_MASK 0x300
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_5_ADDR32 20
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_5_SHAMT 10
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_5_MASK 0xc00
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_6_ADDR32 20
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_6_SHAMT 12
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_6_MASK 0x3000
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_7_ADDR32 20
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_7_SHAMT 14
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_7_MASK 0xc000
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_8_ADDR32 20
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_8_SHAMT 16
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_8_MASK 0x30000
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_9_ADDR32 20
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_9_SHAMT 18
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_9_MASK 0xc0000
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_10_ADDR32 20
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_10_SHAMT 20
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_10_MASK 0x300000
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_11_ADDR32 20
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_11_SHAMT 22
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_11_MASK 0xc00000
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_12_ADDR32 20
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_12_SHAMT 24
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_12_MASK 0x3000000
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_13_ADDR32 20
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_13_SHAMT 26
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_13_MASK 0xc000000
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_14_ADDR32 20
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_14_SHAMT 28
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_14_MASK 0x30000000
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_15_ADDR32 20
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_15_SHAMT 30
#define TILE_ROW_SET_MAPPING_0_row_set_mapping_15_MASK 0xc0000000
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_0_ADDR32 21
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_0_SHAMT 0
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_0_MASK 0x3
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_1_ADDR32 21
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_1_SHAMT 2
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_1_MASK 0xc
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_2_ADDR32 21
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_2_SHAMT 4
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_2_MASK 0x30
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_3_ADDR32 21
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_3_SHAMT 6
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_3_MASK 0xc0
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_4_ADDR32 21
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_4_SHAMT 8
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_4_MASK 0x300
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_5_ADDR32 21
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_5_SHAMT 10
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_5_MASK 0xc00
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_6_ADDR32 21
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_6_SHAMT 12
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_6_MASK 0x3000
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_7_ADDR32 21
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_7_SHAMT 14
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_7_MASK 0xc000
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_8_ADDR32 21
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_8_SHAMT 16
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_8_MASK 0x30000
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_9_ADDR32 21
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_9_SHAMT 18
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_9_MASK 0xc0000
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_10_ADDR32 21
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_10_SHAMT 20
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_10_MASK 0x300000
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_11_ADDR32 21
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_11_SHAMT 22
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_11_MASK 0xc00000
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_12_ADDR32 21
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_12_SHAMT 24
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_12_MASK 0x3000000
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_13_ADDR32 21
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_13_SHAMT 26
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_13_MASK 0xc000000
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_14_ADDR32 21
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_14_SHAMT 28
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_14_MASK 0x30000000
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_15_ADDR32 21
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_15_SHAMT 30
#define TILE_ROW_SET_MAPPING_1_row_set_mapping_15_MASK 0xc0000000
#define PCK_EDGE_OFFSET_SEC0_mask_ADDR32 24
#define PCK_EDGE_OFFSET_SEC0_mask_SHAMT 0
#define PCK_EDGE_OFFSET_SEC0_mask_MASK 0xffff
#define PCK_EDGE_MODE_mode_ADDR32 24
#define PCK_EDGE_MODE_mode_SHAMT 16
#define PCK_EDGE_MODE_mode_MASK 0x10000
#define PCK_EDGE_TILE_ROW_SET_SELECT_select_ADDR32 24
#define PCK_EDGE_TILE_ROW_SET_SELECT_select_SHAMT 17
#define PCK_EDGE_TILE_ROW_SET_SELECT_select_MASK 0x1fe0000
#define PCK_EDGE_OFFSET_SEC1_mask_ADDR32 25
#define PCK_EDGE_OFFSET_SEC1_mask_SHAMT 0
#define PCK_EDGE_OFFSET_SEC1_mask_MASK 0xffff
#define PCK_EDGE_OFFSET_SEC2_mask_ADDR32 26
#define PCK_EDGE_OFFSET_SEC2_mask_SHAMT 0
#define PCK_EDGE_OFFSET_SEC2_mask_MASK 0xffff
#define PCK_EDGE_OFFSET_SEC3_mask_ADDR32 27
#define PCK_EDGE_OFFSET_SEC3_mask_SHAMT 0
#define PCK_EDGE_OFFSET_SEC3_mask_MASK 0xffff
#define PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32 28
#define PACK_COUNTERS_SEC0_pack_per_xy_plane_SHAMT 0
#define PACK_COUNTERS_SEC0_pack_per_xy_plane_MASK 0xff
#define PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_ADDR32 28
#define PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_SHAMT 8
#define PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_MASK 0xff00
#define PACK_COUNTERS_SEC0_pack_xys_per_tile_ADDR32 28
#define PACK_COUNTERS_SEC0_pack_xys_per_tile_SHAMT 16
#define PACK_COUNTERS_SEC0_pack_xys_per_tile_MASK 0x7f0000
#define PACK_COUNTERS_SEC0_pack_yz_transposed_ADDR32 28
#define PACK_COUNTERS_SEC0_pack_yz_transposed_SHAMT 23
#define PACK_COUNTERS_SEC0_pack_yz_transposed_MASK 0x800000
#define PACK_COUNTERS_SEC0_auto_ctxt_inc_xys_cnt_ADDR32 28
#define PACK_COUNTERS_SEC0_auto_ctxt_inc_xys_cnt_SHAMT 24
#define PACK_COUNTERS_SEC0_auto_ctxt_inc_xys_cnt_MASK 0xff000000
// Registers for THCON
#define THCON_SEC0_REG1_Row_start_section_size_ADDR32 68
#define THCON_SEC0_REG1_Row_start_section_size_SHAMT 0
#define THCON_SEC0_REG1_Row_start_section_size_MASK 0xffff
#define THCON_SEC0_REG1_Exp_section_size_ADDR32 68
#define THCON_SEC0_REG1_Exp_section_size_SHAMT 16
#define THCON_SEC0_REG1_Exp_section_size_MASK 0xffff0000
#define THCON_SEC0_REG1_L1_Dest_addr_ADDR32 69
#define THCON_SEC0_REG1_L1_Dest_addr_SHAMT 0
#define THCON_SEC0_REG1_L1_Dest_addr_MASK 0xffffffff
#define THCON_SEC0_REG1_Disable_zero_compress_ADDR32 70
#define THCON_SEC0_REG1_Disable_zero_compress_SHAMT 0
#define THCON_SEC0_REG1_Disable_zero_compress_MASK 0x1
#define THCON_SEC0_REG1_Add_l1_dest_addr_offset_ADDR32 70
#define THCON_SEC0_REG1_Add_l1_dest_addr_offset_SHAMT 1
#define THCON_SEC0_REG1_Add_l1_dest_addr_offset_MASK 0x2
#define THCON_SEC0_REG1_Disable_pack_zero_flags_ADDR32 70
#define THCON_SEC0_REG1_Disable_pack_zero_flags_SHAMT 2
#define THCON_SEC0_REG1_Disable_pack_zero_flags_MASK 0x4
#define THCON_SEC0_REG1_ovrd_default_throttle_mode_ADDR32 70
#define THCON_SEC0_REG1_ovrd_default_throttle_mode_SHAMT 3
#define THCON_SEC0_REG1_ovrd_default_throttle_mode_MASK 0x8
#define THCON_SEC0_REG1_Out_data_format_ADDR32 70
#define THCON_SEC0_REG1_Out_data_format_SHAMT 4
#define THCON_SEC0_REG1_Out_data_format_MASK 0xf0
#define THCON_SEC0_REG1_In_data_format_ADDR32 70
#define THCON_SEC0_REG1_In_data_format_SHAMT 8
#define THCON_SEC0_REG1_In_data_format_MASK 0xf00
#define THCON_SEC0_REG1_Dis_shared_exp_assembler_ADDR32 70
#define THCON_SEC0_REG1_Dis_shared_exp_assembler_SHAMT 12
#define THCON_SEC0_REG1_Dis_shared_exp_assembler_MASK 0x1000
#define THCON_SEC0_REG1_Auto_set_last_pacr_intf_sel_ADDR32 70
#define THCON_SEC0_REG1_Auto_set_last_pacr_intf_sel_SHAMT 13
#define THCON_SEC0_REG1_Auto_set_last_pacr_intf_sel_MASK 0x2000
#define THCON_SEC0_REG1_Enable_out_fifo_ADDR32 70
#define THCON_SEC0_REG1_Enable_out_fifo_SHAMT 14
#define THCON_SEC0_REG1_Enable_out_fifo_MASK 0x4000
#define THCON_SEC0_REG1_Sub_l1_tile_header_size_ADDR32 70
#define THCON_SEC0_REG1_Sub_l1_tile_header_size_SHAMT 15
#define THCON_SEC0_REG1_Sub_l1_tile_header_size_MASK 0x8000
#define THCON_SEC0_REG1_Source_interface_selection_ADDR32 70
#define THCON_SEC0_REG1_Source_interface_selection_SHAMT 16
#define THCON_SEC0_REG1_Source_interface_selection_MASK 0x10000
#define THCON_SEC0_REG1_pack_start_intf_pos_ADDR32 70
#define THCON_SEC0_REG1_pack_start_intf_pos_SHAMT 17
#define THCON_SEC0_REG1_pack_start_intf_pos_MASK 0x1e0000
#define THCON_SEC0_REG1_All_pack_disable_zero_compress_ovrd_ADDR32 70
#define THCON_SEC0_REG1_All_pack_disable_zero_compress_ovrd_SHAMT 21
#define THCON_SEC0_REG1_All_pack_disable_zero_compress_ovrd_MASK 0x200000
#define THCON_SEC0_REG1_Add_tile_header_size_ADDR32 70
#define THCON_SEC0_REG1_Add_tile_header_size_SHAMT 22
#define THCON_SEC0_REG1_Add_tile_header_size_MASK 0x400000
#define THCON_SEC0_REG1_pack_dis_y_pos_start_offset_ADDR32 70
#define THCON_SEC0_REG1_pack_dis_y_pos_start_offset_SHAMT 23
#define THCON_SEC0_REG1_pack_dis_y_pos_start_offset_MASK 0x800000
#define THCON_SEC0_REG1_L1_source_addr_ADDR32 70
#define THCON_SEC0_REG1_L1_source_addr_SHAMT 24
#define THCON_SEC0_REG1_L1_source_addr_MASK 0xff000000
#define THCON_SEC0_REG1_Downsample_mask_ADDR32 71
#define THCON_SEC0_REG1_Downsample_mask_SHAMT 0
#define THCON_SEC0_REG1_Downsample_mask_MASK 0xffff
#define THCON_SEC0_REG1_Downsample_rate_ADDR32 71
#define THCON_SEC0_REG1_Downsample_rate_SHAMT 16
#define THCON_SEC0_REG1_Downsample_rate_MASK 0x70000
#define THCON_SEC0_REG1_Pack_L1_Acc_ADDR32 71
#define THCON_SEC0_REG1_Pack_L1_Acc_SHAMT 19
#define THCON_SEC0_REG1_Pack_L1_Acc_MASK 0x80000
#define THCON_SEC0_REG1_Exp_threshold_en_ADDR32 71
#define THCON_SEC0_REG1_Exp_threshold_en_SHAMT 20
#define THCON_SEC0_REG1_Exp_threshold_en_MASK 0x100000
#define THCON_SEC0_REG1_Unp_LF8_4b_exp_ADDR32 71
#define THCON_SEC0_REG1_Unp_LF8_4b_exp_SHAMT 22
#define THCON_SEC0_REG1_Unp_LF8_4b_exp_MASK 0x400000
#define THCON_SEC0_REG1_Pac_LF8_4b_exp_ADDR32 71
#define THCON_SEC0_REG1_Pac_LF8_4b_exp_SHAMT 23
#define THCON_SEC0_REG1_Pac_LF8_4b_exp_MASK 0x800000
#define THCON_SEC0_REG1_Exp_threshold_ADDR32 71
#define THCON_SEC0_REG1_Exp_threshold_SHAMT 24
#define THCON_SEC0_REG1_Exp_threshold_MASK 0xff000000
#define THCON_SEC0_REG8_Exp_section_size_ADDR32 96
#define THCON_SEC0_REG8_Exp_section_size_SHAMT 16
#define THCON_SEC0_REG8_Exp_section_size_MASK 0xffff0000
// Registers for GLOBAL
#define DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32 180
#define DEST_TARGET_REG_CFG_PACK_SEC0_Offset_SHAMT 0
#define DEST_TARGET_REG_CFG_PACK_SEC0_Offset_MASK 0xfff
#define DEST_TARGET_REG_CFG_PACK_SEC0_ZOffset_ADDR32 180
#define DEST_TARGET_REG_CFG_PACK_SEC0_ZOffset_SHAMT 12
#define DEST_TARGET_REG_CFG_PACK_SEC0_ZOffset_MASK 0x3f000
#define SCRATCH_SEC2_val_ADDR32 211
#define SCRATCH_SEC2_val_SHAMT 0
#define SCRATCH_SEC2_val_MASK 0xffffffff
""",
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_gpr_map.h": """
namespace ckernel
{
struct p_gpr
{
    constexpr static std::uint32_t ZERO         = 0; // Always stores 0
    constexpr static std::uint32_t DBG_RESERVED = 1; // Reserved for future use
    constexpr static std::uint32_t DBG_MSG      = 2; // Firmware debug message
    constexpr static std::uint32_t DBG_CKID     = 3; // Ckernel ID
};
struct p_gpr_pack
{
    constexpr static std::uint32_t DEST_OFFSET_LO = 4;  // dest lower bank offsets
    constexpr static std::uint32_t DEST_OFFSET_HI = 8;  // dest upper bank offsets
    constexpr static std::uint32_t OUTPUT_ADDR    = 12; // output address that packer is writing to
    constexpr static std::uint32_t TILE_HEADER    = 16; // tile header - ID + tile size

    constexpr static std::uint32_t TEMP_TILE_OFFSET    = 20; // Temp var which holds tile offset in dest
    constexpr static std::uint32_t NUM_MSGS_RECEIVED   = 24; // holds tile count and word size
    constexpr static std::uint32_t ONE_MSG_RECEIVED    = 25; // by default holds 1 tile count and word size for streaming per tile
    constexpr static std::uint32_t HEADER_ADDR         = 26; // Holds the address of the header (used by pack shift kernel only)
    constexpr static std::uint32_t TMP0                = 28; // Temp data
    constexpr static std::uint32_t TMP1                = 29; // Temp data
    constexpr static std::uint32_t TMP_LO              = 30; // Temp data, upper 16-bit always 0
    constexpr static std::uint32_t TMP_HI              = 31; // Temp data, lower 16-bit always 0
    constexpr static std::uint32_t PACK_STREAM_SYNC    = 32; // sync between pack and output stream [32:63]
    constexpr static std::uint32_t OUTPUT_ADDR_OFFSET  = 50; // output offset address that's added to OUTPUT_ADDR
    constexpr static std::uint32_t PERF_PACK_NUM_TILES = 51; // output operand num tiles
    constexpr static std::uint32_t EXP0_SEC_SIZE_BFP   = 52; // pack0,1,2,3 exp section size for bfp8,4,2
    constexpr static std::uint32_t EXP1_SEC_SIZE_BFP8  = 53; // pack1 exp section size for bfp8
    constexpr static std::uint32_t EXP2_SEC_SIZE_BFP8  = 54; // pack2 exp section size for bfp8
    constexpr static std::uint32_t EXP3_SEC_SIZE_BFP8  = 55; // pack2 exp section size for bfp8
    constexpr static std::uint32_t EXP1_SEC_SIZE_BFP4  = 57; // pack1 exp section size for bfp4
    constexpr static std::uint32_t EXP2_SEC_SIZE_BFP4  = 58; // pack2 exp section size for bfp4
    constexpr static std::uint32_t EXP3_SEC_SIZE_BFP4  = 59; // pack3 exp section size for bfp4
    constexpr static std::uint32_t EXP1_SEC_SIZE_BFP2  = 61; // pack1 exp section size for bfp2
    constexpr static std::uint32_t EXP2_SEC_SIZE_BFP2  = 62; // pack2 exp section size for bfp2
    constexpr static std::uint32_t EXP3_SEC_SIZE_BFP2  = 63; // pack3 exp section size for bfp2
};
}
""",
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h": """
// ---- cpack_common.h:18-164 (verbatim, main 8a7beeeaede)
namespace ckernel::packer
{
using DataFormatType = std::underlying_type_t<DataFormat>;

constexpr std::uint32_t replay_buf_offset = 16; // split replay buffer usage between fpu/sfpu
                                                // fist 16 for sfpu, next 16 for fpu
constexpr std::uint32_t NUM_PACKERS = 1;        // Number of packers

// Pack config
typedef struct
{
    // word 0
    std::uint32_t row_ptr_section_size : 16;
    std::uint32_t exp_section_size     : 16;
    // word 1
    std::uint32_t l1_dest_addr : 32;
    // word 2
    std::uint32_t uncompress              : 1;
    std::uint32_t add_l1_dest_addr_offset : 1;
    std::uint32_t disable_pack_zero_flag  : 1;
    std::uint32_t reserved_0              : 1;
    std::uint32_t out_data_format : DATA_FORMAT_BIT_COUNT;
    std::uint32_t in_data_format : DATA_FORMAT_BIT_COUNT;
    std::uint32_t dis_shared_exp_assembler            : 1;
    std::uint32_t auto_set_last_pacr_intf_sel         : 1;
    std::uint32_t enable_out_fifo                     : 1;
    std::uint32_t sub_l1_tile_header_size             : 1;
    std::uint32_t src_if_sel                          : 1;
    std::uint32_t pack_start_intf_pos                 : 4;
    std::uint32_t all_pack_disable_zero_compress_ovrd : 1;
    std::uint32_t add_tile_header_size                : 1;
    std::uint32_t pack_dis_y_pos_start_offset         : 1;
    std::uint32_t l1_src_addr                         : 8;
    //   The bit unp_lf8_4b_exp is configured in the unpack, remove word 3 to avoid potential race condition
    //   //word 3
    //   std::uint32_t downsample_mask : 16;
    //   std::uint32_t downsample_shift_count  : 3;
    //   std::uint32_t pack_l1_acc : 1; //Not new to BH, but moved
    //   //std::uint32_t read_mode : 1; //Removed in BH
    //   std::uint32_t exp_threshold_en  : 1;
    //   std::uint32_t reserved_2 : 1;
    //   std::uint32_t unp_lf8_4b_exp: 1;
    //   std::uint32_t pac_lf8_4b_exp: 1;
    //   std::uint32_t exp_threshold : 8;
} pack_config_t;

static_assert(sizeof(pack_config_t) == (sizeof(std::uint32_t) * 3));

typedef union
{
    std::uint32_t val[3];
    pack_config_t f;
} pack_config_u;

// Relu Config
typedef struct
{
    std::uint32_t ALU_ACC_CTRL_Zero_Flag_disabled_src      : 1;
    std::uint32_t ALU_ACC_CTRL_Zero_Flag_disabled_dst      : 1;
    std::uint32_t STACC_RELU_ApplyRelu                     : 4;
    std::uint32_t STACC_RELU_ReluThreshold                 : 16;
    std::uint32_t DISABLE_RISC_BP_Disable_main             : 1;
    std::uint32_t DISABLE_RISC_BP_Disable_trisc            : 3;
    std::uint32_t DISABLE_RISC_BP_Disable_ncrisc           : 1;
    std::uint32_t DISABLE_RISC_BP_Disable_bmp_clear_main   : 1;
    std::uint32_t DISABLE_RISC_BP_Disable_bmp_clear_trisc  : 3;
    std::uint32_t DISABLE_RISC_BP_Disable_bmp_clear_ncrisc : 1;
} relu_config_t;

static_assert(sizeof(relu_config_t) == (sizeof(std::uint32_t)));

typedef union
{
    std::uint32_t val[1];
    relu_config_t r;
} relu_config_u;

// Dest rd control
typedef struct
{
    std::uint32_t PCK_DEST_RD_CTRL_Read_32b_data  : 1;
    std::uint32_t PCK_DEST_RD_CTRL_Read_unsigned  : 1;
    std::uint32_t PCK_DEST_RD_CTRL_Read_int8      : 1;
    std::uint32_t PCK_DEST_RD_CTRL_Round_10b_mant : 1;
    std::uint32_t PCK_DEST_RD_CTRL_Reserved       : 28;
} dest_rd_ctrl_t;

static_assert(sizeof(dest_rd_ctrl_t) == (sizeof(std::uint32_t)));

typedef union
{
    std::uint32_t val;
    dest_rd_ctrl_t f;
} dest_rd_ctrl_u;

// PACK_EDGE_OFFSET_SEC[0:3] register structure
//
// Lower 16b represent a mask that is applied on a single row of one face on the packer output
// Higher 16b contain information about which TILE_ROW_SET_MAPPING register is used for each packer (only in PACK_EDGE_OFFSET_SEC0)
//
// There are 4 PACK_EDGE_OFFSET_SEC[0:3] registers and 4 TILE_ROW_SET_MAPPING[0:3] registers.
// TILE_ROW_SET_MAPPING[0:3] have 2 bits for each row inside a face that determine which PACK_EDGE_OFFSET_SEC[0:3] mask is used.
// Only PACK_EDGE_OFFSET_SEC0 register has higher 16b configured to determine TILE_ROW_SET_MAPPING[0:3] registers used for each packer.
// Other PACK_EDGE_OFFSET_SEC[1:3] registers are used only for the masks in the lower 16b.
typedef struct
{
    std::uint32_t mask                      : 16;
    std::uint32_t mode                      : 1;
    std::uint32_t tile_row_set_select_pack0 : 2;
    std::uint32_t tile_row_set_select_pack1 : 2;
    std::uint32_t tile_row_set_select_pack2 : 2;
    std::uint32_t tile_row_set_select_pack3 : 2;
    std::uint32_t reserved                  : 7;
} pck_edge_offset_t;

static_assert(sizeof(pck_edge_offset_t) == (sizeof(std::uint32_t)));

typedef union
{
    std::uint32_t val;
    pck_edge_offset_t f;
} pck_edge_offset_u;

// Pack counters
typedef struct
{
    std::uint32_t pack_per_xy_plane        : 8;
    std::uint32_t pack_reads_per_xy_plane  : 8;
    std::uint32_t pack_xys_per_til         : 7;
    std::uint32_t pack_yz_transposed       : 1;
    std::uint32_t pack_per_xy_plane_offset : 8;
} pack_counters_t;

static_assert(sizeof(pack_counters_t) == (sizeof(std::uint32_t)));

typedef union
{
    std::uint32_t val;
    pack_counters_t f;
} pack_counters_u;

// Set unpacker offsets to 0, except for unpacker 0, channel 1, X, which is the tile X dimension
inline void packer_addr_counter_init()
{
    TTI_SETADCXY(0b100, 0, 0, 0, 0, 0b1011);
    TTI_SETADCZW(0b100, 0, 0, 0, 0, 0b1111);
}
// ---- cpack_common.h:300-332 (verbatim, main 8a7beeeaede)
template <PackMode pack_mode = PackMode::Default>
inline void set_packer_strides(const std::uint32_t pack_src_format, const std::uint32_t tile_c_dim)
{
    std::uint32_t x_stride = datum_size_in_bytes(pack_src_format);
    std::uint32_t y_stride = FACE_C_DIM * x_stride;
    std::uint32_t w_stride = TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * x_stride;

    // Untilize mode has 2 packer interfaces active, so z counter needs to jump by 2
    // faces, since z counter is only 1 bit (can't be programmed to inc by 2)
    const std::uint32_t z_stride = ((pack_mode != PackMode::Default) && (tile_c_dim == TILE_C_DIM)) ? 2 * FACE_R_DIM * y_stride : FACE_R_DIM * y_stride;

    TT_SETDMAREG(0, LOWER_HALFWORD((y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT)), 0, LO_16(p_gpr_pack::TMP0)); // x-stride not used!
    TT_SETDMAREG(0, UPPER_HALFWORD((y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT)), 0, HI_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, LOWER_HALFWORD((z_stride << PCK0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT)), 0, LO_16(p_gpr_pack::TMP1));
    TT_SETDMAREG(0, UPPER_HALFWORD((w_stride << PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT)), 0, HI_16(p_gpr_pack::TMP1));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32);
    // Channel-1 (L1 output) strides are zero for every pack mode: output placement comes from
    // L1_Dest_addr and the packer's write pointer. Only the fast-untilize strided pack sets a
    // channel-1 (Y) stride, in its own init.
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_1_Xstride_ADDR32);
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
    TTI_NOP;
    TTI_NOP;
}

inline void reconfigure_packer_l1_acc(const std::uint32_t pack_l1_acc)
{
    // Stall to avoid clobbering current packer configuration, then enable/disable L1 accumulation
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    cfg_reg_rmw_tensix<THCON_SEC0_REG1_Pack_L1_Acc_RMW>(pack_l1_acc);
}
// ---- cpack_common.h:345-467 (verbatim, main 8a7beeeaede)
template <bool is_fp32_dest_acc_en>
inline void reconfigure_exp_threshold(const std::uint32_t pack_dst_format)
{
    bool enable             = false;
    std::uint32_t threshold = 0;

    // Workaround for HW bug: tenstorrent/budabackend#1394
    if constexpr (is_fp32_dest_acc_en)
    {
        if (IS_BFP_A_FORMAT(pack_dst_format))
        {
            // After early format conversion the exponent is EXP_B; packing to BFP-A uses EXP_A.
            // Zero out values too small to represent in EXP_A.
            //
            // For a given EXP_B number to be representable in EXP_A, it must satisfy:
            // EXP_BIAS_B = 127, EXP_BIAS_A = 15, and EXP_MIN_A = 1 - EXP_BIAS_A = -14
            //   EXP_B - EXP_BIAS_B >= EXP_MIN_A
            //   EXP_B - 127 >= -14
            //   EXP_B >= 113
            // The packer compares against threshold 113 and forces datums with EXP_B < 113 to zero.
            enable    = true;
            threshold = 113;
        }
    }

    static_assert(
        THCON_SEC0_REG1_Exp_threshold_en_ADDR32 == THCON_SEC0_REG1_Exp_threshold_ADDR32,
        "THCON_SEC0_REG1_Exp_threshold_en and Exp_threshold must share ADDR32 for combined RMW");

    constexpr std::uint32_t THRESHOLD_RMW_MASK = THCON_SEC0_REG1_Exp_threshold_en_MASK | THCON_SEC0_REG1_Exp_threshold_MASK;

    std::uint32_t threshold_rmw_data = (threshold << THCON_SEC0_REG1_Exp_threshold_SHAMT) | (enable << THCON_SEC0_REG1_Exp_threshold_en_SHAMT);

    cfg_reg_rmw_tensix<THCON_SEC0_REG1_Exp_threshold_ADDR32, 0, THRESHOLD_RMW_MASK>(threshold_rmw_data);
}

template <bool is_fp32_dest_acc_en>
inline void set_packer_config(
    const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format, const std::uint32_t num_faces = 4, const bool partial_face = false)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    // Get pointer to registers for current state ID
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    const std::uint32_t pack_output_src_format = masked_data_format(pack_src_format);
    const std::uint32_t pack_output_dst_format = masked_data_format(pack_dst_format);
    // Gasket converts Float16_b -> Float16 before the packer, so hardware in_data_format must be Float16 for Fp8 output.
    const std::uint32_t pack_hw_src_format =
        ((pack_dst_format & 0x1F) == to_underlying(DataFormat::Fp8_e4m3)) ? to_underlying(DataFormat::Float16) : pack_output_src_format;

    // Set packer config
    pack_config_u config;
    for (std::uint32_t i = 0; i < 3; i++)
    {
        config.val[i] = 0;
    }

    config.f.exp_section_size =
        ((pack_output_dst_format == to_underlying(DataFormat::Lf8)) || (pack_output_dst_format == to_underlying(DataFormat::Int8)))
            ? 0
            : (partial_face ? 1 : num_faces); // set to num_faces as exp section size is not used for non-bfp formats except for lf8/int8

    config.f.uncompress      = 1;
    config.f.out_data_format = pack_output_dst_format;
    config.f.in_data_format  = pack_hw_src_format;

    reconfigure_exp_threshold<is_fp32_dest_acc_en>(pack_output_dst_format);

    // Program:
    // THCON_SEC0_REG1_Row_start_section_size = cfg_reg_array[1][0 +: 16];
    // THCON_SEC0_REG1_Exp_section_size = cfg_reg_array[1][16 +: 16];
    // This is filled with garbage, and will be set up on every pack:
    //           THCON_SEC0_REG1_L1_Dest_addr = cfg_reg_array[1][32 +: 32];
    // THCON_SEC0_REG1_Disable_zero_compress = cfg_reg_array[1][64 +: 1];
    // THCON_SEC0_REG1_Add_l1_dest_addr_offset = cfg_reg_array[1][65 +: 1];
    // THCON_SEC0_REG1_Unused0 = cfg_reg_array[1][66 +: 2];
    // THCON_SEC0_REG1_Out_data_format = cfg_reg_array[1][68 +: 4];
    // THCON_SEC0_REG1_In_data_format = cfg_reg_array[1][72 +: 4];
    // THCON_SEC0_REG1_Unused00 = cfg_reg_array[1][76 +: 4];
    // THCON_SEC0_REG1_Source_interface_selection = cfg_reg_array[1][80 +: 1];
    // THCON_SEC0_REG1_Packs_per_xy_plane = cfg_reg_array[1][81 +: 7];
    // THCON_SEC0_REG1_L1_source_addr = cfg_reg_array[1][88 +: 8];
    // THCON_SEC0_REG1_Downsample_mask = cfg_reg_array[1][96 +: 16];
    // THCON_SEC0_REG1_Downsample_shift_count = cfg_reg_array[1][112 +: 3];
    // THCON_SEC0_REG1_Read_mode = cfg_reg_array[1][115 +: 1];
    // THCON_SEC0_REG1_Exp_threshold_en = cfg_reg_array[1][116 +: 1];
    // THCON_SEC0_REG1_Unused1 = cfg_reg_array[1][117 +: 3];
    // THCON_SEC0_REG1_Exp_threshold = cfg_reg_array[1][120 +: 8];
    // for (std::uint32_t i=0; i<4; i++) cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32+i]=config.val[i];
    cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 0] = config.val[0];
    cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 2] = config.val[2];
    // cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32+3]=config.val[3];

    // Reset L1 accumulation flag
    reconfigure_packer_l1_acc(0);

    dest_rd_ctrl_u dest_rd_ctrl;
    dest_rd_ctrl.val = 0;

    bool is_32b_format = pack_src_format == to_underlying(DataFormat::Int32) || pack_src_format == to_underlying(DataFormat::UInt32) ||
                         pack_src_format == to_underlying(DataFormat::Float32);
    bool is_int8_format = pack_src_format == to_underlying(DataFormat::Int8) || pack_src_format == to_underlying(DataFormat::UInt8);

    dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Read_32b_data = is_32b_format || is_fp32_dest_acc_en;
    dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Read_int8     = !(is_fp32_dest_acc_en || is_32b_format) && is_int8_format;

    if (pack_dst_format == to_underlying(DataFormat::UInt8))
    {
        dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Read_unsigned = 1;
    }

    // Round to 10 bit mantissa from fp32 dest
    if ((is_fp32_dest_acc_en && (pack_src_format == to_underlying(DataFormat::Float16))) ||
        (!IS_A_FORMAT(pack_src_format) && (pack_dst_format & 0x1F) == to_underlying(DataFormat::Fp8_e4m3)))
    {
        dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Round_10b_mant = 1;
    }
    cfg[PCK_DEST_RD_CTRL_Read_32b_data_ADDR32] = dest_rd_ctrl.val;

    // Save to GPR for quick data format reconfig
    regfile[p_gpr_pack::EXP0_SEC_SIZE_BFP] = (partial_face ? 1 : num_faces) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
    sync_regfile_write(p_gpr_pack::EXP0_SEC_SIZE_BFP);
}
// ---- cpack_common.h:569-710 (verbatim, main 8a7beeeaede)
template <bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default>
inline void configure_pack(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    [[maybe_unused]] const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t tile_c_dim                  = TILE_C_DIM,
    const std::uint32_t num_faces                   = 4,
    const bool partial_face                         = false,
    const std::uint32_t relu_config                 = 0)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(
        is_packer_to_L1_conversion_supported(static_cast<DataFormat>(pack_src_format & 0xF), static_cast<DataFormat>(pack_dst_format & 0xF)),
        "Unsupported packer to L1 conversion.");
    // Get pointer to registers for current state ID
    volatile std::uint32_t* cfg = get_cfg_pointer();

    // Drain the packer before any of the configuration below is rewritten, whatever the formats. The raw
    // cfg[] stores in this function and in set_packer_config are RISC MMIO, which no STALLWAIT can hold
    // back: the STALLWAIT only keeps new packer instructions out, and tensix_sync() blocks this RISC until
    // the packer has gone idle, which is what orders those stores behind a pack still in flight. Without
    // it the pack samples the new format/width mid-tile.
    TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::PACK);
    tensix_sync();

    const std::uint32_t pack_output_src_format = masked_data_format(pack_src_format);

    set_packer_strides<pack_mode>(pack_src_format, tile_c_dim);

    // NOTE: the packer X (datum) counter (SETADCXX PAC) is intentionally NOT programmed here. It is
    // owned by _llk_pack_init_, which runs after hw-configure and sets its own value per the
    // "inits own SETADCXX" contract.

    t6_mutex_acquire(mutex::REG_RMW);

    // Set Fp8 E4M3 mode for packer
    bool is_fp8_e4m3 = (pack_dst_format & 0x1F) == (std::uint32_t)DataFormat::Fp8_e4m3;
    cfg_reg_rmw_tensix<THCON_SEC0_REG1_Pac_LF8_4b_exp_RMW>(is_fp8_e4m3);

    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG2_Dstacc_RMW>(is_fp8_e4m3 ? to_underlying(DataFormat::Float16) : pack_output_src_format);

    // Establish the no-override baseline for the Dstacc ALU format-select fields (ADDR32=0), consumed
    // by the packer: clear Dstacc_val/override so the base REG2_Dstacc format programmed above is
    // used. The SrcA/SrcB override fields in the same word are owned by the math thread
    // (_llk_math_hw_configure_); the two writers touch disjoint bits and rely on per-byte RMWCIB
    // atomicity, so this write does not depend on the surrounding REG_RMW mutex for cross-thread safety.
    constexpr std::uint32_t dstacc_fmt_override_mask = ALU_FORMAT_SPEC_REG_Dstacc_val_MASK | ALU_FORMAT_SPEC_REG_Dstacc_override_MASK;
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_Dstacc_val_ADDR32, 0, dstacc_fmt_override_mask>(0);

    // Config RELU
    relu_config_u hw_relu_config;
    hw_relu_config.r.STACC_RELU_ApplyRelu     = relu_config & 0xffff;
    hw_relu_config.r.STACC_RELU_ReluThreshold = (relu_config >> 16) & 0xffff;

    static_assert(STACC_RELU_ApplyRelu_ADDR32 == STACC_RELU_ReluThreshold_ADDR32, "STACC_RELU_ApplyRelu and ReluThreshold must share ADDR32 for combined RMW");

    constexpr std::uint32_t hw_relu_mask = STACC_RELU_ApplyRelu_MASK | STACC_RELU_ReluThreshold_MASK;
    cfg_reg_rmw_tensix<STACC_RELU_ApplyRelu_ADDR32, 0, hw_relu_mask>(hw_relu_config.val[0]);

    t6_mutex_release(mutex::REG_RMW);

    set_packer_config<is_fp32_dest_acc_en>(pack_src_format, pack_dst_format, num_faces, partial_face);

    // PACK_COUNTERS_SEC0_pack_per_xy_plane = cfg_reg_array[3][0 +: 8];
    // PACK_COUNTERS_SEC0_pack_reads_per_xy_plane = cfg_reg_array[3][8 +: 8];
    // PACK_COUNTERS_SEC0_pack_xys_per_tile = cfg_reg_array[3][16 +: 7];
    // PACK_COUNTERS_SEC0_pack_yz_transposed = cfg_reg_array[3][23 +: 1];
    pack_counters_u pack_counters;
    pack_counters.val = 0;
    // Number of reads per face used for resetting tile position generator for edge masks.
    pack_counters.f.pack_reads_per_xy_plane = 1;
    for (std::uint32_t i = 0; i < NUM_PACKERS; i++)
    {
        cfg[PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32 + i] = pack_counters.val; // disable auto last generation
    }

    pck_edge_offset_u pck_edge_offset;
    pck_edge_offset.val    = 0;
    pck_edge_offset.f.mask = 0xffff;

    cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32]                = pck_edge_offset.val;
    cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32] = 0x0; // All packers use row set mapping 0, edge offset 0 mask

    regfile[p_gpr_pack::TILE_HEADER]     = tile_size;
    regfile[p_gpr_pack::TILE_HEADER + 1] = 0;
    regfile[p_gpr_pack::TILE_HEADER + 2] = 0;
    regfile[p_gpr_pack::TILE_HEADER + 3] = 0;
    sync_regfile_write(p_gpr_pack::TILE_HEADER + 3);
}

inline std::uint8_t get_packer_dest_offset_index()
{
    return (dest_offset_id ? p_gpr_pack::DEST_OFFSET_HI : p_gpr_pack::DEST_OFFSET_LO);
}

inline std::uint32_t get_packer_dest_offset()
{
    return (dest_offset_id ? DEST_REGISTER_HALF_SIZE : 0x0);
}

inline void flip_packer_dest_offset_id()
{
    dest_offset_id = 1 - dest_offset_id;
}

// Flip packer dest register offset to 0 or DEST_REGISTER_HALF_SIZE
// flip-flopping between two halves
template <DstSync Dst>
inline void select_packer_dest_registers()
{
    if constexpr (Dst == DstSync::SyncFull)
    {
        TTI_WRCFG(p_gpr_pack::DEST_OFFSET_LO, p_cfg::WRCFG_32b, DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32);
    }
    else
    {
        TT_WRCFG(get_packer_dest_offset_index(), p_cfg::WRCFG_32b, DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32);
    }
    TTI_DMANOP;
    TTI_DMANOP;
}

// Program packer destination addresses from GPRs
inline void program_packer_destination(std::uint32_t addr)
{
    LLK_ASSERT(is_valid_L1_address(addr), "L1 address must be in valid L1 memory region");
    // No packer-drain STALLWAIT is needed before reprogramming the destination: the pack thread issues its
    // instruction stream in order, so each tile's PACR has already started -- and latched L1_Dest_addr,
    // sampled at PACR start -- before the next call's WRCFG reprograms it, and the Last=1 PACR that ends each
    // pack MOP drains the packer and forces the next pack to re-sample L1_Dest_addr. The STALLWAIT below is
    // only the GPR-producer fence: it ensures the SETDMAREG write to OUTPUT_ADDR retires before WRCFG reads it.
    std::uint32_t new_l1_addr = (1 << 31) | addr;
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);

    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
}
// ---- cpack_common.h:740-773 (verbatim, main 8a7beeeaede)
inline void program_packer_dest_offset_registers(std::uint32_t dest_tile_offset)
{
    TT_SETDMAREG(0, LOWER_HALFWORD(dest_tile_offset), 0, LO_16(p_gpr_pack::TEMP_TILE_OFFSET));
    TT_SETDMAREG(0, UPPER_HALFWORD(dest_tile_offset), 0, HI_16(p_gpr_pack::TEMP_TILE_OFFSET));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON | p_stall::PACK);
    TTI_WRCFG(p_gpr_pack::TEMP_TILE_OFFSET, p_cfg::WRCFG_32b, PCK0_ADDR_BASE_REG_0_Base_ADDR32);
    TTI_DMANOP;
    TTI_DMANOP;
}

// READERS FOR CONFIG STRUCTS

inline pack_config_t read_pack_config_helper(std::uint32_t reg_addr, const volatile std::uint32_t tt_reg_ptr* cfg)
{
    pack_config_u config = {.val = 0};

    config.val[0] = cfg[reg_addr];
    config.val[1] = cfg[reg_addr + 1];
    config.val[2] = cfg[reg_addr + 2];

    return config.f;
}

__attribute__((noinline)) std::array<pack_config_t, NUM_PACKERS> read_pack_config()
{
    std::array<pack_config_t, NUM_PACKERS> config_vec;

    // Get pointer to registers for current state ID
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    config_vec[0] = read_pack_config_helper(THCON_SEC0_REG1_Row_start_section_size_ADDR32, cfg);

    return config_vec;
}
// ---- cpack_common.h:925-925 (verbatim, main 8a7beeeaede)
} // namespace ckernel::packer
""",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_pack_fast_tilize.h": """
// ---- llk_pack_fast_tilize.h:21-25 (verbatim, main 8a7beeeaede)
constexpr std::uint32_t REPLAY_TILE_OFFSET = 0;
constexpr std::uint32_t REPLAY_TILE_LEN    = 16;

constexpr std::uint32_t REPLAY_ADDR_UPDATE_OFFSET = ckernel::packer::replay_buf_offset;
constexpr std::uint32_t REPLAY_ADDR_UPDATE_LEN    = 4;
// ---- llk_pack_fast_tilize.h:35-107 (verbatim, main 8a7beeeaede)
#define EMIT_FACE_PACRS(boundary_am, last) \\
    TTI_PACR(                              \\
        p_pacr::CFG_CTXT_0,                \\
        p_pacr::NO_ROW_PAD_ZERO,           \\
        p_pacr::DST_ACCESS_STRIDED_MODE,   \\
        ADDR_MOD_0,                        \\
        p_pacr::ADDR_CNT_CTXT_0,           \\
        0,                                 \\
        p_pacr::ALL_INTF_ACTIVE,           \\
        0,                                 \\
        0,                                 \\
        p_pacr::NO_CTXT_CTRL,              \\
        0,                                 \\
        0);                                \\
    TTI_PACR(                              \\
        p_pacr::CFG_CTXT_0,                \\
        p_pacr::NO_ROW_PAD_ZERO,           \\
        p_pacr::DST_ACCESS_STRIDED_MODE,   \\
        ADDR_MOD_0,                        \\
        p_pacr::ADDR_CNT_CTXT_0,           \\
        0,                                 \\
        p_pacr::ALL_INTF_ACTIVE,           \\
        0,                                 \\
        0,                                 \\
        p_pacr::NO_CTXT_CTRL,              \\
        0,                                 \\
        0);                                \\
    TTI_PACR(                              \\
        p_pacr::CFG_CTXT_0,                \\
        p_pacr::NO_ROW_PAD_ZERO,           \\
        p_pacr::DST_ACCESS_STRIDED_MODE,   \\
        ADDR_MOD_0,                        \\
        p_pacr::ADDR_CNT_CTXT_0,           \\
        0,                                 \\
        p_pacr::ALL_INTF_ACTIVE,           \\
        0,                                 \\
        0,                                 \\
        p_pacr::NO_CTXT_CTRL,              \\
        0,                                 \\
        0);                                \\
    TTI_PACR(                              \\
        p_pacr::CFG_CTXT_0,                \\
        p_pacr::NO_ROW_PAD_ZERO,           \\
        p_pacr::DST_ACCESS_STRIDED_MODE,   \\
        boundary_am,                       \\
        p_pacr::ADDR_CNT_CTXT_0,           \\
        0,                                 \\
        p_pacr::ALL_INTF_ACTIVE,           \\
        0,                                 \\
        0,                                 \\
        p_pacr::NO_CTXT_CTRL,              \\
        0,                                 \\
        last)

__attribute__((noinline)) void _llk_pack_fast_tilize_load_replay_()
{
    // Tile replay [0..15]: 16 PACRs, Last=1 on final.
    TTI_REPLAY(REPLAY_TILE_OFFSET, REPLAY_TILE_LEN, 0, 1);
    EMIT_FACE_PACRS(ADDR_MOD_3, 0);
    EMIT_FACE_PACRS(ADDR_MOD_2, 0);
    EMIT_FACE_PACRS(ADDR_MOD_3, 0);
    EMIT_FACE_PACRS(ADDR_MOD_1, 1); // face3: Last=1

    // Address-update replay [16..19]: advance L1_Dest_addr for next tile.
    // Same pattern as BH pack-untilize (llk_pack_untilize.h:94-108).
    TTI_REPLAY(REPLAY_ADDR_UPDATE_OFFSET, REPLAY_ADDR_UPDATE_LEN, 0, 1);
    TTI_ADDDMAREG(0, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
    TTI_NOP;
}

#undef EMIT_FACE_PACRS
""",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack.h": """
// ---- llk_pack.h:17-18 (verbatim, main 8a7beeeaede)
using namespace ckernel;
using namespace ckernel::packer;
// ---- llk_pack.h:589-609 (verbatim, main 8a7beeeaede)
template <DstSync Dst, bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default, bool mutex_ADC = false>
inline void _llk_pack_(const std::uint32_t tile_index, const std::uint32_t address)
{
    static_assert(
        pack_mode == PackMode::Default || pack_mode == PackMode::Untilize, "Blackhole: _llk_pack_ supports PackMode::Default and PackMode::Untilize only");
    static_assert(
        !mutex_ADC || pack_mode == PackMode::Default,
        "mutex_ADC is supported for PackMode::Default only: other modes issue ADC instructions from inside the MOP, "
        "which the mutex cannot cover");

    set_dst_write_addr<mutex_ADC>(tile_index);

    program_packer_destination(address);

    ckernel::ckernel_template::run();

    {
        T6MutexLockGuard<mutex_ADC> guard(mutex::THREAD2_ADC);
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101); // reset z counters
    }
}
""",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack_common.h": """
// ---- llk_pack_common.h:16-47 (verbatim, main 8a7beeeaede)
using namespace ckernel;
using namespace ckernel::packer;

/**
 * @brief Stall the packer until the math thread has produced data to pack.
 *
 * Waits on the MATH_PACK semaphore so the packer does not run ahead of the math result.
 */
// wait until math is done and has produced something to pack
inline void _llk_packer_wait_for_math_done_()
{
    // The mask must cover STALL_SYNC as well as STALL_TDMA: the Wait Gate only blocks classes named in it, so with a
    // TDMA-only mask the Sync-class ATGETM of a mutexed _llk_pack_ slips past this unmet wait and takes
    // mutex::THREAD2_ADC, the SETADC behind it blocks head-of-line, and the ATRELM is stranded behind that. The packer
    // then waits on MATH_PACK holding the mutex the unpack thread needs every iteration: deadlock. Naming the Sync
    // class costs nothing when unmutexed.
    TTI_SEMWAIT(p_stall::STALL_TDMA | p_stall::STALL_SYNC, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO);
}

/**
 * @brief Signal the MATH_PACK semaphore (decrement via SEMGET) to release the math thread.
 *
 * Tells math it may overwrite the destination register again now that the packer has consumed it.
 *
 * @tparam WaitRes: p_stall resource mask to stall on before signalling (e.g. p_stall::NONE, p_stall::PACK); default p_stall::NONE issues no stall.
 */
// Tell math that it can write again
template <std::uint32_t WaitRes = p_stall::NONE>
inline void _llk_packer_set_math_semaphore_()
{
    t6_semaphore_get<WaitRes>(semaphore::MATH_PACK); // Indicate that packer is done and header is written into L1
}
// ---- llk_pack_common.h:61-95 (verbatim, main 8a7beeeaede)
// Wait for all writes to complete in L1 (header + data)
// Tell math it can write again
// Clear dest only when clear_dest is true
template <DstSync Dst, bool is_fp32_dest_acc_en, bool clear_dest = true>
inline void _llk_pack_dest_section_done_()
{
    if constexpr (clear_dest)
    {
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK); // wait for pack to finish

        if constexpr (Dst == DstSync::SyncFull)
        {
            TTI_ZEROACC(p_zeroacc::CLR_ALL, is_fp32_dest_acc_en, 0, ADDR_MOD_1, 0);
        }
        else
        {
            static_assert(Dst == DstSync::SyncHalf);
            TT_ZEROACC(p_zeroacc::CLR_HALF, is_fp32_dest_acc_en, 0, ADDR_MOD_1, dest_offset_id % 2);
        }

        // Tell math that it can write again
        _llk_packer_set_math_semaphore_<p_stall::NONE>();
    }
    else
    {
        // Without ZEROACC, block SEMGET itself until PACK has finished reading DST.
        _llk_packer_set_math_semaphore_<p_stall::PACK>();
    }

    if constexpr (Dst == DstSync::SyncHalf)
    {
        flip_packer_dest_offset_id();
        select_packer_dest_registers<Dst>();
    }
}
// ---- llk_pack_common.h:152-157 (verbatim, main 8a7beeeaede)
template <bool mutex_ADC = false>
inline void set_dst_write_addr(const std::uint32_t tile_index)
{
    T6MutexLockGuard<mutex_ADC> guard(mutex::THREAD2_ADC);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, tile_index);
}
// ---- llk_pack_common.h:166-193 (verbatim, main 8a7beeeaede)
TT_ALWAYS_INLINE void _llk_pack_relu_config_(const ckernel::ReluConfig& relu_config)
{
    const std::uint32_t mode = static_cast<std::uint32_t>(relu_config.get_mode());
    const std::uint32_t val  = (relu_config.get_threshold() << STACC_RELU_ReluThreshold_SHAMT) | (mode << STACC_RELU_ApplyRelu_SHAMT);

    // STACC_RELU shares this cfg word with ALU_ACC_CTRL_Zero_Flag_disabled_src/dst (bits 0-1, owned by the
    // MATH/UNPACK threads). A whole-word WRCFG_32b would clobber those bits, so use a masked RMW under
    // mutex::REG_RMW -- matching how configure_pack programs relu.
    static_assert(STACC_RELU_ApplyRelu_ADDR32 == STACC_RELU_ReluThreshold_ADDR32, "STACC_RELU ApplyRelu and ReluThreshold must share ADDR32 for combined RMW");
    constexpr std::uint32_t hw_relu_mask = STACC_RELU_ApplyRelu_MASK | STACC_RELU_ReluThreshold_MASK;

    // Only the packer needs draining: RMWCIB takes the data as an immediate (no GPR), so there is no
    // SETDMAREG->WRCFG producer to fence -- the THCON wait the old whole-word path used is no longer needed.
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    t6_mutex_acquire(mutex::REG_RMW);
    cfg_reg_rmw_tensix<STACC_RELU_ApplyRelu_ADDR32, 0, hw_relu_mask>(val);
    t6_mutex_release(mutex::REG_RMW);
}

/**
 * @brief Enable or disable packer L1 accumulation.
 *
 * @param enable: Non-zero to accumulate packed output into existing L1 data, zero to overwrite.
 */
inline void _llk_pack_reconfig_l1_acc_(const std::uint32_t enable)
{
    reconfigure_packer_l1_acc(enable);
}
// ---- llk_pack_common.h:207-325 (verbatim, main 8a7beeeaede)
template <ReduceDim dim, PackMode pack_mode = PackMode::Default>
inline void _llk_pack_reduce_mask_config_(const std::uint32_t face_r_dim = FACE_R_DIM)
{
    ckernel::packer::pck_edge_offset_u pack_edge_offset = {.val = 0};

    // We initialize PCK_EDGE_OFFSET_SEC0 mask to clear out all the datums in the row
    pack_edge_offset.f.mask             = 0x0;
    std::uint32_t row_set_mapping_1     = 0;
    std::uint32_t edge_offset_sec1_mask = 0;

    if constexpr (dim == ReduceDim::REDUCE_ROW)
    {
        // All packers use TILE_ROW_SET_MAPPING_1 to support both narrow tiles (packers 0,1)
        // and wide tiles (packers 0,2)
        pack_edge_offset.f.tile_row_set_select_pack0 = 1;
        pack_edge_offset.f.tile_row_set_select_pack1 = 1;
        pack_edge_offset.f.tile_row_set_select_pack2 = 1;
        pack_edge_offset.f.tile_row_set_select_pack3 = 1;

        // PCK_EDGE_OFFSET_SEC1 mask will clear out all the datums in the row except the first one
        edge_offset_sec1_mask = 0x0001;
        if constexpr (pack_mode == PackMode::Untilize)
        {
            row_set_mapping_1 = 0x11111111; // each packer packs 1x32 row
        }
        else
        {
            // TILE_ROW_SET_MAPPING_1 configuration sets all rows to use PCK_EDGE_OFFSET_SEC1 mask
            row_set_mapping_1 = 0x55555555; // each packer packs 1x16 row
        }
    }
    else if constexpr (dim == ReduceDim::REDUCE_COL)
    {
        // PCK_EDGE_OFFSET_SEC1 mask will pass through all the datums in the row as they are
        edge_offset_sec1_mask = 0xffff;

        // Packer 0 and 1 will use TILE_ROW_SET_MAPPING_1, while packer 2 and 3 will keep using
        // TILE_ROW_SET_MAPPING_0 configuration which is the default one
        pack_edge_offset.f.tile_row_set_select_pack0 = 1;
        pack_edge_offset.f.tile_row_set_select_pack1 = 1;

        if constexpr (pack_mode == PackMode::Untilize)
        {
            row_set_mapping_1 = 0x00000005; // each packer packs 1x32 row
        }
        else
        {
            // TILE_ROW_SET_MAPPING_1 configuration sets only first row to use PCK_EDGE_OFFSET_SEC1 mask
            row_set_mapping_1 = 0x00000001; // each packer packs 1x16 row
        }
    }
    else if constexpr (dim == ReduceDim::REDUCE_SCALAR)
    {
        // PCK_EDGE_OFFSET_SEC1 mask will clear out all the datums in the row except the first one
        edge_offset_sec1_mask = 0x0001;
        // Packer 0  will use TILE_ROW_SET_MAPPING_1, while packers 1,2 and 3 will keep using
        // TILE_ROW_SET_MAPPING_0 configuration which is the default one
        pack_edge_offset.f.tile_row_set_select_pack0 = 1;

        // TILE_ROW_SET_MAPPING_1 configuration sets only first row to use PCK_EDGE_OFFSET_SEC1 mask
        row_set_mapping_1 = 0x00000001;
    }

    // Initialize TMP registers with values we need to write in CFG registers
    TTI_SETDMAREG(0, LOWER_HALFWORD(pack_edge_offset.val), 0, LO_16(p_gpr_pack::TMP0));
    TTI_SETDMAREG(0, UPPER_HALFWORD(pack_edge_offset.val), 0, HI_16(p_gpr_pack::TMP0));
    TTI_SETDMAREG(0, LOWER_HALFWORD(edge_offset_sec1_mask), 0, LO_16(p_gpr_pack::TMP_LO));
    TTI_SETDMAREG(0, LOWER_HALFWORD(row_set_mapping_1), 0, LO_16(p_gpr_pack::TMP1));
    TTI_SETDMAREG(0, UPPER_HALFWORD(row_set_mapping_1), 0, HI_16(p_gpr_pack::TMP1));

    // Wait for packer to finish to avoid breaking its current configuration
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);

    cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(face_r_dim);

    // Configure packer
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP_LO, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC1_mask_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, TILE_ROW_SET_MAPPING_1_row_set_mapping_0_ADDR32);

    TTI_NOP;
    TTI_NOP;
}

/**
 * @brief Restore the default packer edge masks and tile-row-set mapping after a reduce.
 *
 * Resets the edge-offset masks to pass-through and points all tile-row-set mappings back to
 * PCK_EDGE_OFFSET_SEC0, undoing @ref _llk_pack_reduce_mask_config_.
 *
 * @note Pairs with @ref _llk_pack_reduce_mask_config_.
 */
inline void _llk_pack_reduce_mask_clear_()
{
    // By default, all packers are set to use TILE_ROW_SET_MAPPING_0 and
    // mask is configured to pass through all the datums
    pck_edge_offset_u pack_edge_offset = {.val = 0};
    pack_edge_offset.f.mask            = 0xffff;

    // Initialize TMP registers with values we need to write in CFG registers
    TTI_SETDMAREG(0, LOWER_HALFWORD(pack_edge_offset.val), 0, LO_16(p_gpr_pack::TMP0));
    TTI_SETDMAREG(0, UPPER_HALFWORD(pack_edge_offset.val), 0, HI_16(p_gpr_pack::TMP0));

    // Wait for packer to finish to avoid breaking its current configuration
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);

    cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(1);

    // Clear out packer configuration for reduce
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC1_mask_ADDR32);

    // All mappings point to PCK_EDGE_OFFSET_SEC0_mask_ADDR32
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32);
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, TILE_ROW_SET_MAPPING_1_row_set_mapping_0_ADDR32);

    TTI_NOP;
    TTI_NOP;
}
""",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack_untilize.h": """
// ---- llk_pack_untilize.h:17-18 (verbatim, main 8a7beeeaede)
using namespace ckernel;
using namespace ckernel::packer;
// ---- llk_pack_untilize.h:64-167 (verbatim, main 8a7beeeaede)
template <std::uint32_t block_ct_dim, bool narrow_row = false, bool dense = false>
inline void _llk_pack_untilize_mop_config_(const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4)
{
    static_assert(!dense || (block_ct_dim % 2 == 0), "block_ct_dim must be even when dense");
    static_assert(!dense || (!narrow_row), "narrow_row must be false when dense");
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(!dense || (num_faces == 2), "num_faces must be 2 when dense");
    /*
    Outer loop iterates over the rows in the block, while the inner loop iterates
    over each tile in the block.
    When dense, we use all 4 interfaces to pack out a row each from 4 faces (2 tiles) that end up contiguous in L1
    because offsets align well and it improves perf, thus we halve the number of mop inner loops.
    */
    constexpr std::uint32_t MOP_INNER_LOOP = dense ? block_ct_dim / 2 : block_ct_dim;
    const std::uint32_t MOP_OUTER_LOOP     = face_r_dim;

    // For narrow row, the faces are stored in the first column of the tile, therefore requiring only one packer interface.
    const std::uint32_t PACK_INTF_SEL = (dense)                          ? p_pacr::ALL_INTF_ACTIVE
                                        : (narrow_row || num_faces == 1) ? p_pacr::SINGLE_INTF_ACTIVE
                                                                         : p_pacr::TWO_INTFS_ACTIVE;
    /*
    When using DST_STRIDED_MODE, each packer interface has a stride of 16*block_size,
    where block_size is set to be the size of a row within face.
    Each PACR instruction packs 2x16 datums if (num_faces>1), meaning that it would
    pack out one row for each tile in the block.
    In the inner loop, for each tile, the rows that get packed from dest register
    in the first outer loop iteration are:
    tile 0: row 0, row 16
    tile 1: row 64, row 80
    tile block_ct_dim-1: row 64*(block_ct_dim-1), row 64*(block_ct_dim-1)+16
    This processes is repeated for each row of the block in dest.
    */
    ckernel::ckernel_template tmp(
        MOP_OUTER_LOOP,
        MOP_INNER_LOOP,
        TT_OP_INCADCZW(p_setadc::PAC, 0, 0, 1, 0), // w cnt points to the next tile
        TT_OP_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_STRIDED_MODE,
            ADDR_MOD_0,
            p_pacr::ADDR_CNT_CTXT_0,
            0,
            PACK_INTF_SEL,
            0,
            0,
            p_pacr::NO_CTXT_CTRL,
            0,
            0));

    /*
    Since there are two inner loop operations, the instruction set by set_last_inner_loop_instr
    will replace the second inner loop operation (in the last iteration, call the PACR instruction
    with the Last bit set to 1 instead of 0 to close the row).
    The W counter CR shadow (W_Cr) is established by TT_SETADC(...SET_W...) in _llk_pack_untilize_
    before run() is called; the SETADCZW there only initializes Z.
    ADDRCRZW with increment 0 resets W to the stored W_Cr value at the start of each outer loop
    iteration (row), without needing tile_dst_offset baked into this MOP template.
    */
    tmp.set_start_op(TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 0, 0, 0b0010 /*CH0_W*/)); // W = W_Cr (restore W to start of block)

    const std::uint32_t replay_buf_len = 2;
    load_replay_buf(
        ckernel::packer::replay_buf_offset,
        replay_buf_len,
        []
        {
            // THCON_SEC0_REG1_L1_Dest_addr_ADDR32 += SCRATCH_SEC[CurrentThread].val
            // Scratch slot loaded in _llk_pack_untilize_init_ holds the per-row L1 stride.
            // Replaces ADDDMAREG + STALLWAIT + WRCFG + NOP — saves ~3 cyc + 1 STALLWAIT per row.
            // Mirrors llk_unpack_tilize.h:285 precedent.
            TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
            TTI_NOP;
        });

    // After the inner loop finishes, update L1 address. The "advance Dst face-row" is folded
    // into the row-closing PACR's AddrMod (ADDR_MOD_1, set below), so no INCADCXY end_op is needed.
    tmp.set_end_op(lltt::replay_insn(ckernel::packer::replay_buf_offset, replay_buf_len));

    /*
    Close the row in the block by setting the Last bit to 1 in the last inner loop instruction.
    Use ADDR_MOD_1 so the packer auto-advances y_src by 1 (next row in face) post-PACR.
    Revisit after #22820 to convert last_loop_op to constexpr.
    */
    std::uint32_t last_loop_op = TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_STRIDED_MODE,
        ADDR_MOD_1,
        p_pacr::ADDR_CNT_CTXT_0,
        0,
        PACK_INTF_SEL,
        0,
        0,
        p_pacr::NO_CTXT_CTRL,
        0,
        1);

    tmp.set_last_inner_loop_instr(last_loop_op);

    tmp.set_last_outer_loop_instr(last_loop_op);

    tmp.program();
}
""",
}


FILES = {**REAL_SNIPPETS, **STUBS}
CPACK = f"{LLK}/common/inc/cpack_common.h"
PACK_COMMON = f"{LLK}/llk_lib/llk_pack_common.h"
LLK_PACK = f"{LLK}/llk_lib/llk_pack.h"
UNTILIZE = f"{LLK}/llk_lib/llk_pack_untilize.h"
FAST_TILIZE = f"{LLK}/llk_lib/experimental/llk_pack_fast_tilize.h"
IO_PACK = "tt_metal/hw/ckernels/blackhole/metal/llk_io/llk_io_pack.h"


def line_of(path: str, needle: str, nth: int = 0) -> str:
    """``path:line`` of the nth line of a fixture file containing ``needle``."""
    hits = [i for i, text in enumerate(FILES[path].splitlines(), 1) if needle in text]
    return f"{path}:{hits[nth]}"


class PackFamilyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        for relative, text in FILES.items():
            path = cls.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        cls.db = effects_db.build(cls.root, families=["pack"])

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    # ---- helpers ------------------------------------------------------------ #
    def fn(self, name: str):
        (model,) = self.db.by_name(name)
        return model

    def pack_effects(self, name: str, pattern: str | None = None):
        return [
            e
            for e in self.fn(name).effects
            if e.family == "pack" and (pattern is None or e.pattern == pattern)
        ]

    @staticmethod
    def rows(effects):
        return sorted((e.resource, e.access, e.value, e.condition) for e in effects)

    # ---- 1. set_packer_strides: WRCFG fields resolved through the SETDMAREG halves ---- #
    def test_set_packer_strides_fields(self) -> None:
        effects = self.pack_effects("set_packer_strides", "pack.cfg_field")
        y_stride = "FACE_C_DIM * (datum_size_in_bytes(pack_src_format))"
        by_resource = {e.resource: e for e in effects}
        self.assertEqual(by_resource["CFG.PCK0_ADDR_CTRL_XY_REG_0_Xstride"].value, "0")
        self.assertEqual(
            by_resource["CFG.PCK0_ADDR_CTRL_XY_REG_0_Ystride"].value, y_stride
        )
        self.assertEqual(
            by_resource["CFG.PCK0_ADDR_CTRL_XY_REG_0_Ystride"].params,
            ("pack_src_format",),
        )
        self.assertEqual(
            by_resource["CFG.PCK0_ADDR_CTRL_ZW_REG_0_Wstride"].value,
            "TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * (datum_size_in_bytes(pack_src_format))",
        )
        zstride = by_resource["CFG.PCK0_ADDR_CTRL_ZW_REG_0_Zstride"]
        self.assertEqual(
            zstride.value,
            f"((pack_mode != PackMode::Default) && (tile_c_dim == TILE_C_DIM)) ? 2 * FACE_R_DIM * ({y_stride}) : FACE_R_DIM * ({y_stride})",
        )
        self.assertEqual(zstride.params, ("pack_mode", "pack_src_format", "tile_c_dim"))
        self.assertEqual(
            by_resource["CFG.PCK0_ADDR_CTRL_XY_REG_0_Xstride"].evidence,
            line_of(
                CPACK,
                "TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32)",
            ),
        )
        self.assertEqual(
            by_resource["CFG.PCK0_ADDR_CTRL_XY_REG_0_Ystride"].detail,
            "src=wrcfg;gpr=TMP0",
        )
        for name in (
            "CFG.PCK0_ADDR_CTRL_XY_REG_1_Xstride",
            "CFG.PCK0_ADDR_CTRL_XY_REG_1_Ystride",
            "CFG.PCK0_ADDR_CTRL_ZW_REG_1_Zstride",
            "CFG.PCK0_ADDR_CTRL_ZW_REG_1_Wstride",
        ):
            self.assertEqual(
                (by_resource[name].value, by_resource[name].detail),
                ("0", "src=wrcfg;gpr=ZERO;word=0"),
            )
        self.assertEqual(len(effects), 8)
        self.assertTrue(
            all(
                e.granularity == "field" and e.access == "write" and e.thread == "pack"
                for e in effects
            )
        )

    # ---- 2. reconfigure_exp_threshold: branch-dependent values -------------------- #
    def test_exp_threshold_depends_on_branch(self) -> None:
        effects = self.pack_effects("reconfigure_exp_threshold", "pack.cfg_field")
        # disjoint cubes in source nesting order: if constexpr (is_fp32) { if (IS_BFP_A_FORMAT) .. }
        self.assertEqual(
            self.rows(effects),
            [
                (
                    "CFG.THCON_SEC0_REG1_Exp_threshold",
                    "write",
                    "0",
                    "!(is_fp32_dest_acc_en)",
                ),
                (
                    "CFG.THCON_SEC0_REG1_Exp_threshold",
                    "write",
                    "0",
                    "is_fp32_dest_acc_en && !(IS_BFP_A_FORMAT(pack_dst_format))",
                ),
                (
                    "CFG.THCON_SEC0_REG1_Exp_threshold",
                    "write",
                    "0x71",
                    "is_fp32_dest_acc_en && IS_BFP_A_FORMAT(pack_dst_format)",
                ),
                (
                    "CFG.THCON_SEC0_REG1_Exp_threshold_en",
                    "write",
                    "0",
                    "!(is_fp32_dest_acc_en)",
                ),
                (
                    "CFG.THCON_SEC0_REG1_Exp_threshold_en",
                    "write",
                    "0",
                    "is_fp32_dest_acc_en && !(IS_BFP_A_FORMAT(pack_dst_format))",
                ),
                (
                    "CFG.THCON_SEC0_REG1_Exp_threshold_en",
                    "write",
                    "1",
                    "is_fp32_dest_acc_en && IS_BFP_A_FORMAT(pack_dst_format)",
                ),
            ],
        )
        sink = line_of(
            CPACK,
            "cfg_reg_rmw_tensix<THCON_SEC0_REG1_Exp_threshold_ADDR32, 0, THRESHOLD_RMW_MASK>",
        )
        self.assertEqual({e.evidence for e in effects}, {sink})
        self.assertEqual({e.detail for e in effects}, {"src=rmw"})
        # bound from a caller: fp32 dest off -> the threshold is unconditionally disabled
        off = [
            e
            for e in self.db.call_effects(
                "reconfigure_exp_threshold", template_args=("false",), args=("fmt",)
            )
            if e.pattern == "pack.cfg_field"
        ]
        self.assertEqual(
            self.rows(off),
            [
                ("CFG.THCON_SEC0_REG1_Exp_threshold", "write", "0", ""),
                ("CFG.THCON_SEC0_REG1_Exp_threshold_en", "write", "0", ""),
            ],
        )
        on = [
            e
            for e in self.db.call_effects(
                "reconfigure_exp_threshold", template_args=("true",), args=("fmt",)
            )
            if e.pattern == "pack.cfg_field" and e.resource.endswith("_en")
        ]
        decided = sorted(
            (
                e.value,
                [
                    lit
                    for lit in pack._split_literals(e.condition)
                    if scan.truth(pack._literal_text(*lit)) is not True
                ],
            )
            for e in on
        )
        self.assertEqual(
            decided,
            [
                ("0", [("IS_BFP_A_FORMAT(fmt)", False)]),
                ("1", [("IS_BFP_A_FORMAT(fmt)", True)]),
            ],
        )

    # ---- 3. _llk_pack_relu_config_: both relu fields, param-dependent ------------- #
    def test_relu_config_covers_both_fields(self) -> None:
        effects = self.pack_effects("_llk_pack_relu_config_")
        sink = line_of(
            PACK_COMMON,
            "cfg_reg_rmw_tensix<STACC_RELU_ApplyRelu_ADDR32, 0, hw_relu_mask>(val)",
        )
        fields = {
            (e.resource, e.value) for e in effects if e.pattern == "pack.cfg_field"
        }
        self.assertEqual(
            fields,
            {
                (
                    "CFG.STACC_RELU_ApplyRelu",
                    "static_cast<std::uint32_t>(relu_config.get_mode())",
                ),
                ("CFG.STACC_RELU_ReluThreshold", "relu_config.get_threshold()"),
            },
        )
        for effect in effects:
            if effect.pattern == "pack.cfg_field":
                self.assertEqual(
                    (effect.evidence, effect.params, effect.detail),
                    (sink, ("relu_config",), "src=rmw"),
                )
        (stall,) = [e for e in effects if e.pattern == "pack.stall"]
        self.assertEqual((stall.resource, stall.access), ("PACK.BUSY", "read"))
        self.assertEqual(stall.value, "stall=p_stall::STALL_CFG;wait=p_stall::PACK")
        # the llk_api wrapper forwards the parameter
        wrapper = [
            e
            for e in self.db.effects(self.fn("llk_pack_relu_config").key)
            if e.pattern == "pack.cfg_field"
        ]
        self.assertEqual(
            {(e.resource, e.value, e.value_kind) for e in wrapper},
            {
                (
                    "CFG.STACC_RELU_ApplyRelu",
                    "static_cast<std::uint32_t>(relu_config.get_mode())",
                    "expr",
                ),
                ("CFG.STACC_RELU_ReluThreshold", "relu_config.get_threshold()", "expr"),
            },
        )
        self.assertEqual(
            {e.via[0][0] for e in wrapper}, {self.fn("_llk_pack_relu_config_").key}
        )

    # ---- 4. configure_pack: union-typed RMW data, MMIO stores, drains -------------- #
    def test_configure_pack_hw_state(self) -> None:
        effects = self.pack_effects("configure_pack")
        fields = {
            (e.resource, e.value, e.detail)
            for e in effects
            if e.pattern == "pack.cfg_field"
        }
        for row in (
            ("CFG.STACC_RELU_ApplyRelu", "relu_config & 0xffff", "src=rmw"),
            ("CFG.STACC_RELU_ReluThreshold", "(relu_config >> 16) & 0xffff", "src=rmw"),
            ("CFG.ALU_FORMAT_SPEC_REG_Dstacc_val", "0", "src=rmw;data=0"),
            ("CFG.ALU_FORMAT_SPEC_REG_Dstacc_override", "0", "src=rmw;data=0"),
            (
                "CFG.PACK_COUNTERS_SEC0_pack_reads_per_xy_plane",
                "1",
                "src=mmio;word=0x100",
            ),
            ("CFG.PACK_COUNTERS_SEC0_pack_per_xy_plane", "0", "src=mmio;word=0x100"),
            ("CFG.PCK_EDGE_OFFSET_SEC0_mask", "0xffff", "src=mmio;word=0xffff"),
            ("CFG.PCK_EDGE_TILE_ROW_SET_SELECT_select", "0", "src=mmio;word=0xffff"),
            ("CFG.TILE_ROW_SET_MAPPING_0_row_set_mapping_0", "0", "src=mmio"),
        ):
            self.assertIn(row, fields)
        counters = [
            e
            for e in effects
            if e.resource == "CFG.PACK_COUNTERS_SEC0_pack_reads_per_xy_plane"
        ]
        self.assertEqual(
            [e.evidence for e in counters],
            [
                line_of(
                    CPACK,
                    "cfg[PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32 + i] = pack_counters.val",
                )
            ],
        )
        (mapping,) = [
            e
            for e in effects
            if e.resource == "CFG.TILE_ROW_SET_MAPPING_0_row_set_mapping_0"
        ]
        self.assertEqual(
            mapping.granularity, "word"
        )  # array-like word: one effect, whole-word value
        stalls = sorted(
            (e.detail, e.value, e.evidence)
            for e in effects
            if e.pattern == "pack.stall"
        )
        self.assertEqual(
            stalls,
            [
                (
                    "stallwait",
                    "stall=p_stall::STALL_PACK;wait=p_stall::PACK",
                    line_of(
                        CPACK, "TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::PACK);"
                    ),
                ),
                ("tensix_sync", "", line_of(CPACK, "    tensix_sync();")),
            ],
        )
        # relu disabled by the caller folds to constants
        bound = self.db.call_effects(
            "configure_pack",
            template_args=("false", "PackMode::Default"),
            args=("src", "dst", "size", "FACE_R_DIM", "TILE_C_DIM", "4", "false", "0"),
        )
        relu = {
            (e.resource, e.value, e.value_kind)
            for e in bound
            if e.pattern == "pack.cfg_field" and "RELU" in e.resource
        }
        self.assertEqual(
            relu,
            {
                ("CFG.STACC_RELU_ApplyRelu", "0 & 0xffff", "const"),
                ("CFG.STACC_RELU_ReluThreshold", "(0 >> 16) & 0xffff", "const"),
            },
        )

    # ---- 5. set_packer_config: data-format word and dest read control -------------- #
    def test_set_packer_config_formats(self) -> None:
        effects = self.pack_effects("set_packer_config", "pack.cfg_field")
        by_resource: dict[str, list] = {}
        for effect in effects:
            by_resource.setdefault(effect.resource, []).append(effect)
        (out_format,) = by_resource["CFG.THCON_SEC0_REG1_Out_data_format"]
        self.assertEqual(
            (out_format.value, out_format.detail),
            ("masked_data_format(pack_dst_format)", "src=mmio"),
        )
        (uncompress,) = by_resource["CFG.THCON_SEC0_REG1_Disable_zero_compress"]
        self.assertEqual(uncompress.value, "1")
        self.assertEqual(
            sorted(
                (e.value, e.condition)
                for e in by_resource["CFG.PCK_DEST_RD_CTRL_Read_unsigned"]
            ),
            [
                ("0", "!(pack_dst_format == to_underlying(DataFormat::UInt8))"),
                ("1", "pack_dst_format == to_underlying(DataFormat::UInt8)"),
            ],
        )
        self.assertIn(
            "CFG.PCK_DEST_RD_CTRL_Read_int8", by_resource
        )  # only the hw-configure path writes Read_int8
        self.assertEqual(
            out_format.evidence,
            line_of(
                CPACK,
                "cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 2] = config.val[2];",
            ),
        )
        # word 2 of THCON_SEC0_REG1 is written whole: every field of it is reported
        word2 = {
            name
            for name in by_resource
            if self.db.tables.storage(name) and self.db.tables.storage(name)[1] == 70
        }
        self.assertEqual(len(word2), 16)

    # ---- 6. _llk_pack_reduce_mask_config_: per-dim edge masks -------------------- #
    def test_reduce_mask_config_branches(self) -> None:
        effects = self.pack_effects("_llk_pack_reduce_mask_config_", "pack.cfg_field")
        sec1 = sorted(
            (e.value, e.condition)
            for e in effects
            if e.resource == "CFG.PCK_EDGE_OFFSET_SEC1_mask"
        )
        self.assertEqual(
            sec1,
            [
                (
                    "0",
                    "!(dim == ReduceDim::REDUCE_ROW) && !(dim == ReduceDim::REDUCE_COL) && !(dim == ReduceDim::REDUCE_SCALAR)",
                ),
                ("0xffff", "dim == ReduceDim::REDUCE_COL"),
                ("1", "dim == ReduceDim::REDUCE_ROW"),
                ("1", "dim == ReduceDim::REDUCE_SCALAR"),
            ],
        )
        mapping = sorted(
            (e.value, e.condition)
            for e in effects
            if e.resource == "CFG.TILE_ROW_SET_MAPPING_1_row_set_mapping_0"
        )
        self.assertIn(
            (
                "0x11111111",
                "dim == ReduceDim::REDUCE_ROW && pack_mode == PackMode::Untilize",
            ),
            mapping,
        )
        self.assertIn(
            (
                "0x55555555",
                "dim == ReduceDim::REDUCE_ROW && !(pack_mode == PackMode::Untilize)",
            ),
            mapping,
        )
        self.assertTrue(all(e.params for e in effects if e.condition))
        bound = [
            e
            for e in self.db.call_effects(
                "_llk_pack_reduce_mask_config_",
                template_args=("ReduceDim::REDUCE_ROW", "PackMode::Default"),
                args=("16",),
            )
            if e.pattern == "pack.cfg_field"
        ]
        self.assertEqual(
            sorted((e.resource, e.value, e.condition) for e in bound),
            [
                ("CFG.PCK_EDGE_MODE_mode", "0", ""),
                ("CFG.PCK_EDGE_OFFSET_SEC0_mask", "0", ""),
                ("CFG.PCK_EDGE_OFFSET_SEC1_mask", "1", ""),
                ("CFG.PCK_EDGE_TILE_ROW_SET_SELECT_select", "0x55", ""),
                ("CFG.TILE_ROW_SET_MAPPING_1_row_set_mapping_0", "0x55555555", ""),
            ],
        )

    # ---- 7. _llk_pack_reduce_mask_clear_: restore of the pass-through masks ------- #
    def test_reduce_mask_clear_restores(self) -> None:
        effects = self.pack_effects("_llk_pack_reduce_mask_clear_", "pack.cfg_field")
        self.assertEqual(
            sorted((e.resource, e.value, e.granularity, e.detail) for e in effects),
            [
                (
                    "CFG.PCK_EDGE_MODE_mode",
                    "0",
                    "field",
                    "src=wrcfg;gpr=TMP0;word=0xffff",
                ),
                (
                    "CFG.PCK_EDGE_OFFSET_SEC0_mask",
                    "0xffff",
                    "field",
                    "src=wrcfg;gpr=TMP0;word=0xffff",
                ),
                (
                    "CFG.PCK_EDGE_OFFSET_SEC1_mask",
                    "0xffff",
                    "field",
                    "src=wrcfg;gpr=TMP0;word=0xffff",
                ),
                (
                    "CFG.PCK_EDGE_TILE_ROW_SET_SELECT_select",
                    "0",
                    "field",
                    "src=wrcfg;gpr=TMP0;word=0xffff",
                ),
                (
                    "CFG.TILE_ROW_SET_MAPPING_0_row_set_mapping_0",
                    "0",
                    "word",
                    "src=wrcfg;gpr=ZERO",
                ),
                (
                    "CFG.TILE_ROW_SET_MAPPING_1_row_set_mapping_0",
                    "0",
                    "word",
                    "src=wrcfg;gpr=ZERO",
                ),
            ],
        )
        self.assertEqual(self.fn("_llk_pack_reduce_mask_clear_").role, "uninit")

    # ---- 8./9. destination address and dest tile offset --------------------------- #
    def test_destination_programming(self) -> None:
        (dest,) = self.pack_effects("program_packer_destination", "pack.cfg_field")
        self.assertEqual(
            (dest.resource, dest.value, dest.params, dest.evidence),
            (
                "CFG.THCON_SEC0_REG1_L1_Dest_addr",
                "addr | 0x80000000",
                ("addr",),
                line_of(
                    CPACK,
                    "TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);",
                ),
            ),
        )
        self.assertEqual(self.fn("program_packer_destination").role, "helper")
        key = self.fn("program_packer_dest_offset_registers").key
        (base,) = [e for e in self.db.effects(key) if e.pattern == "pack.cfg_field"]
        self.assertEqual(
            (base.resource, base.value, base.value_kind, base.detail),
            (
                "CFG.PCK0_ADDR_BASE_REG_0_Base",
                "dest_tile_offset",
                "param",
                "src=wrcfg;gpr=TEMP_TILE_OFFSET",
            ),
        )

    # ---- 10. select_packer_dest_registers: which dest-offset GPR is read ---------- #
    def test_dest_offset_index_reads(self) -> None:
        effects = self.pack_effects(
            "select_packer_dest_registers", "pack.gpr_read.dest_offset"
        )
        self.assertEqual(
            self.rows(effects),
            [
                (
                    "GPR.pack.DEST_OFFSET_HI",
                    "read",
                    "",
                    "!(Dst == DstSync::SyncFull) && dest_offset_id",
                ),
                (
                    "GPR.pack.DEST_OFFSET_LO",
                    "read",
                    "",
                    "!(Dst == DstSync::SyncFull) && !(dest_offset_id)",
                ),
                ("SW.dest_offset_id", "read", "", "!(Dst == DstSync::SyncFull)"),
            ],
        )
        full = [
            e
            for e in self.db.call_effects(
                "select_packer_dest_registers", template_args=("DstSync::SyncFull",)
            )
            if e.pattern == "pack.gpr_read.dest_offset"
        ]
        self.assertEqual(full, [])
        half = [
            e
            for e in self.db.call_effects(
                "select_packer_dest_registers", template_args=("DstSync::SyncHalf",)
            )
            if e.pattern == "pack.gpr_read.dest_offset"
        ]
        # SyncHalf: both GPRs stay possible, selected by dest_offset_id (the bound Dst literal is
        # true; the core folds a condition only as a whole, so it may remain in the text)
        rows = {}
        for effect in half:
            literals = pack._split_literals(effect.condition)
            rest = [lit for lit in literals if lit[0] != "dest_offset_id"]
            self.assertTrue(
                all(scan.truth(pack._literal_text(*lit)) is True for lit in rest),
                effect.condition,
            )
            rows[effect.resource] = [
                lit for lit in literals if lit[0] == "dest_offset_id"
            ]
        self.assertEqual(
            rows,
            {
                "GPR.pack.DEST_OFFSET_HI": [("dest_offset_id", True)],
                "GPR.pack.DEST_OFFSET_LO": [("dest_offset_id", False)],
                "SW.dest_offset_id": [],
            },
        )

    # ---- 11. _llk_pack_: a MOP run reads the packer state -------------------------- #
    def test_pack_execution_reads(self) -> None:
        effects = self.pack_effects("_llk_pack_")
        run_line = line_of(LLK_PACK, "ckernel::ckernel_template::run();")
        reads = {e.resource for e in effects if e.pattern == "pack.exec.read"}
        expected = {
            f"CFG.{name}"
            for name, _ in pack.PACK_EXEC_READS
            if name in self.db.tables.cfg.fields
        }
        expected |= set(pack.PACK_EXEC_ADC) | {"REPLAY"}
        self.assertEqual(reads, expected)
        self.assertIn("CFG.STACC_RELU_ApplyRelu", reads)
        self.assertIn("CFG.THCON_SEC0_REG1_Pack_L1_Acc", reads)
        self.assertEqual(
            {(e.evidence, e.detail, e.thread) for e in effects},
            {(run_line, "mop_run", "pack")},
        )
        (busy,) = [e for e in effects if e.pattern == "pack.exec.busy"]
        self.assertEqual(
            (busy.resource, busy.access, busy.value), ("PACK.BUSY", "write", "mop_run")
        )
        granularity = {e.resource: e.granularity for e in effects}
        self.assertEqual(granularity["CFG.PCK_EDGE_OFFSET_SEC0_mask"], "word")
        self.assertEqual(
            granularity["CFG.STACC_RELU_ReluThreshold"], "field"
        )  # word shared with ALU zero flags

    # ---- 12. replay-recorded instructions are not executions ---------------------- #
    def test_recorded_instructions_are_deferred(self) -> None:
        self.assertEqual(self.pack_effects("_llk_pack_fast_tilize_load_replay_"), [])
        self.assertEqual(self.pack_effects("_llk_pack_untilize_mop_config_"), [])
        # the core still reports the recorded WRCFG; the family adds no field values for it
        core = [
            e
            for e in self.fn("_llk_pack_fast_tilize_load_replay_").effects
            if e.pattern == "core.wrcfg.cfg"
        ]
        self.assertEqual(
            [e.resource for e in core], ["CFG.THCON_SEC0_REG1_L1_Dest_addr"]
        )
        self.assertEqual(self.fn("_llk_pack_fast_tilize_load_replay_").role, "init")

    # ---- 13. cfg[] MMIO reads re-normalize under substitution ---------------------- #
    def test_cfg_reads(self) -> None:
        direct = self.pack_effects("read_pack_config_helper", "pack.cfg_read")
        self.assertEqual(
            sorted((e.resource, e.resolved) for e in direct),
            [
                ("CFG.?reg_addr", False),
                ("CFG.?reg_addr + 1", False),
                ("CFG.?reg_addr + 2", False),
            ],
        )
        reads = [
            e
            for e in self.db.effects(self.fn("read_pack_config").key)
            if e.pattern == "pack.cfg_read"
        ]
        self.assertEqual(
            sorted((e.resource, e.granularity) for e in reads),
            [
                ("CFG.THCON_SEC0_REG1_Disable_zero_compress", "word"),
                ("CFG.THCON_SEC0_REG1_L1_Dest_addr", "word"),
                ("CFG.THCON_SEC0_REG1_Row_start_section_size", "word"),
            ],
        )

    # ---- 14. pack-side CB pointers and STOREREG ----------------------------------- #
    def test_cb_state_and_storereg(self) -> None:
        effects = self.pack_effects("llk_push_tiles", "pack.cb_state")
        writes = sorted(
            (e.resource, e.value, e.condition) for e in effects if e.access == "write"
        )
        self.assertEqual(
            writes,
            [
                ("SW.cb_interface.fifo_wr_ptr", "+= num_words", ""),
                (
                    "SW.cb_interface.fifo_wr_ptr",
                    "-= cb.fifo_size",
                    "cb.fifo_wr_ptr >= cb.fifo_limit",
                ),
                ("SW.cb_interface.fifo_wr_tile_ptr", "0", ""),
            ],
        )
        self.assertEqual({e.detail for e in effects}, {"cb=output"})
        (store,) = self.pack_effects("llk_push_to_brisc", "pack.storereg")
        self.assertEqual(
            (store.resource, store.access), ("GPR.pack.NUM_MSGS_RECEIVED", "read")
        )
        self.assertEqual(
            store.evidence,
            line_of(IO_PACK, "TT_STOREREG(p_gpr_pack::NUM_MSGS_RECEIVED"),
        )

    # ---- 15. _llk_pack_dest_section_done_: branch on clear_dest / Dst ------------- #
    def test_dest_section_done_branches(self) -> None:
        effects = self.pack_effects("_llk_pack_dest_section_done_")
        self.assertEqual(
            sorted((e.resource, e.pattern, e.condition) for e in effects),
            [
                ("PACK.BUSY", "pack.stall", "clear_dest"),
                (
                    "SW.dest_offset_id",
                    "pack.sw_read",
                    "clear_dest && !(Dst == DstSync::SyncFull)",
                ),
            ],
        )
        kept = self.db.call_effects(
            "_llk_pack_dest_section_done_",
            template_args=("DstSync::SyncFull", "false", "false"),
        )
        self.assertEqual(
            [
                e
                for e in kept
                if e.family == "pack"
                and e.pattern in ("pack.stall", "pack.sw_read")
                and not e.via
            ],
            [],
        )

    # ---- hooks, groups, scope, determinism ----------------------------------------- #
    def test_hooks_groups_and_scope(self) -> None:
        self.assertEqual(self.fn("set_dst_write_addr").role, "helper")
        self.assertEqual(self.fn("_llk_pack_reconfig_l1_acc_").role, "reconfig")
        groups = self.db.tables.groups_of
        self.assertIn("pack.hw_config_owned", groups("CFG.STACC_RELU_ReluThreshold"))
        self.assertIn("pack.hw_config_owned", groups("CFG.THCON_SEC0_REG1_Pack_L1_Acc"))
        self.assertIn(
            "pack.reconfig_owned", groups("CFG.THCON_SEC0_REG1_L1_source_addr")
        )
        self.assertIn("pack.init_owned", groups("ADDR_MOD_PACK.0"))
        self.assertIn("pack.execute_owned", groups("CFG.THCON_SEC0_REG1_L1_Dest_addr"))
        self.assertTrue(
            pack.in_scope(CPACK)
            and pack.in_scope(IO_PACK)
            and pack.in_scope(FAST_TILIZE)
        )
        self.assertFalse(pack.in_scope(f"{LLK}/llk_lib/llk_unpack_A.h"))
        self.assertFalse(pack.in_scope(f"{LLK}/common/inc/ckernel.h"))
        outside = [e for e in self.fn("tensix_sync").effects if e.family == "pack"]
        self.assertEqual(outside, [])
        own = pack.ownership(self.db)
        self.assertEqual(
            own["CFG.STACC_RELU_ApplyRelu"], {"config": ["llk_pack_relu_config"]}
        )

    def test_deterministic_json(self) -> None:
        first = effects_db.dumps_effects(self.db.to_json())
        second = effects_db.dumps_effects(
            effects_db.build(self.root, families=["pack"]).to_json()
        )
        self.assertEqual(first, second)
        self.assertNotIn(str(self.root), first)


class PackHelpersTest(unittest.TestCase):
    def test_split_literals(self) -> None:
        self.assertEqual(
            pack._split_literals("!(a == B::C) && x && !(p && q)"),
            [("a == B::C", False), ("x", True), ("p && q", False)],
        )
        self.assertEqual(pack._split_literals(None), [])

    def test_decision_tree_cover_is_disjoint_and_ordered(self) -> None:
        atoms = ["d == R::A", "u", "d == R::B"]
        table = {}
        for a in (True, False):
            for u in (True, False):
                for b in (True, False):
                    if a and b:
                        continue  # infeasible: d cannot equal both
                    value = "x" if a else ("y" if b else "z")
                    table[
                        tuple(sorted({"d == R::A": a, "u": u, "d == R::B": b}.items()))
                    ] = value
        cover = pack._decide(atoms, table, {})
        self.assertEqual(
            {
                value: [pack._simplify_cube(c) for c in cubes]
                for value, cubes in cover.items()
            },
            {
                "x": [{"d == R::A": True}],
                "y": [{"d == R::B": True}],
                "z": [{"d == R::A": False, "d == R::B": False}],
            },
        )
        self.assertFalse(pack._feasible({"d == R::A": True, "d == R::B": True}))

    def test_field_extraction(self) -> None:
        cfg = pack._Cfg.__new__(pack._Cfg)
        cfg.fields, cfg.words = {}, {}
        cfg.tables = type(
            "T",
            (),
            {
                "int_value": staticmethod(
                    lambda text: {"0": 0, "31": 31, "1": 1, "16": 16}.get(text.strip())
                )
            },
        )()
        self.assertEqual(
            pack._extract("(1 << 31) | addr", 0, 32, cfg), "addr | 0x80000000"
        )
        self.assertEqual(
            pack._extract("(1 << 31) | addr", 31, 1, cfg), None
        )  # addr may reach bit 31
        self.assertEqual(pack._extract("y << 16", 0, 16, cfg), 0)
        self.assertEqual(
            pack._extract("y << 16", 16, 16, cfg), "y"
        )  # top field: nothing to mask
        self.assertEqual(pack._extract("y << 16", 16, 8, cfg), "(y) & 0xff")


if __name__ == "__main__":
    unittest.main()
