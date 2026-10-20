# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Blackhole math family (``bh/families/math.py``) on real snippets.

Every snippet below is a verbatim excerpt of the Blackhole math scope on main
(8a7beeeaede), stored under its real repo-relative path. A function body is never
edited; ``reduce_custom`` and ``eltwise_binary_custom`` keep the real function head
and close it after the excerpted lines. The tree also gets the real
``cfg_defines.h`` and ``ckernel_gpr_map.h`` of the checkout. Expected evidence lines
are computed from the snippet text, so every expectation is an exact
``path:line``.
"""

import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.llk_state_audit.bh import effects_db, resources
from tools.llk_state_audit.bh.families import math as math_family

REPO_ROOT = Path(__file__).resolve().parents[5]
LLK = "tt_metal/tt-llk/tt_llk_blackhole/"
API = "tt_metal/hw/ckernels/blackhole/metal/llk_api/"
CMATH = LLK + "common/inc/cmath_common.h"
CKERNEL = LLK + "common/inc/ckernel.h"
TEMPLATE = LLK + "common/inc/ckernel_template.h"
COMMON = LLK + "llk_lib/llk_math_common.h"
BINARY = LLK + "llk_lib/llk_math_eltwise_binary.h"
REDUCE = LLK + "llk_lib/experimental/llk_math_reduce_custom.h"
CUSTOM = LLK + "llk_lib/experimental/llk_math_eltwise_binary_custom.h"
MATMUL = LLK + "llk_lib/llk_math_matmul.h"
BINARY_API = API + "llk_math_binary_api.h"

SNIPPETS = {
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_addrmod.h": r"""
namespace ckernel
{

constexpr std::uint8_t ADDR_MOD_0 = 0;
constexpr std::uint8_t ADDR_MOD_1 = 1;
constexpr std::uint8_t ADDR_MOD_2 = 2;
constexpr std::uint8_t ADDR_MOD_3 = 3;
constexpr std::uint8_t ADDR_MOD_4 = 4;
constexpr std::uint8_t ADDR_MOD_5 = 5;
constexpr std::uint8_t ADDR_MOD_6 = 6;
constexpr std::uint8_t ADDR_MOD_7 = 7;

constexpr std::uint32_t SRC_INCR_MASK  = 0x3F;
constexpr std::uint32_t DEST_INCR_MASK = 0x3FF;

// FIXME: These should be updated from cfg_defines.h

struct addr_mod_t
{
    // CLR, CR, INCR(4 bits)
    struct addr_mod_src_t
    {
        std::uint8_t incr = 0;
        std::uint8_t clr  = 0;
        std::uint8_t cr   = 0;

        constexpr std::uint8_t val() const
        {
            return (incr & SRC_INCR_MASK) | ((cr & 0x1) << 6) | ((clr & 0x1) << 7);
        }
    };

    // CLR, CR, INCR(8 bits)
    struct addr_mod_dest_t
    {
        std::int16_t incr    = 0;
        std::uint8_t clr     = 0;
        std::uint8_t cr      = 0;
        std::uint8_t c_to_cr = 0;

        constexpr std::uint16_t val() const
        {
            return (incr & DEST_INCR_MASK) | ((cr & 0x1) << 10) | ((clr & 0x1) << 11) | ((c_to_cr & 0x1) << 12);
        }
    };

    // CLR, INCT (2 bits)
    struct addr_mod_fidelity_t
    {
        std::uint8_t incr = 0;
        std::uint8_t clr  = 0;

        constexpr std::uint16_t val() const
        {
            return (incr & 0x3) | ((clr & 0x1) << 2);
        }
    };

    // CLR, INCT (4 bits)
    struct addr_mod_bias_t
    {
        std::uint8_t incr = 0;
        std::uint8_t clr  = 0;

        constexpr std::uint16_t val() const
        {
            return (incr & 0xF) | ((clr & 0x1) << 4);
        }
    };

    // Set defaults so that we can skip unchanged in initialization list
    addr_mod_src_t srca          = {};
    addr_mod_src_t srcb          = {};
    addr_mod_dest_t dest         = {};
    addr_mod_fidelity_t fidelity = {};
    addr_mod_bias_t bias         = {};
    addr_mod_src_t pack_ysrc     = {};
    addr_mod_src_t pack_ydst     = {};

    // SrcA/B register is combination of A and B values
    constexpr std::uint16_t src_val() const
    {
        return srca.val() | (srcb.val() << 8);
    }

    constexpr std::uint16_t pack_val() const
    {
        return pack_ysrc.val() | (pack_ydst.val() << 6);
    }

    // List of addresses of src/dest registers
    constexpr static std::uint32_t addr_mod_src_reg_addr[] = {
        ADDR_MOD_AB_SEC0_SrcAIncr_ADDR32,
        ADDR_MOD_AB_SEC1_SrcAIncr_ADDR32,
        ADDR_MOD_AB_SEC2_SrcAIncr_ADDR32,
        ADDR_MOD_AB_SEC3_SrcAIncr_ADDR32,
        ADDR_MOD_AB_SEC4_SrcAIncr_ADDR32,
        ADDR_MOD_AB_SEC5_SrcAIncr_ADDR32,
        ADDR_MOD_AB_SEC6_SrcAIncr_ADDR32,
        ADDR_MOD_AB_SEC7_SrcAIncr_ADDR32};

    constexpr static std::uint32_t addr_mod_dest_reg_addr[] = {
        ADDR_MOD_DST_SEC0_DestIncr_ADDR32,
        ADDR_MOD_DST_SEC1_DestIncr_ADDR32,
        ADDR_MOD_DST_SEC2_DestIncr_ADDR32,
        ADDR_MOD_DST_SEC3_DestIncr_ADDR32,
        ADDR_MOD_DST_SEC4_DestIncr_ADDR32,
        ADDR_MOD_DST_SEC5_DestIncr_ADDR32,
        ADDR_MOD_DST_SEC6_DestIncr_ADDR32,
        ADDR_MOD_DST_SEC7_DestIncr_ADDR32};

    constexpr static std::uint32_t addr_mod_bias_reg_addr[] = {
        ADDR_MOD_BIAS_SEC0_BiasIncr_ADDR32,
        ADDR_MOD_BIAS_SEC1_BiasIncr_ADDR32,
        ADDR_MOD_BIAS_SEC2_BiasIncr_ADDR32,
        ADDR_MOD_BIAS_SEC3_BiasIncr_ADDR32,
        ADDR_MOD_BIAS_SEC4_BiasIncr_ADDR32,
        ADDR_MOD_BIAS_SEC5_BiasIncr_ADDR32,
        ADDR_MOD_BIAS_SEC6_BiasIncr_ADDR32,
        ADDR_MOD_BIAS_SEC7_BiasIncr_ADDR32};

    constexpr static std::uint32_t addr_mod_pack_reg_addr[] = {
        ADDR_MOD_PACK_SEC0_YsrcIncr_ADDR32, ADDR_MOD_PACK_SEC1_YsrcIncr_ADDR32, ADDR_MOD_PACK_SEC2_YsrcIncr_ADDR32, ADDR_MOD_PACK_SEC3_YsrcIncr_ADDR32};

    // Program source and dest registers
    __attribute__((always_inline)) inline void set(const std::uint8_t mod_index) const
    {
        // KCM - This gets around issue: error: impossible constraint in 'asm'
        // TTI_SETC16(addr_mod_src_reg_addr[mod_index], src_val());
        TTI_SETC16(addr_mod_src_reg_addr[mod_index], srca.val() | (srcb.val() << 8));
        TTI_SETC16(addr_mod_dest_reg_addr[mod_index], dest.val() | (fidelity.val() << 13));
        TTI_SETC16(addr_mod_bias_reg_addr[mod_index], bias.val());
    }
};
} // namespace ckernel
""",
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_template.h": r"""
namespace ckernel
{
class ckernel_template
{
    // Here is the basic outline of a MOP loop and the definition of the
    // variables used.
    // LOOP_OUTER: <OUTER_LOOP_COUNT>
    //   START_OP
    //   LOOP_INNER: <INNER_LOOP_COUNT>
    //     LOOP_OP0
    //     LOOP_OP1
    //   END_LOOP_INNER
    //   END_OP0
    //   END_OP1
    // END_LOOP_OUTER

    const std::uint32_t m_outer_loop_len;
    const std::uint32_t m_inner_loop_len;
    std::uint32_t m_loop_op0;
    std::uint32_t m_loop_op1;
    std::uint32_t m_end_op0, m_end_op1, m_start_op0;
    std::uint32_t
        m_loop0_last_instr; // In the last iteration of the outer loop, this instruction replaces the inner loop instruction, if constructed with one inner
                            // loop instruction or the second inner loop instruction, if constructed with two inner loop instructions (see below example).
    std::uint32_t
        m_loop1_last_instr; // In the last iteration of the inner loop, this instruction replaces the inner loop instruction, if constructed with one inner
                            // loop instruction or the second inner loop instruction, if constructed with two inner loop instructions (see below example).

    // Note: The last iteration of inner loop will also be the last iteration of the outer loop when outer loop length = 1.
    // This means that in this case, last_inner_loop_instr will be replaced by the last_outer_loop_instr
    // NOTE:
    // This is how m_loop0_last_instr and m_loop1_last_instr are executed:
    //
    // if(last_inner_loop_iter && last_outer_loop_iter) m_loop_op1 = m_loop0_last_instr;
    // else if(last_inner_loop_iter)                    m_loop_op1 = m_loop1_last_instr;
    // else                                             m_loop_op1 = m_loop_op1;

public:
    ckernel_template() = delete;
    ~ckernel_template();
    ckernel_template(ckernel_template const &)            = delete;
    ckernel_template(ckernel_template &&)                 = delete;
    ckernel_template &operator=(ckernel_template const &) = delete;
    ckernel_template &operator=(ckernel_template &&)      = delete;

    ckernel_template(std::uint32_t outer_loop_len, std::uint32_t inner_loop_len);
    ckernel_template(std::uint32_t outer_loop_len, std::uint32_t inner_loop_len, std::uint32_t loop_op);
    ckernel_template(std::uint32_t outer_loop_len, std::uint32_t inner_loop_len, std::uint32_t loop_op0, std::uint32_t loop_op1);
    void set_loop_op0(std::uint32_t loop_op);
    void set_loop_op1(std::uint32_t loop_op);
    void set_end_ops(std::uint32_t end_op0, std::uint32_t end_op1);
    void set_end_op(std::uint32_t end_op0);
    void set_start_op(std::uint32_t start_op0);
    void set_last_inner_loop_instr(std::uint32_t op);
    void set_last_outer_loop_instr(std::uint32_t op);

    void program();         // just programs the registers
    static void run();      // runs - assumes that registers were already programmed
    void program_and_run(); // calls program, then run
};

inline ckernel_template::ckernel_template(std::uint32_t outer_loop_len, std::uint32_t inner_loop_len) :
    m_outer_loop_len(outer_loop_len),
    m_inner_loop_len(inner_loop_len),
    m_loop_op0(TT_OP_NOP),
    m_loop_op1(TT_OP_NOP),
    m_end_op0(TT_OP_NOP),
    m_end_op1(TT_OP_NOP),
    m_start_op0(TT_OP_NOP),
    m_loop0_last_instr(TT_OP_NOP),
    m_loop1_last_instr(TT_OP_NOP)
{
}

inline ckernel_template::ckernel_template(std::uint32_t outer_loop_len, std::uint32_t inner_loop_len, std::uint32_t loop_op) :
    m_outer_loop_len(outer_loop_len),
    m_inner_loop_len(inner_loop_len),
    m_loop_op0(loop_op),
    m_loop_op1(TT_OP_NOP),
    m_end_op0(TT_OP_NOP),
    m_end_op1(TT_OP_NOP),
    m_start_op0(TT_OP_NOP)
{
    m_loop0_last_instr = loop_op;
    m_loop1_last_instr = loop_op;
}

inline ckernel_template::ckernel_template(std::uint32_t outer_loop_len, std::uint32_t inner_loop_len, std::uint32_t loop_op0, std::uint32_t loop_op1) :
    m_outer_loop_len(outer_loop_len),
    m_inner_loop_len(inner_loop_len),
    m_loop_op0(loop_op0),
    m_loop_op1(loop_op1),
    m_end_op0(TT_OP_NOP),
    m_end_op1(TT_OP_NOP),
    m_start_op0(TT_OP_NOP)
{
    m_loop0_last_instr = loop_op1;
    m_loop1_last_instr = loop_op1;
}

inline ckernel_template::~ckernel_template()
{
    // Empty destructor.
}

inline void ckernel_template::set_loop_op0(std::uint32_t loop_op)
{
    m_loop_op0 = loop_op;
}

inline void ckernel_template::set_loop_op1(std::uint32_t loop_op)
{
    m_loop_op1 = loop_op;
}

inline void ckernel_template::set_end_ops(std::uint32_t end_op0, std::uint32_t end_op1)
{
    m_end_op0 = end_op0;
    m_end_op1 = end_op1;
}

inline void ckernel_template::set_end_op(std::uint32_t end_op0)
{
    set_end_ops(end_op0, TT_OP_NOP);
}

inline void ckernel_template::set_start_op(std::uint32_t start_op0)
{
    m_start_op0 = start_op0;
}

inline void ckernel_template::set_last_inner_loop_instr(std::uint32_t op)
{
    m_loop1_last_instr = op;
}

inline void ckernel_template::set_last_outer_loop_instr(std::uint32_t op)
{
    m_loop0_last_instr = op;
}

inline void ckernel_template::program_and_run()
{
    program();
    run();
}

inline void ckernel_template::run()
{
    TTI_MOP(1, 0, 0); // run the double-loop template
}

inline void ckernel_template::program()
{
    volatile std::uint32_t *mop_cfg = reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);

    mop_sync(); // wait until previous mops have completed

    mop_cfg[0] = m_outer_loop_len;
    mop_cfg[1] = m_inner_loop_len;
    mop_cfg[2] = m_start_op0;
    mop_cfg[3] = m_end_op0;
    mop_cfg[4] = m_end_op1;
    mop_cfg[5] = m_loop_op0;
    mop_cfg[6] = m_loop_op1;
    mop_cfg[7] = m_loop0_last_instr;
} // namespace ckernel
""",
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel.h": r"""
namespace ckernel
{
extern std::uint32_t dest_offset_id;

inline void reset_dest_offset_id()
{
    dest_offset_id = 0;
}

inline void update_dest_offset_id()
{
    // ping-pong between 0 and 1
    dest_offset_id = 1 - dest_offset_id;
}

inline std::uint32_t get_dest_buffer_base()
{
    return (0 != dest_offset_id) ? DEST_REGISTER_HALF_SIZE : 0x0;
}

#if defined(COMPILE_FOR_TRISC)
// Place instructions into the replay buffer. EXEC is true to execute
// when loading (default is false). START is where to place in the
// replay buffer, and LEN is the number of instructions to record
// (should match the expansion of CALLABLE). CALLABLE is a callable,
// to which ARGS are forwarded.
// When we move to c++23 we can use 'using enum lltt::ExecBool;'
enum ExecBool : bool
{
    NoExec,
    Exec
};

template <ExecBool Exec = NoExec, typename Callable, typename... Args>
[[gnu::always_inline, gnu::flatten]] inline void load_replay_buf(std::uint32_t start, std::uint32_t len, Callable &&callable, Args &&...args)
{
    // ENABLE_GATHERING is controlled by JIT build.
    // Not enabled by default due to tt-metal#16439.
#if defined(ENABLE_GATHERING)
    disable_gathering();
#endif

    // Issue instruction to load replay buffer
    lltt::record<lltt::ExecBool(Exec)>(start, len);

    // Send in the user's desired instructions
    callable(std::forward<Args>(args)...);

#if defined(ENABLE_GATHERING)
    enable_gathering();
#endif
}
#endif // defined(COMPILE_FOR_TRISC)
} // namespace ckernel
""",
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/cmath_common.h": r"""
using namespace ckernel;

namespace ckernel::math
{

constexpr std::uint32_t replay_buf_offset = 16; // split replay buffer usage between fpu/sfpu

static std::uint32_t src_zero_flag_hw       = 0xff; // last value written to the flag; 0xff = unknown
static std::uint32_t src_zero_flag_srca_fmt = 0xff; // cached operand formats feeding the compute default
static std::uint32_t src_zero_flag_srcb_fmt = 0xff;

// The one writer. Out-of-line so the write exists in a single copy (code size — a matmul kernel
// otherwise overflows its slot).
// No STALLWAIT: every reader (SrcB zero-flag generation, the SrcA MOV path) latches this flag when its
// FPU instruction issues and carries it down the pipe, so the write cannot change an instruction already
// in flight. The SFPU never reads it.
inline __attribute__((noinline)) void _apply_src_zero_flag_(const std::uint32_t value)
{
    src_zero_flag_hw = value;
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(value);
}

// Set the flag to `disable`; skip if it already holds that value. The check is inlined at every call
// site so hot loops (whose operand formats change but map to the same flag value) stay call-free; only
// a genuine value change pays the out-of-line write.
inline void _configure_src_zero_flag_(const bool disable)
{
    const std::uint32_t value = disable ? 1u : 0u;
    if (src_zero_flag_hw == value)
    {
        return;
    }
    _apply_src_zero_flag_(value);
}

// A kernel tight on program-config space (e.g. ring-joint SDPA, which reconfigs ~30x) can
// #define LLK_ZEROFLAG_OUTLINE before its includes to force this configurator out-of-line -- one copy
// called from each reconfig/init site instead of an inlined fast path at every one, trading a call for
// code size. Perf-critical kernels (groupnorm welford) leave it inlined (the default).
#ifdef LLK_ZEROFLAG_OUTLINE
#define LLK_ZEROFLAG_DEFAULT_ATTR __attribute__((noinline))
#else
#define LLK_ZEROFLAG_DEFAULT_ATTR
#endif

// FP compute / format reconfig: the flag follows the operand formats. Reads the cached SrcA/SrcB formats
// -- maintained by the reconfig sites, the only places the SrcA/SrcB format actually changes -- and applies
// the operand-driven value, skipping the write when the flag already holds it (the steady
// state in hot loops). Takes no format args and stores nothing, so the inlined fast path at every init site
// stays tiny AND the caches can never diverge (they are refreshed on every format change, independent of
// whether the resulting flag value changed).
// TODO(tt-metal#53652): the flag must be CLEARED for all FPU compute; once that lands this collapses to
// _configure_src_zero_flag_(false) and requires_disabled_src_zero_flag() / the format cache are dropped.
inline LLK_ZEROFLAG_DEFAULT_ATTR void _configure_default_zero_flag_state_()
{
    const std::uint32_t value = requires_disabled_src_zero_flag(src_zero_flag_srca_fmt, src_zero_flag_srcb_fmt) ? 1u : 0u;
    if (src_zero_flag_hw == value)
    {
        return;
    }
    _apply_src_zero_flag_(value);
}

// Data-movement ops keep the flag set so values pass through faithfully (bf16 -0.0, 16b/32b ints).
inline void _configure_preserve_zero_flag_state_()
{
    _configure_src_zero_flag_(true);
}

// Datacopy zero-flag, chosen by the source (SrcA) format. Default is preserve (keep) so bf16 -0.0 and
// 16-bit integer datums survive the move. Exception: fp8 (e4m3 / e5m2) sources widen into a SrcA datum
// whose zero carries a nonzero high residual, so preserve would read it back as ~2^-15; those must be
// flushed (zero-substituted) to produce a clean 0. Src format carries extra high bits, so mask to 0x1F.
inline void _configure_copy_zero_flag_state_(const std::uint32_t src_dst_format)
{
    const std::uint32_t fmt = src_dst_format & 0x1F;
    const bool flush_fp8    = (fmt == static_cast<std::uint32_t>(DataFormat::Fp8_e4m3)) || (fmt == static_cast<std::uint32_t>(DataFormat::Lf8));
    _configure_src_zero_flag_(!flush_fp8);
}

// After a raw cfg write that bypassed the setter, mark the tracked value unknown so the next
// _configure_ re-applies from a known baseline.
inline void _invalidate_src_zero_flag_state_()
{
    src_zero_flag_hw = 0xff;
}

inline void reset_counters(const std::uint32_t setrwc)
{
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, setrwc);
}

inline void incr_counters(const std::uint32_t incr_a, const std::uint32_t incr_b, const std::uint32_t incr_d, const std::uint32_t incr_cr)
{
    TT_INCRWC(incr_cr, incr_d, incr_b, incr_a);
}

inline void move_d2a_fixed_face(const std::uint8_t addrmod)
{
    // Drain preceding math instructions so their source-bank release is visible before testing
    // the DVALID of the bank that MOVD2A will write.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCA_VLD);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 0, addrmod, p_movd2a::MOV_4_ROWS, 0);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 4, addrmod, p_movd2a::MOV_4_ROWS, 4);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 8, addrmod, p_movd2a::MOV_4_ROWS, 8);
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 12, addrmod, p_movd2a::MOV_4_ROWS, 12);
}

template <std::uint32_t SrcReg>
inline void clear_bank_valid()
{
    if constexpr (SrcReg == Srcs::SrcA)
    {
        TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_A);
    }
    else
    {
        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_B);
    }
}

template <DstTileShape tile_shape, UnpackDestination unpack_destination>
inline void set_dst_write_addr(std::uint32_t tile_index)
{
    static_assert(
        tile_shape == DstTileShape::Tile32x32 || tile_shape == DstTileShape::Tile32x16 || tile_shape == DstTileShape::Tile16x16, "Invalid tile shape");
    static_assert(DstTileShape::Tile32x32 == 0, "DstTileShape::Tile32x32 must equal 0");
    static_assert(DstTileShape::Tile32x16 == 1, "DstTileShape::Tile32x16 must equal 1");
    static_assert(DstTileShape::Tile16x16 == 2, "DstTileShape::Tile16x16 must equal 2");
    static_assert(DstTileSizeLog2[DstTileShape::Tile32x32] == 6, "DstTileSizeLog2[Tile32x32] must equal 6");
    static_assert(DstTileSizeLog2[DstTileShape::Tile32x16] == 5, "DstTileSizeLog2[Tile32x16] must equal 5");
    static_assert(DstTileSizeLog2[DstTileShape::Tile16x16] == 4, "DstTileSizeLog2[Tile16x16] must equal 4");

    std::uint32_t dst_index = tile_index << DstTileSizeLog2[tile_shape];
    dst_index               = dst_index + get_dest_buffer_base();
    if constexpr (unpack_destination == UnpackDestination::DestReg)
    {
        mailbox_write(ThreadId::UnpackThreadId, dst_index); // Send to unpacker
    }
    else
    {
        TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index);
    }
}

// Programming a dst write addr offset that gets added to base
//
inline void clear_dst_reg_addr()
{
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

template <std::uint32_t num_rows = 8>
inline void inc_dst_addr()
{
    static_assert(num_rows <= 15, "num_rows must be <= 15");
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, num_rows, 0, 0, p_setrwc::SET_D);
}

inline void dest_section_flip()
{
    update_dest_offset_id();
    std::uint32_t base_addr = get_dest_buffer_base();
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::SFPU1);
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, base_addr);
}

template <DstStart Dst>
inline void set_dest_section_base()
{
    if constexpr (Dst == DstStart::StartZero)
    {
        TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, 0);
    }
    else
    {
        TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, DEST_REGISTER_HALF_SIZE);
    }
}

inline constexpr bool is_32bit_input(const std::uint32_t src_format, const std::uint32_t dst_format)
{
    const std::uint32_t input_df  = masked_data_format(src_format);
    const std::uint32_t output_df = masked_data_format(dst_format);
    return ((input_df == to_underlying(DataFormat::Int32)) || (input_df == to_underlying(DataFormat::Float32))) &&
           ((output_df == to_underlying(DataFormat::Int32)) || (output_df == to_underlying(DataFormat::Float32)));
}

inline constexpr bool is_high_fidelity(const MathFidelity math_fidelity_desc)
{
    return math_fidelity_desc != MathFidelity::LoFi;
}

} // namespace ckernel::math
""",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_common.h": r"""
using namespace ckernel::math;

template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_hw_configure_(const std::uint32_t srca_data_format, const std::uint32_t srcb_data_format)
{
    // SFPU_Fp32_enabled (written below) is read by SFPLOAD/SFPSTORE (MOD0_FMT_SRCB), so the SFPU must drain too.
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
    // Configure ZEROACC to auto-detect destination bank (non-legacy mode).
    cfg_reg_rmw_tensix<DEST_ACCESS_CFG_zeroacc_absolute_tile_mode_RMW>(0);
    std::uint32_t int8_math_enabled = is_int8_or_int32_format(srca_data_format) || is_int8_or_int32_format(srcb_data_format);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_INT8_math_enabled_RMW>(int8_math_enabled);

    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(is_fp32_dest_acc_en);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW>(is_fp32_dest_acc_en);

    // Establish the no-override baseline for the SrcA/SrcB ALU format-select fields (ADDR32=0),
    // consumed by the FPU (SrcA) and SFPU (SrcB) on this thread: clear SrcA_val/override and
    // SrcB_val/override so the base (or Blackhole per-bank implied) SrcA/SrcB formats are used. The
    // Dstacc override fields in the same word are owned by the pack thread (configure_pack); the two
    // writers touch disjoint bits and rely on per-byte RMWCIB atomicity, so no cross-thread mutex is
    // needed.
    constexpr std::uint32_t src_fmt_override_mask =
        ALU_FORMAT_SPEC_REG_SrcA_val_MASK | ALU_FORMAT_SPEC_REG_SrcA_override_MASK | ALU_FORMAT_SPEC_REG_SrcB_val_MASK | ALU_FORMAT_SPEC_REG_SrcB_override_MASK;
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32, 0, src_fmt_override_mask>(0);

    // Establish the operand-driven baseline for the Src zero-substitution flag. This is a SrcA+SrcB format
    // change, so refresh both format caches before applying (the no-arg configurator and the compute-op
    // asserts read them; refreshing here is what keeps them from ever going stale).
    src_zero_flag_srca_fmt = srca_data_format;
    src_zero_flag_srcb_fmt = srcb_data_format;
    _configure_default_zero_flag_state_();
}

/**
 * @brief Enable or disable the destination read-address remap (stride-of-16) used by untilize mode.
 *
 * Waits for all in-flight DEST accesses and pending packs to finish before toggling the remap_addrs and
 * swizzle_32b config bits, since changing them mid-access would corrupt reads.
 *
 * @param remap_enable: True to enable the stride-of-16 remap (untilize), false to restore linear addressing.
 */
inline void _llk_math_reconfig_remap_(const bool remap_enable)
{
    // Need to wait for all DEST accesses to be finished before changing
    // remap_addrs and swizzle_32b bits
    tensix_sync();
    while (semaphore_read(semaphore::MATH_PACK) > 0)
    {
    } // Wait for previous packs to finish before claiming all dest

    // Untilize mode needs dest read access with a stride of 16
    // Following bits are needed for enabling stride of 16
    cfg_reg_rmw_tensix<DEST_ACCESS_CFG_remap_addrs_RMW>(remap_enable);
    cfg_reg_rmw_tensix<DEST_ACCESS_CFG_swizzle_32b_RMW>(remap_enable);
}

template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_clear_dest_section_()
{
    // Startup or the preceding commit has already drained this thread's DST writes.
    if constexpr (Dst == DstSync::SyncFull)
    {
        TTI_ZEROACC(p_zeroacc::CLR_ALL, is_fp32_dest_acc_en, 0, ADDR_MOD_1, 0);
    }
    else
    {
        static_assert(Dst == DstSync::SyncHalf);
        TT_ZEROACC(p_zeroacc::CLR_HALF, is_fp32_dest_acc_en, 0, ADDR_MOD_1, dest_offset_id % 2);
    }
    // Do not let SFPU writes or an unpack-to-DST ready signal overtake the clear.
    TTI_STALLWAIT(p_stall::STALL_SFPU | p_stall::STALL_SYNC, p_stall::MATH);
}

/**
 * @brief Initialize the math/pack synchronization semaphore and reset the destination section base.
 *
 * Waits for any in-flight packs to finish, then seeds the MATH_PACK semaphore (max count 1 for SyncFull, 2 for
 * SyncHalf double-buffering) and resets the DEST offset and section base.
 *
 * @tparam Dst: Destination sync mode, values = <SyncHalf/SyncFull>
 * @tparam is_fp32_dest_acc_en: Whether FP32 accumulation in the destination register is enabled.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_pack_sync_init_()
{
    tensix_sync();
    while (semaphore_read(semaphore::MATH_PACK) > 0)
    {
    } // Wait for previous packs to finish before claiming all dest
    if constexpr (Dst == DstSync::SyncFull)
    {
        TTI_SEMINIT(1, 0, p_stall::SEMAPHORE_1);
        reset_dest_offset_id();
        set_dest_section_base<StartZero>();
    }
    else
    {
        static_assert(Dst == DstSync::SyncHalf);
        TTI_SEMINIT(2, 0, p_stall::SEMAPHORE_1);
        reset_dest_offset_id();
        set_dest_section_base<StartZero>();
    }
}
""",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h": r"""
using namespace ckernel;

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity>
inline void eltwise_binary_configure_addrmod()
{
    static_assert(
        math_fidelity == MathFidelity::LoFi || eltwise_binary_type == EltwiseBinaryType::ELWMUL,
        "Math fidelity larger than LoFi only works with Eltwise multiply");
    static_assert(
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
            (eltwise_binary_type == EltwiseBinaryType::ELWMUL),
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");

    constexpr std::uint32_t fidelity_increment = is_high_fidelity(math_fidelity) ? 1 : 0;
    constexpr std::uint8_t srcb_incr           = (bcast_type == BroadcastType::NONE || bcast_type == BroadcastType::COL) ? MAX_FPU_ROWS : 0;
    addr_mod_t {
        .srca = {.incr = MAX_FPU_ROWS},
        .srcb = {.incr = srcb_incr},
        .dest = {.incr = MAX_FPU_ROWS},
    }
        .set(ADDR_MOD_0);

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_1);

    addr_mod_t {.srca = {.incr = 0, .clr = 1}, .srcb = {.incr = 0, .clr = 1}, .dest = {.incr = 0, .clr = 0, .cr = 1}, .fidelity = {.incr = fidelity_increment}}
        .set(ADDR_MOD_2);

    addr_mod_t {
        .srca     = {.incr = 0, .clr = 1},
        .srcb     = {.incr = 0, .clr = 1},
        .dest     = {.incr = MAX_FPU_ROWS, .clr = 0, .cr = 0, .c_to_cr = 1},
        .fidelity = {.incr = 0, .clr = 1}}
        .set(ADDR_MOD_3);
}

/**
 * @brief Build the encoded FPU instruction (ELWADD/ELWSUB/ELWMUL) for the given binary op type.
 *
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @param clr_src: Source-clear mode passed to the instruction.
 * @param acc_to_dest: Accumulate result into dest instead of overwriting.
 * @param broadcast_type: Source B broadcast mode (p_elwise::SRCB_* value).
 * @param addr_mod: Address-mod slot the instruction uses.
 * @return The encoded TT_OP instruction word.
 */
template <EltwiseBinaryType eltwise_binary_type>
inline auto eltwise_binary_func(std::uint8_t clr_src, std::uint8_t acc_to_dest, std::uint8_t broadcast_type, std::uint8_t addr_mod)
{
    static_assert(
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
            (eltwise_binary_type == EltwiseBinaryType::ELWMUL),
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");

    if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWADD)
    {
        return TT_OP_ELWADD(clr_src, acc_to_dest, broadcast_type, addr_mod, 0 /*dst*/);
    }
    else if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWSUB)
    {
        return TT_OP_ELWSUB(clr_src, acc_to_dest, broadcast_type, addr_mod, 0 /*dst*/);
    }
    else
    {
        return TT_OP_ELWMUL(clr_src, acc_to_dest, broadcast_type, addr_mod, 0 /*dst*/);
    }
}

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void eltwise_binary_configure_mop_standard(const std::uint32_t acc_to_dest, const ckernel::TensorShape tensor_shape)
{
    static_assert(
        math_fidelity == MathFidelity::LoFi || eltwise_binary_type == EltwiseBinaryType::ELWMUL,
        "Math fidelity larger than LoFi only works with Eltwise multiply");
    LLK_VALIDATE_TENSOR_SHAPE_MATH("eltwise_binary_configure_mop_standard", tensor_shape);
    const std::uint32_t num_faces       = tensor_shape.total_num_faces();
    const std::uint32_t num_faces_c_dim = tensor_shape.num_faces_c_dim;
    constexpr bool high_fidelity        = is_high_fidelity(math_fidelity);
    constexpr std::uint8_t addr_mod     = ADDR_MOD_0;

    // Inner loop: number of MAX_FPU_ROWS (8-row) operations per face
    // Even if face_r_dim < 16, we still process at least 1 inner loop iteration
    const std::uint8_t innerloop = tensor_shape.face_r_dim > MAX_FPU_ROWS ? (tensor_shape.face_r_dim >> MAX_FPU_ROWS_LOG2) : 1;

    // Outer loop depends on broadcast type:
    // - COL broadcast: MOP processes num_faces_c_dim faces (one row of faces)
    //                  Runtime calls MOP num_faces_r_dim times (one call per row)
    // - Other broadcasts: MOP processes all num_faces in one call
    const std::uint32_t outerloop = (bcast_type == BroadcastType::COL) ? num_faces_c_dim : num_faces;

    constexpr auto broadcast_type = (bcast_type == BroadcastType::COL)      ? p_elwise::SRCB_BCAST_COL
                                    : (bcast_type == BroadcastType::ROW)    ? p_elwise::SRCB_BCAST_ROW
                                    : (bcast_type == BroadcastType::SCALAR) ? p_elwise::SRCB_BCAST_ALL
                                                                            : p_elwise::SRCB_NO_BCAST;

    // Scalar and Col broadcast should not Clear B within a MOP - B is cleared outside of MOP
    constexpr auto CLR_SRC = (bcast_type == BroadcastType::COL || bcast_type == BroadcastType::SCALAR) ? p_setrwc::CLR_A : p_setrwc::CLR_AB;

    if constexpr ((eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB))
    {
        ckernel_template tmp(outerloop, innerloop, eltwise_binary_func<eltwise_binary_type>(0 /*clr_src*/, acc_to_dest, broadcast_type, addr_mod));
        if (tensor_shape.face_r_dim <= MAX_FPU_ROWS)
        {
            // For partial faces (face_r_dim < 16), we still need to increment counters by MAX_FPU_ROWS
            // to maintain proper 16-row spacing between faces
            tmp.set_loop_op1(TT_OP_INCRWC(0, MAX_FPU_ROWS, MAX_FPU_ROWS, MAX_FPU_ROWS));
        }
        tmp.set_end_op(TT_OP_SETRWC(CLR_SRC, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
        tmp.program();
    }
    else if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
    {
        if constexpr (high_fidelity)
        {
            ckernel_template tmp(
                to_underlying(math_fidelity),
                innerloop,
                eltwise_binary_func<EltwiseBinaryType::ELWMUL>(0 /*clr_src*/, 0 /*acc_to_dest*/, broadcast_type, addr_mod));
            tmp.set_last_inner_loop_instr(eltwise_binary_func<EltwiseBinaryType::ELWMUL>(0 /*clr_src*/, 0 /*acc_to_dest*/, broadcast_type, ADDR_MOD_2));
            tmp.set_last_outer_loop_instr(eltwise_binary_func<EltwiseBinaryType::ELWMUL>(CLR_SRC, 0 /*acc_to_dest*/, broadcast_type, ADDR_MOD_3));
            // HiFi partial face advancement is handled by runtime INCRWC between face MOP runs,
            // NOT by end_op (which would incorrectly advance dest between fidelity phases)
            tmp.program();
        }
        else if (tensor_shape.face_r_dim <= MAX_FPU_ROWS)
        {
            // Partial faces: INCRWC as loop_op1 via two-arg constructor to maintain 16-row face spacing.
            // Must use two-arg constructor so m_loop0/1_last_instr = INCRWC, preventing the
            // last-iteration override from replacing INCRWC with a second ELWMUL instruction.
            ckernel_template tmp(
                outerloop,
                innerloop,
                eltwise_binary_func<EltwiseBinaryType::ELWMUL>(0 /*clr_src*/, 0 /*acc_to_dest*/, broadcast_type, addr_mod),
                TT_OP_INCRWC(0, MAX_FPU_ROWS, MAX_FPU_ROWS, MAX_FPU_ROWS));
            tmp.set_end_op(TT_OP_SETRWC(CLR_SRC, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program();
        }
        else
        {
            ckernel_template tmp(
                outerloop, innerloop, eltwise_binary_func<EltwiseBinaryType::ELWMUL>(0 /*clr_src*/, 0 /*acc_to_dest*/, broadcast_type, addr_mod));
            tmp.set_end_op(TT_OP_SETRWC(CLR_SRC, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program();
        }
    }
}

/**
 * @brief Initialize FPU for standard elementwise binary operations (no dest reuse).
 *
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param tensor_shape: Tensor shape describing tile dimensions
 * @param acc_to_dest: Accumulate result to destination register instead of overwriting
 * @note @ref _llk_math_eltwise_binary_standard_ runs the configured op with matching template args.
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType src_b_bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_binary_standard_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    LLK_VALIDATE_TENSOR_SHAPE_MATH("_llk_math_eltwise_binary_standard_init_", tensor_shape);

    eltwise_binary_configure_addrmod<eltwise_binary_type, src_b_bcast_type, math_fidelity>();
    eltwise_binary_configure_mop_standard<eltwise_binary_type, src_b_bcast_type, math_fidelity>(acc_to_dest, tensor_shape);

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_binary_standard_(const ckernel::TensorShape tensor_shape, std::uint32_t dst_index)
{
    static_assert(
        math_fidelity == MathFidelity::LoFi || eltwise_binary_type == EltwiseBinaryType::ELWMUL,
        "Math fidelity larger than LoFi only works with Eltwise multiply");
    static_assert(
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
            (eltwise_binary_type == EltwiseBinaryType::ELWMUL),
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");
    LLK_VALIDATE_TENSOR_SHAPE_MATH("_llk_math_eltwise_binary_standard_", tensor_shape);
    const std::uint32_t num_faces_r_dim = tensor_shape.num_faces_r_dim;
    constexpr bool high_fidelity        = is_high_fidelity(math_fidelity);

    // Dest counter always jumps by 32x32 tile spacing regardless of actual tile size
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    if constexpr ((eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB))
    {
        if constexpr (src_b_bcast_type == BroadcastType::COL)
        {
            // COL broadcast: MOP processes num_faces_c_dim faces (one row of faces)
            // Runtime calls MOP num_faces_r_dim times (once per row of faces)
            // After each row, CLR_B to allow next B column to be loaded
#pragma GCC unroll 0
            for (std::uint32_t face_row = 0; face_row < num_faces_r_dim; face_row++)
            {
                ckernel_template::run();
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
            }
        }
        else
        {
            // NONE/ROW/SCALAR: MOP handles all faces in one call
            ckernel_template::run();
            if constexpr (src_b_bcast_type == BroadcastType::SCALAR)
            {
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_D);
            }
        }
    }
    else if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
    {
        if constexpr (src_b_bcast_type == BroadcastType::COL)
        {
            // COL broadcast: MOP processes fidelity phases for one face (HiFi) or all face columns (LoFi)
            // With high fidelity, call MOP once per face column per face row
            const std::uint32_t num_faces_c_dim = tensor_shape.num_faces_c_dim;
            const std::uint32_t fidelity_loop   = high_fidelity ? num_faces_c_dim : 1;
#pragma GCC unroll 0
            for (std::uint32_t face_row = 0; face_row < num_faces_r_dim; face_row++)
            {
#pragma GCC unroll 0
                for (std::uint32_t i = 0; i < fidelity_loop; i++)
                {
                    ckernel_template::run();
                    if constexpr (high_fidelity)
                    {
                        if (tensor_shape.face_r_dim <= MAX_FPU_ROWS)
                        {
                            TTI_INCRWC(p_setrwc::CR_D, MAX_FPU_ROWS, 0, 0);
                        }
                    }
                    // LoFi: MOP handles face spacing internally via loop_op1, no runtime INCRWC needed
                }
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
            }
        }
        else
        {
            // NONE/ROW/SCALAR: MOP handles all faces, fidelity requires multiple runs
            const std::uint32_t num_faces     = tensor_shape.total_num_faces();
            const std::uint32_t fidelity_loop = high_fidelity ? num_faces : 1;
#pragma GCC unroll 0
            for (std::uint32_t i = 0; i < fidelity_loop; i++)
            {
                ckernel_template::run();
                if constexpr (high_fidelity)
                {
                    if (tensor_shape.face_r_dim <= MAX_FPU_ROWS)
                    {
                        TTI_INCRWC(p_setrwc::CR_D, MAX_FPU_ROWS, 0, 0);
                    }
                }
                // LoFi: MOP handles face spacing internally via loop_op1, no runtime INCRWC needed
            }
            if constexpr (src_b_bcast_type == BroadcastType::SCALAR)
            {
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_D);
            }
        }
    }
    math::clear_dst_reg_addr();
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void _llk_math_eltwise_binary_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::NONE)
    {
        _llk_math_eltwise_binary_standard_init_<eltwise_binary_type, src_b_bcast_type, math_fidelity>(tensor_shape, acc_to_dest);
    }
    else
    {
        _llk_math_eltwise_binary_with_dest_reuse_init_<eltwise_binary_type, src_b_bcast_type, math_fidelity, binary_reuse_dest>(tensor_shape, acc_to_dest);
    }

    // ELWADD/ELWMUL/ELWSUB read the Src zero-substitution flag but
    // eltwise-binary init never sets it, so re-establish the operand-driven DEFAULT here — otherwise a preceding
    // copy_init/datacopy op that left PRESERVE leaks into the MOP (denormal Src results differ). Also
    // covers bcast add/sub/mul, which route through this init. Mirrors reduce/transpose/datacopy.
    math::_configure_default_zero_flag_state_();
}
""",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_math_reduce_custom.h": r"""
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_mop_config_(const ckernel::TensorShape tensor_shape)
{
    if (tensor_shape.num_faces_r_dim == 1)
    {
        if constexpr (is_fp32_dest_acc_en)
        {
            lltt::record(0, 15);

            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);
            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);

            // Move high 16 bits from DEST row 0 to SrcB rows 16 - 31 and transpose
            TTI_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_6, p_movd2b::MOV_1_ROW, 0);
            TTI_TRNSPSRCB;
            // Move high 16 bits back to Dest
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_6, p_movb2d::MOV_4_ROWS, 0);
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_6, p_movb2d::MOV_4_ROWS, 4);
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_6, p_movb2d::MOV_4_ROWS, 8);
            TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_6, p_movb2d::MOV_4_ROWS, 12);
            // Move low 16 bits to SrcB rows 16 - 31 and transpose
            TTI_MOVD2B(p_mov::DEST_32B_LOW, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_6, p_movd2b::MOV_1_ROW, 0);
            TTI_TRNSPSRCB;
            // Move low 16 bits from SrcB rows 16 - 31 to DEST rows 0, 4, 8, 12.
            // ADDR_MOD_2 increments CR_D and Dest counter val by 4. No F2 for a single face-row, so
            // the last write uses ADDR_MOD_2 (no +20 jump); CLR_B resets the counters afterwards.
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
            TTI_MOVB2D(p_mov::DEST_32B_LOW, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
            // Clear B valid bits at the end and all address counters
            TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
        }
    }
}
""",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_math_eltwise_binary_custom.h": r"""
template <EltwiseBinaryType eltwise_binary_type>
inline void _llk_math_bcast_cols_reuse_custom_(
    const std::uint32_t ct_dim = 1, const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE, const std::uint32_t dst_index = 0)
{
    static_assert(
        eltwise_binary_type == EltwiseBinaryType::ELWMUL || eltwise_binary_type == EltwiseBinaryType::ELWSUB,
        "blocked bcast-col reuse scaffold supports ELWMUL and ELWSUB only");
    LLK_ASSERT(
        math::src_zero_flag_hw == (requires_disabled_src_zero_flag(math::src_zero_flag_srca_fmt, math::src_zero_flag_srcb_fmt) ? 1u : 0u),
        "bcast-col custom: Src zero-substitution flag does not hold the operand-driven value; an op between the init and "
        "this call left it at keep, so denormal Src operands would not be flushed");

}
""",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h": r"""
using namespace ckernel;

template <MathFidelity math_fidelity>
inline void matmul_configure_mop(
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false)
{
    // in0 - loaded to SrcB
    // in1 - loaded to SrcA
    // Unpacker will always load faces in f0,f1,f2,f3 order
    // if in1 is transposed then faces 1&2 need to be swapped during read
    // by changing address increment amount via addr_mods
    // Col major layout in dest only impacts destination address increment
    // if col major layout faces are ordered as f0,f2,f1,f3
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);

    const bool reuse_a        = ct_dim >= rt_dim;
    const std::uint32_t t_dim = reuse_a ? rt_dim : ct_dim;

    const bool is_in0_16x32 = (in0_tile_r_dim <= FACE_R_DIM) && (in0_tile_c_dim > FACE_C_DIM);
    const bool is_in1_32x16 = (in1_tile_r_dim > FACE_R_DIM) && (in1_tile_c_dim <= FACE_C_DIM);
    const bool is_in0_32x16 = (in0_tile_r_dim > FACE_R_DIM) && (in0_tile_c_dim <= FACE_C_DIM);
    const bool is_in1_16x32 = (in1_tile_r_dim <= FACE_R_DIM) && (in1_tile_c_dim > FACE_C_DIM);

    const std::uint32_t replay_buf_len =
        (is_in0_16x32 && is_in1_32x16) ? 4 : ((is_in0_16x32 || is_in1_32x16 || is_in0_32x16 || is_in1_16x32) ? (partial_face ? 4 : 8) : 16);

    load_replay_buf(
        ckernel::math::replay_buf_offset,
        replay_buf_len,
        // Lambda function to load reply buffer
        [high_fidelity, reuse_a, partial_face, is_in1_32x16, is_in0_16x32, is_in0_32x16, is_in1_16x32, t_dim]
        {
            if (is_in1_32x16)
            {
                if (is_in0_16x32)
                {
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A0 // srca=srca, srcb+=8,  dest+=8
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B0A0 // srca+=16,  srcb+=8,  dest=0
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A1 // srca=srca, srcb+=8,  dest=+8, bias=1
                }
                else
                {
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A0 // srca=srca, srcb+=8,  dest+=8
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B0A0 // srca+=16,  srcb+=8,  dest=0
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A1 // srca=srca, srcb+=8,  dest=+8
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // B1A1 // srca=0,    srcb+=8,  dest=16 (addr_mod_4), bias=0

                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A0 // srca=srca, srcb+=8,  dest+=8
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B2A0 // srca+=16,  srcb+=8,  dest=16
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B3A1 // srca=srca, srcb+=8,  dest+=8
                }
            }
            else if (is_in0_16x32 || is_in0_32x16)
            {
                if (partial_face)
                {
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B0A0 // srca+=16,  srcb=0,   dest=+16
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // B0A1 // srca+=16,  srcb+=16,  dest=0 (addr_mod_4), bias=0
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B1A2 // srca+=16,  srcb=0,  dest=+16
                }
                else
                {
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A0 // srca=srca, srcb+=8,  dest+=8
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B0A0 // srca+=16,  srcb=0,   dest+=8
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A1 // srca=srca, srcb+=8,  dest+=8
                    TTI_MVMUL(
                        p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // B0A1 // srca+=16/=0,  srcb=16,  dest=0/+=8 (addr_mod_4), bias=0 // srca=0 dest+=8 if in0_32x16

                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A2 // srca=srca, srcb+=8,  dest+=8
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B1A2 // srca+=16,  srcb=16,  dest+=8/24 // dest+=24 if transposed
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A3 // srca=srca, srcb+=8,  dest+=8
                }
            }
            else
            {
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A0 // srca=srca, srcb+=8,  dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B0A0 // srca+=16/32, srcb=0, dest+=8  // srca+=32 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A1 // srca=srca, srcb+=8,  dest+=8  // A1 -> A2 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B0A1 // srca=0,    srcb=32,  dest+=8  // A1 -> A2 if transposed

                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A0 // srca=srca, srcb+=8,  dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B2A0 // srca+=16/32, srcb=0, dest+=8 // srca+=32 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A1 // srca=srca, srcb+=8,  dest+=8 // A1 -> A2 if transposed
                if (!is_in1_16x32)
                {
                    // TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A1 // srca=32/16,srcb=16,  dest=0 (addr_mod_4) // A1 -> A2 && srca=16 if transposed
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // B2A1 // srca=32/16,srcb=16,  dest=0 (addr_mod_4) // A1 -> A2 && srca=16 if transposed

                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A2 // srca=srca, srcb+=8,  dest+=8 // A2 -> A1 if transposed
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B1A2 // srca+=16,  srcb=16,  dest+=8 // A2 -> A1 if transposed
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A3 // srca=srca, srcb+=8,  dest+=8
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B1A3 // srca=32,   srcb=48,  dest+=8

                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B3A2 // srca=srca, srcb+=8,  dest+=8 // A2 -> A1 if transposed
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B3A2 // srca+=16,  srcb=0,   dest+=8 // A2 -> A1 if transposed
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B3A3 // srca=srca, srcb+=8,  dest+=8
                }
            }

            if constexpr (high_fidelity)
            {
                // TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B3A3 or B3A2 // reset srca/srcb/dest, increment phase (addr_mod_5)
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0); // B3A3 or B3A2 // reset srca/srcb/dest, increment phase (addr_mod_5)
            }
            else
            {
                if (reuse_a)
                {
                    TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_5, 0); // B3A3 or B3A2 // reset srca/srcb/dest, increment phase (addr_mod_5), clear src A
                }
                else
                {
                    TTI_MVMUL(p_setrwc::CLR_B, 0, ADDR_MOD_5, 0); // B3A3 or B2A1 // reset srca/srcb/dest, increment phase (addr_mod_5), clear src A
                }
            }
        });

    // TODO: can we commonize this?
    constexpr std::uint32_t inner_loops = high_fidelity ? to_underlying(math_fidelity) : 1;
    ckernel_template tmp(1 /* outer loop */, inner_loops, lltt::replay_insn(ckernel::math::replay_buf_offset, replay_buf_len));

    if constexpr (high_fidelity)
    {
        if (reuse_a)
        {
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F));
        }
        else
        {
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F));
        }
    }
    tmp.program();
}
""",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_common_api.h": r"""
template <EltwiseBinaryType eltwise_binary_type, MathFidelity math_fidelity>
inline constexpr MathFidelity get_effective_math_fidelity() {
    return (eltwise_binary_type == EltwiseBinaryType::ELWMUL) ? math_fidelity : MathFidelity::LoFi;
}
""",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_binary_api.h": r"""
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary_init_impl(
    const ckernel::TensorShape& tensor_shape, const std::uint32_t acc_to_dest) {
    constexpr auto effective_math_fidelity = get_effective_math_fidelity<eltwise_binary_type, math_fidelity>();

    SAN_HOOK(init<OperationFpuEltwiseBinary>(
        StateVal<OperationFpuEltwiseBinary::EltwiseBinaryType>(to_underlying(eltwise_binary_type)),
        StateVal<OperationFpuEltwiseBinary::BroadcastType>(to_underlying(src_b_bcast_type)),
        StateVal<OperationFpuEltwiseBinary::MathFidelity>(to_underlying(effective_math_fidelity)),
        StateVal<OperationFpuEltwiseBinary::ReuseDest>(to_underlying(binary_reuse_dest)),
        StateVal<OperationFpuEltwiseBinary::AccToDest>(acc_to_dest),
        StateVal<OperationFpuEltwiseBinary::FaceHeight>(tensor_shape.face_r_dim),
        StateVal<OperationFpuEltwiseBinary::NumFaces>(tensor_shape.total_num_faces()),
        StateVal<OperationFpuEltwiseBinary::NumFacesCDim>(tensor_shape.num_faces_c_dim)));

    _llk_math_eltwise_binary_init_<eltwise_binary_type, src_b_bcast_type, effective_math_fidelity, binary_reuse_dest>(
        tensor_shape, acc_to_dest);
}
""",
}


def line_of(path: str, needle: str, nth: int = 1) -> int:
    """1-based line of the ``nth`` occurrence of ``needle`` in the snippet at ``path``."""
    seen = 0
    for number, text in enumerate(SNIPPETS[path].split("\n"), 1):
        if needle in text:
            seen += 1
            if seen == nth:
                return number
    raise AssertionError(f"{needle!r} #{nth} not in {path}")


def ev(path: str, needle: str, nth: int = 1) -> str:
    return f"{path}:{line_of(path, needle, nth)}"


def signature(name: str, path: str, line: int, params: str) -> str:
    digest = hashlib.sha256(f"{path}:{line}".encode("utf-8")).hexdigest()[:8]
    return f"mop::{name}#0x{digest}({params})"


class MathFamilyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        real = {
            path: REPO_ROOT / path
            for path in (resources.CFG_DEFINES, resources.GPR_MAP)
        }
        if not all(path.is_file() for path in real.values()):
            raise unittest.SkipTest(
                "needs cfg_defines.h and ckernel_gpr_map.h of the checkout"
            )
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        files = dict(SNIPPETS)
        for relative, source in real.items():
            files[relative] = source.read_text(encoding="utf-8")
        for relative, text in files.items():
            target = cls.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        cls.db = effects_db.build(cls.root, families=["math"])

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    # ---- helpers ---------------------------------------------------------- #
    def fn(self, name: str):
        (model,) = self.db.by_name(name)
        return model

    @staticmethod
    def rows(effects, *patterns):
        return sorted(
            (e.resource, e.access, e.value, e.condition, e.evidence, e.detail)
            for e in effects
            if not patterns or e.pattern in patterns
        )

    # ---- ADDR_MOD ---------------------------------------------------------- #
    def test_addr_mod_set_values_are_canonical(self) -> None:
        model = self.fn("eltwise_binary_configure_addrmod")
        self.assertEqual(
            self.rows(model.effects, "math.addr_mod_set"),
            [
                (
                    "ADDR_MOD.0",
                    "write",
                    "{.srca.incr=MAX_FPU_ROWS, .srcb.incr=srcb_incr, .dest.incr=MAX_FPU_ROWS}",
                    "",
                    ev(BINARY, ".set(ADDR_MOD_0);"),
                    "",
                ),
                ("ADDR_MOD.1", "write", "{}", "", ev(BINARY, ".set(ADDR_MOD_1);"), ""),
                (
                    "ADDR_MOD.2",
                    "write",
                    "{.srca.clr=1, .srcb.clr=1, .dest.cr=1, .fidelity.incr=fidelity_increment}",
                    "",
                    ev(BINARY, ".set(ADDR_MOD_2);"),
                    "",
                ),
                (
                    "ADDR_MOD.3",
                    "write",
                    "{.srca.clr=1, .srcb.clr=1, .dest.incr=MAX_FPU_ROWS, .dest.c_to_cr=1, .fidelity.clr=1}",
                    "",
                    ev(BINARY, ".set(ADDR_MOD_3);"),
                    "",
                ),
            ],
        )
        # addr_mod_t::set is a stop primitive: its three SETC16 never surface.
        transitive = self.db.effects(model.key)
        self.assertEqual(
            sorted(e.resource for e in transitive),
            ["ADDR_MOD.0", "ADDR_MOD.1", "ADDR_MOD.2", "ADDR_MOD.3"],
        )
        kinds = {e.resource: (e.value_kind, e.params) for e in transitive}
        self.assertEqual(kinds["ADDR_MOD.1"], ("const", ()))
        self.assertEqual(kinds["ADDR_MOD.3"], ("const", ()))
        self.assertEqual(kinds["ADDR_MOD.0"], ("expr", ("bcast_type",)))
        self.assertEqual(kinds["ADDR_MOD.2"], ("expr", ("math_fidelity",)))

    def test_addr_mod_value_depends_on_template_arguments(self) -> None:
        def slot(bcast: str, fidelity: str, resource: str) -> str:
            (effect,) = [
                e
                for e in self.db.call_effects(
                    "eltwise_binary_configure_addrmod",
                    template_args=("EltwiseBinaryType::ELWMUL", bcast, fidelity),
                    evidence="kernel.cpp:1",
                )
                if e.resource == resource
            ]
            return effect.value

        self.assertEqual(
            slot("BroadcastType::SCALAR", "MathFidelity::LoFi", "ADDR_MOD.0"),
            "{.srca.incr=MAX_FPU_ROWS, .srcb.incr=((BroadcastType::SCALAR == BroadcastType::NONE"
            " || BroadcastType::SCALAR == BroadcastType::COL) ? MAX_FPU_ROWS : 0), .dest.incr=MAX_FPU_ROWS}",
        )
        self.assertNotEqual(
            slot("BroadcastType::NONE", "MathFidelity::HiFi4", "ADDR_MOD.2"),
            slot("BroadcastType::NONE", "MathFidelity::LoFi", "ADDR_MOD.2"),
        )
        self.assertEqual(
            slot("BroadcastType::NONE", "MathFidelity::LoFi", "ADDR_MOD.3"),
            slot("BroadcastType::ROW", "MathFidelity::HiFi4", "ADDR_MOD.3"),
        )

    def test_canonical_addr_mod(self) -> None:
        canonical = math_family.canonical_addr_mod
        self.assertEqual(
            canonical(
                "addr_mod_t {.srca = {.incr = 0, .clr = 0}, .dest = {.incr = 8,},}"
            ),
            "{.dest.incr=8}",
        )
        self.assertEqual(
            canonical("addr_mod_t{.dest = {.incr = 8}}"),
            canonical(
                "addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 8, .clr = 0}}"
            ),
        )
        self.assertEqual(canonical("ckernel::addr_mod_t {}"), "{}")
        self.assertIsNone(canonical("addr_mod_t {{0}, {0}}"))
        self.assertIsNone(canonical("am"))

    def test_addr_mod_operands_and_implicit_fpu_reads(self) -> None:
        model = self.fn("move_d2a_fixed_face")
        reads = self.rows(model.effects, "math.addr_mod_use")
        self.assertEqual(
            reads,
            [
                (
                    "ADDR_MOD.?addrmod",
                    "read",
                    "TTI_MOVD2A",
                    "",
                    ev(CMATH, "TTI_MOVD2A", n),
                    "",
                )
                for n in (1, 2, 3, 4)
            ],
        )
        self.assertTrue(
            all(
                not e.resolved
                for e in model.effects
                if e.pattern == "math.addr_mod_use"
            )
        )
        implicit = sorted(
            (e.resource, e.evidence, e.detail, e.confidence)
            for e in model.effects
            if e.pattern == "math.fpu_implicit"
        )
        self.assertEqual(
            implicit,
            [
                (
                    "CFG.ALU_ACC_CTRL_Fp32_enabled",
                    ev(CMATH, "TTI_MOVD2A"),
                    "implicit",
                    "medium",
                ),
                (
                    "THDCFG.DEST_TARGET_REG_CFG_MATH_Offset",
                    ev(CMATH, "TTI_MOVD2A"),
                    "implicit",
                    "medium",
                ),
            ],
        )
        resolved = {
            e.resource
            for e in self.db.call_effects(
                "move_d2a_fixed_face", args=("ADDR_MOD_1",), evidence="k.cpp:3"
            )
            if e.resource.startswith("ADDR_MOD")
        }
        self.assertEqual(resolved, {"ADDR_MOD.1"})

    # ---- RWC --------------------------------------------------------------- #
    def test_setrwc_writes_the_masked_counters(self) -> None:
        (effect,) = self.fn("reset_counters").effects
        self.assertEqual(
            (
                effect.resource,
                effect.access,
                effect.value,
                effect.resolved,
                effect.params,
            ),
            ("RWC.?setrwc", "write", ".cr=0;.d=0;.b=0;.a=0", False, ("setrwc",)),
        )
        init = self.fn("_llk_math_eltwise_binary_standard_init_")
        counters = sorted(
            e.resource
            for e in self.db.effects(init.key)
            if e.pattern == "math.setrwc" and e.evidence == ev(CMATH, "setrwc);")
        )
        self.assertEqual(counters, ["RWC.DEST", "RWC.FIDELITY", "RWC.SRCA", "RWC.SRCB"])
        # TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0) only clears SrcB valid: no counter is written.
        execute = self.fn("_llk_math_eltwise_binary_standard_")
        cleared = ev(BINARY, "TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);")
        self.assertFalse([e for e in execute.effects if e.evidence == cleared])
        self.assertEqual(
            self.rows(
                [
                    e
                    for e in execute.effects
                    if e.evidence
                    == ev(
                        BINARY,
                        "TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_D);",
                    )
                ],
                "math.setrwc",
            ),
            [
                (
                    "RWC.DEST",
                    "write",
                    ".cr=0;.d=0;.b=0;.a=0",
                    # the ``||`` condition is parenthesized when it is joined under ``&&``
                    "((eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB))"
                    " && !(src_b_bcast_type == BroadcastType::COL) && src_b_bcast_type == BroadcastType::SCALAR",
                    ev(
                        BINARY,
                        "TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_D);",
                    ),
                    "",
                )
            ],
        )

    def test_setrwc_branch_folds_under_template_argument(self) -> None:
        model = self.fn("clear_bank_valid")
        self.assertEqual(
            self.rows(model.effects, "math.setrwc"),
            [
                (
                    "RWC.SRCA",
                    "write",
                    ".cr=0;.d=0;.b=0;.a=0",
                    "SrcReg == Srcs::SrcA",
                    ev(CMATH, "p_setrwc::SET_A);"),
                    "",
                ),
                (
                    "RWC.SRCB",
                    "write",
                    ".cr=0;.d=0;.b=0;.a=0",
                    "!(SrcReg == Srcs::SrcA)",
                    ev(CMATH, "p_setrwc::SET_B);"),
                    "",
                ),
            ],
        )
        folded = self.db.call_effects(
            "clear_bank_valid", template_args=("Srcs::SrcB",), evidence="k.cpp:7"
        )
        self.assertEqual(
            [(e.resource, e.condition, e.evidence) for e in folded],
            [("RWC.SRCB", "", ev(CMATH, "p_setrwc::SET_B);"))],
        )

    def test_incrwc_writes_only_nonzero_increments(self) -> None:
        self.assertEqual(
            self.rows(self.fn("incr_counters").effects, "math.incrwc"),
            [
                (
                    "RWC.DEST",
                    "write",
                    "+incr_d;.cr=incr_cr",
                    "",
                    ev(CMATH, "TT_INCRWC"),
                    "",
                ),
                (
                    "RWC.SRCA",
                    "write",
                    "+incr_a;.cr=incr_cr",
                    "",
                    ev(CMATH, "TT_INCRWC"),
                    "",
                ),
                (
                    "RWC.SRCB",
                    "write",
                    "+incr_b;.cr=incr_cr",
                    "",
                    ev(CMATH, "TT_INCRWC"),
                    "",
                ),
            ],
        )
        execute = self.fn("_llk_math_eltwise_binary_standard_")
        incr = [e for e in execute.effects if e.pattern == "math.incrwc"]
        self.assertEqual(
            sorted({(e.resource, e.value, e.detail) for e in incr}),
            [("RWC.DEST", "+MAX_FPU_ROWS;.cr=p_setrwc::CR_D", "")],
        )
        self.assertTrue(
            all("tensor_shape.face_r_dim <= MAX_FPU_ROWS" in e.condition for e in incr)
        )
        mop = self.fn("eltwise_binary_configure_mop_standard")
        words = {
            (e.resource, e.value, e.detail)
            for e in mop.effects
            if e.pattern == "math.incrwc"
        }
        self.assertEqual(
            words,
            {
                ("RWC.DEST", "+MAX_FPU_ROWS", "deferred"),
                ("RWC.SRCA", "+MAX_FPU_ROWS", "deferred"),
                ("RWC.SRCB", "+MAX_FPU_ROWS", "deferred"),
            },
        )

    # ---- MOP --------------------------------------------------------------- #
    def test_mop_program_value_is_a_signature_of_the_flowing_params(self) -> None:
        model = self.fn("eltwise_binary_configure_mop_standard")
        name = "eltwise_binary_configure_mop_standard"
        programs = [line_of(BINARY, "tmp.program();", n) for n in (1, 2, 3, 4)]
        self.assertEqual(
            [
                (e.value, e.detail, e.evidence)
                for e in sorted(
                    (e for e in model.effects if e.pattern == "math.mop_program"),
                    key=lambda e: e.sort_key(),
                )
            ],
            [
                (
                    signature(
                        name,
                        BINARY,
                        programs[0],
                        "eltwise_binary_type, bcast_type, acc_to_dest, tensor_shape",
                    ),
                    "MOP:*,INCRWC,SETRWC",
                    f"{BINARY}:{programs[0]}",
                ),
                (
                    signature(
                        name,
                        BINARY,
                        programs[1],
                        "bcast_type, math_fidelity, tensor_shape",
                    ),
                    "MOP:*,*,*",
                    f"{BINARY}:{programs[1]}",
                ),
                (
                    signature(name, BINARY, programs[2], "bcast_type, tensor_shape"),
                    "MOP:*,INCRWC,SETRWC",
                    f"{BINARY}:{programs[2]}",
                ),
                (
                    signature(name, BINARY, programs[3], "bcast_type, tensor_shape"),
                    "MOP:*,SETRWC",
                    f"{BINARY}:{programs[3]}",
                ),
            ],
        )
        # ckernel_template::program is a stop primitive: its mop_cfg[] stores never surface.
        writes = {e.pattern for e in self.db.effects(model.key) if e.resource == "MOP"}
        self.assertEqual(writes, {"math.mop_program"})

    def test_mop_value_distinguishes_acc_to_dest_through_the_llk_api(self) -> None:
        def program(acc_to_dest: str):
            (effect,) = [
                e
                for e in self.db.call_effects(
                    "llk_math_eltwise_binary_init_impl",
                    template_args=(
                        "EltwiseBinaryType::ELWADD",
                        "BroadcastType::NONE",
                        "MathFidelity::HiFi4",
                    ),
                    args=("ckernel::DEFAULT_TENSOR_SHAPE", acc_to_dest),
                    evidence="kernel.cpp:9",
                )
                if e.resource == "MOP" and e.access == "write" and not e.condition
            ]
            # The ELWMUL branches of the same template also surface, still conditional: the
            # graph caps conditions at 300 chars per level, so their false
            # "ELWADD == ELWMUL" conjunct no longer folds (out/requests/math.md).
            return effect

        line = line_of(BINARY, "tmp.program();", 1)
        zero, one = program("0"), program("1")
        self.assertEqual(
            (zero.value, zero.value_kind, zero.condition),
            (
                signature(
                    "eltwise_binary_configure_mop_standard",
                    BINARY,
                    line,
                    "EltwiseBinaryType::ELWADD, BroadcastType::NONE, 0, ckernel::DEFAULT_TENSOR_SHAPE",
                ),
                "const",
                "",
            ),
        )
        self.assertNotEqual(zero.value, one.value)
        self.assertEqual(program("1").value, one.value)
        self.assertEqual(
            [step[0].split("<")[0] for step in one.via],
            [
                "llk_math_eltwise_binary_init_impl",
                "_llk_math_eltwise_binary_init_",
                "_llk_math_eltwise_binary_standard_init_",
                "eltwise_binary_configure_mop_standard",
            ],
        )

    def test_mop_run_reads_mop_and_implicit_alu_state(self) -> None:
        model = self.fn("_llk_math_eltwise_binary_standard_")
        reads = {
            (e.resource, e.pattern)
            for e in self.db.effects(model.key)
            if e.resource == "MOP"
        }
        self.assertEqual(reads, {("MOP", "core.mop_run.insn")})
        implicit = {
            (e.resource, e.detail, e.confidence)
            for e in model.effects
            if e.pattern == "math.fpu_implicit"
        }
        self.assertEqual(
            implicit,
            {
                (resource, "implicit:mop", "low")
                for resource in (
                    "CFG.ALU_ACC_CTRL_Fp32_enabled",
                    "CFG.ALU_ACC_CTRL_INT8_math_enabled",
                    "CFG.ALU_ACC_CTRL_Zero_Flag_disabled_src",
                    "CFG.ALU_FORMAT_SPEC_REG0_SrcA",
                    "THDCFG.DEST_TARGET_REG_CFG_MATH_Offset",
                )
            },
        )

    # ---- replay ------------------------------------------------------------ #
    def test_record_window_marks_exactly_len_instructions_deferred(self) -> None:
        model = self.fn("_llk_math_reduce_block_max_row_mop_config_")
        self.assertEqual(
            self.rows(model.effects, "math.replay_record"),
            [
                (
                    "REPLAY",
                    "write",
                    "noexec",
                    "tensor_shape.num_faces_r_dim == 1 && is_fp32_dest_acc_en",
                    ev(REDUCE, "lltt::record(0, 15);"),
                    "start=0;len=15",
                )
            ],
        )
        addr = [e for e in model.effects if e.pattern == "math.addr_mod_use"]
        self.assertEqual(
            len(addr), 12
        )  # 2 GMPOOL + 2 MOVD2B + 8 MOVB2D (TRNSPSRCB has no ADDR_MOD)
        self.assertEqual({e.detail for e in addr}, {"deferred"})
        self.assertEqual(
            sorted(
                {
                    (e.resource, e.detail)
                    for e in model.effects
                    if e.pattern == "math.setrwc"
                }
            ),
            [
                ("RWC.DEST", "deferred"),
                ("RWC.SRCA", "deferred"),
                ("RWC.SRCB", "deferred"),
            ],
        )
        self.assertFalse([e for e in model.effects if e.pattern == "math.fpu_implicit"])

    def test_load_replay_buf_lambda_is_deferred(self) -> None:
        model = self.fn("matmul_configure_mop")
        addr = [e for e in model.effects if e.pattern == "math.addr_mod_use"]
        self.assertEqual(len(addr), 38)
        self.assertEqual({e.detail for e in addr}, {"deferred"})
        self.assertEqual(
            self.rows(
                model.effects,
                "math.load_replay_buf",
                "math.replay_run",
                "math.mop_program",
            )[:2],
            [
                (
                    "MOP",
                    "write",
                    signature(
                        "matmul_configure_mop",
                        MATMUL,
                        line_of(MATMUL, "tmp.program();"),
                        "math_fidelity, ct_dim, rt_dim, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face",
                    ),
                    "",
                    ev(MATMUL, "tmp.program();"),
                    "MOP:REPLAY,SETRWC,SETRWC",
                ),
                (
                    "REPLAY",
                    "read",
                    "",
                    "",
                    ev(MATMUL, "lltt::replay_insn("),
                    "start=replay_buf_offset;len=replay_buf_len;deferred",
                ),
            ],
        )
        (write,) = [e for e in model.effects if e.pattern == "math.load_replay_buf"]
        self.assertEqual(
            (write.access, write.value, write.evidence, write.detail),
            (
                "write",
                "noexec",
                ev(MATMUL, "load_replay_buf("),
                "start=replay_buf_offset;len=replay_buf_len",
            ),
        )

    def test_dest_write_addr_composes_the_reassigned_local(self) -> None:
        model = self.fn("set_dst_write_addr")
        composed = (
            "(tile_index << DstTileSizeLog2[tile_shape]) + get_dest_buffer_base()"
        )
        self.assertEqual(
            self.rows(model.effects, "math.setc16", "math.mailbox_write"),
            [
                (
                    "MAILBOX.unpack",
                    "write",
                    composed,
                    "unpack_destination == UnpackDestination::DestReg",
                    ev(CMATH, "mailbox_write(ThreadId::UnpackThreadId, dst_index);"),
                    "",
                ),
                (
                    "THDCFG.DEST_TARGET_REG_CFG_MATH_Offset",
                    "write",
                    composed,
                    "!(unpack_destination == UnpackDestination::DestReg)",
                    ev(
                        CMATH,
                        "TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index);",
                    ),
                    "",
                ),
            ],
        )
        execute = self.fn("_llk_math_eltwise_binary_standard_")
        dest = [
            e
            for e in self.db.effects(execute.key)
            if e.resource
            in ("THDCFG.DEST_TARGET_REG_CFG_MATH_Offset", "MAILBOX.unpack")
            and e.access == "write"
        ]
        self.assertEqual(
            [(e.resource, e.value, e.value_kind, e.params, e.condition) for e in dest],
            [
                (
                    "THDCFG.DEST_TARGET_REG_CFG_MATH_Offset",
                    "(dst_index << DstTileSizeLog2[DstTileShape::Tile32x32]) + get_dest_buffer_base()",
                    "expr",
                    ("dst_index",),
                    "",
                )
            ],
        )
        reads = {
            e.resource
            for e in self.db.effects(execute.key)
            if e.pattern == "math.sw_read"
        }
        self.assertEqual(reads, {"SW.dest_offset_id"})

    # ---- software shadows, semaphores, cfg --------------------------------- #
    def test_dest_section_flip_reads_and_writes_dest_offset(self) -> None:
        model = self.fn("dest_section_flip")
        self.assertEqual(
            self.rows(self.db.effects(model.key)),
            [
                (
                    "SW.dest_offset_id",
                    "read",
                    "",
                    "",
                    ev(CKERNEL, "dest_offset_id = 1 - dest_offset_id;"),
                    "",
                ),
                (
                    "SW.dest_offset_id",
                    "read",
                    "",
                    "",
                    ev(CKERNEL, "return (0 != dest_offset_id)"),
                    "",
                ),
                (
                    "SW.dest_offset_id",
                    "write",
                    "1 - dest_offset_id",
                    "",
                    ev(CKERNEL, "dest_offset_id = 1 - dest_offset_id;"),
                    "",
                ),
                (
                    "THDCFG.DEST_TARGET_REG_CFG_MATH_Offset",
                    "write",
                    "get_dest_buffer_base()",
                    "",
                    ev(
                        CMATH,
                        "TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, base_addr);",
                    ),
                    "",
                ),
            ],
        )

    def test_clear_dest_section_branches(self) -> None:
        model = self.fn("_llk_math_clear_dest_section_")
        full = "Dst == DstSync::SyncFull"
        half = "!(Dst == DstSync::SyncFull)"
        self.assertEqual(
            self.rows(
                model.effects, "math.sw_read", "math.addr_mod_use", "math.fpu_implicit"
            ),
            [
                (
                    "ADDR_MOD.1",
                    "read",
                    "TTI_ZEROACC",
                    full,
                    ev(COMMON, "TTI_ZEROACC(p_zeroacc::CLR_ALL"),
                    "",
                ),
                (
                    "ADDR_MOD.1",
                    "read",
                    "TT_ZEROACC",
                    half,
                    ev(COMMON, "TT_ZEROACC(p_zeroacc::CLR_HALF"),
                    "",
                ),
                (
                    "CFG.DEST_ACCESS_CFG_zeroacc_absolute_tile_mode",
                    "read",
                    "TTI_ZEROACC",
                    full,
                    ev(COMMON, "TTI_ZEROACC(p_zeroacc::CLR_ALL"),
                    "implicit",
                ),
                (
                    "CFG.DEST_ACCESS_CFG_zeroacc_absolute_tile_mode",
                    "read",
                    "TT_ZEROACC",
                    half,
                    ev(COMMON, "TT_ZEROACC(p_zeroacc::CLR_HALF"),
                    "implicit",
                ),
                (
                    "SW.dest_offset_id",
                    "read",
                    "",
                    half,
                    ev(COMMON, "TT_ZEROACC(p_zeroacc::CLR_HALF"),
                    "",
                ),
                (
                    "THDCFG.DEST_TARGET_REG_CFG_MATH_Offset",
                    "read",
                    "TTI_ZEROACC",
                    full,
                    ev(COMMON, "TTI_ZEROACC(p_zeroacc::CLR_ALL"),
                    "implicit",
                ),
                (
                    "THDCFG.DEST_TARGET_REG_CFG_MATH_Offset",
                    "read",
                    "TT_ZEROACC",
                    half,
                    ev(COMMON, "TT_ZEROACC(p_zeroacc::CLR_HALF"),
                    "implicit",
                ),
            ],
        )
        half_only = self.db.call_effects(
            "_llk_math_clear_dest_section_",
            template_args=("DstSync::SyncHalf", "false"),
            evidence="k.cpp:5",
        )
        self.assertEqual(
            {e.resource for e in half_only if e.pattern == "math.sw_read"},
            {"SW.dest_offset_id"},
        )
        self.assertTrue(all(e.condition == "" for e in half_only))

    def test_zero_flag_shadow_reads(self) -> None:
        model = self.fn("_configure_src_zero_flag_")
        self.assertEqual(
            self.rows(model.effects, "math.sw_read"),
            [
                (
                    "SW.src_zero_flag_hw",
                    "read",
                    "",
                    "",
                    ev(CMATH, "if (src_zero_flag_hw == value)"),
                    "",
                )
            ],
        )
        writes = self.rows(
            [e for e in self.db.effects(model.key) if e.access == "write"]
        )
        self.assertEqual(
            [(row[0], row[2], row[4]) for row in writes],
            [
                (
                    "CFG.ALU_ACC_CTRL_Zero_Flag_disabled_src",
                    "disable ? 1u : 0u",
                    ev(
                        CMATH,
                        "cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(value);",
                    ),
                ),
                (
                    "SW.src_zero_flag_hw",
                    "disable ? 1u : 0u",
                    ev(CMATH, "src_zero_flag_hw = value;"),
                ),
            ],
        )
        # LLK_ASSERT arguments are unevaluated in production builds: the scanner blanks
        # them, so the shadow reads inside the assert are not effects
        asserted = self.fn("_llk_math_bcast_cols_reuse_custom_")
        self.assertEqual(self.rows(asserted.effects, "math.sw_read"), [])

    def test_pack_sync_init_semaphores_by_sync_mode(self) -> None:
        model = self.fn("_llk_math_pack_sync_init_")
        self.assertEqual(
            self.rows(model.effects, "core.semaphore_read", "core.seminit"),
            [
                (
                    "SEM.MATH_PACK",
                    "read",
                    "",
                    "",
                    ev(COMMON, "while (semaphore_read(semaphore::MATH_PACK) > 0)", 2),
                    "",
                ),
                (
                    "SEM.MATH_PACK",
                    "write",
                    "max=1;init=0",
                    "Dst == DstSync::SyncFull",
                    ev(COMMON, "TTI_SEMINIT(1, 0"),
                    "",
                ),
                (
                    "SEM.MATH_PACK",
                    "write",
                    "max=2;init=0",
                    "!(Dst == DstSync::SyncFull)",
                    ev(COMMON, "TTI_SEMINIT(2, 0"),
                    "",
                ),
            ],
        )
        half = self.db.call_effects(
            "_llk_math_pack_sync_init_",
            template_args=("DstSync::SyncHalf", "false"),
            evidence="k.cpp:2",
        )
        self.assertEqual(
            sorted(
                (e.resource, e.access, e.value)
                for e in half
                if e.resource.startswith("SEM.")
            ),
            [("SEM.MATH_PACK", "read", ""), ("SEM.MATH_PACK", "write", "max=2;init=0")],
        )

    def test_hw_configure_writes_every_field_of_the_override_mask(self) -> None:
        model = self.fn("_llk_math_hw_configure_")
        line = ev(
            COMMON,
            "cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32, 0, src_fmt_override_mask>(0);",
        )
        self.assertEqual(
            sorted(
                (e.resource, e.value, e.pattern, e.granularity)
                for e in model.effects
                if e.evidence == line
            ),
            # the core resolves the constexpr mask and names every field it covers
            [
                (
                    "CFG.ALU_FORMAT_SPEC_REG_SrcA_override",
                    "0",
                    "core.cfg_reg_rmw_tensix",
                    "field",
                ),
                (
                    "CFG.ALU_FORMAT_SPEC_REG_SrcA_val",
                    "0",
                    "core.cfg_reg_rmw_tensix",
                    "field",
                ),
                (
                    "CFG.ALU_FORMAT_SPEC_REG_SrcB_override",
                    "0",
                    "core.cfg_reg_rmw_tensix",
                    "field",
                ),
                (
                    "CFG.ALU_FORMAT_SPEC_REG_SrcB_val",
                    "0",
                    "core.cfg_reg_rmw_tensix",
                    "field",
                ),
            ],
        )
        fp32 = [
            e
            for e in self.db.effects(model.key)
            if e.resource == "CFG.ALU_ACC_CTRL_Fp32_enabled"
        ]
        self.assertEqual(
            [(e.value, e.value_kind) for e in fp32], [("is_fp32_dest_acc_en", "param")]
        )

    # ---- roles, threads, determinism ---------------------------------------- #
    def test_roles_and_threads(self) -> None:
        self.assertEqual(self.fn("eltwise_binary_func").role, "helper")
        self.assertEqual(self.fn("get_effective_math_fidelity").role, "helper")
        self.assertEqual(self.fn("_llk_math_eltwise_binary_standard_").role, "execute")
        self.assertEqual(
            self.fn("eltwise_binary_configure_mop_standard").role, "config"
        )
        scoped = [
            fn for fn in self.db.functions.values() if math_family.in_scope(fn.path)
        ]
        self.assertTrue(scoped)
        self.assertEqual({fn.thread for fn in scoped}, {"math"})
        self.assertFalse(
            math_family.in_scope(LLK + "llk_lib/llk_math_eltwise_unary_sfpu.h")
        )

    def test_output_is_deterministic(self) -> None:
        first = effects_db.dumps_effects(self.db.to_json())
        second = effects_db.dumps_effects(
            effects_db.build(self.root, families=["math"]).to_json()
        )
        self.assertEqual(first, second)
        self.assertNotIn(str(self.root), first)


if __name__ == "__main__":
    unittest.main()
