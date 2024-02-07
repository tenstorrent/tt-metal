#include "ckernel_template.h"
#include "fw_debug.h"

namespace ckernel
{
extern volatile uint *cfg_regs;

ckernel_template::ckernel_template(uint outer_loop_len, uint inner_loop_len, uint loop_op)
    : m_outer_loop_len(outer_loop_len)
    , m_inner_loop_len(inner_loop_len)
    , m_loop_op0(loop_op)
    , m_loop_op1(TT_OP_NOP)
    , m_end_op0(TT_OP_NOP)
    , m_end_op1(TT_OP_NOP)
    , m_start_op0(TT_OP_NOP)
{
    m_loop0_last_instr = loop_op;
    m_loop1_last_instr = loop_op;
}

ckernel_template::ckernel_template(uint outer_loop_len, uint inner_loop_len, uint loop_op0, uint loop_op1)
    : m_outer_loop_len(outer_loop_len)
    , m_inner_loop_len(inner_loop_len)
    , m_loop_op0(loop_op0)
    , m_loop_op1(loop_op1)
    , m_end_op0(TT_OP_NOP)
    , m_end_op1(TT_OP_NOP)
    , m_start_op0(TT_OP_NOP)
{
    m_loop0_last_instr = loop_op1;
    m_loop1_last_instr = loop_op1;
}

void ckernel_template::set_loop_op0(uint loop_op)
{
    m_loop_op0 = loop_op;
}

void ckernel_template::set_loop_op1(uint loop_op)
{
    m_loop_op1 = loop_op;
}

void ckernel_template::set_end_ops(uint end_op0, uint end_op1)
{
    m_end_op0 = end_op0;
    m_end_op1 = end_op1;
}

void ckernel_template::set_end_op(uint end_op0)
{
    set_end_ops(end_op0, TT_OP_NOP);
}

void ckernel_template::set_start_op(uint start_op0)
{
    m_start_op0 = start_op0;
}

void ckernel_template::set_last_inner_loop_instr(uint op)
{
    m_loop1_last_instr = op;
}

void ckernel_template::set_last_outer_loop_instr(uint op)
{
    m_loop0_last_instr = op;
}

void ckernel_template::program_and_run(volatile uint *instrn_buffer)
{
    program(instrn_buffer);
    run(instrn_buffer);
}

void ckernel_template::run(volatile uint *instrn_buffer)
{
    TTI_MOP(1, 0, 0); // run the double-loop template
}

void ckernel_template::program(volatile uint *instrn_buffer)
{
    volatile uint *mop_cfg = reinterpret_cast<volatile uint *>(TENSIX_MOP_CFG_BASE);

    mop_sync(); // wait until previous mops have completed

    mop_cfg[0] = m_outer_loop_len;
    mop_cfg[1] = m_inner_loop_len;
    mop_cfg[2] = m_start_op0;
    mop_cfg[3] = m_end_op0;
    mop_cfg[4] = m_end_op1;
    mop_cfg[5] = m_loop_op0;
    mop_cfg[6] = m_loop_op1;
    mop_cfg[7] = m_loop0_last_instr;
    mop_cfg[8] = m_loop1_last_instr;
}

void ckernel_unpack_template::program_and_run(volatile uint *instrn_buffer, const uint8_t count, const uint32_t zmask)
{
    program(instrn_buffer);
    run(instrn_buffer, count, zmask);
}

void ckernel_unpack_template::run(volatile uint *instrn_buffer, const uint8_t count, const uint32_t zmask)
{
    FWASSERT("Unpack template only supports loops up to 128", count <= 128);
    TT_MOP_CFG(zmask >> 16);              // Set the top 16 bits of zmask - we could skip this for count <= 16
    TT_MOP(0, count - 1, zmask & 0xFFFF); // Run the template
}

// Version without zmask, should be slightly faster by eliminating one instruction.
void ckernel_unpack_template::run(volatile uint *instrn_buffer, const uint8_t count)
{
    FWASSERT("Unpack template only supports loops up to 128", count <= 128);
    TT_MOP(0, count - 1, 0); // Run the template
}

void ckernel_unpack_template::program(volatile uint *instrn_buffer) const
{
    volatile uint *mop_cfg = reinterpret_cast<volatile uint *>(TENSIX_MOP_CFG_BASE);

    mop_sync(); // wait until previous mops have completed

    mop_cfg[1] = m_unpackB | (m_unpack_halo << 1);
    mop_cfg[2] = m_B_instr;
    mop_cfg[3] = m_A0_instr;
    mop_cfg[4] = m_A1_instr;
    mop_cfg[5] = m_A2_instr;
    mop_cfg[6] = m_A3_instr;
    mop_cfg[7] = m_skipA_instr;
    mop_cfg[8] = m_skipB_instr;
}

ckernel_unpack_template ckernel_unpack_template::lA(uint A_instr, uint skipA_instr)
{
    return ckernel_unpack_template(false, // src B
        false,                            // halo
        A_instr, 0, 0, 0, skipA_instr, 0, 0);
}

ckernel_unpack_template ckernel_unpack_template::lB(uint B_instr, uint skipB_instr)
{
    return ckernel_unpack_template(false, // src B
        false,                            // halo
        B_instr, 0, 0, 0, skipB_instr, 0, 0);
}

ckernel_unpack_template ckernel_unpack_template::lzA(bool neginf, uint A_instr, uint skipA_instr)
{
    return ckernel_unpack_template(false, // src B
        true,                             // halo
        neginf ? DEF_NINFSRCA : DEF_ZEROSRCA, A_instr, DEF_UNPACR_NOP, DEF_UNPACR_NOP, skipA_instr, 0, 0);
}

ckernel_unpack_template ckernel_unpack_template::lhA(const uint32_t halo_mask)
{
    // Figure out which unpack is last
    const uint last_mask = (halo_mask == 0x1) ? 0x1 : (halo_mask <= 0x3) ? 0x2 : (halo_mask <= 0x7) ? 0x4 : 0;

    return ckernel_unpack_template(false, // src B
        true,                             // halo
        ((halo_mask >> 0) & 0x1) ? ((last_mask >> 0) & 0x1) ? DEF_A0_last_instr : DEF_A0_instr : SKIP_A0_instr,
        ((halo_mask >> 1) & 0x1) ? ((last_mask >> 1) & 0x1) ? DEF_A1_last_instr : DEF_A1_instr : SKIP_A1_instr,
        ((halo_mask >> 2) & 0x1) ? ((last_mask >> 2) & 0x1) ? DEF_A2_last_instr : DEF_A2_instr : SKIP_A2_instr,
        ((halo_mask >> 3) & 0x1) ? DEF_A3_instr : SKIP_A3_instr, DEF_SKIP_A, 0, 0);
}

ckernel_unpack_template ckernel_unpack_template::flhA(const uint32_t halo_mask)
{
    // Figure out which unpack is last
    const uint last_mask = (halo_mask == 0x1) ? 0x1 : (halo_mask <= 0x3) ? 0x2 : (halo_mask <= 0x7) ? 0x4 : 0;

    return ckernel_unpack_template(false, // src B
        true,                             // halo
        ((halo_mask >> 0) & 0x1) ? ((last_mask >> 0) & 0x1) ? DEF_A0_fconv_last_instr : DEF_A0_fconv_instr : SKIP_A0_instr,
        ((halo_mask >> 1) & 0x1) ? ((last_mask >> 1) & 0x1) ? DEF_A1_fconv_last_instr : DEF_A1_fconv_instr : SKIP_A1_instr,
        ((halo_mask >> 2) & 0x1) ? ((last_mask >> 2) & 0x1) ? DEF_A2_fconv_last_instr : DEF_A2_fconv_instr : SKIP_A2_instr,
        ((halo_mask >> 3) & 0x1) ? DEF_A3_fconv_instr : SKIP_A3_instr, TT_OP_NOP, 0, 0);
}

ckernel_unpack_template ckernel_unpack_template::lBhA(const uint32_t halo_mask, const bool rarefy)
{
    // Figure out which unpack is last
    const uint last_mask = (halo_mask == 0x1) ? 0x1 : (halo_mask <= 0x3) ? 0x2 : (halo_mask <= 0x7) ? 0x4 : 0;

    return ckernel_unpack_template(true, // src B
        true,                            // halo
        ((halo_mask >> 0) & 0x1) ? ((last_mask >> 0) & 0x1) ? DEF_A0_last_instr : DEF_A0_instr : SKIP_A0_instr,
        ((halo_mask >> 1) & 0x1) ? ((last_mask >> 1) & 0x1) ? DEF_A1_last_instr : DEF_A1_instr : SKIP_A1_instr,
        ((halo_mask >> 2) & 0x1) ? ((last_mask >> 2) & 0x1) ? DEF_A2_last_instr : DEF_A2_instr : SKIP_A2_instr,
        ((halo_mask >> 3) & 0x1) ? DEF_A3_instr : SKIP_A3_instr, DEF_SKIP_A, rarefy ? DEF_B_rarefy_cntx_ovrd_instr : DEF_B_cntx_ovrd_instr, DEF_SKIP_B);
}

ckernel_unpack_template ckernel_unpack_template::flBhA(const uint32_t halo_mask)
{
    // Figure out which unpack is last
    const uint last_mask = (halo_mask == 0x1) ? 0x1 : (halo_mask <= 0x3) ? 0x2 : (halo_mask <= 0x7) ? 0x4 : 0;

    return ckernel_unpack_template(true, // src B
        true,                            // halo
        ((halo_mask >> 0) & 0x1) ? ((last_mask >> 0) & 0x1) ? DEF_A0_fconv_last_instr : DEF_A0_fconv_instr : SKIP_A0_instr,
        ((halo_mask >> 1) & 0x1) ? ((last_mask >> 1) & 0x1) ? DEF_A1_fconv_last_instr : DEF_A1_fconv_instr : SKIP_A1_instr,
        ((halo_mask >> 2) & 0x1) ? ((last_mask >> 2) & 0x1) ? DEF_A2_fconv_last_instr : DEF_A2_fconv_instr : SKIP_A2_instr,
        ((halo_mask >> 3) & 0x1) ? DEF_A3_fconv_instr : SKIP_A3_instr, TT_OP_NOP, DEF_B_cntx_ovrd_no_z_inc_instr, DEF_SKIP_B);
}

ckernel_unpack_template ckernel_unpack_template::lBA(uint A_instr, uint skipA_instr,

    uint B_instr, uint skipB_instr)
{
    return ckernel_unpack_template(true, // src B
        false,                           // halo
        A_instr, 0, 0, 0, skipA_instr, B_instr, skipB_instr);
}

ckernel_unpack_template ckernel_unpack_template::loopx1instr(uint instr0, uint skip0){
    return ckernel_unpack_template::lA(instr0, skip0);
}

ckernel_unpack_template ckernel_unpack_template::loopx2instr(uint instr0, uint instr1, uint skip0, uint skip1){
    // Note - 2 instr loop so we will hijack B_instr slot for 2nd instruction via lBA.
    return ckernel_unpack_template::lBA(instr0, skip0, instr1, skip1);
}

} // namespace ckernel
