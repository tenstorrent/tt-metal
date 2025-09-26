// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Compiler hint that a branch is unlikely to be taken
#define UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)
#define tt_l1_ptr           __attribute__((rvtt_l1_ptr))
#define tt_reg_ptr          __attribute__((rvtt_reg_ptr))
#include "ckernel_include.h"
#include "ckernel_ops.h"
// #include "fw_debug.h"
#include "t6_debug_map.h"
#include "tensix.h"

namespace ckernel
{

// MM Oct 13 2021: Now only 1024 rows in dest
static int const DEST_MAX_ADDR_HALF_16B = 512;
static int const DEST_MAX_ADDR_HALF_32B = 256;
static int const DEST_MAX_ADDR_16B      = 1024;
static int const DEST_MAX_ADDR_32B      = 512;

constexpr uint8_t TENSIX_MATH_SEMAPHORE   = p_stall::SEMAPHORE_1;
constexpr uint8_t TENSIX_PERF_SEMAPHORE   = p_stall::SEMAPHORE_2;
constexpr uint8_t MATH_SEMAPHORE          = 1;
constexpr uint8_t PC_BUF_SEMAPHORE_BASE   = 32; // base address for semaphores in PC buffer. FIXME: must be kept in sync with SEM_COUNT parameter... ugly...
constexpr uint8_t STREAM_SEMAPHORE        = 5;  // semaphore used by unpack thread to sync between trisc and unpacker
constexpr uint8_t TENSIX_STREAM_SEMAPHORE = p_stall::SEMAPHORE_5; // semaphore used by unpack thread to sync between trisc and unpacker
constexpr uint8_t PARAM_ITERATIONS        = 0;
constexpr uint8_t TENSIX_UNPACK_TO_DEST_UNPACK_SEMAPHORE = p_stall::SEMAPHORE_4;
constexpr uint8_t UNPACK_TO_DEST_UNPACK_SEMAPHORE        = 4;
constexpr uint8_t TENSIX_PACK_STREAM_SEMAPHORE           = p_stall::SEMAPHORE_6;
constexpr uint8_t PACK_STREAM_SEMAPHORE                  = 6;
constexpr uint8_t TENSIX_UNPACK_TO_DEST_PACK_SEMAPHORE   = p_stall::SEMAPHORE_7;
constexpr uint8_t UNPACK_TO_DEST_PACK_SEMAPHORE          = 7;

constexpr uint32_t KERNEL_COMPLETE = 0x1;

volatile uint *const reg_base        = (volatile uint *)0xFFB10000;
volatile uint *const pc_buf_base     = (volatile uint *)PC_BUF_BASE;
volatile uint *const regfile         = (volatile uint *)REGFILE_BASE;
volatile uint *const instrn_buffer   = (volatile uint *)INSTRN_BUF_BASE;
volatile uint *const mailbox_base[4] = {
    (volatile uint *)TENSIX_MAILBOX0_BASE, (volatile uint *)TENSIX_MAILBOX1_BASE, (volatile uint *)TENSIX_MAILBOX2_BASE, (volatile uint *)TENSIX_MAILBOX3_BASE};
volatile uint *const replay_mmap = (uint32_t volatile *)(INSTRN_BUF_BASE + (1 << 10));

inline void mmio_register_write(register_space_e space, uint addr, uint data)
{
    const uint regaddr = (space << 6) | (addr & 0x3F);
    // FWLOG2("Regaddr: 0x%x, data: 0x%x", regaddr, data);
    reg_base[regaddr] = data;
}

inline void sync_regfile_write(const uint index)
{
    volatile uint foo     = 0xdeadbeef;
    volatile uint *fooptr = &foo;
    *fooptr               = regfile[index];
}

inline uint8_t semaphore_read(const uint8_t index)
{
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
}

inline void semaphore_post(const uint8_t index)
{
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0;
}

inline void semaphore_get(const uint8_t index)
{
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1;
}

// Register read (workaround for bug
// tenstorrent/tensix#976
// now handled by the compiler)
// workaround is needed only for GS
inline uint reg_read(uint32_t addr)
{
    volatile uint tt_reg_ptr *p_reg = reinterpret_cast<volatile uint tt_reg_ptr *>(addr);
    return p_reg[0];
}

inline void reg_write(uint32_t addr, uint32_t data)
{
    volatile uint tt_reg_ptr *p_reg = reinterpret_cast<volatile uint tt_reg_ptr *>(addr);
    p_reg[0]                        = data;
}

//
//
// inline void wait_math_semaphores()
//{
//  FWLOG0("wait_math_semaphore");
//  // wait while math semaphore is on max, no room to write math results
//  TTI_SEMWAIT(p_stall::STALL_MATH, p_stall::STALL_ON_MAX, 0, p_stall::SEMAPHORE_1);
//}
//
// inline void set_math_semaphores()
//{
//	FWLOG0("set_math_semaphore");
//  // Tell packer that it has something to pack
//  TTI_STALLWAIT(p_stall::STALL_SYNC, 0, 0, p_stall::MATH); // when using HF, it could take 4 cycles before math is really done
//  TTI_SEMPOST(0, p_stall::SEMAPHORE_1);
//}
//
//// Flip math dest register offset to 0 or 0x80, depending on the iteration,
//// flip-flopping between two halves
// inline void select_math_dest_registers(const uint iteration, const uint32_t dest_offset=DEST_MAX_ADDR_HALF_16B)
//{
//	TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, (iteration%2 != dest_offset_id) ? dest_offset : 0x0);
// }
// inline uint32_t get_dest_offset(){
//	volatile uint * cfg_regs       = reinterpret_cast<volatile uint *>(TENSIX_CFG_BASE);
//	const uint32_t cfg_addr0 = (cfg_state_id == 0) ? ALU_ROUNDING_MODE_Fpu_srnd_en_ADDR32 : (CFG_STATE_SIZE * 4) + ALU_ROUNDING_MODE_Fpu_srnd_en_ADDR32;
//     const uint32_t cfg_addr1 = (cfg_state_id == 0) ? THCON_UNPACKER0_REG0_OUT_DATA_FORMAT_ADDR32 : (CFG_STATE_SIZE * 4) +
//     THCON_UNPACKER0_REG0_OUT_DATA_FORMAT_ADDR32;
//	uint32_t cfg_data0 = cfg_regs[cfg_addr0];
//	DataFormat cfg_data1 = static_cast<DataFormat>(cfg_regs[cfg_addr1]);
//	uint32_t dest_offset =
//       (
//         (cfg_data0 >> ALU_ACC_CTRL_Fp32_enabled_SHAMT) ||
//         (cfg_data1 == DataFormat::Float32) ||
//         (cfg_data1 == DataFormat::Int32)
//       )
//       ? DEST_MAX_ADDR_HALF_32B
//       : DEST_MAX_ADDR_HALF_16B
//     ; //Tests both Fp32 snd int32
//
//	FWLOG2("StateID:%d, dest_offset:%x", cfg_state_id, dest_offset);
//	return dest_offset;
// }

inline void tensix_sync()
{
    volatile uint foo     = 0xdeadbeef;
    volatile uint *fooptr = &foo;
    // Write to pc buffer to push all writes ahead of us.. otherwise, the pc buffer read can bypass older writes
    pc_buf_base[1] = foo;

    // Now read -- this read will block until we're idle
    *fooptr = pc_buf_base[1];
}

inline void mop_sync()
{
    volatile uint foo     = 0xdeadbeef;
    volatile uint *fooptr = &foo;
    // Write to pc buffer to push all writes ahead of us.. otherwise, the pc buffer read can bypass older writes
    pc_buf_base[2] = foo;

    // Now read -- this read will block until mops are done
    *fooptr = pc_buf_base[2];
}

inline void cfg_rmw(uint32_t cfg_addr32, uint32_t cfg_shamt, uint32_t cfg_mask, uint32_t val)
{
    //  FWLOG1("cfg_rmw() addr:%d", cfg_addr32 );
    //  FWLOG1("cfg_rmw() shamt:%d", cfg_shamt );
    //  FWLOG1("cfg_rmw() mask :%x", cfg_mask  );
    //  FWLOG1("cfg_rmw() data :%x", val  );

    // uint32_t bit_offset = cfg_shamt - cfg_shamt 0x1ffff;

    uint32_t wrdata = val << cfg_shamt;
    uint8_t mask_b0 = cfg_mask & 0xff;
    if (mask_b0 != 0)
    {
        uint8_t data_b0 = wrdata & 0xff;
        // FWLOG1("cfg_rmw() data_b0 :%x", data_b0  );
        TT_RMWCIB0(cfg_addr32, mask_b0, data_b0);
    }
    wrdata >>= 8;

    uint8_t mask_b1 = (cfg_mask >> 8) & 0xff;
    if (mask_b1 != 0)
    {
        uint8_t data_b1 = (wrdata) & 0xff;
        // FWLOG1("cfg_rmw() data_b1 :%x", data_b1  );
        TT_RMWCIB1(cfg_addr32, mask_b1, data_b1);
    }
    wrdata >>= 8;

    uint8_t mask_b2 = (cfg_mask >> 16) & 0xff;
    if (mask_b2 != 0)
    {
        uint8_t data_b2 = (wrdata) & 0xff;
        // FWLOG1("cfg_rmw() data_b2 :%x", data_b2  );
        TT_RMWCIB2(cfg_addr32, mask_b2, data_b2);
    }
    wrdata >>= 8;

    uint8_t mask_b3 = (cfg_mask >> 24) & 0xff;
    if (mask_b3 != 0)
    {
        uint8_t data_b3 = (wrdata) & 0xff;
        // FWLOG1("cfg_rmw() data_b3 :%x", data_b3  );
        TT_RMWCIB3(cfg_addr32, mask_b3, data_b3);
    }
}

// WTF is this?????
// inline void cfg_rmw_gpr(uint32_t cfg_addr32, uint32_t cfg_shamt, uint32_t cfg_mask, uint32_t gpr_index){
// 	 const uint32_t wrdata = regfile[gpr_index];
// 	 cfg_rmw(cfg_addr32, cfg_shamt, cfg_mask, wrdata);
// }
//
// inline void cfg_rmw_tensix(uint32_t cfg_addr32, uint32_t cfg_shamt, uint32_t cfg_mask, uint32_t gpr_index, uint32_t tmp_gpr0, uint32_t tmp_gpr1, uint32_t
// tmp_gpr2) {
// 	//Read config reg for RMW (R53 (tmp_gpr0))
// 	TTI_RDCFG(tmp_gpr0,cfg_addr32);													// R53 =
// CFG[LOOP_CNT_REG8_LoopCnt_SET0_ADDR32]
//
// 	// Set up gpr for mask (R54 (tmp_gpr1))
// 	TTI_SETGPR(0, (~cfg_mask) & 0xffff, 0, tmp_gpr1*2+0);            		// R54(L) = ~MASK_LO
// 	TTI_SETGPR(0, (~(cfg_mask>>16)) & 0xffff, 0, tmp_gpr1*2+1);          // R54(H) = ~MASK_HI
//
// 	//Zero out relevant bits in cfg word
// 	TTI_BITWOPGPR(0, 0, tmp_gpr0,tmp_gpr0,tmp_gpr1);
//
// 	// Set up gpr for mask (R54 (tmp_gpr1))
// 	TTI_SETGPR(0, (cfg_mask) & 0xffff, 0, tmp_gpr1*2+0);            		// R54(L) = MASK_LO
// 	TTI_SETGPR(0, ((cfg_mask>>16)) & 0xffff, 0, tmp_gpr1*2+1);          // R54(H) = MASK_HI
//
// 	//Setup gpr for shamt (R55 (tmp_gpr2))
// 	TTI_SETGPR(0, cfg_shamt, 0, tmp_gpr2*2+0);            					// R55(L) = SHAMT
// 	TTI_SETGPR(0, 0, 0, tmp_gpr2*2+1); // R55(H) = 0x0000
//
// 	//Left shift data to be written to align with cfg bits and reuse shamt gpr for result 		// R55 = R25<<R55
// 	TTI_SHIFTGPR(0, 0,tmp_gpr2,tmp_gpr2, gpr_index);
//
// 	//Apply Mask to ensure only relevant bits are non-zero 												// R55 =
// R55 & R54 	TTI_BITWOPGPR(0, 0, tmp_gpr2,tmp_gpr2,tmp_gpr1);
//
// 	//OR with current cfg bits
// // R55 = R55 | R53 	TTI_BITWOPGPR(0, 1, tmp_gpr2,tmp_gpr2,tmp_gpr0);
//
// 	TTI_STALLWAIT(p_stall::STALL_CFG, 0, 0, p_stall::THCON);
// 	TTI_WRCFG(tmp_gpr2,p_cfg::WRCFG_32b,cfg_addr32);
// }

// CHECKME: does this need to change now that BRISC is gone?
inline void mailbox_write(const uint8_t thread, const uint32_t data)
{
    mailbox_base[thread][0] = data;
}

// Blocking read
inline uint32_t mailbox_read(const uint8_t thread)
{
    return mailbox_base[thread][0];
}

inline bool mailbox_not_empty(const uint8_t thread)
{
    return mailbox_base[thread][1] > 0;
}

// If the TRACK_x bit is set, then the Tensix hardware will automatically
// stall TRISC memory accesses and/or Tensix instructions to x in order
// to guarantee correct ordering. This should eliminate most cases where
// we used to need a tensix_sync() or a sync_regfile_write().
//
// The EN_SUBDIVIDED_CFG_FOR_UNPACR is more subtle. If it is turned off,
// then the global CFG registers are treated as one big entity, and ANY
// access from Tensix instructions will be synchronized with ANY access
// from this TRISC. If it is on, then we distinguish between accesses
// target CFG regs for unpacker 0, CFG regs for unpacker 1, and all the
// others (meaning that no synchronization will happen between, for
// example, a TRISC access to an unpacker 1 register and an UNPACR
// instruction that targets unpacker 0).
//
constexpr static uint TRACK_GLOBAL_CFG             = 1 << 0;
constexpr static uint EN_SUBDIVIDED_CFG_FOR_UNPACR = 1 << 1;
constexpr static uint TRACK_GPR                    = 1 << 2;
constexpr static uint TRACK_TDMA                   = 1 << 3;
constexpr static uint TRACK_TENSIX_INSTRUCTIONS    = 1 << 4;
constexpr static uint TRACK_ALL                    = 0x1F;

// HACK: I inverted this signal in RTL, should probably clean this up at some point
template <uint bitmask>
inline void set_ttsync_enables(uint thread_id = 0xdeadface)
{
    static_assert((bitmask & ~TRACK_ALL) == 0, "The given bitmask targets bits outside the allowable range");
    auto t6dbg = RISCV_DEBUG_REGS;

    if (thread_id > 3)
    {
        // FWLOG0(
        //     "WARNING: automatically writing to all TTSync enable regs, because the given thread ID (or lack thereof) implies that you don't know what thread
        //     " "you want");
        for (int i = 0; i < 3; i++)
        {
            t6dbg->TENSIX_TRISC_SYNC[i] = ~bitmask;
        }
    }
    else
    {
        t6dbg->TENSIX_TRISC_SYNC[thread_id] = ~bitmask;
    }
}

template <bool add_nops = false>
__attribute__((always_inline)) inline void disable_gathering()
{
    // Disable gathering: set bit 18
    asm(R"ASM(
        .option push
        fence
        li   t1, 0x1
        slli t1, t1, 18
        csrrs zero, 0x7c0, t1
        .option pop
         )ASM" ::
            : "t1");

    // Gathering is done early in the pipeline, so we need to make sure
    // the above csrrw gets processed before the load-replay instructions
    if (add_nops)
    {
        TTI_NOP;
        TTI_NOP;
        TTI_NOP;
    }
}

__attribute__((always_inline)) inline void enable_gathering()
{
    // Enable gathering: clear bit 18
    asm(R"ASM(
        .option push
        li   t1, 0x1
        slli t1, t1, 18
        csrrc zero, 0x7c0, t1
        .option pop
         )ASM" ::
            : "t1");
}

// Pass a lambda function (or a regular function pointer) that takes void,
// returns void, and issues the instructions you want to load into the
// replay buffer. start, len, and exec_while_loading have the same meaning
// as they do for the REPLAY instruction, as descired in assembly.yaml.
template <uint start, uint len, bool exec_while_loading = false, uint set_mutex = 0, uint last = 0, typename F>
__attribute__((always_inline)) inline void load_replay_buf(F fn)
{
    if (len > 0)
    {
        // disable_gathering(); // MM Jun 11 / 2025: no longer needed

        // Issue instruction to load replay buffer
        TTI_REPLAY(start, len, last, set_mutex, exec_while_loading, 1);

        // Send in the user's desired instructions
        fn();

        // enable_gathering(); // MM Jun 11 / 2025: no longer needed
    }
}

// Same as above, but used if start/len/exec_while_loading are not known
// at compile time.
template <typename F>
__attribute__((always_inline)) inline void load_replay_buf(uint start, uint len, bool exec_while_loading, uint set_mutex, uint last, F fn)
{
    disable_gathering();

    // Issue instruction to load replay buffer
    TT_REPLAY(start, len, last, set_mutex, exec_while_loading, 1);

    // Send in the user's desired instructions
    fn();

    enable_gathering();
}

enum class CSR : uint16_t
{
    tensix_queue_status = 0xBC0,
    tensix_busy_status  = 0xBC1,
    NEO_ID              = 0xBC2,
    TRISC_ID            = 0xBC3,
    UNUSED_0            = 0xBC4,
    UNUSED_1            = 0xBC5,
    UNUSED_2            = 0xBC6,
    UNUSED_3            = 0xBC7,
    UNUSED_4            = 0xBC8,
    UNUSED_5            = 0xBC9,
    MISA                = 0x301, // Machine ISA, expected to be 0x40201123 in BH
};

template <CSR csr_num, bool fence = true>
inline uint32_t csr_read()
{
    uint32_t ret;

    if (fence)
    {
        asm volatile("fence");
    }
    asm volatile("csrr %[ret], %[csr_num] \n" : [ret] "=r"(ret) : [csr_num] "i"(csr_num));

    return ret;
}

// Use at your own risk :-)
template <uint16_t csr_num, bool fence = true>
inline uint32_t csr_read()
{
    static_assert(csr_num < (1 << 12), "Given CSR number is out of range");
    uint32_t ret;

    if (fence)
    {
        asm volatile("fence");
    }
    asm volatile("csrr %[ret], %[csr_num] \n" : [ret] "=r"(ret) : [csr_num] "i"(csr_num));

    return ret;
}

template <CSR csr_num, bool fence = true>
inline void csr_write(uint32_t val)
{
    if (fence)
    {
        asm volatile("fence");
    }
    asm volatile("csrw %[csr_num], %[val] \n" : : [csr_num] "i"(csr_num), [val] "r"(val));
}

// Use at your own risk :-)
template <uint16_t csr_num, bool fence = true>
inline void csr_write(uint32_t val)
{
    static_assert(csr_num < (1 << 12), "Given CSR number is out of range");

    if (fence)
    {
        asm volatile("fence");
    }
    asm volatile("csrw %[csr_num], %[val] \n" : : [csr_num] "i"(csr_num), [val] "r"(val));
}

union qstatus_u
{
    uint32_t val;

    struct
    {
        unsigned replay : 1;
        unsigned mop    : 1;
        unsigned thcon  : 1;
        unsigned xmov   : 1;
        unsigned unpack : 1;
        unsigned pack   : 1;
        unsigned cfg    : 1;
        unsigned sync   : 1;
        unsigned tdma   : 1;
        unsigned _sfpu  : 1; // ugh.... the "sfpu" and "SFPU" identifiers are already in use...
        unsigned fpu    : 1;
        unsigned sfpucc : 2;

        unsigned global_replay : 1;
        unsigned global_mop    : 1;
        unsigned global_thcon  : 1;
        unsigned global_xmov   : 1;
        unsigned global_unpack : 1;
        unsigned global_pack   : 1;
        unsigned global_cfg    : 1;
        unsigned global_sync   : 1;
        unsigned global_tdma   : 1;
        unsigned global_sfpu   : 1;
        unsigned global_fpu    : 1;
        unsigned global_sfpucc : 2;
    };
};

union bstatus_u
{
    uint32_t val;

    struct
    {
        unsigned replay : 1;
        unsigned mop    : 1;
        unsigned thcon  : 1;
        unsigned xmov   : 1;
        unsigned unpack : 1;
        unsigned pack   : 1;
        unsigned cfg    : 1;
        unsigned sync   : 1;
        unsigned tdma   : 1;
        unsigned _sfpu  : 1; // ugh.... the "sfpu" and "SFPU" identifiers are already in use...
        unsigned fpu    : 1;

        unsigned global_replay : 1;
        unsigned global_mop    : 1;
        unsigned global_thcon  : 1;
        unsigned global_xmov   : 1;
        unsigned global_unpack : 1;
        unsigned global_pack   : 1;
        unsigned global_cfg    : 1;
        unsigned global_sync   : 1;
        unsigned global_tdma   : 1;
        unsigned global_sfpu   : 1;
        unsigned global_fpu    : 1;
    };
};

// Helper functions to wait on specific parts Tensix to go idle

// For convenience, some named functions. Generated with the following Lua code:
/*

ex2field = setmetatable({}, {__index = function(t,k) return k end})
ex2field.sfpu = "_sfpu"

template = [[
inline void wait_@exunit@_idle() {bstatus_u bstatus; do bstatus.val = csr_read<CSR::tensix_busy_status>(); while(bstatus.@exfield@ != 0);}
]]

ex_units = {"replay", "mop", "thcon", "xmov", "unpack", "pack", "cfg", "sync", "tdma", "sfpu", "fpu"}

for _,v in ipairs(ex_units) do
    local str = template:gsub("@exunit@", v):gsub("@exfield@", ex2field[v])
    io.write(str)
end

*/
inline void wait_replay_idle()
{
    bstatus_u bstatus;
    do
    {
        bstatus.val = csr_read<CSR::tensix_busy_status>();
    } while (bstatus.replay != 0);
}

inline void wait_mop_idle()
{
    bstatus_u bstatus;
    do
    {
        bstatus.val = csr_read<CSR::tensix_busy_status>();
    } while (bstatus.mop != 0);
}

inline void wait_thcon_idle()
{
    bstatus_u bstatus;
    do
    {
        bstatus.val = csr_read<CSR::tensix_busy_status>();
    } while (bstatus.thcon != 0);
}

inline void wait_unpack_idle()
{
    bstatus_u bstatus;
    do
    {
        bstatus.val = csr_read<CSR::tensix_busy_status>();
    } while (bstatus.unpack != 0);
}

inline void wait_pack_idle()
{
    bstatus_u bstatus;
    do
    {
        bstatus.val = csr_read<CSR::tensix_busy_status>();
    } while (bstatus.pack != 0);
}

inline void wait_cfg_idle()
{
    bstatus_u bstatus;
    do
    {
        bstatus.val = csr_read<CSR::tensix_busy_status>();
    } while (bstatus.cfg != 0);
}

inline void wait_sync_idle()
{
    bstatus_u bstatus;
    do
    {
        bstatus.val = csr_read<CSR::tensix_busy_status>();
    } while (bstatus.sync != 0);
}

inline void wait_tdma_idle()
{
    bstatus_u bstatus;
    do
    {
        bstatus.val = csr_read<CSR::tensix_busy_status>();
    } while (bstatus.tdma != 0);
}

inline void wait_sfpu_idle()
{
    bstatus_u bstatus;
    do
    {
        bstatus.val = csr_read<CSR::tensix_busy_status>();
    } while (bstatus._sfpu != 0);
}

inline void wait_fpu_idle()
{
    bstatus_u bstatus;
    do
    {
        bstatus.val = csr_read<CSR::tensix_busy_status>();
    } while (bstatus.fpu != 0);
}

// Slightly more general CSR waiting function. Can be used like this:
/*
    bstatus_u wait_for_me;
    wait_for_me.cfg = 1;
    wait_for_me.sync = 1;
    //etc...
    wait_bstatus_low(wait_for_me.val);
*/

inline void wait_bstatus_low(uint32_t val)
{
    bstatus_u bstatus;
    do
    {
        bstatus.val = csr_read<CSR::tensix_busy_status>();
    } while ((bstatus.val & val) != 0);
}

// Little utility function to help with certain single-core-io-custom-kernel tests.
// Marked static as a dumb way to avoid multiple definition error
[[maybe_unused]]
static void time_waster(int num_iters)
{
    uint32_t volatile *some_random_l1_address = (uint32_t volatile *)0x14FFE;
    uint32_t dont_optimize_me_away;

    for (int i = 0; i < num_iters; i++)
    {
        // FWLOG1("Just wasting time... i = %d", i);
        dont_optimize_me_away = *some_random_l1_address;
        if (dont_optimize_me_away == 0xFACEBEEF)
        {
            // This condition should hopefully keep the C++ compiler from
            // optimizing out the loop (I'm banking on the fact that it's
            // vanishingly unlikely for 0xFACEBEEF to be sitting at some
            // random L1 address)
            break;
        }
    }
}

// For unpacker, only UNPACK_TO_DEST will require unpacker to sync on dest
enum class dest_dvalid_client : int
{
    UNPACK = 0,
    FPU,
    SFPU,
    PACK
};

// I wish C++ had better code generation, instead of this messy templating stuff
template <uint32_t wait_mask, uint32_t wait_polarity, uint32_t toggle_mask>
struct mk_dest_dvalid_reg
{
    static_assert(wait_mask < 16);
    static_assert(wait_polarity < 16);
    static_assert(toggle_mask < 16);

    static uint32_t const val = ((wait_mask << 0) | (wait_polarity << 4) | (toggle_mask << 8));
};

/* Examples:

The usual case with FPU producing and pack consuming:
    set_up_dest_dvalid({
        dest_dvalid_client::FPU,
        dest_dvalid_client::PACK
    });

A case where FPU produces, SFPU modifies, and pack consumes:
    set_up_dest_dvalid({
        dest_dvalid_client::FPU,
        dest_dvalid_client::SFPU,
        dest_dvalid_client::PACK
    });

Using unpack-to-dest with SFPU and then packing out:
    set_up_dest_dvalid({
        dest_dvalid_client::UNPACK_TO_DEST,
        dest_dvalid_client::SFPU,
        dest_dvalid_client::PACK
    });

*/
template <dest_dvalid_client CURR_CLIENT, int N /*deduced*/>
void set_up_dest_dvalid_per_thread(dest_dvalid_client const (&clients)[N])
{
    static uint32_t const ctrl_regs[] = {
        UNPACK_TO_DEST_DVALID_CTRL_wait_mask_ADDR32,
        MATH_DEST_DVALID_CTRL_wait_mask_ADDR32,
        SFPU_DEST_DVALID_CTRL_wait_mask_ADDR32,
        PACK_DEST_DVALID_CTRL_wait_mask_ADDR32};

    static_assert(N > 1, "Doesn't make sense to set dvalid control if you only have one client");
    static_assert(N <= 4, "We only support up to 3 producer-consumer pairs");

    auto cfg = (uint32_t volatile *)TENSIX_CFG_BASE;

    if (CURR_CLIENT == clients[0])
    {
        cfg[ctrl_regs[static_cast<int>(clients[0])]] = mk_dest_dvalid_reg<0b1111, 0b0000, 0b0001>::val;
    }
    for (int i = 1; i < N - 1; i++)
    {
        if (CURR_CLIENT == clients[i])
        {
            cfg[ctrl_regs[static_cast<int>(clients[i])]] = mk_dest_dvalid_reg<0b0001, 0b0001, 0b0011>::val << (i - 1);
        }
    }
    if (CURR_CLIENT == clients[N - 1])
    {
        cfg[ctrl_regs[static_cast<int>(clients[N - 1])]] = mk_dest_dvalid_reg<0b0001, 0b0001, 0b0001>::val << (N - 2);
    }
}

// d e e p e s t l o r e
__attribute__((always_inline)) inline void rv_wrcfg(
    uint32_t const &wrdata_hi, uint32_t const &wrdata_lo, uint32_t const &cfg_addr, uint32_t const write_64b = 0, uint32_t const byte_mask = 0xFF)
{
    uint32_t const base_instrn = TRISC_OP_SWIZZLE(TT_OP_RV_WRCFG(byte_mask, write_64b, 0, 0, 0));
    asm volatile(
        ".equ reg_lut_zero, 0\n"
        ".equ reg_lut_ra, 1\n"
        ".equ reg_lut_sp, 2\n"
        ".equ reg_lut_gp, 3\n"
        ".equ reg_lut_tp, 4\n"
        ".equ reg_lut_t0, 5\n"
        ".equ reg_lut_t1, 6\n"
        ".equ reg_lut_t2, 7\n"
        ".equ reg_lut_s0, 8\n"
        ".equ reg_lut_fp, 8\n"
        ".equ reg_lut_s1, 9\n"
        ".equ reg_lut_a0, 10\n"
        ".equ reg_lut_a1, 11\n"
        ".equ reg_lut_a2, 12\n"
        ".equ reg_lut_a3, 13\n"
        ".equ reg_lut_a4, 14\n"
        ".equ reg_lut_a5, 15\n"
        ".equ reg_lut_a6, 16\n"
        ".equ reg_lut_a7, 17\n"
        ".equ reg_lut_s2, 18\n"
        ".equ reg_lut_s3, 19\n"
        ".equ reg_lut_s4, 20\n"
        ".equ reg_lut_s5, 21\n"
        ".equ reg_lut_s6, 22\n"
        ".equ reg_lut_s7, 23\n"
        ".equ reg_lut_s8, 24\n"
        ".equ reg_lut_s9, 25\n"
        ".equ reg_lut_s10, 26\n"
        ".equ reg_lut_s11, 27\n"
        ".equ reg_lut_t3, 28\n"
        ".equ reg_lut_t4, 29\n"
        ".equ reg_lut_t5, 30\n"
        ".equ reg_lut_t6, 31\n"
        ".word %[base_instrn] + (reg_lut_%[reg0] << 2) + (reg_lut_%[reg1] << 7) + (reg_lut_%[reg2] << 12)"
        :
        : [base_instrn] "i"(base_instrn), [reg2] "r"(wrdata_lo), [reg1] "r"(wrdata_hi), [reg0] "r"(cfg_addr)
        :);
}

// Refer to the comments about the Ports for RISC memory-mapped access near
// to the top of tt_replay_unit.sv
// The division by 4 is because the comment uses byte addresses.
static const uint32_t replay_live_mutex = 256 >> 2;
static const uint32_t mutex_0_unbanked  = 264 >> 2;
static const uint32_t mutex_1_unbanked  = 268 >> 2;
static const uint32_t replay_write_id   = 272 >> 2; // replay_instr_bank_write_id in the RTL
static const uint32_t replay_read_id    = 276 >> 2; // replay_instr_bank_read_id in the RTL
static const uint32_t mmio_write_id     = 280 >> 2; // mm_bank_write_id in the RTL

inline void start_using_replay_mmio_load(uint32_t double_banked = true)
{
    // State at the end of this routine, once replay is idle
    //    o Both Mutexes == 1
    //    o mm_bank_write_id == replay_instrn_bank_read_id
    // A normal "wait while replay_live_mutex is 0" is a sufficient guard
    // for the next MMIO replay buffer load,
    // and all subsequent MMIO replay buffer loads.

    // Make sure MMIO reads and writes come out in program order.
    // Ref TEN-2139
    asm(R"ASM(
        .option push
        fence
        li   t1, 0x1
        csrrs zero, 0x7c0, t1
        .option pop
         )ASM" ::
            : "t1");

    wait_replay_idle();
    uint32_t initial_read_id = replay_mmap[replay_read_id];
    if (replay_mmap[mmio_write_id] == initial_read_id)
    {
        // Already lined up. Just grab both Mutexes.
        replay_mmap[mutex_0_unbanked] = 1;
        if (double_banked)
        {
            replay_mmap[mutex_1_unbanked] = 1;
        }
    }
    else
    {
        // The code below flips the MMIO write ID twice. The
        // second NOP load will be REPLAYed out, which gives us the
        // side effect of flipping replay_instrn_bank_read_id
        replay_mmap[0]                 = TT_OP_NOP; // I don't think we actually need this one
        replay_mmap[replay_live_mutex] = 0;
        replay_mmap[0]                 = TT_OP_NOP;
        replay_mmap[replay_live_mutex] = 0;
        // Grab the mutex of the bank that we are NOT about
        // to replay a NOP out of
        if (initial_read_id)
        {
            replay_mmap[mutex_0_unbanked] = 1;
        }
        else
        {
            replay_mmap[mutex_1_unbanked] = 1;
        }
        TTI_REPLAY(0, 1, 1, 1, 0, 0); // last=1 will flip replay_read.
                                      // set_mutex will return the bank to software
    }
}

inline void finish_using_replay_mmio_load()
{
    // State at the end of this routine, once replay is idle:
    //    o Both Mutexes == 0
    //    o replay_instr_bank_write_id == replay_instr_bank_read_id
    // No additional guard or synchronization is needed before the next REPLAY-load
    wait_replay_idle();
    replay_mmap[mutex_0_unbanked] = 0;
    replay_mmap[mutex_1_unbanked] = 0;
    if (replay_mmap[replay_write_id] != replay_mmap[replay_read_id])
    {
        TTI_REPLAY(0, 1, 1, 0, 0, 1); // Load a NOP, use done=1 to flip replay_instrn_bank_write_id
        TTI_NOP;
    }
}

} // namespace ckernel
