/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include "risc_attribs.h"

// Compiler hint that a branch is unlikely to be taken
#define UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)
#define UNROLL_LOOP(factor) GCC unroll factor

#ifndef EN_DEST_DOUBLE_BUFFERING
#define EN_DEST_DOUBLE_BUFFERING 1
#endif

#ifndef LOCAL_MEM_EN
#define LOCAL_MEM_EN 0
#endif

#ifndef GPR_DEBUG_TTI
#define GPR_DEBUG_TTI 0
#endif

#ifndef GPR_DEBUG_REGFILE
#define GPR_DEBUG_REGFILE 0
#endif

#include <cstdint>

#include "ckernel_include.h"
#include "debug/fw_debug.h"
#include "tensix.h"
#include "eth_l1_address_map.h"
#include "noc_overlay_parameters.h"
#include "stream_io_map.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "limits.h"
// #include <cstring>
//#include "perf_lib/scratch_api.h" // not used unless perf dump enabled?


namespace ckernel
{

#define get_compile_time_arg_val(arg_idx) KERNEL_COMPILE_TIME_ARG_ ## arg_idx

constexpr uint PACK_FLUSH_COUNTERS = // counters flush
    (1 << PACK_COUNTERS_SEC2_pack_per_xy_plane_SHAMT) |
    (1 << PACK_COUNTERS_SEC2_pack_reads_per_xy_plane_SHAMT) |
    (1 << PACK_COUNTERS_SEC2_pack_xys_per_tile_SHAMT);

extern volatile uint tt_reg_ptr * const reg_base;
extern volatile uint tt_reg_ptr * const pc_buf_base;
extern volatile uint tt_reg_ptr * const regfile;
extern uint tt_reg_ptr * regmem;
extern volatile uint tt_reg_ptr * const instrn_buffer;
extern volatile uint tt_reg_ptr *dbg_event_scratch;

extern uint32_t cfg_state_id;
extern uint32_t dest_offset_id;
extern uint32_t dbg_event_index;
extern uint32_t dbg_event_end;

extern uint32_t op_info_offset;
// Internal scope to namespace methods only (C++ does not allow namespace private ownership)
namespace internal {
}

void tensix_sync();
void mop_sync();

inline void sync_regfile_write(const uint index);

// Field value overflow check
template<typename T>
static constexpr bool is_valid(const T val, const uint8_t wid)
{
	const T mask = (1 << wid) - 1;
	return (val & mask) == val;
}

inline void mmio_register_write(register_space_e space, uint addr, uint data)
{
    const uint regaddr = (space << 6) | (addr & 0x3F);
    reg_base[regaddr] = data;
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

// Tensix thread semaphore post optionally stalled
template <uint WaitRes = p_stall::NONE>
inline void t6_semaphore_post(const uint8_t index)
{
    if constexpr (WaitRes != p_stall::NONE)
        TTI_STALLWAIT(p_stall::STALL_SYNC, WaitRes);

    TTI_SEMPOST(semaphore::t6_sem(index));
}

// Tensix thread semaphore get optionally stalled
template <uint WaitRes = p_stall::NONE>
inline void t6_semaphore_get(const uint8_t index)
{
    if constexpr (WaitRes != p_stall::NONE)
        TTI_STALLWAIT(p_stall::STALL_SYNC, WaitRes);

    TTI_SEMGET(semaphore::t6_sem(index));
}

// Tensix thread semaphore get optionally stalled
inline void t6_semaphore_init(const uint8_t index, const uint8_t min_value, const uint8_t max_value)
{
    TTI_SEMINIT(max_value, min_value, semaphore::t6_sem(index));
}

inline void t6_mutex_acquire(const uint8_t index)
{
    TTI_ATGETM(index);
}

inline void t6_mutex_release(const uint8_t index)
{
    TTI_ATRELM(index);
}

// Return address of the current state ID register
inline uint cfg_addr(uint cfg_addr32)
{
    return (cfg_state_id == 0) ? cfg_addr32 : (CFG_STATE_SIZE * 4) + cfg_addr32;
}

inline void cfg_write(uint cfg_addr32, uint data)
{
    // Declared here instead of globally to prevent direct access, which might ignore current state ID
    volatile uint tt_reg_ptr *cfg_regs = reinterpret_cast<volatile uint tt_reg_ptr *>(TENSIX_CFG_BASE);
    cfg_regs[cfg_addr(cfg_addr32)] = data;
}

inline uint cfg_read(uint cfg_addr32)
{
    // Declared here instead of globally to prevent direct access, which might ignore current state ID
    volatile uint32_t tt_reg_ptr *cfg_regs = reinterpret_cast<volatile uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
    return cfg_regs[cfg_addr(cfg_addr32)];
}

// Return pointer to CFG with the right base address for the current state
inline volatile uint * tt_reg_ptr get_cfg_pointer()
{
    if (cfg_state_id == 0)
        return reinterpret_cast<volatile uint tt_reg_ptr *>(TENSIX_CFG_BASE);

    return reinterpret_cast<volatile uint tt_reg_ptr *>(TENSIX_CFG_BASE + CFG_STATE_SIZE * 16);
}

inline volatile uint short * tt_reg_ptr get_cfg16_pointer()
{
    if (cfg_state_id == 0)
        return reinterpret_cast<volatile uint short tt_reg_ptr *>(TENSIX_CFG_BASE);

    return reinterpret_cast<volatile uint short tt_reg_ptr *>(TENSIX_CFG_BASE + CFG_STATE_SIZE * 16);
}

inline void flip_cfg_state_id()
{
    cfg_state_id = 1 - cfg_state_id;
    TT_SETC16(CFG_STATE_ID_StateID_ADDR32, cfg_state_id); // Program the current state ID
    TTI_NOP;
    TTI_NOP;
}

inline void reset_cfg_state_id()
{
    cfg_state_id = 0;
}

inline void reset_dest_offset_id()
{
    dest_offset_id = 0;
}

// MOP run version without zmask
inline void mop_run(const uint8_t type, const uint8_t count)
{
    TTI_MOP(type, count - 1, 0); // Run the MOP
}

inline __attribute__((always_inline)) uint32_t reg_read(uint32_t addr)
{
    volatile uint tt_reg_ptr *p_reg = reinterpret_cast<volatile uint tt_reg_ptr *> (addr);
    return p_reg[0];
}

inline void reg_write(uint32_t addr, uint32_t data)
{
    volatile uint tt_reg_ptr *p_reg = reinterpret_cast<volatile uint tt_reg_ptr *> (addr);
    p_reg[0] = data;
}

inline void wait(uint32_t cycles) {
    volatile uint tt_reg_ptr * clock_lo = reinterpret_cast<volatile uint tt_reg_ptr * >(RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile uint tt_reg_ptr * clock_hi = reinterpret_cast<volatile uint tt_reg_ptr * >(RISCV_DEBUG_REG_WALL_CLOCK_H);
    uint64_t wall_clock_timestamp = clock_lo[0] | ((uint64_t)clock_hi[0]<<32);
    uint64_t wall_clock = 0;
    do {
       wall_clock = clock_lo[0] | ((uint64_t)clock_hi[0]<<32);
    }
    while (wall_clock < (wall_clock_timestamp+cycles));
}

// Clear dest
inline void zeroacc() {
    // Clear dest
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }.set(ADDR_MOD_1);
    TT_ZEROACC(p_zeroacc::CLR_ALL, ADDR_MOD_1, 0);
}

inline void zerosrc() {
    TTI_ZEROSRC(0,0,1,3); // Zero all srcA&B banks
}

inline void sync_regfile_write(const uint index)
{
    volatile uint foo = 0x0;
    volatile uint *fooptr = &foo;
    *fooptr = regfile[index];
}

inline void cfg_rmw(uint32_t cfg_addr32, uint32_t cfg_shamt, uint32_t cfg_mask, uint32_t val)
{
    uint32_t wrdata = val;

    // Avoid multiplication of variables!
    //const uint32_t addr = (cfg_state_id * CFG_STATE_SIZE * 4) + cfg_addr32;
    const uint32_t addr = (cfg_state_id == 0) ? cfg_addr32 : (CFG_STATE_SIZE * 4) + cfg_addr32;

    // Declared here instead of globally to prevent direct access, which might ignore current state ID
    volatile uint tt_reg_ptr *cfg_regs = reinterpret_cast<volatile uint tt_reg_ptr *>(TENSIX_CFG_BASE);
    uint32_t cfg_data = cfg_regs[addr];

    // Shift and mask wrdata to properly align withn 32-bit DWORD
    wrdata <<= cfg_shamt;
    wrdata &= cfg_mask;

    // Zero-out relevant bits in cfg data
    cfg_data &= ~cfg_mask;

    // Or new data bits
    cfg_data |= wrdata;

    //Update cfg regs
    cfg_regs[addr] = cfg_data;
}

inline void cfg_rmw_gpr(uint32_t cfg_addr32, uint32_t cfg_shamt, uint32_t cfg_mask, uint32_t gpr_index)
{
    const uint32_t wrdata = regfile[gpr_index];
    cfg_rmw(cfg_addr32, cfg_shamt, cfg_mask, wrdata);
}

template <uint CfgAddr32, uint Shamt, uint Mask>
inline void cfg_reg_rmw_tensix(uint32_t val)
{
    uint32_t wrdata = val<<Shamt;
    uint8_t mask_b0 = Mask & 0xff;

    if (mask_b0!=0){
        uint8_t data_b0 = wrdata & 0xff;
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
    }
    wrdata>>=8;
    uint8_t mask_b1 = (Mask>>8) & 0xff;

    if (mask_b1!=0){
        uint8_t data_b1 = (wrdata) & 0xff;
        TT_RMWCIB1(mask_b1, data_b1, CfgAddr32);
    }

    wrdata>>=8;
    uint8_t mask_b2 = (Mask>>16) & 0xff;

    if (mask_b2!=0){
        uint8_t data_b2 = (wrdata) & 0xff;
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
    }

    wrdata>>=8;
    uint8_t mask_b3 = (Mask>>24) & 0xff;
    if (mask_b3!=0){
        uint8_t data_b3 = (wrdata) & 0xff;
        TT_RMWCIB3(mask_b3, data_b3, CfgAddr32);
    }
}

template <class T>
inline std::uint32_t memory_cast(T *object_ptr)
{
    return reinterpret_cast<uint32_t>(object_ptr);
}

inline uint64_t read_wall_clock()
{
   uint32_t timestamp_low = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
   uint32_t timestamp_high = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
   return ((uint64_t)timestamp_high << 32) | timestamp_low;
}

void debug_dump(const uint8_t *data, uint32_t byte_size);
void debug_dump_seek(uint8_t offset);


inline __attribute__((always_inline)) unsigned int mulsi3 (unsigned int a, unsigned int b)
{
  unsigned int r = 0;
  while (a)
    {
      if (a & 1)
        r += b;
      a >>= 1;
      b <<= 1;
    }
  return r;
}

inline __attribute__((always_inline)) uint32_t fast_udiv_12(uint32_t n)
{
    // Uses embedding style magic number
    // * fixed point 1/12 then shifting.
    // https://web.archive.org/web/20190703172151/http://www.hackersdelight.org/magic.htm
    return (((uint64_t) n * 0xAAAAAAAB) >> 32) >> 3;
}

inline __attribute__((always_inline)) uint32_t fast_udiv_94(uint32_t n)
{
    // Uses embedding style magic number
    // * fixed point 1/12 then shifting.
    // https://web.archive.org/web/20190703172151/http://www.hackersdelight.org/magic.htm
    return (((uint64_t) n * 0xAE4C415D) >> 32) >> 6;
}

template <uint32_t d>
inline __attribute__((always_inline)) uint32_t udivsi3_const_divisor(uint32_t n)
{
    if constexpr (d == 12) {
        // fast divide for 12 divisor
        return fast_udiv_12(n);
    } else if constexpr (d == 94) {
        // fast divide for 94 divisor. Handles Banked L1 address generation for E75
        return fast_udiv_94(n);
    } else {
        // generic divide from llvm
        const unsigned n_uword_bits = sizeof(uint32_t) * CHAR_BIT;
        unsigned int q;
        unsigned int r;
        unsigned sr;
        /* special cases */
        if (d == 0)
            return 0; /* ?! */
        if (n == 0)
            return 0;
        sr = __builtin_clz(d) - __builtin_clz(n);
        /* 0 <= sr <= n_uword_bits - 1 or sr large */
        if (sr > n_uword_bits - 1)  /* d > r */
            return 0;
        if (sr == n_uword_bits - 1)  /* d == 1 */
            return n;
        ++sr;
        /* 1 <= sr <= n_uword_bits - 1 */
        /* Not a special case */
        q = n << (n_uword_bits - sr);
        r = n >> sr;
        unsigned int  carry = 0;
        for (; sr > 0; --sr)
        {
            /* r:q = ((r:q)  << 1) | carry */
            r = (r << 1) | (q >> (n_uword_bits - 1));
            q = (q << 1) | carry;
            /* carry = 0;
             * if (r.all >= d.all)
             * {
             *      r.all -= d.all;
             *      carry = 1;
             * }
             */
            const int s = (unsigned int)(d - r - 1) >> (n_uword_bits - 1);
            carry = s & 1;
            r -= d & s;
        }
        q = (q << 1) | carry;
        return q;
    }
}
template <uint32_t d>
inline __attribute__((always_inline)) uint32_t umodsi3_const_divisor(uint32_t a)
{
    return a - udivsi3_const_divisor<d>(a) * d;
}

inline void tensix_sync()
{
    volatile uint foo = 0x0;
    volatile uint *fooptr = &foo;
    // Write to pc buffer to push all writes ahead of us.. otherwise, the pc buffer read can bypass older writes
    pc_buf_base[1] = foo;

    // Now read -- this read will block until we're idle
    *fooptr = pc_buf_base[1];
}

inline void mop_sync()
{
    volatile uint foo = 0x0;
    volatile uint *fooptr = &foo;
    // Write to pc buffer to push all writes ahead of us.. otherwise, the pc buffer read can bypass older writes
    pc_buf_base[2] = foo;

    // Now read -- this read will block until mops are done
    *fooptr = pc_buf_base[2];
}

inline void llk_get_next_op_info(tt::op_info_t& op_info_struct) {

    uint32_t* op_info_ptr = reinterpret_cast<uint32_t*>(OP_INFO_BASE_ADDR + op_info_offset);
    static constexpr uint32_t op_info_num_items = 7;

    volatile uint32_t* op_info_struct_ptr = reinterpret_cast<volatile uint32_t*>(&op_info_struct);
    for (uint32_t i = 0; i < op_info_num_items; i++) {
        op_info_struct_ptr[i] = op_info_ptr[i];
    }
    op_info_offset += 28;

    if (op_info_offset == OP_INFO_SIZE) {
        op_info_offset = 0; // In case we go out of bounds
    }
}

}
