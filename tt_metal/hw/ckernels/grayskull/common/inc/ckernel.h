// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#pragma once

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

constexpr uint PACK_FLUSH_COUNTERS = // counters flush
    (1 << PACK_COUNTERS_SEC2_pack_per_xy_plane_SHAMT) |
    (1 << PACK_COUNTERS_SEC2_pack_reads_per_xy_plane_SHAMT) |
    (1 << PACK_COUNTERS_SEC2_pack_xys_per_tile_SHAMT);

extern volatile uint * reg_base;
extern volatile uint * pc_buf_base;
extern volatile uint * regfile;
extern uint *regmem;
extern volatile uint * instrn_buffer;
extern volatile uint *dbg_event_scratch;
extern volatile uint local_mem_barrier;

extern uint32_t cfg_state_id;
extern uint32_t dest_offset_id;
extern uint32_t dbg_event_index;
extern uint32_t dbg_event_end;

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
    volatile uint *cfg_regs = reinterpret_cast<volatile uint *>(TENSIX_CFG_BASE);
    cfg_regs[cfg_addr(cfg_addr32)] = data;
}

//inline uint cfg_read(uint cfg_addr32)
//{
//    // Declared here instead of globally to prevent direct access, which might ignore current state ID
//    volatile uint *cfg_regs = reinterpret_cast<volatile uint *>(TENSIX_CFG_BASE);
//    return cfg_regs[cfg_addr(cfg_addr32)];
//}

inline uint cfg_read(uint cfg_addr32)
{
    // Declared here instead of globally to prevent direct access, which might ignore current state ID
    volatile uint32_t tt_reg_ptr *cfg_regs = reinterpret_cast<volatile uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
    return cfg_regs[cfg_addr(cfg_addr32)];
}

// Return pointer to CFG with the right base address for the current state
inline volatile uint *get_cfg_pointer()
{
    if (cfg_state_id == 0)
        return reinterpret_cast<volatile uint *>(TENSIX_CFG_BASE);

    return reinterpret_cast<volatile uint *>(TENSIX_CFG_BASE + CFG_STATE_SIZE * 16);
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

// MOP run version without zmask
inline void mop_run(const uint8_t type, const uint8_t count)
{
    TTI_MOP(type, count - 1, 0); // Run the MOP
}

inline __attribute__((always_inline)) uint32_t reg_read(uint32_t addr)
{
    volatile uint32_t tt_reg_ptr *p_reg = reinterpret_cast<volatile uint32_t tt_reg_ptr *> (addr);
    return p_reg[0];
}


inline void reg_write(uint32_t addr, uint32_t data)
{
    volatile uint *p_reg = reinterpret_cast<volatile uint *> (addr);
    p_reg[0] = data;
}

inline void wait(uint32_t cycles) {
    volatile uint * clock_lo = reinterpret_cast<volatile uint * >(RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile uint * clock_hi = reinterpret_cast<volatile uint * >(RISCV_DEBUG_REG_WALL_CLOCK_H);
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
    volatile uint foo = 0;
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
    volatile uint32_t tt_reg_ptr *cfg_regs = reinterpret_cast<volatile uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
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

template <class T>
inline std::uint32_t memory_cast(T *object_ptr)
{
    return reinterpret_cast<uint32_t>(object_ptr);
}

inline void debug_dump(uint8_t *data, uint32_t byte_size) {
    // TODO(pk) re-implement
}

inline void tensix_sync()
{
    volatile uint foo = 0;
    volatile uint *fooptr = &foo;
    // Write to pc buffer to push all writes ahead of us.. otherwise, the pc buffer read can bypass older writes
    pc_buf_base[1] = foo;

    // Now read -- this read will block until we're idle
    *fooptr = pc_buf_base[1];
}

inline void mop_sync()
{
    volatile uint foo = 0;
    volatile uint *fooptr = &foo;
    // Write to pc buffer to push all writes ahead of us.. otherwise, the pc buffer read can bypass older writes
    pc_buf_base[2] = foo;

    // Now read -- this read will block until mops are done
    *fooptr = pc_buf_base[2];
}

}
