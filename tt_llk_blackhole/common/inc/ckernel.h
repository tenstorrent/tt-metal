// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "llk_defs.h"
#include "risc_attribs.h"

// MT: This should be dissolved and moved to the appropriate place
#include "tensix.h"

// This header is included on non-trisc builds, for reasons
// unknown. lltt is only available on trisc
#if defined(COMPILE_FOR_TRISC)
#include <utility>

#include "lltt.h"
#endif

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

#ifdef PERF_DUMP
#define DECOUPLINGS_EN (SKIP_UNP || MATH_PACK_DECOUPLE)
#else
#define SKIP_UNP           0
#define MATH_PACK_DECOUPLE 0
#define DECOUPLINGS_EN     0
#define OVERLAY_DECOUPLE   0
#endif

#if defined(EN_KERNEL_SLOWDOWN)
#include "kernel_slowdown_config.h"
#endif

#ifndef INSERT_UNPACK_DELAY
#define INSERT_UNPACK_DELAY 0
#endif

#ifndef INSERT_MATH_DELAY
#define INSERT_MATH_DELAY 0
#endif

#ifndef INSERT_PACK_DELAY
#define INSERT_PACK_DELAY 0
#endif

#define DELAY_EN (INSERT_UNPACK_DELAY || INSERT_PACK_DELAY || INSERT_MATH_DELAY)

#define TT_ALWAYS_INLINE inline __attribute__((always_inline))

#include <cstdint>

#include "ckernel_include.h"
#include "fw_debug.h"

// #include <cstring>
#if defined(PERF_DUMP) || DELAY_EN > 0
#include <l1_address_map.h>

#include "perf_lib/scratch_api.h"
#endif

namespace ckernel
{

constexpr uint PACK_FLUSH_COUNTERS = // counters flush
    (1 << PACK_COUNTERS_SEC2_pack_per_xy_plane_SHAMT) | (1 << PACK_COUNTERS_SEC2_pack_reads_per_xy_plane_SHAMT) |
    (1 << PACK_COUNTERS_SEC2_pack_xys_per_tile_SHAMT);

constexpr uint RESET_VAL          = 0;
constexpr uint KERNEL_IN_PROGRESS = 15;
constexpr uint KERNEL_COMPLETE    = 1;

extern volatile uint tt_reg_ptr *reg_base;
extern volatile uint tt_reg_ptr *pc_buf_base;
extern volatile uint tt_reg_ptr *regfile;
extern volatile uint tt_reg_ptr *instrn_buffer;
extern volatile uint tt_reg_ptr *mailbox_base[4];
extern volatile uint tt_reg_ptr *dbg_event_scratch;
extern volatile uint tt_reg_ptr *trisc_l1_mailbox;
extern volatile uint8_t tt_l1_ptr *debug_buffer;

extern uint32_t cfg_state_id;
extern uint32_t dest_offset_id;
extern uint32_t dbg_event_index;
extern uint32_t dbg_event_end;

extern volatile uint16_t tt_reg_ptr *debug_mailbox_base;
extern uint8_t mailbox_index;
const extern uint8_t mailbox_end;

// Internal scope to namespace methods only (C++ does not allow namespace private ownership)
namespace internal
{
}

inline void tensix_sync()
{
    volatile uint foo     = 0;
    volatile uint *fooptr = &foo;
    // Write to pc buffer to push all writes ahead of us.. otherwise, the pc buffer read can bypass older writes
    pc_buf_base[1] = foo;

    // Now read -- this read will block until we're idle
    *fooptr = pc_buf_base[1];
}

inline void mop_sync()
{
    volatile uint foo     = 0;
    volatile uint *fooptr = &foo;
    // Write to pc buffer to push all writes ahead of us.. otherwise, the pc buffer read can bypass older writes
    pc_buf_base[2] = foo;

    // Now read -- this read will block until mops are done
    *fooptr = pc_buf_base[2];
}

inline void sync_regfile_write(const uint index);

// Field value overflow check
template <typename T>
static constexpr bool is_valid(const T val, const uint8_t wid)
{
    const T mask = (1 << wid) - 1;
    return (val & mask) == val;
}

inline void mmio_register_write(register_space_e space, uint addr, uint data)
{
    const uint regaddr = (space << 6) | (addr & 0x3F);
    // FWLOG2("Regaddr: 0x%x, data: 0x%x", regaddr, data);
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
    {
        TTI_STALLWAIT(p_stall::STALL_SYNC, WaitRes);
    }

    TTI_SEMPOST(semaphore::t6_sem(index));
}

// Tensix thread semaphore get optionally stalled
template <uint WaitRes = p_stall::NONE>
inline void t6_semaphore_get(const uint8_t index)
{
    if constexpr (WaitRes != p_stall::NONE)
    {
        TTI_STALLWAIT(p_stall::STALL_SYNC, WaitRes);
    }

    TTI_SEMGET(semaphore::t6_sem(index));
}

template <uint WaitRes>
inline void t6_semaphore_wait_on_max(const uint8_t index)
{
    TTI_SEMWAIT(WaitRes, semaphore::t6_sem(index), p_stall::STALL_ON_MAX);
}

template <uint WaitRes>
inline void t6_semaphore_wait_on_zero(const uint8_t index)
{
    TTI_SEMWAIT(WaitRes, semaphore::t6_sem(index), p_stall::STALL_ON_ZERO);
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
    cfg_regs[cfg_addr(cfg_addr32)]     = data;
}

inline uint cfg_read(uint cfg_addr32)
{
    // Declared here instead of globally to prevent direct access, which might ignore current state ID
    volatile uint *cfg_regs = reinterpret_cast<volatile uint *>(TENSIX_CFG_BASE);
    return cfg_regs[cfg_addr(cfg_addr32)];
}

// Return pointer to CFG with the right base address for the current state
inline volatile uint *tt_reg_ptr get_cfg_pointer()
{
    if (cfg_state_id == 0)
    {
        return reinterpret_cast<volatile uint tt_reg_ptr *>(TENSIX_CFG_BASE);
    }

    return reinterpret_cast<volatile uint tt_reg_ptr *>(TENSIX_CFG_BASE + CFG_STATE_SIZE * 16);
}

inline volatile uint short *tt_reg_ptr get_cfg16_pointer()
{
    if (cfg_state_id == 0)
    {
        return reinterpret_cast<volatile uint short tt_reg_ptr *>(TENSIX_CFG_BASE);
    }

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

inline void update_dest_offset_id()
{
    // ping-pong between 0 and 1
    dest_offset_id = 1 - dest_offset_id;
}

inline uint32_t get_dest_buffer_base()
{
    return (0 != dest_offset_id) ? DEST_REGISTER_HALF_SIZE : 0x0;
}

// MOP run version without zmask
inline void mop_run(const uint8_t type, const uint8_t count)
{
    TTI_MOP(type, count - 1, 0); // Run the MOP
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

inline void wait(uint32_t cycles)
{
    volatile uint tt_reg_ptr *clock_lo = reinterpret_cast<volatile uint tt_reg_ptr *>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile uint tt_reg_ptr *clock_hi = reinterpret_cast<volatile uint tt_reg_ptr *>(RISCV_DEBUG_REG_WALL_CLOCK_H);
    uint64_t wall_clock_timestamp      = clock_lo[0] | ((uint64_t)clock_hi[0] << 32);
    uint64_t wall_clock                = 0;
    do
    {
        wall_clock = clock_lo[0] | ((uint64_t)clock_hi[0] << 32);
    } while (wall_clock < (wall_clock_timestamp + cycles));
}

// Clear dest
inline void zeroacc()
{
    // Clear dest
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_1);
    TT_ZEROACC(p_zeroacc::CLR_ALL, 0, 0, ADDR_MOD_1, 0);
}

inline void zerosrc()
{
    TTI_ZEROSRC(0, 0, 1, 3); // Zero all srcA&B banks
}

inline void sync_regfile_write(const uint index)
{
    volatile uint foo     = 0x0;
    volatile uint *fooptr = &foo;
    *fooptr               = regfile[index];
}

inline void cfg_rmw(uint32_t cfg_addr32, uint32_t cfg_shamt, uint32_t cfg_mask, uint32_t val)
{
    uint32_t wrdata = val;

    // Avoid multiplication of variables!
    // const uint32_t addr = (cfg_state_id * CFG_STATE_SIZE * 4) + cfg_addr32;
    const uint32_t addr = (cfg_state_id == 0) ? cfg_addr32 : (CFG_STATE_SIZE * 4) + cfg_addr32;

    // Declared here instead of globally to prevent direct access, which might ignore current state ID
    volatile uint tt_reg_ptr *cfg_regs = reinterpret_cast<volatile uint tt_reg_ptr *>(TENSIX_CFG_BASE);
    uint32_t cfg_data                  = cfg_regs[addr];

    // Shift and mask wrdata to properly align withn 32-bit DWORD
    wrdata <<= cfg_shamt;
    wrdata &= cfg_mask;

    // Zero-out relevant bits in cfg data
    cfg_data &= ~cfg_mask;

    // Or new data bits
    cfg_data |= wrdata;

    // Update cfg regs
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
    uint32_t wrdata = val << Shamt;
    uint8_t mask_b0 = Mask & 0xff;

    if (mask_b0 != 0)
    {
        uint8_t data_b0 = wrdata & 0xff;
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
    }
    wrdata >>= 8;
    uint8_t mask_b1 = (Mask >> 8) & 0xff;

    if (mask_b1 != 0)
    {
        uint8_t data_b1 = (wrdata) & 0xff;
        TT_RMWCIB1(mask_b1, data_b1, CfgAddr32);
    }

    wrdata >>= 8;
    uint8_t mask_b2 = (Mask >> 16) & 0xff;

    if (mask_b2 != 0)
    {
        uint8_t data_b2 = (wrdata) & 0xff;
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
    }

    wrdata >>= 8;
    uint8_t mask_b3 = (Mask >> 24) & 0xff;
    if (mask_b3 != 0)
    {
        uint8_t data_b3 = (wrdata) & 0xff;
        TT_RMWCIB3(mask_b3, data_b3, CfgAddr32);
    }
}

inline void mailbox_write(const uint8_t thread, const uint32_t data)
{
    mailbox_base[thread + 1][0] = data;
}

// Blocking read
inline uint32_t mailbox_read(const uint8_t thread)
{
    return mailbox_base[thread + 1][0];
}

inline bool mailbox_not_empty(const uint8_t thread)
{
    return mailbox_base[thread + 1][1] > 0;
}

inline void mailbox_write_full(const uint8_t thread, const uint32_t data)
{
    mailbox_base[thread][0] = data;
}

// Blocking read
inline uint32_t mailbox_read_full(const uint8_t thread)
{
    return mailbox_base[thread][0];
}

inline bool mailbox_not_empty_full(const uint8_t thread)
{
    return mailbox_base[thread][1] > 0;
}

inline void trisc_l1_mailbox_write(const uint data)
{
    trisc_l1_mailbox[0] = data;
}

inline uint trisc_l1_mailbox_read()
{
    return trisc_l1_mailbox[0];
}

template <class T>
inline std::uint32_t memory_cast(T *object_ptr)
{
    return reinterpret_cast<uint32_t>(object_ptr);
}

inline void record_mailbox_value(uint16_t event_value)
{
    if (mailbox_index < mailbox_end)
    {
        debug_mailbox_base[mailbox_index] = event_value;
        mailbox_index++;
    }
}

inline void record_mailbox_value_with_index(uint8_t index, uint16_t event_value)
{
    if (index < mailbox_end)
    {
        debug_mailbox_base[index] = event_value;
    }
}

// Initialize debug scratch mailbox values and range
inline void clear_mailbox_values(uint16_t value = 0)
{
    for (int i = 0; i < mailbox_end; i++)
    {
        debug_mailbox_base[i] = value;
    }
}

inline uint64_t read_wall_clock()
{
    uint32_t timestamp_low  = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t timestamp_high = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
    return ((uint64_t)timestamp_high << 32) | timestamp_low;
}

inline void record_kernel_runtime(uint64_t kernel_runtime)
{
    debug_mailbox_base[mailbox_end - 4] = kernel_runtime & 0xffff;
    debug_mailbox_base[mailbox_end - 3] = (kernel_runtime >> 16) & 0xffff;
    debug_mailbox_base[mailbox_end - 2] = (kernel_runtime >> 32) & 0xffff;
    debug_mailbox_base[mailbox_end - 1] = (kernel_runtime >> 48) & 0xffff;
}

void debug_dump(const uint8_t *data, uint32_t byte_size);
void debug_dump_seek(uint8_t offset);

inline void stall_kernel(uint32_t num_cycles)
{
#if DELAY_EN > 0
    TT_LLK_DUMP("stall_kernel({})", num_cycles);
    uint32_t start_clk_l  = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t elapsed_time = 0;
    while (elapsed_time <= num_cycles)
    {
        uint32_t current_clk_l = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        if (current_clk_l >= start_clk_l)
        {
            elapsed_time = current_clk_l - start_clk_l;
        }
        else
        {
            elapsed_time = 0xffffffff - (start_clk_l - current_clk_l);
        }
    }
#endif
}

#if defined(PERF_DUMP) || DELAY_EN > 0
extern bool record_perf_events;
#endif

// This api is inserted in the beginning of each input loop
// Wait for all instructions of previous loop to finish before starting the next loop
// If PERF_DUMP is enabled, always wait but only for the inputs that perf dump is enabled for
// If PERF_DUMP is enabled, and delay is not, no need to insert these apis for unpack and math
template <int thread_id>
inline void serialize_input_loop_start()
{
#if defined(PERF_DUMP) || DELAY_EN > 0
    TT_LLK_DUMP("serialize_input_loop_start<{}>()", thread_id);
    if constexpr (thread_id == 0)
    {
#if DELAY_EN > 0
        t6_semaphore_post(semaphore::UNPACK_MATH_DONE);
        while (semaphore_read(semaphore::UNPACK_MATH_DONE) == 0)
        {
        }
#endif
    }
    else if (thread_id == 1)
    {
#if DELAY_EN > 0
        t6_semaphore_post(semaphore::UNPACK_MATH_DONE);
        while (semaphore_read(semaphore::UNPACK_MATH_DONE) == 0)
        {
        }
#endif
    }
    else if (thread_id == 2)
    {
#if DELAY_EN == 0
        if (record_perf_events)
        {
#endif
            t6_semaphore_post(semaphore::PACK_DONE);
            while (semaphore_read(semaphore::PACK_DONE) == 0)
            {
            }
#if DELAY_EN == 0
        }
#endif
    }
#endif
}

template <int thread_id>
inline void serialize_input_loop_end()
{
#if defined(PERF_DUMP) || DELAY_EN > 0
    TT_LLK_DUMP("serialize_input_loop_end<{}>()", thread_id);
    if constexpr (thread_id == 0)
    {
#if DELAY_EN > 0
        t6_semaphore_get<p_stall::UNPACK>(semaphore::UNPACK_MATH_DONE);
        while (semaphore_read(semaphore::UNPACK_MATH_DONE) > 0)
        {
        }
#endif
    }
    else if (thread_id == 1)
    {
#if DELAY_EN > 0
        t6_semaphore_get<p_stall::MATH>(semaphore::UNPACK_MATH_DONE);
        while (semaphore_read(semaphore::UNPACK_MATH_DONE) > 0)
        {
        }
#endif
    }
    else if (thread_id == 2)
    {
#if DELAY_EN == 0
        if (record_perf_events)
        {
#endif
            t6_semaphore_get<p_stall::PACK>(semaphore::PACK_DONE);
            while (semaphore_read(semaphore::PACK_DONE) > 0)
            {
            }
#if DELAY_EN == 0
        }
#endif
    }
#endif
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

// Uses a template to guarantee compiletime execution (could probably
// get away with constexpr but this seems better)
template <uint bitmask>
inline void set_ttsync_enables()
{
    static_assert((bitmask & ~TRACK_ALL) == 0, "The given bitmask targets bits outside the allowable range");
    TTI_SETC16(TENSIX_TRISC_SYNC_TrackGlobalCfg_ADDR32, bitmask);
}

template <bool add_nops = true>
inline void disable_gathering()
{
    asm("csrrs zero, 0x7c0, %0" : : "r"(1 << 1));
    asm("fence");
    // Disable gathering: set bit 18
    asm("csrrs zero, 0x7c0, %0" : : "r"(1 << 18));
    asm("csrrc zero, 0x7c0, %0" : : "r"(1 << 1));
    asm("fence");

    // Gathering is done early in the pipeline, so we need to make sure
    // the above csrrw gets processed before the load-replay instructions
    if constexpr (add_nops)
    {
        TTI_NOP;
        TTI_NOP;
        TTI_NOP;
    }
}

inline void enable_gathering()
{
    // Enable gathering: clear bit 18
    asm("csrrc zero, 0x7c0, %0" : : "r"(1 << 18));
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
[[gnu::always_inline, gnu::flatten]] inline void load_replay_buf(uint start, uint len, Callable &&callable, Args &&...args)
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

enum class CSR : uint16_t
{
    tensix_queue_status        = 0xBC0,
    tensix_busy_status         = 0xBC1,
    stream_curr_phase_0        = 0xBC2,
    stream_curr_phase_1        = 0xBC3,
    stream_curr_phase_2        = 0xBC4,
    stream_curr_phase_3        = 0xBC5,
    stream_num_msgs_received_0 = 0xBC6,
    stream_num_msgs_received_1 = 0xBC7,
    stream_num_msgs_received_2 = 0xBC8,
    stream_num_msgs_received_3 = 0xBC9
};

template <CSR csr_num, bool fence = true>
inline uint32_t csr_read()
{
    uint32_t ret;

    if constexpr (fence)
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

    if constexpr (fence)
    {
        asm volatile("fence");
    }
    asm volatile("csrr %[ret], %[csr_num] \n" : [ret] "=r"(ret) : [csr_num] "i"(csr_num));

    return ret;
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

inline void init_prng_seed(const uint seed)
{
    // The seed for PRNG should at least be initialzied during chip bootup time.
    volatile uint tt_reg_ptr *cfg  = get_cfg_pointer();
    cfg[PRNG_SEED_Seed_Val_ADDR32] = seed;

    // TODO: ckernel::wait does not work properly. Use ckernel::wait when fixed.
    for (int i = 0; i < 600; i++)
    {
        TTI_SFPNOP;
    }
}

inline constexpr bool is_valid_instruction_mode(InstrModLoadStore mode)
{
    return mode == InstrModLoadStore::INT32_2S_COMP || mode == InstrModLoadStore::INT32 || mode == InstrModLoadStore::LO16;
}

inline void apply_sign_magnitude_conversion(uint src, uint dst, InstrModCast cast_mode)
{
    TTI_SFPCAST(src /*lreg*/, dst /*ldest*/, cast_mode);
    // Required after cast due to a bug in Blackhole RTL (Refer tenstorrent/tt-llk-bh#16)
    TTI_SFPSETSGN(0 /* imm */, dst /*lreg_c*/, src /*ldest*/, 0 /*imod*/);
}

} // namespace ckernel
