// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_common_ops.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "internal/risc_attribs.h"
#include "llk_assert.h"
#include "llk_defs.h"

// MT: This should be dissolved and moved to the appropriate place
#include "tensix.h"

// This header is included on non-trisc builds, for reasons
// unknown. lltt is only available on trisc
#if defined(COMPILE_FOR_TRISC)
#include <utility>

#include "lltt.h"
#endif

// compiler hints
#define LIKELY(condition)   __builtin_expect(static_cast<bool>(condition), 1)
#define UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)
#define UNREACHABLE()       __builtin_unreachable()

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

#define TT_ALWAYS_INLINE inline __attribute__((always_inline))

#include <cstdint>

#include "ckernel_include.h"

namespace ckernel
{

constexpr std::uint32_t PACK_FLUSH_COUNTERS = // counters flush
    (1 << PACK_COUNTERS_SEC2_pack_per_xy_plane_SHAMT) | (1 << PACK_COUNTERS_SEC2_pack_reads_per_xy_plane_SHAMT) |
    (1 << PACK_COUNTERS_SEC2_pack_xys_per_tile_SHAMT);

constexpr std::uint32_t RESET_VAL          = 0;
constexpr std::uint32_t KERNEL_IN_PROGRESS = 15;
constexpr std::uint32_t KERNEL_COMPLETE    = 0xFF;

extern volatile std::uint32_t tt_reg_ptr *reg_base;
extern volatile std::uint32_t tt_reg_ptr *pc_buf_base;
extern volatile std::uint32_t tt_reg_ptr *regfile;
} // namespace ckernel

extern volatile std::uint32_t __instrn_buffer[];

namespace ckernel
{
constexpr inline volatile std::uint32_t(tt_reg_ptr &instrn_buffer)[] = __instrn_buffer;
extern volatile std::uint32_t tt_reg_ptr *mailbox_base[4];

extern std::uint32_t cfg_state_id;
extern std::uint32_t dest_offset_id;
extern std::uint32_t dbg_event_index;
extern std::uint32_t dbg_event_end;

extern volatile std::uint16_t tt_reg_ptr *debug_mailbox_base;
extern std::uint8_t mailbox_index;
const extern std::uint8_t mailbox_end;

// Internal scope to namespace methods only (C++ does not allow namespace private ownership)
namespace internal
{
}

inline void tensix_sync()
{
    volatile std::uint32_t foo     = 0;
    volatile std::uint32_t *fooptr = &foo;
    // Write to pc buffer to push all writes ahead of us.. otherwise, the pc buffer read can bypass older writes
    pc_buf_base[1] = foo;

    // Now read -- this read will block until we're idle
    *fooptr = pc_buf_base[1];
}

inline void mop_sync()
{
    volatile std::uint32_t foo     = 0;
    volatile std::uint32_t *fooptr = &foo;
    // Write to pc buffer to push all writes ahead of us.. otherwise, the pc buffer read can bypass older writes
    pc_buf_base[2] = foo;

    // Now read -- this read will block until mops are done
    *fooptr = pc_buf_base[2];
}

inline void sync_regfile_write(const std::uint32_t index);

// Field value overflow check
template <typename T>
static constexpr bool is_valid(const T val, const std::uint8_t wid)
{
    const T mask = (1 << wid) - 1;
    return (val & mask) == val;
}

inline void mmio_register_write(register_space_e space, std::uint32_t addr, std::uint32_t data)
{
    const std::uint32_t regaddr = (space << 6) | (addr & 0x3F);
    reg_base[regaddr]           = data;
}

inline std::uint8_t semaphore_read(const std::uint8_t index)
{
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
}

inline void semaphore_post(const std::uint8_t index)
{
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0;
}

inline void semaphore_get(const std::uint8_t index)
{
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1;
}

// Tensix thread semaphore post optionally stalled
template <std::uint32_t WaitRes = p_stall::NONE>
inline void t6_semaphore_post(const std::uint8_t index)
{
    if constexpr (WaitRes != p_stall::NONE)
    {
        TTI_STALLWAIT(p_stall::STALL_SYNC, WaitRes);
    }

    TTI_SEMPOST(semaphore::t6_sem(index));
}

// Tensix thread semaphore get optionally stalled
template <std::uint32_t WaitRes = p_stall::NONE>
inline void t6_semaphore_get(const std::uint8_t index)
{
    if constexpr (WaitRes != p_stall::NONE)
    {
        TTI_STALLWAIT(p_stall::STALL_SYNC, WaitRes);
    }

    TTI_SEMGET(semaphore::t6_sem(index));
}

template <std::uint32_t WaitRes>
inline void t6_semaphore_wait_on_max(const std::uint8_t index)
{
    TTI_SEMWAIT(WaitRes, semaphore::t6_sem(index), p_stall::STALL_ON_MAX);
}

template <std::uint32_t WaitRes>
inline void t6_semaphore_wait_on_zero(const std::uint8_t index)
{
    TTI_SEMWAIT(WaitRes, semaphore::t6_sem(index), p_stall::STALL_ON_ZERO);
}

// Tensix thread semaphore get optionally stalled
inline void t6_semaphore_init(const std::uint8_t index, const std::uint8_t min_value, const std::uint8_t max_value)
{
    TTI_SEMINIT(max_value, min_value, semaphore::t6_sem(index));
}

inline void t6_mutex_acquire(const std::uint8_t index)
{
    TTI_ATGETM(index);
}

inline void t6_mutex_release(const std::uint8_t index)
{
    TTI_ATRELM(index);
}

// Return address of the current state ID register
inline std::uint32_t cfg_addr(std::uint32_t cfg_addr32)
{
    return (cfg_state_id == 0) ? cfg_addr32 : (CFG_STATE_SIZE * 4) + cfg_addr32;
}

inline void cfg_write(std::uint32_t cfg_addr32, std::uint32_t data)
{
    // Declared here instead of globally to prevent direct access, which might ignore current state ID
    volatile std::uint32_t tt_reg_ptr *cfg_regs = reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
    cfg_regs[cfg_addr(cfg_addr32)]              = data;
}

inline std::uint32_t cfg_read(std::uint32_t cfg_addr32)
{
    // Declared here instead of globally to prevent direct access, which might ignore current state ID
    volatile std::uint32_t *cfg_regs = reinterpret_cast<volatile std::uint32_t *>(TENSIX_CFG_BASE);
    return cfg_regs[cfg_addr(cfg_addr32)];
}

// Return pointer to CFG with the right base address for the current state
inline volatile std::uint32_t *tt_reg_ptr get_cfg_pointer()
{
    if (cfg_state_id == 0)
    {
        return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
    }

    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE + CFG_STATE_SIZE * 16);
}

inline volatile std::uint32_t short *tt_reg_ptr get_cfg16_pointer()
{
    if (cfg_state_id == 0)
    {
        return reinterpret_cast<volatile std::uint32_t short tt_reg_ptr *>(TENSIX_CFG_BASE);
    }

    return reinterpret_cast<volatile std::uint32_t short tt_reg_ptr *>(TENSIX_CFG_BASE + CFG_STATE_SIZE * 16);
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

inline std::uint32_t get_dest_buffer_base()
{
    return (0 != dest_offset_id) ? DEST_REGISTER_HALF_SIZE : 0x0;
}

// MOP run version without zmask
inline void mop_run(const std::uint8_t type, const std::uint8_t count)
{
    TTI_MOP(type, count - 1, 0); // Run the MOP
}

// Register read (workaround for bug
// tenstorrent/tensix#976
// now handled by the compiler)
// workaround is needed only for GS
inline std::uint32_t reg_read(std::uint32_t addr)
{
    volatile std::uint32_t tt_reg_ptr *p_reg = reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(addr);
    return p_reg[0];
}

inline void reg_write(std::uint32_t addr, std::uint32_t data)
{
    volatile std::uint32_t tt_reg_ptr *p_reg = reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(addr);
    p_reg[0]                                 = data;
}

inline void wait(std::uint32_t cycles)
{
    volatile std::uint32_t tt_reg_ptr *clock_lo = reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile std::uint32_t tt_reg_ptr *clock_hi = reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(RISCV_DEBUG_REG_WALL_CLOCK_H);
    std::uint64_t wall_clock_timestamp          = clock_lo[0] | (static_cast<std::uint64_t>(clock_hi[0]) << 32);
    std::uint64_t wall_clock                    = 0;
    do
    {
        wall_clock = clock_lo[0] | (static_cast<std::uint64_t>(clock_hi[0]) << 32);
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

inline void sync_regfile_write(const std::uint32_t index)
{
    volatile std::uint32_t foo     = 0x0;
    volatile std::uint32_t *fooptr = &foo;
    *fooptr                        = regfile[index];
}

inline void cfg_rmw(std::uint32_t cfg_addr32, std::uint32_t cfg_shamt, std::uint32_t cfg_mask, std::uint32_t val)
{
    std::uint32_t wrdata = val;

    // Avoid multiplication of variables!
    // const uint32_t addr = (cfg_state_id * CFG_STATE_SIZE * 4) + cfg_addr32;
    const std::uint32_t addr = (cfg_state_id == 0) ? cfg_addr32 : (CFG_STATE_SIZE * 4) + cfg_addr32;

    // Declared here instead of globally to prevent direct access, which might ignore current state ID
    volatile std::uint32_t tt_reg_ptr *cfg_regs = reinterpret_cast<volatile std::uint32_t tt_reg_ptr *>(TENSIX_CFG_BASE);
    std::uint32_t cfg_data                      = cfg_regs[addr];

    // Shift and mask wrdata to properly align within 32-bit DWORD
    wrdata <<= cfg_shamt;
    wrdata &= cfg_mask;

    // Zero-out relevant bits in cfg data
    cfg_data &= ~cfg_mask;

    // Or new data bits
    cfg_data |= wrdata;

    // Update cfg regs
    cfg_regs[addr] = cfg_data;
}

inline void cfg_rmw_gpr(std::uint32_t cfg_addr32, std::uint32_t cfg_shamt, std::uint32_t cfg_mask, std::uint32_t gpr_index)
{
    const std::uint32_t wrdata = regfile[gpr_index];
    cfg_rmw(cfg_addr32, cfg_shamt, cfg_mask, wrdata);
}

template <std::uint32_t CfgAddr32, std::uint32_t Shamt, std::uint32_t Mask>
inline void cfg_reg_rmw_tensix(std::uint32_t val)
{
    std::uint32_t wrdata = val << Shamt;
    std::uint8_t mask_b0 = Mask & 0xff;

    if (mask_b0 != 0)
    {
        std::uint8_t data_b0 = wrdata & 0xff;
        TT_RMWCIB0(mask_b0, data_b0, CfgAddr32);
    }
    wrdata >>= 8;
    std::uint8_t mask_b1 = (Mask >> 8) & 0xff;

    if (mask_b1 != 0)
    {
        std::uint8_t data_b1 = (wrdata) & 0xff;
        TT_RMWCIB1(mask_b1, data_b1, CfgAddr32);
    }

    wrdata >>= 8;
    std::uint8_t mask_b2 = (Mask >> 16) & 0xff;

    if (mask_b2 != 0)
    {
        std::uint8_t data_b2 = (wrdata) & 0xff;
        TT_RMWCIB2(mask_b2, data_b2, CfgAddr32);
    }

    wrdata >>= 8;
    std::uint8_t mask_b3 = (Mask >> 24) & 0xff;
    if (mask_b3 != 0)
    {
        std::uint8_t data_b3 = (wrdata) & 0xff;
        TT_RMWCIB3(mask_b3, data_b3, CfgAddr32);
    }
}

inline void mailbox_write(const std::uint8_t thread, const std::uint32_t data)
{
    mailbox_base[thread][0] = data;
}

// Blocking read
inline std::uint32_t mailbox_read(const std::uint8_t thread)
{
    return mailbox_base[thread][0];
}

inline bool mailbox_not_empty(const std::uint8_t thread)
{
    return mailbox_base[thread][1] > 0;
}

template <class T>
inline std::uint32_t memory_cast(T *object_ptr)
{
    return reinterpret_cast<std::uint32_t>(object_ptr);
}

inline void record_mailbox_value(std::uint16_t event_value)
{
    if (mailbox_index < mailbox_end)
    {
        debug_mailbox_base[mailbox_index] = event_value;
        mailbox_index++;
    }
}

inline void record_mailbox_value_with_index(std::uint8_t index, std::uint16_t event_value)
{
    if (index < mailbox_end)
    {
        debug_mailbox_base[index] = event_value;
    }
}

// Initialize debug scratch mailbox values and range
inline void clear_mailbox_values(std::uint16_t value = 0)
{
    for (int i = 0; i < mailbox_end; i++)
    {
        debug_mailbox_base[i] = value;
    }
}

inline std::uint64_t read_wall_clock()
{
    std::uint32_t timestamp_low  = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    std::uint32_t timestamp_high = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
    return (static_cast<std::uint64_t>(timestamp_high) << 32) | timestamp_low;
}

inline void record_kernel_runtime(std::uint64_t kernel_runtime)
{
    debug_mailbox_base[mailbox_end - 4] = kernel_runtime & 0xffff;
    debug_mailbox_base[mailbox_end - 3] = (kernel_runtime >> 16) & 0xffff;
    debug_mailbox_base[mailbox_end - 2] = (kernel_runtime >> 32) & 0xffff;
    debug_mailbox_base[mailbox_end - 1] = (kernel_runtime >> 48) & 0xffff;
}

void debug_dump(const std::uint8_t *data, std::uint32_t byte_size);
void debug_dump_seek(std::uint8_t offset);

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
constexpr static std::uint32_t TRACK_GLOBAL_CFG             = 1 << 0;
constexpr static std::uint32_t EN_SUBDIVIDED_CFG_FOR_UNPACR = 1 << 1;
constexpr static std::uint32_t TRACK_GPR                    = 1 << 2;
constexpr static std::uint32_t TRACK_TDMA                   = 1 << 3;
constexpr static std::uint32_t TRACK_TENSIX_INSTRUCTIONS    = 1 << 4;
constexpr static std::uint32_t TRACK_ALL                    = 0x1F;

// Uses a template to guarantee compile time execution (could probably
// get away with constexpr but this seems better)
template <std::uint32_t bitmask>
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

enum class CSR : std::uint16_t
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
inline std::uint32_t csr_read()
{
    std::uint32_t ret;

    if constexpr (fence)
    {
        asm volatile("fence");
    }
    asm volatile("csrr %[ret], %[csr_num] \n" : [ret] "=r"(ret) : [csr_num] "i"(csr_num));

    return ret;
}

// Use at your own risk :-)
template <std::uint16_t csr_num, bool fence = true>
inline std::uint32_t csr_read()
{
    static_assert(csr_num < (1 << 12), "Given CSR number is out of range");
    std::uint32_t ret;

    if constexpr (fence)
    {
        asm volatile("fence");
    }
    asm volatile("csrr %[ret], %[csr_num] \n" : [ret] "=r"(ret) : [csr_num] "i"(csr_num));

    return ret;
}

union qstatus_u
{
    std::uint32_t val;

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
    std::uint32_t val;

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

inline void init_prng_seed(const std::uint32_t seed)
{
    // The seed for PRNG should at least be initialized during chip boot-up time.
    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer();
    cfg[PRNG_SEED_Seed_Val_ADDR32]         = seed;

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

inline void apply_sign_magnitude_conversion(std::uint32_t src, std::uint32_t dst, InstrModCast cast_mode)
{
    TTI_SFPCAST(src /*lreg*/, dst /*ldest*/, cast_mode);
    // Required after cast due to a bug in Blackhole RTL (Refer tenstorrent/tt-llk-bh#16)
    TTI_SFPSETSGN(0 /* imm */, dst /*lreg_c*/, src /*ldest*/, 0 /*imod*/);
}

constexpr std::uint32_t DstTileSizeLog2[3] = {
    6, // 32x32 tile shape
    5, // 32x16, 16x32 tile shape
    4  // 16x16 tile shape
};

/**
 * @brief Calculates the maximum number of destination tiles that can fit in the destination register.
 *
 * @tparam SYNC_MODE   Destination synchronization mode (SyncHalf or SyncFull)
 * @tparam ACCUM_MODE Accumulation mode: true for 32-bit (FP32), false for 16-bit
 * @tparam TILE_SHAPE      Tile shape enum value (e.g., 32x32, 16x16, etc.)
 * @return constexpr std::uint32_t   Maximum number of destination tiles
 *
 * The calculation is based on the destination register size and the tile shape.
 *
 * Formula:
 *   DEST_REGISTER_SIZE >> DstTileSizeLog2[static_cast<int>(TILE_SHAPE)]
 *
 * Where DEST_REGISTER_SIZE is selected based on SYNC_MODE and ACCUM_MODE.
 */
template <DstSync SYNC_MODE, bool ACCUM_MODE, DstTileShape TILE_SHAPE>
constexpr std::uint32_t get_dest_max_tiles()
{
    constexpr std::uint32_t DEST_REGISTER_SIZE = SYNC_MODE == DstSync::SyncHalf ? (ACCUM_MODE ? DEST_REGISTER_HALF_SIZE >> 1 : DEST_REGISTER_HALF_SIZE)
                                                                                : (ACCUM_MODE ? DEST_REGISTER_FULL_SIZE >> 1 : DEST_REGISTER_FULL_SIZE);

    return DEST_REGISTER_SIZE >> DstTileSizeLog2[static_cast<int>(TILE_SHAPE)];
}

/**
 * @brief Used to invalidate the RISCV core's DCache.
 * On Blackhole this happens as a side effect of the FENCE instruction.
 */
inline void invalidate_data_cache()
{
    // clobber memory to prevent code reordering by the compiler.
    asm volatile("fence" ::: "memory");
}

} // namespace ckernel
