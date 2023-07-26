#ifndef _RISC_COMMON_H_
#define _RISC_COMMON_H_

#include <cstdint>
#include <stdint.h>

#include "noc_parameters.h"
#include "tensix.h"
#include "risc.h"
#include "eth_l1_address_map.h"
#include "noc_overlay_parameters.h"
#include "stream_io_map.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "limits.h"

#define NOC_X(x) (loading_noc == 0 ? (x) : (noc_size_x-1-(x)))
#define NOC_Y(y) (loading_noc == 0 ? (y) : (noc_size_y-1-(y)))

#define TILE_WORD_2_BIT ((256 + 64 + 32) >> 4)
#define TILE_WORD_4_BIT ((512 + 64 + 32) >> 4)
#define TILE_WORD_8_BIT ((32*32*1 + 64 + 32) >> 4)
#define TILE_WORD_16_BIT ((32*32*2 + 32) >> 4)
#define TILE_WORD_32_BIT ((32*32*4 + 32) >> 4)

#ifdef COMPILE_FOR_BRISC
constexpr std::uint32_t L1_ARG_BASE = BRISC_L1_ARG_BASE;
constexpr std::uint32_t L1_RESULT_BASE = BRISC_L1_RESULT_BASE;
#elif defined(COMPILE_FOR_NCRISC)
constexpr std::uint32_t L1_ARG_BASE = NCRISC_L1_ARG_BASE;
constexpr std::uint32_t L1_RESULT_BASE = NCRISC_L1_RESULT_BASE;
#endif

const uint32_t STREAM_RESTART_CHECK_MASK = (0x1 << 3) - 1;

const uint32_t MAX_TILES_PER_PHASE = 2048;

extern uint8_t my_x[NUM_NOCS];
extern uint8_t my_y[NUM_NOCS];
extern uint8_t noc_size_x;
extern uint8_t noc_size_y;
extern volatile uint32_t local_mem_barrier;

inline void WRITE_REG(uint32_t addr, uint32_t val) {
  volatile tt_reg_ptr uint32_t* ptr = (volatile tt_reg_ptr uint32_t*)addr;
  ptr[0] = val;
}

inline uint32_t READ_REG(uint32_t addr) {
  volatile tt_reg_ptr uint32_t* ptr = (volatile tt_reg_ptr uint32_t*)addr;
  return ptr[0];
}

inline uint32_t dram_io_incr_ptr(uint32_t curr_ptr, uint32_t incr, uint32_t buf_size_q_slots) {
  uint32_t next_ptr = curr_ptr + incr;
  uint32_t double_buf_size_q_slots = 2*buf_size_q_slots;
  if (next_ptr >= double_buf_size_q_slots) {
    next_ptr -= double_buf_size_q_slots;
  }
  return next_ptr;
}

inline __attribute__((always_inline)) uint32_t dram_io_empty(uint32_t rd_ptr, uint32_t wr_ptr) {
  return (rd_ptr == wr_ptr);
}

inline __attribute__((always_inline)) uint32_t dram_io_local_empty(uint32_t local_rd_ptr, uint32_t rd_ptr, uint32_t wr_ptr) {
  if (rd_ptr == wr_ptr)
    return true;

  uint32_t case1 = rd_ptr < wr_ptr && (local_rd_ptr < rd_ptr || local_rd_ptr >= wr_ptr);
  uint32_t case2 = rd_ptr > wr_ptr && wr_ptr <= local_rd_ptr && local_rd_ptr < rd_ptr;

  return case1 || case2;
}

inline uint32_t dram_io_full(uint32_t rd_ptr, uint32_t wr_ptr, uint32_t buf_size_q_slots) {
  uint32_t wr_ptr_reduced_by_q_slots = wr_ptr - buf_size_q_slots;
  uint32_t rd_ptr_reduced_by_q_slots = rd_ptr - buf_size_q_slots;
  uint32_t case1 = (wr_ptr_reduced_by_q_slots == rd_ptr);
  uint32_t case2 = (rd_ptr_reduced_by_q_slots == wr_ptr);
  return case1 || case2;
}

inline __attribute__((always_inline)) uint32_t buf_ptr_inc_wrap(uint32_t buf_ptr, uint32_t inc, uint32_t buf_size) {
  uint32_t result = buf_ptr + inc;
  if (result >= buf_size) {
    result -= buf_size;
  }
  return result;
}

inline __attribute__((always_inline)) uint32_t buf_ptr_dec_wrap(uint32_t buf_ptr, uint32_t dec, uint32_t buf_size) {
  uint32_t result = buf_ptr;
  if (dec > result) {
    result += buf_size;
  }
  result -= dec;
  return result;
}

inline uint32_t reg_read_barrier(uint32_t addr)
{
    volatile tt_reg_ptr uint32_t *p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t *> (addr);
    uint32_t data = p_reg[0];
    local_mem_barrier = data;
    return data;
}

inline uint32_t reg_read_barrier_l1(uint32_t addr)
{
    volatile tt_reg_ptr uint32_t *p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t *> (addr);
    uint32_t data = p_reg[0];
    local_mem_barrier = data;
    return data;
}

inline void assert_trisc_reset() {
  uint32_t soft_reset_0 = READ_REG(RISCV_DEBUG_REG_SOFT_RESET_0);
  uint32_t trisc_reset_mask = 0x7000;
  WRITE_REG(RISCV_DEBUG_REG_SOFT_RESET_0, soft_reset_0 | trisc_reset_mask);
}


inline void deassert_trisc_reset() {
  uint32_t soft_reset_0 = READ_REG(RISCV_DEBUG_REG_SOFT_RESET_0);
  uint32_t trisc_reset_mask = 0x7000;
  WRITE_REG(RISCV_DEBUG_REG_SOFT_RESET_0, soft_reset_0 & ~trisc_reset_mask);
}

inline uint32_t special_mult(uint32_t a, uint32_t special_b) {
  if (special_b == TILE_WORD_8_BIT)
    return a * TILE_WORD_8_BIT;
  else if (special_b == TILE_WORD_16_BIT)
    return a * TILE_WORD_16_BIT;
  else if (special_b == TILE_WORD_4_BIT)
    return a * TILE_WORD_4_BIT;
  else if (special_b == TILE_WORD_2_BIT)
    return a * TILE_WORD_2_BIT;
  else if (special_b == TILE_WORD_32_BIT)
    return a * TILE_WORD_32_BIT;

  RISC_POST_STATUS(0xDEAD0002);
  while(true);
  return 0;
}

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

template <uint32_t d>
inline __attribute__((always_inline)) uint32_t udivsi3_const_divisor(uint32_t n)
{
    if constexpr (d == 12) {
        // fast divide for 12 divisor
        return fast_udiv_12(n);
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

void risc_init();
void replicate(uint32_t noc_id, uint32_t src_addr, uint64_t dest_addr, uint32_t chunk_size_bytes, uint32_t times_to_replicate);
void replicate_l1(uint32_t noc_id, uint32_t src_addr, uint64_t dest_addr, uint32_t chunk_size_bytes, uint32_t times_to_replicate);
void tile_header_buffer_init();

// This call blocks until NCRISC indicates that all epoch start state
// has been loaded from DRAM to L1.
void risc_get_next_epoch();
// This call signals to NCRISC that the current epoch is done and can
// be overwritten with the next epoch state from DRAM.
void risc_signal_epoch_done();

inline void breakpoint_(uint32_t line) {
    /*
        When called, writes the stack pointer to a known location
        in memory (unique for each core) and then hangs until the
        user explicitly continues
    */
    uint32_t BREAKPOINT;
    uint32_t LNUM;
    volatile tt_l1_ptr uint32_t* bp;
    volatile tt_l1_ptr uint32_t* lnum;

    #define MACRO_SP_AUX(SP) #SP
    #define MACRO_SP(SP) MACRO_SP_AUX(SP)

    // Need to use macros for inline assembly in order to create a string literal
    #if defined(COMPILE_FOR_NCRISC)
        asm("li t0, " MACRO_SP(NCRISC_SP_MACRO));
        BREAKPOINT = NCRISC_BREAKPOINT;
        LNUM = NCRISC_BP_LNUM;
    #elif defined(COMPILE_FOR_BRISC)
        asm("li t0, " MACRO_SP(BRISC_SP_MACRO));
        BREAKPOINT = BRISC_BREAKPOINT;
        LNUM = BRISC_BP_LNUM;
    #else
    extern uint32_t __firmware_start[];
    if ((uint32_t)__firmware_start == (uint32_t)l1_mem::address_map::TRISC0_BASE) {
        asm("li t0, " MACRO_SP(TRISC0_SP_MACRO));
        BREAKPOINT = TRISC0_BREAKPOINT;
        LNUM = TRISC0_BP_LNUM
    } else if ((uint32_t)__firmware_start == (uint32_t)l1_mem::address_map::TRISC1_BASE) {
        asm("li t0, " MACRO_SP(TRISC1_SP_MACRO));
        BREAKPOINT = TRISC1_BREAKPOINT;
        LNUM = TRISC1_BP_LNUM
    } else if ((uint32_t)__firmware_start == (uint32_t)l1_mem::address_map::TRISC2_BASE) {
        asm("li t0, " MACRO_SP(TRISC2_SP_MACRO));
        BREAKPOINT = TRISC2_BREAKPOINT;
        LNUM = TRISC2_BP_LNUM
    }
    #endif

    // Write '1' to breakpoint location so that this core keeps
    // busy looping until host releases it
    asm("sw sp, 0(t0)");
    bp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(BREAKPOINT);
    bp[0] = 1;

    lnum    = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(LNUM);
    lnum[0] = line;

    while (bp[0] == 1);
}

#define breakpoint() breakpoint_(__LINE__);

#endif
