#pragma once

#include <type_traits>
#include <cstdint>
#include <initializer_list>

#include "fw_debug.h"
#include "tensix.h"
#include "tensix_functions.h"
#include "noc_overlay_parameters.h"

class c_tensix_core {

public:
    static const bool is_emulated = false;

    static vptr_uint instrn_buf_base(uint32_t thread_id)
    {
        const uint32_t addr[] = { INSTRN_BUF_BASE, INSTRN1_BUF_BASE, INSTRN2_BUF_BASE };
        return reinterpret_cast<uint32_t*>(addr[thread_id]);
    }
    static vptr_pc_buf pc_buf_base(uint32_t thread_id)
    {
        const uint32_t addr[] = { PC_BUF_BASE, PC1_BUF_BASE, PC2_BUF_BASE };
        return reinterpret_cast<uint32_t*>(addr[thread_id]);
    }
    static vptr_uint regfile_base() { return reinterpret_cast<uint32_t*>(REGFILE_BASE); }
    static vptr_uint cfg_regs_base(uint state_id = 0)
    {
        if (state_id == 0)
            return reinterpret_cast<uint32_t *>(TENSIX_CFG_BASE);

        return reinterpret_cast<uint32_t *>(TENSIX_CFG_BASE + CFG_STATE_SIZE * 4 * 4);
    }
    static vptr_mailbox mailbox_base(uint32_t thread_id)
    {
        const uint32_t addr[] = { TENSIX_MAILBOX1_BASE, TENSIX_MAILBOX2_BASE, TENSIX_MAILBOX3_BASE };
        return reinterpret_cast<uint32_t*>(addr[thread_id]);
    }

    static volatile uint64_t *wall_clock_mailbox()
    {
        return (uint64_t *)MEM_WALL_CLOCK_MAILBOX_ADDRESS;
    }

    static void ex_setc16(uint addr, uint val, vptr_uint instrn_buf) { ::ex_setc16(addr, val, instrn_buf); }
    static void ex_rmw_cfg(uint8_t state_id, uint cfg_addr32, uint cfg_shamt, uint32_t cfg_mask, uint wr_val)
      { ::ex_rmw_cfg(cfg_addr32, cfg_shamt, cfg_mask, wr_val, cfg_regs_base(state_id)); }

    static void ex_nop(vptr_uint instrn_buf) { :: ex_nop(instrn_buf); }

    static void ex_sem_init(uint semaphore, uint max_value, uint init_value, vptr_uint instrn_buffer);
    static void ex_zeroacc(vptr_uint instrn_buf, uint clear_mode = 3, uint dest_register = 0, uint addressing_mode = 0);
    static void ex_encc(vptr_uint instrn_buf);
    static void ex_load_const(vptr_uint instrn_buf);

    static uint wait(int cycles);

    static uint64_t read_wall_clock();
    static uint32_t read_wall_clock_l();

    static std::uint32_t l1_size() { return L1_SIZE; }

    static void noc_copy(uint64_t src_addr, uint64_t dst_addr, uint32_t size, bool linked, bool posted, bool wr_blocking = false, bool rd_blocking = false, uint16_t be = 0xffff);
    static void noc_atomic_increment(uint64_t addr, uint32_t incr, uint32_t wrap, bool linked);
    // if blocking copy is requested, set num_blocking_cores to the number of receiving cores
    static void noc_multicast_copy(uint64_t src_addr, uint64_t dst_addr, uint32_t multicast_mode, uint32_t size, bool linked, bool posted, uint32_t num_blocking_cores = 0);
    static void noc_multicast_atomic_increment(uint64_t addr, uint32_t multicast_mode, uint32_t incr, uint32_t wrap, bool linked);

    static std::uint32_t noc_id();

    static inline void write_stream_register(uint32_t stream_id, uint32_t index, uint32_t value);
    static inline uint32_t read_stream_register(uint32_t stream_id, uint32_t index);
    static inline uint32_t read_stream_register_field(uint32_t stream_id, uint32_t index, uint32_t shift, uint32_t width);

private:
    static inline volatile tt_reg_ptr uint32_t* noc_stream_registers(uint32_t stream_id);
};


inline void c_tensix_core::ex_sem_init(uint semaphore, uint max_value, uint init_value, vptr_uint instrn_buffer)
{
    ::ex_sem_init(semaphore, max_value, init_value, instrn_buffer);
}

inline void c_tensix_core::ex_zeroacc(vptr_uint instrn_buf, uint clear_mode, uint dest_register, uint addressing_mode)
{
    ::ex_zeroacc(instrn_buf, clear_mode, dest_register, addressing_mode);
}

inline void c_tensix_core::ex_encc(vptr_uint instrn_buf)
{
    ::ex_encc(instrn_buf);
}

inline void c_tensix_core::ex_load_const(vptr_uint instrn_buf)
{
    // Do nothing on grayskull
}

inline uint c_tensix_core::wait(int cycles)
{
  int count = 0;
  uint bla = 0;

  volatile uint * mailbox = mailbox_base(0);
  while (count < cycles) {
      bla = mailbox[0];
      count++;
  }
  return bla;
}

// NOC API
inline void c_tensix_core::noc_copy(uint64_t src_addr, uint64_t dst_addr, uint32_t size, bool linked, bool posted, bool wr_blocking, bool rd_blocking, uint16_t be) {

    FWASSERT("Write-Blocking behaviour is only supported when posted=false", wr_blocking == false || posted == false);
    FWASSERT("Byte-enable is only supported for a word copy", ( be == 0xffff || size <= 16 ));

    uint32_t acks = wr_blocking ? noc_wr_ack_received() : noc_rd_resp_received();
    uint32_t num_acks = size / NOC_MAX_BURST_SIZE + ((size % NOC_MAX_BURST_SIZE) ? 1 : 0);

    if(be != 0xffff)
    {
      ::noc_copy_word_be(src_addr, dst_addr, be, linked, posted, false, 0);
    }
    else
    {
      ::noc_copy(src_addr, dst_addr, size, linked, posted, false, 0, 0);
    }

    // if blocking copy, wait until all the wacks have been received
    while((wr_blocking && (acks + num_acks != noc_wr_ack_received())) || // block on wacks
          (rd_blocking && (acks + num_acks != noc_rd_resp_received())));  // block on read-responses
}

inline void c_tensix_core::noc_atomic_increment(uint64_t addr, uint32_t incr, uint32_t wrap, bool linked) {
    ::noc_atomic_increment(addr, incr, wrap, linked);
 }

inline void c_tensix_core::noc_multicast_copy(uint64_t src_addr, uint64_t dst_addr, uint32_t multicast_mode, uint32_t size, bool linked, bool posted, uint32_t num_blocking_cores) {

    uint32_t wacks = noc_wr_ack_received();
    uint32_t num_wacks = size / NOC_MAX_BURST_SIZE + ((size % NOC_MAX_BURST_SIZE) ? 1 : 0);
    num_wacks *= num_blocking_cores;

    FWASSERT("Blocking behaviour is only supported when posted=false", num_blocking_cores == 0 || posted == false);

    ::noc_multicast_copy(src_addr, dst_addr, multicast_mode, size, linked, posted, false, 0);

    // if blocking copy, wait until all the wacks have been received
    while(num_blocking_cores && (wacks + num_wacks != noc_wr_ack_received()));
}

inline void c_tensix_core::noc_multicast_atomic_increment(uint64_t addr, uint32_t multicast_mode, uint32_t incr, uint32_t wrap, bool linked) {
    ::noc_multicast_atomic_increment(addr, multicast_mode, incr, wrap, linked);
}

inline std::uint32_t c_tensix_core::noc_id()
{
    std::uint32_t id = ::noc_local_node_id();
    return (id & 0xFFF);
}

inline void c_tensix_core::write_stream_register(uint32_t stream_id, uint32_t index, uint32_t value)
{
  NOC_STREAM_WRITE_REG(stream_id, index, value);
}

inline uint32_t c_tensix_core::read_stream_register(uint32_t stream_id, uint32_t index)
{
  return NOC_STREAM_READ_REG(stream_id, index);
}

inline uint32_t c_tensix_core::read_stream_register_field(uint32_t stream_id, uint32_t index, uint32_t shift, uint32_t width)
{
  return ( read_stream_register(stream_id, index) >> shift ) & ((1 << width)-1);
}

inline uint32_t c_tensix_core::read_wall_clock_l()
{
  return memory_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
}

inline uint64_t c_tensix_core::read_wall_clock()
{
  uint32_t low = memory_read(RISCV_DEBUG_REG_WALL_CLOCK_L); // latches high
  uint32_t high = memory_read(RISCV_DEBUG_REG_WALL_CLOCK_H);

  return ((uint64_t)high << 32) | low;
}
