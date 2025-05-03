// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include <cstdint>
#include <initializer_list>

#include "debug/fw_debug.h"
#include "tensix.h"
#include "tensix_functions.h"
#include "noc_overlay_parameters.h"
#include "ckernel_structs.h"

class c_tensix_core {
public:
    static const bool is_emulated = false;

    static vptr_uint instrn_buf_base(uint32_t thread_id) {
        const uint32_t addr[] = {INSTRN_BUF_BASE, INSTRN1_BUF_BASE, INSTRN2_BUF_BASE};
        return reinterpret_cast<uint32_t*>(addr[thread_id]);
    }
    static vptr_pc_buf pc_buf_base(uint32_t thread_id) {
        const uint32_t addr[] = {PC_BUF_BASE, PC1_BUF_BASE, PC2_BUF_BASE};
        return reinterpret_cast<uint32_t*>(addr[thread_id]);
    }
    static vptr_uint regfile_base() { return reinterpret_cast<uint32_t*>(REGFILE_BASE); }
    static vptr_uint cfg_regs_base(uint state_id = 0) {
        if (state_id == 0) {
            return reinterpret_cast<uint32_t*>(TENSIX_CFG_BASE);
        }

        return reinterpret_cast<uint32_t*>(TENSIX_CFG_BASE + CFG_STATE_SIZE * 4 * 4);
    }

    static volatile uint32_t* get_io_queue_pointer_base(uint32_t base_addr, uint32_t id) {
        return reinterpret_cast<volatile uint32_t*>(base_addr) + (id << 2) + id;
    }

    // These are used to track dynamic allocation/deallocations for perf analysis. They don't do anything by default,
    // but writes to perf scratch area could be added.
    static void record_dynamic_allocation(int buffer_id, int loc, std::intptr_t ptr, uint32_t size) {}
    static void record_dynamic_deallocation(int buffer_id) {}

    static void ex_setc16(uint addr, uint val, vptr_uint instrn_buf) { ::ex_setc16(addr, val, instrn_buf); }
    static void ex_instrn_wrcfg(uint gpr, uint cfg_addr, vptr_uint instrn_buf) {
        ::ex_instrn_wrcfg(gpr, cfg_addr, instrn_buf);
    }
    static void ex_instrn_rdcfg(uint gpr, uint cfg_addr, vptr_uint instrn_buf) {
        ::ex_instrn_rdcfg(gpr, cfg_addr, instrn_buf);
    }
    static void ex_rmw_cfg_gpr(uint state_id, uint cfg_addr32, uint cfg_shamt, uint32_t cfg_mask, uint gpr_index) {
        ::ex_rmw_cfg_gpr(cfg_addr32, cfg_shamt, cfg_mask, gpr_index, regfile_base(), cfg_regs_base(state_id));
    }
    static void ex_rmw_cfg(uint8_t state_id, uint cfg_addr32, uint cfg_shamt, uint32_t cfg_mask, uint wr_val) {
        ::ex_rmw_cfg(cfg_addr32, cfg_shamt, cfg_mask, wr_val, cfg_regs_base(state_id));
    }

    static void ex_nop(vptr_uint instrn_buf) { ::ex_nop(instrn_buf); }

    // static void ex_set_stride_prepacked(cnt_id_t cntset_ind, uint chan_ind, uint xy_stride, uint zw_stride, vptr_uint
    // instrn_buf);
    static void ex_setadc(cnt_id_t cnt_ind, uint chan_ind, uint dim_ind, uint val, vptr_uint instrn_buf);
    static void ex_setpkedgof(uint edge_mask, vptr_uint instrn_buf);
    static void ex_clear_dvalid(uint clear_ab, uint reset, vptr_uint instrn_buffer);
    static void ex_sem_init(uint semaphore, uint max_value, uint init_value, vptr_uint instrn_buffer);
    static void ex_zeroacc(vptr_uint instrn_buf, uint clear_mode = 3, uint dest_register = 0, uint addressing_mode = 0);
    static void ex_encc(vptr_uint instrn_buf);
    static void ex_load_const(vptr_uint instrn_buf);

    static void ex_instrn(vptr_uint instrn_buffer, unsigned int instruction) {
        ::execute_instruction(instrn_buffer, instruction);
    }
    static void thcon_load_ind(
        vptr_uint instrn_buffer,
        std::uint32_t base_addr_index,
        std::uint32_t dst_data_index,
        std::uint32_t offset_index,
        std::uint32_t autoinc,
        std::uint32_t size);
    static void thcon_incr_get_ptr(
        vptr_uint instrn_buffer,
        std::uint32_t mem_addr_index,
        std::uint32_t data_reg_index,
        std::uint32_t incr_val,
        std::uint32_t wrap_val,
        bool rd_wr,
        bool l0_l1_sel);
    static void thcon_incr_get_ptr_noinc(
        vptr_uint instrn_buffer,
        std::uint32_t mem_addr_index,
        std::uint32_t data_reg_index,
        std::uint32_t incr_val,
        std::uint32_t wrap_val,
        bool rd_wr,
        bool l0_l1_sel);
    static void thcon_reg_to_flops(
        vptr_uint instrn_buffer,
        uint32_t mode_32b_16B,
        uint32_t reg_index,
        uint32_t flop_index,
        uint32_t target_select = 0,
        uint32_t byte_offset = 0);
    static void thcon_set_descriptor(vptr_uint instrn_buf, uint reg_index, uint unpacker_id);

    static uint read_packed_size(uint thread);              // Return size in bytes of last packer output for a thread.
    static uint read_accumulated_packed_size(uint thread);  // Return accumulated size of tiles processed by the packer

    static void initialize_tensix_semaphores(vptr_uint instrn_buf);  // Initialize tensix semaphores

    static uint wait(int cycles);

    static uint64_t read_wall_clock();
    static uint32_t read_wall_clock_l();

    template <class T, std::enable_if_t<std::is_pointer<T>::value, int> = 0>
    static T l1_cast(uint32_t l1_offset) {
        return reinterpret_cast<T>(l1_offset);
    }

    template <class T>
    static std::uint32_t l1_cast(T* l1_pointer) {
        return reinterpret_cast<uint32_t>(l1_pointer);
    }

    static std::uint32_t l1_size() { return SIM_L1_SIZE; }

    static void noc_copy(
        uint64_t src_addr,
        uint64_t dst_addr,
        uint32_t size,
        bool linked,
        bool posted,
        bool wr_blocking = false,
        bool rd_blocking = false,
        uint16_t be = 0xffff);
    static void noc_atomic_increment(uint64_t addr, uint32_t incr, uint32_t wrap, bool linked);
    // if blocking copy is requested, set num_blocking_cores to the number of receiving cores
    static void noc_multicast_copy(
        uint64_t src_addr,
        uint64_t dst_addr,
        uint32_t multicast_mode,
        uint32_t size,
        bool linked,
        bool posted,
        uint32_t num_blocking_cores = 0);
    static void noc_multicast_atomic_increment(
        uint64_t addr, uint32_t multicast_mode, uint32_t incr, uint32_t wrap, bool linked);

    static std::uint32_t noc_id();

    static inline void write_stream_register(uint32_t stream_id, uint32_t index, uint32_t value);
    static inline uint32_t read_stream_register(uint32_t stream_id, uint32_t index);
    static inline uint32_t read_stream_register_field(
        uint32_t stream_id, uint32_t index, uint32_t shift, uint32_t width);

    static inline void ExtraKernelParams(
        uint /*thread_id*/, uint /*kernel_id*/, std::initializer_list<uint32_t> /*params*/) {}

    static inline void check_l1_address_range(std::uint32_t byte_addr, std::size_t length);

private:
    static inline volatile uint32_t* noc_stream_registers(uint32_t stream_id);
};

/*inline void c_tensix_core::ex_set_stride_prepacked(cnt_id_t cntset_ind, uint chan_ind, uint xy_stride, uint zw_stride,
volatile uint * instrn_buf)
{
    ::ex_set_stride_prepacked(cntset_ind, chan_ind, xy_stride, zw_stride, instrn_buf);
}*/

inline void c_tensix_core::ex_setpkedgof(uint edge_mask, vptr_uint instrn_buf) {
    ::ex_setpkedgof(edge_mask, instrn_buf);
}

inline void c_tensix_core::ex_clear_dvalid(uint clear_ab, uint reset, vptr_uint instrn_buffer) {
    ::ex_clear_dvalid(clear_ab, reset, instrn_buffer);
}

inline void c_tensix_core::ex_sem_init(uint semaphore, uint max_value, uint init_value, vptr_uint instrn_buffer) {
    ::ex_sem_init(semaphore, max_value, init_value, instrn_buffer);
}

inline void c_tensix_core::ex_zeroacc(vptr_uint instrn_buf, uint clear_mode, uint dest_register, uint addressing_mode) {
    ::ex_zeroacc(instrn_buf, clear_mode, dest_register, addressing_mode);
}

inline void c_tensix_core::ex_encc(vptr_uint instrn_buf) { ::ex_encc(instrn_buf); }

inline void c_tensix_core::ex_load_const(vptr_uint instrn_buf) {
    // Load LREG11 w/ -1.0f by convention
    uint instrn;
    instrn = (0xbf80 << 0);  // Load LREG0 w/ -1.0f
    ex_push_insn(instrn_buf, INSTRN_SFPLOADI(instrn));
    instrn = (11 << 4);  // Set LREG11 to LREG0
    ex_push_insn(instrn_buf, INSTRN_SFPCONFIG(instrn));
}

inline void c_tensix_core::ex_setadc(cnt_id_t cnt_ind, uint chan_ind, uint dim_ind, uint val, vptr_uint instrn_buf) {
    ::ex_setadc(cnt_ind, chan_ind, dim_ind, val, instrn_buf);
}

inline void c_tensix_core::thcon_load_ind(
    vptr_uint instrn_buffer, uint base_addr_index, uint dst_data_index, uint offset_index, uint autoinc, uint size) {
    ::thcon_load_ind(instrn_buffer, base_addr_index, dst_data_index, offset_index, autoinc, size);
}

inline void c_tensix_core::thcon_incr_get_ptr(
    vptr_uint instrn_buffer,
    uint mem_addr_index,
    uint data_reg_index,
    uint incr_val,
    uint wrap_val,
    bool rd_wr,
    bool l0_l1_sel) {
    ::thcon_incr_get_ptr(instrn_buffer, mem_addr_index, data_reg_index, incr_val, wrap_val, rd_wr, l0_l1_sel);
}

inline void c_tensix_core::thcon_incr_get_ptr_noinc(
    vptr_uint instrn_buffer,
    uint mem_addr_index,
    uint data_reg_index,
    uint incr_val,
    uint wrap_val,
    bool rd_wr,
    bool l0_l1_sel) {
    ::thcon_incr_get_ptr_noinc(instrn_buffer, mem_addr_index, data_reg_index, incr_val, wrap_val, rd_wr, l0_l1_sel);
}

inline void c_tensix_core::thcon_reg_to_flops(
    vptr_uint instrn_buffer, uint mode_32b_16B, uint reg_index, uint flop_index, uint target_select, uint byte_offset) {
    ::thcon_reg_to_flops(instrn_buffer, mode_32b_16B, reg_index, flop_index, target_select, byte_offset);
}

inline void c_tensix_core::thcon_set_descriptor(vptr_uint instrn_buf, uint reg_index, uint unpacker_id) {
    ::thcon_set_descriptor(instrn_buf, reg_index, unpacker_id);
}

inline uint c_tensix_core::read_packed_size(uint thread) {
    uint packed_size = memory_read(RISCV_TDMA_REG_PACKED_SIZE);
    if (thread == 0) {
        packed_size &= 0xFFFF;
    } else {
        packed_size >>= 16;
    }

    return packed_size;
}

inline uint c_tensix_core::read_accumulated_packed_size(uint thread) {
    uint packed_size = memory_read(RISCV_TDMA_REG_ACC_PACKED_SIZE);
    if (thread == 0) {
        packed_size &= 0xFFFF;
    } else {
        packed_size >>= 16;
    }

    return packed_size;
}

inline void c_tensix_core::initialize_tensix_semaphores(vptr_uint instrn_buf) {
    // Initialize sempahores - check if we need to do this still
    // math->packer semaphore - max set to 1, as double-buffering is disabled by default
    ex_sem_init(ckernel::semaphore::MATH_PACK, 1, 0, instrn_buf);
    ex_sem_init(ckernel::semaphore::UNPACK_TO_DEST, 1, 0, instrn_buf);
    ex_sem_init(ckernel::semaphore::MATH_DONE, 1, 0, instrn_buf);
}

// NOC API
inline void c_tensix_core::noc_copy(
    uint64_t src_addr,
    uint64_t dst_addr,
    uint32_t size,
    bool linked,
    bool posted,
    bool wr_blocking,
    bool rd_blocking,
    uint16_t be) {
    FWASSERT("Write-Blocking behaviour is only supported when posted=false", wr_blocking == false || posted == false);
    FWASSERT("Byte-enable is only supported for a word copy", (be == 0xffff || size <= 16));

    uint32_t acks = wr_blocking ? noc_wr_ack_received() : noc_rd_resp_received();
    uint32_t num_acks = size / NOC_MAX_BURST_SIZE + ((size % NOC_MAX_BURST_SIZE) ? 1 : 0);

    if (be != 0xffff) {
        ::noc_copy_word_be(src_addr, dst_addr, be, linked, posted, false, 0, 0);
    } else {
        ::noc_copy(src_addr, dst_addr, size, linked, posted, false, 0, 0, 0);
    }

    // if blocking copy, wait until all the wacks have been received
    while ((wr_blocking && (acks + num_acks != noc_wr_ack_received())) ||  // block on wacks
           (rd_blocking && (acks + num_acks != noc_rd_resp_received())));  // block on read-responses
}

inline void c_tensix_core::noc_atomic_increment(uint64_t addr, uint32_t incr, uint32_t wrap, bool linked) {
    ::noc_atomic_increment(addr, incr, wrap, linked);
}

inline void c_tensix_core::noc_multicast_copy(
    uint64_t src_addr,
    uint64_t dst_addr,
    uint32_t multicast_mode,
    uint32_t size,
    bool linked,
    bool posted,
    uint32_t num_blocking_cores) {
    uint32_t wacks = noc_wr_ack_received();
    uint32_t num_wacks = size / NOC_MAX_BURST_SIZE + ((size % NOC_MAX_BURST_SIZE) ? 1 : 0);
    num_wacks *= num_blocking_cores;

    FWASSERT("Blocking behaviour is only supported when posted=false", num_blocking_cores == 0 || posted == false);

    ::noc_multicast_copy(src_addr, dst_addr, multicast_mode, size, linked, posted, false, 0, 0);

    // if blocking copy, wait until all the wacks have been received
    while (num_blocking_cores && (wacks + num_wacks != noc_wr_ack_received()));
}

inline void c_tensix_core::noc_multicast_atomic_increment(
    uint64_t addr, uint32_t multicast_mode, uint32_t incr, uint32_t wrap, bool linked) {
    ::noc_multicast_atomic_increment(addr, multicast_mode, incr, wrap, linked);
}

inline std::uint32_t c_tensix_core::noc_id() {
    std::uint32_t id = ::noc_local_node_id();
    return (id & 0xFFF);
}

inline void c_tensix_core::write_stream_register(uint32_t stream_id, uint32_t index, uint32_t value) {
    NOC_STREAM_WRITE_REG(stream_id, index, value);
}

inline uint32_t c_tensix_core::read_stream_register(uint32_t stream_id, uint32_t index) {
    return NOC_STREAM_READ_REG(stream_id, index);
}

inline uint32_t c_tensix_core::read_stream_register_field(
    uint32_t stream_id, uint32_t index, uint32_t shift, uint32_t width) {
    return (read_stream_register(stream_id, index) >> shift) & ((1 << width) - 1);
}

inline uint32_t c_tensix_core::read_wall_clock_l() { return memory_read(RISCV_DEBUG_REG_WALL_CLOCK_L); }

inline uint64_t c_tensix_core::read_wall_clock() {
    uint32_t low = memory_read(RISCV_DEBUG_REG_WALL_CLOCK_L);  // latches high
    uint32_t high = memory_read(RISCV_DEBUG_REG_WALL_CLOCK_H);

    return ((uint64_t)high << 32) | low;
}

inline void c_tensix_core::check_l1_address_range(std::uint32_t byte_addr, std::size_t length) {
    FWASSERT("Exceeded L1 of 1MB!!", ((byte_addr + length) <= (1U << 20)));
}
