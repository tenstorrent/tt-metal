// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensix.h"
#include "tensix_types.h"
#include "vptr_uint.h"

#ifndef CPU_JAWBRIDGE
#include <stdarg.h>
#endif

/**
 * Notify compiler that any memory address could have been written by external processes or could be read by external
 * process.
 */
inline void clobber_all_memory(void) { asm volatile("" ::: "memory"); }

/**
 * Wait for kernel to transfer program flow from Tensix to RISCV.
 */
#ifdef CPU_JAWBRIDGE
inline void ex_sync_kernel(vptr_mailbox mailbox) {
#ifndef MODELT
    while (mailbox[0] == 0);
    clobber_all_memory();
#endif
}
#endif

/**
 * Push an instruction into the tensix instruction fifo.
 *
 * See documentation of INSTRN_BUF_BASE for conditions.
 */
inline void ex_push_insn(vptr_uint instrn_buffer, uint instrn) { instrn_buffer[0] = instrn; }

/**
 * Push an instruction into the tensix instruction fifo. (Two-word instruction variant.)
 *
 * See documentation of INSTRN_BUF_BASE for conditions.
 */
inline void ex_push_insn(vptr_uint instrn_buffer, uint instrn1, uint instrn2) {
    instrn_buffer[0] = instrn1;
    instrn_buffer[0] = instrn2;
}

inline void ex_pacr(uint addr_mode, uint zero_write, uint flush, uint last, vptr_uint instrn_buf) {
    uint instrn;
    instrn = 0x0 | (addr_mode << 15) | (zero_write << 12) | (flush << 8) | (last);
    ex_push_insn(instrn_buf, INSTRN_PACRNL(instrn));
}

inline void ex_upacr(uint block_sel, uint addr_mode, uint zero_write, uint last, vptr_uint instrn_buf) {
    uint instrn;
    instrn = 0x0 | (block_sel << 23) | (addr_mode << 15) | (zero_write << 4) | (last);
    ex_push_insn(instrn_buf, INSTRN_UNPACR(instrn));
}

inline void ex_xsearch(uint block_sel, vptr_uint instrn_buf) {
    uint instrn;
    instrn = 0x0 | (block_sel << 23);
    ex_push_insn(instrn_buf, INSTRN_SEARCHX(instrn));
}

inline void ex_nop(vptr_uint instrn_buf) {
    uint instrn;
    instrn = 0x0;
    ex_push_insn(instrn_buf, INSTRN_NOP(instrn));
}

inline void ex_flush(vptr_uint instrn_buf) {
    uint instrn;
    instrn = 0x0;
    ex_push_insn(instrn_buf, INSTRN_FLUSH(instrn));
}

#define ZEROSRC_A (0x1)
#define ZEROSRC_B (0x2)
#define ZEROSRC (0x3)

inline void ex_zerosrc(uint src, vptr_uint instrn_buf) {
    uint instrn;
    instrn = 0x0 | src;
    ex_push_insn(instrn_buf, INSTRN_ZEROSRC(instrn));
}

inline void ex_mova2d(uint addr_mode, uint srca_transp, uint dest_index, uint srca_index, vptr_uint instrn_buf) {
    uint instrn;
    instrn = 0x0 | (addr_mode << 15) | (srca_transp << 12) | (dest_index << 4) | (srca_index);
    ex_push_insn(instrn_buf, INSTRN_MOVA2D(instrn));
}

inline void ex_stallwait(uint wait_res, uint stall_res, vptr_uint instrn_buf) {
    uint instrn;
    instrn = 0x0 | (stall_res << 12) | (wait_res);
    ex_push_insn(instrn_buf, INSTRN_STALLWAIT(instrn));
}

inline void ex_setc16(uint addr, uint val, vptr_uint instrn_buf) {
    uint instrn;
    instrn = 0x0 | (addr << 16) | (val & 0xFFFF);
    ex_push_insn(instrn_buf, INSTRN_SETC16(instrn));
}

inline void ex_instrn_wrcfg(uint gpr, uint cfg_addr, vptr_uint instrn_buf) {
    uint instrn;
    instrn = 0x0 | (gpr << 16) | (cfg_addr);
    ex_push_insn(instrn_buf, INSTRN_WRCFG(instrn));
}

inline void ex_instrn_rdcfg(uint gpr, uint cfg_addr, vptr_uint instrn_buf) {
    uint instrn;
    instrn = 0x0 | (gpr << 16) | (cfg_addr);
    ex_push_insn(instrn_buf, INSTRN_RDCFG(instrn));
}

inline uint32_t rmw_cfg_value(uint cfg_shamt, uint32_t cfg_mask, uint32_t wrdata, uint32_t l_cfg_data) {
    uint32_t cfg_data = l_cfg_data;

    // Shift and mask wrdata to properly align withn 32-bit DWORD
    wrdata <<= cfg_shamt;
    wrdata &= cfg_mask;

    // Zero-out relevant bits in cfg data
    cfg_data &= ~cfg_mask;

    // Or new data bits
    cfg_data |= wrdata;

    return cfg_data;
}

inline void ex_rmw_cfg(uint cfg_addr32, uint cfg_shamt, uint32_t cfg_mask, uint wr_val, vptr_uint cfg_regs) {
    uint addr = cfg_addr32;
    uint32_t cfg_data = cfg_regs[addr];
    cfg_regs[addr] = rmw_cfg_value(cfg_shamt, cfg_mask, wr_val, cfg_data);
}

inline void ex_rmw_cfg_gpr(
    uint cfg_addr32, uint cfg_shamt, uint32_t cfg_mask, uint gpr_index, vptr_uint regfile, vptr_uint cfg_regs) {
    uint32_t wrdata = regfile[gpr_index];
    ex_rmw_cfg(cfg_addr32, cfg_shamt, cfg_mask, wrdata, cfg_regs);
}

/**
 * TODO
 */
inline void ex_setadc(cnt_id_t cnt_ind, uint chan_ind, uint dim_ind, uint val, vptr_uint instrn_buf) {
    uint instrn;
    instrn = 0x0 | (cnt_ind << 21) | (chan_ind << 20) | (dim_ind << 18) | (val & 0xFFFF);
    ex_push_insn(instrn_buf, INSTRN_SETADC(instrn));
}

inline void ex_zeroacc(vptr_uint instrn_buf, uint clear_mode = 3, uint dest_register = 0, uint addressing_mode = 0) {
    uint instrn;
    instrn = 0x0 | (clear_mode << 19) | (addressing_mode << 15) | (dest_register << 0);
    ex_push_insn(instrn_buf, INSTRN_ZEROACC(instrn));
}

inline void ex_encc(vptr_uint instrn_buf) {
    uint instrn;
    instrn = (3 << 12) | (10 << 0);  // Set CC enable and result
    ex_push_insn(instrn_buf, INSTRN_SFPENCC(instrn));
    ex_push_insn(instrn_buf, INSTRN_NOP(0));
}

/**
 * Set address counters for X and Y dimensions.
 * @param dstac_y    TODO
 * @param dstac_x    TODO
 * @param srcb_y     TODO
 * @param srcb_x     TODO
 * @param srca_y     TODO
 * @param srca_x     TODO
 * @param instrn_buf TODO
 */
inline void ex_setadcxy(cnt_id_t cntset_ind, uint srcb_y, uint srcb_x, uint srca_y, uint srca_x, vptr_uint instrn_buf) {
    uint instrn;
    instrn = 0x0 | (cntset_ind << 21) | (srcb_y << 15) | (srcb_x << 12) | (srca_y << 9) | (srca_x << 6) | 0x0f;
    ex_push_insn(instrn_buf, INSTRN_SETADCXY(instrn));
}

inline void ex_setadczw(cnt_id_t cntset_ind, uint srcb_w, uint srcb_z, uint srca_w, uint srca_z, vptr_uint instrn_buf) {
    uint instrn;
    instrn = 0x0 | (cntset_ind << 21) | (srcb_w << 15) | (srcb_z << 12) | (srca_w << 9) | (srca_z << 6) | 0x3f;
    ex_push_insn(instrn_buf, INSTRN_SETADCZW(instrn));
}

#define COUNTER_SEL(cnt_sel, cntset_index, channel_index, channel_reg) \
    do {                                                               \
        if ((channel_index == 0) && (cntset_index == UNP0))            \
            cnt_sel = UNP0_##channel_reg##_0;                          \
        else if ((channel_index == 0) && (cntset_index == UNP1))       \
            cnt_sel = UNP1_##channel_reg##_0;                          \
        else if ((channel_index == 0) && (cntset_index == PCK0))       \
            cnt_sel = PCK0_##channel_reg##_0;                          \
        else if ((channel_index == 1) && (cntset_index == UNP0))       \
            cnt_sel = UNP0_##channel_reg##_1;                          \
        else if ((channel_index == 1) && (cntset_index == UNP1))       \
            cnt_sel = UNP1_##channel_reg##_1;                          \
        else if ((channel_index == 1) && (cntset_index == PCK0))       \
            cnt_sel = PCK0_##channel_reg##_1;                          \
        else                                                           \
            cnt_sel = 0;                                               \
    } while (0)

// #define CHANNEL_REG(channel_index, channel_reg) (channel_index == 0 ? channel_reg##_0 : channel_reg##_1)

/*
inline void ex_set_stride(cnt_id_t cntset_ind, uint chan_ind, uint x_stride, uint y_stride, uint z_stride, uint
w_stride, vptr_uint  instrn_buf)
{
  uint addr;
  COUNTER_SEL(addr, cntset_ind, chan_ind, ADDR_CTRL_XY_REG);

  uint regval;
  regval = 0x0 | (x_stride) | (y_stride << 16);
  ex_setc(addr, regval, instrn_buf);
  addr++; // TODO: Replace with CHANNEL_REG(chan_ind, ADDR_CTRL_ZW_REG) and test.
  regval = 0x0 | (z_stride) | (w_stride << 16);
  ex_setc(addr, regval, instrn_buf);
}

inline void ex_set_stride_prepacked(cnt_id_t cntset_ind, uint chan_ind, uint xy_stride, uint zw_stride, vptr_uint
instrn_buf)
{
  uint addr;
  COUNTER_SEL(addr, cntset_ind, chan_ind, ADDR_CTRL_XY_REG);

  ex_setc(addr, xy_stride, instrn_buf);
  ex_setc(addr+1, zw_stride, instrn_buf);
}



inline void ex_set_base(cnt_id_t cntset_ind, uint chan_ind, uint base, vptr_uint  instrn_buf)
{
  // uint addr = CHANNEL_REG(chan_ind, CNTSET_REG(cntset_ind, ADDR_BASE_REG));
  uint addr;
  COUNTER_SEL(addr, cntset_ind, chan_ind, ADDR_CTRL_XY_REG);

  ex_setc(addr, base, instrn_buf);
}
*/

inline void ex_setpkedgof(uint edge_mask, vptr_uint instrn_buf) {
    ex_push_insn(instrn_buf, INSTRN_SETPKEDGEOF(edge_mask));
}

inline void execute_kernel_loop(uint kernel_count, uint loop_count, vptr_pc_buf pc_buf) {
    // FWASSERT("Loop count must be at least 1", loop_count > 0);
    // FWASSERT("Kernel count must be at least 1", kernel_count > 0);
    clobber_all_memory();
    uint32_t val = ((kernel_count - 1) << 16) | (loop_count - 1);
    pc_buf[0] = TENSIX_LOOP_PC_VAL(val);
}

inline void execute_kernel_sync(vptr_pc_buf pc_buf, vptr_mailbox mailbox) {
#ifndef MODELT
    volatile uint foo = 0xdeadbeef;
    volatile uint* fooptr = &foo;
    clobber_all_memory();

    /*
    // Wait until all kernels have completed
    pc_buf[0] = TENSIX_PC_SYNC(0);
    clobber_all_memory();
    pc_buf[0] = TENSIX_PC_SYNC(0); // get blocked in WB
    clobber_all_memory();
    *fooptr = pc_buf[0];               // this read will block on the previous write until it goes through
    */

    // Write to pc buffer to push all writes ahead of us.. otherwise, the pc buffer read can bypass older writes
    pc_buf[1] = foo;

    *fooptr = pc_buf[1];  // sync read - block until everything is idle

    // Clear the mailbox if it was set by one of the previous kernels
    clobber_all_memory();
#ifdef CPU_JAWBRIDGE
    *fooptr = mailbox[0];
#endif

#else
    modelt_accessor_mailbox& mbox = reinterpret_cast<modelt_accessor_mailbox&>(mailbox.acc);
    mbox.sync_kernels();
#endif
}

inline void unhalt_tensix() {
    clobber_all_memory();
    volatile uint* pc_buf = reinterpret_cast<volatile uint*>(PC_BUF_BASE);
    pc_buf[0] = TENSIX_UNHALT_VAL;
}

inline void memory_write(uint addr, uint value) {
#ifndef MODELT
    volatile uint* buf = reinterpret_cast<volatile uint*>(addr);
    buf[0] = value;
#endif
}

inline uint memory_read(uint addr) {
#ifndef MODELT
    volatile uint* buf = reinterpret_cast<volatile uint*>(addr);
    return buf[0];
#else
    // FWASSERT("memory_read in modelt not supported yet", 0);
    return 0;
#endif
}

inline void execute_instruction(vptr_uint instrn_buffer, unsigned int instruction) {
    ex_push_insn(instrn_buffer, instruction);
}

inline void thcon_flush_dma(vptr_uint instrn_buffer, uint arg) {
    execute_instruction(instrn_buffer, INSTRN_FLUSH_DMA(arg));
}

//////
// Address index is in 32b quants, offset index is in 16b quants, data index is in 32b quants
inline void thcon_load_ind(
    vptr_uint instrn_buffer, uint base_addr_index, uint dst_data_index, uint offset_index, uint autoinc, uint size) {
    uint instrn_arg;

    instrn_arg =
        0x0 | (base_addr_index) | (dst_data_index << 6) | (autoinc << 12) | (offset_index << 14) | (size << 22);
    execute_instruction(instrn_buffer, INSTRN_LOAD_IND(instrn_arg));
}

inline void thcon_store_ind(
    vptr_uint instrn_buffer,
    uint base_index,
    uint src_data_index,
    uint offset_index,
    uint autoinc,
    uint mode_32b_16B,
    bool l0_l1_sel,
    uint tile_mode) {
    uint instrn_arg;
    uint sel = l0_l1_sel;

    instrn_arg = 0x0 | (base_index) | (src_data_index << 6) | (autoinc << 12) | (offset_index << 14) |
                 (tile_mode << 21) | (mode_32b_16B << 22) | (sel << 23);
    execute_instruction(instrn_buffer, INSTRN_STORE_IND(instrn_arg));
}

inline void thcon_incr_get_ptr(
    vptr_uint instrn_buffer,
    uint mem_addr_index,
    uint data_reg_index,
    uint incr_val,
    uint wrap_val,
    bool rd_wr,
    bool l0_l1_sel) {
    uint instrn_arg;
    uint sel = l0_l1_sel;
    uint rd_wr_sel = (uint)rd_wr;

    // Below, src_data_index is shifted 8 times instead of 6 in order to convert from 16B quants to 32b quants which
    // instrn expects
    instrn_arg = 0x0 | (mem_addr_index) | (data_reg_index << 6) | (wrap_val << 14) | (rd_wr_sel << 12) |
                 (incr_val << 18) | (sel << 23);
    execute_instruction(instrn_buffer, INSTRN_AT_INCR_GET_PTR(instrn_arg));
}

inline void thcon_incr_get_ptr_noinc(
    vptr_uint instrn_buffer,
    uint mem_addr_index,
    uint data_reg_index,
    uint incr_val,
    uint wrap_val,
    bool rd_wr,
    bool l0_l1_sel) {
    uint instrn_arg;
    uint sel = l0_l1_sel;
    uint rd_wr_sel = (uint)rd_wr;

    instrn_arg = 0x0 | (mem_addr_index) | (data_reg_index << 6) | (wrap_val << 14) | (rd_wr_sel << 12) |
                 (incr_val << 18) | (1 << 22) | (sel << 23);
    execute_instruction(instrn_buffer, INSTRN_AT_INCR_GET_PTR(instrn_arg));
}

inline void thcon_reg_to_flops(
    vptr_uint instrn_buffer,
    uint mode_32b_16B,
    uint reg_index,
    uint flop_index,
    uint target_select = 0,
    uint byte_offset = 0) {
    int instrn_arg;
    instrn_arg =
        0x0 | reg_index | (flop_index << 6) | (byte_offset << 18) | (target_select << 20) | (mode_32b_16B << 22);
    execute_instruction(instrn_buffer, INSTRN_MV_REG_TO_FLOPS(instrn_arg));
}

inline void ex_clear_dvalid(uint clear_ab, uint reset, vptr_uint instrn_buffer) {
    int instrn_arg;
    instrn_arg = 0x0 | (reset & 0x1) | (clear_ab << 22);
    execute_instruction(instrn_buffer, INSTRN_CLEAR_DVALID(instrn_arg));
}

inline void ex_sem_init(uint semaphore, uint max_value, uint init_value, vptr_uint instrn_buffer) {
    int instrn_arg;
    instrn_arg = 0x0 | (0x1 << (semaphore + 2)) | (init_value << 16) | (max_value << 20);
    execute_instruction(instrn_buffer, INSTRN_SEMINIT(instrn_arg));
}

/**
 * Atomic Compare-and-Swap
 *
 * @param address_register_index Which register points to the memory location that will be changed.
 *                               This address refers to 16B addressing space.
 * @param data_register_index    This seems to be unused.
 *                               TODO: Investigate why this is not used.
 * @param word_select            Which of the four 32-bit addresses within the 16-byte word referenced
 *                               by address_register_index will be used.
 *                               Effectively, the address of the memory to be changed is split between
 *                               this parameter and address_register_index.
 * @param compare_value          A 4-bit number that controls when the CAS operation can complete.
 *                               Once the CAS instruction is invoked, it will block until the memory
 *                               pointed two by address_register_idnex and word_select becomes equal
 *                               to this "compare_value".
 * @param swap_value             The 4-bit number that will be written to memory.
 */
inline void thcon_cas(
    vptr_uint instrn_buffer,
    uint8_t address_register_index,
    uint8_t data_register_index,
    uint8_t word_select,
    uint8_t compare_value,
    uint8_t swap_value,
    bool mem_heirarchy_select) {
    const uint32_t instrn_arg = ((uint32_t)address_register_index << 0) | ((uint32_t)data_register_index << 6) |
                                ((uint32_t)word_select << 12) | ((uint32_t)compare_value << 14) |
                                ((uint32_t)swap_value << 18) | ((uint32_t)mem_heirarchy_select << 23);

    execute_instruction(instrn_buffer, INSTRN_AT_CAS(instrn_arg));
}

inline void thcon_at_swap(
    vptr_uint instrn_buffer, uint mem_addr_index, uint src_data_index, uint mask_16b, bool l0_l1_sel) {
    uint instrn_arg;
    uint sel = l0_l1_sel;
    // Below, src_data_index is shifted 8 times instead of 6 in order to convert from 16B quants to 32b quants which
    // instrn expects
    instrn_arg = 0x0 | (mem_addr_index) | (src_data_index << 8) | (mask_16b << 14) | (sel << 23);
    execute_instruction(instrn_buffer, INSTRN_AT_SWAP(instrn_arg));
}

///////
// Address is in 16b quants
inline void thcon_write_16b_reg(
    vptr_uint instrn_buffer, uint addr /* 16b quants */, uint val, bool set_signals_mode = false) {
    uint setdma_payload;
    uint instrn_arg;

    setdma_payload = val;
    instrn_arg = 0x0 | addr | (setdma_payload << 8);
    if (set_signals_mode) {
        instrn_arg |= (1 << 7);
    }
    execute_instruction(instrn_buffer, INSTRN_SET_DMA_REG(instrn_arg));
}

///////
// Address is in 16b quants
inline void thcon_sigwrite_16b_reg(vptr_uint instrn_buffer, uint addr /* 16b quants */, uint sig_addr) {
    uint instrn_arg;

    instrn_arg = 0x0 | addr | (1 << 7) | (sig_addr << 8);
    execute_instruction(instrn_buffer, INSTRN_SET_DMA_REG(instrn_arg));
}

///////
// Address is in 32b quants
inline void thcon_write_32b_reg(uint addr /*32b quants*/, uint val) {
    volatile uint* regfile = reinterpret_cast<uint*>(REGFILE_BASE);
    regfile[addr] = val;
}

///////
// Address is in 16B quants
inline void thcon_write_16B_reg(uint addr, const uint* val) {
    uint addr_bot;
    addr_bot = addr << 2;
    int i;

    for (i = 0; i < 4; ++i) {
        thcon_write_32b_reg(addr_bot + i, val[i]);
    }
}

inline void thcon_set_packer_section_conf(vptr_uint instrn_buf, uint rowstart_size, uint exp_size) {
    // printf("CONFIG: setting rowstart to %d and exp size to %d\n",rowstart_size,exp_size);
    // FIXME MT: just use a free register for now
    uint reg_index = 9;
    uint flop_index = 4;
    uint reg;
    reg = 0x0 | rowstart_size | (exp_size << 16);
    thcon_write_32b_reg(reg_index, reg);
    thcon_reg_to_flops(instrn_buf, 1, reg_index, flop_index);
}

inline void thcon_write_tile_addr(vptr_uint instrn_buf, uint reg_index, uint unpacker_id) {
    //    printf("CONFIG: setting TILE ADDRESS to %d\n",tile_addr);
    // uint reg_index  = 0;
    // uint flop_index = 3+(4*2);

    // thcon_write_32b_reg(reg_index, tile_addr, NULL);
    thcon_reg_to_flops(instrn_buf, 1, reg_index, 3 + (4 * 2) + TDMA_FLOPREG_IDX_BASE(unpacker_id));
    // WAIT_SHORT;
}

inline void thcon_set_packer_l1_dest_addr(vptr_uint instrn_buf, uint l1_dest_addr) {
    // printf("CONFIG: setting PACKER L1 destination to %d\n",l1_dest_addr);
    uint reg_index = 9;
    uint flop_index = 1 + (4 * 1);
    uint reg;
    reg = l1_dest_addr;
    thcon_write_32b_reg(reg_index, reg);
    thcon_reg_to_flops(instrn_buf, 1, reg_index, flop_index);
}

inline void thcon_set_packer_misc_conf(
    vptr_uint instrn_buf, uint disable_zcomp, uint in_data_format, uint out_data_format, uint dest_digest_offset) {
    // printf("CONFIG: setting PACKER disable zcomp to %d IN data format to %d OUT data format to %d DIGEST_DESC offset
    // to %d\n",disable_zcomp,in_data_format,out_data_format,dest_digest_offset);
    uint reg_index = 9;
    uint flop_index = 2 + (4 * 1);
    uint reg;
    reg = 0x0 | disable_zcomp | (in_data_format << 8) | (out_data_format << 4) | (dest_digest_offset << 12);
    thcon_write_32b_reg(reg_index, reg);
    thcon_reg_to_flops(instrn_buf, 1, reg_index, flop_index);
}

inline void thcon_set_unpacker_misc_conf(vptr_uint instrn_buf, uint out_data_format, uint unpacker_id) {
    // printf("CONFIG: setting UNPACK OUT data format to %d", out_data_format);
    uint reg_index = 0;
    uint flop_index = 3 + (4 * 1) + TDMA_FLOPREG_IDX_BASE(unpacker_id);
    uint reg;
    reg = 0x0 | out_data_format;
    thcon_write_32b_reg(reg_index, reg);
    thcon_reg_to_flops(instrn_buf, 1, reg_index, flop_index);
}

/////
// Register index is in BIG registers (16B quants)
inline void thcon_set_descriptor(vptr_uint instrn_buf, uint reg_index, uint unpacker_id) {
    uint reg_index_fixed;
    reg_index_fixed = reg_index << 2;
    thcon_reg_to_flops(instrn_buf, 0, reg_index_fixed, 0 + TDMA_FLOPREG_IDX_BASE(unpacker_id));
}

inline tile_descriptor_u thcon_build_descriptor(
    uint tile_id, uint tile_type, uint x_dim, uint y_dim, uint z_dim, uint w_dim, uint digest_type, uint digest_size) {
    tile_descriptor_u td;

    tile_id &= bitmask<uint>(
        16);  // existing firmware passes in a 32-bit tile_id. It's incorrect but must be supported for now.

    td.val[0] = pack_field(tile_id, 16, 8) | pack_field(tile_type, 8, 0) | pack_field(x_dim, 8, 0, 24);
    td.val[1] = pack_field(x_dim, 8, 8, 0) | pack_field(y_dim, 16, 0, 8) | pack_field(z_dim, 8, 0, 24);
    td.val[2] = pack_field(z_dim, 8, 8, 0) | pack_field(w_dim, 16, 0, 8);
    td.val[3] = pack_field(digest_type, 8, 24) | pack_field(digest_size, 8, 16);

    return td;
}

/////
// Register index is in BIG registers (16B quants)
inline void thcon_write_descriptor_to_reg(
    uint reg_index,
    uint tile_id,
    uint tile_type,
    uint x_dim,
    uint y_dim,
    uint z_dim,
    uint w_dim,
    uint digest_type,
    uint digest_size) {
    tile_descriptor_u td =
        thcon_build_descriptor(tile_id, tile_type, x_dim, y_dim, z_dim, w_dim, digest_type, digest_size);

    thcon_write_16B_reg(reg_index, td.val);
}

/////
inline void thcon_write_descriptor_to_l1(
    uint addr,
    uint tile_id,
    uint tile_type,
    uint x_dim,
    uint y_dim,
    uint z_dim,
    uint w_dim,
    uint digest_type,
    uint digest_size) {
    volatile uint* ptr = reinterpret_cast<volatile uint*>(addr);

    tile_descriptor_u td =
        thcon_build_descriptor(tile_id, tile_type, x_dim, y_dim, z_dim, w_dim, digest_type, digest_size);

    ptr[0] = td.val[0];
    ptr[1] = td.val[1];
    ptr[2] = td.val[2];
    ptr[3] = td.val[3];
}

/////
// Breakpoint functions
//
// localparam RESUME = 3'b000, SET = 3'b001, CLEAR = 3'b010, DATASEL = 3'b011, SET_COND = 3'b100, CLEAR_COND = 3'b101;
#define BKPT_CMD_RESUME 0x0
#define BKPT_CMD_SET 0x1
#define BKPT_CMD_CLEAR 0x2
#define BKPT_CMD_DATASEL 0x3
#define BKPT_CMD_SET_COND 0x4
#define BKPT_CMD_CLEAR_COND 0x5

#define BKPT_CMD_PAYLOAD(thread, cmd, data) ((thread << 31) | (cmd << 28) | data)
#define BKPT_CMD_ID_PAYLOAD(thread, cmd, id, data) ((thread << 31) | (cmd << 28) | (id << 26) | data)

inline void breakpoint_set(uint thread, uint bkpt_index, bool pc_valid, uint pc = 0) {
    memory_write(
        RISCV_DEBUG_REG_BREAKPOINT_CTRL, BKPT_CMD_ID_PAYLOAD(thread, BKPT_CMD_SET, bkpt_index, (pc_valid << 21 | pc)));
}

inline void breakpoint_clear(uint thread, uint bkpt_index) {
    memory_write(RISCV_DEBUG_REG_BREAKPOINT_CTRL, BKPT_CMD_ID_PAYLOAD(thread, BKPT_CMD_CLEAR, bkpt_index, 0));
}

// Set which breakpoint will be returning data
inline void breakpoint_set_data(uint thread, uint bkpt_index, uint data_index) {
    memory_write(
        RISCV_DEBUG_REG_BREAKPOINT_CTRL, BKPT_CMD_ID_PAYLOAD(thread, BKPT_CMD_DATASEL, bkpt_index, data_index));
}

inline void breakpoint_set_condition_op(uint thread, uint bkpt_index, uint opcode, uint opcode_mask = 0xFF) {
    memory_write(
        RISCV_DEBUG_REG_BREAKPOINT_CTRL,
        BKPT_CMD_ID_PAYLOAD(thread, BKPT_CMD_SET_COND, bkpt_index, (opcode_mask << 8 | opcode)));
}

inline void breakpoint_clear_condition_op(uint thread, uint bkpt_index) {
    memory_write(RISCV_DEBUG_REG_BREAKPOINT_CTRL, BKPT_CMD_ID_PAYLOAD(thread, BKPT_CMD_CLEAR_COND, bkpt_index, 0));
}

inline void breakpoint_set_condition_loop(uint thread, uint bkpt_index, uint loop) {
    memory_write(
        RISCV_DEBUG_REG_BREAKPOINT_CTRL,
        BKPT_CMD_ID_PAYLOAD(thread, BKPT_CMD_SET_COND, bkpt_index, (0x1 << 16) | loop));
}

inline void breakpoint_clear_condition_loop(uint thread, uint bkpt_index) {
    memory_write(
        RISCV_DEBUG_REG_BREAKPOINT_CTRL, BKPT_CMD_ID_PAYLOAD(thread, BKPT_CMD_CLEAR_COND, bkpt_index, 0x1 << 16));
}

inline void breakpoint_set_condition_other_thread(uint thread, uint bkpt_index) {
    memory_write(
        RISCV_DEBUG_REG_BREAKPOINT_CTRL, BKPT_CMD_ID_PAYLOAD(thread, BKPT_CMD_SET_COND, bkpt_index, (0x2 << 16)));
}

inline void breakpoint_clear_condition_other_thread(uint thread, uint bkpt_index) {
    memory_write(
        RISCV_DEBUG_REG_BREAKPOINT_CTRL, BKPT_CMD_ID_PAYLOAD(thread, BKPT_CMD_CLEAR_COND, bkpt_index, 0x2 << 16));
}

inline void breakpoint_resume_execution(uint thread) {
    memory_write(RISCV_DEBUG_REG_BREAKPOINT_CTRL, BKPT_CMD_PAYLOAD(thread, BKPT_CMD_RESUME, 0));
}

// return status for a specific breakpoint
inline uint breakpoint_status(uint thread, uint bkpt_index) {
    uint status = memory_read(RISCV_DEBUG_REG_BREAKPOINT_STATUS);
    return (status >> ((bkpt_index + thread * 4) * 4)) & 0xF;
}

// return status for all breakpoints
inline uint breakpoint_status() {
    uint status = memory_read(RISCV_DEBUG_REG_BREAKPOINT_STATUS);
    return status;
}

inline uint breakpoint_data() {
    volatile uint* ptr = reinterpret_cast<volatile uint*>(RISCV_DEBUG_REG_BREAKPOINT_DATA);
    *ptr = 0;  // Ensure ordering with any previous control writes
    return *ptr;
}

// Read debug array functions
#define SRCA_ARRAY_ID 0x0
#define SRCB_ARRAY_ID 0x1
#define DEST_ARRAY_ID 0x2
#define MAX_EXP_ARRAY_ID 0x3
#define DBG_RD_CMD_PAYLOAD(thread, array_id, addr) ((thread << 19) | (array_id << 16) | addr)

inline void dbg_dump_array_enable() { memory_write(RISCV_DEBUG_REG_DBG_ARRAY_RD_EN, 1); }

inline void dbg_dump_array_disable() {
    // Invalidate array_id to invalid to set logic rd_en to 0
    memory_write(RISCV_DEBUG_REG_DBG_ARRAY_RD_CMD, DBG_RD_CMD_PAYLOAD(0, 0xF, 0));
    memory_write(RISCV_DEBUG_REG_DBG_ARRAY_RD_EN, 0);
}

inline void dbg_dump_array_rd_cmd(uint thread, uint array_id, uint addr) {
    memory_write(RISCV_DEBUG_REG_DBG_ARRAY_RD_CMD, DBG_RD_CMD_PAYLOAD(thread, array_id, addr));
    volatile uint dummy_wait;
    volatile uint* dummy_wait_ptr = &dummy_wait;
    *dummy_wait_ptr = memory_read(RISCV_DEBUG_REG_DBG_ARRAY_RD_CMD);
}

inline void dbg_dump_array_to_l1(uint thread, uint addr) {
    // This will trigger debug bus input to L1
}

inline void dbg_instrn_buf_wait_for_ready() {
    while (1) {
        volatile uint status = memory_read(RISCV_DEBUG_REG_INSTRN_BUF_STATUS);
        if (status == 0x77) {
            break;
        }
    }
}

inline void dbg_instrn_buf_set_override_en() {
    // Set override enable
    memory_write(RISCV_DEBUG_REG_INSTRN_BUF_CTRL0, 0x7);
}

inline void dbg_instrn_buf_push_instrn(uint instrn) {
    // write instrn
    memory_write(RISCV_DEBUG_REG_INSTRN_BUF_CTRL1, instrn);
    // write -> 1
    memory_write(RISCV_DEBUG_REG_INSTRN_BUF_CTRL0, 0x17);
    // write -> 0
    memory_write(RISCV_DEBUG_REG_INSTRN_BUF_CTRL0, 0x07);
}

inline void dbg_instrn_buf_clear_override_en() {
    // Set override enable
    memory_write(RISCV_DEBUG_REG_INSTRN_BUF_CTRL0, 0x0);
}

extern "C" void wzerorange(uint32_t* start, uint32_t* end);
inline void wzeromem(uint32_t start, uint32_t len) { wzerorange((uint32_t*)start, (uint32_t*)(start + len)); }
