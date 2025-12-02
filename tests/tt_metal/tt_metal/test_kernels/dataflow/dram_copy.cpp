// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"
// #include "noc_functions.h"
#include "noc_parameters.h"
#include "tt-2xx/quasar/overlay/cmdbuff_api.hpp"
#include "tt-2xx/quasar/noc_nonblocking_api.h"

#define TT_CLUSTER_CTRL_REG_MAP_BASE_ADDR (0x03000000)
#define TT_CLUSTER_CTRL_SCRATCH_0__REG_OFFSET (0x00000040)
#define SCRATCH_0_OFFSET TT_CLUSTER_CTRL_SCRATCH_0__REG_OFFSET
#define PERIPH_PORT_BASE_ADDR (uint64_t)TT_CLUSTER_CTRL_REG_MAP_BASE_ADDR
#define WRITE_PERIPH_PORT32(offset, val) ((*((volatile uint32_t*)((PERIPH_PORT_BASE_ADDR + offset)))) = (val))
#define WRITE_SCRATCH(num, val) (WRITE_PERIPH_PORT32(SCRATCH_0_OFFSET + (0x8 * num), val))

/**
 * NOC APIs are prefixed w/ "ncrisc" (legacy name) but there's nothing NCRISC specific, they can be used on BRISC or
 * other RISCs Any two RISC processors cannot use the same CMD_BUF non_blocking APIs shouldn't be mixed with slow noc.h
 * APIs explicit flushes need to be used since the calls are non-blocking
 * */
void kernel_main() {
    std::uint32_t l1_buffer_addr = get_arg_val<uint32_t>(0);

    std::uint32_t dram_buffer_src_addr = get_arg_val<uint32_t>(1);
    std::uint32_t dram_src_bank_id = get_arg_val<uint32_t>(2);

    std::uint32_t dram_buffer_dst_addr = get_arg_val<uint32_t>(3);
    std::uint32_t dram_dst_bank_id = get_arg_val<uint32_t>(4);

    std::uint32_t dram_buffer_size = get_arg_val<uint32_t>(5);

#if defined(SIGNAL_COMPLETION_TO_DISPATCHER)
    // We will assert later. This kernel will hang.
    // Need to signal completion to dispatcher before hanging so that
    // Dispatcher Kernel is able to finish.
    // Device Close () requires fast dispatch kernels to finish.
#if defined(COMPILE_FOR_ERISC)
    tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE);
#else
    tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE);
#endif
    uint64_t dispatch_addr = NOC_XY_ADDR(
        NOC_X(mailboxes->go_message.master_x),
        NOC_Y(mailboxes->go_message.master_y),
        DISPATCH_MESSAGE_ADDR + NOC_STREAM_REG_SPACE_SIZE * mailboxes->go_message.dispatch_message_offset);
    noc_fast_write_dw_inline<DM_DEDICATED_NOC>(
        noc_index,
        NCRISC_AT_CMD_BUF,
        1 << REMOTE_DEST_BUF_WORDS_FREE_INC,
        dispatch_addr,
        0xF,  // byte-enable
        NOC_UNICAST_WRITE_VC,
        false,  // mcast
        true    // posted
    );
#endif

    DPRINT << "dprint test" << ENDL();

    uint32_t noc_id_reg = MY_NOC_ENCODING(0);
    DPRINT << "noc_id_reg : " << noc_id_reg << ENDL();

    uint32_t my_x = noc_id_reg & NOC_NODE_ID_MASK;
    uint32_t my_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    DPRINT << "my_x : " << my_x << " my_y : " << my_y << ENDL();

    // noc_read(0, 0, 0, 0, 0, 0);
    // uint32_t noc,
    // uint64_t src_coordinate,
    // uint64_t src_addr,
    // uint64_t dest_coordinate,
    // uint64_t dest_addr,
    // uint64_t len_bytes,
    // uint32_t transaction_id,
    // uint32_t static_vc,
    // uint32_t cmd_buf) {
    //     NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, dest_addr);
    //     NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)src_addr);
    //     NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_MID, (uint32_t)(src_addr >> 32) &
    //     NOC_PCIE_MASK); NOC_CMD_BUF_WRITE_REG(
    //         noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(src_addr >>
    //         NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    //     NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN, len_bytes);
    //     NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

    // uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(0, 0, NOC_NODE_ID);
    // uint32_t my_x = noc_id_reg & NOC_NODE_ID_MASK;
    // uint32_t my_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    // uint64_t my_coords = NOC_XY_COORD(my_x, my_y);

    // DPRINT << "X : " << my_x << " Y : " << my_y << ENDL();
    DPRINT << "mX : " << mx() << " mY : " << my() << ENDL();

    std::uint64_t dram_buffer_src_noc_addr =
        NOC_XY_ADDR(0, 0, 0x100000);  // get_noc_addr_from_bank_id<true>(dram_src_bank_id, dram_buffer_src_addr);

    // noc_read(
    //     0,
    //     ((uint32_t)(dram_buffer_src_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK),
    //     dram_buffer_src_noc_addr,
    //     my_coords,
    //     l1_buffer_addr,
    //     dram_buffer_size,
    //     0,
    //     0,
    //     0);

    // all_noc_reads_flushed(0);

    // DRAM NOC src address
    DPRINT << CMDBUF_TR_ACK(0) << ENDL();
    noc_async_read(dram_buffer_src_noc_addr, 0x100000, dram_buffer_size);
    DPRINT << "Read is issued" << ENDL();
    DPRINT << CMDBUF_TR_ACK(0) << ENDL();

    // WRITE_PERIPH_PORT32(C0_POSTCODE , POSTCODE_FAIL);
    DPRINT << "Write to reg feed beef" << ENDL();
    WRITE_SCRATCH(0, 0xfeedbeef);
    uint32_t test = CMDBUF_TR_ACK(0);

    // __builtin_riscv_ttrocc_cmdbuf_tr_ack(0);

    // DPRINT << __builtin_riscv_ttrocc_cmdbuf_tr_ack(0) << ENDL();

    noc_async_read_barrier();
    DPRINT << "Read is Done " << ENDL();

    // DRAM NOC dst address
    // std::uint64_t dram_buffer_dst_noc_addr = get_noc_addr_from_bank_id<true>(dram_dst_bank_id, dram_buffer_dst_addr);
    noc_async_write(0x100000, dram_buffer_src_noc_addr, dram_buffer_size);

    DPRINT << "Acks received : " << NOC_STATUS_READ_REG(0, NIU_MST_WR_ACK_RECEIVED) << ENDL();
    DPRINT << "Writes sent : " << NOC_STATUS_READ_REG(0, NIU_MST_NONPOSTED_WR_REQ_SENT) << ENDL();
    DPRINT << "Var : " << noc_nonposted_writes_acked[0] << ENDL();
    DPRINT << "Var : " << noc_nonposted_writes_num_issued[0] << ENDL();
    DPRINT << "Var : " << noc_reads_num_issued[0] << ENDL();

    noc_async_write_barrier();
    DPRINT << "write is Done " << ENDL();
}
