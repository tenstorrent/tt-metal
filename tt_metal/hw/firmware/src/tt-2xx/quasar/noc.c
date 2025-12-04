
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "noc.h"
#include <stdint.h>
#include <stdbool.h>
#include "noc_parameters.h"

// generated code
#include "registers/noc_niu_reg.h"
#include "registers/noc_config_reg.h"
#include "registers/noc_status_reg.h"

#ifdef TB_NOC

#include "noc_api_dpi.h"

#else

#define NOC_WRITE_REG(addr, val)                                      \
    ((*((volatile uint32_t*)(noc_get_cmd_buf() * NOC_CMD_BUF_OFFSET + \
                             noc_get_active_instance() * NOC_INSTANCE_OFFSET + ((uintptr_t)addr)))) = (val))
#define NOC_READ_REG(addr)                                                                                             \
    (*((volatile uint32_t*)(noc_get_cmd_buf() * NOC_CMD_BUF_OFFSET + noc_get_active_instance() * NOC_INSTANCE_OFFSET + \
                            ((uintptr_t)addr))))

#endif

#ifdef ARC_FW_NOC
#include "arc_fw_noc.h"
#endif

static uint32_t active_cmd_buf = 0;
static uint32_t active_noc_instance = 0;

void noc_set_cmd_buf(uint32_t cmd_buf_id) {
#ifdef TB_NOC
    api_set_active_cmd_buf(cmd_buf_id);
#else
    active_cmd_buf = cmd_buf_id;
#endif
}

uint32_t noc_get_cmd_buf() {
#ifdef TB_NOC
    return api_get_active_cmd_buf();
#else
    return active_cmd_buf;
#endif
}

void noc_set_active_instance(uint32_t noc_id) {
#ifdef TB_NOC
    api_set_active_noc_instance(noc_id);
#else
    active_noc_instance = noc_id;
#endif
}

uint32_t noc_get_active_instance() {
#ifdef TB_NOC
    return api_get_active_noc_instance();
#else
    return active_noc_instance;
#endif
}

// Define the struct
typedef struct {
    bool src_local;
    uint32_t src_coordinate;
    uint64_t src_addr;
    uint32_t dst_coordinate;
    uint64_t dst_addr;
    uint32_t size;
    bool linked;
    bool posted;
    bool write_be;
    uint32_t be[8];
    uint32_t static_vc;
    uint32_t resp_static_vc;
    bool multicast;
    bool multicast_mode;
    bool src_include;
    uint32_t multicast_lo;
    uint32_t multicast_hi;
    uint32_t transaction_id;
    uint32_t cmd_select;
    uint64_t val;
    bool write_inline;
    bool l1_acc;
    uint32_t data_format;
    bool disable_saturation;
    bool atomic;
    uint32_t atomic_data;
    uint32_t wrap;
    uint32_t port_req_mask;
    bool dynamic_routing_enable;
    bool cmd_snoop;
    bool cmd_flush;
    bool cmd_mem_rd_drop_ack;
} TransferParams;

void noc_transfer(TransferParams* p) {
    // if (size > 0x4000) {
    //    LOGC_("Size of noc bytes is over 16KB ! Req sent should be incremented\n");
    // }

    while (!noc_command_ready(p->cmd_select));

    noc_set_cmd_buf(p->cmd_select);

    if (p->write_inline) {
        p->src_local = true;
        NOC_WRITE_REG(NOC_NIU_TARG_ADDR_LO_REG_ADDR, (uint32_t)(p->dst_addr & 0xFFFFFFFF));
        NOC_WRITE_REG(NOC_NIU_TARG_ADDR_MID_REG_ADDR, (uint32_t)(p->dst_addr >> 32));
        NOC_WRITE_REG(NOC_NIU_TARG_ADDR_HI_REG_ADDR, p->dst_coordinate);

        NOC_WRITE_REG(NOC_NIU_INLINE_DATA_LO_REG_ADDR, (uint32_t)(p->val & 0xFFFFFFFF));
        NOC_WRITE_REG(NOC_NIU_INLINE_DATA_HI_REG_ADDR, (uint32_t)(p->val >> 32));
        NOC_WRITE_REG(NOC_NIU_AT_LEN_REG_ADDR, (uint32_t)(p->size));

    } else {
        NOC_WRITE_REG(NOC_NIU_RET_ADDR_LO_REG_ADDR, (uint32_t)(p->dst_addr & 0xFFFFFFFF));
        NOC_WRITE_REG(NOC_NIU_RET_ADDR_MID_REG_ADDR, (uint32_t)(p->dst_addr >> 32));
        NOC_WRITE_REG(NOC_NIU_RET_ADDR_HI_REG_ADDR, p->dst_coordinate);

        NOC_WRITE_REG(NOC_NIU_TARG_ADDR_LO_REG_ADDR, (uint32_t)(p->src_addr & 0xFFFFFFFF));
        NOC_WRITE_REG(NOC_NIU_TARG_ADDR_MID_REG_ADDR, (uint32_t)(p->src_addr >> 32));
        NOC_WRITE_REG(NOC_NIU_TARG_ADDR_HI_REG_ADDR, p->src_coordinate);

        if (p->write_be) {
            p->src_local = true;
            NOC_WRITE_REG(NOC_NIU_BYTE_ENABLE_0__REG_ADDR, p->be[0]);
            NOC_WRITE_REG(NOC_NIU_BYTE_ENABLE_1__REG_ADDR, p->be[1]);
            NOC_WRITE_REG(NOC_NIU_BYTE_ENABLE_2__REG_ADDR, p->be[2]);
            NOC_WRITE_REG(NOC_NIU_BYTE_ENABLE_3__REG_ADDR, p->be[3]);
            NOC_WRITE_REG(NOC_NIU_BYTE_ENABLE_4__REG_ADDR, p->be[4]);
            NOC_WRITE_REG(NOC_NIU_BYTE_ENABLE_5__REG_ADDR, p->be[5]);
            NOC_WRITE_REG(NOC_NIU_BYTE_ENABLE_6__REG_ADDR, p->be[6]);
            NOC_WRITE_REG(NOC_NIU_BYTE_ENABLE_7__REG_ADDR, p->be[7]);

        } else if (p->atomic) {
            NOC_NIU_AT_LEN_reg_u at_len_value;
            at_len_value.val = 0x0;

            at_len_value.f.at_ind_32 = (p->src_addr >> 2) & 0x3;
            at_len_value.f.at_wrap = p->wrap;
            at_len_value.f.at_ind_32_src = 0x0;
            at_len_value.f.at_instrn = NOC_AT_INS_INCR_GET;

            NOC_WRITE_REG(NOC_NIU_AT_LEN_REG_ADDR, at_len_value.val);
            NOC_WRITE_REG(NOC_NIU_AT_DATA_REG_ADDR, p->atomic_data);

        } else {
            NOC_WRITE_REG(NOC_NIU_AT_LEN_REG_ADDR, (uint32_t)(p->size));
        }
    }

    if (p->l1_acc) {
        NOC_NIU_L1_ACC_AT_INSTRN_reg_u l1_acc_value;
        l1_acc_value.val = 0x0;

        l1_acc_value.f.data_format = p->data_format;
        l1_acc_value.f.disable_sat = p->disable_saturation;
        l1_acc_value.f.at_instrn = NOC_AT_INS_ACC;

        NOC_WRITE_REG(NOC_NIU_L1_ACC_AT_INSTRN_REG_ADDR, l1_acc_value.val);
    }

    NOC_NIU_CMD_LO_reg_u cmd_value_lo;
    NOC_NIU_CMD_HI_reg_u cmd_value_hi;

    // dangerous it latched the values
    cmd_value_lo.val = NOC_READ_REG(NOC_NIU_CMD_LO_REG_ADDR);
    cmd_value_hi.val = NOC_READ_REG(NOC_NIU_CMD_HI_REG_ADDR);

    cmd_value_lo.f.cmd_at_cpy_bit = p->atomic;
    cmd_value_lo.f.cmd_rw_bit = p->src_local;
    cmd_value_lo.f.cmd_wr_be_bit = p->write_be;
    cmd_value_lo.f.cmd_wr_inline_bit = p->write_inline;
    cmd_value_lo.f.cmd_resp_marked_bit = !p->posted;
    cmd_value_lo.f.cmd_linked_bit = p->linked;
    cmd_value_lo.f.cmd_port_req_mask = p->port_req_mask;
    cmd_value_lo.f.cmd_l1_acc_at_en = p->l1_acc;
    cmd_value_lo.f.cmd_static_vc = p->static_vc;
    cmd_value_lo.f.resp_static_vc = p->resp_static_vc;
    cmd_value_hi.f.cmd_pkt_tag_id = p->transaction_id;

    if (p->multicast) {
        cmd_value_lo.f.cmd_brcst_bit = p->multicast;
        cmd_value_lo.f.cmd_path_reserve = 1;

        NOC_NIU_BRCST_LO_reg_u brcst_value_lo;

        brcst_value_lo.val = 0x0;

        brcst_value_lo.f.brcst_xy_bit = p->multicast_mode;
        brcst_value_lo.f.brcst_src_include = p->src_include;
        // These 3 signals are internal to the NOC and should not be set by the user
        // brcst_value_lo.f.brcst_ctrl_state
        // brcst_value_lo.f.brcst_active_node
        // brcst_value_lo.f.brcst_end_node
        //
        if (p->multicast_lo != 0x0) {
            uint32_t shifted_value = p->multicast_lo;

            shifted_value &= 0xFFFFF800;  // Mask for bits 11 to 31

            NOC_WRITE_REG(
                NOC_NIU_BRCST_LO_REG_ADDR,
                (uint32_t)(shifted_value | (brcst_value_lo.val & 0xEFF)));  // upper 10 bits not used.
            NOC_WRITE_REG(NOC_NIU_BRCST_HI_REG_ADDR, (uint32_t)(p->multicast_hi));
        } else {
            NOC_WRITE_REG(NOC_NIU_BRCST_LO_REG_ADDR, brcst_value_lo.val);
            NOC_WRITE_REG(NOC_NIU_BRCST_HI_REG_ADDR, 0x0);
        }
    } else {
        cmd_value_lo.f.cmd_brcst_bit = p->multicast;
        cmd_value_lo.f.cmd_path_reserve = 0;
    }

    // LOG_C(
    //     "Writing CMD_BUF: %d, CMD_LO: cmd_at_cpy_bit=%d,  cmd_rw_bit=%d,  cmd_wr_be_bit=%d,  cmd_wr_inline_bit=%d, "
    //     "cmd_resp_marked_bit=%d,  cmd_brcst_bit=%d,  cmd_linked_bit=%d,  cmd_path_reserve=%d, cmd_mem_rd_drop_ack=%d,
    //     " " cmd_dyna_routing_en=%d,  cmd_l1_acc_at_en=%d,  cmd_flush_bit=%d,  cmd_snoop_bit=%d,  cmd_static_vc=%d,  "
    //     "resp_static_vc=%d, cmd_port_req_mask=%d \n",
    //     p->cmd_select,
    //     cmd_value_lo.f.cmd_at_cpy_bit,
    //     cmd_value_lo.f.cmd_rw_bit,
    //     cmd_value_lo.f.cmd_wr_be_bit,
    //     cmd_value_lo.f.cmd_wr_inline_bit,
    //     cmd_value_lo.f.cmd_resp_marked_bit,
    //     cmd_value_lo.f.cmd_brcst_bit,
    //     cmd_value_lo.f.cmd_linked_bit,
    //     cmd_value_lo.f.cmd_path_reserve,
    //     cmd_value_lo.f.cmd_mem_rd_drop_ack,
    //     cmd_value_lo.f.cmd_dyna_routing_en,
    //     cmd_value_lo.f.cmd_l1_acc_at_en,
    //     cmd_value_lo.f.cmd_flush_bit,
    //     cmd_value_lo.f.cmd_snoop_bit,
    //     cmd_value_lo.f.cmd_static_vc,
    //     cmd_value_lo.f.resp_static_vc,
    //     cmd_value_lo.f.cmd_port_req_mask);

    // LOG_C(
    //     "Writing CMD_HI: cmd_pkt_tag_id=%d, cmd_force_dim_routing=%d \n",
    //     cmd_value_hi.f.cmd_pkt_tag_id,
    //     cmd_value_hi.f.cmd_force_dim_routing);

    // if (cmd_value_lo.f.cmd_brcst_bit) {
    //     if (cmd_value_lo.f.cmd_static_vc < NOC_BCAST_VC_START ||
    //         cmd_value_lo.f.cmd_static_vc > NOC_BCAST_VC_START + 3) {
    //         LOG_C("ERROR: Invalid static VC for multicast\n");
    //         exit(1);
    //     }
    // } else {
    //     if (cmd_value_lo.f.cmd_static_vc < 0 || cmd_value_lo.f.cmd_static_vc > 7) {
    //         LOG_C("ERROR: Invalid static VC for unicast\n");
    //         exit(1);
    //     }
    // }

    NOC_WRITE_REG(NOC_NIU_CMD_LO_REG_ADDR, cmd_value_lo.val);

    NOC_WRITE_REG(NOC_NIU_CMD_HI_REG_ADDR, cmd_value_hi.val);

    NOC_WRITE_REG(NOC_NIU_CMD_CTRL_REG_ADDR, 0x1);
}

////
///
static bool unicast_addr_local(uint32_t noc_coordinate) {
    uint32_t local_node_id = noc_local_node_id();
    uint32_t local_x = local_node_id & NOC_NODE_ID_MASK;
    uint32_t local_y = (local_node_id >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;

    if (NOC_UNICAST_COORDINATE_X(noc_coordinate) == local_x && NOC_UNICAST_COORDINATE_Y(noc_coordinate) == local_y) {
        return true;
    }

    return false;
}

// Normally when you write to the l1 memory you get an ack from it
// however when we do autocfg we do not get an ack from the overlay registers, this bit disables the logic that expects
// an ack back
void set_rd_ack_drop(uint32_t drop, uint32_t cmd_buf) {
    noc_set_cmd_buf(cmd_buf);
    NOC_NIU_CMD_LO_reg_u cmd_value_lo;
    cmd_value_lo.val = NOC_READ_REG(NOC_NIU_CMD_LO_REG_ADDR);
    cmd_value_lo.f.cmd_mem_rd_drop_ack = drop;
    NOC_WRITE_REG(NOC_NIU_CMD_LO_REG_ADDR, cmd_value_lo.val);
}

// When set, this bit guarantees that transactions have completed before issuing the next one
void set_flush_bit(uint32_t flush, uint32_t cmd_buf) {
    noc_set_cmd_buf(cmd_buf);
    NOC_NIU_CMD_LO_reg_u cmd_value_lo;
    cmd_value_lo.val = NOC_READ_REG(NOC_NIU_CMD_LO_REG_ADDR);
    cmd_value_lo.f.cmd_flush_bit = flush;
    NOC_WRITE_REG(NOC_NIU_CMD_LO_REG_ADDR, cmd_value_lo.val);
}

// force dimension routing for outside mesh space, within mesh it is guaranteed to be dim-order
void force_dim_routing(uint32_t enable, uint32_t cmd_buf) {
    noc_set_cmd_buf(cmd_buf);
    NOC_NIU_CMD_HI_reg_u cmd_value_hi;
    cmd_value_hi.val = NOC_READ_REG(NOC_NIU_CMD_HI_REG_ADDR);
    cmd_value_hi.f.cmd_force_dim_routing = enable;
    NOC_WRITE_REG(NOC_NIU_CMD_HI_REG_ADDR, cmd_value_hi.val);
}

void noc_copy(
    uint32_t src_coordinate,
    uint64_t src_addr,
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint32_t size,
    bool linked,
    bool posted,
    uint32_t static_vc,
    uint32_t resp_static_vc,
    uint32_t transaction_id) {
    // Create TransferParams struct and initialize to zero
    TransferParams transferParams = {0};

    bool src_local = unicast_addr_local(src_coordinate);

    // Assign specific values
    transferParams.src_coordinate = src_coordinate;
    transferParams.src_addr = src_addr;
    transferParams.dst_coordinate = dst_coordinate;
    transferParams.dst_addr = dst_addr;
    transferParams.size = size;
    transferParams.linked = false;
    transferParams.posted = posted;
    transferParams.static_vc = static_vc;
    transferParams.resp_static_vc = resp_static_vc;
    transferParams.src_local = src_local;
    transferParams.transaction_id = transaction_id;

    // Call noc_transfer with the created TransferParams struct
    noc_transfer(&transferParams);
}

void noc_copy_cmd(
    uint32_t src_coordinate,
    uint64_t src_addr,
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint32_t size,
    bool linked,
    bool posted,
    uint32_t static_vc,
    uint32_t resp_static_vc,
    uint32_t transaction_id,
    uint32_t cmd_select) {
    // Create TransferParams struct and initialize to zero
    TransferParams transferParams = {0};

    bool src_local = unicast_addr_local(src_coordinate);

    // Assign specific values
    transferParams.src_coordinate = src_coordinate;
    transferParams.src_addr = src_addr;
    transferParams.dst_coordinate = dst_coordinate;
    transferParams.dst_addr = dst_addr;
    transferParams.size = size;
    transferParams.linked = false;
    transferParams.posted = posted;
    transferParams.static_vc = static_vc;
    transferParams.resp_static_vc = resp_static_vc;
    transferParams.src_local = src_local;
    transferParams.transaction_id = transaction_id;
    transferParams.cmd_select = cmd_select;  // Set the 'cmd_select' parameter

    // Call noc_transfer with the created TransferParams struct
    noc_transfer(&transferParams);
}

void noc_accumulate(
    uint32_t src_coordinate,
    uint64_t src_addr,
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint32_t size,
    bool linked,
    bool posted,
    uint32_t static_vc,
    uint32_t resp_static_vc,
    bool multicast,
    bool multicast_mode,
    uint32_t transaction_id,
    uint32_t data_format,
    bool disable_saturation) {
    // Create TransferParams struct and initialize to zero
    TransferParams transferParams = {0};

    bool src_local = unicast_addr_local(src_coordinate);

    // Assign specific values
    transferParams.src_coordinate = src_coordinate;
    transferParams.src_addr = src_addr;
    transferParams.dst_coordinate = dst_coordinate;
    transferParams.dst_addr = dst_addr;
    transferParams.size = size;
    transferParams.linked = linked;
    transferParams.posted = posted;
    transferParams.static_vc = static_vc;
    transferParams.resp_static_vc = resp_static_vc;
    transferParams.multicast = multicast;
    transferParams.multicast_mode = multicast_mode;
    transferParams.src_local = src_local;
    transferParams.transaction_id = transaction_id;
    transferParams.l1_acc = true;
    transferParams.data_format = data_format;
    transferParams.disable_saturation = disable_saturation;

    // Call noc_transfer with the created TransferParams struct
    noc_transfer(&transferParams);
}

void noc_write_inline(
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint64_t val,
    uint32_t size,
    bool linked,
    bool posted,
    uint32_t static_vc,
    uint32_t resp_static_vc,
    uint32_t transaction_id) {
    // Create TransferParams struct and initialize to zero
    TransferParams transferParams = {0};

    // Assign specific values
    transferParams.dst_coordinate = dst_coordinate;
    transferParams.dst_addr = dst_addr;
    transferParams.val = val;
    transferParams.linked = false;
    transferParams.posted = posted;
    transferParams.static_vc = static_vc;
    transferParams.resp_static_vc = resp_static_vc;
    transferParams.write_inline = true;
    transferParams.size = size;
    transferParams.transaction_id = transaction_id;

    // Call noc_transfer with the created TransferParams struct
    noc_transfer(&transferParams);
}

void noc_multicast_write_inline(
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint64_t val,
    uint32_t size,
    bool multicast_mode,
    bool linked,
    bool posted,
    uint32_t static_vc,
    uint32_t resp_static_vc,
    uint32_t transaction_id) {
    // Create TransferParams struct and initialize to zero
    TransferParams transferParams = {0};

    // Assign specific values
    transferParams.dst_coordinate = dst_coordinate;
    transferParams.dst_addr = dst_addr;
    transferParams.val = val;
    transferParams.linked = linked;
    transferParams.posted = posted;
    transferParams.static_vc = static_vc;
    transferParams.resp_static_vc = resp_static_vc;
    transferParams.multicast = true;
    transferParams.multicast_mode = multicast_mode;
    transferParams.write_inline = true;
    transferParams.size = size;
    transferParams.transaction_id = transaction_id;

    // Call noc_transfer with the created TransferParams struct
    noc_transfer(&transferParams);
}

void noc_multicast_write_inline_src_include(
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint64_t val,
    uint32_t size,
    bool multicast_mode,
    bool linked,
    bool posted,
    uint32_t static_vc,
    uint32_t resp_static_vc,
    uint32_t transaction_id) {
    // Create TransferParams struct and initialize to zero
    TransferParams transferParams = {0};

    // Assign specific values
    transferParams.dst_coordinate = dst_coordinate;
    transferParams.dst_addr = dst_addr;
    transferParams.val = val;
    transferParams.linked = linked;
    transferParams.posted = posted;
    transferParams.static_vc = static_vc;
    transferParams.resp_static_vc = resp_static_vc;
    transferParams.multicast = true;
    transferParams.multicast_mode = multicast_mode;
    transferParams.write_inline = true;
    transferParams.size = size;
    transferParams.src_include = true;
    transferParams.transaction_id = transaction_id;

    // Call noc_transfer with the created TransferParams struct
    noc_transfer(&transferParams);
}

void noc_copy_word_be(
    uint32_t src_coordinate,
    uint64_t src_addr,
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint32_t* be,
    bool linked,
    bool posted,
    uint32_t static_vc,
    uint32_t resp_static_vc,
    uint32_t transaction_id) {
    // Create TransferParams struct and initialize to zero
    TransferParams transferParams = {0};

    // Assign specific values
    transferParams.src_coordinate = src_coordinate;
    transferParams.src_addr = src_addr;
    transferParams.dst_coordinate = dst_coordinate;
    transferParams.dst_addr = dst_addr;
    for (int i = 0; i < 8; i++) {
        transferParams.be[i] = *(be + i);
    }
    transferParams.write_be = true;
    transferParams.linked = false;
    transferParams.posted = posted;
    transferParams.static_vc = static_vc;
    transferParams.resp_static_vc = resp_static_vc;
    transferParams.transaction_id = transaction_id;

    // Call noc_transfer with the created TransferParams struct
    noc_transfer(&transferParams);
}

void noc_multicast_copy_word_be(
    uint32_t src_coordinate,
    uint64_t src_addr,
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    bool multicast_mode,
    uint32_t* be,
    bool linked,
    bool posted,
    uint32_t static_vc,
    uint32_t resp_static_vc,
    uint32_t transaction_id) {
    // Create TransferParams struct and initialize to zero
    TransferParams transferParams = {0};

    // Assign specific values
    transferParams.src_coordinate = src_coordinate;
    transferParams.src_addr = src_addr;
    transferParams.dst_coordinate = dst_coordinate;
    transferParams.dst_addr = dst_addr;
    for (int i = 0; i < 8; i++) {
        transferParams.be[i] = *(be + i);
    }
    transferParams.write_be = true;
    transferParams.linked = linked;
    transferParams.posted = posted;
    transferParams.static_vc = static_vc;
    transferParams.resp_static_vc = resp_static_vc;
    transferParams.multicast = true;
    transferParams.multicast_mode = multicast_mode;
    transferParams.transaction_id = transaction_id;

    // Call noc_transfer with the created TransferParams struct
    noc_transfer(&transferParams);
}

void noc_multicast_copy(
    uint32_t src_coordinate,
    uint64_t src_addr,
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    bool multicast_mode,
    uint32_t size,
    bool linked,
    bool posted,
    uint32_t static_vc,
    uint32_t resp_static_vc,
    uint32_t transaction_id) {
    // Create TransferParams struct and initialize to zero
    TransferParams transferParams = {0};

    bool src_local = unicast_addr_local(src_coordinate);

    // Assign specific values
    transferParams.src_coordinate = src_coordinate;
    transferParams.src_addr = src_addr;
    transferParams.dst_coordinate = dst_coordinate;
    transferParams.dst_addr = dst_addr;
    transferParams.size = size;
    transferParams.linked = linked;
    transferParams.posted = posted;
    transferParams.src_local = src_local;
    transferParams.static_vc = static_vc;
    transferParams.resp_static_vc = resp_static_vc;
    transferParams.multicast = true;
    transferParams.multicast_mode = multicast_mode;
    transferParams.transaction_id = transaction_id;

    // Call noc_transfer with the created TransferParams struct
    noc_transfer(&transferParams);
}

void noc_multicast_copy_src_include(
    uint32_t src_coordinate,
    uint64_t src_addr,
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    bool multicast_mode,
    uint32_t size,
    bool linked,
    bool posted,
    uint32_t static_vc,
    uint32_t resp_static_vc,
    uint32_t transaction_id) {
    // Create TransferParams struct and initialize to zero
    TransferParams transferParams = {0};

    bool src_local = unicast_addr_local(src_coordinate);

    // Assign specific values
    transferParams.src_coordinate = src_coordinate;
    transferParams.src_addr = src_addr;
    transferParams.dst_coordinate = dst_coordinate;
    transferParams.dst_addr = dst_addr;
    transferParams.size = size;
    transferParams.linked = linked;
    transferParams.posted = posted;
    transferParams.src_local = src_local;
    transferParams.static_vc = static_vc;
    transferParams.resp_static_vc = resp_static_vc;
    transferParams.multicast = true;
    transferParams.multicast_mode = multicast_mode;
    transferParams.src_include = true;
    transferParams.transaction_id = transaction_id;

    // Call noc_transfer with the created TransferParams struct
    noc_transfer(&transferParams);
}

void noc_multicast_copy_exclude(
    uint32_t src_coordinate,
    uint64_t src_addr,
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    bool multicast_mode,
    uint32_t multicast_lo,
    uint32_t multicast_hi,
    uint32_t size,
    bool linked,
    bool posted,
    uint32_t static_vc,
    uint32_t resp_static_vc,
    uint32_t transaction_id) {
    // Create TransferParams struct and initialize to zero
    TransferParams transferParams = {0};

    bool src_local = unicast_addr_local(src_coordinate);

    // Assign specific values
    transferParams.src_coordinate = src_coordinate;
    transferParams.src_addr = src_addr;
    transferParams.dst_coordinate = dst_coordinate;
    transferParams.dst_addr = dst_addr;
    transferParams.size = size;
    transferParams.linked = linked;
    transferParams.posted = posted;
    transferParams.src_local = src_local;
    transferParams.static_vc = static_vc;
    transferParams.resp_static_vc = resp_static_vc;
    transferParams.multicast = true;
    transferParams.multicast_mode = multicast_mode;
    transferParams.multicast_lo = multicast_lo;
    transferParams.multicast_hi = multicast_hi;
    transferParams.transaction_id = transaction_id;

    // Call noc_transfer with the created TransferParams struct
    noc_transfer(&transferParams);
}

void noc_atomic_increment(
    uint32_t noc_coordinate,
    uint64_t addr,
    uint32_t incr,
    uint32_t wrap,
    bool linked,
    uint32_t static_vc,
    uint32_t transaction_id) {
    // Create TransferParams struct and initialize to zero
    TransferParams transferParams = {0};

    // Assign specific values
    transferParams.src_coordinate = noc_coordinate;
    transferParams.src_addr = addr;
    transferParams.atomic = true;
    transferParams.atomic_data = incr;
    transferParams.wrap = wrap;
    transferParams.linked = false;
    transferParams.posted = true;
    transferParams.static_vc = static_vc;
    transferParams.transaction_id = transaction_id;

    // Call noc_transfer with the created TransferParams struct
    noc_transfer(&transferParams);
}

void noc_atomic_read_and_increment(
    uint32_t noc_coordinate,
    uint64_t addr,
    uint32_t incr,
    uint32_t wrap,
    uint32_t read_coordinate,
    uint64_t read_addr,
    bool linked,
    uint32_t static_vc,
    uint32_t resp_static_vc,
    uint32_t transaction_id) {
    // Create TransferParams struct and initialize to zero
    TransferParams transferParams = {0};

    // Assign specific values
    transferParams.src_coordinate = noc_coordinate;
    transferParams.src_addr = addr;
    transferParams.dst_coordinate = read_coordinate;
    transferParams.dst_addr = read_addr;
    transferParams.atomic = true;
    transferParams.atomic_data = incr;
    transferParams.wrap = wrap;
    transferParams.linked = false;
    transferParams.transaction_id = transaction_id;
    transferParams.posted = false;
    transferParams.static_vc = static_vc;
    transferParams.resp_static_vc = resp_static_vc;

    // Call noc_transfer with the created TransferParams struct
    noc_transfer(&transferParams);
}

void noc_multicast_atomic_increment(
    uint32_t noc_coordinate,
    uint64_t addr,
    bool multicast_mode,
    uint32_t incr,
    uint32_t wrap,
    bool linked,
    uint32_t static_vc,
    uint32_t transaction_id) {
    // Create TransferParams struct and initialize to zero
    TransferParams transferParams = {0};

    // Assign specific values
    transferParams.src_coordinate = noc_coordinate;
    transferParams.src_addr = addr;
    transferParams.atomic = true;
    transferParams.atomic_data = incr;
    transferParams.wrap = wrap;
    transferParams.linked = linked;
    transferParams.multicast = true;
    transferParams.multicast_mode = multicast_mode;
    transferParams.posted = true;
    transferParams.static_vc = static_vc;
    transferParams.transaction_id = transaction_id;

    // Call noc_transfer with the created TransferParams struct
    noc_transfer(&transferParams);
}

void noc_multicast_atomic_read_and_increment(
    uint32_t noc_coordinate,
    uint64_t addr,
    bool multicast_mode,
    uint32_t incr,
    uint32_t wrap,
    uint32_t read_coordinate,
    uint64_t read_addr,
    bool linked,
    uint32_t static_vc,
    uint32_t resp_static_vc,
    uint32_t transaction_id) {
    // Create TransferParams struct and initialize to zero
    TransferParams transferParams = {0};

    // Assign specific values
    transferParams.src_coordinate = noc_coordinate;
    transferParams.src_addr = addr;
    transferParams.dst_coordinate = read_coordinate;
    transferParams.dst_addr = read_addr;
    transferParams.atomic = true;
    transferParams.atomic_data = incr;
    transferParams.wrap = wrap;
    transferParams.linked = linked;
    transferParams.multicast = true;
    transferParams.multicast_mode = multicast_mode;
    transferParams.transaction_id = transaction_id;
    transferParams.posted = false;
    transferParams.static_vc = static_vc;
    transferParams.resp_static_vc = resp_static_vc;

    // Call noc_transfer with the created TransferParams struct
    noc_transfer(&transferParams);
}

bool noc_command_ready(uint32_t cmd_select) {
    noc_set_cmd_buf(cmd_select);

    return (NOC_READ_REG(NOC_CMD_CTRL) == NOC_CTRL_STATUS_READY);
}

uint32_t noc_atomic_read_updates_completed() {
    uint32_t save_cmd_buf = noc_get_cmd_buf();
    noc_set_cmd_buf(0);
    uint32_t result = noc_status_reg(NIU_MST_ATOMIC_RESP_RECEIVED);
    noc_set_cmd_buf(save_cmd_buf);
    return result;
}

volatile uint32_t noc_wr_ack_received() {
    uint32_t save_cmd_buf = noc_get_cmd_buf();
    noc_set_cmd_buf(0);
    uint32_t result = noc_status_reg(NIU_MST_WR_ACK_RECEIVED);
    noc_set_cmd_buf(save_cmd_buf);
    return result;
}

volatile uint32_t noc_rd_resp_received() {
    uint32_t save_cmd_buf = noc_get_cmd_buf();
    noc_set_cmd_buf(0);
    uint32_t result = noc_status_reg(NIU_MST_RD_RESP_RECEIVED);
    noc_set_cmd_buf(save_cmd_buf);
    return result;
}

uint32_t noc_local_node_id() {
    uint32_t save_cmd_buf = noc_get_cmd_buf();
    noc_set_cmd_buf(0);
    uint32_t result = NOC_READ_REG(NOC_NODE_ID);
    noc_set_cmd_buf(save_cmd_buf);
    return result;
}

uint32_t noc_status_reg(uint32_t status_reg_id) {
    uint32_t save_cmd_buf = noc_get_cmd_buf();
    noc_set_cmd_buf(0);
    uint32_t result = NOC_READ_REG(NOC_STATUS(status_reg_id));
    noc_set_cmd_buf(save_cmd_buf);
    return result;
}

void noc_set_cfg_reg(uint32_t cfg_reg_id, uint32_t val) {
    uint32_t save_cmd_buf = noc_get_cmd_buf();
    noc_set_cmd_buf(0);
    NOC_WRITE_REG(NOC_CFG(cfg_reg_id), val);
    noc_set_cmd_buf(save_cmd_buf);
}

uint32_t noc_get_cfg_reg(uint32_t cfg_reg_id) {
    uint32_t save_cmd_buf = noc_get_cmd_buf();
    noc_set_cmd_buf(0);
    uint32_t result = NOC_READ_REG(NOC_CFG(cfg_reg_id));
    noc_set_cmd_buf(save_cmd_buf);
    return result;
}

//////////////////////////////////////////////////////////////////
//////////////////////// ECC Functions ///////////////////////////
//////////////////////////////////////////////////////////////////

void noc_ecc_cfg_stage_1(bool header_ckh_bits_en) {
    uint32_t mask;
    uint32_t cfg_reg;

    cfg_reg = noc_get_cfg_reg(ROUTER_CFG_0);
    mask = (1 << ROUTER_CFG_0_ECC_HEADER_CHKBITS_EN);
    cfg_reg = (cfg_reg & ~mask) | (header_ckh_bits_en << ROUTER_CFG_0_ECC_HEADER_CHKBITS_EN);
    noc_set_cfg_reg(ROUTER_CFG_0, cfg_reg);
}

void noc_ecc_cfg_stage_2(
    bool niu_mem_parity_en,
    bool router_mem_parity_en,
    bool header_secded_en,
    bool mem_parity_int_en,
    bool header_sec_int_en,
    bool header_ded_int_en) {
    uint32_t mask;
    uint32_t cfg_reg;

    cfg_reg = noc_get_cfg_reg(NIU_CFG_0);
    mask = (1 << NIU_CFG_0_ECC_NIU_MEM_PARITY_EN) | (1 << NIU_CFG_0_ECC_MEM_PARITY_INT_EN) |
           (1 << NIU_CFG_0_ECC_HEADER_1B_INT_EN) | (1 << NIU_CFG_0_ECC_HEADER_2B_INT_EN);
    cfg_reg = (cfg_reg & ~mask) | (niu_mem_parity_en << NIU_CFG_0_ECC_NIU_MEM_PARITY_EN) |
              (mem_parity_int_en << NIU_CFG_0_ECC_MEM_PARITY_INT_EN) |
              (header_sec_int_en << NIU_CFG_0_ECC_HEADER_1B_INT_EN) |
              (header_ded_int_en << NIU_CFG_0_ECC_HEADER_2B_INT_EN);
    noc_set_cfg_reg(NIU_CFG_0, cfg_reg);

    cfg_reg = noc_get_cfg_reg(ROUTER_CFG_0);
    mask = (1 << ROUTER_CFG_0_ECC_ROUTER_MEM_PARITY_EN) | (1 << ROUTER_CFG_0_ECC_HEADER_SECDED_EN);
    cfg_reg = (cfg_reg & ~mask) | (router_mem_parity_en << ROUTER_CFG_0_ECC_ROUTER_MEM_PARITY_EN) |
              (header_secded_en << ROUTER_CFG_0_ECC_HEADER_SECDED_EN);
    noc_set_cfg_reg(ROUTER_CFG_0, cfg_reg);
}

void noc_ecc_clear_err(bool clear_mem_parity_err, bool clear_header_sec, bool clear_header_ded) {
    uint32_t save_cmd_buf = noc_get_cmd_buf();
    noc_set_cmd_buf(0);
    NOC_WRITE_REG(ECC_CTRL, ((clear_mem_parity_err | (clear_header_sec << 1) | (clear_header_ded << 2)) & 0x7));
    noc_set_cmd_buf(save_cmd_buf);
}

void noc_ecc_force_err(bool force_mem_parity_err, bool force_header_sec, bool force_header_ded) {
    uint32_t save_cmd_buf = noc_get_cmd_buf();
    noc_set_cmd_buf(0);
    NOC_WRITE_REG(ECC_CTRL, (((force_mem_parity_err | (force_header_sec << 1) | (force_header_ded << 2)) & 0x7) << 3));
    noc_set_cmd_buf(save_cmd_buf);
}

uint32_t noc_ecc_get_num_mem_parity_errs() {
    uint32_t save_cmd_buf = noc_get_cmd_buf();
    noc_set_cmd_buf(0);
    uint32_t result = NOC_READ_REG(NUM_MEM_PARITY_ERR);
    noc_set_cmd_buf(save_cmd_buf);
    return result;
}

uint32_t noc_ecc_get_num_header_sec() {
    uint32_t save_cmd_buf = noc_get_cmd_buf();
    noc_set_cmd_buf(0);
    uint32_t result = NOC_READ_REG(NUM_HEADER_1B_ERR);
    noc_set_cmd_buf(save_cmd_buf);
    return result;
}

uint32_t noc_ecc_get_num_header_ded() {
    uint32_t save_cmd_buf = noc_get_cmd_buf();
    noc_set_cmd_buf(0);
    uint32_t result = NOC_READ_REG(NUM_HEADER_2B_ERR);
    noc_set_cmd_buf(save_cmd_buf);
    return result;
}

void noc_clear_req_id_cnt(uint32_t id_mask) {
    uint32_t save_cmd_buf = noc_get_cmd_buf();
    noc_set_cmd_buf(0);
    NOC_WRITE_REG(NOC_CLEAR_OUTSTANDING_REQ_CNT, id_mask);
    noc_set_cmd_buf(save_cmd_buf);
}
