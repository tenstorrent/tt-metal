// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "noc.h"
#include <stdint.h>
#include <stdbool.h>
#include "noc_parameters.h"

////

#ifdef TB_NOC

#include "noc_api_dpi.h"

#else

#define NOC_WRITE_REG(addr, val)                                      \
    ((*((volatile uint32_t*)(noc_get_cmd_buf() * NOC_CMD_BUF_OFFSET + \
                             noc_get_active_instance() * NOC_INSTANCE_OFFSET + (addr)))) = (val))
#define NOC_READ_REG(addr)                                                                                             \
    (*((volatile uint32_t*)(noc_get_cmd_buf() * NOC_CMD_BUF_OFFSET + noc_get_active_instance() * NOC_INSTANCE_OFFSET + \
                            (addr))))

#endif

#ifdef ARC_FW_NOC
#include "arc_fw_noc.h"
#endif

///

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

static void noc_transfer(
    uint32_t src_coordinate,
    uint64_t src_addr,
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint32_t size,
    bool linked,
    bool posted,
    bool static_vc_alloc,
    uint32_t static_vc,
    bool multicast,
    uint32_t multicast_mode,
    uint32_t multicast_exclude,
    bool src_local,
    uint32_t vc_arb_priority,
    bool src_include,
    uint8_t transaction_id) {
    while (!noc_command_ready());
    NOC_WRITE_REG(NOC_TARG_ADDR_LO, (uint32_t)(src_addr & 0xFFFFFFFF));
    NOC_WRITE_REG(NOC_TARG_ADDR_MID, (uint32_t)(src_addr >> 32));
    NOC_WRITE_REG(NOC_TARG_ADDR_HI, src_coordinate);
    NOC_WRITE_REG(NOC_RET_ADDR_LO, (uint32_t)(dst_addr & 0xFFFFFFFF));
    NOC_WRITE_REG(NOC_RET_ADDR_MID, (uint32_t)(dst_addr >> 32));
    NOC_WRITE_REG(NOC_RET_ADDR_HI, dst_coordinate);
    NOC_WRITE_REG(NOC_AT_LEN_BE, size);
    NOC_WRITE_REG(NOC_AT_LEN_BE_1, 0x0);
    NOC_WRITE_REG(NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(transaction_id));
    if (multicast) {
        NOC_WRITE_REG(
            NOC_CTRL,
            (linked ? NOC_CMD_VC_LINKED : 0x0) | (static_vc_alloc ? NOC_CMD_VC_STATIC : 0x0) |
                NOC_CMD_STATIC_VC(static_vc) | NOC_CMD_PATH_RESERVE | NOC_CMD_CPY |
                (src_local ? NOC_CMD_WR : NOC_CMD_RD) | NOC_CMD_BRCST_PACKET | (posted ? 0x0 : NOC_CMD_RESP_MARKED) |
                (src_include ? NOC_CMD_BRCST_SRC_INCLUDE : 0x0) | NOC_CMD_BRCST_XY(multicast_mode));
        NOC_WRITE_REG(NOC_BRCST_EXCLUDE, multicast_exclude);
    } else {
        NOC_WRITE_REG(
            NOC_CTRL,
            (linked ? NOC_CMD_VC_LINKED : 0x0) | (static_vc_alloc ? NOC_CMD_VC_STATIC : 0x0) |
                NOC_CMD_STATIC_VC(static_vc) | NOC_CMD_CPY | (src_local ? NOC_CMD_WR : NOC_CMD_RD) |
                (posted ? 0x0 : NOC_CMD_RESP_MARKED) | NOC_CMD_ARB_PRIORITY(vc_arb_priority));
    }
    NOC_WRITE_REG(NOC_CMD_CTRL, 0x1);
}

static bool unicast_addr_local(uint32_t noc_coordinate) {
    bool noc_id_translate_en = (noc_get_cfg_reg(NIU_CFG_0) >> NIU_CFG_0_NOC_ID_TRANSLATE_EN) & 0x1;
    uint32_t local_node_id = noc_id_translate_en ? noc_get_cfg_reg(NOC_ID_LOGICAL) : noc_local_node_id();
    uint32_t local_x = (local_node_id & NOC_NODE_ID_MASK);
    uint32_t local_y = ((local_node_id >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK);
    return (NOC_UNICAST_COORDINATE_X(noc_coordinate) == local_x) &&
           (NOC_UNICAST_COORDINATE_Y(noc_coordinate) == local_y);
}

////

void noc_copy(
    uint32_t src_coordinate,
    uint64_t src_addr,
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint32_t size,
    bool linked,
    bool posted,
    bool static_vc_alloc,
    uint32_t static_vc,
    uint32_t vc_arb_priority,
    uint8_t transaction_id) {
    bool src_local = unicast_addr_local(src_coordinate);
    if (!src_local) {
        posted = true;
    }
    noc_transfer(
        src_coordinate,
        src_addr,
        dst_coordinate,
        dst_addr,
        size,
        linked,
        posted,
        static_vc_alloc,
        static_vc,
        false,
        0,
        0,
        src_local,
        vc_arb_priority,
        false,
        transaction_id);
}

void noc_accumulate(
    uint32_t src_coordinate,
    uint64_t src_addr,
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint32_t size,
    bool linked,
    bool posted,
    bool static_vc_alloc,
    uint32_t static_vc,
    bool multicast,
    uint32_t multicast_mode,
    uint32_t vc_arb_priority,
    uint8_t transaction_id,
    uint8_t data_format,
    bool disable_saturation) {
    bool src_local = unicast_addr_local(src_coordinate);
    if (!src_local) {
        posted = true;
    }
    while (!noc_command_ready());
    NOC_WRITE_REG(NOC_TARG_ADDR_LO, (uint32_t)(src_addr & 0xFFFFFFFF));
    NOC_WRITE_REG(NOC_TARG_ADDR_MID, (uint32_t)(src_addr >> 32));
    NOC_WRITE_REG(NOC_TARG_ADDR_HI, src_coordinate);
    NOC_WRITE_REG(NOC_RET_ADDR_LO, (uint32_t)(dst_addr & 0xFFFFFFFF));
    NOC_WRITE_REG(NOC_RET_ADDR_MID, (uint32_t)(dst_addr >> 32));
    NOC_WRITE_REG(NOC_RET_ADDR_HI, dst_coordinate);
    NOC_WRITE_REG(NOC_AT_LEN_BE, size);
    NOC_WRITE_REG(NOC_AT_LEN_BE_1, 0x0);
    NOC_WRITE_REG(
        NOC_L1_ACC_AT_INSTRN,
        NOC_AT_INS(NOC_AT_INS_ACC) | NOC_AT_ACC_FORMAT(data_format) | NOC_AT_ACC_SAT_DIS(disable_saturation));
    NOC_WRITE_REG(NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(transaction_id));
    NOC_WRITE_REG(NOC_BRCST_EXCLUDE, 0x0);
    if (multicast) {
        NOC_WRITE_REG(
            NOC_CTRL,
            (linked ? NOC_CMD_VC_LINKED : 0x0) | (static_vc_alloc ? NOC_CMD_VC_STATIC : 0x0) |
                NOC_CMD_STATIC_VC(static_vc) | NOC_CMD_PATH_RESERVE | NOC_CMD_CPY | NOC_CMD_L1_ACC_AT_EN |
                (src_local ? NOC_CMD_WR : NOC_CMD_RD) | NOC_CMD_BRCST_PACKET | (posted ? 0x0 : NOC_CMD_RESP_MARKED) |
                NOC_CMD_BRCST_XY(multicast_mode));
        NOC_WRITE_REG(NOC_BRCST_EXCLUDE, 0x0);
    } else {
        NOC_WRITE_REG(
            NOC_CTRL,
            (linked ? NOC_CMD_VC_LINKED : 0x0) | (static_vc_alloc ? NOC_CMD_VC_STATIC : 0x0) |
                NOC_CMD_STATIC_VC(static_vc) | NOC_CMD_CPY | NOC_CMD_L1_ACC_AT_EN |
                (src_local ? NOC_CMD_WR : NOC_CMD_RD) | (posted ? 0x0 : NOC_CMD_RESP_MARKED) |
                NOC_CMD_ARB_PRIORITY(vc_arb_priority));
    }
    NOC_WRITE_REG(NOC_CMD_CTRL, 0x1);
}

static void transfer_word_be(
    uint32_t src_coordinate,
    uint64_t src_addr,
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint64_t be,
    bool linked,
    bool posted,
    bool static_vc_alloc,
    uint32_t static_vc,
    bool multicast,
    uint32_t multicast_mode,
    uint8_t transaction_id) {
    while (!noc_command_ready());
    NOC_WRITE_REG(NOC_TARG_ADDR_LO, (uint32_t)(src_addr & 0xFFFFFFFF));
    NOC_WRITE_REG(NOC_TARG_ADDR_MID, (uint32_t)(src_addr >> 32));
    NOC_WRITE_REG(NOC_TARG_ADDR_HI, src_coordinate);
    NOC_WRITE_REG(NOC_RET_ADDR_LO, (uint32_t)(dst_addr & 0xFFFFFFFF));
    NOC_WRITE_REG(NOC_RET_ADDR_MID, (uint32_t)(dst_addr >> 32));
    NOC_WRITE_REG(NOC_RET_ADDR_HI, dst_coordinate);
    NOC_WRITE_REG(NOC_AT_LEN_BE, (uint32_t)(be & 0xFFFFFFFF));
    NOC_WRITE_REG(NOC_AT_LEN_BE_1, (uint32_t)(be >> 32));
    NOC_WRITE_REG(NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(transaction_id));
    NOC_WRITE_REG(NOC_BRCST_EXCLUDE, 0x0);
    if (multicast) {
        NOC_WRITE_REG(
            NOC_CTRL,
            (linked ? NOC_CMD_VC_LINKED : 0x0) | (static_vc_alloc ? NOC_CMD_VC_STATIC : 0x0) |
                NOC_CMD_STATIC_VC(static_vc) | NOC_CMD_PATH_RESERVE | NOC_CMD_WR | NOC_CMD_WR_BE |
                NOC_CMD_BRCST_PACKET | (posted ? 0x0 : NOC_CMD_RESP_MARKED) | NOC_CMD_BRCST_XY(multicast_mode));
        NOC_WRITE_REG(NOC_BRCST_EXCLUDE, 0x0);
    } else {
        NOC_WRITE_REG(
            NOC_CTRL,
            (linked ? NOC_CMD_VC_LINKED : 0x0) | (static_vc_alloc ? NOC_CMD_VC_STATIC : 0x0) |
                NOC_CMD_STATIC_VC(static_vc) | NOC_CMD_WR | NOC_CMD_WR_BE | (posted ? 0x0 : NOC_CMD_RESP_MARKED));
    }
    NOC_WRITE_REG(NOC_CMD_CTRL, 0x1);
}

static void noc_transfer_dw_inline(
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint32_t val,
    uint8_t be,
    bool linked,
    bool posted,
    bool static_vc_alloc,
    uint32_t static_vc,
    bool multicast,
    uint32_t multicast_mode,
    uint8_t transaction_id) {
    while (!noc_command_ready());
    NOC_WRITE_REG(NOC_TARG_ADDR_LO, (uint32_t)(dst_addr & 0xFFFFFFFF));
    NOC_WRITE_REG(NOC_TARG_ADDR_MID, (uint32_t)(dst_addr >> 32));
    NOC_WRITE_REG(NOC_TARG_ADDR_HI, dst_coordinate);

    uint64_t be64 = be;
    uint32_t be_shift = (dst_addr & (NOC_WORD_BYTES - 1));
    be64 = (be64 << be_shift);
    NOC_WRITE_REG(NOC_AT_LEN_BE, (uint32_t)(be64 & 0xFFFFFFFF));
    NOC_WRITE_REG(NOC_AT_LEN_BE_1, (uint32_t)(be64 >> 32));

    NOC_WRITE_REG(NOC_AT_DATA, val);
    NOC_WRITE_REG(NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(transaction_id));
    NOC_WRITE_REG(NOC_BRCST_EXCLUDE, 0x0);
    if (multicast) {
        NOC_WRITE_REG(
            NOC_CTRL,
            (linked ? NOC_CMD_VC_LINKED : 0x0) | (static_vc_alloc ? NOC_CMD_VC_STATIC : 0x0) |
                NOC_CMD_STATIC_VC(static_vc) | NOC_CMD_PATH_RESERVE | NOC_CMD_WR | NOC_CMD_WR_INLINE |
                NOC_CMD_BRCST_PACKET | (posted ? 0x0 : NOC_CMD_RESP_MARKED) | NOC_CMD_BRCST_XY(multicast_mode));
        NOC_WRITE_REG(NOC_BRCST_EXCLUDE, 0x0);
    } else {
        NOC_WRITE_REG(
            NOC_CTRL,
            (linked ? NOC_CMD_VC_LINKED : 0x0) | (static_vc_alloc ? NOC_CMD_VC_STATIC : 0x0) |
                NOC_CMD_STATIC_VC(static_vc) | NOC_CMD_WR | NOC_CMD_WR_INLINE | (posted ? 0x0 : NOC_CMD_RESP_MARKED));
    }
    NOC_WRITE_REG(NOC_CMD_CTRL, 0x1);
}

void noc_write_dw_inline(
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint32_t val,
    uint8_t be,
    bool linked,
    bool posted,
    bool static_vc_alloc,
    uint32_t static_vc,
    uint8_t transaction_id) {
    noc_transfer_dw_inline(
        dst_coordinate, dst_addr, val, be, linked, posted, static_vc_alloc, static_vc, false, 0, transaction_id);
}

void noc_multicast_write_dw_inline(
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint32_t val,
    uint32_t multicast_mode,
    uint8_t be,
    bool linked,
    bool posted,
    bool static_vc_alloc,
    uint32_t static_vc,
    uint8_t transaction_id) {
    noc_transfer_dw_inline(
        dst_coordinate,
        dst_addr,
        val,
        be,
        linked,
        posted,
        static_vc_alloc,
        static_vc,
        true,
        multicast_mode,
        transaction_id);
}

void noc_copy_word_be(
    uint32_t src_coordinate,
    uint64_t src_addr,
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint64_t be,
    bool linked,
    bool posted,
    bool static_vc_alloc,
    uint32_t static_vc,
    uint8_t transaction_id) {
    transfer_word_be(
        src_coordinate,
        src_addr,
        dst_coordinate,
        dst_addr,
        be,
        linked,
        posted,
        static_vc_alloc,
        static_vc,
        false,
        0,
        transaction_id);
}

void noc_multicast_copy_word_be(
    uint32_t src_coordinate,
    uint64_t src_addr,
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint32_t multicast_mode,
    uint64_t be,
    bool linked,
    bool posted,
    bool static_vc_alloc,
    uint32_t static_vc,
    uint8_t transaction_id) {
    transfer_word_be(
        src_coordinate,
        src_addr,
        dst_coordinate,
        dst_addr,
        be,
        linked,
        posted,
        static_vc_alloc,
        static_vc,
        true,
        multicast_mode,
        transaction_id);
}

void noc_multicast_copy(
    uint32_t src_coordinate,
    uint64_t src_addr,
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint32_t multicast_mode,
    uint32_t size,
    bool linked,
    bool posted,
    bool static_vc_alloc,
    uint32_t static_vc,
    uint8_t transaction_id) {
    noc_transfer(
        src_coordinate,
        src_addr,
        dst_coordinate,
        dst_addr,
        size,
        linked,
        posted,
        static_vc_alloc,
        static_vc,
        true,
        multicast_mode,
        0,
        unicast_addr_local(src_coordinate),
        0,
        false,
        transaction_id);
}

void noc_multicast_copy_src_include(
    uint32_t src_coordinate,
    uint64_t src_addr,
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint32_t multicast_mode,
    uint32_t size,
    bool linked,
    bool posted,
    bool static_vc_alloc,
    uint32_t static_vc,
    uint8_t transaction_id) {
    noc_transfer(
        src_coordinate,
        src_addr,
        dst_coordinate,
        dst_addr,
        size,
        linked,
        posted,
        static_vc_alloc,
        static_vc,
        true,
        multicast_mode,
        0,
        unicast_addr_local(src_coordinate),
        0,
        true,
        transaction_id);
}

void noc_multicast_copy_exclude(
    uint32_t src_coordinate,
    uint64_t src_addr,
    uint32_t dst_coordinate,
    uint64_t dst_addr,
    uint32_t multicast_mode,
    uint32_t multicast_exclude,
    uint32_t size,
    bool linked,
    bool posted,
    bool static_vc_alloc,
    uint32_t static_vc,
    uint8_t transaction_id) {
    noc_transfer(
        src_coordinate,
        src_addr,
        dst_coordinate,
        dst_addr,
        size,
        linked,
        posted,
        static_vc_alloc,
        static_vc,
        true,
        multicast_mode,
        multicast_exclude,
        unicast_addr_local(src_coordinate),
        0,
        false,
        transaction_id);
}

void noc_atomic_increment(uint32_t noc_coordinate, uint64_t addr, uint32_t incr, uint32_t wrap, bool linked) {
    while (!noc_command_ready());
    NOC_WRITE_REG(NOC_TARG_ADDR_LO, (uint32_t)(addr & 0xFFFFFFFF));
    NOC_WRITE_REG(NOC_TARG_ADDR_MID, (uint32_t)(addr >> 32));
    NOC_WRITE_REG(NOC_TARG_ADDR_HI, noc_coordinate);
    // NOC_WRITE_REG(NOC_RET_ADDR_LO, 0);
    // NOC_WRITE_REG(NOC_RET_ADDR_MID, 0);
    NOC_WRITE_REG(NOC_CTRL, (linked ? NOC_CMD_VC_LINKED : 0x0) | NOC_CMD_AT);
    NOC_WRITE_REG(
        NOC_AT_LEN_BE,
        NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(wrap) | NOC_AT_IND_32((addr >> 2) & 0x3) | NOC_AT_IND_32_SRC(0));
    NOC_WRITE_REG(NOC_AT_LEN_BE_1, 0x0);
    NOC_WRITE_REG(NOC_AT_DATA, incr);
    NOC_WRITE_REG(NOC_CMD_CTRL, 0x1);
}

void noc_atomic_read_and_increment(
    uint32_t noc_coordinate,
    uint64_t addr,
    uint32_t incr,
    uint32_t wrap,
    uint32_t read_coordinate,
    uint64_t read_addr,
    bool linked,
    uint8_t transaction_id) {
    while (!noc_command_ready());
    NOC_WRITE_REG(NOC_TARG_ADDR_LO, (uint32_t)(addr & 0xFFFFFFFF));
    NOC_WRITE_REG(NOC_TARG_ADDR_MID, (uint32_t)(addr >> 32));
    NOC_WRITE_REG(NOC_TARG_ADDR_HI, noc_coordinate);
    NOC_WRITE_REG(NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(transaction_id));
    NOC_WRITE_REG(NOC_RET_ADDR_LO, (uint32_t)(read_addr & 0xFFFFFFFF));
    NOC_WRITE_REG(NOC_RET_ADDR_MID, (uint32_t)(read_addr >> 32));
    NOC_WRITE_REG(NOC_RET_ADDR_HI, read_coordinate);
    NOC_WRITE_REG(NOC_CTRL, (linked ? NOC_CMD_VC_LINKED : 0x0) | NOC_CMD_AT | NOC_CMD_RESP_MARKED);
    NOC_WRITE_REG(
        NOC_AT_LEN_BE,
        NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(wrap) | NOC_AT_IND_32((addr >> 2) & 0x3) | NOC_AT_IND_32_SRC(0));
    NOC_WRITE_REG(NOC_AT_LEN_BE_1, 0x0);
    NOC_WRITE_REG(NOC_AT_DATA, incr);
    NOC_WRITE_REG(NOC_CMD_CTRL, 0x1);
}

void noc_multicast_atomic_increment(
    uint32_t noc_coordinate, uint64_t addr, uint32_t multicast_mode, uint32_t incr, uint32_t wrap, bool linked) {
    while (!noc_command_ready());
    NOC_WRITE_REG(NOC_TARG_ADDR_LO, (uint32_t)(addr & 0xFFFFFFFF));
    NOC_WRITE_REG(NOC_TARG_ADDR_MID, (uint32_t)(addr >> 32));
    NOC_WRITE_REG(NOC_TARG_ADDR_HI, noc_coordinate);
    // NOC_WRITE_REG(NOC_RET_ADDR_LO, 0);
    // NOC_WRITE_REG(NOC_RET_ADDR_MID, 0);
    NOC_WRITE_REG(
        NOC_CTRL,
        (linked ? NOC_CMD_VC_LINKED : 0x0) | NOC_CMD_PATH_RESERVE | NOC_CMD_AT | NOC_CMD_BRCST_PACKET |
            NOC_CMD_BRCST_XY(multicast_mode));
    NOC_WRITE_REG(NOC_BRCST_EXCLUDE, 0x0);
    NOC_WRITE_REG(
        NOC_AT_LEN_BE,
        NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(wrap) | NOC_AT_IND_32((addr >> 2) & 0x3) | NOC_AT_IND_32_SRC(0));
    NOC_WRITE_REG(NOC_AT_LEN_BE_1, 0x0);
    NOC_WRITE_REG(NOC_AT_DATA, incr);
    NOC_WRITE_REG(NOC_CMD_CTRL, 0x1);
}

void noc_multicast_atomic_read_and_increment(
    uint32_t noc_coordinate,
    uint64_t addr,
    uint32_t multicast_mode,
    uint32_t incr,
    uint32_t wrap,
    uint32_t read_coordinate,
    uint64_t read_addr,
    bool linked,
    uint8_t transaction_id) {
    while (!noc_command_ready());
    NOC_WRITE_REG(NOC_TARG_ADDR_LO, (uint32_t)(addr & 0xFFFFFFFF));
    NOC_WRITE_REG(NOC_TARG_ADDR_MID, (uint32_t)(addr >> 32));
    NOC_WRITE_REG(NOC_TARG_ADDR_HI, noc_coordinate);
    NOC_WRITE_REG(NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(transaction_id));
    NOC_WRITE_REG(NOC_RET_ADDR_LO, (uint32_t)(read_addr & 0xFFFFFFFF));
    NOC_WRITE_REG(NOC_RET_ADDR_MID, (uint32_t)(read_addr >> 32));
    NOC_WRITE_REG(NOC_RET_ADDR_HI, read_coordinate);
    NOC_WRITE_REG(
        NOC_CTRL,
        (linked ? NOC_CMD_VC_LINKED : 0x0) | NOC_CMD_PATH_RESERVE | NOC_CMD_AT | NOC_CMD_RESP_MARKED |
            NOC_CMD_BRCST_PACKET | NOC_CMD_BRCST_XY(multicast_mode));
    NOC_WRITE_REG(NOC_BRCST_EXCLUDE, 0x0);
    NOC_WRITE_REG(
        NOC_AT_LEN_BE,
        NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(wrap) | NOC_AT_IND_32((addr >> 2) & 0x3) | NOC_AT_IND_32_SRC(0));
    NOC_WRITE_REG(NOC_AT_LEN_BE_1, 0x0);
    NOC_WRITE_REG(NOC_AT_DATA, incr);
    NOC_WRITE_REG(NOC_CMD_CTRL, 0x1);
}

bool noc_command_ready() { return (NOC_READ_REG(NOC_CMD_CTRL) == NOC_CTRL_STATUS_READY); }

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
