
#include "noc.h"
#include <stdint.h>
#include <stdbool.h>
#include "noc_parameters.h"
#include "tt_metal/src/firmware/riscv/common/risc_attribs.h"

////

#ifdef TB_NOC

#include "noc_api_dpi.h"

#else

#define NOC_WRITE_REG(addr, val) ((*((volatile tt_reg_ptr uint32_t*)(noc_get_cmd_buf()*NOC_CMD_BUF_OFFSET+noc_get_active_instance()*NOC_INSTANCE_OFFSET+(addr)))) = (val))
#define NOC_READ_REG(addr)       (*((volatile tt_reg_ptr uint32_t*)(noc_get_cmd_buf()*NOC_CMD_BUF_OFFSET+noc_get_active_instance()*NOC_INSTANCE_OFFSET+(addr))))

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


static void noc_burst(uint64_t src_addr, uint64_t dst_addr, uint16_t len, bool linked, bool posted, bool static_vc_alloc, uint32_t static_vc, bool multicast, uint32_t multicast_mode, bool src_local, uint32_t vc_arb_priority) {
  while (!noc_command_ready());
  NOC_WRITE_REG(NOC_TARG_ADDR_LO, (uint32_t)(src_addr & 0xFFFFFFFF));
  NOC_WRITE_REG(NOC_TARG_ADDR_MID, (uint32_t)(src_addr >> 32));
  NOC_WRITE_REG(NOC_RET_ADDR_LO, (uint32_t)(dst_addr & 0xFFFFFFFF));
  NOC_WRITE_REG(NOC_RET_ADDR_MID, (uint32_t)(dst_addr >> 32));
  NOC_WRITE_REG(NOC_AT_LEN_BE, len);
  if (multicast) {
    NOC_WRITE_REG(NOC_CTRL, (linked ? NOC_CMD_VC_LINKED : 0x0) |
                            (static_vc_alloc ? NOC_CMD_VC_STATIC : 0x0) |
                            NOC_CMD_STATIC_VC(static_vc) |
                            NOC_CMD_PATH_RESERVE |
                            NOC_CMD_CPY |
                            (src_local ? NOC_CMD_WR : NOC_CMD_RD) |
                            NOC_CMD_BRCST_PACKET |
                            (posted ? 0x0 : NOC_CMD_RESP_MARKED) |
                            NOC_CMD_BRCST_XY(multicast_mode));
  } else {
    NOC_WRITE_REG(NOC_CTRL, (linked ? NOC_CMD_VC_LINKED : 0x0) |
                            (static_vc_alloc ? NOC_CMD_VC_STATIC : 0x0) |
                            NOC_CMD_STATIC_VC(static_vc) |
                            NOC_CMD_CPY |
                            (src_local ? NOC_CMD_WR : NOC_CMD_RD) |
                            (posted ? 0x0 : NOC_CMD_RESP_MARKED) |
                            NOC_CMD_ARB_PRIORITY(vc_arb_priority));
  }
  NOC_WRITE_REG(NOC_CMD_CTRL, 0x1);
}


static void noc_transfer(uint64_t src_addr, uint64_t dst_addr, uint32_t size, bool linked, bool posted, bool static_vc_alloc, uint32_t static_vc, bool multicast, uint32_t multicast_mode, bool src_local, uint32_t vc_arb_priority) {
  while (size > NOC_MAX_BURST_SIZE) {
    noc_burst(src_addr, dst_addr, NOC_MAX_BURST_SIZE, linked, posted, static_vc_alloc, static_vc, multicast, multicast_mode, src_local, vc_arb_priority);
    src_addr += NOC_MAX_BURST_SIZE;
    dst_addr += NOC_MAX_BURST_SIZE;
    size -= NOC_MAX_BURST_SIZE;
  }
  noc_burst(src_addr, dst_addr, size, linked, posted, static_vc_alloc, static_vc, multicast, multicast_mode, src_local, vc_arb_priority);
}


static bool unicast_addr_local(uint64_t addr) {
  uint32_t local_node_id = noc_local_node_id();
  uint32_t local_x = (local_node_id & NOC_NODE_ID_MASK);
  uint32_t local_y = ((local_node_id>>NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK);
  return (NOC_UNICAST_ADDR_X(addr) == local_x) && (NOC_UNICAST_ADDR_Y(addr) == local_y);
}

////

void noc_copy(uint64_t src_addr, uint64_t dst_addr, uint32_t size, bool linked, bool posted, bool static_vc_alloc, uint32_t static_vc, uint32_t vc_arb_priority) {
  bool src_local = unicast_addr_local(src_addr);
  if (!src_local) {
    posted = true;
  }
  noc_transfer(src_addr, dst_addr, size, linked, posted, static_vc_alloc, static_vc, false, 0, src_local, vc_arb_priority);
}

static void transfer_word_be(uint64_t src_addr, uint64_t dst_addr, uint32_t be, bool linked, bool posted, bool static_vc_alloc, uint32_t static_vc, bool multicast, uint32_t multicast_mode) {
  while (!noc_command_ready());
  NOC_WRITE_REG(NOC_TARG_ADDR_LO, (uint32_t)(src_addr & 0xFFFFFFFF));
  NOC_WRITE_REG(NOC_TARG_ADDR_MID, (uint32_t)(src_addr >> 32));
  NOC_WRITE_REG(NOC_RET_ADDR_LO, (uint32_t)(dst_addr & 0xFFFFFFFF));
  NOC_WRITE_REG(NOC_RET_ADDR_MID, (uint32_t)(dst_addr >> 32));
  NOC_WRITE_REG(NOC_AT_LEN_BE, be);
  if (multicast) {
    NOC_WRITE_REG(NOC_CTRL, (linked ? NOC_CMD_VC_LINKED : 0x0) |
                            (static_vc_alloc ? NOC_CMD_VC_STATIC : 0x0) |
                            NOC_CMD_STATIC_VC(static_vc) |
                            NOC_CMD_PATH_RESERVE |
                            NOC_CMD_WR |
                            NOC_CMD_WR_BE |
                            NOC_CMD_BRCST_PACKET |
                            (posted ? 0x0 : NOC_CMD_RESP_MARKED) |
                            NOC_CMD_BRCST_XY(multicast_mode));
  } else {
    NOC_WRITE_REG(NOC_CTRL, (linked ? NOC_CMD_VC_LINKED : 0x0) |
                            (static_vc_alloc ? NOC_CMD_VC_STATIC : 0x0) |
                            NOC_CMD_STATIC_VC(static_vc) |
                            NOC_CMD_WR |
                            NOC_CMD_WR_BE |
                            (posted ? 0x0 : NOC_CMD_RESP_MARKED));
  }
  NOC_WRITE_REG(NOC_CMD_CTRL, 0x1);
}

static void noc_transfer_dw_inline(uint64_t dst_addr, uint32_t val, uint8_t be, bool linked, bool posted, bool static_vc_alloc, uint32_t static_vc, bool multicast, uint32_t multicast_mode) {

  while (!noc_command_ready());
  NOC_WRITE_REG(NOC_TARG_ADDR_LO, (uint32_t)(dst_addr & 0xFFFFFFFF));
  NOC_WRITE_REG(NOC_TARG_ADDR_MID, (uint32_t)(dst_addr >> 32));

  uint32_t be32 = be;
  uint32_t be_shift = (dst_addr & (NOC_WORD_BYTES-1));
  be32 = (be32 << be_shift);
  NOC_WRITE_REG(NOC_AT_LEN_BE, be32);

  NOC_WRITE_REG(NOC_AT_DATA, val);

  if (multicast) {
    NOC_WRITE_REG(NOC_CTRL,
                  (linked ? NOC_CMD_VC_LINKED : 0x0) |
                  (static_vc_alloc ? NOC_CMD_VC_STATIC : 0x0) |
                  NOC_CMD_STATIC_VC(static_vc) |
                  NOC_CMD_PATH_RESERVE |
                  NOC_CMD_WR |
                  NOC_CMD_WR_INLINE |
                  NOC_CMD_BRCST_PACKET |
                  (posted ? 0x0 : NOC_CMD_RESP_MARKED) |
                  NOC_CMD_BRCST_XY(multicast_mode));
  } else {
    NOC_WRITE_REG(NOC_CTRL,
                  (linked ? NOC_CMD_VC_LINKED : 0x0) |
                  (static_vc_alloc ? NOC_CMD_VC_STATIC : 0x0) |
                  NOC_CMD_STATIC_VC(static_vc) |
                  NOC_CMD_WR |
                  NOC_CMD_WR_INLINE |
                  (posted ? 0x0 : NOC_CMD_RESP_MARKED));
  }
  NOC_WRITE_REG(NOC_CMD_CTRL, 0x1);
}


void noc_write_dw_inline(uint64_t dst_addr, uint32_t val, uint8_t be, bool linked, bool posted, bool static_vc_alloc, uint32_t static_vc) {
  noc_transfer_dw_inline(dst_addr, val, be, linked, posted, static_vc_alloc, static_vc, false, 0);
}

void noc_multicast_write_dw_inline(uint64_t dst_addr, uint32_t val, uint32_t multicast_mode, uint8_t be, bool linked, bool posted, bool static_vc_alloc, uint32_t static_vc) {
  noc_transfer_dw_inline(dst_addr, val, be, linked, posted, static_vc_alloc, static_vc, true, multicast_mode);
}


void noc_copy_word_be(uint64_t src_addr, uint64_t dst_addr, uint32_t be, bool linked, bool posted, bool static_vc_alloc, uint32_t static_vc) {
  transfer_word_be(src_addr, dst_addr, be, linked, posted, static_vc_alloc, static_vc, false, 0);
}


void noc_multicast_copy_word_be(uint64_t src_addr, uint64_t dst_addr, uint32_t multicast_mode, uint32_t be, bool linked, bool posted, bool static_vc_alloc, uint32_t static_vc) {
  transfer_word_be(src_addr, dst_addr, be, linked, posted, static_vc_alloc, static_vc, true, multicast_mode);
}



void noc_multicast_copy(uint64_t src_addr, uint64_t dst_addr, uint32_t multicast_mode, uint32_t size, bool linked, bool posted, bool static_vc_alloc, uint32_t static_vc) {
  noc_transfer(src_addr, dst_addr, size, linked, posted, static_vc_alloc, static_vc, true, multicast_mode, unicast_addr_local(src_addr), 0);
}


void noc_atomic_increment(uint64_t addr, uint32_t incr, uint32_t wrap, bool linked) {
  while (!noc_command_ready());
  NOC_WRITE_REG(NOC_TARG_ADDR_LO, (uint32_t)(addr & 0xFFFFFFFF));
  NOC_WRITE_REG(NOC_TARG_ADDR_MID, (uint32_t)(addr >> 32));
  // NOC_WRITE_REG(NOC_RET_ADDR_LO, 0);
  // NOC_WRITE_REG(NOC_RET_ADDR_MID, 0);
  NOC_WRITE_REG(NOC_CTRL, (linked ? NOC_CMD_VC_LINKED : 0x0) | NOC_CMD_AT);
  NOC_WRITE_REG(NOC_AT_LEN_BE, NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(wrap) | NOC_AT_IND_32((addr>>2) & 0x3) | NOC_AT_IND_32_SRC(0));
  NOC_WRITE_REG(NOC_AT_DATA, incr);
  NOC_WRITE_REG(NOC_CMD_CTRL, 0x1);
}


void noc_atomic_read_and_increment(uint64_t addr, uint32_t incr, uint32_t wrap, uint64_t read_addr, bool linked) {
  while (!noc_command_ready());
  NOC_WRITE_REG(NOC_TARG_ADDR_LO, (uint32_t)(addr & 0xFFFFFFFF));
  NOC_WRITE_REG(NOC_TARG_ADDR_MID, (uint32_t)(addr >> 32));
  NOC_WRITE_REG(NOC_RET_ADDR_LO, (uint32_t)(read_addr & 0xFFFFFFFF));
  NOC_WRITE_REG(NOC_RET_ADDR_MID, (uint32_t)(read_addr >> 32));
  NOC_WRITE_REG(NOC_CTRL, (linked ? NOC_CMD_VC_LINKED : 0x0) |
                         NOC_CMD_AT |
                         NOC_CMD_RESP_MARKED);
  NOC_WRITE_REG(NOC_AT_LEN_BE, NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(wrap) | NOC_AT_IND_32((addr>>2) & 0x3) | NOC_AT_IND_32_SRC(0));
  NOC_WRITE_REG(NOC_AT_DATA, incr);
  NOC_WRITE_REG(NOC_CMD_CTRL, 0x1);
}


void noc_multicast_atomic_increment(uint64_t addr, uint32_t multicast_mode, uint32_t incr, uint32_t wrap, bool linked) {
  while (!noc_command_ready());
  NOC_WRITE_REG(NOC_TARG_ADDR_LO, (uint32_t)(addr & 0xFFFFFFFF));
  NOC_WRITE_REG(NOC_TARG_ADDR_MID, (uint32_t)(addr >> 32));
  // NOC_WRITE_REG(NOC_RET_ADDR_LO, 0);
  // NOC_WRITE_REG(NOC_RET_ADDR_MID, 0);
  NOC_WRITE_REG(NOC_CTRL, (linked ? NOC_CMD_VC_LINKED : 0x0) |
                         NOC_CMD_PATH_RESERVE |
                         NOC_CMD_AT |
                         NOC_CMD_BRCST_PACKET |
                         NOC_CMD_BRCST_XY(multicast_mode));
  NOC_WRITE_REG(NOC_AT_LEN_BE, NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(wrap) | NOC_AT_IND_32((addr>>2) & 0x3) | NOC_AT_IND_32_SRC(0));
  NOC_WRITE_REG(NOC_AT_DATA, incr);
  NOC_WRITE_REG(NOC_CMD_CTRL, 0x1);
}


void noc_multicast_atomic_read_and_increment(uint64_t addr, uint32_t multicast_mode, uint32_t incr, uint32_t wrap, uint64_t read_addr, bool linked) {
  while (!noc_command_ready());
  NOC_WRITE_REG(NOC_TARG_ADDR_LO, (uint32_t)(addr & 0xFFFFFFFF));
  NOC_WRITE_REG(NOC_TARG_ADDR_MID, (uint32_t)(addr >> 32));
  NOC_WRITE_REG(NOC_RET_ADDR_LO, (uint32_t)(read_addr & 0xFFFFFFFF));
  NOC_WRITE_REG(NOC_RET_ADDR_MID, (uint32_t)(read_addr >> 32));
  NOC_WRITE_REG(NOC_CTRL, (linked ? NOC_CMD_VC_LINKED : 0x0) |
                         NOC_CMD_PATH_RESERVE |
                         NOC_CMD_AT |
                         NOC_CMD_RESP_MARKED |
                         NOC_CMD_BRCST_PACKET |
                         NOC_CMD_BRCST_XY(multicast_mode));
  NOC_WRITE_REG(NOC_AT_LEN_BE, NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(wrap) | NOC_AT_IND_32((addr>>2) & 0x3) | NOC_AT_IND_32_SRC(0));
  NOC_WRITE_REG(NOC_AT_DATA, incr);
  NOC_WRITE_REG(NOC_CMD_CTRL, 0x1);
}


bool noc_command_ready() {
  return (NOC_READ_REG(NOC_CMD_CTRL) == NOC_CTRL_STATUS_READY);
}

uint32_t noc_atomic_read_updates_completed() {
  uint32_t save_cmd_buf = noc_get_cmd_buf();
  noc_set_cmd_buf(0);
  uint32_t result =noc_status_reg(NIU_MST_ATOMIC_RESP_RECEIVED);
  noc_set_cmd_buf(save_cmd_buf);
  return result;
}

volatile uint32_t noc_wr_ack_received() {
  uint32_t save_cmd_buf = noc_get_cmd_buf();
  noc_set_cmd_buf(0);
  uint32_t result =noc_status_reg(NIU_MST_WR_ACK_RECEIVED);
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

void noc_ecc_cfg_stage_1(bool header_ckh_bits_en)
{
  uint32_t mask;
  uint32_t cfg_reg;

  cfg_reg = noc_get_cfg_reg(ROUTER_CFG_0);
  mask = (1 << ROUTER_CFG_0_ECC_HEADER_CHKBITS_EN);
  cfg_reg = (cfg_reg & ~mask) | (header_ckh_bits_en << ROUTER_CFG_0_ECC_HEADER_CHKBITS_EN);
  noc_set_cfg_reg(ROUTER_CFG_0, cfg_reg);
}

void noc_ecc_cfg_stage_2(bool niu_mem_parity_en, bool router_mem_parity_en, bool header_secded_en, bool mem_parity_int_en, bool header_sec_int_en, bool header_ded_int_en)
{
  uint32_t mask;
  uint32_t cfg_reg;

  cfg_reg = noc_get_cfg_reg(NIU_CFG_0);
  mask = (1 << NIU_CFG_0_ECC_NIU_MEM_PARITY_EN) | (1 << NIU_CFG_0_ECC_MEM_PARITY_INT_EN) | (1 << NIU_CFG_0_ECC_HEADER_1B_INT_EN) | (1 << NIU_CFG_0_ECC_HEADER_2B_INT_EN);
  cfg_reg = (cfg_reg & ~mask) | (niu_mem_parity_en << NIU_CFG_0_ECC_NIU_MEM_PARITY_EN) | (mem_parity_int_en << NIU_CFG_0_ECC_MEM_PARITY_INT_EN) | (header_sec_int_en << NIU_CFG_0_ECC_HEADER_1B_INT_EN) | (header_ded_int_en << NIU_CFG_0_ECC_HEADER_2B_INT_EN);
  noc_set_cfg_reg(NIU_CFG_0, cfg_reg);

  cfg_reg = noc_get_cfg_reg(ROUTER_CFG_0);
  mask = (1 << ROUTER_CFG_0_ECC_ROUTER_MEM_PARITY_EN) | (1 << ROUTER_CFG_0_ECC_HEADER_SECDED_EN);
  cfg_reg = (cfg_reg & ~mask) | (router_mem_parity_en << ROUTER_CFG_0_ECC_ROUTER_MEM_PARITY_EN) | (header_secded_en << ROUTER_CFG_0_ECC_HEADER_SECDED_EN);
  noc_set_cfg_reg(ROUTER_CFG_0, cfg_reg);
}

void noc_ecc_clear_err(bool clear_mem_parity_err, bool clear_header_sec, bool clear_header_ded)
{
  uint32_t save_cmd_buf = noc_get_cmd_buf();
  noc_set_cmd_buf(0);
  NOC_WRITE_REG(ECC_CTRL, ((clear_mem_parity_err | (clear_header_sec << 1) | (clear_header_ded << 2)) & 0x7));
  noc_set_cmd_buf(save_cmd_buf);
}

void noc_ecc_force_err(bool force_mem_parity_err, bool force_header_sec, bool force_header_ded)
{
  uint32_t save_cmd_buf = noc_get_cmd_buf();
  noc_set_cmd_buf(0);
  NOC_WRITE_REG(ECC_CTRL, (((force_mem_parity_err | (force_header_sec << 1) | (force_header_ded << 2)) & 0x7) << 3));
  noc_set_cmd_buf(save_cmd_buf);
}

uint32_t noc_ecc_get_num_mem_parity_errs()
{
  uint32_t save_cmd_buf = noc_get_cmd_buf();
  noc_set_cmd_buf(0);
  uint32_t result = NOC_READ_REG(NUM_MEM_PARITY_ERR);
  noc_set_cmd_buf(save_cmd_buf);
  return result;
}

uint32_t noc_ecc_get_num_header_sec()
{
  uint32_t save_cmd_buf = noc_get_cmd_buf();
  noc_set_cmd_buf(0);
  uint32_t result = NOC_READ_REG(NUM_HEADER_1B_ERR);
  noc_set_cmd_buf(save_cmd_buf);
  return result;
}

uint32_t noc_ecc_get_num_header_ded()
{
  uint32_t save_cmd_buf = noc_get_cmd_buf();
  noc_set_cmd_buf(0);
  uint32_t result = NOC_READ_REG(NUM_HEADER_2B_ERR);
  noc_set_cmd_buf(save_cmd_buf);
  return result;
}
