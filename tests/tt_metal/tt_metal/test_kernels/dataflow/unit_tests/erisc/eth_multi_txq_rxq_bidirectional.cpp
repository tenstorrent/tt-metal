
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_handshake.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "tt_metal/hw/inc/ethernet/tunneling.h"
#include <tuple>
#include <cstdint>
#include <cstddef>
#include "debug/dprint.h"
#include "tt_metal/fabric/impl/kernels/edm_fabric/eth_core_a_reg.h"

constexpr uint32_t data_txq_id = get_compile_time_arg_val(0);
constexpr uint32_t ack_txq_id = get_compile_time_arg_val(1);

void eth_reg_write(uint32_t addr, uint32_t val) { ETH_WRITE_REG(addr, val); }

uint32_t eth_reg_read(uint32_t addr) { return ETH_READ_REG(addr); }

// void eth_txq_reg_write(uint32_t qnum, uint32_t offset, uint32_t val) {
//   ETH_WRITE_REG(ETH_CORE_A_ETH_TXQ_0_REG_MAP_BASE_ADDR+(qnum*ETH_TXQ_REGS_SIZE)+offset, val);
// }

// uint32_t eth_txq_reg_read(uint32_t qnum, uint32_t offset) {
//   return ETH_READ_REG(ETH_CORE_A_ETH_TXQ_0_REG_MAP_BASE_ADDR+(qnum*ETH_TXQ_REGS_SIZE)+offset);
// }

void eth_rxq_reg_write(uint32_t qnum, uint32_t offset, uint32_t val) {
    ETH_WRITE_REG(ETH_CORE_A_ETH_RXQ_0_REG_MAP_BASE_ADDR + (qnum * ETH_RXQ_REGS_SIZE) + offset, val);
}

uint32_t eth_rxq_reg_read(uint32_t qnum, uint32_t offset) {
    return ETH_READ_REG(ETH_CORE_A_ETH_RXQ_0_REG_MAP_BASE_ADDR + (qnum * ETH_RXQ_REGS_SIZE) + offset);
}

typedef enum {
    PORT_UNKNOWN,
    PORT_UP,
    PORT_DOWN,
    PORT_UNUSED,
} port_status_e;

typedef enum {
    LINK_TRAIN_TRAINING,
    LINK_TRAIN_SKIP,
    LINK_TRAIN_PASS,
    LINK_TRAIN_INT_LB,
    LINK_TRAIN_EXT_LB,
    LINK_TRAIN_TIMEOUT_MANUAL_EQ,
    LINK_TRAIN_TIMEOUT_ANLT,
    LINK_TRAIN_TIMEOUT_CDR_LOCK,
    LINK_TRAIN_TIMEOUT_BIST_LOCK,
    LINK_TRAIN_TIMEOUT_LINK_UP,
    LINK_TRAIN_TIMEOUT_CHIP_INFO,
} link_train_status_e;

static constexpr uint32_t NUM_SERDES_LANES = 8;

struct serdes_results_t {
    uint32_t postcode;           // 0
    uint32_t serdes_inst;        // 1
    uint32_t serdes_lane_mask;   // 2
    uint32_t target_speed;       // 3 - Target speed from the boot params
    uint32_t data_rate;          // 4
    uint32_t data_width;         // 5
    uint32_t spare_main[8 - 6];  // 6-7

    // Training retries
    uint32_t lt_retry_cnt;   // 8
    uint32_t spare[16 - 9];  // 9-15

    // BIST
    uint32_t bist_mode;       // 16
    uint32_t bist_test_time;  // 17
    // test_time in cycles for bist mode 0 and ms for bist mode 1
    uint32_t bist_err_cnt_nt[NUM_SERDES_LANES];           // 18-25
    uint32_t bist_err_cnt_55t32_nt[NUM_SERDES_LANES];     // 26-33
    uint32_t bist_err_cnt_overflow_nt[NUM_SERDES_LANES];  // 34-41

    uint32_t spare2[48 - 42];  // 42-47

    // Training times
    uint32_t man_eq_cmn_pstate_time;      // 48
    uint32_t man_eq_tx_ack_time;          // 49
    uint32_t man_eq_rx_ack_time;          // 50
    uint32_t man_eq_rx_iffsm_time;        // 51
    uint32_t man_eq_rx_eq_assert_time;    // 52
    uint32_t man_eq_rx_eq_deassert_time;  // 53
    uint32_t anlt_auto_neg_time;          // 54
    uint32_t anlt_link_train_time;        // 55
    uint32_t cdr_lock_time;               // 56
    uint32_t bist_lock_time;              // 57

    uint32_t spare_time[64 - 58];  // 58-63
};

struct eth_status_t {
    // Basic status
    uint32_t postcode;                 // 0
    port_status_e port_status;         // 1
    link_train_status_e train_status;  // 2
    uint32_t train_speed;              // 3 - Actual resulting speed from training

    // Live status/retrain related
    uint32_t retrain_count;   // 4
    uint32_t mac_pcs_errors;  // 5
    uint32_t corr_dw_hi;      // 6
    uint32_t corr_dw_lo;      // 7
    uint32_t uncorr_dw_hi;    // 8
    uint32_t uncorr_dw_lo;    // 9
    uint32_t frames_rxd_hi;   // 10
    uint32_t frames_rxd_lo;   // 11
    uint32_t bytes_rxd_hi;    // 12
    uint32_t bytes_rxd_lo;    // 13

    uint32_t spare[28 - 14];  // 14-27

    // Heartbeat
    uint32_t heartbeat[4];  // 28-31
};
struct ChipUID {
    uint64_t board_id;
    uint8_t asic_location;

    bool operator<(const ChipUID& other) const {
        return std::tie(board_id, asic_location) < std::tie(other.board_id, other.asic_location);
    }

    const bool operator==(const ChipUID& other) const {
        return board_id == other.board_id && asic_location == other.asic_location;
    }
};
struct macpcs_results_t {
    uint32_t postcode;  // 0

    uint32_t spare[24 - 1];  // 1-23

    // Training times
    uint32_t link_up_time;    // 24
    uint32_t chip_info_time;  // 25

    uint32_t spare_time[32 - 26];  // 26-31
};
struct chip_info_t {
    uint8_t pcb_type;  // 0
    uint8_t asic_location;
    uint8_t eth_id;
    uint8_t logical_eth_id;
    uint32_t board_id_hi;   // 1
    uint32_t board_id_lo;   // 2
    uint32_t mac_addr_org;  // 3
    uint32_t mac_addr_id;   // 4
    uint32_t spare[2];      // 5-6
    uint32_t ack;           // 7

    ChipUID get_chip_uid() const {
        ChipUID chip_uid;
        chip_uid.board_id = ((uint64_t)board_id_hi << 32) | board_id_lo;
        chip_uid.asic_location = asic_location;
        return chip_uid;
    }
};
struct fw_version_t {
    uint32_t patch : 8;
    uint32_t minor : 8;
    uint32_t major : 8;
    uint32_t unused : 8;
};
struct boot_results_t {
    eth_status_t eth_status;          // 0-31
    serdes_results_t serdes_results;  // 32 - 95
    macpcs_results_t macpcs_results;  // 96 - 127

    uint32_t spare[238 - 128];  // 128 - 237

    fw_version_t serdes_fw_ver;  // 238
    fw_version_t eth_fw_ver;     // 239
    chip_info_t local_info;      // 240 - 247
    chip_info_t remote_info;     // 248 - 255
};

// Initialize RX queue controller
// void eth_rxq_init(uint32_t q_num, /*uint32_t rec_buf_word_addr, uint32_t rec_buf_size_words,*/ bool rec_buf_wrap,
// bool packet_mode) {
//   // set up raw mode RX buffer
// //   eth_rxq_reg_write(q_num, ETH_CORE_A_ETH_RXQ_0_BUF_START_WORD_ADDR_REG_OFFSET, rec_buf_word_addr);
// //   eth_rxq_reg_write(q_num, ETH_CORE_A_ETH_RXQ_0_BUF_PTR_REG_OFFSET, 0);
// //   eth_rxq_reg_write(q_num, ETH_CORE_A_ETH_RXQ_0_BUF_SIZE_WORDS_REG_OFFSET, rec_buf_size_words);
// //   eth_rxq_reg_write(q_num, ETH_CORE_A_ETH_RXQ_1_BUF_START_WORD_ADDR_REG_OFFSET, rec_buf_word_addr);
// //   eth_rxq_reg_write(q_num, ETH_CORE_A_ETH_RXQ_1_BUF_PTR_REG_OFFSET, 0);
// //   eth_rxq_reg_write(q_num, ETH_CORE_A_ETH_RXQ_1_BUF_SIZE_WORDS_REG_OFFSET, rec_buf_size_words);

//   //uint32_t ctrl_val = eth_rxq_reg_read(q_num, ETH_CORE_A_ETH_RXQ_0_CTRL_REG_OFFSET);
//   ETH_RXQ_CTRL_reg_u ctrl_reg;
// //   ctrl_reg.val = eth_rxq_reg_read(q_num, ETH_CORE_A_ETH_RXQ_0_CTRL_REG_OFFSET);
//   ctrl_reg.val = eth_rxq_reg_read(q_num, ETH_CORE_A_ETH_RXQ_1_CTRL_REG_OFFSET);
//   DPRINT << "old ctrl_reg.val: " << (uint32_t)ctrl_reg.val << "\n";
//   DPRINT << "old packet_mode: " << (uint32_t)ctrl_reg.f.packet_mode << "\n";
//   DPRINT << "old ctrl_reg.f.buf_wrap: " << (uint32_t)ctrl_reg.f.buf_wrap << "\n";
//   if (packet_mode) {
//       ctrl_reg.f.packet_mode = 1;
//     } else {
//         ctrl_reg.f.packet_mode = 0;
//     }
//     if (rec_buf_wrap) {
//         ctrl_reg.f.buf_wrap = 1;
//     } else {
//         ctrl_reg.f.buf_wrap = 0;
//     }
//     //   eth_rxq_reg_write(q_num, ETH_CORE_A_ETH_RXQ_0_CTRL_REG_OFFSET, ctrl_reg.val);
//     eth_rxq_reg_write(q_num, ETH_CORE_A_ETH_RXQ_1_CTRL_REG_OFFSET, ctrl_reg.val);
//     ctrl_reg.val = eth_rxq_reg_read(q_num, ETH_CORE_A_ETH_RXQ_1_CTRL_REG_OFFSET);
//     DPRINT << "new ctrl_reg.val: " << (uint32_t)ctrl_reg.val << "\n";
//     DPRINT << "new packet_mode: " << (uint32_t)ctrl_reg.f.packet_mode << "\n";
//     DPRINT << "new ctrl_reg.f.buf_wrap: " << (uint32_t)ctrl_reg.f.buf_wrap << "\n";
// }

constexpr uint64_t BCAST_MAC_ADDR = 0xFFFF'FFFF'FFFF;
constexpr uint64_t UNICAST_MAC_ADDR = 0x0000'0000'0000;
constexpr uint64_t MCAST_MAC_ADDR = 0x0100'0000'0001;

constexpr uint64_t RXQ0_MAC_ADDR = BCAST_MAC_ADDR;
constexpr uint64_t RXQ1_MAC_ADDR = MCAST_MAC_ADDR;

constexpr uint64_t TXQ0_MAC_ADDR = BCAST_MAC_ADDR;
constexpr uint64_t TXQ1_MAC_ADDR = MCAST_MAC_ADDR;

static void map_tx_queues_to_rx_queues() {
    static constexpr uint32_t rx_chan_bcast_data_bit_offset = 0;
    static constexpr uint32_t rx_chan_bcast_regwrite_bit_offset = 2;
    static constexpr uint32_t rx_chan_mcast_data_bit_offset = 4;
    static constexpr uint32_t rx_chan_mcast_regwrite_bit_offset = 6;
    static constexpr uint32_t rx_chan_other_data_bit_offset = 8;
    static constexpr uint32_t rx_chan_other_regwrite_bit_offset = 10;

    uint32_t register_fields_value = eth_reg_read(ETH_CORE_A_ETH_CTRL_A_MAC_RX_ADDR_ROUTING_REG_ADDR) & ~0xFF;
    DPRINT << "BEFORE register_fields_value: " << (uint32_t)register_fields_value << "\n";
    // send all TXQ0 packets to RXQ0 -- we mapped TXQ0 to non-mcast,non-bcast packet types
    // for (auto offset :
    //      {rx_chan_other_data_bit_offset,
    //       rx_chan_other_regwrite_bit_offset}) {
    //     register_fields_value |= (0x0 << offset);
    // }
    auto apply_rxq_side_routing = [](uint32_t RXQ_ID, uint64_t ADDR_MAPPING, uint32_t& register_fields_value_out) {
        if (ADDR_MAPPING == BCAST_MAC_ADDR) {
            for (auto offset : {rx_chan_bcast_data_bit_offset, rx_chan_bcast_regwrite_bit_offset}) {
                register_fields_value_out |= (RXQ_ID << offset);
            }
        } else if (ADDR_MAPPING == UNICAST_MAC_ADDR) {
            for (auto offset : {rx_chan_other_data_bit_offset, rx_chan_other_regwrite_bit_offset}) {
                register_fields_value_out |= (RXQ_ID << offset);
            }
        } else if (ADDR_MAPPING == MCAST_MAC_ADDR) {
            for (auto offset : {rx_chan_mcast_data_bit_offset, rx_chan_mcast_regwrite_bit_offset}) {
                register_fields_value_out |= (RXQ_ID << offset);
            }
        }
    };

    // send all TXQ1 packets to RXQ1 -- we mapped TXQ1 to mcast,bcast packet types
    DPRINT << "AFTER register_fields_value: " << (uint32_t)register_fields_value << "\n";
    apply_rxq_side_routing(0, RXQ0_MAC_ADDR, register_fields_value);
    apply_rxq_side_routing(1, RXQ1_MAC_ADDR, register_fields_value);
    eth_reg_write(ETH_CORE_A_ETH_CTRL_A_MAC_RX_ADDR_ROUTING_REG_ADDR, register_fields_value);
    eth_reg_write(ETH_CORE_A_ETH_CTRL_A_MAC_RX_ROUTING_REG_ADDR, 0);
}

constexpr uint32_t BOOT_RESULTS_ADDR = 0x7CC00;
// #define TMP_8K_BUF_ADDR 0x7E000
// #define MAX_ETH_PACKET_SIZE 4096 // BH ETH supports 4KB packets
// #define ETH_PACKET_HEADER_SIZE 18 // TODO: Kind of arbitrary number, figure out if this is exact
static void eth_enable_packet_mode(
    uint32_t qnum,
    uint32_t max_packet_size_bytes,
    uint32_t min_packet_size_words,
    uint32_t remote_seq_timeout,
    uint32_t local_seq_update_timeout) {
    // Initializing the rx queue to enable packet mode
    //   eth_rxq_init(qnum, (TMP_8K_BUF_ADDR + MAX_ETH_PACKET_SIZE) >> 4, MAX_ETH_PACKET_SIZE >> 4, false,  true);

    // enable packet resend
    ETH_TXQ_CTRL_reg_u txq_ctrl_reg;
    txq_ctrl_reg.f.packet_resend_mode_active = 1;
    eth_txq_reg_write(qnum, ETH_CORE_A_ETH_TXQ_0_CTRL_REG_OFFSET, txq_ctrl_reg.val);
    eth_txq_reg_write(qnum, ETH_CORE_A_ETH_TXQ_1_CTRL_REG_OFFSET, txq_ctrl_reg.val);

    //   volatile boot_results_t* boot_results = reinterpret_cast<volatile boot_results_t*>(BOOT_RESULTS_ADDR);

    //   uint64_t src_mac_addr = ((uint64_t)(boot_results->local_info.mac_addr_org) << 24) |
    //   (uint64_t)(boot_results->local_info.mac_addr_id);
    // //   eth_reg_write(ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_0__MAC_SA_HI_REG_ADDR, src_mac_addr >> 32);
    // //   eth_reg_write(ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_0__MAC_SA_LO_REG_ADDR, src_mac_addr & 0xFFFFFFFF);
    //   eth_reg_write(ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_1__MAC_SA_HI_REG_ADDR, src_mac_addr >> 32);
    //   eth_reg_write(ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_1__MAC_SA_LO_REG_ADDR, src_mac_addr & 0xFFFFFFFF);

    // What are we doing here?
    // Essentially we are abusing the ethernet protocol and arbitrarily deciding to use mcast/bcast vs unicast
    // message types to distinguish destination receiver queue.
    //
    // For TXQ0, we are we are setting the destination mac address to 0 (RXQ0)
    //  -> these will map to "unicast" packets
    // For TXQ1, we are we are setting the destination mac address to 1 (RXQ1)
    //  -> these will map to "bcast"/"mcast" packets
    //
    // Further down, we will initialize the RX queues to be forwarded the ethernet packets of each
    // respective type described above.
    //   uint64_t unicast_mac_addr = 0;
    //   uint64_t bcast_mac_addr = 1;

    DPRINT << "OLD txq0 mac addr(hi): " << HEX()
           << (uint32_t)(eth_reg_read(ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_0__MAC_DA_HI_REG_ADDR)) << "\n";
    DPRINT << "OLD txq0 mac addr(lo): "
           << (uint32_t)(eth_reg_read(ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_0__MAC_DA_LO_REG_ADDR)) << "\n";
    DPRINT << "OLD txq1 mac addr: "
           << (uint32_t)(eth_reg_read(ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_1__MAC_DA_HI_REG_ADDR)) << "\n";

    DPRINT << "BEFORE ETH_CORE_A_ETH_CTRL_A_MAC_RX_ADDR_ROUTING_REG_ADDR: "
           << (uint32_t)eth_reg_read(ETH_CORE_A_ETH_CTRL_A_MAC_RX_ADDR_ROUTING_REG_ADDR) << "\n";
    DPRINT << "BEFORE ETH_CORE_A_ETH_CTRL_A_MAC_RX_ROUTING_REG_ADDR: "
           << (uint32_t)eth_reg_read(ETH_CORE_A_ETH_CTRL_A_MAC_RX_ROUTING_REG_ADDR) << "\n";
    DPRINT << "BEFORE ETH_CORE_A_ETH_RXQ_0_CTRL_REG_ADDR: "
           << (uint32_t)eth_reg_read(ETH_CORE_A_ETH_RXQ_0_CTRL_REG_ADDR) << "\n";
    DPRINT << "BEFORE ETH_CORE_A_ETH_RXQ_1_CTRL_REG_ADDR: "
           << (uint32_t)eth_reg_read(ETH_CORE_A_ETH_RXQ_1_CTRL_REG_ADDR) << "\n";

    //   eth_reg_write(ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_0__MAC_SA_HI_REG_ADDR, (TXQ0_MAC_ADDR >> 32) & 0xFFFF);
    //   eth_reg_write(ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_0__MAC_SA_LO_REG_ADDR, TXQ0_MAC_ADDR & 0xFFFFFFFF);
    //   eth_reg_write(ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_1__MAC_SA_HI_REG_ADDR, (TXQ1_MAC_ADDR >> 32) & 0xFFFF);
    //   eth_reg_write(ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_1__MAC_SA_LO_REG_ADDR, TXQ1_MAC_ADDR & 0xFFFFFFFF);

    // snijjar -- keep just in case
    //   eth_reg_write(ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_0__MAC_DA_HI_REG_ADDR, (TXQ0_MAC_ADDR >> 32) & 0xFFFF);
    //   eth_reg_write(ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_0__MAC_DA_LO_REG_ADDR, TXQ0_MAC_ADDR & 0xFFFFFFFF);
    //   eth_reg_write(ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_0__MAC_DA_HI_REG_ADDR, (TXQ0_MAC_ADDR >> 32) & 0xFFFF);
    //   eth_reg_write(ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_0__MAC_DA_LO_REG_ADDR, TXQ0_MAC_ADDR & 0xFFFFFFFF);
    eth_reg_write(ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_1__MAC_DA_HI_REG_ADDR, (TXQ1_MAC_ADDR >> 32) & 0xFFFF);
    eth_reg_write(ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_1__MAC_DA_LO_REG_ADDR, TXQ1_MAC_ADDR & 0xFFFFFFFF);

    // enable packet resend on each TXQ:
    auto enable_packet_resend_txq = [](uint32_t addr) {
        uint32_t val = 0;
        val |= (1 << 0);  // packet resend
        eth_reg_write(addr, val);
    };
    enable_packet_resend_txq(ETH_CORE_A_ETH_TXQ_0_CTRL_REG_ADDR);
    enable_packet_resend_txq(ETH_CORE_A_ETH_TXQ_1_CTRL_REG_ADDR);

    // Set each TXQ to use separate entries in the tx pkt config table
    eth_reg_write(ETH_CORE_A_ETH_TXQ_0_TXPKT_CFG_SEL_HW_REG_ADDR, 0);
    eth_reg_write(ETH_CORE_A_ETH_TXQ_1_TXPKT_CFG_SEL_HW_REG_ADDR, 1);
    auto set_txq_config_table_mapping = [](uint32_t txpkt_cfg_sel_sw_reg_addr, uint32_t mapping) {
        uint32_t val = 0;
        val |= (mapping << 0);  // raw
        val |= (mapping << 4);  // eth reg writes
        val |= (mapping << 8);  // eth packet sends
        eth_reg_write(txpkt_cfg_sel_sw_reg_addr, val);
    };
    // Set each TXQ to use separate entries in the tx pkt config table
    set_txq_config_table_mapping(ETH_CORE_A_ETH_TXQ_0_TXPKT_CFG_SEL_SW_REG_ADDR, 0);
    set_txq_config_table_mapping(ETH_CORE_A_ETH_TXQ_1_TXPKT_CFG_SEL_SW_REG_ADDR, 1);

    auto enable_packet_mode_rxq = [](uint32_t addr) {
        uint32_t val = 0;
        val |= (1 << 1);  // packet mode
        eth_reg_write(addr, val);
    };
    enable_packet_mode_rxq(ETH_CORE_A_ETH_RXQ_0_CTRL_REG_ADDR);
    enable_packet_mode_rxq(ETH_CORE_A_ETH_RXQ_1_CTRL_REG_ADDR);

    map_tx_queues_to_rx_queues();
    //   eth_rxq_init(ack_txq_id, false, true);
    DPRINT << "AFTER ETH_CORE_A_ETH_CTRL_A_MAC_RX_ADDR_ROUTING_REG_ADDR: " << HEX()
           << (uint32_t)eth_reg_read(ETH_CORE_A_ETH_CTRL_A_MAC_RX_ADDR_ROUTING_REG_ADDR) << "\n";
    DPRINT << "AFTER 0xffb98154: " << (uint32_t)eth_reg_read(0xffb98154) << "\n";
    DPRINT << "AFTER ETH_CORE_A_ETH_CTRL_A_MAC_RX_ROUTING_REG_ADDR: "
           << (uint32_t)eth_reg_read(ETH_CORE_A_ETH_CTRL_A_MAC_RX_ROUTING_REG_ADDR) << "\n";
    DPRINT << "AFTER MAC_DA_HI_REG_ADDR_0: "
           << (uint32_t)eth_reg_read(ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_0__MAC_DA_HI_REG_ADDR) << "\n";
    DPRINT << "AFTER MAC_DA_LO_REG_ADDR_0: "
           << (uint32_t)eth_reg_read(ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_0__MAC_DA_LO_REG_ADDR) << "\n";
    DPRINT << "AFTER MAC_DA_HI_REG_ADDR_1: "
           << (uint32_t)eth_reg_read(ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_1__MAC_DA_HI_REG_ADDR) << "\n";
    DPRINT << "AFTER MAC_DA_LO_REG_ADDR_1: "
           << (uint32_t)eth_reg_read(ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_1__MAC_DA_LO_REG_ADDR) << "\n";
    DPRINT << "AFTER ETH_CORE_A_ETH_TXQ_0_TXPKT_CFG_SEL_HW_REG_ADDR: "
           << (uint32_t)eth_reg_read(ETH_CORE_A_ETH_TXQ_0_TXPKT_CFG_SEL_HW_REG_ADDR) << "\n";
    DPRINT << "AFTER ETH_CORE_A_ETH_TXQ_1_TXPKT_CFG_SEL_HW_REG_ADDR: "
           << (uint32_t)eth_reg_read(ETH_CORE_A_ETH_TXQ_1_TXPKT_CFG_SEL_HW_REG_ADDR) << "\n";
    DPRINT << "AFTER ETH_CORE_A_ETH_TXQ_0_TXPKT_CFG_SEL_SW_REG_ADDR: "
           << (uint32_t)eth_reg_read(ETH_CORE_A_ETH_TXQ_0_TXPKT_CFG_SEL_SW_REG_ADDR) << "\n";
    DPRINT << "AFTER ETH_CORE_A_ETH_TXQ_1_TXPKT_CFG_SEL_SW_REG_ADDR: "
           << (uint32_t)eth_reg_read(ETH_CORE_A_ETH_TXQ_1_TXPKT_CFG_SEL_SW_REG_ADDR) << "\n";
    DPRINT << "AFTER ETH_CORE_A_ETH_RXQ_0_CTRL_REG_ADDR: " << (uint32_t)eth_reg_read(ETH_CORE_A_ETH_RXQ_0_CTRL_REG_ADDR)
           << "\n";
    DPRINT << "AFTER ETH_CORE_A_ETH_RXQ_1_CTRL_REG_ADDR: " << (uint32_t)eth_reg_read(ETH_CORE_A_ETH_RXQ_1_CTRL_REG_ADDR)
           << "\n";
    DPRINT << "AFTER 0xffb98290: " << (uint32_t)eth_reg_read(0xffb98290) << "\n";
    DPRINT << "-------------\n";
    DPRINT << "AFTER 0xffb90000: " << (uint32_t)eth_reg_read(0xffb90000) << "\n";
    DPRINT << "AFTER 0xffb90004: " << (uint32_t)eth_reg_read(0xffb90004) << "\n";
    DPRINT << "AFTER 0xffb90008: " << (uint32_t)eth_reg_read(0xffb90008) << "\n";
    DPRINT << "AFTER 0xffb9000C: " << (uint32_t)eth_reg_read(0xffb9000C) << "\n";
    DPRINT << "-------------\n";
    DPRINT << "AFTER 0xffb91080: " << (uint32_t)eth_reg_read(0xffb91080) << "\n";
    DPRINT << "AFTER 0xffb91084: " << (uint32_t)eth_reg_read(0xffb91084) << "\n";
    DPRINT << "AFTER 0xffb91088: " << (uint32_t)eth_reg_read(0xffb91088) << "\n";
    DPRINT << "AFTER 0xffb9108C: " << (uint32_t)eth_reg_read(0xffb9108C) << "\n";
    DPRINT << "-------------\n";
    DPRINT << "AFTER 0xffb91000: " << (uint32_t)eth_reg_read(0xffb91000) << "\n";
    DPRINT << "AFTER 0xffb91004: " << (uint32_t)eth_reg_read(0xffb91004) << "\n";
    DPRINT << "AFTER 0xffb91008: " << (uint32_t)eth_reg_read(0xffb91008) << "\n";
    DPRINT << "AFTER 0xffb9100C: " << (uint32_t)eth_reg_read(0xffb9100C) << "\n";
    DPRINT << "-------------\n";
    DPRINT << "AFTER 0xFFB94000: " << (uint32_t)eth_reg_read(0xFFB94000) << "\n";
    DPRINT << "AFTER 0xFFB94004: " << (uint32_t)eth_reg_read(0xFFB94004) << "\n";
    DPRINT << "AFTER 0xFFB94008: " << (uint32_t)eth_reg_read(0xFFB94008) << "\n";
    DPRINT << "AFTER 0xFFB9400C: " << (uint32_t)eth_reg_read(0xFFB9400C) << "\n";
    DPRINT << "-------------\n";
    DPRINT << "AFTER 0xffb95000: " << (uint32_t)eth_reg_read(0xffb95000) << "\n";
    DPRINT << "AFTER 0xffb95004: " << (uint32_t)eth_reg_read(0xffb95004) << "\n";
    DPRINT << "AFTER 0xffb95008: " << (uint32_t)eth_reg_read(0xffb95008) << "\n";
    DPRINT << "AFTER 0xffb9500C: " << (uint32_t)eth_reg_read(0xffb9500C) << "\n";
    DPRINT << "-------------\n";
}

static constexpr uint32_t CREDITS_STREAM_ID = 0;
static constexpr uint32_t ACK_STREAM_ID = 1;

// static constexpr uint32_t data_txq_id = 0;
// static constexpr uint32_t ack_txq_id = 1;

static constexpr uint32_t PAYLOAD_SIZE = 64;

void kernel_main() {
    size_t arg_idx = 0;
    uint32_t handshake_addr = get_arg_val<uint32_t>(arg_idx++);
    bool is_handshake_sender = get_arg_val<uint32_t>(arg_idx++);
    uint32_t local_eth_l1_src_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_eth_l1_dst_addr = get_arg_val<uint32_t>(arg_idx++);
    int num_messages = get_arg_val<uint32_t>(arg_idx++);

    DPRINT << "local_eth_l1_src_addr: " << (uint32_t)local_eth_l1_src_addr << "\n";
    DPRINT << "remote_eth_l1_dst_addr: " << (uint32_t)remote_eth_l1_dst_addr << "\n";
    DPRINT << "data_txq_id: " << (uint32_t)data_txq_id << "\n";
    DPRINT << "ack_txq_id: " << (uint32_t)ack_txq_id << "\n";

    if constexpr (data_txq_id != ack_txq_id) {
        DPRINT << "Enabling packet mode for ack txq\n";
        eth_enable_packet_mode(ack_txq_id, 4096, 0, 40000, 8000);
    }

    init_ptr_val(CREDITS_STREAM_ID, 0);
    init_ptr_val(ACK_STREAM_ID, 0);

    DPRINT << "Handshake\n";
    // Handshake to make sure it's safe to start sending
    if (is_handshake_sender) {
        erisc::datamover::handshake::sender_side_handshake(handshake_addr);
    } else {
        erisc::datamover::handshake::receiver_side_handshake(handshake_addr);
    }
    // if constexpr (data_txq_id != ack_txq_id) {
    //     eth_reg_write(ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_0__MAC_DA_HI_REG_ADDR, TXQ0_MAC_ADDR >> 32);
    //     eth_reg_write(ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_0__MAC_DA_LO_REG_ADDR, TXQ0_MAC_ADDR & 0xFFFFFFFF);
    // }
    DPRINT << "Handshake done\n";
    bool has_unsent_messages = true;
    bool has_unsent_acks = true;
    int num_messages_sent = 0;
    int num_acks_sent = 0;
    int last_printed_ack = 0;
    size_t idle_count = 0;
    while (has_unsent_messages || has_unsent_acks) {
        // Send Messages
        bool current_ack = get_ptr_val<ACK_STREAM_ID>();
        if (current_ack != last_printed_ack) {
            DPRINT << "ACK_CNT: " << (uint32_t)current_ack << "\n";
            last_printed_ack = current_ack;
        }
        if (has_unsent_messages) {
            *reinterpret_cast<uint32_t*>(local_eth_l1_src_addr) = num_messages_sent + 1;
            uint32_t credits_available = get_ptr_val<CREDITS_STREAM_ID>();
            DPRINT << "Sending message\n";
            while (internal_::eth_txq_is_busy(data_txq_id)) {
            }
            internal_::eth_send_packet_bytes_unsafe(
                data_txq_id, local_eth_l1_src_addr, remote_eth_l1_dst_addr, PAYLOAD_SIZE);
            while (internal_::eth_txq_is_busy(data_txq_id)) {
            }
            remote_update_ptr_val<CREDITS_STREAM_ID, data_txq_id>(1);
            num_messages_sent++;
            has_unsent_messages = num_messages_sent < num_messages;
            if (!has_unsent_messages) {
                DPRINT << "No more messages to send\n";
            }
            idle_count = 0;
            DPRINT << "\tdone\n";
        }

        // Send Acks
        if (has_unsent_acks) {
            if (get_ptr_val<CREDITS_STREAM_ID>() > num_acks_sent) {
                DPRINT << "Sending ack\n";
                size_t count = 0;
                while (internal_::eth_txq_is_busy(ack_txq_id)) {
                    if (count > 1000000000) {
                        DPRINT << "STALL\n";
                        auto STATUS = eth_txq_reg_read(ack_txq_id, ETH_CORE_A_ETH_TXQ_1_STATUS_REG_OFFSET);
                        DPRINT << "TXQ0 STATUS: "
                               << (uint32_t)eth_txq_reg_read(data_txq_id, ETH_CORE_A_ETH_TXQ_0_STATUS_REG_OFFSET)
                               << "\n";
                        DPRINT << "TXQ1 STATUS: "
                               << (uint32_t)eth_txq_reg_read(ack_txq_id, ETH_CORE_A_ETH_TXQ_1_STATUS_REG_OFFSET)
                               << "\n";
                        DPRINT << "RXQ0 STATUS: "
                               << (uint32_t)eth_rxq_reg_read(data_txq_id, ETH_CORE_A_ETH_RXQ_0_RXSTATUS_REG_OFFSET)
                               << "\n";
                        DPRINT << "RXQ1 STATUS: "
                               << (uint32_t)eth_rxq_reg_read(ack_txq_id, ETH_CORE_A_ETH_RXQ_1_RXSTATUS_REG_OFFSET)
                               << "\n";
                        count = 0;
                    }
                    count++;
                }
                DPRINT << "\tdone\n";
                remote_update_ptr_val<ACK_STREAM_ID, ack_txq_id>(1);
                num_acks_sent++;
                has_unsent_acks = num_acks_sent < num_messages;
                if (!has_unsent_acks) {
                    DPRINT << "No more acks to send\n";
                }
                idle_count = 0;
            }
        }
        idle_count++;
        if (idle_count > 1000000000) {
            DPRINT << "IDLE\n";
            DPRINT << "TXQ0 STATUS: " << (uint32_t)eth_txq_reg_read(data_txq_id, ETH_CORE_A_ETH_TXQ_0_STATUS_REG_OFFSET)
                   << "\n";
            DPRINT << "TXQ1 STATUS: " << (uint32_t)eth_txq_reg_read(ack_txq_id, ETH_CORE_A_ETH_TXQ_1_STATUS_REG_OFFSET)
                   << "\n";
            DPRINT << "RXQ0 STATUS: "
                   << (uint32_t)eth_rxq_reg_read(data_txq_id, ETH_CORE_A_ETH_RXQ_0_RXSTATUS_REG_OFFSET) << "\n";
            DPRINT << "RXQ1 STATUS: "
                   << (uint32_t)eth_rxq_reg_read(ack_txq_id, ETH_CORE_A_ETH_RXQ_1_RXSTATUS_REG_OFFSET) << "\n";
            idle_count = 0;
        }
    }

    DPRINT << "Waiting for remaining messages\n";
    while (get_ptr_val<CREDITS_STREAM_ID>() < num_messages_sent) {
    }

    DPRINT << "Checking if messages were received\n";
    if (*reinterpret_cast<int32_t*>(remote_eth_l1_dst_addr) != num_messages_sent) {
        DPRINT << "val: " << (uint32_t)*reinterpret_cast<int32_t*>(remote_eth_l1_dst_addr) << "\n";
        while (true) {
        }  // didn't get data
    }

    DPRINT << "KERNEL DONE\n";
    // constexpr uint32_t num_bytes_per_send = get_compile_time_arg_val(0);
    // constexpr uint32_t num_bytes_per_send_word_size = get_compile_time_arg_val(1);
}
