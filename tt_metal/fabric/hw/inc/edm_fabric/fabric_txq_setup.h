// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/hw/inc/ethernet/tt_eth_api.h"

constexpr uint64_t BCAST_MAC_ADDR = 0xFFFF'FFFF'FFFF;
constexpr uint64_t UNICAST_MAC_ADDR = 0x0000'0000'0000;
constexpr uint64_t MCAST_MAC_ADDR = 0x0100'0000'0001;

constexpr uint64_t RXQ0_MAC_ADDR = BCAST_MAC_ADDR;
constexpr uint64_t RXQ1_MAC_ADDR = MCAST_MAC_ADDR;

constexpr uint64_t TXQ0_MAC_ADDR = BCAST_MAC_ADDR;
constexpr uint64_t TXQ1_MAC_ADDR = MCAST_MAC_ADDR;

#define ETH_CORE_A_ETH_CTRL_A_MAC_RX_ADDR_ROUTING_REG_ADDR (0xFFB98154)
#define ETH_CORE_A_ETH_CTRL_A_MAC_RX_ROUTING_REG_ADDR (0xFFB98150)
#define ETH_CORE_A_ETH_TXQ_0_CTRL_REG_OFFSET (0x00000000)
#define ETH_CORE_A_ETH_TXQ_1_CTRL_REG_OFFSET (0x00000000)
#define ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_1__MAC_DA_HI_REG_ADDR (0xFFB9829C)
#define ETH_CORE_A_ETH_CTRL_A_ETH_TXPKT_CFG_A_1__MAC_DA_LO_REG_ADDR (0xFFB98298)
#define ETH_CORE_A_ETH_TXQ_0_CTRL_REG_ADDR (0xFFB90000)
#define ETH_CORE_A_ETH_TXQ_1_CTRL_REG_ADDR (0xFFB91000)
#define ETH_CORE_A_ETH_TXQ_0_TXPKT_CFG_SEL_HW_REG_ADDR (0xFFB90084)
#define ETH_CORE_A_ETH_TXQ_1_TXPKT_CFG_SEL_HW_REG_ADDR (0xFFB91084)
#define ETH_CORE_A_ETH_RXQ_0_CTRL_REG_ADDR (0xFFB94000)
#define ETH_CORE_A_ETH_RXQ_1_CTRL_REG_ADDR (0xFFB95000)
#define ETH_CORE_A_ETH_TXQ_0_TXPKT_CFG_SEL_SW_REG_ADDR (0xFFB90080)
#define ETH_CORE_A_ETH_TXQ_1_TXPKT_CFG_SEL_SW_REG_ADDR (0xFFB91080)

static void map_tx_queues_to_rx_queues() {
    static constexpr uint32_t rx_chan_bcast_data_bit_offset = 0;
    static constexpr uint32_t rx_chan_bcast_regwrite_bit_offset = 2;
    static constexpr uint32_t rx_chan_mcast_data_bit_offset = 4;
    static constexpr uint32_t rx_chan_mcast_regwrite_bit_offset = 6;
    static constexpr uint32_t rx_chan_other_data_bit_offset = 8;
    static constexpr uint32_t rx_chan_other_regwrite_bit_offset = 10;

    uint32_t register_fields_value = eth_reg_read(ETH_CORE_A_ETH_CTRL_A_MAC_RX_ADDR_ROUTING_REG_ADDR) & ~0xFF;

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
    apply_rxq_side_routing(0, RXQ0_MAC_ADDR, register_fields_value);
    apply_rxq_side_routing(1, RXQ1_MAC_ADDR, register_fields_value);
    eth_reg_write(ETH_CORE_A_ETH_CTRL_A_MAC_RX_ADDR_ROUTING_REG_ADDR, register_fields_value);
    eth_reg_write(ETH_CORE_A_ETH_CTRL_A_MAC_RX_ROUTING_REG_ADDR, 0);
}
typedef struct {
    uint32_t packet_resend_mode_active : 1;
    uint32_t rsvd_0 : 2;
    uint32_t disable_remote_drop_notification : 1;
} ETH_TXQ_CTRL_reg_t;

typedef union {
    uint32_t val;
    ETH_TXQ_CTRL_reg_t f;
} ETH_TXQ_CTRL_reg_u;

static void eth_enable_packet_mode(uint32_t qnum) {
    // enable packet resend
    ETH_TXQ_CTRL_reg_u txq_ctrl_reg;
    txq_ctrl_reg.f.packet_resend_mode_active = 1;
    eth_txq_reg_write(qnum, ETH_CORE_A_ETH_TXQ_0_CTRL_REG_OFFSET, txq_ctrl_reg.val);
    eth_txq_reg_write(qnum, ETH_CORE_A_ETH_TXQ_1_CTRL_REG_OFFSET, txq_ctrl_reg.val);

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
}
