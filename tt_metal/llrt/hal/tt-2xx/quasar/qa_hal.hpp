// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llrt/hal.hpp"

namespace tt::tt_metal::quasar {

// If you are trying to include this file and you aren't hal...you are doing something wrong
struct EthFwMailbox {
    uint32_t msg;
    uint32_t arg[3];
};

struct AllEthMailbox {
    EthFwMailbox mailbox[4];
};

uint32_t get_l1_size(IpVariant variant);
uint32_t get_grendel_l1_size();
uint32_t get_horizon_l1_size();
uint32_t get_trinity_l1_size();

dev_msgs::Factory create_tensix_dev_msgs_factory();
HalCoreInfoType create_tensix_mem_map(IpVariant variant);
HalCoreInfoType create_active_eth_mem_map();
HalCoreInfoType create_idle_eth_mem_map();
HalCoreInfoType create_dispatch_mem_map(IpVariant variant);
// drisc_l1_noc_offset 0 keeps the Mimir-local tag. A package ATT descriptor passes the CCE SRAM window.
HalCoreInfoType create_dram_mem_map(uint64_t drisc_l1_noc_offset = 0);

}  // namespace tt::tt_metal::quasar
