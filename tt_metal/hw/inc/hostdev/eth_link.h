// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// The Ethernet-link record of a Keraunos CCE, at MEM_IERISC_FABRIC_ETH_LINK_BASE (HalL1MemAddrType::ETH_LINK). The
// host writes it before launch; the router reads it (tt-2xx/quasar/eth_cce_send.h).
struct EthLink {
    uint32_t peer_tile;  // HSIO tile of the link peer's CCE
    uint32_t valid;      // ETH_LINK_VALID once the host has written the record
    uint32_t reserved[2];
};

static_assert(sizeof(EthLink) == 16);

inline constexpr uint32_t ETH_LINK_VALID = 0x4B4E494C;  // "LINK"
