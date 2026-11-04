// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ethernet send from a Keraunos CCE (Quasar ACTIVE_ETH), behind metal's internal_::eth_* primitives in tunneling.h.
// Every send is an iDMA copy on command buffer 1, on TX queue q's request VC, counted under one transaction id.
// Addresses the router passes are flat addresses in this CCE's SRAM, cached or through the uncached alias; the
// destination is the same offset in the link peer's CCE SRAM. Only CCE0 is handled: its SRAM region starts at 0.

#pragma once

#include <cstdint>

#include "api/debug/assert.h"
#include "dev_mem_map.h"
#include "internal/risc_attribs.h"
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"

namespace internal_::cce_eth {

using namespace overlay;

constexpr CmdBuf kCmdBuf = CMDBUF_1;
constexpr uint32_t kTrid = CMDBUF_1_TRID_STATIC;
// Request VCs 12-15 reach the Ethernet backends: TX queue q is VC 12 + q.
constexpr uint32_t kFirstTxqReqVc = 12;
constexpr uint32_t kNumTxqs = 4;
constexpr uint64_t kMaxBytesInPacket = 0x1000;
// AXI_OPT_1.src_protocol of an inline write: the source register carries the data.
constexpr uint8_t kInlineSrcProtocol = 4;

// Keraunos local address of an HSIO tile's CCE SRAM: base + tile * stride + offset. The command buffer takes this
// view on both ends (chippy metal compat, CM-001). The Keraunos-E100 spec (v1.0.9, ch. 12) says a CCE issues Grendel
// system addresses and the HSIO remap translates them; the two have not been reconciled.
constexpr uint64_t kHsioLocalBase = 0x20000000;
constexpr uint64_t kHsioTileStride = 0x4000000;
constexpr uint64_t kHsioCceSramOffset = 0x2800000;

// MEM_IERISC_FABRIC_ETH_LINK_BASE, as the host writes it.
struct EthLink {
    uint32_t local_tile;
    uint32_t peer_tile;
    uint32_t reserved[2];
};
static_assert(sizeof(EthLink) == MEM_IERISC_FABRIC_ETH_LINK_SIZE);

FORCE_INLINE const volatile EthLink& eth_link() {
    return *reinterpret_cast<const volatile EthLink*>(
        static_cast<uintptr_t>(MEM_IERISC_FABRIC_ETH_LINK_BASE + MEM_IERISC_UNCACHED_BASE));
}

FORCE_INLINE uint32_t txq_req_vc(uint32_t q_num) {
    ASSERT(q_num < kNumTxqs);
    return kFirstTxqReqVc + q_num;
}

// Keraunos local address of the flat address @p flat (cached or uncached alias) in HSIO tile @p tile's CCE SRAM.
FORCE_INLINE uint64_t sram_local_addr(uint32_t tile, uint32_t flat) {
    const uint32_t offset = flat >= MEM_IERISC_UNCACHED_BASE ? flat - MEM_IERISC_UNCACHED_BASE : flat;
    ASSERT(offset < MEM_IERISC_L1_SIZE);
    return kHsioLocalBase + tile * kHsioTileStride + kHsioCceSramOffset + offset;
}

// True while the command buffer has no space on TX queue @p q_num's request VC. Never blocks.
FORCE_INLINE bool txq_busy(uint32_t q_num) { return idma_free_space_cmdbuf<kCmdBuf>(txq_req_vc(q_num)) == 0; }

FORCE_INLINE void setup_send(uint32_t q_num, uint32_t dest) {
    const uint32_t vc = txq_req_vc(q_num);
    reset_cmdbuf<kCmdBuf>();
    idma_setup_as_copy_cmdbuf<kCmdBuf>();
    setup_ongoing_cmdbuf<kCmdBuf>(AutoIncConfig{});
    setup_wrapping_vcs_cmdbuf<kCmdBuf>(VcRange{.start = vc, .end = vc});
    setup_trids_cmdbuf<kCmdBuf>(kTrid);
    set_dest_cmdbuf<kCmdBuf>(sram_local_addr(eth_link().peer_tile, dest));
}

// Copy @p num_bytes from @p src to @p dest on the link peer on TX queue @p q_num; returns once issued.
// Assumes !txq_busy(q_num).
FORCE_INLINE void send_bytes(uint32_t q_num, uint32_t src, uint32_t dest, uint32_t num_bytes) {
    // A core store to the source must reach SRAM before the copy reads it.
    asm volatile("fence" ::: "memory");
    setup_send(q_num, dest);
    set_src_cmdbuf<kCmdBuf>(sram_local_addr(eth_link().local_tile, src));
    set_len_cmdbuf<kCmdBuf>(num_bytes);
    setup_max_bytes_in_packet_cmdbuf<kCmdBuf>(kMaxBytesInPacket);
    setup_packet_tags_cmdbuf<kCmdBuf>(PacketTags{});
    issue_cmdbuf<kCmdBuf>();
}

// Write the 32-bit @p val to @p dest on the link peer as an inline write on TX queue @p q_num; returns once issued.
// Assumes !txq_busy(q_num). Unverified on hardware: compat's inline Ethernet writes are 8 bytes.
FORCE_INLINE void send_word(uint32_t q_num, uint32_t dest, uint32_t val) {
    setup_send(q_num, dest);
    set_src_cmdbuf<kCmdBuf>(val);
    set_len_cmdbuf<kCmdBuf>(sizeof(val));
    TT_ROCC_CMD_BUF_AXI_OPT_1_reg_u axi_opt_1;
    axi_opt_1.val = TT_ROCC_CMD_BUF_AXI_OPT_1_REG_DEFAULT;
    set_axi_opt_1_cmdbuf<kCmdBuf>(kInlineSrcProtocol, axi_opt_1.f.decouple_aw);
    setup_packet_tags_cmdbuf<kCmdBuf>(PacketTags{});
    issue_cmdbuf<kCmdBuf>();
}

}  // namespace internal_::cce_eth
