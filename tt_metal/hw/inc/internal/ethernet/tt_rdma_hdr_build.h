// SPDX-License-Identifier: Apache-2.0
//
// TT-RDMA v1 header builder + CRC-32, shared by the on-core kernel and the host.
// Fills a tt_rdma_hdr_t (tt_rdma_wire.h) per the spec field offsets and computes
// header_cksum = CRC-32 over header bytes [0..27] (tt-rdma-wire-protocol-v1.md §1).
//
// Polynomial choice: standard CRC-32 (0x04C11DB7, reflected 0xEDB88320, init/final
// ~0) — the SAME polynomial as the ETH-CTRL ROCE_ICRC hardware engine on Blackhole
// (regs @ 0xFFB98100, RX_INIT/CTRL default residual 0xC704DD7B => this variant).
// Keeping the wire header_cksum on the HW polynomial lets the RX path offload the
// check to that engine (RX_CHECK_EN + RX_CALCULATED vs RX_RECEIVED, zero RISC
// cycles). This C routine is the reference/fallback and the host+gateway codec.
//
// Both x86 host and the erisc are little-endian, and tt_rdma_hdr_t is packed, so
// native struct stores already give the exact on-wire byte order. The golden
// vectors in tt_rdma_wire.h (TT_GOLDEN_SEND_HDR etc.) are the correctness oracle:
// filling the §7.1 SEND fields must reproduce header_cksum 0x69B1EDCC, and the
// canonical vector crc32("123456789") == 0xCBF43926.
#pragma once

#include "tt_rdma_wire.h"

// CRC-32 (IEEE 802.3 / RoCE iCRC poly): reflected, poly 0x04C11DB7 -> 0xEDB88320,
// init/final ~0. Bit-serial reference (matches the ETH-CTRL ROCE_ICRC engine).
static inline uint32_t tt_rdma_crc32(const uint8_t* p, uint32_t n) {
    uint32_t crc = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < n; ++i) {
        crc ^= (uint32_t)p[i];
        for (int b = 0; b < 8; ++b) {
            uint32_t mask = (uint32_t)(-(int32_t)(crc & 1u));
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return crc ^ 0xFFFFFFFFu;
}

// Populate a 32-byte header in place and stamp header_cksum. `h` must point at
// 32 writable bytes (a local tt_rdma_hdr_t on either side; copy to L1 after).
static inline void tt_rdma_build_hdr(
    tt_rdma_hdr_t* h,
    uint8_t opcode,
    uint8_t version_flags,
    uint16_t tag,
    uint32_t length,
    uint32_t seq,
    uint32_t rkey,
    uint64_t remote_offset,
    uint32_t imm_data) {
    h->opcode = opcode;
    h->version_flags = version_flags;
    h->tag = tag;
    h->length = length;
    h->seq = seq;
    h->rkey = rkey;
    h->remote_offset = remote_offset;
    h->imm_data = imm_data;
    h->header_cksum = tt_rdma_crc32((const uint8_t*)h, 28);
}
