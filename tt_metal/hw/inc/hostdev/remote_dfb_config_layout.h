// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Remote-DFB config page layouts used by CrossNodeDFB and PersistentDFB.
//
// Shared prefix:
//   word[0]  is_sender (1) | is_receiver (0)
//   word[1]  num_receivers
//   word[2]  fifo_start_addr
//   word[3]  fifo_size (CrossNode: entry_size * num_entries; Persistent: ring bytes)
//
// CrossNode:
//   word[4]  fifo_ptr_checkpoint   // reserved / ignored; ctor resets ptrs to word[2]
//   word[5]  noc_xy_offset         // page-relative → word[8]
//   word[6]  pages_sent_offset     // page-relative
//   word[7]  pages_acked_offset    // page-relative
//   + sender NOC XY table + L1-aligned sent/acked pairs
//   + receiver sender XY after header
//
// Persistent:
//   word[4]  fifo_ptr_checkpoint   // sender wr / receiver rd; commit stores; ctor loads
//   word[5]  applied_entry_size    // epoch + last successful resize
//   word[6]  noc_xy_offset         // page-relative → word[9]
//   word[7]  pages_sent_offset
//   word[8]  pages_acked_offset
//   + sender NOC XY table + L1-aligned sent/acked pairs
//   + receiver sender XY after header

inline constexpr uint32_t REMOTE_DFB_CFG_IS_SENDER = 0;
inline constexpr uint32_t REMOTE_DFB_CFG_NUM_RECEIVERS = 1;
inline constexpr uint32_t REMOTE_DFB_CFG_FIFO_START = 2;
inline constexpr uint32_t REMOTE_DFB_CFG_FIFO_SIZE = 3;

// --- CrossNodeDFB header ---
inline constexpr uint32_t CROSS_NODE_DFB_CONFIG_HEADER_WORDS = 8;
inline constexpr uint32_t CROSS_NODE_DFB_CFG_FIFO_PTR_CHECKPOINT = 4;
inline constexpr uint32_t CROSS_NODE_DFB_CFG_NOC_XY_OFFSET = 5;
inline constexpr uint32_t CROSS_NODE_DFB_CFG_PAGES_SENT_OFFSET = 6;
inline constexpr uint32_t CROSS_NODE_DFB_CFG_PAGES_ACKED_OFFSET = 7;

inline constexpr uint32_t cross_node_dfb_noc_xy_byte_offset() {
    return CROSS_NODE_DFB_CONFIG_HEADER_WORDS * static_cast<uint32_t>(sizeof(uint32_t));
}

// --- PersistentDFB header ---
inline constexpr uint32_t PERSISTENT_DFB_CONFIG_HEADER_WORDS = 9;
inline constexpr uint32_t PERSISTENT_DFB_CFG_FIFO_PTR_CHECKPOINT = 4;
inline constexpr uint32_t PERSISTENT_DFB_CFG_APPLIED_ENTRY_SIZE = 5;
inline constexpr uint32_t PERSISTENT_DFB_CFG_NOC_XY_OFFSET = 6;
inline constexpr uint32_t PERSISTENT_DFB_CFG_PAGES_SENT_OFFSET = 7;
inline constexpr uint32_t PERSISTENT_DFB_CFG_PAGES_ACKED_OFFSET = 8;

inline constexpr uint32_t persistent_dfb_noc_xy_byte_offset() {
    return PERSISTENT_DFB_CONFIG_HEADER_WORDS * static_cast<uint32_t>(sizeof(uint32_t));
}
