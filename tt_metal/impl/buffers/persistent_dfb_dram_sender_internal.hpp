// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Impl-internal DRAM-sender accessors for PersistentDFB. These read back the DRISC L1 layout that
// the DRAM-sender PersistentDFB constructor stamps out; they are consumed only inside tt_metal/
// (the Tensor prefetcher manager and the DRAM-sender PersistentDFB tests), so they live here
// rather than on the public experimental surface in tt-metalium/experimental/persistent_dfb.hpp.
// That header keeps only what ttnn consumes.

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>

#include "impl/dataflow_buffer/persistent_dfb.hpp"

namespace tt::tt_metal::experimental {

// DRISC L1 base of this PersistentDFB's per-sender block: a PersistentDfbDramSenderState prefix
// followed by the sender's PersistentDFB config page (9-word header, receiver NOC XY table, and
// the per-receiver entries_sent/entries_acked pairs). Pre-written by the constructor on every
// (device, sender_core) at a uniform offset. The DRISC kernel reads the prefix for its slab base,
// then hands the config-page address to setup_persistent_dfb_interface.
//
// Layout: tt_metal/impl/buffers/persistent_dfb_dram_sender_state.hpp. Zero for worker-sender
// PersistentDFBs.
DeviceAddr persistent_dfb_sender_state_drisc_l1_base(const PersistentDFB& persistent_dfb);

// Physical worker NOC XY for each sender's receivers, in bank-local slab order. Empty for
// worker-sender PersistentDFBs.
const std::vector<std::vector<CoreCoord>>& persistent_dfb_receiver_coords_per_sender(
    const PersistentDFB& persistent_dfb);

// Per-sender bank-local slab indices: entry [s][r] is the slab index (recv_index_base + r) that
// sender s's local receiver r reads, in sender_receiver_core_mapping() order. Same contract (and
// same underlying helper) as the GlobalCircularBuffer accessor of this name, including its
// deliberate order-agnosticism: mapping a slab index to a consumer "ring position" depends on the
// tensor's shard distribution, which is the caller's concept. DRAM-sender PersistentDFBs only.
std::vector<std::vector<uint32_t>> persistent_dfb_receiver_slab_indices(const PersistentDFB& persistent_dfb);

}  // namespace tt::tt_metal::experimental
