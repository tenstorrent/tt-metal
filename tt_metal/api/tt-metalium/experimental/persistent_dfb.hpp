// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Experimental DRAM-sender extension to PersistentDFB: a durable remote dataflow buffer whose
// senders are programmable DRAM cores (Blackhole DRISCs) rather than worker cores. This is the
// delivery target the Tensor prefetcher streams into as an alternative to a DRAM-sender
// GlobalCircularBuffer.
//
// Consumers are unchanged from an ordinary PersistentDFB: a consumer program calls
// AttachPersistentDFB on the receiver cores and its kernels use the device-side
// experimental::PersistentDFB (wait_front / get_read_ptr / pop_front). Only the producer side
// differs, and it is owned by the prefetcher.
//
// The functions here are free functions taking a const PersistentDFB& so that ttnn (which does
// not include tt_metal/impl headers) can reach the DRAM-sender properties.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/global_circular_buffer.hpp>

namespace tt::tt_metal {

class Program;

namespace distributed {
class MeshDevice;
}  // namespace distributed

namespace experimental {

class PersistentDFB;

// Construct a PersistentDFB whose senders are programmable DRAM cores identified by DRAM bank id,
// sized to hold `num_entries` entries of `entry_size` bytes per receiver.
//
// Sender placement, the dual-sender receiver split, and slab numbering are identical to
// CreateGlobalCircularBufferForTensorPrefetcher -- the two share build_dram_sender_mapping -- so a
// tensor laid out for one transport is laid out for the other. See that function for what
// `support_multi_receiver_shards` promises about the source layout.
//
// `entry_size` is the per-receiver push granularity and must equal the streamed tensor's
// per-receiver page size: this transport does not resize mid-flight, so every queued tensor must
// agree with it (enforced when the request is queued).
//
// Returned by shared_ptr because PersistentDFB is neither copyable nor movable (it owns the durable
// ring for its lifetime) while callers -- op attributes, Python bindings -- need a holder. Keep it
// alive for as long as any program Attaches or uses it; destroying it frees the ring and config.
//
// MeshDevice-only: the DRISC L1 arena backing the sender config pages lives on MeshDeviceImpl.
std::shared_ptr<PersistentDFB> CreatePersistentDFBForTensorPrefetcher(
    distributed::MeshDevice& mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t entry_size,
    uint32_t num_entries,
    BufferType buffer_type = BufferType::L1,
    bool support_multi_receiver_shards = false);

// Sender domain of a PersistentDFB: SenderCoreType::Worker for one built by the ordinary
// CreatePersistentDFB, SenderCoreType::Dram for one from CreatePersistentDFBForTensorPrefetcher.
// Reuses the GlobalCircularBuffer enum -- it names the same distinction.
SenderCoreType persistent_dfb_sender_core_type(const PersistentDFB& persistent_dfb);

// Geometry and placement, readable without naming PersistentDFB's definition.
uint32_t persistent_dfb_entry_size(const PersistentDFB& persistent_dfb);
uint32_t persistent_dfb_num_entries(const PersistentDFB& persistent_dfb);
uint32_t persistent_dfb_ring_size(const PersistentDFB& persistent_dfb);
uint32_t persistent_dfb_buffer_address(const PersistentDFB& persistent_dfb);
const CoreRangeSet& persistent_dfb_receiver_cores(const PersistentDFB& persistent_dfb);
const CoreRangeSet& persistent_dfb_sender_cores(const PersistentDFB& persistent_dfb);
const std::vector<std::pair<CoreCoord, CoreRangeSet>>& persistent_dfb_sender_receiver_core_mapping(
    const PersistentDFB& persistent_dfb);

// Attach a PersistentDFB to `program` on `cores`, returning the program-local slot id kernels
// name. Re-declared here (against the forward declaration above) so a consumer op can Attach
// without including the impl header. Same function as the one in
// impl/dataflow_buffer/persistent_dfb.hpp, which carries the default for entry_size_override.
uint8_t AttachPersistentDFB(
    Program& program,
    PersistentDFB& persistent_dfb,
    const CoreRangeSet& cores,
    std::optional<uint32_t> entry_size_override);

// The impl-internal DRAM-sender L1-layout accessors (sender-state base / receiver coords / slab
// indices) are consumed only inside tt_metal/ and live in
// tt_metal/impl/buffers/persistent_dfb_dram_sender_internal.hpp.

}  // namespace experimental
}  // namespace tt::tt_metal
