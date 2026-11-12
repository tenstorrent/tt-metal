// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "impl/buffers/tensor_prefetcher_manager.hpp"
#include "impl/buffers/tensor_prefetcher_transport.hpp"

namespace tt::tt_metal {

class Buffer;
class DriscL1Arena;
class Hal;
class IDevice;
class MeshTensor;
class Program;

namespace distributed {

class MeshDevice;

// Per-core carve of the DRISC L1 kernel working region (above the GCB zone), the same on every sender core: the
// backend's request buffers, which include the CQ signal slots, then the stage ring. noc_xy table, config block, and
// the RemoteSenderCBInterface live inside each GCB's sender state block, owned by the GCB rather than the prefetcher
// kernel.
struct KernelRegionCarve {
    uint32_t cq_signal_l1_addr = 0;
    // One L1 alignment per slot rather than packed -- see cq_signal_slot_stride_.
    uint32_t cq_signal_slot_stride = 0;
    uint32_t stage_ring_base = 0;
    uint32_t stage_ring_size = 0;
    // Empty when the carve fits the region; otherwise why it does not.
    std::string failure_reason;
};

// `carve_request_buffers` places the CQ signal slots and the `request_buffers` at the front of the region, from
// `kernel_region_base`, and sets where the stage ring starts. The stage ring takes the rest, capped at
// `stage_ring_cap_bytes` and aligned down to `stage_ring_alignment_bytes` (unset is 6 x the L1 alignment).
KernelRegionCarve carve_kernel_region(
    const Hal& hal,
    const DriscL1Arena& arena,
    const char* request_buffers,
    std::optional<uint32_t> stage_ring_cap_bytes,
    std::optional<uint32_t> stage_ring_alignment_bytes,
    const std::function<void(KernelRegionCarve&, uint32_t)>& carve_request_buffers);

// How to resolve a weight whose two candidate layouts describe the same bytes: one shard per bank,
// feeding that bank's single receiver. There K-row-major and receiver-contiguous read identically
// (n_per_recv == n_per_bank, one slab at offset 0), so the bytes cannot settle it.
//
// What still differs is what each sizer accepts. K-row-major *ceils* K into block_count blocks, so
// it takes a K the block count does not divide (the short last block over-reads inside the bank,
// which is that receiver's own data); receiver-contiguous requires exact division, because in
// general the over-read would cross into the next receiver's slab. K-row-major also gets a whole
// stage half per K-row where receiver-contiguous gets a third, its loop keeping 3 slots rather
// than 2. In exchange, receiver-contiguous is the loop with dynamic multi-block batching.
//
// So a GlobalCircularBuffer keeps the K-row-major reading it has always had, and PrefetcherPipe
// delivery -- which consumes receiver-contiguous only -- gets the other. (K-row-major's third fit
// rung, chunking a too-wide K-row across receivers, is not among the differences: its M is capped
// at num_receivers, so at one receiver per bank neither layout can chunk N.)
enum class LayoutTieBreak : uint8_t {
    PreferKRowMajor,
    PreferReceiverContiguous,
};

// Detection keys on how the weight was allocated, NOT the shard count: receiver-contiguous
// weights are created with an NdShardSpec (num_shards == ring_size), K-row-major weights use a
// legacy (WIDTH_SHARDED) shard spec. Counting shards is ambiguous when total_receivers ==
// num_banks (one receiver per bank, num_shards == num_banks == total_receivers): the count tie
// would route a recv-contig tensor down the K-row path, where compute_tensor_layout_krow_major
// calls shard_spec() and TT_FATALs because the buffer only has an NdShardSpec.
TensorPrefetcherLayoutMode detect_layout_mode(
    const MeshTensor& t, const Buffer& buf, uint32_t total_receivers, LayoutTieBreak tie_break);

// What differs between the Blackhole DRISC prefetcher and the Quasar CCE prefetcher: config resolution, the carve,
// the sender Programs and the transport that feeds them, and the checks on targets and requests. start(), worker_loop()
// and stop() reach the architecture only through this and that transport.
class PrefetcherSenderBackend {
public:
    // The Programs, one per device and not yet compiled or launched, and the transport that delivers their request
    // pages.
    struct PreparedSenders {
        std::vector<std::unique_ptr<Program>> programs;
        std::unique_ptr<RequestTransport> transport;
    };

    virtual ~PrefetcherSenderBackend() = default;

    // This architecture's part of tensor_prefetcher_preconditions, run once the shared checks pass.
    virtual TensorPrefetcherPreconditions preconditions(const MeshDevice& mesh_device) const = 0;

    // Validates `config` for this architecture and keeps what the rest of start() reads from it.
    virtual void resolve_config(
        const MeshDevice& mesh_device,
        const experimental::TensorPrefetcherConfig& config,
        const std::optional<QuasarTensorPrefetcherConfig>& quasar_config) = 0;

    // DRAM banks to provision senders on; empty provisions every bank.
    virtual std::vector<uint32_t> sender_banks() const = 0;

    // Sender roles per DRAM bank, in dram_sender_logical_cores order.
    virtual uint32_t senders_per_bank() const = 0;

    virtual KernelRegionCarve carve(const MeshDevice& mesh_device) const = 0;

    virtual PreparedSenders prepare_senders(
        MeshDevice& mesh_device,
        const std::vector<IDevice*>& devices,
        const std::vector<CoreCoord>& sender_logical_cores,
        const KernelRegionCarve& carve) const = 0;

    // Whether start() times the session for the benchmark tuner.
    virtual bool benchmark_enabled() const = 0;

    // Throws unless the senders can deliver to PrefetcherPipes.
    virtual void require_prefetcher_pipe_delivery(const MeshDevice& mesh_device) const = 0;

    // Throws unless the receivers in `mapping` can run while the provisioned senders do.
    virtual void require_receivers_off_senders(
        const MeshDevice& mesh_device,
        const std::vector<IDevice*>& devices,
        const std::vector<CoreCoord>& sender_logical_cores,
        const std::vector<std::pair<CoreCoord, CoreRangeSet>>& mapping) const = 0;

    // Request checks serialize_request_pages runs at three points: over every tensor before the streaming topology
    // guard, on each tensor before its layout is computed, and on each tensor after.
    virtual void validate_request_tensors(
        const MeshDevice& mesh_device, const std::vector<experimental::TensorPrefetcherInput>& tensors) const = 0;
    virtual void validate_tensor_before_layout(
        const MeshDevice& mesh_device,
        const experimental::TensorPrefetcherInput& input,
        uint32_t total_receivers,
        size_t tensor_idx) const = 0;
    virtual void validate_tensor_after_layout(
        const MeshDevice& mesh_device,
        const IDevice* reference_device,
        const MeshTensor& tensor,
        const TensorPrefetcherTensorLayout& layout,
        const std::vector<std::pair<CoreCoord, CoreRangeSet>>& mapping,
        const std::vector<uint32_t>& recv_index_base_per_sender,
        size_t tensor_idx) const = 0;

    // How a GlobalCircularBuffer target resolves a weight whose two layouts describe the same bytes.
    virtual LayoutTieBreak global_cb_layout_tie_break() const = 0;
};

}  // namespace distributed
}  // namespace tt::tt_metal
