// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Experimental Start / Queue / Stop lifecycle API for the queueable Tensor
// prefetcher (DRISC). A long-running DRISC kernel on every DRAM sender core
// reads requests off a per-core H2D socket; each request identifies a
// DRAM-sender GlobalCircularBuffer and a per-tensor work list. The host can
// keep queueing new requests after Start returns; per-GCB ring-buffer state
// (fifo_wr_ptr, pages_sent) is preserved in DRISC L1 so successive requests
// can target different GCBs and switch back and forth across requests.
//
// State (Program(s), sockets, host worker thread, held tensor refs) lives on
// MeshDeviceImpl. Only one prefetcher may be active per mesh device at a time.

#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace tt::tt_metal {

class MeshTensor;

namespace distributed {
class MeshDevice;
class MeshCoordinateRangeSet;
}  // namespace distributed

namespace experimental {

class GlobalCircularBuffer;

struct TensorPrefetcherConfig {
    // When true, drive each DRAM bank with two DRISC sender cores instead of one:
    // the free non-endpoint subchannel plus the bank's NOC1-endpoint subchannel
    // (both write on NOC0). The bank's receivers are split ceil/floor across the two
    // cores, adding a second DMA engine + NoC initiator per bank. Only supported for
    // the receiver-contiguous DRAM layout. The GlobalCircularBuffer must be created
    // with the matching `dual_senders_per_bank` flag so its sender cores agree with
    // the prefetcher's; a mismatch is rejected at QueueTensorPrefetcherRequest.
    bool dual_senders_per_bank = false;
};

// Returns true if the Tensor prefetcher is supported on `mesh_device`, i.e.
// programmable DRAM cores are available (Blackhole with
// TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES=1). When this returns false,
// StartTensorPrefetcher would TT_FATAL, so callers (e.g. tests) can use this
// to skip rather than fail.
bool IsTensorPrefetcherSupported(const distributed::MeshDevice& mesh_device);

// One prefetch work item: a weight tensor plus the number of K-blocks to split
// its K dimension into. `block_count` is used in place of the GCB ring size when
// dividing K (k_block_w_tiles = ceil(K_tiles / block_count)), so different
// tensors — within one request or across GCBs — can use different K-block
// counts. The consuming matmul must wait_front(block_count) per layer.
//
// Required `tensor` layout (validated at Queue time, when the request is
// serialized):
//   - DRAM-resident, TILE layout,
//   - width-sharded across all DRAM banks (one shard per bank; each shard holds
//     the full K dimension and `N / num_dram_banks` columns),
//   - tile-aligned in both K and N (shard_shape divisible by TILE_HEIGHT/WIDTH).
// The per-bank N columns are further split evenly across that bank's receivers,
// so `N_per_bank` (= N / num_dram_banks) must divide the GCB's receiver count:
// receiver r in bank b owns columns
// `[b*N_per_bank + r*N_per_recv, b*N_per_bank + (r+1)*N_per_recv)`, where
// N_per_recv = N_per_bank / num_receivers_per_sender.
struct TensorPrefetcherInput {
    std::reference_wrapper<const MeshTensor> tensor;
    uint32_t block_count = 0;
};

// Build per-device Programs (one DRISC kernel per DRAM sender core), allocate
// the per-(device, sender) H2D sockets, and spawn the host worker thread that
// drains the request queue. Returns immediately. No prefetch work is scheduled
// yet — kernels park on socket_wait_for_pages.
//
// Receiver count is owned by each GCB (read from the per-GCB sender state
// block on every request), so a single prefetcher can serve GCBs with
// different num_receivers values.
//
// Preconditions (TT_FATAL):
//   - No other prefetcher is currently active on this mesh device.
//   - DRAM programmable cores are available on this mesh
//     (TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES=1).
void StartTensorPrefetcher(distributed::MeshDevice& mesh_device, const TensorPrefetcherConfig& config);

// Queue one prefetch request. Non-blocking.
//
//   - `gcb` must be a DRAM-sender GlobalCircularBuffer constructed against the
//     same mesh device.
//   - `device_subset` defaults to the full mesh when std::nullopt. Devices
//     outside the subset do not process this request.
//   - `input_tensors` is the full, flattened list of weight tensors to prefetch
//     (at least one), each paired with the block_count to divide its K dimension
//     into. They are streamed to the receivers in list order; pass distinct
//     tensors for distinct model layers, or repeat a tensor to replay it. Tensors
//     that share a layout are deduplicated on the wire, and a list that overflows
//     one request page is transparently split across pages.
//   - Per-GCB ring-buffer state is preserved across requests, so successive
//     Queue calls against the same GCB resume where the previous call left off.
//   - `cq_id` is the command queue on which a trace may be recording. When that
//     CQ is mid trace-capture, the request is captured into the trace instead of
//     being sent immediately, and is (re)sent on every replay of that trace
//     (ReplayTrace / ttnn.execute_trace). When the CQ is not capturing, the
//     request is sent immediately. Defaults (std::nullopt) to the current/default
//     command queue.
//
// The caller is responsible for keeping the tensors in `input_tensors` and
// `gcb` alive until Stop returns.
void QueueTensorPrefetcherRequest(
    distributed::MeshDevice& mesh_device,
    const GlobalCircularBuffer& gcb,
    const std::optional<distributed::MeshCoordinateRangeSet>& device_subset,
    const std::vector<TensorPrefetcherInput>& input_tensors,
    std::optional<uint8_t> cq_id = std::nullopt);

// Fence the prefetcher against command queue `cq_id`: every prefetch request queued
// after this call waits until all work previously enqueued on `cq_id` has completed
// on device before it reads DRAM. Use this to guarantee that data written over
// `cq_id` (e.g. the EnqueueWriteBuffer that populates the weights) has landed before
// the prefetcher streams it.
//
// Call this synchronously on the host thread that issued the data writes — after
// those writes, and before the QueueTensorPrefetcherRequest that consumes them.
//
//   - `cq_id` selects the command queue to fence against.
//   - `device_subset` defaults to the full mesh when std::nullopt.
//
// Preconditions (TT_FATAL): a prefetcher is active on this mesh device.
void WaitForCqOnTensorPrefetcher(
    distributed::MeshDevice& mesh_device,
    uint8_t cq_id,
    const std::optional<distributed::MeshCoordinateRangeSet>& device_subset);

// Block until all previously queued requests have been delivered and the
// kernels have exited, then release the prefetcher's resources. No-op if no
// prefetcher is active.
void StopTensorPrefetcher(distributed::MeshDevice& mesh_device);

}  // namespace experimental
}  // namespace tt::tt_metal
