// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Experimental Start/Stop lifecycle API for the DRAM-core (DRISC) prefetcher.
// The prefetcher streams tensor shards from DRAM into a receiver ring via a
// DRAM-sender GlobalCircularBuffer; only one prefetcher may be active on a
// mesh device at a time. State (in-flight Program, runtime args, lifetime
// references to inputs and the GCB) lives on MeshDeviceImpl.

#pragma once

#include <cstdint>
#include <vector>

namespace tt::tt_metal {

class MeshTensor;

namespace distributed {
class MeshDevice;
}  // namespace distributed

namespace experimental {

class GlobalCircularBuffer;

struct DramCorePrefetcherConfig {
    uint32_t num_layers = 1;
    bool enable_performance_mode = false;
};

// Launch the DRAM-core prefetcher on `mesh_device`. The prefetcher streams
// each tensor in `input_tensors` from its DRAM bank into the receivers
// configured in `gcb` for `config.num_layers` iterations. Non-blocking — the
// host thread returns immediately and is free to enqueue consumer programs
// (matmuls) that read from the GCB while the prefetcher kernel runs.
//
// Input tensor layout:
//   Each data tensor in `input_tensors` (all entries except the last; the last
//   is the addrs tensor, kept for op-contract parity with the worker-core
//   path and unused on this path) must be:
//     - DRAM-resident, TILE layout,
//     - width-sharded across all DRAM banks (one shard per bank; each shard
//       holds the full K dimension and `N / num_dram_banks` columns),
//     - tile-aligned in both K and N (validated at Start).
//
//   Interaction with `gcb`:
//     - Bank `b` is paired with the receiver CoreRangeSet at index `b` in
//       the GCB's sender_receiver_core_mapping.
//     - Per (layer, tensor), each receiver is pushed `num_blocks = num_senders
//       * num_receivers_per_sender` pages, one per K-block (= ceil(K_tiles /
//       num_blocks) tile-rows).
//     - The N-stripe a given receiver sees is its per-bank slice:
//       receiver r within bank b owns columns
//       `[b * N_per_bank + r * N_per_recv, b * N_per_bank + (r+1) * N_per_recv)`,
//       where N_per_bank = N / num_dram_banks and
//       N_per_recv = N_per_bank / num_receivers_per_sender.
//
// Preconditions (TT_FATAL on violation):
//   - sender_core_type(gcb) == SenderCoreType::Dram.
//   - No other DRAM-core prefetcher is currently active on this mesh device
//     (single-prefetcher-at-a-time invariant).
//   - All MeshTensors live on `mesh_device` and outlive the Stop call.
void StartDramCorePrefetcher(
    distributed::MeshDevice* mesh_device,
    const std::vector<const MeshTensor*>& input_tensors,
    const GlobalCircularBuffer& gcb,
    const DramCorePrefetcherConfig& config);

// Block until the active DRAM-core prefetcher finishes the natural exit of
// its `num_layers` loop, then tear down the held Program and release the
// resources it held. No-op if no prefetcher is active.
//
// Callers invoke Stop after enqueuing all the consuming matmuls — Stop is
// what drains the pipeline.
void StopDramCorePrefetcher(distributed::MeshDevice* mesh_device);

}  // namespace experimental
}  // namespace tt::tt_metal
