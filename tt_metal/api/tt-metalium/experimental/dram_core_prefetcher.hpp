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
    uint32_t dram_core_k_block_w_tiles = 1;
};

// Launch the DRAM-core prefetcher on `mesh_device`. The prefetcher streams
// each tensor in `input_tensors` from its DRAM bank into the receivers
// configured in `gcb` for `config.num_layers` iterations. Non-blocking — the
// host thread returns immediately and is free to enqueue consumer programs
// (matmuls) that read from the GCB while the prefetcher kernel runs.
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
