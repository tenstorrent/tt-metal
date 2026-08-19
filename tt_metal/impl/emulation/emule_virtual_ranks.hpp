// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

namespace tt::tt_metal::distributed {
class MeshDevice;
}

// In-process virtual ranks: K host "ranks" as threads of one process, so blaze reaches its K>1 code
// path (and therefore real d2d sockets) with no MPI. See tt-emule docs/multi-rank-emulation.md §8.

namespace tt::tt_metal::emule {

// Install a world of `k` virtual ranks as the current DistributedContext. k <= 1 uninstalls.
// Call before anything reads the world size; blaze latches K when it builds its pipeline.
void install_virtual_ranks(uint32_t k);

// Bind the CALLING THREAD to a virtual rank. Each of the k threads must call this once, with a
// distinct rank, before it touches ttnn — rank() is answered per thread.
void set_current_virtual_rank(uint32_t rank);

// The calling thread's virtual rank. A helper thread spawned to do work ON BEHALF of a rank must
// adopt it via set_current_virtual_rank: the rank is thread-local, so a fresh std::thread would
// otherwise speak as rank 0 and its collectives would address the wrong peer.
uint32_t current_virtual_rank();

// Abandon every rank parked in a collective or a point-to-point wait, making each throw. A rank
// whose thread dies must call this: an MPI rank takes the job down with it, but a dead thread is
// invisible and its peers would block until the test timeout, hiding the real exception.
void fault_virtual_ranks();

// Number of IN-PROCESS virtual ranks, or 1 when they are not installed (including under MPI, where
// the world is K but each rank is its own process). Callers use this to decide whether process-wide
// state has to be partitioned by rank by hand.
uint32_t virtual_rank_count();

// Publish the submesh a rank drives. A rank-addressed socket resolves its PEER's submesh-local
// coordinates through that peer's host-rank binding, but in one process the ranks share a host, so
// every submesh would fold onto the same chips. Held weakly: the mesh outlives its use here only by
// accident, and pinning it deadlocks a fixture teardown.
void register_virtual_rank_mesh(uint32_t rank, const std::shared_ptr<distributed::MeshDevice>& mesh);

// The submesh `rank` published, or null (no in-process ranks, never registered, already closed).
std::shared_ptr<distributed::MeshDevice> virtual_rank_mesh(uint32_t rank);

}  // namespace tt::tt_metal::emule
