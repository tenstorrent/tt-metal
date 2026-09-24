// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A socket built inside a world larger than itself. Ranks that are neither
// endpoint allocate nothing and never call the socket's barriers, so anything
// the socket does collectively over the full context deadlocks the endpoints
// against a rank that is not coming.

#include <gtest/gtest.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/sockets/host_mesh_socket.hpp>
#include <tt-metalium/host_api.hpp>

#include "host_socket_test_utils.hpp"

#include <chrono>
#include <thread>
#include <vector>

namespace tt::tt_metal::distributed::host_socket_test {
namespace {

constexpr uint32_t kMrPageSize = 2048;
constexpr uint32_t kMrNumPages = 16;
constexpr uint32_t kMrFifoSize = kMrPageSize * kMrNumPages;

// Endpoints are the first and last rank, so every rank in between is a
// non-participant and the two endpoints still land on different hosts.
TEST(HostSocketMultiRankTest, NonParticipantsAreNotWaitedOn) {
    const auto& context = multihost::DistributedContext::get_current_world();
    const int size = *context->size();
    if (size < 3) {
        GTEST_SKIP() << "needs a world larger than the socket's two endpoints (got " << size << ")";
    }

    const multihost::Rank sender{0};
    const multihost::Rank receiver{size - 1};
    const int rank = *context->rank();
    const bool participates = rank == *sender || rank == *receiver;

    // Every rank opens a device, including the non-participant. Mesh creation is
    // collective over the world, so letting only the endpoints call it leaves
    // them waiting on a rank that never arrives -- and worse, the absent rank's
    // next collective silently matches one of theirs, since MPI pairs calls by
    // order rather than by call site.
    auto mesh_device = MeshDevice::create_unit_mesh(0);
    if (!host_sockets_supported(mesh_device)) {
        GTEST_SKIP() << "pinned host memory cannot be mapped to the NOC; enable vIOMMU";
    }

    if (participates) {
        const CoreCoord core{0, 0};
        SocketConnection connection{
            MeshCoreCoord(MeshCoordinate(0, 0), core), MeshCoreCoord(MeshCoordinate(0, 0), core)};
        SocketMemoryConfig mem_config(BufferType::L1, kMrFifoSize);
        SocketConfig socket_config({connection}, mem_config, sender, receiver, context);

        HostMeshSocket::TransportConfig transport;
        transport.page_size = kMrPageSize;

        // Construction is what the full-context barrier used to hang in. The
        // socket is torn down immediately: reaching this line is the assertion.
        HostMeshSocket socket(mesh_device, socket_config, transport);
        EXPECT_EQ(socket.get_config().sender_rank, sender);
    }

    // Every rank, endpoint or not, must reach this. A regression strands the
    // endpoints inside the socket and this never completes on any rank, so the
    // launcher's timeout is what reports it.
    context->barrier();
    SUCCEED() << "rank " << rank << " of " << size << (participates ? " (endpoint)" : " (non-participant)");
}

}  // namespace
}  // namespace tt::tt_metal::distributed::host_socket_test
