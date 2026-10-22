// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/host_transport/host_transport.hpp"

#include "tt_metal/distributed/host_transport/loopback_transport.hpp"
#ifdef TT_METAL_HOST_TRANSPORT_MPI
#include "tt_metal/distributed/host_transport/mpi_transport.hpp"
#endif

#include <tt-metalium/distributed_context.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::distributed::host_transport {

bool host_transport_available() {
#ifdef TT_METAL_HOST_TRANSPORT_MPI
    return true;
#else
    return false;
#endif
}

std::unique_ptr<HostTransport> make_host_transport(const TransportParams& params) {
    // A peer in this process needs no network.
    if (params.context == nullptr || params.peer_rank == *params.context->rank()) {
        return std::make_unique<LoopbackTransport>(params);
    }
#ifdef TT_METAL_HOST_TRANSPORT_MPI
    return std::make_unique<MpiTransport>(params);
#else
    TT_THROW("The host socket needs an MPI build (ENABLE_DISTRIBUTED), which this build does not have.");
#endif
}

}  // namespace tt::tt_metal::distributed::host_transport
