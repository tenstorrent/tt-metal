// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/host_transport/host_transport.hpp"

#ifdef TT_METAL_HOST_TRANSPORT_MPI
#include "tt_metal/distributed/host_transport/mpi_transport.hpp"
#endif
#ifdef TT_METAL_HOST_TRANSPORT_RDMA
#include "tt_metal/distributed/host_transport/rdma_link.hpp"
#include "tt_metal/distributed/host_transport/rdma_transport.hpp"
#endif

#include <tt_stl/assert.hpp>

namespace tt::tt_metal::distributed::host_transport {

bool host_transport_available(TransportKind kind) {
    switch (kind) {
        case TransportKind::Mpi:
#ifdef TT_METAL_HOST_TRANSPORT_MPI
            return true;
#else
            return false;
#endif
        case TransportKind::Rdma:
#ifdef TT_METAL_HOST_TRANSPORT_RDMA
            try {
                RdmaContext probe;
                return true;
            } catch (const std::exception&) {
                return false;
            }
#else
            return false;
#endif
    }
    return false;
}

std::unique_ptr<HostTransport> make_host_transport(const TransportParams& params) {
    switch (params.kind) {
        case TransportKind::Mpi:
#ifdef TT_METAL_HOST_TRANSPORT_MPI
            return std::make_unique<MpiTransport>(params);
#else
            TT_THROW("TransportKind::Mpi needs an MPI build (USE_MPI), which this build does not have.");
#endif
        case TransportKind::Rdma:
#ifdef TT_METAL_HOST_TRANSPORT_RDMA
            return std::make_unique<RdmaTransport>(params);
#else
            TT_THROW(
                "TransportKind::Rdma needs libibverbs, which this build does not have. Rebuild with rdma-core "
                "installed, or select TransportKind::Mpi (portable, much slower).");
#endif
    }
    TT_THROW("unknown TransportKind {}", static_cast<int>(params.kind));
}

}  // namespace tt::tt_metal::distributed::host_transport
