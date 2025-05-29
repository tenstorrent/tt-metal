// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/tt-metalium/distributed_context.hpp"
#if defined(OPEN_MPI)
#include "mpi_distributed_context.hpp"
#else
#include "single_host_context.hpp"
#endif

namespace tt::tt_metal::distributed::multihost {
/* -------------------- factory for generic interface --------------------- */
void DistributedContext::create(int argc, char** argv) {
#if defined(OPEN_MPI)
    MPIContext::create(argc, argv);
#else
    SingleHostContext::create(argc, argv);
#endif
}

const ContextPtr& DistributedContext::get_current_world() {
#if defined(OPEN_MPI)
    return MPIContext::get_current_world();
#else
    return SingleHostContext::get_current_world();
#endif
}

void DistributedContext::set_current_world(const ContextPtr& ctx) {
#if defined(OPEN_MPI)
    MPIContext::set_current_world(ctx);
#else
    SingleHostContext::set_current_world(ctx);
#endif
}
}  // namespace tt::tt_metal::distributed::multihost
