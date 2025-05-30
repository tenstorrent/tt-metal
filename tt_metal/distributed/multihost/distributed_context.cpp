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

#if defined(OPEN_MPI)
using ContextImpl = MPIContext;
#else
using ContextImpl = SingleHostContext;
#endif

/* -------------------- factory for generic interface --------------------- */
void DistributedContext::create(int argc, char** argv) { ContextImpl::create(argc, argv); }

const ContextPtr& DistributedContext::get_current_world() { return ContextImpl::get_current_world(); }

void DistributedContext::set_current_world(const ContextPtr& ctx) { ContextImpl::set_current_world(ctx); }
}  // namespace tt::tt_metal::distributed::multihost
