// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/tt-metalium/distributed_context.hpp"
#include <tt-logger/tt-logger.hpp>
#include <string>

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
void DistributedContext::create(int argc, char** argv) {
#if defined(OPEN_MPI)
    log_info(tt::LogFabric, "DIAG DistributedContext::create: compiled with OPEN_MPI, using MPIContext");
#else
    log_info(tt::LogFabric, "DIAG DistributedContext::create: compiled WITHOUT OPEN_MPI, using SingleHostContext");
#endif
    ContextImpl::create(argc, argv);
}

const ContextPtr& DistributedContext::get_current_world() {
    const auto& ctx = ContextImpl::get_current_world();
    static bool logged_once = false;
    if (!logged_once) {
        logged_once = true;
#if defined(OPEN_MPI)
        log_info(tt::LogFabric, "DIAG DistributedContext::get_current_world: OPEN_MPI build, impl=MPIContext, rank={}, size={}", *ctx->rank(), *ctx->size());
#else
        log_info(tt::LogFabric, "DIAG DistributedContext::get_current_world: non-MPI build, impl=SingleHostContext, rank={}, size={}", *ctx->rank(), *ctx->size());
#endif
    }
    return ctx;
}

void DistributedContext::set_current_world(const ContextPtr& ctx) {
    log_info(tt::LogFabric, "DIAG DistributedContext::set_current_world called: ctx rank={}, size={}",
        ctx ? std::to_string(*ctx->rank()) : "NULL",
        ctx ? std::to_string(*ctx->size()) : "NULL");
    ContextImpl::set_current_world(ctx);
}

bool DistributedContext::is_initialized() {
    bool init = ContextImpl::is_initialized();
    static bool logged_once = false;
    if (!logged_once) {
        logged_once = true;
        log_info(tt::LogFabric, "DIAG DistributedContext::is_initialized: result={}", init);
    }
    return init;
}

DistributedContextId DistributedContext::id() const { return id_; }

/* -------------------- DistributedContext ID generation --------------------- */
DistributedContextId DistributedContext::generate_unique_id() {
    static std::size_t next_id = 0;
    return DistributedContextId(next_id++);
}

}  // namespace tt::tt_metal::distributed::multihost
