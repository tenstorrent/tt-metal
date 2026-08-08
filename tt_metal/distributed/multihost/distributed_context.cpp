// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/tt-metalium/distributed_context.hpp"

#include <array>
#include <cstdlib>
#include <string_view>

#include <tt_stl/assert.hpp>

#if defined(OPEN_MPI)
#include "mpi_distributed_context.hpp"
#else
#include "single_host_context.hpp"
#endif

#if defined(TT_DISTRIBUTED_ZMQ)
#include "zmq_distributed_context.hpp"
#endif

namespace tt::tt_metal::distributed::multihost {

#if defined(OPEN_MPI)
using ContextImpl = MPIContext;
#else
using ContextImpl = SingleHostContext;
#endif

#if defined(TT_DISTRIBUTED_ZMQ)
namespace {

// Optional runtime override: when the ZMQ backend is compiled in
// (TT_DISTRIBUTED_ZMQ) and TT_DISTRIBUTED_BACKEND=zmq, use the ZMQ context
// instead of the compile-time default (MPI or single-host). Evaluated once and
// cached, since the choice must be stable for the lifetime of the process.
bool use_zmq_backend() {
    static const bool selected = [] {
        const char* b = std::getenv("TT_DISTRIBUTED_BACKEND");
        return b != nullptr && std::string_view(b) == "zmq";
    }();
    return selected;
}

}  // namespace
#endif

/* -------------------- factory for generic interface --------------------- */
void DistributedContext::create(int argc, char** argv) {
#if defined(TT_DISTRIBUTED_ZMQ)
    if (use_zmq_backend()) {
        ZmqContext::create(argc, argv);
        return;
    }
#endif
    ContextImpl::create(argc, argv);
}

const ContextPtr& DistributedContext::get_current_world() {
#if defined(TT_DISTRIBUTED_ZMQ)
    if (use_zmq_backend()) {
        return ZmqContext::get_current_world();
    }
#endif
    return ContextImpl::get_current_world();
}

ContextPtr DistributedContext::get_world_context() {
#if defined(TT_DISTRIBUTED_ZMQ)
    if (use_zmq_backend()) {
        return ZmqContext::get_world_context();
    }
#endif
#if defined(OPEN_MPI)
    return MPIContext::get_world_context();
#else
    return ContextImpl::get_current_world();
#endif
}

void DistributedContext::set_current_world(const ContextPtr& ctx) {
#if defined(TT_DISTRIBUTED_ZMQ)
    if (use_zmq_backend()) {
        ZmqContext::set_current_world(ctx);
        return;
    }
#endif
    ContextImpl::set_current_world(ctx);
}

Size DistributedContext::subcontext_size(SubcontextId subcontext_id) const {
    TT_FATAL(*subcontext_id == 0, "subcontext_id {} invalid for default single-context layout", *subcontext_id);
    return size();
}

ttsl::Span<const int> DistributedContext::subcontext_sizes() const {
    static thread_local std::array<int, 1> backing;
    backing[0] = *size();
    return {backing.data(), backing.size()};
}

Rank DistributedContext::local_to_world_rank(SubcontextId subcontext_id, Rank local_rank) const {
    TT_FATAL(*subcontext_id == 0, "subcontext_id {} invalid for default single-context layout", *subcontext_id);
    TT_FATAL(
        *local_rank >= 0 && *local_rank < *size(),
        "local_rank {} out of range for sub-context 0 (size {})",
        *local_rank,
        *size());
    return local_rank;
}

bool DistributedContext::is_initialized() {
#if defined(TT_DISTRIBUTED_ZMQ)
    if (use_zmq_backend()) {
        return ZmqContext::is_initialized();
    }
#endif
    return ContextImpl::is_initialized();
}

DistributedContextId DistributedContext::id() const { return id_; }

/* -------------------- DistributedContext ID generation --------------------- */
DistributedContextId DistributedContext::generate_unique_id() {
    static std::size_t next_id = 0;
    return DistributedContextId(next_id++);
}

}  // namespace tt::tt_metal::distributed::multihost
