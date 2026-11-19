// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/tt-metalium/distributed_context.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string_view>
#include <typeinfo>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "single_host_context.hpp"
#if defined(OPEN_MPI)
#include "mpi_distributed_context.hpp"
#endif

namespace tt::tt_metal::distributed::multihost {

namespace {

#if defined(OPEN_MPI)
// Set for every rank by mpirun/prterun (OMPI_COMM_WORLD_SIZE) and by any PMIx launcher, including tt-run (which wraps
// mpirun) and srun --mpi=pmix (PMIX_RANK).
constexpr std::array<const char*, 2> kMpiLauncherEnvVars = {"OMPI_COMM_WORLD_SIZE", "PMIX_RANK"};

// Starting MPI outside an MPI job makes OpenMPI run as a singleton, which leaves a shared-memory segment in /dev/shm
// and a session directory in TMPDIR behind on every run. TT_METAL_MPI_INIT=always forces it anyway.
bool decide_use_mpi() {
    const char* mode_env = std::getenv("TT_METAL_MPI_INIT");
    const std::string_view mode = mode_env != nullptr ? mode_env : "auto";
    TT_FATAL(
        mode == "auto" || mode == "always" || mode == "never",
        "TT_METAL_MPI_INIT must be one of auto, always, never; got '{}'",
        mode);
    if (mode != "auto") {
        return mode == "always";
    }
    int mpi_already_initialized = 0;
    MPI_Initialized(&mpi_already_initialized);
    const bool launched_by_mpi =
        std::any_of(kMpiLauncherEnvVars.begin(), kMpiLauncherEnvVars.end(), [](const char* var) {
            return std::getenv(var) != nullptr;
        });
    if (!launched_by_mpi && mpi_already_initialized == 0) {
        log_debug(tt::LogMetal, "No MPI launcher detected; using the single-host distributed context");
    }
    return launched_by_mpi || mpi_already_initialized != 0;
}

bool use_mpi() {
    static const bool use = decide_use_mpi();
    return use;
}
#endif

}  // namespace

/* -------------------- factory for generic interface --------------------- */
void DistributedContext::create(int argc, char** argv) {
#if defined(OPEN_MPI)
    if (use_mpi()) {
        MPIContext::create(argc, argv);
        return;
    }
#endif
    SingleHostContext::create(argc, argv);
}

const ContextPtr& DistributedContext::get_current_world() {
    if (!current_world_) {
        create(0, nullptr);
    }
    return current_world_;
}

ContextPtr DistributedContext::get_world_context() { return get_current_world()->job_world(); }

void DistributedContext::set_current_world(const ContextPtr& ctx) {
    TT_FATAL(ctx != nullptr, "set_current_world: context is nullptr");
    if (current_world_) {
        const DistributedContext& incoming = *ctx;
        const DistributedContext& active = *current_world_;
        TT_FATAL(
            typeid(incoming) == typeid(active), "set_current_world: context type does not match the active backend");
    }
    current_world_ = ctx;
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
    if (current_world_) {
        return current_world_->is_runtime_initialized();
    }
    return false;
}

DistributedContextId DistributedContext::id() const { return id_; }

/* -------------------- DistributedContext ID generation --------------------- */
DistributedContextId DistributedContext::generate_unique_id() {
    static std::size_t next_id = 0;
    return DistributedContextId(next_id++);
}

}  // namespace tt::tt_metal::distributed::multihost
