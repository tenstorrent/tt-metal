// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/advanced_options.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt_stl/strong_type.hpp>
#include <hostdevcommon/sem_scope.h>

namespace tt::tt_metal::experimental {

/**
 * @brief Host-side per-semaphore scope INTENT, baked into a SemScope by the host.
 *
 * AUTO = the host derives the effective SemScope from the semaphore's reach (who
 * binds it, on how many nodes). An off-node CONSUME binder is rejected under EVERY
 * scope (a guaranteed hang: down() spins on the consumer's local word). Beyond that:
 * forcing DM_LOCAL_CACHED is validated at build time (a contradiction is a host
 * FATAL); forcing EXTERNAL or LOCAL_NONATOMIC skips AUTO's SET-race FATAL -- the
 * escape for phase-separated init-then-write, which the census cannot see.
 * Host-only; not used on the device.
 */
enum class SemaphoreScope : uint8_t {
    AUTO = 0,
    LOCAL_NONATOMIC = 1,
    DM_LOCAL_CACHED = 2,
    EXTERNAL = 3,
};

// ============================================================================
//  SemaphoreSpec API
// ============================================================================
//
// A SemaphoreSpec is a descriptor for a Tenstorrent semaphore,
// which can be used for inter-kernel instance synchronization.
//
// INSTANCING: One SRAM ("L1") cell per node in the set of target_nodes.
//
// PLACEMENT: Specified directly via target_nodes. Unlike DFBs, semaphores are
//   remote resources for kernels. Placement cannot be inferred from kernel
//   bindings.
//
// BINDING SCOPE: Any kernel can bind to any semaphore in the ProgramSpec and
//   signal it (up() takes explicit coordinates for remote targets). Consumers
//   (down(), labeled CONSUME) must run on the semaphore's node -- the host
//   rejects off-node CONSUME binders at build time, under every scope.
//
// ============================================================================

// A name identifying a SemaphoreSpec within a ProgramSpec.
using SemaphoreSpecName = ttsl::StrongType<std::string, struct SemaphoreSpecNameTag>;

struct SemaphoreSpec {
    // Semaphore identifier: used to reference this Semaphore within the ProgramSpec
    SemaphoreSpecName unique_id;

    // Target nodes
    Nodes target_nodes;

    //////////////////////////////////////////////////////////////////////////////
    // Advanced options (see advanced_options.hpp)
    //////////////////////////////////////////////////////////////////////////////
    SemaphoreAdvancedOptions advanced_options;

    // Physical-path INTENT for this semaphore. AUTO lets the host derive it. An off-node
    // CONSUME binder is rejected under every scope (guaranteed hang). Forcing
    // DM_LOCAL_CACHED is additionally validated at build time (e.g. multi-node is a
    // FATAL). The host resolves this to a device SemScope that the kernel picks up via
    // CTAD (see noc_semaphore.h).
    SemaphoreScope scope = SemaphoreScope::AUTO;
};

}  // namespace tt::tt_metal::experimental
