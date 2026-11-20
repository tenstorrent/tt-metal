// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device_types.hpp>
#include <tt-metalium/hal_types.hpp>

// EXPERIMENTAL: declarations in this header may change or be removed without notice.

namespace tt::tt_metal::distributed {
class MeshDevice;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::experimental {

/**
 * @brief Outcome of quiesce_for_soft_reset().
 */
enum class QuiesceStatus : uint8_t {
    // All dispatch, fabric (if active), and worker tiles reached QUIESCE_STATUS_QUIESCED,
    // all host background pollers stopped, and zero NOC / NOC2AXI traffic remains.
    Quiesced,

    // Timed out in Phase 1 waiting for Fast Dispatch cores to cut off fetch_q/exec_buf,
    // drain in-flight workers, and park. Indicates a hung/deadlocked kernel or dispatcher.
    // AI-IP soft-reset is UNSAFE.
    TimedOutInDispatch,

    // Timed out waiting for Fabric muxes / ERISC routers to terminate after Phase 1.
    // AI-IP soft-reset is UNSAFE.
    TimedOutInFabric,

    // Timed out in Phase 2 waiting for worker or idle dispatch tiles to drain TRIDs and park
    // (e.g., hung kernel in Slow Dispatch mode). AI-IP soft-reset is UNSAFE.
    TimedOutInWorkers,
};

/**
 * @brief Diagnostic snapshot for a core that failed to reach QUIESCE_STATUS_QUIESCED before timeout.
 */
struct QuiesceCoreDiagnostic {
    ChipId chip_id = 0;
    CoreCoord virtual_core{0, 0};
    HalProgrammableCoreType core_type = HalProgrammableCoreType::TENSIX;
    uint32_t raw_status = 0;
    uint32_t control = 0;
    uint32_t op_started_count_cq0 = 0;
    uint32_t op_started_count_cq1 = 0;
};

/**
 * @brief Configuration options for quiesce_for_soft_reset().
 */
struct QuiesceOptions {
    // Total wall-clock timeout budget across all quiesce phases.
    // Pass std::nullopt only when an unbounded wait is explicitly desired (e.g., slow RTL debug).
    // Passing <= 0 ms is rejected via TT_FATAL.
    std::optional<std::chrono::milliseconds> timeout = std::chrono::milliseconds{5000};
};

/**
 * @brief Result returned by quiesce_for_soft_reset().
 */
struct QuiesceResult {
    QuiesceStatus status = QuiesceStatus::Quiesced;
    std::chrono::milliseconds elapsed{0};

    // Empty when ok() == true; populated with every unquiesced core on timeout.
    std::vector<QuiesceCoreDiagnostic> unquiesced_cores;

    bool ok() const noexcept { return status == QuiesceStatus::Quiesced; }
    explicit operator bool() const noexcept { return ok(); }
};

/**
 * @brief Gracefully quiesce and park all tiles on a Quasar MeshDevice in preparation for an AI-IP soft-reset.
 *
 * Preconditions (checked via TT_FATAL):
 * - `mesh_device.arch() == tt::ARCH::QUASAR` and target is not Mock or Emule.
 * - `mesh_device.is_initialized()` and owns all open devices in its context (`is_parent_mesh()` or a standalone
 *   unit mesh from `open_unit_mesh` / `create_unit_mesh`, not a carved submesh).
 * - `options.timeout`, if engaged, is strictly greater than 0 ms.
 * - Caller is not concurrently invoking Enqueue* / Finish / close() on `mesh_device` or its submeshes
 *   from another host thread.
 *
 * Postconditions:
 * - On `result.ok() == true`:
 *   - Every dispatch and worker tile on `mesh_device` has drained all 32 NOC TRIDs + FDS and parked all
 *     harts in `nop` (`QUIESCE_STATUS_QUIESCED`).
 *   - All host background threads (`FDMeshCommandQueue` completion reader, `WatcherServer`, `DPrintServer`,
 *     profiler threads) are stopped.
 *   - `mesh_device` is closed host-side (subsequent `mesh_device.close()` or `~MeshDevice()` is a safe no-op),
 *     and post-quiesce hardware MMIO writes in Metalium teardown are suppressed so zero `NOC2AXI` transactions
 *     occur after this call returns.
 * - On `result.ok() == false` (timeout due to hung/deadlocked workload):
 *   - AI-IP soft-reset MUST NOT be issued by the caller (NOC / `NOC2AXI` may still be active).
 *   - Host DMA/sysmem mappings remain pinned until process exit so any still-running kernel does not trigger
 *     SMMU / host UAF faults, and `mesh_device` is marked poisoned so destructors do not hang in `cq.terminate()`.
 */
[[nodiscard]] QuiesceResult quiesce_for_soft_reset(
    distributed::MeshDevice& mesh_device, const QuiesceOptions& options = {});

}  // namespace tt::tt_metal::experimental
