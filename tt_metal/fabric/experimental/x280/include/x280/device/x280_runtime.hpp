// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "x280/device/translated_coord.hpp"
#include "x280/device/x280_active.hpp"
#include "x280/device/x280_tile.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// Coordinate callers include <tt-metalium/core_coord.hpp> and
// <umd/device/types/core_coordinates.hpp> for these Metal/UMD definitions.
namespace tt {
struct xy_pair;
enum class CoreType;
}  // namespace tt

namespace tt::tt_metal {
using CoreCoord = tt::xy_pair;
enum NOC : uint8_t;
}  // namespace tt::tt_metal

namespace tt::tt_metal::distributed {
class MeshDevice;
class MeshCoordinate;
}  // namespace tt::tt_metal::distributed

namespace x280 {

namespace detail {
class RuntimeImpl;
}

// TileIdentity is the stable public identifier. The shorter name keeps runtime
// lookup call sites focused on identity rather than topology details.
using TileId = TileIdentity;

// Reset release and recovery reads on an otherwise unclocked tile require this
// low chip-global PLL frequency.
inline constexpr uint32_t kSafePllMhz = 200;

// PLL4 ownership, which every clock-moving entry point below shares.
//
// One VCO serves all four L2CPU tiles, so a frequency change is chip-global
// even where the post-dividers are per-output. Three operations move it:
// open() applies kSafePllMhz to probe an unreadable reset-held tile and again
// before priming; a cold boot() applies it while the chip is quiescent, before
// releasing reset; set_pll() applies whatever the owner asks for.
//
// A second owner exists outside this process. CMFW takes PLL4 via its
// power-setting message, and tt-kmd sends one on every open() of
// /dev/tenstorrent/N (tt_cdev_open -> tenstorrent_set_aggregated_power_state).
// A message landing mid-slew can leave CMFW rejecting power states, after
// which the chip answers PCIe with a continuous NonFatal error stream until an
// ASIC reset -- enough to take a host down when the error recovery saturates
// its CPUs. Nothing here can serialise against another process, so this is a
// precondition on the caller and not a check any of the three can make: do not
// move this clock on a host where anything else holds the device open. A telemetry
// daemon counts.

enum class RuntimeOpenOutcome {
    Ready,
    PreparationRequired,
    ResetRequired,
    Failed,
};

enum class LimEccPrimingPolicy {
    // Prime unprepared reset-held hardware and return ResetRequired.
    AutoPrime,
    // Return PreparationRequired on unprimed hardware without increasing WayEnable.
    RequirePrepared,
};

struct RuntimeOpenConfig {
    LimEccPrimingPolicy priming_policy = LimEccPrimingPolicy::AutoPrime;
    // Applied only after all selected chips preflight successfully and only
    // while every affected L2CPU remains held in reset.
    LimShape lim_shape{};
    // Selected component memory layout profile. Default is Legacy.
    LimLayout lim_layout = LimLayout::Legacy;
};

const char* to_string(RuntimeOpenOutcome outcome);
bool preparation_required(RuntimeOpenOutcome outcome);
bool reset_required(RuntimeOpenOutcome outcome);

struct ChipOpenResult {
    int chip_id = -1;
    RuntimeOpenOutcome outcome = RuntimeOpenOutcome::Failed;
    std::string diagnostic;
};

struct RuntimeOpenResult;

struct MultiTileBootRequest {
    // Tiles to boot, all on one chip, each a member of tile_ids(). Duplicates
    // and foreign tiles are refused before anything is touched.
    std::vector<TileId> tiles;

    std::string idle_image_path;

    // Bounds the wait for each tile's harts to reach IDLE after the release.
    std::chrono::milliseconds boot_timeout{3000};

    // Bounds ARC telemetry confirmation of the safe reset-release clock.
    std::chrono::milliseconds clock_confirm_timeout{4000};

    // Reset release is one-way; batch boot requires explicit opt-in.
    bool allow_reset_release = false;
};

struct TileBootReport {
    TileId id;
    BootResult result;
};

struct MultiTileBootResult {
    // True once the reset-release write may have been issued; no further cold
    // boot is possible in that chip-reset lifetime.
    bool reset_released = false;

    // One entry per selected tile, in request order. Empty on early refusal.
    std::vector<TileBootReport> tiles;

    std::string diagnostic;

    bool ready() const {
        if (tiles.empty()) {
            return false;
        }
        for (const auto& report : tiles) {
            if (!report.result.ready()) {
                return false;
            }
        }
        return true;
    }
};

// The X280 stateful initialization boundary for one live caller-owned
// MeshDevice. A different mesh cannot bind until this mesh closes.
class X280Runtime {
public:
    // Binds the mesh, preflights every selected chip, and exposes the tile
    // inventory. Not read-only: an unreadable reset-held tile is clocked at
    // kSafePllMhz so it can be probed, and the AutoPrime policy commits
    // WayEnable, which an ASIC reset alone undoes. Both are subject to the PLL4
    // ownership rules above.
    static RuntimeOpenResult open(
        const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh, RuntimeOpenConfig config = {});

    X280Runtime(const X280Runtime&) = delete;
    X280Runtime& operator=(const X280Runtime&) = delete;
    X280Runtime(X280Runtime&&) noexcept;
    X280Runtime& operator=(X280Runtime&&) noexcept;

    // Destroying the last runtime value sweeps every still-live session
    // through the same bounded restoration cleanup() performs, using each
    // request's cleanup_timeout. A closed or dropped MeshDevice admits no
    // transport write, so such a session is abandoned with a logged warning.
    ~X280Runtime();

    // Enabled, unharvested tiles ordered by chip id and hardware tile index.
    std::vector<TileId> tile_ids() const;

    /** The translation point for every coordinate the host hands to device
     * code: resolves a per-core-type logical coordinate to the virtual (UMD
     * TRANSLATED) coordinate common to every chip of `scope`, for a consumer
     * addressing it on `noc`. Tensix, Ethernet and DRAM resolve through
     * Metal's mesh resolver; L2CPU through the SoC descriptors. Needs no open
     * runtime, so Metal-only callers use it too.
     *
     * Throws for a closed or empty scope, an unavailable core, an unsupported
     * type, coordinates differing across chips, a coordinate outside six bits,
     * or, on NoC 1, a translated Y < 2 with X != 8: firmware leaves X
     * untranslated there, so the pair names a mirrored tile on NoC 1
     * (BlackholeA0/NoC/Coordinates.md, "Coordinate Translation"). No device
     * I/O; the borrowed scope must stay live and unchanged through the call. */
    static TranslatedCoord virtual_core_from_logical_core(
        const tt::tt_metal::distributed::MeshDevice& scope,
        const tt::tt_metal::CoreCoord& logical_coord,
        const tt::CoreType& core_type,
        tt::tt_metal::NOC noc);

    /** As above, on the one chip at `chip` within `scope`, for a selection
     * that differs per chip (each chip's end of an Ethernet link). Also throws
     * for a coordinate outside scope or a remote chip there. */
    static TranslatedCoord virtual_core_from_logical_core(
        const tt::tt_metal::distributed::MeshDevice& scope,
        const tt::tt_metal::distributed::MeshCoordinate& chip,
        const tt::tt_metal::CoreCoord& logical_coord,
        const tt::CoreType& core_type,
        tt::tt_metal::NOC noc);

    /** Returns the selected tile's virtual coordinate on its chip, for a
     * consumer addressing it on `noc`. Throws for a closed mesh, an identity
     * outside this runtime's inventory, a missing mapping, or the range and
     * NoC 1 refusals of virtual_core_from_logical_core.
     * No device I/O; mesh close or reshape must not race this query. */
    TranslatedCoord virtual_core_from_tile(const TileId& id, tt::tt_metal::NOC noc) const;

    // Effective cache/LIM partition established before this runtime was exposed.
    LimShape lim_shape() const;

    // Returns the runtime-owned tile for this exact identity. The reference is
    // valid only while a runtime value for this live mesh remains alive.
    X280Tile& tile(const TileId& id);

    // Establishes and confirms the chip-global L2CPU clock, subject to the PLL4
    // ownership rules above. Unlike open() and boot(), this moves the clock
    // whatever the chip is doing: the owner coordinates the frequency across
    // tiles, and a tile already out of reset sees its frequency move under it.
    void set_pll(
        int chip_id, uint32_t mhz, std::chrono::milliseconds confirm_timeout = std::chrono::milliseconds{4000});

    // Prepares all selected tiles, then releases their resets in one write.
    // Requires every enabled tile reset-held under the chip lock; preparation
    // failure releases none. Already-running tiles are skipped.
    //
    // Leaves PLL4 at the safe release frequency. A post-release failure
    // requires an ASIC reset before another cold boot.
    MultiTileBootResult boot_tiles(const MultiTileBootRequest& request);

    // Submits one active image to a healthy, resident-idle tile of this
    // runtime. Performs preflight, session attachment, image load and readback,
    // per-hart record clearing, run/entry/args publication, and JUMP
    // publication in the caller's thread -- an immediate submission, not a
    // background FIFO. One execution per tile: a second call while one is live
    // returns Busy. Every tile returns Busy while an X280Tile::prepare_pinned_input
    // session is live.
    //
    // request.args_addr is validated for placement before the slot is reserved
    // and never for content; EnqueueOutcome::ArgumentsRejected leaves the tile
    // untouched and healthy.
    EnqueueResult enqueue(X280Tile& tile, const ActiveRunRequest& request);

    // Live executions this runtime owns, ordered by tile, and each id names the
    // tile it runs on. Includes a session cleanup() is still restoring: no
    // active-protocol traffic is ever in flight for a tile absent from this
    // list. No device I/O.
    std::vector<SessionId> sessions() const;

    // The runtime-owned session for this id. Throws when the id is unknown or
    // already reaped. The reference is valid only until that session is
    // cleaned up or the runtime is destroyed; a concurrent cleanup() of the same
    // id invalidates it, which is the caller's obligation to order, exactly as
    // it is for X280Tile&.
    X280ActiveSession& session(SessionId id);

    // Non-throwing existence query.
    bool session_live(SessionId id) const;

    // Bounded STOP / RETURN_ACK / resident re-arm, then reap. Single-shot: the
    // session no longer exists once this returns, so a later call returns
    // CleanupOutcome::UnknownSession, and a concurrent one that loses the claim
    // returns CleanupOutcome::AlreadyRestoring rather than driving the protocol
    // into the same four harts twice. A failed restoration leaves the evidence
    // in the returned result and in X280Tile::last_failure().
    CleanupResult cleanup(SessionId id, std::chrono::milliseconds timeout);

private:
    static RuntimeOpenResult open_impl(
        const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh, RuntimeOpenConfig config);

    explicit X280Runtime(std::shared_ptr<detail::RuntimeImpl> impl);
    std::shared_ptr<detail::RuntimeImpl> impl_;
};

struct RuntimeOpenResult {
    RuntimeOpenOutcome outcome = RuntimeOpenOutcome::Failed;

    // Present only when every selected chip is ready.
    std::optional<X280Runtime> runtime;
    std::vector<ChipOpenResult> chips;
    std::string diagnostic;

    bool ready() const { return runtime.has_value(); }
    // True when at least one selected chip needs operator preparation before
    // open can safely act on it.
    bool requires_preparation() const;
    // Broader than reset_required(outcome): a mixed failed/reset-required
    // multi-chip open still requires the caller to reset committed hardware.
    bool requires_reset() const;
};

}  // namespace x280
