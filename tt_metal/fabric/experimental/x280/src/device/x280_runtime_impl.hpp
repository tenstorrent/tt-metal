// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "x280/device/x280_runtime.hpp"
#include "x280_chip_impl.hpp"

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace x280::detail {

class TileImpl;

class RuntimeImpl : public std::enable_shared_from_this<RuntimeImpl> {
public:
    RuntimeImpl(
        std::weak_ptr<tt::tt_metal::distributed::MeshDevice> mesh,
        std::vector<std::shared_ptr<ChipImpl>> chips,
        LimShape lim_shape,
        LimLayout lim_layout = LimLayout::Legacy);

    // Sweeps every still-live session before the tiles it restores are
    // destroyed.
    ~RuntimeImpl();

    bool mesh_live() const;
    /** Locks the borrowed mesh lifetime; throws if it was dropped or closed. */
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh() const;
    bool owns_mesh(const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh) const;
    void require_mesh_live(const char* operation, const TileIdentity* tile = nullptr) const;
    std::vector<TileId> tile_ids() const;
    LimShape lim_shape() const { return lim_shape_; }
    LimLayout lim_layout() const { return lim_layout_; }
    std::shared_ptr<ChipImpl> chip(int chip_id) const;
    X280Tile& tile(const TileId& id);
    TileImpl& tile_impl(const TileId& id);

    const std::vector<std::shared_ptr<ChipImpl>>& chips() const { return chips_; }

    // These acquire their own locks; a caller holds neither the chip lock nor
    // sessions_mutex_ across them.
    //
    // enqueue takes the chip lock for the whole submission and sessions_mutex_
    // inside it to register. cleanup takes sessions_mutex_ to claim the session,
    // releases it and only then restores, because the restoration polls the
    // tile, and re-takes it afterwards to erase. sessions, session and
    // session_live take sessions_mutex_ alone and touch no hardware.
    //
    // An exclusive enqueue is refused while any session is registered or being
    // submitted anywhere in this runtime, and while its own session is live
    // every other enqueue is refused Busy.
    EnqueueResult enqueue(X280Tile& tile, const ActiveRunRequest& request, bool exclusive = false);
    std::vector<SessionId> sessions() const;
    X280ActiveSession& session(SessionId id);
    bool session_live(SessionId id) const;
    CleanupResult cleanup(SessionId id, std::chrono::milliseconds timeout);

private:
    // Takes sessions_mutex_, which is what keeps two callers minting on
    // different tiles from drawing the same value.
    SessionId mint_session_id(const TileKey& tile);

    std::weak_ptr<tt::tt_metal::distributed::MeshDevice> mesh_;
    std::vector<std::shared_ptr<ChipImpl>> chips_;
    LimShape lim_shape_;
    LimLayout lim_layout_ = LimLayout::Legacy;
    // Declared after chips so tiles are destroyed first.
    std::vector<std::unique_ptr<X280Tile>> tiles_;

    // Keyed by tile, so this map is itself the session-to-tile mapping and one
    // execution per tile is structural. A session stays here for its whole life
    // and is erased only once its restoration has finished, so no
    // active-protocol traffic is ever in flight for a tile absent from this map.
    //
    // sessions_mutex_ guards the map, ActiveSessionImpl's restoring_ flag,
    // next_session_id_, submitting_ and exclusive_, and is never held across
    // device I/O. Registration in enqueue() takes it under the chip lock, so
    // chip then sessions is the one permitted nesting; taking the chip lock
    // under this one would deadlock against it.
    std::map<TileKey, std::unique_ptr<X280ActiveSession>> sessions_;
    mutable std::mutex sessions_mutex_;

    // Only distinguishes successive executions on one tile, which the tile key
    // cannot.
    uint64_t next_session_id_ = 1;

    // Enqueues admitted and not yet returned, and the tile holding an exclusive
    // admission until its session is reaped.
    size_t submitting_ = 0;
    std::optional<TileKey> exclusive_;
};

}  // namespace x280::detail
