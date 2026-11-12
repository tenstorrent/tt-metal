// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Coordination for one chip owned by one live MeshDevice runtime.
//
// Private to the target: this header names UMD types freely.

#pragma once

#include "x280_metal_transport.hpp"
#include "x280_pll.hpp"
#include "x280/device/x280_tile_state.hpp"

#include <chrono>
#include <memory>
#include <mutex>
#include <vector>

namespace x280::detail {

class ChipImpl {
public:
    ChipImpl(int chip_id, std::shared_ptr<X280MetalTransport> transport);

    // --- chip-wide serialisation -----------------------------------------

    // UMD locks individual register operations but nothing serialises a
    // sequence, and every chip-global sequence here is one: the L2CPU_RESET
    // read-modify-write, the PLL staircase, and the observe-then-act that gates
    // both one-way steps. Tiles take this lock too, so concurrent per-tile calls
    // are correct without a caller reasoning about which registers are shared.
    //
    // Does not serialise against Metal's own threads (watcher, dprint,
    // profiler, dispatch).
    std::mutex& lock() { return mutex_; }

    // --- topology --------------------------------------------------------

    int chip_id() const { return chip_id_; }
    X280MetalTransport& transport() { return *transport_; }
    const X280MetalTransport& transport() const { return *transport_; }

    // Throws for an index that is harvested or outside the documented set.
    // UMD applies ENABLED_L2CPU as its harvesting mask before returning cores.
    tt::umd::CoreCoord core_for(int hw_index) const;

    const TileIdentity& tile_identity(int hw_index) const;
    const std::vector<TileIdentity>& tile_identities() const { return tile_identities_; }
    std::vector<int> tile_hw_indices() const;

    // --- chip-scoped control ---------------------------------------------

    // Reset is one shared ARC register with a bit per tile, so observation and
    // release remain on the chip context even though the selected bit is
    // tile-specific.
    uint32_t read_l2cpu_reset_raw() const;
    static bool tile_reset_released(uint32_t reset_register, int hw_index);
    bool all_tiles_held_in_reset() const;
    void release_tile_reset(int hw_index);

    // Releases every named L2CPU in one write so no tile runs during another's
    // release. Requires nonempty valid indices; throws on readback mismatch.
    void release_tile_resets(const std::vector<int>& hw_indices);

    // Establishes PLL4 and confirms every enabled tile in fair polling sweeps.
    PllEstablishResult establish_pll(uint32_t mhz, std::chrono::milliseconds confirm_timeout);

private:
    uint32_t read_l2cpu_reset_checked() const;

    int chip_id_;
    std::shared_ptr<X280MetalTransport> transport_;
    tt::umd::CoreCoord arc_{0, 0};
    std::vector<tt::umd::CoreCoord> l2cpu_cores_;
    std::vector<TileIdentity> tile_identities_;
    std::mutex mutex_;
};

}  // namespace x280::detail
