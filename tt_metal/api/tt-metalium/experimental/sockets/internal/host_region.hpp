// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The one region: a static 2 MiB-aligned span holding the credit array and every core's
// arenas, pinned once for the device and exposed once in the RMA window.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <tt-metalium/experimental/sockets/internal/host_uva_layout.hpp>
#include <tt-metalium/experimental/sockets/internal/host_uva.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
}
namespace tt::tt_metal::experimental {
class PinnedMemory;
}

namespace tt::tt_metal::experimental {

// Which of a core's two arenas an overlay covers: TX is the D2H FIFO, RX the H2D ring.
enum class AliasArena : uint32_t { Tx = 0, Rx = 1, Count = 2 };

class HostRegion {
public:
    struct Grid {
        uint32_t width = 0;
        uint32_t height = 0;
    };

    static HostRegion& provision(
        const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
        uint32_t chip,
        uint32_t cores_in_use,
        HostTopology topology,
        Grid grid);

    static bool is_provisioned();

    // Unpins and clears the one-region flag, so a later provision() can succeed. Must run
    // BEFORE the overlays are unmapped: the pin names pages that are about to be swapped.
    static void release();

    // Fixed from program start, so RingAlias can overlay before there is a region to ask.
    static uint8_t* reserved_base();

    // An overlay declares how much of an arena this region may still fill, and where the
    // tail it may fill again begins. Refused after provisioning: see RingAlias.
    static void declare_alias(AliasArena arena, uint32_t core, uint64_t fill_bytes, uint64_t mapped_bytes);
    static void clear_aliases();
    static uint64_t alias_fill_bytes(AliasArena arena, uint32_t core);
    static uint64_t alias_tail_offset(AliasArena arena, uint32_t core);

    uint8_t* base() const { return base_; }
    uint64_t pinned_bytes() const { return pinned_bytes_; }
    uint32_t cores_in_use() const { return cores_in_use_; }

    // How the device addresses this region; both halves go to the socket metadata.
    struct DeviceView {
        uint32_t pcie_xy_enc = 0;
        uint64_t io_base = 0;
    };
    const DeviceView& device() const { return device_; }

    RegionHeader* header() const { return reinterpret_cast<RegionHeader*>(base_); }
    uint8_t* tx_arena(uint32_t core) const { return base_ + tx_arena_offset(core); }
    uint8_t* rx_arena(uint32_t core) const { return base_ + rx_arena_offset(core); }

    // Fills both arenas with the complement of a correct payload, so an unwritten byte
    // always differs and a test cannot pass by accident.
    void reset_arenas(uint8_t fill = kArenaFill);

    // Empty string means the published geometry matches this build.
    std::string verify_header() const;

    static constexpr uint8_t kArenaFill = 0xA5;

private:
    HostRegion() = default;
    // A function-local static behind a private constructor: "one region" is a property
    // the type enforces, not a rule the .cpp happens to follow.
    static HostRegion& storage();

    uint8_t* base_ = nullptr;
    uint64_t pinned_bytes_ = 0;
    uint32_t cores_in_use_ = 0;
    uint32_t chip_ = 0;
    HostTopology topology_{};
    Grid grid_{};
    DeviceView device_{};
    std::shared_ptr<PinnedMemory> pinned_;
};

// Reported before anything is pinned, so an over-large request fails with a number rather
// than inside an ioctl.
struct PinLimits {
    uint64_t rlimit_memlock = 0;
    uint32_t max_pins = 0;
    uint64_t max_total_pin = 0;
    bool can_map_to_noc = false;
};
PinLimits query_pin_limits(const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device);

}  // namespace tt::tt_metal::experimental
