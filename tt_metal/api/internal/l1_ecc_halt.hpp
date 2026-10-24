// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>

#include <tt-umd/types/cluster_descriptor_types.hpp>

namespace tt::tt_metal {
class MetalContext;
namespace distributed {
class MeshDevice;
}  // namespace distributed
}  // namespace tt::tt_metal

// Internal, unstable API - read the stability/usage conditions in tt_metal/api/internal/README.md
namespace tt::tt_metal::internal {

// Guards one host wait loop against a Blackhole Tensix BRISC halted on an L1 ECC error. Once the wait has been
// stalled for about 10 s, and every 10 s after that, it reads the watched chips' halt records
// (mailboxes_t::l1_ecc_halt) and throws std::runtime_error naming the chip, core and cause if one has halted. After
// any wait has reported a halt, every other wait throws at once. A no-op off Blackhole silicon, or with
// TT_METAL_DISABLE_L1_ECC_HALT=1.
class L1EccHaltWatch {
public:
    // Watches one chip.
    L1EccHaltWatch(MetalContext& ctx, ChipId device_id);
    // Watches the mesh's chips. A null mesh_device disables the watch.
    explicit L1EccHaltWatch(const distributed::MeshDevice* mesh_device);

    // Call on every iteration of the wait that has not finished yet. Reads the device only once the wait has been
    // stalled for a while (see above). Throws if a halt is found.
    void stalled();
    // Reads the halt records now and throws if one is found. Call before a loop's own timeout throws.
    void check_now();

private:
    MetalContext* ctx_ = nullptr;                           // null when the feature is off
    const distributed::MeshDevice* mesh_device_ = nullptr;  // set by the mesh constructor
    ChipId device_id_ = 0;                                  // the watched chip otherwise
    uint32_t polls_ = 0;
    std::chrono::steady_clock::time_point next_check_;
};

}  // namespace tt::tt_metal::internal
