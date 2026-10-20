// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>

#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::tt_metal {
class MetalContext;
namespace distributed {
class MeshDevice;
}  // namespace distributed
}  // namespace tt::tt_metal

// Internal, unstable API - read the stability/usage conditions in tt_metal/api/internal/README.md
namespace tt::tt_metal::internal {

// Guards one host wait loop against a Blackhole Tensix BRISC halted on an L1 or RISC memory ECC error (its halt
// record in mailboxes_t::l1_ecc_halt). Checks the watched chips' Tensix cores once the loop has been stalled for
// a while, periodically after that, and at once if another wait has already reported a halt. No-op when the
// feature is off.
class L1EccHaltWatch {
public:
    // Watches one chip.
    L1EccHaltWatch(MetalContext& ctx, ChipId device_id);
    // Watches the mesh's chips. A null mesh_device disables the watch.
    explicit L1EccHaltWatch(const distributed::MeshDevice* mesh_device);

    // Call on every iteration that is still waiting. May throw.
    void stalled();
    // Checks immediately. Call before a loop's own timeout throws.
    void check_now();

private:
    MetalContext* ctx_ = nullptr;                           // null when the feature is off
    const distributed::MeshDevice* mesh_device_ = nullptr;  // set by the mesh constructor
    ChipId device_id_ = 0;                                  // the watched chip otherwise
    uint32_t polls_ = 0;
    std::chrono::steady_clock::time_point next_check_;
};

}  // namespace tt::tt_metal::internal
