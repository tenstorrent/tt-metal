// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The component's single Metal integration and hardware-access boundary.
//
// RuntimeImpl binds one instance to its caller-owned MeshDevice and shares it
// across every chip context. Topology, register, and LIM operations stay on
// tt::Cluster. The raw UMD cluster is borrowed only for ARC telemetry, which
// the wrapper does not expose.
// Wrapper I/O pins UMD's thread-local physical fabric to NOC0 independently of
// converting X280's NOC0 identities to Metal's TRANSLATED coordinates.
//
// Register and block operations are intentionally distinct:
//
//   register path  4-byte aligned, 32-bit accesses through the uncached,
//                  strictly ordered wrapper path. Verified writes force each
//                  PLL/reset/register step to complete before the next.
//   block path     4-byte aligned bulk memory accesses. Used for LIM through
//                  Metal's tile-local static TLB window.

#pragma once

#include <core_coord.hpp>
#include <tt-umd/types/core_coordinates.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace tt::umd {
class Cluster;
}

namespace tt {
class Cluster;
}

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace x280 {

class X280MetalTransport {
public:
    static std::shared_ptr<X280MetalTransport> bind(const tt::tt_metal::distributed::MeshDevice& mesh);

    bool has_chip(int chip_id) const;
    bool is_mmio_local(int chip_id) const;
    std::vector<tt::umd::CoreCoord> l2cpu_cores(int chip_id) const;
    tt::umd::CoreCoord arc_core(int chip_id) const;

    /** Translates an enabled physical NOC0 L2CPU identity to its virtual
     * coordinate. Throws for a nonlocal chip, absent/harvested core, or other core type.
     * Caller keeps the bound mesh live; no device I/O. */
    tt::tt_metal::CoreCoord virtual_core_from_noc0(int chip_id, const tt::tt_metal::CoreCoord& noc0) const;

    uint32_t read_reg32(int chip_id, const tt::umd::CoreCoord& core, uint64_t addr) const;
    uint32_t read_reg32_checked(int chip_id, const tt::umd::CoreCoord& core, uint64_t addr, const char* what) const;
    void write_reg32(int chip_id, const tt::umd::CoreCoord& core, uint64_t addr, uint32_t value) const;
    void write_reg32_verified(int chip_id, const tt::umd::CoreCoord& core, uint64_t addr, uint32_t value) const;

    void read_block(int chip_id, const tt::umd::CoreCoord& core, uint64_t addr, void* dst, size_t size) const;
    void write_block(int chip_id, const tt::umd::CoreCoord& core, uint64_t addr, const void* src, size_t size) const;
    uint64_t read_u64(int chip_id, const tt::umd::CoreCoord& core, uint64_t addr) const;
    void write_u64(int chip_id, const tt::umd::CoreCoord& core, uint64_t addr, uint64_t value) const;

    // nullopt when the chip has no telemetry reader or the entry is absent.
    std::optional<uint32_t> telemetry(int chip_id, uint8_t tag) const;

private:
    X280MetalTransport(const tt::Cluster& cluster, tt::umd::Cluster& telemetry_driver);

    void require_chip(int chip_id) const;
    void require_mmio_chip(int chip_id) const;

    const tt::Cluster* cluster_;
    tt::umd::Cluster* telemetry_driver_;
};

// Returns the mesh's unique physical device ids in ascending order after
// validating that the mesh is open, Blackhole, non-empty, and represented by
// the Metal transport.
std::vector<int> sorted_usable_chip_ids(
    const tt::tt_metal::distributed::MeshDevice& mesh, const X280MetalTransport& transport);

}  // namespace x280
