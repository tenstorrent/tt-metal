// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// l1_layout.h
//
// Host-side index and data types for the dynamic L1 region layout (Quasar / tt-2xx).
//
// The movable shared-L1 region bases are computed once by compute_l1_layout() (dev_mem_map.h,
// host/non-device branch) from measured firmware section sizes. They reach the *device* only as
// absolute linker symbols (`__l1_*`), emitted by the Quasar HAL's linker_flags (plan B2 / D1) from
// that layout and resolved by dev_mem_map.h's device-branch macros (D2). There is no device-side
// runtime accessor and no mailbox descriptor: this file is the host index (L1Region), the layout
// data types, and the fixed image/region counts. It is host-only (not included on device).
//
// Only *movable shared-L1* regions are listed. TRISC TLS/local lives in per-core local data RAM
// (MEM_TRISC_LOCAL_BASE = MEM_LOCAL_BASE = 0x802000), outside the 4 MB L1, so it is NOT a region
// here. The fixed low prefix (mailbox/zeros/llk-debug) and kernel_config_base (already runtime via
// launch_msg) are also excluded.

#pragma once

#include <array>
#include <cstdint>

// Addresses below are DeviceAddr (== std::uint64_t in tt-metalium/hal_types.hpp). This leaf uses the
// underlying std::uint64_t directly so it stays dependency-free (no tt-metalium include), which lets
// it be pulled in from the low-level dev_mem_map.h host branch without an include-path assumption.

enum class L1Region : uint8_t {
    DM_GLOBAL,
    TRISC0_GLOBAL,
    TRISC1_GLOBAL,
    TRISC2_GLOBAL,
    TRISC3_GLOBAL,
    DM_LOCAL,  // L1-resident (DM only; TRISC local is in local RAM, excluded)
    NOC_COUNTER,
    FABRIC_COUNTER,
    FABRIC_CONNECTION_LOCK,
    ROUTING_TABLE,
    ROUTING_PATH,
    EXIT_NODE_TABLE,
    FABRIC_CONNECTIONS,
    PACKET_HEADER_POOL,
    INIT_LOCAL_SCRATCH,
    BANK_TO_NOC_SCRATCH,
    LOGICAL_TO_VIRTUAL_SCRATCH,
    MAP_END,  // start of kernel-config / worker-L1 (the boundary reclamation pushes down)
    // Per-TRISC firmware-text load bases. Static = MEM_TRISC{0..3}_FIRMWARE_BASE; under packing they
    // move up into the reclaimed DM/TRISC text headroom, so DM firmware reads them (instead of the
    // baked macro) to set each TRISC's reset-PC. Appended last so the indices above are unchanged.
    TRISC0_FW,
    TRISC1_FW,
    TRISC2_FW,
    TRISC3_FW,
    COUNT
};

inline constexpr uint8_t NUM_L1_REGIONS = static_cast<uint8_t>(L1Region::COUNT);

// Firmware images in layout order: DM followed by one per TRISC core. Fixed for Quasar; tied to
// dev_mem_map.h's NUM_TRISC_CORES by a static_assert there (this leaf has no macro dependency).
inline constexpr uint8_t NUM_L1_FW_IMAGES = 5;  // 1 (DM) + 4 (TRISC0..3)

// Measured (post-link) firmware section sizes, fed to compute_l1_layout() to reclaim the firmware
// reservation headroom. Image order matches NUM_L1_FW_IMAGES ({DM, TRISC0..3}).
struct FirmwareSectionSizes {
    uint32_t text = 0;
    uint32_t data = 0;
    uint32_t tls = 0;
};

// Packed linker defsyms for one firmware image (plan B2), read at link() time to place that image.
// fw_text is the (XIP) load/reset base; text_size / data_size are the tightened .segments overflow
// limits for .text / firmware .data (R9 packs both to measured+margin). fw_data is the firmware
// globals base; kn_data is the kernel globals base (= fw_data + data_size), where the kernel's own
// .data lands (per-core stride applied by linker_flags). fw_tls is the local-data base (DM: L1,
// shifted down; TRISC: local RAM 0x802000, unchanged). DM local size is not tightened in this cut.
struct PackedFirmwareDefsyms {
    std::uint64_t fw_text = 0;
    uint32_t text_size = 0;
    std::uint64_t fw_data = 0;
    uint32_t data_size = 0;
    std::uint64_t kn_data = 0;
    std::uint64_t fw_tls = 0;
};

// Packed movable-region bases (indexed by L1Region) + the lowered worker-L1 boundary, computed by
// compute_l1_layout() from measured firmware sizes. The bases reach the device as `__l1_*` linker
// symbols (linker_flags / D1); the host consumes them via the HAL mem-map and the allocator
// (worker_l1_size = MEM_L1_SIZE - map_end). `valid` is false only on arches without dynamic L1.
// `fw` holds the per-firmware relink defsyms in image order (Quasar = {DM, TRISC0..3}).
struct PackedL1Layout {
    bool valid = false;
    std::array<std::uint64_t, NUM_L1_REGIONS> region_base{};
    std::uint64_t map_end = 0;
    std::array<PackedFirmwareDefsyms, NUM_L1_FW_IMAGES> fw{};
};
