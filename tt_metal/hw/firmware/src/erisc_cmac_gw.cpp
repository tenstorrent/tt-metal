// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Gateway kernel: moves packets between host-writable L1 ring and CMAC TX.
// Runs on the erisc core of an external CMAC port after PCS lock.
//
// Build note: this file belongs alongside the other erisc firmware sources in
//   tt_metal/hw/firmware/src/tt-1xx/sources.cmake
// Add "erisc_cmac_gw.cpp" to FIRMWARE_JIT_API_FILES once the cross-compiler
// toolchain for erisc is wired up for this target.

#include <cstdint>

// Ring-buffer layout (must match ExternalIfaceSender constants).
static constexpr uint32_t kRingBase = 0x2000;
static constexpr uint32_t kRingSlots = 8;
static constexpr uint32_t kSlotBytes = 2048;

// Slot header: host writes size (bytes) here before marking slot valid.
struct SlotHeader {
    uint32_t size;   // 0 = empty; host sets non-zero to signal a frame
    uint32_t flags;  // reserved for future use
};

void kernel_main() {
    uint32_t tail = 0;
    while (true) {
        auto* slot = reinterpret_cast<volatile SlotHeader*>(kRingBase + tail * kSlotBytes);
        if (slot->size == 0) {
            // No data — yield until host enqueues a frame.
            continue;
        }

        // TODO: forward slot payload to CMAC TX path.
        //   uint8_t* payload = reinterpret_cast<uint8_t*>(slot) + sizeof(SlotHeader);
        //   cmac_tx_send(payload, slot->size);

        slot->size = 0;  // mark slot consumed so host can reuse it
        tail = (tail + 1 == kRingSlots) ? 0 : tail + 1;
    }
}
