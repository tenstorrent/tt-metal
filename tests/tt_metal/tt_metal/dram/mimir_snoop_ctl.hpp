// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test-side snoop controls for MM41 (plan runtime edit R2). Both write CCE registers through the Mimir's SMC with
// tt::Cluster register accesses and restore the value they found when the guard ends. Neither register is set by
// UMD's emulator bring-up beyond its defaults, and nothing shows that uncore reset restores them, so every write is
// undone by its guard.

#pragma once

#include <cstdint>

#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::mimir_test {

// RegionAttrs bit 30 (cmd_snoop, driven onto AxUSER[9]) of DMRISC remap entry 1, the local Mimir's SRAM window.
inline constexpr uint32_t kRemapCmdSnoopBit = 1u << 30;

// NOC_SNOOP_TL_MASTER_CFG of `cce_index` (bit 0 disable_inline_writes, bit 1 invalidate_flush_dirty_en, bit 2
// clean_putdata_en).
uint32_t read_snoop_master_config(const Cluster& cluster, ChipId chip, uint32_t cce_index);

// Sets or clears cmd_snoop on `cce_index`'s remap entry 1 for its lifetime, so that CCE's hart stores to an SPA in its
// Mimir's SRAM window (a sibling CCE's TL1, or its own through the SPA) carry the snoop bit. Uncached-alias stores do
// not pass through the remap. The destructor writes back the word it found; a throw is reported as a failure.
class ScopedSramRemapSnoop {
public:
    ScopedSramRemapSnoop(const Cluster& cluster, ChipId chip, uint32_t cce_index, bool snoop);
    ScopedSramRemapSnoop(const ScopedSramRemapSnoop&) = delete;
    ScopedSramRemapSnoop& operator=(const ScopedSramRemapSnoop&) = delete;
    ~ScopedSramRemapSnoop();

    uint32_t original() const { return original_; }
    uint32_t applied() const { return applied_; }

private:
    const Cluster& cluster_;
    ChipId chip_;
    uint32_t cce_index_;
    uint32_t original_ = 0;
    uint32_t applied_ = 0;
};

// Writes `config` to `cce_index`'s NOC_SNOOP_TL_MASTER_CFG for its lifetime; the destructor writes back the value it
// found. A throw is reported as a failure.
class ScopedSnoopMasterConfig {
public:
    ScopedSnoopMasterConfig(const Cluster& cluster, ChipId chip, uint32_t cce_index, uint32_t config);
    ScopedSnoopMasterConfig(const ScopedSnoopMasterConfig&) = delete;
    ScopedSnoopMasterConfig& operator=(const ScopedSnoopMasterConfig&) = delete;
    ~ScopedSnoopMasterConfig();

    uint32_t applied() const { return applied_; }

private:
    const Cluster& cluster_;
    ChipId chip_;
    uint32_t cce_index_;
    uint32_t original_ = 0;
    uint32_t applied_ = 0;
};

}  // namespace tt::tt_metal::mimir_test
