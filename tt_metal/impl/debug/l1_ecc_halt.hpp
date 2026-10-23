// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <internal/l1_ecc_halt.hpp>
#include <tt-metalium/core_coord.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

// Blackhole silicon: BRISC halts on an L1 DBE and writes mailboxes_t::l1_ecc_halt. Waits detect halts through
// L1EccHaltWatch.
namespace tt::tt_metal::internal {

// A Tensix BRISC still inside an interrupt handler from an earlier process, and the IRQ it is in (BRISC_INT_NO).
struct LatchedBrisc {
    CoreCoord core;
    uint32_t int_no;
};

// Returns every such BRISC on the chip and logs the ones with a halt record. Call before the halt
// records are cleared at init. Runs even when the feature is off.
std::vector<LatchedBrisc> find_latched_briscs(MetalContext& ctx, ChipId device_id);

// Points each latched BRISC's interrupt at code in its mailbox that returns from the handler, so it leaves the handler
// once released from reset. Call while the chip's cores are held in reset, so none can run before both are in place.
void release_latched_briscs(MetalContext& ctx, ChipId device_id, const std::vector<LatchedBrisc>& latched);

// Throws if a released BRISC is still inside an interrupt handler. Call when firmware init fails.
void check_briscs_released(MetalContext& ctx, ChipId device_id, const std::vector<LatchedBrisc>& latched);

// Zeroes the halt record on every Tensix core of the chip. Call before the chip's cores are reset at init, since
// the waits during that reset read the records.
void clear_l1_ecc_halt_records(MetalContext& ctx, ChipId device_id);

// Logs the halt record of every Tensix core that halted.
void report_l1_ecc_at_close(MetalContext& ctx, ChipId device_id);

}  // namespace tt::tt_metal::internal
