// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <internal/l1_ecc_halt.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

// Blackhole silicon: BRISC halts on an L1 DBE, a RISC local memory DBE or an instruction cache parity error, and
// writes mailboxes_t::l1_ecc_halt. Waits detect halts through L1EccHaltWatch.
namespace tt::tt_metal::internal {

// Throws if a Tensix BRISC is still inside an interrupt handler, which only tt-smi -r clears and which makes
// firmware init time out. Runs even when the feature is off.
void check_no_halted_brisc_at_init(MetalContext& ctx, ChipId device_id);

// Zeroes the halt record on every Tensix core of the chip. Call before the chip's cores are reset at init, since
// the waits during that reset read the records.
void clear_l1_ecc_halt_records(MetalContext& ctx, ChipId device_id);

// Logs the halt record of every Tensix core that halted.
void report_l1_ecc_at_close(MetalContext& ctx, ChipId device_id);

}  // namespace tt::tt_metal::internal
