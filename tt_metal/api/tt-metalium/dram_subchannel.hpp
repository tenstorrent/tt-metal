// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal {

class IDevice;

namespace experimental {

// Returns the DRAM subchannel (y coord in the DRAM logical grid) for the given bank
// that is not used as a worker_endpoint or eth_endpoint by either NOC, per the SOC
// descriptor. Intended for DRISC kernels that need to claim an unreserved subchannel.
//
// Throws if no free subchannel exists for the bank.
uint32_t pick_unused_dram_subchannel(IDevice* device, uint32_t bank_id);

}  // namespace experimental

}  // namespace tt::tt_metal
