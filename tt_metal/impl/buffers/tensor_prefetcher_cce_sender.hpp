// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The Tensor prefetcher's sender on DRAM cores that are CCEs: its backend, the TL1 request ring that feeds it, and
// its status.

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <tt-metalium/core_coord.hpp>

#include "impl/buffers/tensor_prefetcher_cce_abi.hpp"

namespace tt::tt_metal {

class IDevice;

namespace distributed {

class MeshDevice;
class PrefetcherSenderBackend;

std::unique_ptr<PrefetcherSenderBackend> make_cce_sender_backend();

// TL1 offset of every CCE sender's control block (tensor_prefetcher_cce_abi.hpp). It depends only on
// the mesh's DRISC L1 arena, so it holds before Start and after Stop.
uint32_t cce_sender_control_base(const MeshDevice& mesh_device);

// The status of the CCE sender on `sender_logical` whose control block is at `control_base`.
CceSenderStatus read_cce_sender_status(
    const MeshDevice& mesh_device, IDevice* device, const CoreCoord& sender_logical, uint32_t control_base);

std::string format_cce_sender_status(const CceSenderStatus& status);

}  // namespace distributed
}  // namespace tt::tt_metal
