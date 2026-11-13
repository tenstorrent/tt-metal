// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// needed for private members
#include "aux_region_allocator.hpp"
#include "cq_backing.hpp"
#include "system_memory_cq.hpp"
#include "system_memory_cq_interface.hpp"
#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/device_types.hpp>  // for tt::ChipId
#include <tt_stl/assert.hpp>

namespace tt::tt_metal {

class Buffer;
class MetalContext;
class MetalEnvImpl;

// The host side of a device's fast dispatch command queues: one SystemMemoryCq per hardware CQ, over one CqBacking.
class SystemMemoryManager {
public:
    // Create a SystemMemoryManager for accessing system memory accessible by the given device
    // The referenced MetalEnvImpl and MetalContext must outlive this SystemMemoryManager.
    SystemMemoryManager(MetalEnvImpl& env, MetalContext& ctx, ChipId device_id, uint8_t num_hw_cqs);

    SystemMemoryCq& cq(uint8_t cq_id) {
        TT_ASSERT(cq_id < cqs_.size(), "CQ {} is out of range: device {} has {} CQs", cq_id, device_id_, cqs_.size());
        return *cqs_[cq_id];
    }
    const SystemMemoryCq& cq(uint8_t cq_id) const {
        TT_ASSERT(cq_id < cqs_.size(), "CQ {} is out of range: device {} has {} CQs", cq_id, device_id_, cqs_.size());
        return *cqs_[cq_id];
    }

    uint8_t num_cqs() const { return static_cast<uint8_t>(cqs_.size()); }

    uint32_t get_cq_size() const { return backing_->cq_size(); }

    ChipId get_device_id() const { return device_id_; }

    MetalEnvImpl& env_impl() const { return env_; }

    MetalContext& metal_context() const { return ctx_; }

    bool is_dram_backed() const { return backing_->is_dram_backed(); }

    uint32_t get_dram_region_base_addr() const;

    uint32_t get_dram_region_bank_id() const;

    AuxRegionAllocator& aux_region_allocator() { return backing_->aux_region_allocator(); }

private:
    MetalEnvImpl& env_;
    MetalContext& ctx_;
    ChipId device_id_ = 0;
    std::unique_ptr<CqBacking> backing_;
    std::vector<std::unique_ptr<SystemMemoryCq>> cqs_;
};

}  // namespace tt::tt_metal
