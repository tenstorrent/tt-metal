// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/allocator.hpp>
#include "dispatch/dispatch_core_manager.hpp"
#include "llrt/core_descriptor.hpp"
#include <cstdint>

#include "impl/allocator/allocator_types.hpp"
#include "impl/allocator/allocator.hpp"
#include "impl/context/context_types.hpp"
#include "impl/context/metal_env_impl.hpp"

namespace tt::tt_metal {

struct AllocatorConfig;

class L1BankingAllocator : public AllocatorImpl {
public:
    explicit L1BankingAllocator(const AllocatorConfig& alloc_config);
    static AllocatorConfig generate_config(
        dispatch_core_manager& dispatch_core_manager,
        MetalEnvImpl& env,
        ChipId device_id,
        uint8_t num_hw_cqs,
        size_t l1_small_size,
        size_t trace_region_size,
        size_t worker_l1_unreserved_start,
        BankMapping l1_bank_remap);
};

}  // namespace tt::tt_metal
