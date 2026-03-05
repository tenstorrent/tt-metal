// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/allocator.hpp>
#include <cstdint>

#include <tt-metalium/experimental/context/context_descriptor.hpp>
#include "impl/allocator/allocator_types.hpp"
#include "impl/allocator/allocator.hpp"

namespace tt::tt_metal {

struct AllocatorConfig;

class L1BankingAllocator : public AllocatorImpl {
public:
    explicit L1BankingAllocator(const AllocatorConfig& alloc_config);
    static AllocatorConfig generate_config(
        int context_id,
        ChipId device_id,
        uint8_t num_hw_cqs,
        size_t l1_small_size,
        size_t trace_region_size,
        size_t worker_l1_unreserved_start,
        BankMapping l1_bank_remap);
};

}  // namespace tt::tt_metal
