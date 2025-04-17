// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/allocator.hpp>
#include <cstdint>

namespace tt {

namespace tt_metal {

struct AllocatorConfig;

class L1BankingAllocator : public Allocator {
public:
    explicit L1BankingAllocator(const AllocatorConfig& alloc_config);
};

}  // namespace tt_metal

}  // namespace tt
