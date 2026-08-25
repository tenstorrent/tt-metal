// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

namespace tt::tt_fabric::experimental {

// Host/device sizing contract for the worker-L1 software transaction-counter
// bank used by FabricDataflowBuffer.
struct FabricTransactionCounterConfig {
    using TransactionId = uint32_t;
    using Counter = uint32_t;

    static constexpr uint32_t counter_size_bytes = sizeof(Counter);

    static constexpr bool is_valid_capacity(uint64_t max_transaction_ids) {
        return max_transaction_ids > 0 &&
               max_transaction_ids <= std::numeric_limits<uint32_t>::max() / counter_size_bytes;
    }

    static constexpr uint64_t storage_size_bytes(uint64_t max_transaction_ids) {
        return max_transaction_ids * counter_size_bytes;
    }
};

}  // namespace tt::tt_fabric::experimental
