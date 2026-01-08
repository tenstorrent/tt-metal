// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tensor_accessor {

/**
 * @brief Represents a page in a tensor with its NOC address and identifiers.
 */
class Page {
public:
    Page(uint64_t noc_addr, uint32_t global_page_id) : noc_addr_(noc_addr), global_page_id_(global_page_id) {}

    uint64_t noc_addr() const { return noc_addr_; }
    uint32_t page_id() const { return global_page_id_; }

private:
    uint64_t noc_addr_;
    uint32_t global_page_id_;
};

}  // namespace tensor_accessor
