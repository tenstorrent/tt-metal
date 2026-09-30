// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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

/**
 * @brief A Page that also remembers the accessor it came from.
 *
 * noc_addr() is still the software-computed address. The extra accessor pointer lets a NoC transfer of this page
 * ask the accessor for a *transfer* address instead (see transfer_noc_addr.h), which on Quasar can come from the
 * hardware address generator. Derives from Page, so code written against `const Page&` keeps working.
 */
template <typename Accessor>
class AccessorPage : public Page {
public:
    AccessorPage(uint64_t noc_addr, uint32_t global_page_id, const Accessor* accessor) :
        Page(noc_addr, global_page_id), accessor_(accessor) {}

    const Accessor& accessor() const { return *accessor_; }

private:
    const Accessor* accessor_;
};

/**
 * @brief A page yielded by shard_pages(): also remembers which shard it is in and its index within that shard.
 *
 * Pages of a shard are consecutive in its bank, so a transfer of the next page_in_shard can continue a hardware
 * walk even though the global page ids jump between shard rows (see transfer_noc_addr.h).
 */
template <typename Accessor>
class ShardPage : public AccessorPage<Accessor> {
public:
    ShardPage(
        uint64_t noc_addr,
        uint32_t global_page_id,
        const Accessor* accessor,
        uint32_t shard_id,
        uint32_t page_in_shard) :
        AccessorPage<Accessor>(noc_addr, global_page_id, accessor),
        shard_id_(shard_id),
        page_in_shard_(page_in_shard) {}

    uint32_t shard_id() const { return shard_id_; }
    uint32_t page_in_shard() const { return page_in_shard_; }

private:
    uint32_t shard_id_;
    uint32_t page_in_shard_;
};

}  // namespace tensor_accessor
