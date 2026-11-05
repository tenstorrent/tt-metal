// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/buf_rw_note.h"
#include "internal/tensor/transfer_noc_addr.h"

namespace tensor_accessor {

template <typename Accessor>
class AccessorPage;
template <typename Accessor>
class ShardPage;

namespace detail {
// The software address of an iterator page, without the op-to-op note: the public noc_addr() adds the note, and the
// generated-address path (internal/tensor/generated_noc_addr.h) uses this as its software fallback, recording its own
// exact access instead. Resolves detail::kLazyNocAddr.
struct PageAccess {
    template <typename Accessor>
    static uint64_t sw_noc_addr(const AccessorPage<Accessor>& page) {
        const uint64_t addr = page.noc_addr_;
        return addr != kLazyNocAddr ? addr : transfer_noc_addr(*page.accessor_, page.page_id(), 0, page.noc_);
    }
    // As above, resolved through the shard (no page-id-to-shard division).
    template <typename Accessor>
    static uint64_t sw_noc_addr(const ShardPage<Accessor>& page) {
        const uint64_t addr = page.noc_addr_;
        return addr != kLazyNocAddr ? addr
                                    : transfer_shard_noc_addr(
                                          page.accessor(),
                                          page.shard_id(),
                                          page.page_in_shard() * page.accessor().get_aligned_page_size(),
                                          page.noc_);
    }
};
}  // namespace detail

/**
 * @brief Represents a page in a tensor with its NOC address and identifiers.
 */
class Page {
public:
    Page(uint64_t noc_addr, uint32_t global_page_id) : noc_addr_(noc_addr), global_page_id_(global_page_id) {}

    // An iterator page used through a base Page reference on the hardware path would hand out detail::kLazyNocAddr as
    // an address: use the AccessorPage / ShardPage the iterator yields, not a Page& to it. Caught in debug builds.
    uint64_t noc_addr() const {
        ASSERT(noc_addr_ != detail::kLazyNocAddr);
        return noc_addr_;
    }
    uint32_t page_id() const { return global_page_id_; }

private:
    friend struct detail::PageAccess;

    uint64_t noc_addr_;
    uint32_t global_page_id_;
};

/**
 * @brief A Page that also remembers the accessor it came from.
 *
 * noc_addr() is the software-computed address. The extra accessor pointer lets a NoC transfer of this page ask the
 * accessor for a generated address instead, which can come from the hardware address generator.
 * When it does (detail::lazy_page_addr_v), the iterators don't compute the software address upfront:
 * the page holds detail::kLazyNocAddr and noc_addr() computes it on demand. Use it as an AccessorPage (or ShardPage),
 * not through a Page reference: Page::noc_addr() can't compute it (it asserts in debug builds).
 */
template <typename Accessor>
class AccessorPage : public Page {
public:
    AccessorPage(uint64_t noc_addr, uint32_t global_page_id, const Accessor* accessor, uint8_t noc = noc_index) :
        Page(noc_addr, global_page_id), accessor_(accessor), noc_(noc) {}

    const Accessor& accessor() const { return *accessor_; }

    // Hides Page::noc_addr: the raw address escapes the binding, so note the tensor as read and written (see
    // TensorAccessor::get_noc_addr). The NoC traits take the generated address and note the exact access instead.
    uint64_t noc_addr() const {
        tt_buf_rw::note_read_write<tt_buf_rw::binding_of<Accessor>>();
        return detail::PageAccess::sw_noc_addr(*this);
    }

private:
    friend struct detail::PageAccess;
    const Accessor* accessor_;
    uint8_t noc_;
};

/**
 * @brief A page yielded by shard_pages(): also remembers which shard it is in and its index within that shard.
 *
 * Pages of a shard are consecutive in its bank, so a transfer of the next page_in_shard can continue a hardware
 * walk even though the global page ids jump between shard rows.
 */
template <typename Accessor>
class ShardPage : public AccessorPage<Accessor> {
public:
    ShardPage(
        uint64_t noc_addr,
        uint32_t global_page_id,
        const Accessor* accessor,
        uint32_t shard_id,
        uint32_t page_in_shard,
        uint8_t noc = noc_index) :
        AccessorPage<Accessor>(noc_addr, global_page_id, accessor, noc),
        shard_id_(shard_id),
        page_in_shard_(page_in_shard) {}

    uint32_t shard_id() const { return shard_id_; }
    uint32_t page_in_shard() const { return page_in_shard_; }

    uint64_t noc_addr() const {
        tt_buf_rw::note_read_write<tt_buf_rw::binding_of<Accessor>>();
        return detail::PageAccess::sw_noc_addr(*this);
    }

private:
    uint32_t shard_id_;
    uint32_t page_in_shard_;
};

}  // namespace tensor_accessor
