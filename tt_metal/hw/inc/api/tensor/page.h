// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/buf_rw_note.h"
#include "internal/tensor/transfer_noc_addr.h"

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
 * noc_addr() is the software-computed address. The extra accessor pointer lets a NoC transfer of this page ask the
 * accessor for a *transfer* address instead (see transfer_noc_addr.h), which on Quasar can come from the hardware
 * address generator. Where it does (detail::lazy_page_addr_v), the iterators don't compute the software address up
 * front: the page holds kLazyNocAddr and noc_addr() computes it on demand.
 */
template <typename Accessor>
class AccessorPage : public Page {
public:
    static constexpr uint64_t kLazyNocAddr = ~0ull;  // the software address isn't computed yet

    AccessorPage(uint64_t noc_addr, uint32_t global_page_id, const Accessor* accessor, uint8_t noc = noc_index) :
        Page(noc_addr, global_page_id), accessor_(accessor), noc_(noc) {}

    const Accessor& accessor() const { return *accessor_; }

    // Hides Page::noc_addr: the raw address escapes the binding, so note the tensor as read and written (see
    // TensorAccessor::get_noc_addr). The NoC traits take the transfer address and note the exact access instead.
    uint64_t noc_addr() const {
        tt_buf_rw::note_read_write<tt_buf_rw::binding_of<Accessor>>();
        return sw_noc_addr();
    }

    // The software address without the note: for the transfer path's software fallback.
    uint64_t sw_noc_addr() const {
        const uint64_t addr = Page::noc_addr();
        return addr != kLazyNocAddr ? addr : detail::transfer_noc_addr(*accessor_, page_id(), 0, noc_);
    }

protected:
    uint8_t noc() const { return noc_; }

private:
    const Accessor* accessor_;
    uint8_t noc_;
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
        uint32_t page_in_shard,
        uint8_t noc = noc_index) :
        AccessorPage<Accessor>(noc_addr, global_page_id, accessor, noc),
        shard_id_(shard_id),
        page_in_shard_(page_in_shard) {}

    uint32_t shard_id() const { return shard_id_; }
    uint32_t page_in_shard() const { return page_in_shard_; }

    uint64_t noc_addr() const {
        tt_buf_rw::note_read_write<tt_buf_rw::binding_of<Accessor>>();
        return sw_noc_addr();
    }

    // As AccessorPage's, resolved through the shard (no page-id-to-shard division).
    uint64_t sw_noc_addr() const {
        const uint64_t addr = Page::noc_addr();
        return addr != AccessorPage<Accessor>::kLazyNocAddr
                   ? addr
                   : detail::transfer_shard_noc_addr(
                         this->accessor(),
                         shard_id_,
                         page_in_shard_ * this->accessor().get_aligned_page_size(),
                         this->noc());
    }

private:
    uint32_t shard_id_;
    uint32_t page_in_shard_;
};

}  // namespace tensor_accessor
