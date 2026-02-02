// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef COMPILE_FOR_TRISC
#include "internal/circular_buffer_interface.h"
#ifdef TRISC_PACK
#include "llk_io_pack.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_io_unpack.h"
#endif
#else  // !COMPILE_FOR_TRISC
#include "experimental/noc.h"
#endif

#include "experimental/lock.h"

namespace experimental {

class CircularBuffer {
public:
    enum class AddrSelector { WRITE_PTR, READ_PTR };

    explicit CircularBuffer(uint32_t cb_id) : cb_id_(cb_id) {}

    uint32_t get_cb_id() const { return cb_id_; }

    void reserve_back(int32_t num_pages) {
#ifdef COMPILE_FOR_TRISC
        PACK((llk_wait_for_free_tiles<false, false, false>(cb_id_, num_pages)));
#else
        cb_reserve_back(cb_id_, num_pages);
#endif
    }

    void push_back(int32_t num_pages) {
#ifdef COMPILE_FOR_TRISC
        PACK((llk_push_tiles<false, false>(cb_id_, num_pages)));
#else
        cb_push_back(cb_id_, num_pages);
#endif
    }

    void wait_front(int32_t num_pages) {
#ifdef COMPILE_FOR_TRISC
        UNPACK((llk_wait_tiles(cb_id_, num_pages)));
#else
        cb_wait_front(cb_id_, num_pages);
#endif
    }

    void pop_front(int32_t num_pages) {
#ifdef COMPILE_FOR_TRISC
        UNPACK((llk_pop_tiles(cb_id_, num_pages)));
#else
        cb_pop_front(cb_id_, num_pages);
#endif
    }

#ifdef COMPILE_FOR_TRISC
    uint32_t get_tile_address(uint32_t tile_index) {
        uint32_t address = 0;

        UNPACK({
            uint32_t operand_id = get_operand_id(cb_id_);
            uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr;
            uint32_t offset_address = get_local_cb_interface(operand_id).fifo_page_size * tile_index;
            address = (base_address + offset_address) << 4;  // Convert to byte address

            mailbox_write(ckernel::ThreadId::MathThreadId, address);
            mailbox_write(ckernel::ThreadId::PackThreadId, address);
        })

        MATH(address = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
        PACK(address = mailbox_read(ckernel::ThreadId::UnpackThreadId);)

        return address;
    }

    uint32_t read_tile_value(uint32_t tile_index, uint32_t element_offset) {
        uint32_t value = 0;

        UNPACK({
            uint32_t operand_id = get_operand_id(cb_id_);
            uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr;
            uint32_t offset_address = get_local_cb_interface(operand_id).fifo_page_size * tile_index;
            uint32_t byte_address = (base_address + offset_address) << 4;  // Convert to byte address

            value = reinterpret_cast<volatile uint32_t*>(byte_address)[element_offset];

            mailbox_write(ckernel::ThreadId::MathThreadId, value);
            mailbox_write(ckernel::ThreadId::PackThreadId, value);
        })

        MATH(value = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
        PACK(value = mailbox_read(ckernel::ThreadId::UnpackThreadId);)

        return value;
    }
#else
#ifdef DATA_FORMATS_DEFINED
    uint32_t get_tile_size() const { return ::get_tile_size(cb_id_); }
    uint32_t get_tile_hw() const { return ::get_tile_hw(cb_id_); }
    DataFormat get_dataformat() const { return ::get_dataformat(cb_id_); }
#endif

    bool pages_reservable_at_back(int32_t num_pages) const { return cb_pages_reservable_at_back(cb_id_, num_pages); }

    bool pages_available_at_front(int32_t num_pages) const { return cb_pages_available_at_front(cb_id_, num_pages); }
#endif

    uint32_t get_write_ptr() const {
        // return byte address (fifo_wr_ptr is 16B address)
        uint32_t wr_ptr_bytes = get_local_cb_interface(cb_id_).fifo_wr_ptr;
        return wr_ptr_bytes;
    }

    uint32_t get_read_ptr() const {
        // return byte address (fifo_rd_ptr is 16B address)
        uint32_t rd_ptr_bytes = get_local_cb_interface(cb_id_).fifo_rd_ptr;
        return rd_ptr_bytes;
    }

    [[nodiscard]] auto scoped_lock() {
        // TODO: Register with the debugger to track the lock
        return Lock([this]() { release_scoped_lock(); });
    }

private:
    void release_scoped_lock() {
        // TODO: Unregister with the debugger
    }

    uint32_t cb_id_;
};

#ifndef COMPILE_FOR_TRISC
template <>
struct noc_traits_t<CircularBuffer> {
    struct src_args_type {
        uint32_t offset_bytes{};
    };
    struct dst_args_type {
        uint32_t offset_bytes{};
    };
    struct dst_args_mcast_type {
        uint32_t noc_x_start{};
        uint32_t noc_y_start{};
        uint32_t noc_x_end{};
        uint32_t noc_y_end{};
        uint32_t offset_bytes{};
    };
    template <Noc::AddressType address_type>
    static auto src_addr(const CircularBuffer& src, const Noc&, const src_args_type& args) {
        static_assert(
            address_type == Noc::AddressType::LOCAL_L1,
            "CircularBuffer without mcast range can only be used as L1 source");
        return src.get_read_ptr() + args.offset_bytes;
    }
    template <Noc::AddressType address_type>
    static auto dst_addr(const CircularBuffer& dst, const Noc& noc, const dst_args_type& args) {
        static_assert(
            address_type == Noc::AddressType::LOCAL_L1,
            "CircularBuffer without mcast range can only be used as L1 source");
        return dst.get_write_ptr() + args.offset_bytes;
    }
    template <Noc::AddressType address_type>
    static auto dst_addr_mcast(const CircularBuffer& dst, const Noc& noc, const dst_args_mcast_type& args) {
        static_assert(
            address_type == Noc::AddressType::NOC, "CircularBuffer with mcast range cannot be used as L1 source");
        auto local_addr = dst.get_write_ptr() + args.offset_bytes;
        return ::get_noc_multicast_addr(
            args.noc_x_start, args.noc_y_start, args.noc_x_end, args.noc_y_end, local_addr, noc.get_noc_id());
    }
};

template <CircularBuffer::AddrSelector AddrSel>
struct CircularBufferView {
    const CircularBuffer& cb;
    explicit constexpr CircularBufferView(const CircularBuffer& c) : cb(c) {}
};

// Convenience helper: use<CircularBuffer::AddrSelector::READ_PTR>(cb)
// This allows user to indicate whether the read or write pointer should be used as the source or destination address
// depending on whether the CircularBuffer is src or dst in the Noc apis
template <CircularBuffer::AddrSelector AddrSel>
constexpr auto use(const CircularBuffer& cb) {
    return CircularBufferView<AddrSel>(cb);
}

template <CircularBuffer::AddrSelector AddrSel>
class noc_traits_t<CircularBufferView<AddrSel>> {
public:
    struct src_args_type {
        uint32_t offset_bytes{};
    };
    struct dst_args_type {
        uint32_t offset_bytes{};
    };
    struct dst_args_mcast_type {
        uint32_t noc_x_start{};
        uint32_t noc_y_start{};
        uint32_t noc_x_end{};
        uint32_t noc_y_end{};
        uint32_t offset_bytes{};
    };
    template <Noc::AddressType address_type>
    static auto src_addr(const CircularBufferView<AddrSel>& view, const Noc&, const src_args_type& args) {
        static_assert(
            address_type == Noc::AddressType::LOCAL_L1,
            "CircularBuffer without mcast range can only be used as L1 source");
        return get_local_addr(view) + args.offset_bytes;
    }
    template <Noc::AddressType address_type>
    static auto dst_addr(const CircularBufferView<AddrSel>& view, const Noc& noc, const dst_args_type& args) {
        static_assert(
            address_type == Noc::AddressType::LOCAL_L1,
            "CircularBuffer without mcast rangecan only be used as L1 source");
        return get_local_addr(view) + args.offset_bytes;
    }
    template <Noc::AddressType address_type>
    static auto dst_addr_mcast(
        const CircularBufferView<AddrSel>& view, const Noc& noc, const dst_args_mcast_type& args) {
        static_assert(
            address_type == Noc::AddressType::NOC, "CircularBuffer with mcast range cannot be used as L1 source");
        auto local_addr = get_local_addr(view) + args.offset_bytes;
        return ::get_noc_multicast_addr(
            args.noc_x_start, args.noc_y_start, args.noc_x_end, args.noc_y_end, local_addr, noc.get_noc_id());
    }

private:
    static constexpr auto get_local_addr(const CircularBufferView<AddrSel>& view) {
        if constexpr (AddrSel == CircularBuffer::AddrSelector::READ_PTR) {
            return view.cb.get_read_ptr();
        } else {
            return view.cb.get_write_ptr();
        }
    }
};
#endif

}  // namespace experimental
