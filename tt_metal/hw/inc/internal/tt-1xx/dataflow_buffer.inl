// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Defines the _impl bodies for DataflowBuffer on tt-1xx architectures

#ifndef ARCH_QUASAR

#include "stream_io_map.h"
#ifdef COMPILE_FOR_TRISC
#include "api/compute/common_globals.h"  // defines PACK/UNPACK/MATH macros
#include "internal/circular_buffer_interface.h"
#ifdef TRISC_PACK
#include "llk_io_pack.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_io_unpack.h"
#endif
#endif  // COMPILE_FOR_TRISC

namespace experimental {

inline DataflowBuffer::DataflowBuffer(uint16_t logical_dfb_id)
    : local_dfb_interface_(get_local_cb_interface(logical_dfb_id)), logical_dfb_id_(logical_dfb_id) {}

inline uint32_t DataflowBuffer::get_entry_size() const { return local_dfb_interface_.fifo_page_size; }

inline uint32_t DataflowBuffer::get_stride_size() const { return local_dfb_interface_.fifo_page_size; }

inline void DataflowBuffer::reserve_back_impl(uint16_t num_entries) {
#ifdef COMPILE_FOR_TRISC
    PACK((llk_wait_for_free_tiles<false, false, false>(logical_dfb_id_, num_entries)));
#else
    cb_reserve_back(logical_dfb_id_, num_entries);
#endif
}

inline void DataflowBuffer::push_back_impl(uint16_t num_entries) {
#ifdef COMPILE_FOR_TRISC
    PACK((llk_push_tiles<false, false>(logical_dfb_id_, num_entries)));
#else
    cb_push_back(logical_dfb_id_, num_entries);
#endif
}

inline void DataflowBuffer::wait_front_impl(uint16_t num_entries) {
#ifdef COMPILE_FOR_TRISC
    UNPACK((llk_wait_tiles(logical_dfb_id_, num_entries)));
#else
    cb_wait_front(logical_dfb_id_, num_entries);
#endif
}

inline void DataflowBuffer::pop_front_impl(uint16_t num_entries) {
#ifdef COMPILE_FOR_TRISC
    UNPACK((llk_pop_tiles(logical_dfb_id_, num_entries)));
#else
    cb_pop_front(logical_dfb_id_, num_entries);
#endif
}

#ifdef COMPILE_FOR_TRISC
inline uint32_t DataflowBuffer::get_tile_address(uint32_t tile_index) {
    uint32_t address = 0;

    UNPACK({
        uint32_t base_address = local_dfb_interface_.fifo_rd_ptr;
        uint32_t offset_address = local_dfb_interface_.fifo_page_size * tile_index;
        address = (base_address + offset_address) << 4;  // Convert to byte address

        mailbox_write(ckernel::ThreadId::MathThreadId, address);
        mailbox_write(ckernel::ThreadId::PackThreadId, address);
    })

    MATH(address = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
    PACK(address = mailbox_read(ckernel::ThreadId::UnpackThreadId);)

    return address;
}

inline uint32_t DataflowBuffer::read_tile_value(uint32_t tile_index, uint32_t element_offset) {
    uint32_t value = 0;

    UNPACK({
        uint32_t base_address = local_dfb_interface_.fifo_rd_ptr;
        uint32_t offset_address = local_dfb_interface_.fifo_page_size * tile_index;
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
inline uint32_t DataflowBuffer::get_tile_size() const { return ::get_tile_size(logical_dfb_id_); }
inline uint32_t DataflowBuffer::get_tile_hw() const { return ::get_tile_hw(logical_dfb_id_); }
inline DataFormat DataflowBuffer::get_dataformat() const { return ::get_dataformat(logical_dfb_id_); }
#endif

inline bool DataflowBuffer::pages_reservable_at_back(int32_t num_pages) const { return cb_pages_reservable_at_back(logical_dfb_id_, num_pages); }

inline bool DataflowBuffer::pages_available_at_front(int32_t num_pages) const { return cb_pages_available_at_front(logical_dfb_id_, num_pages); }

#endif

inline void DataflowBuffer::finish_impl() {}

inline uint32_t DataflowBuffer::get_write_ptr_impl() const { return local_dfb_interface_.fifo_wr_ptr; }

inline uint32_t DataflowBuffer::get_read_ptr_impl() const { return local_dfb_interface_.fifo_rd_ptr; }

}  // namespace experimental

#endif  // !ARCH_QUASAR
