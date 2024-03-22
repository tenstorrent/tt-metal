// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/assert.h"
#include "tt_eager/tt_dnn/op_library/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_eager/tt_dnn/op_library/ccl/ccl_common.hpp"

using ccl::ShardType;

FORCE_INLINE void validate_sane_transaction_counters() {
    ASSERT (NOC_STATUS_READ_REG(noc_index, NIU_MST_WR_ACK_RECEIVED) != 0);
    ASSERT (NOC_STATUS_READ_REG(noc_index, NIU_MST_NONPOSTED_WR_REQ_SENT) != 0);
    ASSERT(noc_nonposted_writes_num_issued[noc_index] != 0);
    ASSERT(noc_nonposted_writes_acked[noc_index] != 0);
}

FORCE_INLINE void validate_sane_transaction_counters_rw() {
    ASSERT (NOC_STATUS_READ_REG(noc_index, NIU_MST_WR_ACK_RECEIVED) != 0);
    ASSERT (NOC_STATUS_READ_REG(noc_index, NIU_MST_NONPOSTED_WR_REQ_SENT) != 0);
    ASSERT(noc_nonposted_writes_num_issued[noc_index] != 0);
    ASSERT(noc_nonposted_writes_acked[noc_index] != 0);
}


template <ShardType TYPE>
struct ShardAddrGen final {
    ShardAddrGen()=default;

    FORCE_INLINE static void build_with_placement_new(ShardAddrGen* placement_new_address, const uint32_t arg_index) {
        ccl::ShardAddrGenArgs<false> input_args;

        uint32_t curr_arg_index = arg_index;
        input_args.is_clockwise = bool(get_arg_val<uint32_t>(curr_arg_index++) == 1);
        input_args.shard_size_in_bytes = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.chunks_per_core_before_advance = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.shards_start_address = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.starting_core_index = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.starting_chunk_into_shard = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.num_dest_cores = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.dest_cores = reinterpret_cast<ccl::WorkerXY*>(get_arg_addr(curr_arg_index));
        curr_arg_index += input_args.num_dest_cores;

        ASSERT(input_args.shard_size_in_bytes != ccl::ShardAddrGenArgs<true>::UNINITIALIZED_VALUE);
        ASSERT(input_args.chunks_per_core_before_advance != ccl::ShardAddrGenArgs<true>::UNINITIALIZED_VALUE);
        ASSERT(input_args.shards_start_address != ccl::ShardAddrGenArgs<true>::UNINITIALIZED_VALUE);
        ASSERT(input_args.starting_core_index != ccl::ShardAddrGenArgs<true>::UNINITIALIZED_VALUE);
        ASSERT(input_args.starting_chunk_into_shard != ccl::ShardAddrGenArgs<true>::UNINITIALIZED_VALUE);
        ASSERT(input_args.num_dest_cores != ccl::ShardAddrGenArgs<true>::UNINITIALIZED_VALUE);

        ASSERT(curr_arg_index - arg_index == input_args.get_expected_num_args());

        new (placement_new_address) ShardAddrGen(
            curr_arg_index - arg_index,
            input_args);
    }

    // This addr gen will dump all tiles from an input shard contiguously, and dump the
    // next input shard contiguously after it. This approach depends on a follow up
    //
    ShardAddrGen(
        uint8_t num_args_consumed,
        ccl::ShardAddrGenArgs<false> const& input_args) :
        dest_cores(input_args.dest_cores),
        num_dest_cores(input_args.num_dest_cores),
        shards_start_address(input_args.shards_start_address),
        shard_size_in_bytes(input_args.shard_size_in_bytes),
        chunks_per_core_before_advance(input_args.chunks_per_core_before_advance),
        curr_worker_index(input_args.starting_core_index),
        curr_core_chunk_index(input_args.starting_chunk_into_shard),
        num_args_consumed(num_args_consumed),
        is_clockwise(input_args.is_clockwise),
        completed_core_wrap(false){};

    static_assert(
        TYPE == ShardType::Width || TYPE == ShardType::Height || TYPE == ShardType::Block, "Invalid ShardType");

    // Clockwise vs counter clockwise only affects worker core traversal order (relative to canonical order). Since the
    // dest core list is a configurable list, we will, for now, require the host side kernel config code to produce the
    // correc order per worker
    FORCE_INLINE void advance() {
        if constexpr (TYPE == ShardType::Width or TYPE == ShardType::Height) {
            if (this->is_clockwise) {
                // Read inputs in reverse order too
                bool do_chunk_wrap = this->curr_core_chunk_index == 0;
                if (do_chunk_wrap) {
                    bool do_core_wrap = this->curr_worker_index == 0;
                    this->curr_core_chunk_index = this->chunks_per_core_before_advance - 1;
                    if (do_core_wrap) {
                        completed_core_wrap = true;
                        this->curr_worker_index = this->num_dest_cores - 1;
                    } else {
                        this->curr_worker_index--;
                    }
                } else {
                    this->curr_core_chunk_index--;
                }


            } else {
                // If I analyzed it properly, then we should never be wrapping back to the first dest core *and* still have
                // tiles/input shards to move
                this->curr_core_chunk_index++;
                bool do_chunk_wrap = this->curr_core_chunk_index == this->chunks_per_core_before_advance;
                if (do_chunk_wrap) {
                    this->curr_core_chunk_index = 0;
                    this->curr_worker_index++;
                    bool do_core_wrap = this->curr_worker_index == this->num_dest_cores;
                    if (do_core_wrap) {
                        this->curr_worker_index = 0;
                        completed_core_wrap = true;
                    }
                }
            }
        } else {
            // Unsupported
            ASSERT(false);
        }
    }

    [[nodiscard]] FORCE_INLINE ccl::WorkerXY get_next_noc_xy_core() const {
        ASSERT(this->curr_worker_index < this->num_dest_cores);
        return this->dest_cores[this->curr_worker_index];
    }

    [[nodiscard]] FORCE_INLINE uint64_t get_next_noc_addr_and_advance() {
        if constexpr (TYPE == ShardType::Width) {
            ccl::WorkerXY dest_worker = this->get_next_noc_xy_core();
            uint32_t curr_address = this->shards_start_address + this->curr_core_chunk_index * this->shard_size_in_bytes;
            ASSERT(curr_address + this->shard_size_in_bytes <= 1499136); // L1 wraparound - oops!
            this->advance();
            return get_noc_addr(dest_worker.x, dest_worker.y, curr_address);
        } else {
            ASSERT(false);
            // Unsupported
            return 0;
        }
    }

    [[nodiscard]] FORCE_INLINE uint32_t get_shard_size_in_bytes() const { return this->shard_size_in_bytes; }

    [[nodiscard]] FORCE_INLINE uint32_t get_num_dest_cores() const { return this->num_dest_cores; }
    [[nodiscard]] FORCE_INLINE uint32_t get_chunks_per_core_before_advance() const {
        return this->chunks_per_core_before_advance;
    }
    [[nodiscard]] FORCE_INLINE uint32_t get_num_args_consumed() const { return this->num_args_consumed;}

    ccl::WorkerXY* dest_cores;
    uint32_t num_dest_cores;
    uint32_t shards_start_address;
    uint32_t shard_size_in_bytes;
    uint32_t chunks_per_core_before_advance;
    uint32_t curr_worker_index;
    uint32_t curr_core_chunk_index;
    uint8_t num_args_consumed;
    bool is_clockwise;
    bool completed_core_wrap;
};

FORCE_INLINE void push_filler_pages_to_cb(const uint32_t& cb_id, uint32_t num_pages) {
    ASSERT(num_pages < cb_interface[cb_id].fifo_num_pages);
    cb_reserve_back(cb_id, num_pages);
    cb_push_back(cb_id, num_pages);
}
FORCE_INLINE void pop_filler_pages_from_cb(const uint32_t& cb_id, uint32_t num_pages) {
    ASSERT(num_pages < cb_interface[cb_id].fifo_num_pages);
    cb_wait_front(cb_id, num_pages);
    cb_pop_front(cb_id, num_pages);
}


FORCE_INLINE void fetch_chunk(
    const uint32_t& cb_id, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_read_addr) {
    cb_reserve_back(cb_id, num_pages);
    uint32_t l1_write_addr = get_write_ptr(cb_id);
    noc_async_read(remote_l1_read_addr, l1_write_addr, page_size * num_pages);
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}
FORCE_INLINE void fetch_chunk_sharded(
    const uint32_t& cb_id, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_read_addr) {
    cb_reserve_back(cb_id, num_pages);
    uint32_t l1_write_addr = get_write_ptr(cb_id);
    noc_async_read(remote_l1_read_addr, l1_write_addr, num_pages * page_size);
    validate_sane_transaction_counters();
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}

FORCE_INLINE void send_chunk(
    const uint32_t& cb_id, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_write_addr) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    noc_async_write(l1_read_addr, remote_l1_write_addr, page_size * num_pages);
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}
FORCE_INLINE void send_chunk_sharded(
    const uint32_t& cb_id, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_write_addr) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    noc_async_write(l1_read_addr, remote_l1_write_addr, page_size * num_pages);
    validate_sane_transaction_counters();
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}

template <ShardType T>
FORCE_INLINE void write_and_send_chunk_sharded(
    const uint32_t& cb_id, ShardAddrGen<T>& addr_gen, uint32_t num_pages, uint64_t remote_eth_l1_write_addr) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    uint64_t dest_worker_noc_addr = addr_gen.get_next_noc_addr_and_advance();
    noc_async_write(l1_read_addr, remote_eth_l1_write_addr, addr_gen.get_shard_size_in_bytes());
    noc_async_write(l1_read_addr, dest_worker_noc_addr, addr_gen.get_shard_size_in_bytes());
    validate_sane_transaction_counters();
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}
template <typename AddrGen>
FORCE_INLINE void write_and_send_chunk(uint32_t& output_page_idx, uint32_t& col_idx, uint32_t& row_idx, const uint32_t& cb_id, const AddrGen& d, const uint32_t num_cols, const uint32_t num_rows, const uint32_t& col_offset, const uint32_t& row_offset, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_write_addr, uint64_t eth_l1_sender_semaphore_addr) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    noc_async_write(l1_read_addr, remote_l1_write_addr, page_size * num_pages);
    noc_semaphore_inc(eth_l1_sender_semaphore_addr, 1);
    // TODO: do eth semaphore inc here
    for (uint32_t i = 0; i < num_pages; ++i) {
#ifdef RM_INTERLEAVED
        uint64_t dst_noc_addr = get_noc_addr(output_page_idx, d);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
        output_page_idx++;
        row_idx++;
        if (row_idx == num_rows) {
            row_idx = 0;
            output_page_idx += row_offset;
        }
#elif defined TILE_INTERLEAVED || defined SHARDED
        noc_async_write_tile(output_page_idx, d, l1_read_addr);
        output_page_idx++;
        col_idx++;
        if (col_idx == num_cols) {
            output_page_idx += col_offset;
            col_idx = 0;
            row_idx++;
            if (row_idx == num_rows) {
                row_idx = 0;
                output_page_idx += row_offset;
            }
        }
#endif
        l1_read_addr += page_size;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}

template <ShardType T>
FORCE_INLINE void write_chunk_sharded(const uint32_t& cb_id, ShardAddrGen<T>& addr_gen, uint32_t num_pages) {
    for (uint32_t i = 0; i < num_pages; ++i) {
        cb_wait_front(cb_id, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id);
        uint64_t dest_worker_noc_addr = addr_gen.get_next_noc_addr_and_advance();

        noc_async_write(l1_read_addr, dest_worker_noc_addr, addr_gen.get_shard_size_in_bytes());

        // validate_sane_transaction_counters();
        validate_sane_transaction_counters_rw();
        noc_async_write_barrier();
        cb_pop_front(cb_id, 1);
    }
}
template <typename AddrGen>
FORCE_INLINE void write_chunk(uint32_t& output_page_idx, uint32_t& col_idx, uint32_t& row_idx, const uint32_t& cb_id, const AddrGen& d, const uint32_t& num_cols, const uint32_t& num_rows, const uint32_t& col_offset, const uint32_t& row_offset, const uint32_t& num_pages, const uint32_t& page_size) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    for (uint32_t i = 0; i < num_pages; ++i) {
        #ifdef RM_INTERLEAVED
        uint64_t dst_noc_addr = get_noc_addr(output_page_idx, d);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
        output_page_idx++;
        row_idx++;
        if (row_idx == num_rows) {
            row_idx = 0;
            output_page_idx += row_offset;
        }
        #elif defined TILE_INTERLEAVED
        noc_async_write_tile(output_page_idx, d, l1_read_addr);
        output_page_idx++;
        col_idx++;
        if (col_idx == num_cols) {
            output_page_idx += col_offset;
            col_idx = 0;
            row_idx++;
            if (row_idx == num_rows) {
                row_idx = 0;
                output_page_idx += row_offset;
            }
        }
        #endif
        l1_read_addr += page_size;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}

template <ShardType T>
FORCE_INLINE void read_chunk_from_input_tensor_sharded(
    const uint32_t& cb_id, ShardAddrGen<T>& addr_gen, uint32_t num_pages) {
    cb_reserve_back(cb_id, num_pages);
    uint32_t local_l1_read_dest_addr = get_write_ptr(cb_id);
    uint64_t src_noc_addr = addr_gen.get_next_noc_addr_and_advance();
    noc_async_read(src_noc_addr, local_l1_read_dest_addr, addr_gen.get_shard_size_in_bytes());
    validate_sane_transaction_counters();
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}
// read chunk from input tensor (local chip)
template <typename AddrGen>
FORCE_INLINE void read_chunk_from_input_tensor(uint32_t& input_page_idx, const uint32_t& cb_id, const AddrGen& s, const uint32_t& num_pages, const uint32_t& page_size) {
    const uint32_t end_read_idx = input_page_idx + num_pages;
    cb_reserve_back(cb_id, num_pages);
    uint32_t local_l1_read_addr = get_write_ptr(cb_id);
    for (; input_page_idx < end_read_idx; ++input_page_idx) {
        #ifdef RM_INTERLEAVED
        uint64_t src_noc_addr = get_noc_addr(input_page_idx, s);
        noc_async_read(src_noc_addr, local_l1_read_addr, page_size);
        #elif defined TILE_INTERLEAVED
        noc_async_read_tile(input_page_idx, s, local_l1_read_addr);
        #endif
        local_l1_read_addr += page_size;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}

// Same function - just different address generators? Commonize later
template <ShardType T>
FORCE_INLINE void read_chunk_from_output_tensor_sharded(
    const uint32_t& cb_id, ShardAddrGen<T>& addr_gen, uint32_t num_pages) {
    cb_reserve_back(cb_id, num_pages);
    uint32_t local_l1_read_dest_addr = get_write_ptr(cb_id);
    uint64_t src_noc_addr = addr_gen.get_next_noc_addr_and_advance();
    noc_async_read(src_noc_addr, local_l1_read_dest_addr, addr_gen.get_shard_size_in_bytes());
    validate_sane_transaction_counters();
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}
// read chunk from output tensor (local chip)
template <typename AddrGen>
FORCE_INLINE void read_chunk_from_output_tensor(uint32_t& input_page_idx, uint32_t& col_idx, uint32_t& row_idx, const uint32_t& cb_id, const AddrGen& s, const uint32_t& num_cols, const uint32_t& num_rows, const uint32_t& col_offset, const uint32_t& row_offset, const uint32_t& num_pages, const uint32_t& page_size) {
    cb_reserve_back(cb_id, num_pages);
    uint32_t local_l1_read_addr = get_write_ptr(cb_id);
    for (uint32_t i = 0; i < num_pages; ++i) {
        #ifdef RM_INTERLEAVED
        uint64_t src_noc_addr = get_noc_addr(input_page_idx, s);
        noc_async_read(src_noc_addr, local_l1_read_addr, page_size);
        input_page_idx++;
        row_idx++;
        if (row_idx == num_rows) {
            row_idx = 0;
            input_page_idx += row_offset;
        }
        #elif defined TILE_INTERLEAVED
        noc_async_read_tile(input_page_idx, s, local_l1_read_addr);
        input_page_idx++;
        col_idx++;
        if (col_idx == num_cols) {
            input_page_idx += col_offset;
            col_idx = 0;
            row_idx++;
            if (row_idx == num_rows) {
                row_idx = 0;
                input_page_idx += row_offset;
            }
        }
        #endif
        local_l1_read_addr += page_size;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}
