// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_eager/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "tt_eager/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

#include "debug/dprint.h"

template<uint32_t tile_bytes, uint32_t num_readers>
constexpr uint32_t get_barrier_read_threshold() {
    return ((512 / num_readers) * (1024 + 128)) / tile_bytes;
}

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);  // batch size
    constexpr uint32_t PNHt = get_compile_time_arg_val(1);  // padded number of heads in tiles
    constexpr uint32_t PSt = get_compile_time_arg_val(2);  // padded layer length in tiles
    constexpr uint32_t St = get_compile_time_arg_val(3);  // full sequence length of kv cache in tiles
    constexpr uint32_t DHt = get_compile_time_arg_val(4);  // head dim
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(5);
    constexpr uint32_t scale_val = get_compile_time_arg_val(6);
    constexpr uint32_t num_cores_per_batch = get_compile_time_arg_val(7);  // num cores per batch
    constexpr uint32_t num_cores = get_compile_time_arg_val(8);  // num running cores in total
    constexpr uint32_t in0_receiver_semaphore_addr   = get_compile_time_arg_val(9);  // semaphore for reciever
    constexpr uint32_t cur_batch = get_compile_time_arg_val(10);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(11);  // number of chunks in K, where k_num_chunks*Sk_chunk_t = PSt
    constexpr uint32_t k_chunk_start = get_compile_time_arg_val(12);
    constexpr uint32_t k_chunk_end = get_compile_time_arg_val(13);
    constexpr bool is_out_sharded = get_compile_time_arg_val(14);

    const uint32_t out_addr  = get_arg_val<uint32_t>(0);

    constexpr uint32_t out_chunk_tiles = PNHt * DHt;
    constexpr uint32_t num_cores_to_wait = num_cores_per_batch-1;
    constexpr uint32_t num_tiles_to_wait = (out_chunk_tiles+2*PNHt)*num_cores_to_wait;

    constexpr bool is_dram = true;
    constexpr uint32_t cb_out = tt::CB::c_out4;
    constexpr uint32_t cb_intermed_out = tt::CB::c_out3;  // this cb holds the output intermediates from other worker cores
    constexpr uint32_t cb_out_o = tt::CB::c_out0;
    constexpr uint32_t cb_m_in = tt::CB::c_in6;
    constexpr uint32_t cb_l_in = tt::CB::c_in7;

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t tile_bytes_intermed = get_tile_size(cb_intermed_out);
    constexpr DataFormat data_format = get_dataformat(cb_out);

    const InterleavedAddrGenFast<is_dram> out_writer = {
        .bank_base_address = out_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    uint64_t intermed_l1_read_addr = get_noc_addr(get_read_ptr(cb_intermed_out));

    volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_receiver_semaphore_addr);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();
    uint32_t barrier_count = 0;

    constexpr uint32_t cb_scale_in = tt::CB::c_in4;
    constexpr uint32_t cb_identity_scale_in = tt::CB::c_in5;

    // generate and send scalar to compute
    generate_bcast_unary_scalar(cb_scale_in, scale_val);
    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);

    // DPRINT << "[Writer Reducer] Pushed statistics to copmute" << ENDL();

    if (k_chunk_end - k_chunk_start < k_num_chunks){
        // This indicates that there are computes done by other workers. Needs to wait for them and send to reducer's compute
        // Wait for compute to deliver output chunk, and write to compute again for reduction
        // data in cb_intermed_out is arranged as [o,m,l,o,m,l,...] with size (out_chunk_tiles + 2*PNHt)*num_cores_to_wait
        // wait on in0 semaphore value to become VALID (set by sender)
        // DPRINT << "[Writer Reducer] Waiting for semaphore to be set from "<< num_cores_to_wait << " cores" << ENDL();
        noc_semaphore_wait(in0_receiver_semaphore_addr_ptr, num_cores_to_wait);
        // DPRINT << "[Writer Reducer] Received signal that semaphore has set" << ENDL();
        // noc_semaphore_set(in0_receiver_semaphore_addr_ptr, 0);

        // cb_wait_front(cb_intermed_out, num_tiles_to_wait);
        constexpr uint32_t q_read_size = out_chunk_tiles*tile_bytes_intermed;
        constexpr uint32_t ml_read_size = PNHt*tile_bytes_intermed;
        // DPRINT << "[Writer Reducer] Received intermediate chunks from worker cores" << ENDL();
        // DPRINT << "[Writer Reducer] Sending intermediate chunks to compute" << ENDL();
        for(uint32_t block = 0; block < num_cores_per_batch; ++block) {

            // DPRINT << "[Writer Reducer] Iteration " << block << ENDL();
            cb_reserve_back(cb_out_o, out_chunk_tiles);
            cb_reserve_back(cb_m_in, PNHt);
            cb_reserve_back(cb_l_in, PNHt);
            // DPRINT << "[Writer Reducer] Reserved space in cb for Q, M, L" << ENDL();

            uint32_t q_write_ptr = get_read_ptr(cb_out_o);
            noc_async_read(intermed_l1_read_addr, q_write_ptr, q_read_size);
            intermed_l1_read_addr+=q_read_size;
            noc_async_read_barrier();
            cb_push_back(cb_out_o, out_chunk_tiles);
            // DPRINT << "[Writer Reducer] pushed Q" << ENDL();

            uint32_t m_write_ptr = get_read_ptr(cb_m_in);
            noc_async_read(intermed_l1_read_addr, m_write_ptr, ml_read_size);
            intermed_l1_read_addr+=ml_read_size;
            noc_async_read_barrier();
            cb_push_back(cb_m_in, PNHt);
            // DPRINT << "[Writer Reducer] pushed M" << ENDL();

            uint32_t l_write_ptr = get_read_ptr(cb_l_in);
            noc_async_read(intermed_l1_read_addr, l_write_ptr, ml_read_size);
            intermed_l1_read_addr+=ml_read_size;
            noc_async_read_barrier();
            cb_push_back(cb_l_in, PNHt);
            // DPRINT << "[Writer Reducer] pushed L" << ENDL();

            // DPRINT << "[Writer Reducer] Done iteration " << block << ENDL();
        }
        cb_pop_front(cb_intermed_out, num_tiles_to_wait);

        // DPRINT << "[Writer Reducer] Done sending intermediate chunks to compute" << ENDL();
    }

    // Offset for current batch
    const uint32_t out_batch_offset = cur_batch * out_chunk_tiles;

    // Write entire out into its corresponding batch
    uint32_t out_tile_id = out_batch_offset;
    cb_wait_front(cb_out, out_chunk_tiles);

    // DPRINT << "[Writer Reducer] recieved output chunk from reduce compute" << ENDL();
    if (! is_out_sharded){
        uint32_t l1_read_addr = get_read_ptr(cb_out);
        for (uint32_t tile = 0; tile < out_chunk_tiles; ++tile) {
            noc_async_write_tile(out_tile_id, out_writer, l1_read_addr);
            ++out_tile_id;
            l1_read_addr += tile_bytes;
            if (++barrier_count == barrier_threshold) {
                noc_async_writes_flushed();
                barrier_count = 0;
            }
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, out_chunk_tiles);

    // DPRINT << "[Writer Reducer] Wrote output chunk to memory. Done Reduce Writer" << ENDL();

}
