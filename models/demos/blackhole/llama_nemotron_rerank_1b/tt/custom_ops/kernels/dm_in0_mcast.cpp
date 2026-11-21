// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// in0 (activation) reader of the model's fused matmuls (RISCV_0, NOC 0), transposed grid. minimal_matmul's
// dm_in0_sender.cpp relays each in0 block from core to core along the N axis (store and forward: a core
// waits for its upstream neighbour, then for its downstream one, then writes); here the core at N index 0
// of each column reads the block from DRAM and multicasts it to the rest of the column in one write.
// Same block order as the stock kernel (in0 reuse of the last K block when striding N, the K order
// reversing every N block), so the compute kernels are unchanged. This kernel never writes the output.
//
// Handshake per block: each receiver reserves CB space, marks its receiver semaphore INVALID and
// increments the sender's sender semaphore; the sender, once the count reaches the number of receivers,
// multicasts the block to the same CB address and then multicasts VALID into the receiver semaphores.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "matmul_dataflow_common.hpp"

void kernel_main() {
    Noc noc;
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t padded_M_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t padded_K_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(6);
    constexpr uint32_t N_blocks_per_core = get_compile_time_arg_val(7);
    constexpr uint32_t in0_tile_size = get_compile_time_arg_val(8);
    Semaphore<> sender_semaphore(get_compile_time_arg_val(9));
    Semaphore<> receiver_semaphore(get_compile_time_arg_val(10));
    constexpr uint32_t is_injector_core = get_compile_time_arg_val(11);
    constexpr auto in0_args = TensorAccessorArgs<12>();

    uint32_t argidx = 0;
    const uint32_t in0_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t mcast_x0 = get_arg_val<uint32_t>(argidx++);
    const uint32_t mcast_y0 = get_arg_val<uint32_t>(argidx++);
    const uint32_t mcast_x1 = get_arg_val<uint32_t>(argidx++);
    const uint32_t mcast_y1 = get_arg_val<uint32_t>(argidx++);
    const uint32_t mcast_n = get_arg_val<uint32_t>(argidx++);
    const uint32_t sender_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t sender_y = get_arg_val<uint32_t>(argidx++);

    const auto in0_reader = TensorAccessor(in0_args, in0_addr);
    const TensorShape2D in0_shape(M_tiles, K_tiles, padded_M_tiles, padded_K_tiles);
    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t cb_in0_id = tt::CBIndex::c_0;
    CircularBuffer cb_in0(cb_in0_id);

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        const uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        const uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
        const uint32_t block_bytes = (m_tile_end - m_tile) * K_block_tiles * in0_tile_size;
        bool k_forward = true;
        bool reuse_block = false;
        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                if (reuse_block && k_block_iter == 0) {
                    // The compute kept the previous N block's last K block.
                    reuse_block = false;
                    continue;
                }
                const uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
                cb_in0.reserve_back(in0_block_num_tiles);
                const uint32_t addr = cb_in0.get_write_ptr();
                if constexpr (is_injector_core) {
                    read_in0_block_sync<M_block_tiles, K_block_tiles>(
                        in0_reader,
                        in0_shape,
                        cb_in0_id,
                        in0_tile_size,
                        m_tile,
                        m_tile_end,
                        k_block * K_block_tiles,
                        (k_block + 1) * K_block_tiles);
                    // Hand the block to this core's compute before multicasting it.
                    cb_in0.push_back(in0_block_num_tiles);
                    if (mcast_n > 0) {
                        sender_semaphore.wait(mcast_n);
                        sender_semaphore.set(0);
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(addr),
                            MulticastEndpoint{},
                            block_bytes,
                            mcast_n,
                            {},
                            {.noc_x_start = mcast_x0,
                             .noc_y_start = mcast_y0,
                             .noc_x_end = mcast_x1,
                             .noc_y_end = mcast_y1,
                             .addr = addr},
                            false);
                        noc.async_writes_flushed();
                        receiver_semaphore.set(VALID);
                        receiver_semaphore.set_multicast(noc, mcast_x0, mcast_y0, mcast_x1, mcast_y1, mcast_n);
                    }
                } else {
                    receiver_semaphore.set(INVALID);
                    sender_semaphore.up(noc, sender_x, sender_y, 1);
                    receiver_semaphore.wait(VALID);
                    cb_in0.push_back(in0_block_num_tiles);
                }
            }
            k_forward = !k_forward;
            reuse_block = true;
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
