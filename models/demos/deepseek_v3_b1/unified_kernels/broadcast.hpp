// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include <array>
#include <cstdint>
#include <utility>

using address_t = uint32_t;

#elif defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#include <cstdint>
#include <utility>
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#if defined(ENABLE_SOCKET_READER)
#include "api/socket_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
using tt::data_movement::common::tt_memmove;
#endif

using address_t = uint32_t;
#endif

namespace deepseek_b1_ops {

// Unified kernel for CCL Broadcast operation
struct Broadcast {
    static constexpr uint32_t MAX_NUM_LINKS = 2;

    // ========================================================================
    // Runtime args structs - different layout per RISC
    // ========================================================================
    template <uint32_t cb0Id, uint32_t NumPagesToRead, uint32_t isRoot, uint32_t useSocket = 0>
    struct ReaderCTArgs {
        static constexpr uint32_t cb0_id = cb0Id;
        static constexpr uint32_t num_pages_to_read = NumPagesToRead;
        static constexpr bool is_root = isRoot != 0;
        static constexpr bool use_socket = useSocket != 0;
    };

    struct ReaderArgs {
        uint32_t socket_config_addr = 0;
        uint32_t socket_page_size = 0;
        uint32_t socket_num_pages = 0;
    };

    template <
        uint32_t cb0Id,
        uint32_t NumPagesToRead,
        uint32_t tensorPageSize,
        uint32_t numNeighbors,
        uint32_t numLinks,
        uint32_t isRoot,
        uint32_t chunkSizeBytes,
        uint32_t lastChunkSizeBytes,
        uint32_t numChunks,
        uint32_t cbOutId = 0,
        uint32_t outNumTiles = 0>
    struct WriterCTArgs {
        static constexpr uint32_t cb0_id = cb0Id;
        static constexpr uint32_t num_pages_to_read = NumPagesToRead;
        static constexpr uint32_t tensor0_page_size = tensorPageSize;
        static constexpr uint32_t num_neighbors = numNeighbors;
        static constexpr uint32_t num_links = numLinks;
        static constexpr uint32_t num_connections = num_neighbors * num_links;
        static constexpr bool is_root = isRoot != 0;
        static constexpr uint32_t chunk_size_bytes = chunkSizeBytes;
        static constexpr uint32_t last_chunk_size_bytes = lastChunkSizeBytes;
        static constexpr uint32_t num_chunks = numChunks;
        static constexpr uint32_t cb_out_id = cbOutId;
        static constexpr uint32_t out_num_tiles = outNumTiles;
        static_assert(num_links <= Broadcast::MAX_NUM_LINKS, "num_links exceeds MAX_NUM_LINKS");
        static_assert(num_chunks > 0, "num_chunks must be greater than 0");
    };
    struct WriterArgs {
        uint32_t tensor_address0;
        uint32_t my_noc_x;
        uint32_t my_noc_y;
        std::array<uint32_t, MAX_NUM_LINKS> sem_bank_addrs;
        uint32_t per_core_rta_arg_idx_offset = 0;
        uint32_t per_core_rta_num_args = 0;
    };

    // TRISC args - not used for CCL broadcast op
    struct ComputeArgs {};
    struct ComputeCTArgs {};

    using RTArgs = unified_kernels::SelectByRISCV<WriterArgs, ReaderArgs, ComputeArgs>;

    template <typename CTArgs, bool IsWorkerCore>
    class Op {
    public:
        void operator()(const RTArgs& args) {
            if constexpr (IsWorkerCore) {
                open_connections_impl(args, /*reset_header_pool=*/true);
                impl(args);
            }
        }

        void open_connections(const RTArgs& args, bool reset_header_pool = true) {
            if constexpr (IsWorkerCore) {
                open_connections_impl(args, reset_header_pool);
            }
        }

        void run(const RTArgs& args) {
            if constexpr (IsWorkerCore) {
                impl(args);
            }
        }

    private:
#if defined(COMPILE_FOR_NCRISC)
        static constexpr uint8_t worker_to_fabric_noc = 0;
        static_assert(
            noc_mode == DM_DYNAMIC_NOC || worker_to_fabric_noc == noc_index, "Custom noc requires DM_DYNAMIC_NOC");
        std::array<tt::tt_fabric::WorkerToFabricEdmSender, CTArgs::num_connections> connections;
        std::array<volatile PACKET_HEADER_TYPE*, CTArgs::num_connections> headers;
        uint64_t dst_noc_base = 0;
        std::array<uint64_t, CTArgs::num_links> sem_nocs;
        std::array<volatile tt_l1_ptr uint32_t*, CTArgs::num_links> sem_ptrs;
#endif

        void open_connections_impl([[maybe_unused]] const RTArgs& args, [[maybe_unused]] bool reset_header_pool) {
#if defined(COMPILE_FOR_NCRISC)
            if constexpr (IsWorkerCore) {
                if (reset_header_pool) {
                    PacketHeaderPool::reset();
                }

                dst_noc_base = get_noc_addr(args.my_noc_x, args.my_noc_y, args.tensor_address0, worker_to_fabric_noc);
                for (uint32_t link_idx = 0; link_idx < CTArgs::num_links; link_idx++) {
                    sem_nocs[link_idx] = safe_get_noc_addr(
                        args.my_noc_x, args.my_noc_y, args.sem_bank_addrs[link_idx], worker_to_fabric_noc);
                    sem_ptrs[link_idx] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.sem_bank_addrs[link_idx]);
                }

                size_t arg_idx = args.per_core_rta_arg_idx_offset;
                for (uint32_t neighbor_idx = 0; neighbor_idx < CTArgs::num_neighbors; neighbor_idx++) {
                    for (uint32_t link_idx = 0; link_idx < CTArgs::num_links; link_idx++) {
                        const uint32_t connection_idx = neighbor_idx * CTArgs::num_links + link_idx;
                        connections[connection_idx] =
                            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(
                                arg_idx);
                        connections[connection_idx].open_start();
                    }
                }

                for (uint32_t neighbor_idx = 0; neighbor_idx < CTArgs::num_neighbors; neighbor_idx++) {
                    const uint32_t dst_mesh_id = get_arg_val<uint32_t>(arg_idx++);
                    const uint32_t dst_chip_id = get_arg_val<uint32_t>(arg_idx++);
                    const auto connection_direction = get_next_hop_router_direction(dst_mesh_id, dst_chip_id);
                    for (uint32_t link_idx = 0; link_idx < CTArgs::num_links; link_idx++) {
                        const uint32_t connection_idx = neighbor_idx * CTArgs::num_links + link_idx;
                        headers[connection_idx] = PacketHeaderPool::allocate_header();
                        fabric_set_single_hop_unicast_route_from_direction(
                            headers[connection_idx], connection_direction, dst_chip_id, dst_mesh_id);
                        headers[connection_idx]->to_noc_fused_unicast_write_atomic_inc(
                            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                                dst_noc_base, sem_nocs[link_idx], 1, false},
                            CTArgs::chunk_size_bytes);
                    }
                }

                for (uint32_t connection_idx = 0; connection_idx < CTArgs::num_connections; connection_idx++) {
                    connections[connection_idx].open_finish();
                }
            }
#endif
        }

        void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC - bcast reader
            // ================================================================
            if constexpr (IsWorkerCore) {
                if (CTArgs::is_root) {
#if defined(ENABLE_SOCKET_READER)
                    if constexpr (CTArgs::use_socket) {
                        static_assert(noc_mode == DM_DYNAMIC_NOC);
                        SocketReceiverInterface recv = create_receiver_socket_interface(args.socket_config_addr);
                        set_receiver_socket_page_size(recv, args.socket_page_size);
                        socket_wait_for_pages(recv, args.socket_num_pages);
                        cb_reserve_back(CTArgs::cb0_id, CTArgs::num_pages_to_read);
                        invalidate_l1_cache();
                        noc_async_read(
                            get_noc_addr(recv.read_ptr),
                            get_write_ptr(CTArgs::cb0_id),
                            args.socket_page_size,
                            1 - noc_index);
                        noc_async_read_barrier(1 - noc_index);
                        cb_push_back(CTArgs::cb0_id, CTArgs::num_pages_to_read);
                        socket_pop_pages(recv, args.socket_num_pages);
                        socket_notify_sender(recv, 1 - noc_index);
                        update_socket_config(recv);
                    } else {
#endif
                        cb_reserve_back(CTArgs::cb0_id, CTArgs::num_pages_to_read);
                        cb_push_back(CTArgs::cb0_id, CTArgs::num_pages_to_read);
#if defined(ENABLE_SOCKET_READER)
                    }
#endif
                }
            }

#elif defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC - bcast writer
            // ================================================================
            if constexpr (IsWorkerCore) {
                std::array<uint32_t, CTArgs::num_connections> cached_free_write_slots = {};

                auto refill_free_write_slots = [&](uint32_t connection_idx) {
                    do {
                        cached_free_write_slots[connection_idx] =
                            connections[connection_idx].get_num_free_write_slots();
                    } while (cached_free_write_slots[connection_idx] == 0);
                };

                auto send_single_chunk = [&](uint32_t connection_idx,
                                             uint32_t src_base_addr) __attribute__((always_inline)) {
                    connections[connection_idx].wait_for_empty_write_slot();
                    connections[connection_idx].send_current_slot_non_blocking(
                        src_base_addr,
                        CTArgs::last_chunk_size_bytes,
                        reinterpret_cast<uint32_t>(headers[connection_idx]));
                };

                auto send_multi_chunk = [&](uint32_t connection_idx,
                                            uint32_t src_base_addr,
                                            uint32_t chunk_idx,
                                            uint32_t size) __attribute__((always_inline)) {
                    uint32_t chunk_offset = chunk_idx * CTArgs::chunk_size_bytes;
                    if constexpr (CTArgs::last_chunk_size_bytes != CTArgs::chunk_size_bytes) {
                        if (size != CTArgs::chunk_size_bytes) {
                            headers[connection_idx]->set_payload_size_bytes(size);
                        }
                    }
                    headers[connection_idx]->set_fused_unicast_write_atomic_inc_write_noc_address(
                        dst_noc_base + chunk_offset);
                    if (cached_free_write_slots[connection_idx] == 0) {
                        refill_free_write_slots(connection_idx);
                    }
                    connections[connection_idx].send_current_slot_non_blocking(
                        src_base_addr + chunk_offset, size, reinterpret_cast<uint32_t>(headers[connection_idx]));
                    cached_free_write_slots[connection_idx]--;
                };

                std::array<uint32_t, CTArgs::num_links> link_counters = {};
                auto forward_chunks = [&](uint32_t src_base_addr, auto&& wait_for_link_chunk) {
                    if constexpr (CTArgs::num_chunks == 1) {
                        constexpr uint32_t single_chunk_link = 0;
                        link_counters[single_chunk_link]++;
                        wait_for_link_chunk(single_chunk_link, link_counters[single_chunk_link]);

                        for (uint32_t neighbor_idx = 0; neighbor_idx < CTArgs::num_neighbors; neighbor_idx++) {
                            const uint32_t connection_idx = neighbor_idx * CTArgs::num_links + single_chunk_link;
                            send_single_chunk(connection_idx, src_base_addr);
                        }
                        return;
                    }

                    uint32_t current_link = 0;

                    for (uint32_t chunk_idx = 0; chunk_idx < CTArgs::num_chunks; chunk_idx++) {
                        link_counters[current_link]++;
                        wait_for_link_chunk(current_link, link_counters[current_link]);

                        const uint32_t chunk_size = (chunk_idx < CTArgs::num_chunks - 1)
                                                        ? CTArgs::chunk_size_bytes
                                                        : CTArgs::last_chunk_size_bytes;

                        for (uint32_t neighbor_idx = 0; neighbor_idx < CTArgs::num_neighbors; neighbor_idx++) {
                            const uint32_t connection_idx = neighbor_idx * CTArgs::num_links + current_link;
                            send_multi_chunk(connection_idx, src_base_addr, chunk_idx, chunk_size);
                        }

                        if (++current_link == CTArgs::num_links) {
                            current_link = 0;
                            // flush only when about to reuse a packet header
                            if constexpr (CTArgs::num_neighbors > 0) {
                                if (chunk_idx + 1 < CTArgs::num_chunks) {
                                    noc_async_writes_flushed(worker_to_fabric_noc);
                                }
                            }
                        }
                    }
                };

                // Roles:
                // - Root node: no semaphore wait, sources chunks from local CB read pointer.
                // - Non-root node: waits for chunk arrival and forwards from local output tensor storage.
                //   Non-root can be either:
                //   * forwarding node (num_neighbors > 0), or
                //   * leaf node (num_neighbors == 0), where forwarding loops are no-ops.
                // In the leaf case, num_links remains configured (> 0), wait/reset semantics are still
                // preserved for non-root nodes, and no fabric send occurs due to zero neighbors.
                if constexpr (CTArgs::out_num_tiles != 0) {
                    cb_reserve_back(CTArgs::cb_out_id, CTArgs::out_num_tiles);
                }
                if constexpr (CTArgs::is_root) {
                    cb_wait_front(CTArgs::cb0_id, CTArgs::num_pages_to_read);
                    const uint32_t src = get_read_ptr(CTArgs::cb0_id);
                    if (src != args.tensor_address0) {
                        constexpr uint32_t tensor_size_bytes = CTArgs::tensor0_page_size * CTArgs::num_pages_to_read;
                        noc_async_write(src, dst_noc_base, tensor_size_bytes);
                    }
                    if constexpr (CTArgs::out_num_tiles != 0) {
                        ASSERT(src == args.tensor_address0);
                        cb_push_back(CTArgs::cb_out_id, CTArgs::out_num_tiles);
                    }
                    auto no_wait = [&](uint32_t, uint32_t) {};
                    forward_chunks(src, no_wait);
                    cb_pop_front(CTArgs::cb0_id, CTArgs::num_pages_to_read);
                } else {
                    const uint32_t src = args.tensor_address0;
                    auto sem_wait = [&](uint32_t link_idx, uint32_t link_threshold) {
                        noc_semaphore_wait_min(sem_ptrs[link_idx], link_threshold);
                    };
                    forward_chunks(src, sem_wait);
                    if constexpr (CTArgs::out_num_tiles != 0) {
                        cb_push_back(CTArgs::cb_out_id, CTArgs::out_num_tiles);
                    }
                    for (uint32_t link_idx = 0; link_idx < CTArgs::num_links; link_idx++) {
                        if (link_counters[link_idx] > 0) {
                            unified_kernels::semaphore_dec(sem_ptrs[link_idx], link_counters[link_idx]);
                        }
                    }
                }

                for (uint32_t i = 0; i < CTArgs::num_connections; i++) {
                    connections[i].close();
                }

                noc_async_full_barrier();
            }
#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC - No-op (CCL broadcast is dataflow only)
            // ================================================================
#endif
        }
    };  // class Op

};  // struct Broadcast

}  // namespace deepseek_b1_ops
