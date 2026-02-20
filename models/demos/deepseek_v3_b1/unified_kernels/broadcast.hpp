// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"

using address_t = uint32_t;
using namespace tt::tt_fabric::linear::experimental;

#elif defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#include <cstdint>
#include <utility>
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

using address_t = uint32_t;
#endif

namespace deepseek_b1_ops {

// Unified kernel for CCL Broadcast operation
struct Broadcast {
    // ========================================================================
    // Runtime args structs - different layout per RISC
    // ========================================================================
    template <uint32_t cb0Id, uint32_t NumPagesToRead, uint32_t isSender>
    struct ReaderCTArgs {
        static constexpr uint32_t cb0_id = cb0Id;
        static constexpr uint32_t num_pages_to_read = NumPagesToRead;
        static constexpr uint32_t is_sender = isSender;
    };

    struct ReaderArgs {};

    template <
        uint32_t cb0Id,
        uint32_t NumPagesToRead,
        uint32_t tensorPageSize,
        uint32_t numTargetsForwardDirection,
        uint32_t numTargetsBackwardDirection,
        uint32_t isSender,
        uint32_t coreNocX,
        uint32_t coreNocY,
        uint32_t isSecondarySender,
        uint32_t hasSecondaryTarget,
        uint32_t startDistanceInHopsForward,
        uint32_t rangeHopsForward,
        uint32_t startDistanceInHopsBackward,
        uint32_t rangeHopsBackward>
    struct WriterCTArgs {
        static constexpr uint32_t cb0_id = cb0Id;
        static constexpr uint32_t num_pages_to_read = NumPagesToRead;
        static constexpr uint32_t tensor0_page_size = tensorPageSize;
        static constexpr uint32_t num_targets_forward_direction = numTargetsForwardDirection;
        static constexpr uint32_t num_targets_backward_direction = numTargetsBackwardDirection;
        static constexpr uint32_t is_sender = isSender;
        static constexpr uint32_t core_noc_x = coreNocX;
        static constexpr uint32_t core_noc_y = coreNocY;
        static constexpr uint32_t is_secondary_sender = isSecondarySender;
        static constexpr uint32_t has_secondary_target = hasSecondaryTarget;
        static constexpr uint32_t start_distance_in_hops_forward = startDistanceInHopsForward;
        static constexpr uint32_t range_hops_forward = rangeHopsForward;
        static constexpr uint32_t start_distance_in_hops_backward = startDistanceInHopsBackward;
        static constexpr uint32_t range_hops_backward = rangeHopsBackward;
    };
    struct WriterArgs {
        uint32_t tensor_address0;
        uint32_t out_ready_sem_bank_addr;
        uint32_t wait_output_semaphore;
        uint32_t reset_global_semaphore;
        uint32_t out_ready_sem_noc0_x;
        uint32_t out_ready_sem_noc0_y;
        uint32_t out_ready_sem_wait_value;
        uint32_t barrier_sem;
        uint32_t barrier_sem_noc0_x;
        uint32_t barrier_sem_noc0_y;
        uint32_t ring_index;
        uint32_t secondary_sync_sem;
        uint32_t num_connections;
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
                impl(args);
            }
        }

    private:
        void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC - bcast reader
            // ================================================================
            if constexpr (IsWorkerCore) {
                if (CTArgs::is_sender) {
                    cb_reserve_back(CTArgs::cb0_id, CTArgs::num_pages_to_read);
                    cb_push_back(CTArgs::cb0_id, CTArgs::num_pages_to_read);
                }
            }

#elif defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC - bcast writer
            // ================================================================
            if constexpr (IsWorkerCore) {
                constexpr uint32_t num_primary_connections = (CTArgs::start_distance_in_hops_forward > 0 ? 1 : 0) +
                                                             (CTArgs::start_distance_in_hops_backward > 0 ? 1 : 0);

                constexpr uint32_t secondary_connection_idx = num_primary_connections;
                size_t arg_for_fab = 0;

                auto sem_route_id = PacketHeaderPool::allocate_header_n(num_primary_connections);
                auto fused_route_id = PacketHeaderPool::allocate_header_n(num_primary_connections);
                // Allocate separate route for secondary axis unicast (if applicable)
                auto secondary_route_id = CTArgs::has_secondary_target ? PacketHeaderPool::allocate_header_n(1) : 0;

                tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;

                if constexpr (CTArgs::is_secondary_sender || CTArgs::is_sender) {
                    open_connections(fabric_connection, args.num_connections, arg_for_fab);
                }

                uint8_t starts[] = {
                    static_cast<uint8_t>(CTArgs::start_distance_in_hops_forward),
                    static_cast<uint8_t>(CTArgs::start_distance_in_hops_backward)};
                uint8_t ranges[] = {
                    static_cast<uint8_t>(CTArgs::range_hops_forward),
                    static_cast<uint8_t>(CTArgs::range_hops_backward)};
                if (ranges[0] == 0) {
                    starts[0] = starts[1];
                    ranges[0] = ranges[1];
                }

                // Configure fused route for payload + semaphore increment
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader fused_header(0, 0, 1, true);
                fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state<
                    UnicastFusedAtomicIncUpdateMask::Val | UnicastFusedAtomicIncUpdateMask::Flush>(
                    fabric_connection, fused_route_id, starts, ranges, fused_header, CTArgs::tensor0_page_size);

                uint32_t num_total_targets =
                    CTArgs::num_targets_forward_direction + CTArgs::num_targets_backward_direction;

                uint64_t barrier_sem_noc_addr_in_pkt =
                    safe_get_noc_addr(args.barrier_sem_noc0_x, args.barrier_sem_noc0_y, args.barrier_sem, 0);
                uint64_t secondary_sync_sem_noc_addr_in_pkt =
                    safe_get_noc_addr(args.barrier_sem_noc0_x, args.barrier_sem_noc0_y, args.secondary_sync_sem, 0);

                if (CTArgs::is_sender) {
                    cb_wait_front(CTArgs::cb0_id, CTArgs::num_pages_to_read);
                    size_t l1_read_addr = get_read_ptr(CTArgs::cb0_id);
                    uint64_t dst_noc_addr =
                        get_noc_addr(CTArgs::core_noc_x, CTArgs::core_noc_y, args.tensor_address0, 0);

                    uint64_t out_ready_sem_noc_addr_in_pkt = safe_get_noc_addr(
                        args.out_ready_sem_noc0_x, args.out_ready_sem_noc0_y, args.out_ready_sem_bank_addr, 0);

                    // For dual-axis mode: first unicast to secondary sender, then mcast along primary axis
                    if constexpr (CTArgs::has_secondary_target) {
                        auto& secondary_slot = fabric_connection.get(secondary_connection_idx);
                        volatile PACKET_HEADER_TYPE* secondary_header =
                            PacketHeaderPool::header_table[secondary_route_id].first;

                        // Set up unicast route for 2D fabric
                        fabric_set_unicast_route(fabric_connection, secondary_header, secondary_connection_idx);

                        // Send data + semaphore increment to secondary sender
                        fabric_unicast_noc_fused_unicast_with_atomic_inc(
                            &secondary_slot.sender,
                            secondary_header,
                            l1_read_addr,
                            CTArgs::tensor0_page_size * CTArgs::num_pages_to_read,
                            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                                dst_noc_addr, out_ready_sem_noc_addr_in_pkt, 1, true},
                            1);  // 1 hop to secondary sender
                    }
                    fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state<
                        UnicastFusedAtomicIncUpdateMask::WriteDstAddr | UnicastFusedAtomicIncUpdateMask::SemaphoreAddr |
                        UnicastFusedAtomicIncUpdateMask::PayloadSize>(
                        fabric_connection,
                        fused_route_id,
                        l1_read_addr,
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                            dst_noc_addr, out_ready_sem_noc_addr_in_pkt, 1, true},
                        CTArgs::tensor0_page_size * CTArgs::num_pages_to_read);

                    noc_async_write(l1_read_addr, dst_noc_addr, CTArgs::tensor0_page_size * CTArgs::num_pages_to_read);

                    // increment locally
                    uint64_t out_ready_sem_noc_addr = safe_get_noc_addr(
                        args.out_ready_sem_noc0_x, args.out_ready_sem_noc0_y, args.out_ready_sem_bank_addr);
                    noc_semaphore_inc(out_ready_sem_noc_addr, 1);

                    // 3. wait for mcast output ready semaphore
                    if (args.wait_output_semaphore) {
                        volatile tt_l1_ptr uint32_t* sem_ptr =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.out_ready_sem_bank_addr);
                        noc_semaphore_wait(sem_ptr, args.out_ready_sem_wait_value);
                    }

                    // 4. global semaphore reset
                    if (args.reset_global_semaphore) {
                        noc_semaphore_set(
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.out_ready_sem_bank_addr), 0);
                    }
                    noc_async_writes_flushed();
                    cb_pop_front(CTArgs::cb0_id, CTArgs::num_pages_to_read);

                } else if constexpr (CTArgs::is_secondary_sender) {
                    // Secondary sender: wait for data from primary sender, then broadcast along primary axis
                    // First wait for data to arrive from primary sender
                    if (args.wait_output_semaphore) {
                        volatile tt_l1_ptr uint32_t* sem_ptr =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.out_ready_sem_bank_addr);
                        noc_semaphore_wait(sem_ptr, args.out_ready_sem_wait_value);
                    }

                    // Reset semaphore after receiving data
                    if (args.reset_global_semaphore) {
                        noc_semaphore_set(
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.out_ready_sem_bank_addr), 0);
                    }

                    // broadcast the received data along the primary axis
                    uint64_t src_noc_addr =
                        get_noc_addr(CTArgs::core_noc_x, CTArgs::core_noc_y, args.tensor_address0, 0);

                    // Mcast the data along primary axis
                    uint64_t out_ready_sem_noc_addr_in_pkt = safe_get_noc_addr(
                        args.out_ready_sem_noc0_x, args.out_ready_sem_noc0_y, args.out_ready_sem_bank_addr, 0);

                    fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state<
                        UnicastFusedAtomicIncUpdateMask::WriteDstAddr | UnicastFusedAtomicIncUpdateMask::SemaphoreAddr |
                        UnicastFusedAtomicIncUpdateMask::PayloadSize>(
                        fabric_connection,
                        fused_route_id,
                        static_cast<size_t>(args.tensor_address0),
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                            src_noc_addr, out_ready_sem_noc_addr_in_pkt, 1, true},
                        CTArgs::tensor0_page_size * CTArgs::num_pages_to_read);
                    noc_async_writes_flushed();

                } else {
                    // Receiver: wait for data from broadcaster
                    if (args.wait_output_semaphore) {
                        volatile tt_l1_ptr uint32_t* sem_ptr =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.out_ready_sem_bank_addr);
                        noc_semaphore_wait(sem_ptr, args.out_ready_sem_wait_value);
                    }

                    // Reset global semaphore
                    if (args.reset_global_semaphore) {
                        noc_semaphore_set(
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.out_ready_sem_bank_addr), 0);
                    }
                }
                if constexpr (CTArgs::is_secondary_sender || CTArgs::is_sender) {
                    close_connections(fabric_connection);
                }

                noc_async_write_barrier();
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
