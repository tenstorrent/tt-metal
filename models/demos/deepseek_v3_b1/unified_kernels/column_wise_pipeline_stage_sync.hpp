// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

using namespace tt::tt_fabric::linear::experimental;

#endif

namespace deepseek_b1_ops {

struct ColumnWisePipelineStageSync {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================

    // Reader CTArgs (NCRISC)
    template <
        uint32_t runEntryDeviceLogicOnNCRISC,
        uint32_t runExitDeviceLogicOnNCRISC,
        uint32_t entryDeviceCoreNocXAddr,
        uint32_t entryDeviceCoreNocYAddr,
        uint32_t r1SemaphoreL1Addr,
        uint32_t r2SemaphoreL1Addr,
        uint32_t r3SemaphoreL1Addr,
        uint32_t fabricArgBase>
    struct ReaderCTArgs {
        static constexpr bool run_entry_device_logic_on_ncrisc = runEntryDeviceLogicOnNCRISC == 1;
        static constexpr bool run_exit_device_logic_on_ncrisc = runExitDeviceLogicOnNCRISC == 1;
        static constexpr uint32_t entry_device_core_noc_x_addr = entryDeviceCoreNocXAddr;
        static constexpr uint32_t entry_device_core_noc_y_addr = entryDeviceCoreNocYAddr;
        static constexpr uint32_t r1_semaphore_l1_addr = r1SemaphoreL1Addr;
        static constexpr uint32_t r2_semaphore_l1_addr = r2SemaphoreL1Addr;
        static constexpr uint32_t r3_semaphore_l1_addr = r3SemaphoreL1Addr;
        static constexpr uint32_t fabric_arg_base = fabricArgBase;
    };

    // Compute CTArgs (no-op) (TRISC)
    struct ComputeCTArgs {};

    // Writer CTArgs (BRISC)
    template <
        uint32_t runEntryDeviceLogicOnBRISC,
        uint32_t runExitDeviceLogicOnBRISC,
        uint32_t entryDeviceCoreNocXAddr,
        uint32_t entryDeviceCoreNocYAddr,
        uint32_t r1SemaphoreL1Addr,
        uint32_t r2SemaphoreL1Addr,
        uint32_t r3SemaphoreL1Addr,
        uint32_t fabricArgBase>
    struct WriterCTArgs {
        static constexpr bool run_entry_device_logic_on_brisc = runEntryDeviceLogicOnBRISC == 1;
        static constexpr bool run_exit_device_logic_on_brisc = runExitDeviceLogicOnBRISC == 1;
        static constexpr uint32_t entry_device_core_noc_x_addr = entryDeviceCoreNocXAddr;
        static constexpr uint32_t entry_device_core_noc_y_addr = entryDeviceCoreNocYAddr;
        static constexpr uint32_t r1_semaphore_l1_addr = r1SemaphoreL1Addr;
        static constexpr uint32_t r2_semaphore_l1_addr = r2SemaphoreL1Addr;
        static constexpr uint32_t r3_semaphore_l1_addr = r3SemaphoreL1Addr;
        static constexpr uint32_t fabric_arg_base = fabricArgBase;
    };

    // ========================================================================
    // Runtime args structs
    // ========================================================================

    // NCRISC reader args
    struct ReaderArgs {};

    // TRISC compute args (no-op)
    struct ComputeArgs {};

    // BRISC writer args
    struct WriterArgs {};

    // Select args type based on current RISC
    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // ========================================================================
    // Op
    // ========================================================================
    template <typename CTArgs>
    class Op {
    public:
        void operator()(const RTArgs& args) { impl(args); }

    private:
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
        static FORCE_INLINE void entry_device_impl(
            uint32_t entry_device_core_noc_x_addr,
            uint32_t entry_device_core_noc_y_addr,
            uint32_t r1_semaphore_l1_addr,
            uint32_t r2_semaphore_l1_addr,
            uint32_t r3_semaphore_l1_addr,
            size_t fabric_arg_base) {
            auto r1_semaphore_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r1_semaphore_l1_addr);
            auto r2_semaphore_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r2_semaphore_l1_addr);
            auto r3_semaphore_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r3_semaphore_l1_addr);

            uint64_t r2_semaphore_noc_addr =
                get_noc_addr(entry_device_core_noc_x_addr, entry_device_core_noc_y_addr, r2_semaphore_l1_addr);
            uint64_t r3_semaphore_noc_addr =
                get_noc_addr(entry_device_core_noc_x_addr, entry_device_core_noc_y_addr, r3_semaphore_l1_addr);

            constexpr uint32_t slot_a = 0;
            constexpr uint32_t slot_b = 1;
            constexpr uint32_t num_connections = 2;
            tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
            open_connections(fabric_connection, num_connections, fabric_arg_base);

            PacketHeaderPool::reset();
            auto* packet_header_a_ptr = PacketHeaderPool::allocate_header(1);
            auto* packet_header_b_ptr = PacketHeaderPool::allocate_header(1);

            fabric_set_unicast_route(fabric_connection, packet_header_a_ptr, slot_a);
            fabric_set_unicast_route(fabric_connection, packet_header_b_ptr, slot_b);

            auto& sender_a = fabric_connection.get(slot_a).sender;
            auto& sender_b = fabric_connection.get(slot_b).sender;

            packet_header_a_ptr->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{r2_semaphore_noc_addr, 1});
            packet_header_b_ptr->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{r2_semaphore_noc_addr, 1});

            // == Round 1
            noc_semaphore_wait_min(r1_semaphore_l1_ptr, 1);
            unified_kernels::semaphore_dec(r1_semaphore_l1_ptr);

            // propagate to both neighbours
            sender_a.wait_for_empty_write_slot();
            sender_a.send_payload_flush_blocking_from_address(
                (uint32_t)packet_header_a_ptr, sizeof(PACKET_HEADER_TYPE));

            sender_b.wait_for_empty_write_slot();
            sender_b.send_payload_flush_blocking_from_address(
                (uint32_t)packet_header_b_ptr, sizeof(PACKET_HEADER_TYPE));

            // == Round 2
            noc_semaphore_wait_min(r2_semaphore_l1_ptr, 2);
            unified_kernels::semaphore_dec(r2_semaphore_l1_ptr);
            unified_kernels::semaphore_dec(r2_semaphore_l1_ptr);

            // propagate to just left neighbour
            packet_header_a_ptr->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{r3_semaphore_noc_addr, 1});
            sender_a.wait_for_empty_write_slot();
            sender_a.send_payload_flush_blocking_from_address(
                (uint32_t)packet_header_a_ptr, sizeof(PACKET_HEADER_TYPE));

            // == Round 3
            noc_semaphore_wait_min(r3_semaphore_l1_ptr, 1);
            unified_kernels::semaphore_dec(r3_semaphore_l1_ptr);

            // == Shutdown
            close_connections(fabric_connection);
            noc_async_write_barrier();
            invalidate_l1_cache();
        }

        static FORCE_INLINE void exit_device_impl(
            uint32_t entry_device_core_noc_x_addr,
            uint32_t entry_device_core_noc_y_addr,
            uint32_t r1_semaphore_l1_addr,
            size_t fabric_arg_base) {
            uint64_t r1_semaphore_noc_addr =
                get_noc_addr(entry_device_core_noc_x_addr, entry_device_core_noc_y_addr, r1_semaphore_l1_addr);

            constexpr uint32_t slot = 0;
            constexpr uint32_t num_connections = 1;
            tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
            open_connections(fabric_connection, num_connections, fabric_arg_base);

            PacketHeaderPool::reset();
            auto* packet_header_ptr = PacketHeaderPool::allocate_header(1);
            fabric_set_unicast_route(fabric_connection, packet_header_ptr, slot);
            auto& sender = fabric_connection.get(slot).sender;

            packet_header_ptr->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{r1_semaphore_noc_addr, 1});
            sender.wait_for_empty_write_slot();
            sender.send_payload_flush_blocking_from_address((uint32_t)packet_header_ptr, sizeof(PACKET_HEADER_TYPE));

            close_connections(fabric_connection);
            noc_async_write_barrier();
        }
#endif

        void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC (Reader)
            // ================================================================
            if constexpr (CTArgs::run_entry_device_logic_on_ncrisc) {
                entry_device_impl(
                    CTArgs::entry_device_core_noc_x_addr,
                    CTArgs::entry_device_core_noc_y_addr,
                    CTArgs::r1_semaphore_l1_addr,
                    CTArgs::r2_semaphore_l1_addr,
                    CTArgs::r3_semaphore_l1_addr,
                    CTArgs::fabric_arg_base);
            } else if constexpr (CTArgs::run_exit_device_logic_on_ncrisc) {
                exit_device_impl(
                    CTArgs::entry_device_core_noc_x_addr,
                    CTArgs::entry_device_core_noc_y_addr,
                    CTArgs::r1_semaphore_l1_addr,
                    CTArgs::fabric_arg_base);
            }

#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC - No-op
            // ================================================================

#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC (Writer)
            // ================================================================
            if constexpr (CTArgs::run_entry_device_logic_on_brisc) {
                entry_device_impl(
                    CTArgs::entry_device_core_noc_x_addr,
                    CTArgs::entry_device_core_noc_y_addr,
                    CTArgs::r1_semaphore_l1_addr,
                    CTArgs::r2_semaphore_l1_addr,
                    CTArgs::r3_semaphore_l1_addr,
                    CTArgs::fabric_arg_base);
            } else if constexpr (CTArgs::run_exit_device_logic_on_brisc) {
                exit_device_impl(
                    CTArgs::entry_device_core_noc_x_addr,
                    CTArgs::entry_device_core_noc_y_addr,
                    CTArgs::r1_semaphore_l1_addr,
                    CTArgs::fabric_arg_base);
            }
#endif
        }
    };

};  // struct ColumnWisePipelineStageSync

}  // namespace deepseek_b1_ops
