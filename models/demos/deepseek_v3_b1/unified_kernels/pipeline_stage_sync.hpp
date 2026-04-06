// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/api_common.h"

using namespace tt::tt_fabric::linear::experimental;
using namespace tt::tt_fabric::common::experimental;

#endif

namespace deepseek_b1_ops {

struct PipelineStageSync {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================

    // Reader CTArgs (NCRISC)
    template <
        uint32_t runStallingLogicOnNCRISC,
        uint32_t runSignallingLogicOnNCRISC,
        uint32_t isIntermediateSignaller,
        uint32_t isSignallingToIntermediateSignaller,
        uint32_t stallingCoreNocXAddr,
        uint32_t stallingCoreNocYAddr,
        uint32_t signallingCoreNocXAddr,
        uint32_t signallingCoreNocYAddr,
        uint32_t semaphoreL1Addr,
        uint32_t fabricArgBase>
    struct ReaderCTArgs {
        static constexpr bool run_stalling_logic_on_ncrisc = runStallingLogicOnNCRISC == 1;
        static constexpr bool run_signalling_logic_on_ncrisc = runSignallingLogicOnNCRISC == 1;
        static constexpr bool is_intermediate_signaller = isIntermediateSignaller == 1;
        static constexpr bool is_signalling_to_intermediate_signaller = isSignallingToIntermediateSignaller == 1;
        static constexpr uint32_t stalling_core_noc_x_addr = stallingCoreNocXAddr;
        static constexpr uint32_t stalling_core_noc_y_addr = stallingCoreNocYAddr;
        static constexpr uint32_t signalling_core_noc_x_addr = signallingCoreNocXAddr;
        static constexpr uint32_t signalling_core_noc_y_addr = signallingCoreNocYAddr;
        static constexpr uint32_t semaphore_l1_addr = semaphoreL1Addr;
        static constexpr uint32_t fabric_arg_base = fabricArgBase;
    };

    // Compute CTArgs (no-op) (TRISC)
    struct ComputeCTArgs {};

    // Writer CTArgs (BRISC)
    template <
        uint32_t runStallingLogicOnBRISC,
        uint32_t runSignallingLogicOnBRISC,
        uint32_t isIntermediateSignaller,
        uint32_t isSignallingToIntermediateSignaller,
        uint32_t stallingCoreNocXAddr,
        uint32_t stallingCoreNocYAddr,
        uint32_t signallingCoreNocXAddr,
        uint32_t signallingCoreNocYAddr,
        uint32_t semaphoreL1Addr,
        uint32_t fabricArgBase>
    struct WriterCTArgs {
        static constexpr bool run_stalling_logic_on_brisc = runStallingLogicOnBRISC == 1;
        static constexpr bool run_signalling_logic_on_brisc = runSignallingLogicOnBRISC == 1;
        static constexpr bool is_intermediate_signaller = isIntermediateSignaller == 1;
        static constexpr bool is_signalling_to_intermediate_signaller = isSignallingToIntermediateSignaller == 1;
        static constexpr uint32_t stalling_core_noc_x_addr = stallingCoreNocXAddr;
        static constexpr uint32_t stalling_core_noc_y_addr = stallingCoreNocYAddr;
        static constexpr uint32_t signalling_core_noc_x_addr = signallingCoreNocXAddr;
        static constexpr uint32_t signalling_core_noc_y_addr = signallingCoreNocYAddr;
        static constexpr uint32_t semaphore_l1_addr = semaphoreL1Addr;
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
        static FORCE_INLINE void stalling_impl(uint32_t semaphore_l1_addr) {
            /*
             * - Wait min as multiple signals may have been received before testing semaphore value
             * - Decrement by one (instead of set to 0), so further signals aren't erased
             * - Invalidate cache to ensure value is decremented before proceeding (to prevent wrap around reading same
             * value)
             */
            auto semaphore_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_l1_addr);
            noc_semaphore_wait_min(semaphore_l1_ptr, 1);
            unified_kernels::semaphore_dec(semaphore_l1_ptr);
            invalidate_l1_cache();
        }

        static FORCE_INLINE void signalling_impl(
            bool is_intermediate_signaller,
            bool is_signalling_to_intermediate_signaller,
            uint32_t stalling_core_noc_x_addr,
            uint32_t stalling_core_noc_y_addr,
            uint32_t signalling_core_noc_x_addr,
            uint32_t signalling_core_noc_y_addr,
            uint32_t semaphore_l1_addr,
            size_t fabric_arg_base) {
            if (is_intermediate_signaller) {
                auto semaphore_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_l1_addr);
                noc_semaphore_wait_min(semaphore_l1_ptr, 1);
                unified_kernels::semaphore_dec(semaphore_l1_ptr);
                invalidate_l1_cache();
            }

            uint64_t remote_semaphore_noc_addr;
            if (is_signalling_to_intermediate_signaller) {
                remote_semaphore_noc_addr =
                    get_noc_addr(signalling_core_noc_x_addr, signalling_core_noc_y_addr, semaphore_l1_addr);
            } else {
                remote_semaphore_noc_addr =
                    get_noc_addr(stalling_core_noc_x_addr, stalling_core_noc_y_addr, semaphore_l1_addr);
            }

            constexpr uint32_t num_connections = 1;
            tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
            open_connections(fabric_connection, num_connections, fabric_arg_base);

            PacketHeaderPool::reset();
            auto* packet_header_ptr = PacketHeaderPool::allocate_header(1);
            fabric_set_unicast_route(fabric_connection, packet_header_ptr, 0);

            constexpr uint32_t num_hops = 1;
            packet_header_ptr->to_chip_unicast(num_hops);
            packet_header_ptr->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{remote_semaphore_noc_addr, 1});

            auto& connection = fabric_connection.get(0).sender;
            connection.wait_for_empty_write_slot();
            connection.send_payload_flush_blocking_from_address(
                (uint32_t)packet_header_ptr, sizeof(PACKET_HEADER_TYPE));
            close_connections(fabric_connection);
            noc_async_write_barrier();
        }
#endif

        void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC (Reader)
            // ================================================================
            if constexpr (CTArgs::run_stalling_logic_on_ncrisc) {
                stalling_impl(CTArgs::semaphore_l1_addr);
            } else if constexpr (CTArgs::run_signalling_logic_on_ncrisc) {
                signalling_impl(
                    CTArgs::is_intermediate_signaller,
                    CTArgs::is_signalling_to_intermediate_signaller,
                    CTArgs::stalling_core_noc_x_addr,
                    CTArgs::stalling_core_noc_y_addr,
                    CTArgs::signalling_core_noc_x_addr,
                    CTArgs::signalling_core_noc_y_addr,
                    CTArgs::semaphore_l1_addr,
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
            if constexpr (CTArgs::run_stalling_logic_on_brisc) {
                stalling_impl(CTArgs::semaphore_l1_addr);
            } else if constexpr (CTArgs::run_signalling_logic_on_brisc) {
                signalling_impl(
                    CTArgs::is_intermediate_signaller,
                    CTArgs::is_signalling_to_intermediate_signaller,
                    CTArgs::stalling_core_noc_x_addr,
                    CTArgs::stalling_core_noc_y_addr,
                    CTArgs::signalling_core_noc_x_addr,
                    CTArgs::signalling_core_noc_y_addr,
                    CTArgs::semaphore_l1_addr,
                    CTArgs::fabric_arg_base);
            }
#endif
        }
    };

};  // struct PipelineStageSync

}  // namespace deepseek_b1_ops
