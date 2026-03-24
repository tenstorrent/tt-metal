// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_NCRISC)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#elif defined(COMPILE_FOR_BRISC)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/api_common.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

using namespace tt::tt_fabric::linear::experimental;
using namespace tt::tt_fabric::common::experimental;

#elif defined(COMPILE_FOR_TRISC)

#endif

namespace deepseek_b1_ops {

struct PipelineStageSync {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================

    // Reader CTArgs (NCRISC)
    template <uint32_t isStallingDevice>
    struct ReaderCTArgs {
        static constexpr bool is_stalling_device = isStallingDevice == 1;
    };

    // Writer CTArgs (BRISC)
    template <
        uint32_t isSignallingDevice,
        uint32_t isStallingDeviceEqualSignallingDevice,
        uint32_t stallingDeviceChipID,
        unit32_t stallingDeviceMeshID,
        uint32_t fabricArgBase>
    struct WriterCTArgs {
        static constexpr bool is_signalling_device = isSignallingDevice == 1;
        static constexpr uint32_t is_stalling_device_equal_signalling_device = isStallingDeviceEqualSignallingDevice;
        static constexpr uint32_t stalling_device_chip_id = stallingDeviceChipID;
        static constexpr uint32_t stalling_device_mesh_id = stallingDeviceMeshID;
        static constexpr uint32_t fabric_arg_base = fabricArgBase;
    };

    // ========================================================================
    // Runtime args structs
    // ========================================================================

    // NCRISC reader args
    struct ReaderArgs {
        uint32_t stalling_device_semaphore_l1_addr;
    };

    // BRISC writer args
    struct WriterArgs {
        uint32_t stalling_device_semaphore_noc_x_addr;
        uint32_t stalling_device_semaphore_noc_y_addr;
        uint32_t stalling_device_semaphore_l1_addr;
    };

    // TRISC compute args (no-op)
    struct ComputeArgs {};

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
        void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC (Reader)
            // ================================================================
            if (!CTArgs::is_stalling_device) {
                return;
            }

            /*
             * - Wait min as multiple signals may have been received before testing semaphore value
             * - Decrement by one (instead of set to 0), so further signals aren't erased
             * - Invalidate cache to ensure value is decremented before proceeding
             */
            const uint32_t stalling_device_semaphore_l1_addr = args.stalling_device_semaphore_l1_addr;
            auto stalling_device_semaphore_l1_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stalling_device_semaphore_l1_addr);
            noc_semaphore_wait_min(stalling_device_semaphore_l1_ptr, 1);
            unified_kernels::semaphore_dec(stalling_device_semaphore_l1_ptr);
            invalidate_l1_cache();

#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC (Writer)
            // ================================================================
            if (!CTArgs::is_signalling_device) {
                return;
            }

            const uint32_t stalling_device_semaphore_noc_x_addr = args.stalling_device_semaphore_noc_x_addr;
            const uint32_t stalling_device_semaphore_noc_y_addr = args.stalling_device_semaphore_noc_y_addr;
            const uint32_t stalling_device_semaphore_l1_addr = args.stalling_device_semaphore_l1_addr;

            const uint64_t stalling_device_semaphore_noc_addr = get_noc_addr(
                stalling_device_semaphore_noc_x_addr,
                stalling_device_semaphore_noc_y_addr,
                stalling_device_semaphore_l1_addr);

            /*
             * - If stalling and signalling devices are the same, use a pure NoC semaphore inc
             * - Otherwise, must use a fabric + NoC semaphore inc
             */
            if constexpr (CTArgs::is_stalling_device_equal_signalling_device) {
                // pure NoC
                noc_semaphore_inc(stalling_device_semaphore_noc_addr, 1);
            } else {
                // fabric + NoC
                const uint32_t stalling_device_chip_id = CTArgs::stalling_device_chip_id;
                const uint32_t stalling_device_mesh_id = CTArgs::stalling_device_mesh_id;

                PacketHeaderPool::reset();
                constexpr uint32_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);
                auto route_id = PacketHeaderPool::allocate_header_n(1);
                volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header = PacketHeaderPool::header_table[route_id].first;
                set_unicast_route(
                    packet_header,
                    static_cast<uint16_t>(stalling_device_chip_id),
                    static_cast<uint16_t>(stalling_device_mesh_id),
                    1);
                packet_header->to_noc_unicast_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{stalling_device_semaphore_noc_addr, 1});

                size_t arg_idx = CTArgs::fabric_arg_base;
                auto fabric_sender =
                    tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
                fabric_sender.open();
                fabric_sender.wait_for_empty_write_slot();
                fabric_sender.send_payload_flush_blocking_from_address(
                    reinterpret_cast<uint32_t>(packet_header), packet_header_size_bytes);
                fabric_sender.close();
            }
            noc_async_atomic_barrier();

#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC - No-op
            // ================================================================
#endif
        }
    };

};  // struct PipelineStageSync

}  // namespace deepseek_b1_ops
