// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Unified Reduce-to-Root Kernel for 4x2 mesh 3-level reduction tree.
 *
 * This kernel performs multi-device reduce-to-one operation where:
 * - LEAFs (rows 0, 3) send data to ROOT3/ROOT2/ROOT1
 * - ROOT3 accumulates and sends to ROOT2/ROOT1
 * - ROOT2 accumulates and sends to ROOT1
 * - ROOT1 accumulates all data and gathers to output core
 *
 * Device Roles:
 *   MESH_LEAF = 0: Send data, no compute
 *   MESH_ROOT3 = 1: Receive from LEAF, accumulate, send to ROOT2/ROOT1
 *   MESH_ROOT2 = 2: Receive from LEAF + ROOT3, accumulate, send to ROOT1
 *   MESH_ROOT1 = 3: Receive from LEAF + ROOT3 + ROOT2, accumulate, gather to output
 */

#include <cstdint>

// Device roles
constexpr uint32_t MESH_LEAF = 0;
constexpr uint32_t MESH_ROOT3 = 1;
constexpr uint32_t MESH_ROOT2 = 2;
constexpr uint32_t MESH_ROOT1 = 3;

// =============================================================================
// NCRISC (Reader) Section - Receives data from fabric
// =============================================================================
#if defined(COMPILE_FOR_NCRISC)

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile-time args (named)
    constexpr uint32_t device_role = get_named_compile_time_arg_val("device_role");
    constexpr uint32_t num_tiles = get_named_compile_time_arg_val("num_tiles");
    constexpr uint32_t local_cb = get_named_compile_time_arg_val("local_cb");
    constexpr uint32_t received_cb_r1 = get_named_compile_time_arg_val("received_cb_r1");
    constexpr uint32_t received_cb_r2 = get_named_compile_time_arg_val("received_cb_r2");
    constexpr uint32_t received_cb_r3 = get_named_compile_time_arg_val("received_cb_r3");
    constexpr uint32_t is_fabric_core = get_named_compile_time_arg_val("is_fabric_core");

    DPRINT << "NRISC\n";
    DPRINT << "ct args:\n";
    DPRINT << " device_role: " << (uint32_t)device_role << "\n";
    DPRINT << " num_tiles: " << (uint32_t)num_tiles << "\n";
    DPRINT << " local_cb: " << (uint32_t)local_cb << "\n";
    DPRINT << " received_cb_r1: " << (uint32_t)received_cb_r1 << "\n";
    DPRINT << " received_cb_r2: " << (uint32_t)received_cb_r2 << "\n";
    DPRINT << " received_cb_r3: " << (uint32_t)received_cb_r3 << "\n";
    DPRINT << " is_fabric_core: " << (uint32_t)is_fabric_core << "\n";

    // Fabric cores have no reader work
    if constexpr (is_fabric_core) {
        DPRINT << "Fabric core - no NRISC work\n";
        return;
    }

    // Runtime args - semaphore addresses
    size_t arg_idx = 0;
    const uint32_t recv_sem_round1 = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t recv_sem_round2 = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t recv_sem_round3 = get_arg_val<uint32_t>(arg_idx++);

    DPRINT << "start of NRISC work\n";
    if constexpr (device_role == MESH_ROOT3 || device_role == MESH_ROOT2 || device_role == MESH_ROOT1) {
        DPRINT << "Round 1: Receive from LEAF\n";
        // Push local data to compute (local_cb is in-place on input shard)
        cb_reserve_back(local_cb, num_tiles);
        cb_push_back(local_cb, num_tiles);

        // Round 1: Wait for shard from LEAF
        cb_reserve_back(received_cb_r1, num_tiles);
        volatile tt_l1_ptr uint32_t* recv_sem1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv_sem_round1);
        noc_semaphore_wait(recv_sem1_ptr, 1);
        noc_semaphore_set(recv_sem1_ptr, 0);
        cb_push_back(received_cb_r1, num_tiles);
        DPRINT << "Round 1 done\n";
    }

    if constexpr (device_role == MESH_ROOT2 || device_role == MESH_ROOT1) {
        DPRINT << "Round 2: Receive from ROOT3\n";
        // Round 2: Wait for result from ROOT3
        cb_reserve_back(received_cb_r2, num_tiles);
        volatile tt_l1_ptr uint32_t* recv_sem2_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv_sem_round2);
        noc_semaphore_wait(recv_sem2_ptr, 1);
        noc_semaphore_set(recv_sem2_ptr, 0);
        cb_push_back(received_cb_r2, num_tiles);
        DPRINT << "Round 2 done\n";
    }

    if constexpr (device_role == MESH_ROOT1) {
        DPRINT << "Round 3: Receive from ROOT2\n";
        // Round 3: Wait for result from ROOT2
        cb_reserve_back(received_cb_r3, num_tiles);
        volatile tt_l1_ptr uint32_t* recv_sem3_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv_sem_round3);
        noc_semaphore_wait(recv_sem3_ptr, 1);
        noc_semaphore_set(recv_sem3_ptr, 0);
        cb_push_back(received_cb_r3, num_tiles);
        DPRINT << "Round 3 done\n";
    }
}

// =============================================================================
// BRISC (Writer) Section - Sends data via fabric or NOC
// =============================================================================
#elif defined(COMPILE_FOR_BRISC)

#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"

// Template helper for routing - allows if constexpr to work for both 1D and 2D fabric
template <typename packet_header_t>
FORCE_INLINE void set_unicast_route(
    volatile tt_l1_ptr packet_header_t* header, uint16_t dst_dev_id, uint16_t dst_mesh_id, uint16_t num_hops) {
    if constexpr (std::is_same_v<packet_header_t, tt::tt_fabric::HybridMeshPacketHeader>) {
        fabric_set_unicast_route(header, dst_dev_id, dst_mesh_id);
    } else {
        fabric_set_unicast_route<false>(header, num_hops);
    }
}

void kernel_main() {
    // Compile-time args (named)
    constexpr uint32_t device_role = get_named_compile_time_arg_val("device_role");
    constexpr uint32_t num_tiles = get_named_compile_time_arg_val("num_tiles");
    constexpr uint32_t payload_size_bytes = get_named_compile_time_arg_val("payload_size_bytes");
    constexpr uint32_t local_cb = get_named_compile_time_arg_val("local_cb");
    constexpr uint32_t scratch_cb = get_named_compile_time_arg_val("scratch_cb");
    constexpr uint32_t packet_cb = get_named_compile_time_arg_val("packet_cb");
    constexpr uint32_t num_hops = get_named_compile_time_arg_val("num_hops");
    constexpr uint16_t dst_dev_id = get_named_compile_time_arg_val("dst_fabric_node_chip_id");
    constexpr uint16_t dst_mesh_id = get_named_compile_time_arg_val("dst_fabric_node_mesh_id");
    constexpr uint32_t output_core_noc_x = get_named_compile_time_arg_val("output_core_noc_x");
    constexpr uint32_t output_core_noc_y = get_named_compile_time_arg_val("output_core_noc_y");
    constexpr uint32_t num_workers = get_named_compile_time_arg_val("num_workers");
    constexpr uint32_t slot_size_bytes = get_named_compile_time_arg_val("slot_size_bytes");
    // Use actual sizeof(PACKET_HEADER_TYPE) instead of compile-time arg to ensure correct header size
    constexpr uint32_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);
    constexpr uint32_t is_fabric_core = get_named_compile_time_arg_val("is_fabric_core");

    DPRINT << "BRISC\n";
    DPRINT << "ct args:\n";
    DPRINT << " device_role: " << (uint32_t)device_role << "\n";
    DPRINT << " num_tiles: " << (uint32_t)num_tiles << "\n";
    DPRINT << " payload_size_bytes: " << (uint32_t)payload_size_bytes << "\n";
    DPRINT << " local_cb: " << (uint32_t)local_cb << "\n";
    DPRINT << " scratch_cb: " << (uint32_t)scratch_cb << "\n";
    DPRINT << " packet_cb: " << (uint32_t)packet_cb << "\n";
    DPRINT << " num_hops: " << (uint32_t)num_hops << "\n";
    DPRINT << " dst_dev_id: " << (uint32_t)dst_dev_id << "\n";
    DPRINT << " dst_mesh_id: " << (uint32_t)dst_mesh_id << "\n";
    DPRINT << " output_core_noc_x: " << (uint32_t)output_core_noc_x << "\n";
    DPRINT << " output_core_noc_y: " << (uint32_t)output_core_noc_y << "\n";
    DPRINT << " num_workers: " << (uint32_t)num_workers << "\n";
    DPRINT << " slot_size_bytes: " << (uint32_t)slot_size_bytes << "\n";
    DPRINT << " packet_header_size_bytes: " << (uint32_t)packet_header_size_bytes << "\n";
    DPRINT << " is_fabric_core: " << (uint32_t)is_fabric_core << "\n";
    // Fabric core: forward worker packets via fabric
    if constexpr (is_fabric_core) {
        DPRINT << "Fabric core logic\n";
        // ROOT1 fabric cores have nothing to do
        if constexpr (device_role == MESH_ROOT1) {
            DPRINT << "Fabric core on ROOT1 - no work\n";
            return;
        }

        DPRINT << "Forwarding worker packets via fabric\n";
        size_t arg_idx = 0;
        // Read worker semaphore IDs
        uint32_t worker_sem_addr[num_workers];
        for (uint32_t i = 0; i < num_workers; i++) {
            uint32_t sem_id = get_arg_val<uint32_t>(arg_idx++);
            worker_sem_addr[i] = get_semaphore(sem_id);
            DPRINT << " worker_sem[" << i << "] id=" << sem_id << " addr=" << worker_sem_addr[i] << "\n";
        }

        const uint32_t packet_buffer_addr = get_write_ptr(packet_cb);
        DPRINT << "packet_buffer_addr: " << packet_buffer_addr << "\n";

        // Build fabric connection
        DPRINT << "Building fabric sender from args at idx " << arg_idx << "\n";
        auto fabric_sender =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
        DPRINT << "Opening fabric sender...\n";
        fabric_sender.open();
        DPRINT << "Fabric sender opened\n";

        // Forward worker packets
        uint32_t slot_base = packet_buffer_addr;
        for (uint32_t worker = 0; worker < num_workers; worker++) {
            DPRINT << "Waiting for worker " << worker << " sem at addr " << worker_sem_addr[worker] << "\n";
            uint32_t worker_header_addr = slot_base;
            uint32_t worker_payload_addr = slot_base + packet_header_size_bytes;

            volatile tt_l1_ptr uint32_t* worker_sem_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(worker_sem_addr[worker]);
            noc_semaphore_wait(worker_sem_ptr, 1);
            DPRINT << "Worker " << worker << " semaphore received\n";
            noc_semaphore_set(worker_sem_ptr, 0);

            DPRINT << "Waiting for empty write slot...\n";
            fabric_sender.wait_for_empty_write_slot();
            DPRINT << "Got write slot, sending payload...\n";
            fabric_sender.send_payload_without_header_non_blocking_from_address(
                worker_payload_addr, payload_size_bytes);
            DPRINT << "Sending header (blocking)...\n";
            fabric_sender.send_payload_flush_blocking_from_address(worker_header_addr, packet_header_size_bytes);
            DPRINT << "Worker " << worker << " packet sent\n";

            slot_base += slot_size_bytes;
        }

        fabric_sender.close();
        noc_async_write_barrier();
        DPRINT << "Fabric core done\n";
        return;
    }

    DPRINT << "Worker core logic\n";
    // Worker core logic
    // Runtime args
    uint32_t arg_idx = 0;
    const uint32_t fabric_core_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t fabric_core_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t my_slot_idx = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t worker_sem_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t dst_l1_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t dst_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_base_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t shard_idx = get_arg_val<uint32_t>(arg_idx++);

    const uint32_t arrival_sem_addr = get_semaphore(worker_sem_id);
    const uint32_t my_noc_x = my_x[0];
    const uint32_t my_noc_y = my_y[0];

    DPRINT << "Worker core runtime args:\n";
    DPRINT << " fabric_core_noc_x: " << (uint32_t)fabric_core_noc_x << "\n";
    DPRINT << " fabric_core_noc_y: " << (uint32_t)fabric_core_noc_y << "\n";
    DPRINT << " my_slot_idx: " << (uint32_t)my_slot_idx << "\n";
    DPRINT << " worker_sem_id: " << (uint32_t)worker_sem_id << "\n";
    DPRINT << " dst_l1_addr: " << (uint32_t)dst_l1_addr << "\n";
    DPRINT << " dst_sem_addr: " << (uint32_t)dst_sem_addr << "\n";
    DPRINT << " output_base_addr: " << (uint32_t)output_base_addr << "\n";
    DPRINT << " shard_idx: " << (uint32_t)shard_idx << "\n";

    DPRINT << "arrival_sem_addr: " << (uint32_t)arrival_sem_addr << "\n";

    // ROOT1: gather final results to output core
    if constexpr (device_role == MESH_ROOT1) {
        DPRINT << "ROOT1: Sending final result via NOC write\n";
        uint32_t dst_addr = output_base_addr + shard_idx * payload_size_bytes;
        uint64_t dst_noc_addr = get_noc_addr(output_core_noc_x, output_core_noc_y, dst_addr);

        // Wait for compute to finish
        cb_wait_front(scratch_cb, num_tiles);
        uint32_t src_addr = get_read_ptr(scratch_cb);

        noc_async_write(src_addr, dst_noc_addr, payload_size_bytes);
        noc_async_write_barrier();
        cb_pop_front(scratch_cb, num_tiles);
        DPRINT << "ROOT1 done\n";
        return;
    }

    // Non-ROOT1: send via fabric
    DPRINT << "Non-ROOT1: Sending result via fabric\n";
    const uint32_t packet_buffer_addr = get_write_ptr(packet_cb);

    // Allocate packet header
    auto route_id = PacketHeaderPool::allocate_header_n(1);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header = PacketHeaderPool::header_table[route_id].first;

    // Set routing - works for both 1D (num_hops) and 2D (dst_dev_id, dst_mesh_id) fabric
    set_unicast_route(packet_header, dst_dev_id, dst_mesh_id, static_cast<uint16_t>(num_hops));

    // Set up fused write + atomic inc
    uint64_t dst_noc_addr = get_noc_addr(my_noc_x, my_noc_y, dst_l1_addr);
    uint64_t dst_sem_noc_addr = get_noc_addr(my_noc_x, my_noc_y, dst_sem_addr);
    packet_header->to_noc_fused_unicast_write_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, dst_sem_noc_addr, 1, false},
        payload_size_bytes);

    // Calculate slot in fabric core's packet buffer
    uint32_t slot_offset = my_slot_idx * slot_size_bytes;
    uint32_t header_dest_addr = packet_buffer_addr + slot_offset;
    uint32_t payload_dest_addr = header_dest_addr + packet_header_size_bytes;

    uint64_t header_noc_addr = get_noc_addr(fabric_core_noc_x, fabric_core_noc_y, header_dest_addr);
    uint64_t payload_noc_addr = get_noc_addr(fabric_core_noc_x, fabric_core_noc_y, payload_dest_addr);

    // Source CB: LEAF uses local_cb, others use scratch_cb
    constexpr uint32_t source_cb = (device_role == MESH_LEAF) ? local_cb : scratch_cb;

    // Wait for data
    if constexpr (device_role != MESH_LEAF) {
        cb_wait_front(source_cb, num_tiles);
    }
    uint32_t data_addr = get_read_ptr(source_cb);

    // Send header and payload to fabric core
    noc_async_write(reinterpret_cast<uint32_t>(packet_header), header_noc_addr, packet_header_size_bytes);
    noc_async_write(data_addr, payload_noc_addr, payload_size_bytes);

    // Signal fabric core
    uint64_t arrival_sem_noc_addr = get_noc_addr(fabric_core_noc_x, fabric_core_noc_y, arrival_sem_addr);
    noc_semaphore_inc(arrival_sem_noc_addr, 1);

    if constexpr (device_role != MESH_LEAF) {
        DPRINT << "Popping source CB\n";
        cb_pop_front(source_cb, num_tiles);
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();
    DPRINT << "Non-ROOT1 done\n";
}

// =============================================================================
// TRISC (Compute) Section - Performs reduction
// =============================================================================
#elif defined(COMPILE_FOR_TRISC)

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {

void MAIN {
    // Compile-time args (named)
    constexpr uint32_t device_role = get_named_compile_time_arg_val("device_role");
    constexpr uint32_t num_tiles = get_named_compile_time_arg_val("num_tiles");
    constexpr uint32_t local_cb = get_named_compile_time_arg_val("local_cb");
    constexpr uint32_t received_cb_r1 = get_named_compile_time_arg_val("received_cb_r1");
    constexpr uint32_t received_cb_r2 = get_named_compile_time_arg_val("received_cb_r2");
    constexpr uint32_t received_cb_r3 = get_named_compile_time_arg_val("received_cb_r3");
    constexpr uint32_t output_cb = get_named_compile_time_arg_val("output_cb");
    constexpr uint32_t scratch_cb = get_named_compile_time_arg_val("scratch_cb");
    constexpr uint32_t is_fabric_core = get_named_compile_time_arg_val("is_fabric_core");

    DPRINT << "TRISC\n";
    DPRINT << "ct args:\n";
    DPRINT << " device_role: " << (uint32_t)device_role << "\n";
    DPRINT << " num_tiles: " << (uint32_t)num_tiles << "\n";
    DPRINT << " local_cb: " << (uint32_t)local_cb << "\n";
    DPRINT << " received_cb_r1: " << (uint32_t)received_cb_r1 << "\n";
    DPRINT << " received_cb_r2: " << (uint32_t)received_cb_r2 << "\n";
    DPRINT << " received_cb_r3: " << (uint32_t)received_cb_r3 << "\n";
    DPRINT << " output_cb: " << (uint32_t)output_cb << "\n";
    DPRINT << " scratch_cb: " << (uint32_t)scratch_cb << "\n";
    DPRINT << " is_fabric_core: " << (uint32_t)is_fabric_core << "\n";
    // Fabric cores and LEAFs have no compute
    if constexpr (is_fabric_core || device_role == MESH_LEAF) {
        DPRINT << "No TRISC compute work\n";
        return;
    }

    // Initialize for binary operations
    binary_op_init_common(local_cb, received_cb_r1, scratch_cb);

    // Load local tiles to dest
    copy_tile_to_dst_init_short(local_cb);
    cb_wait_front(local_cb, num_tiles);
    acquire_dst();
    for (uint32_t i = 0; i < num_tiles; i++) {
        copy_tile(local_cb, i, i);
    }
    cb_pop_front(local_cb, num_tiles);

    // Accumulate from received_cb_r1 (LEAF data)
    binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(received_cb_r1);
    cb_wait_front(received_cb_r1, num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(received_cb_r1, i, i);
    }
    cb_pop_front(received_cb_r1, num_tiles);

    if constexpr (device_role == MESH_ROOT2 || device_role == MESH_ROOT1) {
        DPRINT << "Accumulating from ROOT3 data\n";
        // Accumulate from received_cb_r2 (ROOT3 data)
        cb_wait_front(received_cb_r2, num_tiles);
        for (uint32_t i = 0; i < num_tiles; i++) {
            binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(received_cb_r2, i, i);
        }
        cb_pop_front(received_cb_r2, num_tiles);
    }

    if constexpr (device_role == MESH_ROOT1) {
        DPRINT << "Accumulating from ROOT2 data\n";
        // Accumulate from received_cb_r3 (ROOT2 data)
        cb_wait_front(received_cb_r3, num_tiles);
        for (uint32_t i = 0; i < num_tiles; i++) {
            binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(received_cb_r3, i, i);
        }
        cb_pop_front(received_cb_r3, num_tiles);
    }

    // Pack result to scratch_cb
    cb_reserve_back(scratch_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        pack_tile(i, scratch_cb, i);
    }
    release_dst();
    cb_push_back(scratch_cb, num_tiles);
    DPRINT << "TRISC done\n";
}

}  // namespace NAMESPACE

#endif  // COMPILE_FOR_*
