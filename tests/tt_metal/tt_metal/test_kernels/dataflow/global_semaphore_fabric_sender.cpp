// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// up()s a GlobalSemaphore on a core of another chip over fabric; the semaphore's address is common runtime
// arg 0. Runtime args: {dst_noc_x, dst_noc_y, increments, then the chip target (num_hops on a line, or
// dst_dev_id and dst_mesh_id with FABRIC_2D), then the fabric connection args}.
// GLOBAL_SEM_WRITE_AND_UP: each increment is a write_and_up() that first copies payload_bytes from src_addr here
// to dst_addr on the target core. The extra runtime args {src_addr, dst_addr, payload_bytes} come before the chip
// target.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/global_semaphore.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

constexpr GlobalSemaphoreBindingToken sem_token{0};

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t dst_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t dst_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t increments = get_arg_val<uint32_t>(arg_idx++);
#if defined(GLOBAL_SEM_WRITE_AND_UP)
    const uint32_t src_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t dst_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t payload_bytes = get_arg_val<uint32_t>(arg_idx++);
#endif
#if defined(FABRIC_2D)
    const auto dst_dev_id = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const auto dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++));
    const GlobalSemaphore::ToChip chip{.dst_dev_id = dst_dev_id, .dst_mesh_id = dst_mesh_id};
#else
    const GlobalSemaphore::ToChip chip{.num_hops = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++))};
#endif

    auto sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
    sender.open();
    auto* hdr = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(PacketHeaderPool::allocate_header());

    GlobalSemaphore sem(sem_token);
    Noc noc;
#if defined(GLOBAL_SEM_WRITE_AND_UP)
    volatile tt_l1_ptr uint32_t* payload = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
    for (uint32_t i = 0; i < payload_bytes / sizeof(uint32_t); i++) {
        payload[i] = 0xA5000000u + i;
    }
    UnicastEndpoint ep;
    for (uint32_t i = 0; i < increments; i++) {
        sem.write_and_up(
            sender,
            hdr,
            chip,
            noc,
            ep,
            {.addr = src_addr},
            ep,
            {.noc_x = dst_noc_x, .noc_y = dst_noc_y, .addr = dst_addr},
            payload_bytes,
            dst_noc_x,
            dst_noc_y,
            1);
    }
#else
    for (uint32_t i = 0; i < increments; i++) {
        sem.up(sender, hdr, chip, noc, dst_noc_x, dst_noc_y, 1);
    }
#endif

    sender.close();
}
