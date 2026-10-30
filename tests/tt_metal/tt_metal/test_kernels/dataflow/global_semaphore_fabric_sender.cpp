// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// up()s a GlobalSemaphore on a core of another chip over fabric; the semaphore's address is common runtime
// arg 0. Runtime args: {dst_noc_x, dst_noc_y, increments, then the chip target (num_hops on a line, or
// dst_dev_id and dst_mesh_id with FABRIC_2D), then the fabric connection args}.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/global_semaphore.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

constexpr GlobalSemaphoreBindingToken sem_token{0};

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t dst_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t dst_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t increments = get_arg_val<uint32_t>(arg_idx++);
#if defined(FABRIC_2D)
    const auto dst_dev_id = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const auto dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++));
    const ToChip chip{.dst_dev_id = dst_dev_id, .dst_mesh_id = dst_mesh_id};
#else
    const ToChip chip{.num_hops = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++))};
#endif

    auto sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
    sender.open();
    auto* hdr = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(PacketHeaderPool::allocate_header());

    GlobalSemaphore sem(sem_token);
    Noc noc;
    for (uint32_t i = 0; i < increments; i++) {
        sem.up(sender, hdr, chip, noc, dst_noc_x, dst_noc_y, 1);
    }

    sender.close();
}
