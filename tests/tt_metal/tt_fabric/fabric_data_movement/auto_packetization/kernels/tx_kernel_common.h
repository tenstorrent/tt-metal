// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared boilerplate header for all auto-packetization TX (sender) kernels.
//
// Included by all 9 TX kernels:
//   - unicast_tx_writer_raw.cpp
//   - scatter_unicast_tx_writer_raw.cpp
//   - fused_atomic_inc_unicast_tx_writer_raw.cpp
//   - fused_scatter_atomic_inc_unicast_tx_writer_raw.cpp
//   - multicast_tx_writer_raw.cpp
//   - scatter_multicast_tx_writer_raw.cpp
//   - fused_atomic_inc_multicast_tx_writer_raw.cpp
//   - fused_scatter_atomic_inc_multicast_tx_writer_raw.cpp
//   - sparse_multicast_tx_writer_raw.cpp
//
// Provides:
//   1. Common #include block (7 headers + FABRIC_2D-conditional mesh/api.h + linear/api.h)
//   2. Namespace declarations (tt::tt_fabric + FABRIC_2D-conditional mesh/linear experimental)
//   3. TX_KERNEL_PARSE_UNICAST_ARGS(idx) -- RT arg parsing for the 4 unicast kernels
//   4. TX_KERNEL_SETUP(idx)              -- build sender, allocate header, open sender
//   5. TX_KERNEL_TEARDOWN()              -- close sender
//
// Usage for unicast (non-scatter) kernels:
//   void kernel_main() {
//       size_t idx = 0;
//       TX_KERNEL_PARSE_UNICAST_ARGS(idx)
//       TX_KERNEL_SETUP(idx)
//       // unique API calls ...
//       TX_KERNEL_TEARDOWN()
//   }
//
// Usage for scatter unicast kernels (scatter_offset after common args, before SETUP):
//   void kernel_main() {
//       size_t idx = 0;
//       TX_KERNEL_PARSE_UNICAST_ARGS(idx)
//       const uint32_t scatter_offset = get_arg_val<uint32_t>(idx++);
//       TX_KERNEL_SETUP(idx)
//       // unique API calls ...
//       TX_KERNEL_TEARDOWN()
//   }
//
// Usage for multicast/sparse_multicast kernels: include header only; keep kernel_main() intact.

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#ifdef FABRIC_2D
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#endif
#include "tt_metal/fabric/hw/inc/linear/api.h"

using namespace tt::tt_fabric;
#ifdef FABRIC_2D
using namespace tt::tt_fabric::mesh::experimental;
#else
using namespace tt::tt_fabric::linear::experimental;
#endif

// TX_KERNEL_PARSE_UNICAST_ARGS(idx)
//
// Declares and populates the common unicast RT arg variables from the runtime arg array.
// Argument positions:
//   idx+0: src_l1_addr   (u32)
//   idx+1: total_size    (u32)
//   idx+2: dst_base_addr (u32)
//   [FABRIC_2D only] idx+3: dst_mesh_id (u32 -> u16), idx+4: dst_dev_id (u32 -> u8)
//   idx+3 or idx+5: rx_noc_x    (u32)
//   idx+4 or idx+6: rx_noc_y    (u32)
//   idx+5 or idx+7: sem_l1_addr (u32)
//   [!FABRIC_2D only] next idx: num_hops (u32 -> u8)
//
// After the macro, `idx` points to the first fabric-connection arg.
#ifdef FABRIC_2D
#define TX_KERNEL_PARSE_UNICAST_ARGS(idx)                                                      \
    const uint32_t src_l1_addr   = get_arg_val<uint32_t>((idx)++);                            \
    const uint32_t total_size    = get_arg_val<uint32_t>((idx)++);                             \
    const uint32_t dst_base_addr = get_arg_val<uint32_t>((idx)++);                             \
    const uint16_t dst_mesh_id   = static_cast<uint16_t>(get_arg_val<uint32_t>((idx)++));      \
    const uint8_t  dst_dev_id    = static_cast<uint8_t>(get_arg_val<uint32_t>((idx)++));       \
    const uint32_t rx_noc_x      = get_arg_val<uint32_t>((idx)++);                             \
    const uint32_t rx_noc_y      = get_arg_val<uint32_t>((idx)++);                             \
    const uint32_t sem_l1_addr   = get_arg_val<uint32_t>((idx)++);
#else
#define TX_KERNEL_PARSE_UNICAST_ARGS(idx)                                                      \
    const uint32_t src_l1_addr   = get_arg_val<uint32_t>((idx)++);                            \
    const uint32_t total_size    = get_arg_val<uint32_t>((idx)++);                             \
    const uint32_t dst_base_addr = get_arg_val<uint32_t>((idx)++);                             \
    const uint32_t rx_noc_x      = get_arg_val<uint32_t>((idx)++);                             \
    const uint32_t rx_noc_y      = get_arg_val<uint32_t>((idx)++);                             \
    const uint32_t sem_l1_addr   = get_arg_val<uint32_t>((idx)++);                             \
    const uint8_t  num_hops      = static_cast<uint8_t>(get_arg_val<uint32_t>((idx)++));
#endif

// TX_KERNEL_SETUP(idx)
//
// Builds the fabric EDM sender from runtime args, allocates a packet header,
// and opens the sender connection. Must be called after TX_KERNEL_PARSE_UNICAST_ARGS
// (or after any additional arg parsing like scatter_offset). After this macro,
// `sender` and `packet_header` are available, and the sender is open.
#define TX_KERNEL_SETUP(idx)                                                                   \
    auto sender = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx); \
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header = PacketHeaderPool::allocate_header(); \
    sender.open<true>();

// TX_KERNEL_TEARDOWN()
//
// Closes the fabric EDM sender connection. Must be called at the end of kernel_main()
// after all fabric API calls are complete.
#define TX_KERNEL_TEARDOWN() \
    sender.close();
