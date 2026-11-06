// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// All-to-all (a2a) fabric APIs.
//
// For topologies where every device reaches every other device directly over the NoC (e.g. Horizon's AXI
// crossbar). There are no fabric routers, connections, or packet headers: each call resolves the destination
// {mesh_id, dev_id, noc_x, noc_y, addr} to a flat address through the fabric routing table in L1 and issues
// the NoC transaction directly from the calling core.
//
// Destinations are addressed exactly like the mesh/linear APIs: a device and mesh id, plus a NoC command
// header whose noc_address comes from get_noc_addr(noc_x, noc_y, addr) on the destination's coordinates.
//
// All calls are non-blocking. Complete them with the normal NoC barriers (noc_async_write_barrier,
// noc_async_writes_flushed, noc_async_atomic_barrier).
//
// Multicast APIs unicast the same message to every listed device in the mesh. The NoC address (core and
// L1 offset) is the same on every destination device. Destinations are given either as a variadic list of
// device ids or as a DstDevIds view (a C array, or a pointer and count).

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"

namespace tt::tt_fabric::a2a::experimental {

// clang-format off
/**
 * Read-only view of a list of destination device ids. The minimal C++17 stand-in for
 * std::span<const uint8_t> (device builds are C++17). Built implicitly from a C array (count deduced) or
 * from a pointer and a runtime count:
 *
 *   uint8_t devs[3] = {1, 2, 3};
 *   fabric_multicast_noc_unicast_write(mesh, src, size, hdr, devs);       // count = 3
 *   fabric_multicast_noc_unicast_write(mesh, src, size, hdr, {devs, n});  // runtime count
 */
// clang-format on
struct DstDevIds {
    const uint8_t* data;
    uint32_t count;

    template <std::size_t N>
    constexpr DstDevIds(const uint8_t (&ids)[N]) : data(ids), count(static_cast<uint32_t>(N)) {}
    constexpr DstDevIds(const uint8_t* ids, uint32_t num_ids) : data(ids), count(num_ids) {}

    constexpr const uint8_t& operator[](uint32_t i) const { return data[i]; }
    constexpr const uint8_t* begin() const { return data; }
    constexpr const uint8_t* end() const { return data + count; }
    constexpr uint32_t size() const { return count; }
};

// Enables the variadic multicast overloads only for a non-empty list of integral device ids, so they never
// compete with the DstDevIds overloads.
template <typename... DevIdTs>
using enable_if_dev_id_list_t =
    std::enable_if_t<(sizeof...(DevIdTs) > 0) && (std::is_integral_v<std::decay_t<DevIdTs>> && ...)>;

// clang-format off
/**
 * Translates a local NoC address into the flat address of the same core and offset on another device.
 *
 * Return value: The flat 64-bit address to pass to noc_async_write and friends (uint64_t)
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | dst_dev_id                            | Destination device id                   | uint8_t                                       | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                      | True     |
 * | noc_address                           | Destination address from get_noc_addr   | uint64_t                                      | True     |
 */
// clang-format on
FORCE_INLINE uint64_t get_fabric_noc_addr(uint8_t dst_dev_id, uint16_t dst_mesh_id, uint64_t noc_address);

// ============================================================================
// Unicast
// ============================================================================

// clang-format off
/**
 * Issues a unicast write from local L1 memory to a NoC address on another device.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | dst_dev_id                            | Destination device id                   | uint8_t                                       | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                      | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                      | True     |
 * | noc_unicast_command_header            | Destination NOC command header          | tt::tt_fabric::NocUnicastCommandHeader        | True     |
 */
// clang-format on
FORCE_INLINE void fabric_unicast_noc_unicast_write(
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header);

// clang-format off
/**
 * Issues a 32-bit inline write to a NoC address on another device.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | dst_dev_id                            | Destination device id                   | uint8_t                                       | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                      | True     |
 * | noc_unicast_inline_write_command_header | Destination address and 32-bit value  | tt::tt_fabric::NocUnicastInlineWriteCommandHeader | True |
 */
// clang-format on
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write(
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header);

// clang-format off
/**
 * Issues an atomic increment to a semaphore on another device. When the command header's flush is set,
 * waits for this core's outstanding writes to be acked before issuing the increment, so the increment
 * cannot overtake them.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | dst_dev_id                            | Destination device id                   | uint8_t                                       | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                      | True     |
 * | noc_unicast_atomic_inc_command_header | Semaphore address, increment, flush     | tt::tt_fabric::NocUnicastAtomicIncCommandHeader | True   |
 */
// clang-format on
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc(
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header);

// clang-format off
/**
 * Issues a scatter write: splits the payload into chunk_count chunks and writes each to its own NoC
 * address on the destination device. The last chunk's size is size minus the listed chunk sizes.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | dst_dev_id                            | Destination device id                   | uint8_t                                       | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                      | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | size                                  | Total payload size in bytes             | uint32_t                                      | True     |
 * | noc_unicast_scatter_command_header    | Chunk addresses and sizes               | tt::tt_fabric::NocUnicastScatterCommandHeader | True     |
 */
// clang-format on
FORCE_INLINE void fabric_unicast_noc_scatter_write(
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header);

// clang-format off
/**
 * Issues a unicast write followed by an atomic increment on the destination device. When the command
 * header's flush is set, the write is acked before the increment is issued.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | dst_dev_id                            | Destination device id                   | uint8_t                                       | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                      | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                      | True     |
 * | noc_fused_unicast_atomic_inc_command_header | Write and semaphore addresses, increment, flush | tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader | True |
 */
// clang-format on
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc(
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header);

// clang-format off
/**
 * Issues a scatter write (two chunks) followed by an atomic increment on the destination device. When the
 * command header's flush is set, the writes are acked before the increment is issued.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | dst_dev_id                            | Destination device id                   | uint8_t                                       | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                      | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | size                                  | Total payload size in bytes             | uint32_t                                      | True     |
 * | noc_fused_scatter_write_atomic_inc_command_header | Chunks, semaphore address, increment, flush | tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader | True |
 */
// clang-format on
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc(
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_fused_scatter_write_atomic_inc_command_header);

// ============================================================================
// Multicast
//
// Each op has two overloads: a variadic list of device ids (dev_0, ..., dev_n) and a DstDevIds view.
// All destinations are in dst_mesh_id and share the command header's NoC address.
// ============================================================================

// clang-format off
/**
 * Issues a unicast write from local L1 memory to the same NoC address on each listed device.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                      | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                      | True     |
 * | noc_unicast_command_header            | Destination NOC command header          | tt::tt_fabric::NocUnicastCommandHeader        | True     |
 * | dst_dev_ids                           | Destination device ids                  | integral... or DstDevIds                      | True     |
 */
// clang-format on
template <typename... DevIdTs, typename = enable_if_dev_id_list_t<DevIdTs...>>
FORCE_INLINE void fabric_multicast_noc_unicast_write(
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    DevIdTs... dst_dev_ids);

FORCE_INLINE void fabric_multicast_noc_unicast_write(
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    DstDevIds dst_dev_ids);

// clang-format off
/**
 * Issues a 32-bit inline write to the same NoC address on each listed device.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                      | True     |
 * | noc_unicast_inline_write_command_header | Destination address and 32-bit value  | tt::tt_fabric::NocUnicastInlineWriteCommandHeader | True |
 * | dst_dev_ids                           | Destination device ids                  | integral... or DstDevIds                      | True     |
 */
// clang-format on
template <typename... DevIdTs, typename = enable_if_dev_id_list_t<DevIdTs...>>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write(
    uint16_t dst_mesh_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header,
    DevIdTs... dst_dev_ids);

FORCE_INLINE void fabric_multicast_noc_unicast_inline_write(
    uint16_t dst_mesh_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header,
    DstDevIds dst_dev_ids);

// clang-format off
/**
 * Issues an atomic increment to the same semaphore address on each listed device. When the command
 * header's flush is set, this core's outstanding writes are acked once before any increment is issued.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                      | True     |
 * | noc_unicast_atomic_inc_command_header | Semaphore address, increment, flush     | tt::tt_fabric::NocUnicastAtomicIncCommandHeader | True   |
 * | dst_dev_ids                           | Destination device ids                  | integral... or DstDevIds                      | True     |
 */
// clang-format on
template <typename... DevIdTs, typename = enable_if_dev_id_list_t<DevIdTs...>>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc(
    uint16_t dst_mesh_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header,
    DevIdTs... dst_dev_ids);

FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc(
    uint16_t dst_mesh_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header,
    DstDevIds dst_dev_ids);

// clang-format off
/**
 * Issues a scatter write to each listed device (see fabric_unicast_noc_scatter_write).
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                      | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | size                                  | Total payload size in bytes             | uint32_t                                      | True     |
 * | noc_unicast_scatter_command_header    | Chunk addresses and sizes               | tt::tt_fabric::NocUnicastScatterCommandHeader | True     |
 * | dst_dev_ids                           | Destination device ids                  | integral... or DstDevIds                      | True     |
 */
// clang-format on
template <typename... DevIdTs, typename = enable_if_dev_id_list_t<DevIdTs...>>
FORCE_INLINE void fabric_multicast_noc_scatter_write(
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header,
    DevIdTs... dst_dev_ids);

FORCE_INLINE void fabric_multicast_noc_scatter_write(
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header,
    DstDevIds dst_dev_ids);

// clang-format off
/**
 * Issues a unicast write to each listed device, then an atomic increment on each. When the command
 * header's flush is set, all writes are acked once before any increment is issued.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                      | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                      | True     |
 * | noc_fused_unicast_atomic_inc_command_header | Write and semaphore addresses, increment, flush | tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader | True |
 * | dst_dev_ids                           | Destination device ids                  | integral... or DstDevIds                      | True     |
 */
// clang-format on
template <typename... DevIdTs, typename = enable_if_dev_id_list_t<DevIdTs...>>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc(
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header,
    DevIdTs... dst_dev_ids);

FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc(
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header,
    DstDevIds dst_dev_ids);

// clang-format off
/**
 * Issues a two-chunk scatter write to each listed device, then an atomic increment on each. When the
 * command header's flush is set, all writes are acked once before any increment is issued.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                      | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | size                                  | Total payload size in bytes             | uint32_t                                      | True     |
 * | noc_fused_scatter_write_atomic_inc_command_header | Chunks, semaphore address, increment, flush | tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader | True |
 * | dst_dev_ids                           | Destination device ids                  | integral... or DstDevIds                      | True     |
 */
// clang-format on
template <typename... DevIdTs, typename = enable_if_dev_id_list_t<DevIdTs...>>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc(
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_fused_scatter_write_atomic_inc_command_header,
    DevIdTs... dst_dev_ids);

FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc(
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_fused_scatter_write_atomic_inc_command_header,
    DstDevIds dst_dev_ids);

}  // namespace tt::tt_fabric::a2a::experimental

// API IMPL files for architectures with an all-to-all interconnect are included here as they are added
#if defined(ARCH_QUASAR) && defined(IP_VARIANT_HORIZON)
#include "tt_metal/fabric/hw/inc/a2a/tt-2xx/quasar_horizon/api_impl.h"
#else
#error "a2a fabric APIs require an all-to-all interconnect (currently Horizon only)"
#endif
