// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// The DRISC service's commands, as its clients build them and its kernel runs them. The rest of the host <-> kernel
// contract is hostdev/drisc_service/abi.hpp.

namespace tt::tt_metal::internal::drisc_service::abi {

// ServiceHeader.state. Magic values, so L1 left over from anything else reads as not running.
enum ServiceState : uint32_t {
    SERVICE_BOOTING = 0xD15C0001,   // host: image loaded, reset about to be released
    SERVICE_RUNNING = 0xD15C0002,   // kernel: serving slots
    SERVICE_STOPPING = 0xD15C0003,  // kernel: no client left, last rescan before stopping
    SERVICE_STOPPED = 0xD15C0004,   // kernel: writes flushed, about to return; or host: core held in reset
};
constexpr bool is_service_state(uint32_t word) { return word >= SERVICE_BOOTING && word <= SERVICE_STOPPED; }

// One end of a NoC transfer: an address within a tile, and the tile.
struct NocEndpoint {
    uint64_t local_addr;   // tile-local address: DRAM offset, or host IOVA
    uint32_t noc_xy_addr;  // packed (y << 6) | x, as NOC_XY_COORD / *_ADDR_COORDINATE

    constexpr NocEndpoint at_offset(uint64_t offset) const { return NocEndpoint{local_addr + offset, noc_xy_addr}; }
};
static_assert(sizeof(NocEndpoint) == 16);

// Command that tells the kernel to transfer data from one endpoint to another.
struct KernelCommand {
    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t size;
    uint16_t src_noc_xy_addr;
    uint16_t dst_noc_xy_addr;  // low 12 bits coords, top 4 bits CommandKind
};
static_assert(sizeof(KernelCommand) == 24);

// Coords are {y[5:0], x[5:0]} (NOC_XY_ENCODING; the kernel asserts NOC_COORD_BITS against NOC_ADDR_NODE_ID_BITS), so
// dst_noc_xy_addr's top 4 bits carry the kind. CMD_SINGLE's kind is 0, and the kernel never routes the coords of a
// CMD_BATCH or CMD_LOCAL, so kind bits never reach the NIU.
//
// CMD_LOCAL copies within the GDDR bank of the kernel's seat, over the seat's DMA engine instead of the NoC. Its
// addresses are byte offsets within that bank, its coords are unused, and its size must be a multiple of
// LOCAL_SIZE_ALIGN. A batch may carry CMD_SINGLE and CMD_LOCAL entries.
enum CommandKind : uint16_t {
    CMD_SINGLE = 0,
    CMD_BATCH = 1,
    CMD_LOCAL = 2,
};
inline constexpr uint32_t NOC_COORD_BITS = 6;
inline constexpr uint32_t CMD_KIND_SHIFT = 2 * NOC_COORD_BITS;
constexpr uint32_t noc_xy(uint32_t x, uint32_t y) { return (y << NOC_COORD_BITS) | x; }
inline constexpr uint32_t LOCAL_SIZE_ALIGN = 16;
static_assert(__builtin_popcount(LOCAL_SIZE_ALIGN) == 1, "the kernel checks this with a mask");

constexpr CommandKind command_kind(const KernelCommand& command) {
    return static_cast<CommandKind>(command.dst_noc_xy_addr >> CMD_KIND_SHIFT);
}

constexpr KernelCommand make_single_command(NocEndpoint src, NocEndpoint dst, uint32_t size) {
    return KernelCommand{
        .src_addr = src.local_addr,
        .dst_addr = dst.local_addr,
        .size = size,
        .src_noc_xy_addr = static_cast<uint16_t>(src.noc_xy_addr),
        .dst_noc_xy_addr = static_cast<uint16_t>(dst.noc_xy_addr),
    };
}

constexpr KernelCommand make_local_command(uint64_t src_offset, uint64_t dst_offset, uint32_t size) {
    return KernelCommand{
        .src_addr = src_offset,
        .dst_addr = dst_offset,
        .size = size,
        .src_noc_xy_addr = 0,
        .dst_noc_xy_addr = static_cast<uint16_t>(CMD_LOCAL << CMD_KIND_SHIFT),
    };
}

// Most clients one seat serves at once.
inline constexpr uint32_t MAX_CLIENTS = 8;

// Per-client command queue entries in L1.
inline constexpr uint32_t CLIENT_QUEUE_CAPACITY = 8;
static_assert(__builtin_popcount(CLIENT_QUEUE_CAPACITY) == 1, "CLIENT_QUEUE_CAPACITY must be a power of two");

// Most commands one batch carries.
inline constexpr uint32_t BATCH_MAX_COMMANDS = 64;

}  // namespace tt::tt_metal::internal::drisc_service::abi
