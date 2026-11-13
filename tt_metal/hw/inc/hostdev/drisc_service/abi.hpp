// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>

#include "internal/tt-1xx/blackhole/dev_mem_map.h"

// Defines the host <-> kernel contract.

namespace tt::tt_metal::drisc_service::abi {

// Bump on any change to this file that the kernel and the host must agree on.
inline constexpr uint32_t ABI_VERSION = 8;
inline constexpr size_t CONTROL_PAGE_SIZE = 4096;

// The kernel's build. Its ELF carries one in section BUILD_BLOCK_SECTION, which the host checks before loading it,
// and the kernel copies it to ServiceHeader.build when it boots.
inline constexpr uint32_t BUILD_BLOCK_MAGIC = 0x44525342;  // "DRSB"
inline constexpr const char* BUILD_BLOCK_SECTION = ".drisc_service_build";
struct KernelBuildBlock {
    uint32_t magic;
    uint32_t abi_version;
};

// ServiceHeader.state. Magic values, so L1 left over from anything else reads as not running.
enum ServiceState : uint32_t {
    SERVICE_BOOTING = 0xD15C0001,   // host: image loaded, reset about to be released
    SERVICE_RUNNING = 0xD15C0002,   // kernel: serving slots
    SERVICE_STOPPING = 0xD15C0003,  // kernel: no client left, last rescan before stopping
    SERVICE_STOPPED = 0xD15C0004,   // kernel: writes flushed, about to return; or host: core held in reset
};
constexpr bool is_service_state(uint32_t word) { return word >= SERVICE_BOOTING && word <= SERVICE_STOPPED; }

// Client slot handshake words: type in [31:30], generation in [29:0]. The generation is never 0, so a zeroed slot never
// matches a live one, and an exit write left armed by an earlier client of the slot never matches a later one.
enum SlotWordType : uint32_t {
    SLOT_FREE = 0,
    SLOT_ATTACH = 1,  // request: host asks to attach; ack: kernel serves the slot
    SLOT_DETACH = 2,  // exit_mailbox: the client left, cleanly or by dying
    SLOT_FREED = 3,   // ack: kernel finished the client's work and dropped the slot
};
inline constexpr uint32_t SLOT_TYPE_SHIFT = 30;
inline constexpr uint32_t SLOT_GEN_MASK = (1u << SLOT_TYPE_SHIFT) - 1;
constexpr uint32_t slot_word(SlotWordType type, uint32_t gen) {
    return (static_cast<uint32_t>(type) << SLOT_TYPE_SHIFT) | (gen & SLOT_GEN_MASK);
}
constexpr SlotWordType slot_word_type(uint32_t word) { return static_cast<SlotWordType>(word >> SLOT_TYPE_SHIFT); }
constexpr uint32_t slot_word_gen(uint32_t word) { return word & SLOT_GEN_MASK; }

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
inline constexpr uint32_t BANK_ENDPOINTS = 3;  // NoC endpoints of one GDDR bank, one per DRISC

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

constexpr KernelCommand make_batch_command(NocEndpoint batch_slot, uint32_t command_count) {
    return KernelCommand{
        .src_addr = batch_slot.local_addr,
        .dst_addr = 0,
        .size = static_cast<uint32_t>(command_count * sizeof(KernelCommand)),
        .src_noc_xy_addr = static_cast<uint16_t>(batch_slot.noc_xy_addr),
        .dst_noc_xy_addr = static_cast<uint16_t>(CMD_BATCH << CMD_KIND_SHIFT),
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
static_assert(std::has_single_bit(CLIENT_QUEUE_CAPACITY), "CLIENT_QUEUE_CAPACITY must be a power of two");

// Most commands one batch carries.
inline constexpr uint32_t BATCH_MAX_COMMANDS = 64;
// Host batch region, per client: one slot per queue entry, so a batch issued
// into queue index i owns slot i until it retires. A slot holds commands only.
inline constexpr size_t BATCH_SLOT_SIZE = BATCH_MAX_COMMANDS * sizeof(KernelCommand);
static_assert(BATCH_SLOT_SIZE % 64 == 0, "batch slots are NOC read sources and must be 64B-aligned");
inline constexpr size_t BATCH_REGION_SIZE = CLIENT_QUEUE_CAPACITY * BATCH_SLOT_SIZE;

struct ServiceHeader {
    // Kernel, at boot; zero before. First, so it stays at one address across ABI versions.
    KernelBuildBlock build;
    uint32_t state;                 // ServiceState; kernel, and the host before boot
    uint32_t heartbeat;             // kernel: bumped on every slot scan; frozen = not running
    uint32_t free_pull_trid_count;  // kernel: batch window pull TRIDs free, of PULL_TRIDS; written on every slot scan
    // Host, before boot: the seat's bank. The kernel reads a CMD_SINGLE source inside it with the GDDR DMA engine.
    uint64_t bank_size;
    uint16_t bank_noc_xy_addr[BANK_ENDPOINTS];
};

// One client's slot. The host fills control_page and doorbell, then writes
// request = ATTACH|gen. The kernel attaches when request is an ATTACH whose
// gen differs from ack's, and answers ack = request. When the client's
// ExitWrite fd closes, on a clean exit or a crash, the KMD writes
// exit_mailbox = DETACH|gen. The kernel then stops carving the client's commands,
// lets its started reads and CMD_LOCAL copies finish, issues none of its NoC
// writes not yet started, and answers ack = FREED|gen.
struct ClientSlot {
    alignas(64) uint32_t request;  // host
    uint32_t exit_mailbox;         // KMD only, when the client's ExitWrite fd closes
    uint32_t ack;                  // kernel
    uint32_t doorbell;             // host: commands submitted
    NocEndpoint control_page;      // host: {host IOVA, PCIe tile coord}
    alignas(64) KernelCommand command_queue[CLIENT_QUEUE_CAPACITY];
};
static_assert(sizeof(ClientSlot) == 256);

// The command block is the host's bridge to the kernel, in L1.
struct L1CommandBlock {
    alignas(64) ServiceHeader header;
    ClientSlot slots[MAX_CLIENTS];
};
static_assert(offsetof(L1CommandBlock, slots) == 0x40);

// Absolute L1 addresses of the command block's fields. The block sits in tt-metal DRISC firmware's mailbox area, below
// the text the service's own firmware-style ELF loads at MEM_DRISC_FIRMWARE_BASE.
struct L1CommandBlockLayout {
    // Not at the start of the mailbox area: with the block at 0x40, a slot whose doorbell and queue sat below about
    // 0x400 took 1.4-1.9 us per single command against 1.0 us higher up (measured on a p150b, likely L1 bank or ECC
    // read-modify-write contention between the kernel's polling and the host's writes). At 0x8C0 every slot runs at
    // the full rate.
    static constexpr uint32_t base_addr = MEM_DRISC_MAILBOX_BASE + 0x880;
    static constexpr uint32_t end_addr = base_addr + sizeof(L1CommandBlock);
    static constexpr uint32_t build_addr =
        base_addr + offsetof(L1CommandBlock, header) + offsetof(ServiceHeader, build);
    static constexpr uint32_t state_addr =
        base_addr + offsetof(L1CommandBlock, header) + offsetof(ServiceHeader, state);
    static constexpr uint32_t heartbeat_addr =
        base_addr + offsetof(L1CommandBlock, header) + offsetof(ServiceHeader, heartbeat);
    static constexpr uint32_t free_pull_trid_count_addr =
        base_addr + offsetof(L1CommandBlock, header) + offsetof(ServiceHeader, free_pull_trid_count);
    static constexpr uint32_t slot_addr(uint32_t slot) {
        return base_addr + offsetof(L1CommandBlock, slots) + slot * sizeof(ClientSlot);
    }
    static constexpr uint32_t request_addr(uint32_t slot) { return slot_addr(slot) + offsetof(ClientSlot, request); }
    static constexpr uint32_t exit_mailbox_addr(uint32_t slot) {
        return slot_addr(slot) + offsetof(ClientSlot, exit_mailbox);
    }
    static constexpr uint32_t ack_addr(uint32_t slot) { return slot_addr(slot) + offsetof(ClientSlot, ack); }
    static constexpr uint32_t doorbell_addr(uint32_t slot) { return slot_addr(slot) + offsetof(ClientSlot, doorbell); }
    static constexpr uint32_t control_page_addr(uint32_t slot) {
        return slot_addr(slot) + offsetof(ClientSlot, control_page);
    }
    static constexpr uint32_t command_queue_addr(uint32_t slot) {
        return slot_addr(slot) + offsetof(ClientSlot, command_queue);
    }
};

static_assert(L1CommandBlockLayout::base_addr % 64 == 0);

// Kernel -> host completion channel in host memory, one per attached client; done_count owns its 64 B line.
struct ControlPage {
    alignas(64) uint32_t done_count;  // bumped per completed command
};
static_assert(sizeof(ControlPage) <= CONTROL_PAGE_SIZE);

// A tick is one accept/progress pass of the kernel main loop, so a scan period in wall time varies with load; under
// full load it must stay far below ALIVE_PROBE_MS.
inline constexpr uint32_t SLOT_SCAN_TICK_INTERVAL = 256;  // kernel ticks per slot scan + heartbeat bump
inline constexpr uint32_t ALIVE_PROBE_MS = 20;            // host: a heartbeat still this long is not running
inline constexpr uint32_t ATTACH_TIMEOUT_MS = 2000;       // host waits this long for ack = ATTACH|gen
inline constexpr uint32_t DETACH_TIMEOUT_MS = 2000;       // host waits this long for ack = FREED|gen
inline constexpr uint32_t BOOT_TIMEOUT_MS = 10000;        // host waits this long for a launched kernel's ack
inline constexpr uint32_t PULL_TRIDS = 7;                 // NIU transaction IDs the kernel pulls batch lists with
static_assert(std::has_single_bit(SLOT_SCAN_TICK_INTERVAL), "the kernel checks this with a power-of-two modulo");
static_assert(ALIVE_PROBE_MS < ATTACH_TIMEOUT_MS);

}  // namespace tt::tt_metal::drisc_service::abi
