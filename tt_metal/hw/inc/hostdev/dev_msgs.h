// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// dev_msgs.h
//
// Contains the structures/values uses in mailboxes to send messages to/from
// host and device and across brisc/ncrisc/trisc
//
// Note: this file is fed to a script for generating generic accessors for HAL,
// If you modify this file, CMake will invoke the script
//     tt_metal/llrt/hal/codegen/codegen.sh
// to update the generated files.
//
// Only a subset of the C++ language can be used:
// - Structs can only have scalars, structs, and 1-d array fields.
// - Constants and enums will be copied to tt::tt_metal::dev_msgs namespace.
//   - Do not define arch- or core-specific constants/enums here.  Use those in core_config.h.
// - #includes are copied to the generated interface, so make sure
//   those files are not arch- or core- specific.
// - #if... always evaluates to the false branch, so you can use
//   that to hide code from code generator.
// - Other C++ constructs are not supported.

#pragma once

#include <atomic>
#include <cstdint>

#include "hostdevcommon/profiler_common.h"
#include "hostdevcommon/dprint_common.h"

#ifdef HAL_BUILD
// HAL will include this file for different arch/cores, resulting in conflicting definitions that
// compiler will complain (ODR violation when compiling with LTO).
// Wrap the definitions in a unique namespace to avoid that.
namespace HAL_BUILD {  // NOLINT(modernize-concat-nested-namespaces)
#endif

// TODO: move these to processor specific files
#if defined(KERNEL_BUILD) || defined(FW_BUILD) || defined(HAL_BUILD)

// Several firmware/kernel files depend on this file for dev_mem_map.h and/or noc_parameters.h inclusion
// We don't want to pollute host code with those
// Including them here within the guard to make FW/KERNEL happy
// The right thing to do, would be "include what you use" in the other header files
#include "core_config.h"
#include "noc/noc_parameters.h"
#include "dev_mem_map.h"
// Deprecated in favor of dev_mem_map.h. Keep to avoid breaking changes.
#include "eth_l1_address_map.h"

#if defined(COMPILE_FOR_ERISC)
#define GET_MAILBOX_ADDRESS_DEV(x) (&(((mailboxes_t tt_l1_ptr*)eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE)->x))
#elif defined(COMPILE_FOR_IDLE_ERISC)
#define GET_MAILBOX_ADDRESS_DEV(x) (&(((mailboxes_t tt_l1_ptr*)MEM_IERISC_MAILBOX_BASE)->x))
#elif defined(ARCH_QUASAR)
#define GET_MAILBOX_ADDRESS_DEV(x) (&(((mailboxes_t tt_l1_ptr*)(MEM_MAILBOX_BASE + MEM_L1_UNCACHED_BASE))->x))
#else
#define GET_MAILBOX_ADDRESS_DEV(x) (&(((mailboxes_t tt_l1_ptr*)MEM_MAILBOX_BASE)->x))
#endif
// TODO: when device specific headers specify number of processors
// (and hal abstracts them on host), get these from there (same as above for dprint)
#if defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
constexpr uint32_t PROCESSOR_COUNT = static_cast<uint32_t>(EthProcessorTypes::COUNT);
#else
constexpr uint32_t PROCESSOR_COUNT = static_cast<uint32_t>(TensixProcessorTypes::COUNT);
#endif
#else
#error "Host code is not allowed to include dev_msgs.h, please use HAL interface instead."
#endif

struct profiler_msg_buffer_t {
    uint32_t data[kernel_profiler::PROFILER_L1_VECTOR_SIZE];
};

struct profiler_msg_t {
    uint32_t control_vector[kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE];
    profiler_msg_buffer_t buffer[PROCESSOR_COUNT];
};

// Messages for host to tell brisc to go
constexpr uint32_t RUN_MSG_INIT = 0x40;
constexpr uint32_t RUN_MSG_GO = 0x80;
constexpr uint32_t RUN_MSG_RESET_READ_PTR = 0xc0;
constexpr uint32_t RUN_MSG_RESET_READ_PTR_FROM_HOST = 0xe0;
constexpr uint32_t RUN_MSG_REPLAY_TRACE = 0xf0;
constexpr uint32_t RUN_MSG_DONE = 0;

// 0x80808000 is a micro-optimization, calculated with 1 riscv insn
constexpr uint32_t RUN_SYNC_MSG_INIT = 0x40;
constexpr uint32_t RUN_SYNC_MSG_GO = 0x80;
// Trigger loading CBs (and IRAM) before actually running the kernel.
constexpr uint32_t RUN_SYNC_MSG_LOAD = 0x1;
constexpr uint32_t RUN_SYNC_MSG_WAITING_FOR_RESET = 0x2;
constexpr uint32_t RUN_SYNC_MSG_INIT_SYNC_REGISTERS = 0x3;
constexpr uint32_t RUN_SYNC_MSG_DONE = 0;
constexpr uint32_t RUN_SYNC_MSG_ALL_GO = 0x80808080;
constexpr uint32_t RUN_SYNC_MSG_ALL_INIT = 0x40404040;
constexpr uint32_t RUN_SYNC_MSG_ALL_SUBORDINATES_DONE = 0;
constexpr uint64_t RUN_SYNC_MSG_ALL_SUBORDINATES_DMS_DONE = 0;

struct ncrisc_halt_msg_t {
    volatile uint32_t resume_addr;
    volatile uint32_t stack_save;
};

enum dispatch_mode {
    DISPATCH_MODE_DEV,
    DISPATCH_MODE_HOST,
};

enum noc_index {
    NOC_0 = 0,
    NOC_1 = 1,
};

enum noc_mode : uint8_t {
    DM_DEDICATED_NOC = 0,
    DM_DYNAMIC_NOC = 1,
    DM_INVALID_NOC = 2,
};

// Address offsets to kernel runtime configuration components
// struct to densely packs values used by each processor
struct rta_offset_t {
    volatile uint16_t rta_offset;
    volatile uint16_t crta_offset;
};

enum dispatch_enable_flags : uint8_t {
    DISPATCH_ENABLE_FLAG_PRELOAD = 1 << 7,
};

struct kernel_config_msg_t {
    // Ring buffer of kernel configuration data
    volatile uint32_t kernel_config_base[ProgrammableCoreType::COUNT];
    volatile uint16_t sem_offset[ProgrammableCoreType::COUNT];
    volatile uint16_t local_cb_offset;
    volatile uint16_t remote_cb_offset;
    rta_offset_t rta_offset[MaxProcessorsPerCoreType];
    volatile uint8_t mode;     // dispatch mode host/dev
    volatile uint8_t pad2[1];  // CODEGEN:skip
    volatile uint32_t kernel_text_offset[MaxProcessorsPerCoreType];
    volatile uint32_t local_cb_mask;

    volatile uint8_t brisc_noc_id;
    volatile uint8_t brisc_noc_mode;
    volatile uint8_t min_remote_cb_start_index;
    volatile uint8_t exit_erisc_kernel;
    // 32 bit program/launch_msg_id used by the performance profiler
    // [9:0]: physical device id
    // [30:10]: program id
    // [31:31]: 0 (specifies that this id corresponds to a program running on device)
    volatile uint32_t host_assigned_id;
    // bit i set => processor i enabled
    volatile uint32_t enables;
    volatile uint16_t watcher_kernel_ids[MaxProcessorsPerCoreType];
    volatile uint16_t ncrisc_kernel_size16;  // size in 16 byte units

    volatile uint8_t sub_device_origin_x;  // Logical X coordinate of the sub device origin
    volatile uint8_t sub_device_origin_y;  // Logical Y coordinate of the sub device origin
    volatile uint8_t pad3[1 + ((1 - MaxProcessorsPerCoreType % 2) * 2)];  // CODEGEN:skip

    volatile uint8_t preload;  // Must be at end, so it's only written when all other data is written.
} __attribute__((packed));

// Baby riscs don't natively support unaligned accesses, so ensure data alignment to prevent slow compiler workarounds.
static_assert(offsetof(kernel_config_msg_t, kernel_config_base) % sizeof(uint32_t) == 0);
static_assert(offsetof(kernel_config_msg_t, sem_offset) % sizeof(uint16_t) == 0);
static_assert(offsetof(kernel_config_msg_t, local_cb_offset) % sizeof(uint16_t) == 0);
static_assert(offsetof(kernel_config_msg_t, remote_cb_offset) % sizeof(uint16_t) == 0);
static_assert(offsetof(kernel_config_msg_t, remote_cb_offset) % sizeof(uint16_t) == 0);
static_assert(offsetof(kernel_config_msg_t, rta_offset) % sizeof(uint16_t) == 0);
static_assert(offsetof(kernel_config_msg_t, kernel_text_offset) % sizeof(uint32_t) == 0);
static_assert(offsetof(kernel_config_msg_t, local_cb_mask) % sizeof(uint32_t) == 0);
static_assert(offsetof(kernel_config_msg_t, host_assigned_id) % sizeof(uint32_t) == 0);

struct go_msg_t {
    union {
        uint32_t all;
        struct {
            uint8_t dispatch_message_offset;
            uint8_t master_x;
            uint8_t master_y;
            uint8_t signal;  // INIT, GO, DONE, RESET_RD_PTR
        };
    };
} __attribute__((packed));

struct launch_msg_t {  // must be cacheline aligned
    kernel_config_msg_t kernel_config;
} __attribute__((packed));

// save space for the structure, device side will cast to the corrrect structure
struct subordinate_sync_msg_t {
    volatile uint8_t map[subordinate_map_size];
};

constexpr int num_waypoint_bytes_per_riscv = 4;
struct debug_waypoint_msg_t {
    volatile uint8_t waypoint[num_waypoint_bytes_per_riscv];
};

// This structure is populated by the device and read by the host
struct debug_sanitize_addr_msg_t {
    volatile uint64_t noc_addr;
    volatile uint32_t l1_addr;
    volatile uint32_t len;
    volatile uint16_t which_risc;
    volatile uint16_t return_code;
    volatile uint8_t is_multicast;
    volatile uint8_t is_write;
    volatile uint8_t is_target;
    volatile uint8_t pad;  // CODEGEN:skip
};
static_assert(sizeof(debug_sanitize_addr_msg_t) % sizeof(uint32_t) == 0);

// Host -> device. Populated with the information on where we want to insert delays.
struct debug_insert_delays_msg_t {
    volatile uint32_t read_delay_processor_mask = 0;    // Which processors will delay their reads
    volatile uint32_t write_delay_processor_mask = 0;   // Which processors will delay their writes
    volatile uint32_t atomic_delay_processor_mask = 0;  // Which processors will delay their atomics
    volatile uint32_t feedback = 0;                     // Stores the feedback about delays (used for testing)
};

enum debug_sanitize_noc_return_code_enum {
    // 0 and 1 are a common stray values to write, so don't use those
    DebugSanitizeOK = 2,
    DebugSanitizeNocAddrUnderflow = 3,
    DebugSanitizeNocAddrOverflow = 4,
    DebugSanitizeNocAddrZeroLength = 5,
    DebugSanitizeNocTargetInvalidXY = 6,
    DebugSanitizeNocMulticastNonWorker = 7,
    DebugSanitizeNocMulticastInvalidRange = 8,
    DebugSanitizeNocAlignment = 9,
    DebugSanitizeNocMixedVirtualandPhysical = 10,
    DebugSanitizeInlineWriteDramUnsupported = 11,
    DebugSanitizeNocAddrMailbox = 12,
    DebugSanitizeNocLinkedTransactionViolation = 13,
    DebugSanitizeL1AddrOverflow = 14,
    DebugSanitizeEthSrcL1AddrOverflow = 15,
    DebugSanitizeEthDestL1AddrOverflow = 16,
};

struct debug_assert_msg_t {
    volatile uint16_t line_num;
    volatile uint8_t tripped;
    volatile uint8_t which;
};

enum debug_assert_type_t {
    DebugAssertOK = 2,
    DebugAssertTripped = 3,
    DebugAssertNCriscNOCReadsFlushedTripped = 4,
    DebugAssertNCriscNOCNonpostedWritesSentTripped = 5,
    DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped = 6,
    DebugAssertNCriscNOCPostedWritesSentTripped = 7
};

enum debug_transaction_type_t { TransactionRead = 0, TransactionWrite = 1, TransactionAtomic = 2, TransactionNumTypes };

struct debug_pause_msg_t {
    volatile uint8_t flags[MaxProcessorsPerCoreType];
    uint8_t pad[3];  // CODEGEN:skip
};

constexpr static int DEBUG_RING_BUFFER_ELEMENTS = 32;
constexpr static int DEBUG_RING_BUFFER_SIZE = DEBUG_RING_BUFFER_ELEMENTS * sizeof(uint32_t);
struct debug_ring_buf_msg_t {
    int16_t current_ptr;
    uint16_t wrapped;
    uint32_t data[DEBUG_RING_BUFFER_ELEMENTS];
};

struct debug_stack_usage_per_cpu_t {
    // min free stack, offset by +1 (0 == unset)
    volatile uint16_t min_free;
    volatile uint16_t watcher_kernel_id;
};

struct debug_stack_usage_t {
    debug_stack_usage_per_cpu_t cpu[MaxProcessorsPerCoreType];
    uint8_t pad[12];  // CODEGEN:skip
};

struct debug_eth_link_t {
    volatile uint8_t link_down;
};

enum watcher_enable_msg_t {
    WatcherDisabled = 2,
    WatcherEnabled = 3,
};

// TODO: w/ the hal, this can come from core specific defines
constexpr static std::uint32_t MAX_NUM_NOCS_PER_CORE = 2;

struct watcher_msg_t {
    volatile uint32_t enable;
    struct debug_waypoint_msg_t debug_waypoint[MaxProcessorsPerCoreType];
    struct debug_sanitize_addr_msg_t sanitize[MAX_NUM_NOCS_PER_CORE];
    std::atomic<bool> noc_linked_status[MAX_NUM_NOCS_PER_CORE];
    struct debug_eth_link_t eth_status;
    uint8_t pad0;  // CODEGEN:skip
    struct debug_assert_msg_t assert_status;
    struct debug_pause_msg_t pause_status;
    struct debug_stack_usage_t stack_usage;
    struct debug_insert_delays_msg_t debug_insert_delays;
    struct debug_ring_buf_msg_t debug_ring_buf;
};

#ifndef CODEGEN
// Host code does not need to use dprint_buf_msg_t (it uses DebugPrintMemLayout directly), skip because codegen can't
// see DebugPrintMemLayout.
struct dprint_buf_msg_t {
    DebugPrintMemLayout data[PROCESSOR_COUNT];
};
#endif

// NOC aligment max from BH
constexpr uint32_t TT_ARCH_MAX_NOC_WRITE_ALIGNMENT = 16;

enum class AddressableCoreType : uint8_t {
    TENSIX = 0,
    ETH = 1,
    PCIE = 2,
    DRAM = 3,
    HARVESTED = 4,
    UNKNOWN = 5,
    COUNT = 6,
};

struct addressable_core_t {
    volatile uint8_t x;
    volatile uint8_t y;
    volatile AddressableCoreType type;
};

// TODO: This can move into the hal eventually.
// This is the max number of non tensix cores between WH and BH that can be queried through Virtual Coordinates.
// All other Non Worker Cores are not accessible through virtual coordinates. Subject to change, depending on the arch.
// Currently sized for BH (first term is DRAM, second term is PCIe and last term is eth). On WH only Eth and Tensix
// cores are virtualized BH = DRAM(8*2) + 1 PCIe + Eth(12) vs. WH = Eth(16)
constexpr std::uint32_t MAX_VIRTUAL_NON_WORKER_CORES = 29;
// This is the max number of Non Worker Cores across BH and WH.
// BH = DRAM(8) + 1 PCIe + Eth(12) vs. WH = DRAM(18) + 1 PCIe + Eth(16)
constexpr std::uint32_t MAX_PHYSICAL_NON_WORKER_CORES = 35;
constexpr std::uint32_t MAX_HARVESTED_ON_AXIS = 2;
constexpr std::uint8_t CORE_COORD_INVALID = 0xFF;

enum class CoreMagicNumber : uint32_t {
    WORKER = 0x50ec09a3,
    ACTIVE_ETH = 0xc63050d1,
    IDLE_ETH = 0x837b6cae,
};
struct core_info_msg_t {
    volatile uint64_t noc_pcie_addr_base;
    volatile uint64_t noc_pcie_addr_end;
    volatile uint64_t noc_dram_addr_base;
    volatile uint64_t noc_dram_addr_end;
    addressable_core_t non_worker_cores[MAX_PHYSICAL_NON_WORKER_CORES];
    addressable_core_t virtual_non_worker_cores[MAX_VIRTUAL_NON_WORKER_CORES];
    volatile uint8_t harvested_coords[MAX_HARVESTED_ON_AXIS];
    volatile uint8_t virtual_harvested_coords[MAX_HARVESTED_ON_AXIS];
    volatile uint8_t noc_size_x;
    volatile uint8_t noc_size_y;
    volatile uint8_t worker_grid_size_x;
    volatile uint8_t worker_grid_size_y;
    volatile uint8_t absolute_logical_x;  // Logical X coordinate of this core
    volatile uint8_t absolute_logical_y;  // Logical Y coordinate of this core
    volatile uint32_t l1_unreserved_start;
    volatile CoreMagicNumber core_magic_number;
    uint8_t pad;  // CODEGEN:skip
};

constexpr uint32_t launch_msg_buffer_num_entries = 8;
// Equal to the maximum number of subdevices + 1. This allows all workers that aren't assigned to a subdevice to receive
// a dummy entry.
constexpr uint32_t go_message_num_entries = 9;

struct mailboxes_t {
    struct ncrisc_halt_msg_t ncrisc_halt;
    struct subordinate_sync_msg_t subordinate_sync;
    volatile uint32_t launch_msg_rd_ptr;  // Volatile so this can be manually reset by host. TODO: remove volatile when
                                          // dispatch init moves to one-shot.
    struct launch_msg_t launch[launch_msg_buffer_num_entries];
    volatile struct go_msg_t go_messages[go_message_num_entries];
    uint64_t link_status_check_timestamp;  // Next timestamp to check link status (active erisc)
    volatile uint32_t go_message_index;    // Index into go_messages to use. Always 0 on unicast cores.
    struct watcher_msg_t watcher;
    struct dprint_buf_msg_t dprint_buf;  // CODEGEN:skip
    struct core_info_msg_t core_info;
    uint32_t aerisc_run_flag;  // 1: run active ethernet firmware, 0: return to base firmware (active erisc)
    alignas(TT_ARCH_MAX_NOC_WRITE_ALIGNMENT)  // CODEGEN:skip
        profiler_msg_t profiler;
};

// Watcher struct needs to be 32b-divisible, since we need to write it from host using write_core().
static_assert(sizeof(watcher_msg_t) % sizeof(uint32_t) == 0);
static_assert(sizeof(kernel_config_msg_t) % sizeof(uint32_t) == 0);
static_assert(sizeof(core_info_msg_t) % sizeof(uint32_t) == 0);

struct eth_word_t {
    volatile uint32_t bytes_sent;
    volatile uint32_t dst_cmd_valid;
    uint32_t reserved_0;
    uint32_t reserved_1;
};

enum class SyncCBConfigRegion : uint8_t {
    DB_TENSIX = 0,
    TENSIX = 1,
    ROUTER_ISSUE = 2,
    ROUTER_COMPLETION = 3,
};

struct routing_info_t {
    volatile uint32_t routing_enabled;
    volatile uint32_t src_sent_valid_cmd;
    volatile uint32_t dst_acked_valid_cmd;
    volatile uint32_t unused_arg0;
    eth_word_t fd_buffer_msgs[2];
};

#ifdef HAL_BUILD
}  // namespace HAL_BUILD
#endif
