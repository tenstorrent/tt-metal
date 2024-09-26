// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// dev_msgs.h
//
// Contains the structures/values uses in mailboxes to send messages to/from
// host and device and across brisc/ncrisc/trisc
//

#pragma once

#include "core_config.h"
#include "noc/noc_parameters.h"
#include "dev_mem_map.h"
#include "hostdevcommon/profiler_common.h"

// TODO: move these to processor specific files
#if defined(COMPILE_FOR_ERISC)
#define GET_MAILBOX_ADDRESS_DEV(x) (&(((mailboxes_t tt_l1_ptr *)eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE)->x))
#elif defined(COMPILE_FOR_IDLE_ERISC)
#define GET_MAILBOX_ADDRESS_DEV(x) (&(((mailboxes_t tt_l1_ptr *)MEM_IERISC_MAILBOX_BASE)->x))
#else
#define GET_MAILBOX_ADDRESS_DEV(x) (&(((mailboxes_t tt_l1_ptr *)MEM_MAILBOX_BASE)->x))
#endif

// Messages for host to tell brisc to go
constexpr uint32_t RUN_MSG_INIT = 0x40;
constexpr uint32_t RUN_MSG_GO = 0x80;
constexpr uint32_t RUN_MSG_RESET_READ_PTR = 0xc0;
constexpr uint32_t RUN_MSG_DONE = 0;

// 0x80808000 is a micro-optimization, calculated with 1 riscv insn
constexpr uint32_t RUN_SYNC_MSG_INIT = 0x40;
constexpr uint32_t RUN_SYNC_MSG_GO = 0x80;
constexpr uint32_t RUN_SYNC_MSG_DONE = 0;
constexpr uint32_t RUN_SYNC_MSG_ALL_TRISCS_GO = 0x80808000;
constexpr uint32_t RUN_SYNC_MSG_ALL_GO = 0x80808080;
constexpr uint32_t RUN_SYNC_MSG_ALL_SLAVES_DONE = 0;

struct ncrisc_halt_msg_t {
    volatile uint32_t resume_addr;
    volatile uint32_t stack_save;
};

enum dispatch_mode {
    DISPATCH_MODE_DEV,
    DISPATCH_MODE_HOST,
};

enum dispatch_core_processor_classes {
    // Tensix processor classes
    DISPATCH_CLASS_TENSIX_DM0 = 0,
    DISPATCH_CLASS_TENSIX_DM1 = 1,
    DISPATCH_CLASS_TENSIX_COMPUTE = 2,

    // Ethernet processor classes
    DISPATCH_CLASS_ETH_DM0 = 0,

    DISPATCH_CLASS_MAX = 3,
};

enum dispatch_core_processor_masks {
    DISPATCH_CLASS_MASK_TENSIX_ENABLE_DM0     = 1 << DISPATCH_CLASS_TENSIX_DM0,
    DISPATCH_CLASS_MASK_TENSIX_ENABLE_DM1     = 1 << DISPATCH_CLASS_TENSIX_DM1,
    DISPATCH_CLASS_MASK_TENSIX_ENABLE_COMPUTE = 1 << DISPATCH_CLASS_TENSIX_COMPUTE,

    DISPATCH_CLASS_MASK_ETH_DM0 = 1 << DISPATCH_CLASS_ETH_DM0,
};

enum noc_index {
    NOC_0 = 0,
    NOC_1 = 1,
};

enum noc_mode : uint8_t {
    DEDICATED_NOC_PER_DM = 0,
    ANY_NOC_PER_DM = 1,
};

// Address offsets to kernel runtime configuration components
// struct to densely packs values used by each processor
struct dyn_mem_map_t {
    volatile uint16_t rta_offset;
    volatile uint16_t crta_offset;
};

struct kernel_config_msg_t {
    volatile uint16_t watcher_kernel_ids[DISPATCH_CLASS_MAX];
    volatile uint16_t ncrisc_kernel_size16;  // size in 16 byte units

    volatile uint16_t host_assigned_id;

    // Ring buffer of kernel configuration data
    volatile uint32_t kernel_config_base[static_cast<int>(ProgrammableCoreType::COUNT)];
    volatile uint16_t sem_offset[static_cast<int>(ProgrammableCoreType::COUNT)];
    volatile uint16_t cb_offset;
    dyn_mem_map_t mem_map[DISPATCH_CLASS_MAX];

    volatile uint8_t mode;                   // dispatch mode host/dev
    volatile uint8_t brisc_noc_id;
    volatile uint8_t max_cb_index;
    volatile uint8_t exit_erisc_kernel;
    volatile uint8_t brisc_noc_mode;
    volatile uint8_t enables;
} __attribute__((packed));

struct go_msg_t {
    volatile uint8_t pad;
    volatile uint8_t master_x;
    volatile uint8_t master_y;
    volatile uint8_t signal; // INIT, GO, DONE, RESET_RD_PTR
} __attribute__((packed));

struct launch_msg_t {  // must be cacheline aligned
    kernel_config_msg_t kernel_config;
} __attribute__((packed));

struct slave_sync_msg_t {
    union {
        volatile uint32_t all;
        struct {
            volatile uint8_t ncrisc;  // ncrisc must come first, see ncrisc-halt.S
            volatile uint8_t trisc0;
            volatile uint8_t trisc1;
            volatile uint8_t trisc2;
        };
    };
};

constexpr int num_waypoint_bytes_per_riscv = 4;
struct debug_waypoint_msg_t {
    volatile uint8_t waypoint[num_waypoint_bytes_per_riscv];
};

// This structure is populated by the device and read by the host
struct debug_sanitize_noc_addr_msg_t {
    volatile uint64_t noc_addr;
    volatile uint32_t l1_addr;
    volatile uint32_t len;
    volatile uint16_t which_risc;
    volatile uint16_t return_code;
    volatile uint8_t is_multicast;
    volatile uint8_t is_write;
    volatile uint8_t is_target;
    volatile uint8_t pad;
};

// Host -> device. Populated with the information on where we want to insert delays.
struct debug_insert_delays_msg_t {
    volatile uint8_t read_delay_riscv_mask = 0;    // Which Riscs will delay their reads
    volatile uint8_t write_delay_riscv_mask = 0;   // Which Riscs will delay their writes
    volatile uint8_t atomic_delay_riscv_mask = 0;  // Which Riscs will delay their atomics
    volatile uint8_t feedback = 0;                 // Stores the feedback about delays (used for testing)
};

enum debug_sanitize_noc_return_code_enum {
    // 0 and 1 are a common stray values to write, so don't use those
    DebugSanitizeNocOK                    = 2,
    DebugSanitizeNocAddrUnderflow         = 3,
    DebugSanitizeNocAddrOverflow          = 4,
    DebugSanitizeNocAddrZeroLength        = 5,
    DebugSanitizeNocTargetInvalidXY       = 6,
    DebugSanitizeNocMulticastNonWorker    = 7,
    DebugSanitizeNocMulticastInvalidRange = 8,
    DebugSanitizeNocAlignment             = 9,
};

struct debug_assert_msg_t {
    volatile uint16_t line_num;
    volatile uint8_t tripped;
    volatile uint8_t which;
};

enum debug_assert_tripped_enum {
    DebugAssertOK = 2,
    DebugAssertTripped = 3,
};

// XXXX TODO(PGK): why why why do we not have this standardized
typedef enum debug_sanitize_which_riscv {
    DebugBrisc = 0,
    DebugNCrisc = 1,
    DebugTrisc0 = 2,
    DebugTrisc1 = 3,
    DebugTrisc2 = 4,
    DebugErisc = 5,
    DebugIErisc = 6,
    DebugNumUniqueRiscs
} riscv_id_t;

typedef enum debug_transaction_type {
    TransactionRead = 0,
    TransactionWrite = 1,
    TransactionAtomic = 2,
    TransactionNumTypes
} debug_transaction_type_t;

struct debug_pause_msg_t {
    volatile uint8_t flags[DebugNumUniqueRiscs];
    volatile uint8_t pad[8 - DebugNumUniqueRiscs];
};

constexpr static int DEBUG_RING_BUFFER_ELEMENTS = 32;
constexpr static int DEBUG_RING_BUFFER_SIZE = DEBUG_RING_BUFFER_ELEMENTS * sizeof(uint32_t);
struct debug_ring_buf_msg_t {
    int16_t current_ptr;
    uint16_t wrapped;
    uint32_t data[DEBUG_RING_BUFFER_ELEMENTS];
};

struct debug_stack_usage_t {
    volatile uint16_t max_usage[DebugNumUniqueRiscs];
    volatile uint16_t watcher_kernel_id[DebugNumUniqueRiscs];
    volatile uint16_t pad[16 - DebugNumUniqueRiscs * 2];
};

constexpr static std::uint32_t DPRINT_BUFFER_SIZE = 204; // per thread
// TODO: when device specific headers specify number of processors
// (and hal abstracts them on host), get these from there
#if defined(COMPILE_FOR_ERISC) || defined (COMPILE_FOR_IDLE_ERISC)
constexpr static std::uint32_t DPRINT_BUFFERS_COUNT = 1;
#else
constexpr static std::uint32_t DPRINT_BUFFERS_COUNT = 5;
#endif

enum watcher_enable_msg_t {
    WatcherDisabled = 2,
    WatcherEnabled = 3,
};

// TODO: w/ the hal, this can come from core specific defines
constexpr static std::uint32_t MAX_RISCV_PER_CORE = 5;

struct watcher_msg_t {
    volatile uint32_t enable;
    struct debug_waypoint_msg_t debug_waypoint[MAX_RISCV_PER_CORE];
    struct debug_sanitize_noc_addr_msg_t sanitize_noc[NUM_NOCS];
    struct debug_assert_msg_t assert_status;
    struct debug_pause_msg_t pause_status;
    struct debug_stack_usage_t stack_usage;
    struct debug_insert_delays_msg_t debug_insert_delays;
    struct debug_ring_buf_msg_t debug_ring_buf;
};

struct dprint_buf_msg_t {
    uint8_t data[DPRINT_BUFFERS_COUNT][DPRINT_BUFFER_SIZE];
    uint32_t pad; // to 1024 bytes
};


// NOC aligment max from BH
static constexpr uint32_t TT_ARCH_MAX_NOC_WRITE_ALIGNMENT = 16;

// TODO: when device specific headers specify number of processors
// (and hal abstracts them on host), get these from there (same as above for dprint)
#if defined(COMPILE_FOR_ERISC) || defined (COMPILE_FOR_IDLE_ERISC)
static constexpr uint32_t PROFILER_RISC_COUNT = 1;
#else
static constexpr uint32_t PROFILER_RISC_COUNT = 5;
#endif

static constexpr uint32_t LAUNCH_NOC_ALIGMENT_PAD_COUNT = 1;
static constexpr uint32_t PROFILER_NOC_ALIGMENT_PAD_COUNT = 2;

struct profiler_msg_t {
    uint32_t control_vector[kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE];
    uint32_t buffer[PROFILER_RISC_COUNT][kernel_profiler::PROFILER_L1_VECTOR_SIZE];
};

struct addressable_core_t {
    volatile uint8_t x, y;
    volatile AddressableCoreType type;
};

// TODO: This can move into the hal eventually, currently sized for WH.
constexpr static std::uint32_t MAX_NON_WORKER_CORES = 36 + 1 + 16;
constexpr static std::uint32_t MAX_HARVESTED_ROWS = 2;
constexpr static std::uint8_t CORE_COORD_INVALID = 0xFF;
struct core_info_msg_t {
    volatile uint64_t noc_pcie_addr_base;
    volatile uint64_t noc_pcie_addr_end;
    volatile uint64_t noc_dram_addr_base;
    volatile uint64_t noc_dram_addr_end;
    addressable_core_t non_worker_cores[MAX_NON_WORKER_CORES];
    volatile uint8_t harvested_y[MAX_HARVESTED_ROWS];
    volatile uint8_t noc_size_x;
    volatile uint8_t noc_size_y;
    volatile uint8_t pad[29];
};


constexpr uint32_t launch_msg_buffer_num_entries = 4;
struct mailboxes_t {
    struct ncrisc_halt_msg_t ncrisc_halt;
    struct slave_sync_msg_t slave_sync;
    uint32_t launch_msg_rd_ptr;
    struct launch_msg_t launch[launch_msg_buffer_num_entries];
    struct go_msg_t go_message;
    struct watcher_msg_t watcher;
    struct dprint_buf_msg_t dprint_buf;
    uint32_t pads_2[PROFILER_NOC_ALIGMENT_PAD_COUNT];
    struct profiler_msg_t profiler;
    struct core_info_msg_t core_info;
};

// Watcher struct needs to be 32b-divisible, since we need to write it from host using write_hex_vec_to_core().
static_assert(sizeof(watcher_msg_t) % sizeof(uint32_t) == 0);
static_assert(sizeof(kernel_config_msg_t) % sizeof(uint32_t) == 0);
static_assert(sizeof(core_info_msg_t) % sizeof(uint32_t) == 0);

// TODO: move these checks into the HAL?
#ifndef TENSIX_FIRMWARE
// Validate assumptions on mailbox layout on host compile
// Constexpr definitions allow for printing of breaking values at compile time
#ifdef NCRISC_HAS_IRAM
// These are only used in ncrisc-halt.S
static_assert(MEM_MAILBOX_BASE + offsetof(mailboxes_t, slave_sync.ncrisc) == MEM_SLAVE_RUN_MAILBOX_ADDRESS);
static_assert(
    MEM_MAILBOX_BASE + offsetof(mailboxes_t, ncrisc_halt.stack_save) == MEM_NCRISC_HALT_STACK_MAILBOX_ADDRESS);
#endif
#if defined(COMPILE_FOR_ERISC) || defined (COMPILE_FOR_IDLE_ERISC)
static_assert( eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE + sizeof(mailboxes_t) < eth_l1_mem::address_map::ERISC_MEM_MAILBOX_END);
static_assert( MEM_IERISC_MAILBOX_BASE + sizeof(mailboxes_t) < MEM_IERISC_MAILBOX_END);
static constexpr uint32_t ETH_LAUNCH_CHECK = (eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE  + offsetof(mailboxes_t, launch)) % TT_ARCH_MAX_NOC_WRITE_ALIGNMENT;
static constexpr uint32_t ETH_PROFILER_CHECK = (eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE  + offsetof(mailboxes_t, profiler)) % TT_ARCH_MAX_NOC_WRITE_ALIGNMENT;
static_assert( ETH_LAUNCH_CHECK == 0);
static_assert( ETH_PROFILER_CHECK == 0);
static_assert(MEM_IERISC_FIRMWARE_BASE % TT_ARCH_MAX_NOC_WRITE_ALIGNMENT == 0);
static_assert(MEM_IERISC_MAILBOX_BASE + sizeof(mailboxes_t) < MEM_IERISC_MAILBOX_END);
static_assert(MEM_IERISC_MAILBOX_END <= MEM_IERISC_RESERVED2);
#else
static_assert(MEM_MAILBOX_BASE + sizeof(mailboxes_t) < MEM_MAILBOX_END);
static constexpr uint32_t TENSIX_LAUNCH_CHECK = (MEM_MAILBOX_BASE + offsetof(mailboxes_t, launch)) % TT_ARCH_MAX_NOC_WRITE_ALIGNMENT;
static constexpr uint32_t TENSIX_PROFILER_CHECK = (MEM_MAILBOX_BASE + offsetof(mailboxes_t, profiler)) % TT_ARCH_MAX_NOC_WRITE_ALIGNMENT;
static_assert( TENSIX_LAUNCH_CHECK == 0);
static_assert( TENSIX_PROFILER_CHECK == 0);
static_assert( sizeof(launch_msg_t) % TT_ARCH_MAX_NOC_WRITE_ALIGNMENT == 0);
#endif
#endif

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
