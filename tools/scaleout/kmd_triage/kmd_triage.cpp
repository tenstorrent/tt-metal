// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file kmd_triage.cpp
 * @brief chip debugging multitool -- first-step triage, device side
 *
 * WHAT IS THIS:
 *   The device-side half of first-step triage: it answers "is this chip
 *   alive, and if not, how is it dead?" without needing a working runtime.
 *   It talks to tt-kmd directly over its ioctl interface -- no UMD, no
 *   tt_metal, nothing but libc -- so it still reports when the stack above
 *   it cannot load. device_side.sh drives it across every chip on the host
 *   and rolls the results up for the health check; host_side.sh is the
 *   other half, covering host, PCIe and driver state.
 *
 * HOW TO BUILD:
 *   Built as the `kmd_triage` target -- see ../CMakeLists.txt. It lands in
 *   build/tools/scaleout/kmd_triage. Deliberately still buildable on its
 *   own, so it can be dropped onto a machine that has no checkout:
 *     g++ -std=c++20 -O2 -Wall -Wextra -o kmd_triage kmd_triage.cpp -lpthread
 *
 * SUBCOMMANDS:
 *   discover    Enumerate every device: path, arch, PCI location, link
 *               generation/width, and whether the chip looks hung.
 *   info        Device inventory: PCI location and link, IOMMU mode,
 *               board type/id, DRAM status, firmware version, vitals.
 *               Host-side facts print even when the chip is hung.
 *   scratch     Dump the ARC Reset Unit scratch register banks.
 *   telemetry   Dump every telemetry tag published by ARC firmware.
 *   read32      Read one 32-bit word from a NOC endpoint.
 *   write32     Write one 32-bit word to a NOC endpoint.
 *   test        Run a named chip test.
 *   hung        Is the chip hung?  Config space, sysfs telemetry, a double
 *               heartbeat sample, then a NOC read to ARC.
 *   reset       Reset the chip, check firmware, NOC, and DMA, then request
 *               idle power.  --all for every chip at once, --sbr for a link
 *               reset without an ASIC reset, --glx for a Galaxy tray reset.
 *   nuke        Kill every process holding the device open, naming each one.
 *
 * HOW TO RUN:
 *   ./kmd_triage discover
 *   ./kmd_triage info [DEVICE]
 *   ./kmd_triage hung [DEVICE]
 *   ./kmd_triage scratch [DEVICE]
 *   ./kmd_triage telemetry [-r] [DEVICE]
 *   ./kmd_triage read32 [-d PATH] X Y ADDR
 *   ./kmd_triage write32 [-d PATH] X Y ADDR VALUE
 *   ./kmd_triage test --all [DEVICE]
 *   ./kmd_triage test noc_sanity [DEVICE]
 *   ./kmd_triage test dma_loopback [DEVICE]
 *   ./kmd_triage reset [--sbr] DEVICE
 *   ./kmd_triage reset [--sbr] --all
 *   ./kmd_triage reset --glx
 *   ./kmd_triage nuke DEVICE
 *
 * CONVENTIONS:
 *   - Standalone: single .cpp, libc plus the tt-kmd ioctl interface.  The
 *     ioctl structs are inlined rather than included, so this file can be
 *     copied to a machine on its own.
 *   - Power-aware: the device is opened with O_APPEND; subcommands only issue
 *     SET_POWER_STATE if necessary.
 *   - Exit status: 0 and 1 have consistent meanings across subcommands;
 *     2 and 3 are per-subcommand.  See the common device-access layer.
 *   - Error handling: an operational failure -- a syscall or ioctl that can
 *     only fail if the tool, its invocation, or its environment is broken --
 *     dies on the spot via DIE, file:line on stderr, exit status 1.  What
 *     the chip says is never an operational failure: findings are reported
 *     and judged with the subcommand's exit codes.
 *   - Verdict-last: the health-check subcommands (hung, reset, nuke) send
 *     all output to stdout and, unless they die, end with exactly one line
 *     from verdict(), [PASS] or [FAIL], so a wrapper can judge a completed
 *     run by its last line alone.
 */

// g++ defines this itself (libstdc++ requires it); the guard keeps the
// dependency explicit without redefining it.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstdarg>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <getopt.h>
#include <pthread.h>
#include <csetjmp>
#include <csignal>
#include <ctime>
#include <sys/wait.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <sys/types.h>
#include <cerrno>
#include <linux/mman.h>
#include <linux/types.h>

// ===== Copied from ioctl.h (tt-kmd) =====

#define TENSTORRENT_IOCTL_MAGIC 0xFA

#define TENSTORRENT_IOCTL_GET_DEVICE_INFO _IO(TENSTORRENT_IOCTL_MAGIC, 0)
#define TENSTORRENT_IOCTL_RESET_DEVICE _IO(TENSTORRENT_IOCTL_MAGIC, 6)
#define TENSTORRENT_IOCTL_PIN_PAGES _IO(TENSTORRENT_IOCTL_MAGIC, 7)
#define TENSTORRENT_IOCTL_UNPIN_PAGES _IO(TENSTORRENT_IOCTL_MAGIC, 10)
#define TENSTORRENT_IOCTL_ALLOCATE_TLB _IO(TENSTORRENT_IOCTL_MAGIC, 11)
#define TENSTORRENT_IOCTL_FREE_TLB _IO(TENSTORRENT_IOCTL_MAGIC, 12)
#define TENSTORRENT_IOCTL_CONFIGURE_TLB _IO(TENSTORRENT_IOCTL_MAGIC, 13)
#define TENSTORRENT_IOCTL_SET_POWER_STATE _IO(TENSTORRENT_IOCTL_MAGIC, 15)

struct tenstorrent_get_device_info_in {
    __u32 output_size_bytes;
};

struct tenstorrent_get_device_info_out {
    __u32 output_size_bytes;
    __u16 vendor_id;
    __u16 device_id;
    __u16 subsystem_vendor_id;
    __u16 subsystem_id;
    __u16 bus_dev_fn;
    __u16 max_dma_buf_size_log2;
    __u16 pci_domain;
};

struct tenstorrent_get_device_info {
    struct tenstorrent_get_device_info_in in;
    struct tenstorrent_get_device_info_out out;
};

// legacy tenstorrent_reset_device_in.flags
#define TENSTORRENT_RESET_DEVICE_RESTORE_STATE 0
#define TENSTORRENT_RESET_DEVICE_RESET_PCIE_LINK 1
#define TENSTORRENT_RESET_DEVICE_CONFIG_WRITE 2

// tenstorrent_reset_device_in.flags
#define TENSTORRENT_RESET_DEVICE_USER_RESET 3
#define TENSTORRENT_RESET_DEVICE_ASIC_RESET 4
#define TENSTORRENT_RESET_DEVICE_ASIC_DMC_RESET 5
#define TENSTORRENT_RESET_DEVICE_POST_RESET 6

struct tenstorrent_reset_device_in {
    __u32 output_size_bytes;
    __u32 flags;
};

struct tenstorrent_reset_device_out {
    __u32 output_size_bytes;
    __u32 result;
};

struct tenstorrent_reset_device {
    struct tenstorrent_reset_device_in in;
    struct tenstorrent_reset_device_out out;
};

#define TENSTORRENT_PIN_PAGES_NOC_DMA 2

struct tenstorrent_pin_pages_in {
    __u32 output_size_bytes;
    __u32 flags;
    __u64 virtual_address;
    __u64 size;
};

struct tenstorrent_pin_pages_out_extended {
    __u64 physical_address;
    __u64 noc_address;
};

struct tenstorrent_pin_pages {
    struct tenstorrent_pin_pages_in in;
    struct tenstorrent_pin_pages_out_extended out;
};

struct tenstorrent_unpin_pages_in {
    __u64 virtual_address;
    __u64 size;
    __u64 reserved;
};

struct tenstorrent_unpin_pages_out {};

struct tenstorrent_unpin_pages {
    struct tenstorrent_unpin_pages_in in;
    struct tenstorrent_unpin_pages_out out;
};

struct tenstorrent_power_state {
    __u32 argsz;
    __u32 flags;
    __u8 reserved0;
    __u8 validity;
    __u16 power_flags;
    __u16 power_settings[14];
};

#define TT_POWER_VALIDITY(flags_count, settings_count) (((flags_count) & 0xF) | (((settings_count) & 0xF) << 4))

struct tenstorrent_allocate_tlb_in {
    __u64 size;
    __u64 reserved;
};

struct tenstorrent_allocate_tlb_out {
    __u32 id;
    __u32 reserved0;
    __u64 mmap_offset_uc;
    __u64 mmap_offset_wc;
    __u64 reserved1;
};

struct tenstorrent_allocate_tlb {
    struct tenstorrent_allocate_tlb_in in;
    struct tenstorrent_allocate_tlb_out out;
};

struct tenstorrent_free_tlb_in {
    __u32 id;
};

struct tenstorrent_free_tlb_out {};

struct tenstorrent_free_tlb {
    struct tenstorrent_free_tlb_in in;
    struct tenstorrent_free_tlb_out out;
};

struct tenstorrent_noc_tlb_config {
    __u64 addr;
    __u16 x_end;
    __u16 y_end;
    __u16 x_start;
    __u16 y_start;
    __u8 noc;
    __u8 mcast;
    __u8 ordering;
    __u8 linked;
    __u8 static_vc;
    __u8 reserved0[3];
    __u32 reserved1[2];
};

struct tenstorrent_configure_tlb_in {
    __u32 id;
    struct tenstorrent_noc_tlb_config config;
};

struct tenstorrent_configure_tlb_out {
    __u64 reserved;
};

struct tenstorrent_configure_tlb {
    struct tenstorrent_configure_tlb_in in;
    struct tenstorrent_configure_tlb_out out;
};

// ===== End of ioctl.h copy =====

// ============================================================================
// Common device-access layer
// ============================================================================

#define WORMHOLE_PCI_DEVICE_ID 0x401e
#define BLACKHOLE_PCI_DEVICE_ID 0xb140

#define TLB_SIZE (1ULL << 21)
#define ALL_ONES 0xFFFFFFFFu

// Exit statuses 0 and 1 are common; 2 and 3 describe subcommand findings.
#define EXIT_OK 0
#define EXIT_SETUP_ERROR 1     // usage error, or could not reach the point of testing
#define EXIT_CHIP_SILENT 2     // telemetry, hung, reset, discover: the chip is not answering
#define EXIT_BAD_TELEMETRY 3   // telemetry: unpublished or malformed
#define EXIT_FW_SICK 3         // hung, discover: firmware looks sick
#define EXIT_RESET_FAILED 3    // reset: a reset step failed
#define EXIT_HOLDERS_REMAIN 2  // nuke: processes still hold the device open
#define EXIT_TEST_FAILED 2     // test: the test found a chip failure
#define EXIT_TEST_NO_ANSWER 3  // test noc_sanity: a node did not answer at all

// Operational failure: something that can only fail if the tool, its
// invocation, or its environment is broken.  Report where and die with
// EXIT_SETUP_ERROR.  DIE appends strerror(errno) for the usual failed-
// syscall case; DIEX is for a failure that is not a syscall's.
[[noreturn]] __attribute__((format(printf, 4, 5))) static void die_at(
    const char* file, int line, int err, const char* fmt, ...) {
    va_list ap;

    fflush(stdout);
    fprintf(stderr, "%s:%d: ", file, line);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    if (err) {
        fprintf(stderr, ": %s", strerror(err));
    }
    fprintf(stderr, "\n");
    exit(EXIT_SETUP_ERROR);
}
#define DIE(...) die_at(__FILE__, __LINE__, errno, __VA_ARGS__)
#define DIEX(...) die_at(__FILE__, __LINE__, 0, __VA_ARGS__)

// A device named on the command line: a path, or a bare ordinal as
// shorthand, so "-d 0" means "-d /dev/tenstorrent/0".
static const char* device_path_arg(const char* arg) {
    static char buf[sizeof("/dev/tenstorrent/") + 24];
    const char* p;

    for (p = arg; *p >= '0' && *p <= '9'; p++) {
    }
    if (p != arg && *p == '\0' && (size_t)(p - arg) < 24) {
        snprintf(buf, sizeof(buf), "/dev/tenstorrent/%s", arg);
        return buf;
    }
    return arg;
}

// ARC Reset Unit as seen over the NOC on Wormhole and Blackhole.
#define ARC_RESET_UNIT_BASE 0x880030000ULL

// ARC NIU NOC_NODE_ID register as seen over NOC0.
#define WH_ARC_NOC_NODE_ID 0xfffb2002cULL
#define BH_ARC_NOC_NODE_ID 0xffffffffff000044ULL

// The device, opened and identified.  arc_x/arc_y are the ARC tile's NOC0
// coordinates, which differ by architecture.
struct chip {
    int fd;
    int is_blackhole;
    uint16_t arc_x;
    uint16_t arc_y;
    const char* arch_name;
    struct tenstorrent_get_device_info_out info;
};

// Opens the device and identifies the architecture; dies on any failure.
static void chip_open(struct chip* chip, const char* path) {
    struct tenstorrent_get_device_info dev_info;

    memset(chip, 0, sizeof(*chip));

    // O_APPEND tells the driver this is a power-aware client.
    chip->fd = open(path, O_RDWR | O_APPEND | O_CLOEXEC);
    if (chip->fd < 0) {
        DIE("cannot open %s", path);
    }

    memset(&dev_info, 0, sizeof(dev_info));
    dev_info.in.output_size_bytes = sizeof(dev_info.out);
    if (ioctl(chip->fd, TENSTORRENT_IOCTL_GET_DEVICE_INFO, &dev_info) != 0) {
        DIE("%s: ioctl GET_DEVICE_INFO failed", path);
    }
    chip->info = dev_info.out;

    if (dev_info.out.device_id == WORMHOLE_PCI_DEVICE_ID) {
        chip->is_blackhole = 0;
        chip->arch_name = "Wormhole";
        chip->arc_x = 0;
        chip->arc_y = 10;
    } else if (dev_info.out.device_id == BLACKHOLE_PCI_DEVICE_ID) {
        chip->is_blackhole = 1;
        chip->arch_name = "Blackhole";
        chip->arc_x = 8;
        chip->arc_y = 0;
    } else {
        DIEX(
            "%s: unsupported device_id 0x%04x, want Wormhole 0x%04x or Blackhole 0x%04x",
            path,
            dev_info.out.device_id,
            WORMHOLE_PCI_DEVICE_ID,
            BLACKHOLE_PCI_DEVICE_ID);
    }
}

static void chip_close(struct chip* chip) {
    if (chip->fd >= 0) {
        close(chip->fd);
    }
    chip->fd = -1;
}

// A single 2 MiB TLB window that is reconfigured for each NOC access.
struct noc_window {
    int fd;
    uint32_t tlb_id;
    volatile uint8_t* mmio;
};

static void noc_window_open_cache(struct noc_window* win, int fd, int wc) {
    struct tenstorrent_allocate_tlb alloc_tlb;
    void* mmio;

    memset(win, 0, sizeof(*win));
    win->fd = fd;

    memset(&alloc_tlb, 0, sizeof(alloc_tlb));
    alloc_tlb.in.size = TLB_SIZE;
    if (ioctl(fd, TENSTORRENT_IOCTL_ALLOCATE_TLB, &alloc_tlb) != 0) {
        if (errno == ENOMEM) {
            DIEX(
                "ioctl ALLOCATE_TLB failed: no free 2M TLB windows; "
                "another process probably holds them all -- try the nuke subcommand");
        }
        DIE("ioctl ALLOCATE_TLB failed");
    }

    mmio = mmap(
        nullptr,
        TLB_SIZE,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        fd,
        wc ? alloc_tlb.out.mmap_offset_wc : alloc_tlb.out.mmap_offset_uc);
    if (mmio == MAP_FAILED) {
        DIE("mmap of TLB window failed");
    }

    win->tlb_id = alloc_tlb.out.id;
    win->mmio = (volatile uint8_t*)mmio;
}

// Allocates and maps the TLB window, uncached; dies on failure.
static void noc_window_open(struct noc_window* win, int fd) { noc_window_open_cache(win, fd, 0); }

static void noc_window_open_wc(struct noc_window* win, int fd) { noc_window_open_cache(win, fd, 1); }

static void noc_window_close(struct noc_window* win) {
    struct tenstorrent_free_tlb free_tlb;

    if (win->mmio == nullptr) {
        return;
    }

    // The driver requires the mapping to be gone before FREE_TLB.
    if (munmap((void*)win->mmio, TLB_SIZE) != 0) {
        DIE("munmap of TLB window failed");
    }
    win->mmio = nullptr;

    memset(&free_tlb, 0, sizeof(free_tlb));
    free_tlb.in.id = win->tlb_id;
    if (ioctl(win->fd, TENSTORRENT_IOCTL_FREE_TLB, &free_tlb) != 0) {
        DIE("ioctl FREE_TLB failed");
    }
}

// Points the window at (x, y, addr); dies if the TLB cannot be programmed.
static void noc_window_aim(struct noc_window* win, unsigned x, unsigned y, uint64_t addr) {
    struct tenstorrent_configure_tlb cfg;

    memset(&cfg, 0, sizeof(cfg));
    cfg.in.id = win->tlb_id;
    cfg.in.config.addr = addr & ~(TLB_SIZE - 1);
    cfg.in.config.x_end = x;
    cfg.in.config.y_end = y;
    cfg.in.config.noc = 0;
    cfg.in.config.ordering = 1;  // strict, matching tt-kmd's own register reads

    if (ioctl(win->fd, TENSTORRENT_IOCTL_CONFIGURE_TLB, &cfg) != 0) {
        DIE("ioctl CONFIGURE_TLB for (%u, %u) addr 0x%llx failed", x, y, (unsigned long long)addr);
    }
}

// Reads one 32-bit NOC register at (x, y, addr).  A target that does not
// answer is not an error here; it surfaces as a value of all ones.
static uint32_t noc_read32(struct noc_window* win, unsigned x, unsigned y, uint64_t addr) {
    noc_window_aim(win, x, y, addr);
    return *(volatile uint32_t*)(win->mmio + (addr & (TLB_SIZE - 1)));
}

// Writes one 32-bit NOC register at (x, y, addr).  The write is posted;
// nothing here says whether the target accepted it.
static void noc_write32(struct noc_window* win, unsigned x, unsigned y, uint64_t addr, uint32_t value) {
    noc_window_aim(win, x, y, addr);
    *(volatile uint32_t*)(win->mmio + (addr & (TLB_SIZE - 1))) = value;
}

// Emit the final [PASS] or [FAIL] line of a verdict-last subcommand.
__attribute__((format(printf, 2, 3))) static void verdict(int pass, const char* fmt, ...) {
    va_list ap;

    printf("%s ", pass ? "[PASS]" : "[FAIL]");
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
    fflush(stdout);
}

// ============================================================================
// scratch: dump the ARC scratch register banks
// ============================================================================
//
// Both architectures have the small scratch bank at +0x60.  Blackhole also
// has SCRATCH_RAM at +0x400; Wormhole instead exposes the +0x1D0 register bank.
#define ARC_SCRATCH_OFFSET 0x60
#define ARC_SCRATCH_COUNT 8

#define ARC_SCRATCH_RAM_OFFSET 0x400
#define ARC_SCRATCH_RAM_COUNT 64

#define WH_NOC_NODEID_OFFSET 0x1D0
#define WH_NOC_NODEID_COUNT 6

// Firmware roles for each register; NULL means no assigned role.
static const char* const wh_scratch_notes[ARC_SCRATCH_COUNT] = {
    /* 0 */ "POST_CODE",
    /* 1 */ "SPI boot version (0xDEADC0DE = therm trip / watchdog panic)",
    /* 2 */ "ARC msg mailbox 1: message",
    /* 3 */ "ARC msg mailbox 0: args / return",
    /* 4 */ "ARC msg mailbox 1: args / return",
    /* 5 */ "ARC msg mailbox 0: message (0xAA00 | code)",
    /* 6 */ "armisc_info to PCIe controller (incl. DBI bit)",
    /* 7 */ "awmisc_info to PCIe controller (incl. DBI bit)",
};

static const char* const wh_nodeid_notes[WH_NOC_NODEID_COUNT] = {
    /* 0 */ "NOC_NODEID_X_0: telemetry table ptr (| 0x800000000 for NOC)",
    /* 1 */ "NOC_NODEID_Y_0: telemetry data ptr (| 0x800000000 for NOC)",
    /* 2 */ "NOC_NODEID_X_1: ARC msg queue info ptr (WH equivalent of BH SCRATCH_RAM[11])",
    /* 3 */ "NOC_NODEID_Y_1 (unused)",
    /* 4 */ "NOC_DATALINE_X (unused)",
    /* 5 */ "NOC_DATALINE_Y (unused)",
};

static const char* const bh_scratch_notes[ARC_SCRATCH_COUNT] = {
    /* 0 */ "STATUS_POST_CODE",
    /* 1 */ "DMC_CABLE_POWER_LIMIT (0xCAB1....)",
};

static const char* const bh_scratch_ram_notes[ARC_SCRATCH_RAM_COUNT] = {
    /*  0 */ "STATUS_FW_VERSION",
    /*  1 */ "reserved (bootcode security handshake)",
    /*  2 */ "STATUS_BOOT_STATUS0 (booted when (v & 7) == 5)",
    /*  3 */ "STATUS_BOOT_STATUS1",
    /*  4 */ "STATUS_ERROR_STATUS0",
    /*  5 */ "STATUS_ERROR_STATUS1",
    /*  6 */ "STATUS_INTERFACE_TABLE_BASE",
    /*  7 */ "reserved (future interface table)",
    /*  8 */ "STATUS_MSG_Q_STATUS",
    /*  9 */ "STATUS_MSG_Q_ERR_FLAGS",
    /* 10 */ "SPI_BUFFER_INFO",
    /* 11 */ "STATUS_MSG_Q_INFO",
    /* 12 */ "TELEMETRY_DATA ptr",
    /* 13 */ "TELEMETRY_TABLE ptr",
    /* 14 */ "PCIE_INIT_CPL_TIME",
    /* 15 */ "CMFW_START_TIME",
    /* 16 */ "ARC_START_TIME",
    /* 17 */ "PERST_TO_DMFW_INIT_DONE",
    /* 18 */ "PING_DMFW_DURATION",
    /* 19 */ "I2C0_TARGET_DEBUG_STATE",
    /* 20 */ "I2C0_TARGET_DEBUG_STATE_2",
    /* 21 */ "ARC_HANG_PC",
    /* 22 */ "RUNTIME_TELEMETRY_ADDR",
    /* 23 */ "RUNTIME_TELEMETRY_SIZE",
    /* 24 */ nullptr,
    /* 25 */ nullptr,
    /* 26 */ nullptr,
    /* 27 */ nullptr,
    /* 28 */ nullptr,
    /* 29 */ nullptr,
    /* 30 */ nullptr,
    /* 31 */ nullptr,
    /* 32 */ nullptr,
    /* 33 */ nullptr,
    /* 34 */ nullptr,
    /* 35 */ nullptr,
    /* 36 */ nullptr,
    /* 37 */ nullptr,
    /* 38 */ nullptr,
    /* 39 */ nullptr,
    /* 40 */ "STATUS_FW_VUART(0)",
    /* 41 */ "STATUS_FW_VUART(1)",
    /* 42 */ nullptr,
    /* 43 */ nullptr,
    /* 44 */ nullptr,
    /* 45 */ nullptr,
    /* 46 */ nullptr,
    /* 47 */ nullptr,
    /* 48 */ nullptr,
    /* 49 */ nullptr,
    /* 50 */ nullptr,
    /* 51 */ nullptr,
    /* 52 */ nullptr,
    /* 53 */ nullptr,
    /* 54 */ nullptr,
    /* 55 */ nullptr,
    /* 56 */ nullptr,
    /* 57 */ nullptr,
    /* 58 */ nullptr,
    /* 59 */ nullptr,
    /* 60 */ nullptr,
    /* 61 */ nullptr,
    /* 62 */ nullptr,
    /* 63 */ "STATUS_FW_SCRATCH",
};

// Dumps a contiguous bank of 32-bit registers starting at base_addr.  @notes
// annotates registers with a firmware-assigned role; it may be NULL, as may
// individual entries.
static void dump_bank(
    struct noc_window* win,
    uint16_t x,
    uint16_t y,
    const char* label,
    uint64_t base_addr,
    unsigned count,
    const char* const* notes) {
    printf("\n%s (ARC NOC (%u, %u), base 0x%llx):\n", label, x, y, (unsigned long long)base_addr);
    fflush(stdout);

    for (unsigned i = 0; i < count; ++i) {
        uint64_t addr = base_addr + (uint64_t)i * sizeof(uint32_t);
        uint32_t value = noc_read32(win, x, y, addr);

        if (notes != nullptr && notes[i] != nullptr) {
            printf("  [%2u] 0x%09llx = 0x%08x  %s\n", i, (unsigned long long)addr, value, notes[i]);
        } else {
            printf("  [%2u] 0x%09llx = 0x%08x\n", i, (unsigned long long)addr, value);
        }
    }

    fflush(stdout);
}

static void scratch_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [DEVICE]\n", prog);
    fprintf(stderr, "       %s -d DEVICE\n", prog);
    fprintf(stderr, "  -d PATH  Tenstorrent device to open, path or ordinal (default /dev/tenstorrent/0)\n");
    fprintf(stderr, "Exit status:\n");
    fprintf(stderr, "  0 = success\n");
    fprintf(stderr, "  1 = usage or setup error\n");
}

static int scratch_main(int argc, char* argv[], const char* prog) {
    const char* device_path = "/dev/tenstorrent/0";
    int device_path_set = 0;
    struct chip chip;
    struct noc_window win;
    int opt;

    while ((opt = getopt(argc, argv, "d:h")) != -1) {
        switch (opt) {
            case 'd':
                device_path = device_path_arg(optarg);
                device_path_set = 1;
                break;
            case 'h': scratch_usage(prog); return EXIT_OK;
            default: scratch_usage(prog); return EXIT_SETUP_ERROR;
        }
    }
    if (optind < argc) {
        if (device_path_set) {
            fprintf(stderr, "Error: device given both as -d %s and as '%s'\n", device_path, argv[optind]);
            scratch_usage(prog);
            return EXIT_SETUP_ERROR;
        }
        if (argc - optind > 1) {
            fprintf(stderr, "Error: unexpected extra argument '%s'\n", argv[optind + 1]);
            scratch_usage(prog);
            return EXIT_SETUP_ERROR;
        }
        device_path = device_path_arg(argv[optind]);
    }

    printf("Opening device: %s\n", device_path);
    fflush(stdout);
    chip_open(&chip, device_path);

    printf(
        "Device is %s (device_id=0x%04x), ARC at NOC (%u, %u)\n",
        chip.arch_name,
        chip.info.device_id,
        chip.arc_x,
        chip.arc_y);
    fflush(stdout);

    noc_window_open(&win, chip.fd);

    dump_bank(
        &win,
        chip.arc_x,
        chip.arc_y,
        "SCRATCH",
        ARC_RESET_UNIT_BASE + ARC_SCRATCH_OFFSET,
        ARC_SCRATCH_COUNT,
        chip.is_blackhole ? bh_scratch_notes : wh_scratch_notes);

    if (chip.is_blackhole) {
        dump_bank(
            &win,
            chip.arc_x,
            chip.arc_y,
            "SCRATCH_RAM",
            ARC_RESET_UNIT_BASE + ARC_SCRATCH_RAM_OFFSET,
            ARC_SCRATCH_RAM_COUNT,
            bh_scratch_ram_notes);
    } else {
        dump_bank(
            &win,
            chip.arc_x,
            chip.arc_y,
            "NOC_NODEID (telemetry pointers)",
            ARC_RESET_UNIT_BASE + WH_NOC_NODEID_OFFSET,
            WH_NOC_NODEID_COUNT,
            wh_nodeid_notes);
    }

    noc_window_close(&win);
    chip_close(&chip);
    return EXIT_OK;
}

// ============================================================================
// telemetry: dump every telemetry tag published by ARC FW
// ============================================================================
//
// Firmware publishes a table pointer and a data pointer.  The table contains
// a version, entry count, and tag-to-offset directory.

// Wormhole publishes table then data; Blackhole publishes data then table.
#define WH_TELEMETRY_TABLE_PTR_ADDR (ARC_RESET_UNIT_BASE + 0x1D0)
#define WH_TELEMETRY_DATA_PTR_ADDR (ARC_RESET_UNIT_BASE + 0x1D4)

// Wormhole publishes ARC-local addresses; set bit 35 for NOC access.
#define WH_NOC_ADDR_HIGH_BIT 0x800000000ULL

// One past the last Wormhole ARC CSM byte, as a NOC address.
#define WH_CSM_NOC_END (WH_NOC_ADDR_HIGH_BIT | 0x10080000ULL)

#define BH_SCRATCH_RAM_BASE (ARC_RESET_UNIT_BASE + 0x400)
#define BH_TELEMETRY_DATA_PTR_ADDR (BH_SCRATCH_RAM_BASE + 4 * 12)
#define BH_TELEMETRY_TABLE_PTR_ADDR (BH_SCRATCH_RAM_BASE + 4 * 13)

// Telemetry table layout, identical on both architectures:
//   +0x00                 version
//   +0x04                 entry_count
//   +0x08 + 4*i           directory entry i, (offset << 16) | tag
// A value lives at data_base + 4 * offset.  On Blackhole the directory is a
// struct telemetry_entry { u16 tag; u16 offset; } array, which little-endian
// ARC memory makes bit-identical to Wormhole's packed-word directory.  Wormhole
// genuinely publishes a separate data pointer; do not infer data_base from the
// end of its directory.
#define TELEMETRY_VERSION_OFFSET 0x00
#define TELEMETRY_ENTRY_COUNT_OFFSET 0x04
#define TELEMETRY_DIRECTORY_OFFSET 0x08

// Bounds used to reject corrupt directories before issuing unrelated reads.
#define MAX_ENTRY_COUNT 1024

#define MAX_VALUE_OFFSET 1024

// One past the highest tag number for each architecture.
#define WH_TAG_COUNT 82
#define BH_TAG_COUNT 80
#define MAX_TAG_COUNT WH_TAG_COUNT

// Tags referenced by number, in the decoders below or by the info
// subcommand.
#define TAG_BOARD_ID_HIGH 1
#define TAG_BOARD_ID_LOW 2
#define TAG_VCORE 6
#define TAG_TDP 7
#define TAG_TDC 8
#define TAG_ASIC_TEMPERATURE 11
#define TAG_AICLK 14
#define TAG_ETH_LIVE_STATUS 21
#define TAG_GDDR_STATUS 22
#define TAG_GDDR_SPEED 23
#define TAG_FLASH_BUNDLE_VERSION 28
#define TAG_FAN_SPEED 31
#define TAG_ENABLED_GDDR 36
#define TAG_FAN_RPM 41
#define TAG_ASIC_ID_HIGH 61
#define TAG_ASIC_ID_LOW 62
#define TAG_FAN_2_SPEED 78
#define TAG_FAN_2_RPM 79

// Tag names, indexed by tag number.  Transcribed from telemetry.h in
// tt-system-firmware (lib/tenstorrent/bh_arc/telemetry.h).  Index 0 is unused:
// tag 0 is not a tag, it is what an unpopulated directory slot reads back as.
static const char* const bh_tag_names[BH_TAG_COUNT] = {
    /*  0 */ nullptr,
    /*  1 */ "TAG_BOARD_ID_HIGH",
    /*  2 */ "TAG_BOARD_ID_LOW",
    /*  3 */ "TAG_ASIC_ID",
    /*  4 */ "TAG_HARVESTING_STATE",
    /*  5 */ "TAG_UPDATE_TELEM_SPEED",
    /*  6 */ "TAG_VCORE",
    /*  7 */ "TAG_TDP",
    /*  8 */ "TAG_TDC",
    /*  9 */ "TAG_VDD_LIMITS",
    /* 10 */ "TAG_THM_LIMIT_SHUTDOWN",
    /* 11 */ "TAG_ASIC_TEMPERATURE",
    /* 12 */ "TAG_VREG_TEMPERATURE",
    /* 13 */ "TAG_BOARD_TEMPERATURE",
    /* 14 */ "TAG_AICLK",
    /* 15 */ "TAG_AXICLK",
    /* 16 */ "TAG_ARCCLK",
    /* 17 */ "TAG_L2CPUCLK0",
    /* 18 */ "TAG_L2CPUCLK1",
    /* 19 */ "TAG_L2CPUCLK2",
    /* 20 */ "TAG_L2CPUCLK3",
    /* 21 */ "TAG_ETH_LIVE_STATUS",
    /* 22 */ "TAG_GDDR_STATUS",
    /* 23 */ "TAG_GDDR_SPEED",
    /* 24 */ "TAG_ETH_FW_VERSION",
    /* 25 */ "TAG_GDDR_FW_VERSION",
    /* 26 */ "TAG_DM_APP_FW_VERSION",
    /* 27 */ "TAG_DM_BL_FW_VERSION",
    /* 28 */ "TAG_FLASH_BUNDLE_VERSION",
    /* 29 */ "TAG_CM_FW_VERSION",
    /* 30 */ "TAG_L2CPU_FW_VERSION",
    /* 31 */ "TAG_FAN_SPEED",
    /* 32 */ "TAG_TIMER_HEARTBEAT",
    /* 33 */ "TAG_TELEM_ENUM_COUNT",
    /* 34 */ "TAG_ENABLED_TENSIX_COL",
    /* 35 */ "TAG_ENABLED_ETH",
    /* 36 */ "TAG_ENABLED_GDDR",
    /* 37 */ "TAG_ENABLED_L2CPU",
    /* 38 */ "TAG_PCIE_USAGE",
    /* 39 */ "TAG_INPUT_CURRENT",
    /* 40 */ "TAG_NOC_TRANSLATION",
    /* 41 */ "TAG_FAN_RPM",
    /* 42 */ "TAG_GDDR_0_1_TEMP",
    /* 43 */ "TAG_GDDR_2_3_TEMP",
    /* 44 */ "TAG_GDDR_4_5_TEMP",
    /* 45 */ "TAG_GDDR_6_7_TEMP",
    /* 46 */ "TAG_GDDR_0_1_CORR_ERRS",
    /* 47 */ "TAG_GDDR_2_3_CORR_ERRS",
    /* 48 */ "TAG_GDDR_4_5_CORR_ERRS",
    /* 49 */ "TAG_GDDR_6_7_CORR_ERRS",
    /* 50 */ "TAG_GDDR_UNCORR_ERRS",
    /* 51 */ "TAG_MAX_GDDR_TEMP",
    /* 52 */ "TAG_ASIC_LOCATION",
    /* 53 */ "TAG_BOARD_POWER_LIMIT",
    /* 54 */ "TAG_INPUT_POWER",
    /* 55 */ "TAG_TDC_LIMIT_MAX",
    /* 56 */ "TAG_THM_LIMIT_THROTTLE",
    /* 57 */ "TAG_FW_BUILD_DATE",
    /* 58 */ "TAG_TT_FLASH_VERSION",
    /* 59 */ "TAG_ENABLED_TENSIX_ROW",
    /* 60 */ "TAG_THERM_TRIP_COUNT",
    /* 61 */ "TAG_ASIC_ID_HIGH",
    /* 62 */ "TAG_ASIC_ID_LOW",
    /* 63 */ "TAG_AICLK_LIMIT_MAX",
    /* 64 */ "TAG_TDP_LIMIT_MAX",
    /* 65 */ "TAG_AICLK_ARB_MIN",
    /* 66 */ "TAG_AICLK_ARB_MAX",
    /* 67 */ "TAG_ENABLED_MIN_ARB",
    /* 68 */ "TAG_ENABLED_MAX_ARB",
    /* 69 */ "TAG_AICLK_PPM_INFO",
    /* 70 */ "TAG_HOST_AICLK_LIMIT",
    /* 71 */ "TAG_SMBUS_ERRORS",
    /* 72 */ "TAG_GDDR_MRISC_NOC2AXI_PORT",
    /* 73 */ "TAG_GDDR_WEST_IO_POWER",
    /* 74 */ "TAG_GDDR_EAST_IO_POWER",
    /* 75 */ "TAG_KERNEL_THROTTLER",
    /* 76 */ "TAG_NOP_START_COUNT",
    /* 77 */ "TAG_NOP_ON_DURATION",
    /* 78 */ "TAG_FW_CAPABILITIES_0",
    /* 79 */ "TAG_FW_ACTIVE_CONFIG_0",
};

// Wormhole defines tags 1..64.  Keep this separate from Blackhole's table:
// the architectures deliberately use BM versus DM names for tags 26 and 27.
static const char* const wh_tag_names[WH_TAG_COUNT] = {
    /*  0 */ nullptr,
    /*  1 */ "TAG_BOARD_ID_HIGH",
    /*  2 */ "TAG_BOARD_ID_LOW",
    /*  3 */ "TAG_ASIC_ID",
    /*  4 */ "TAG_HARVESTING_STATE",
    /*  5 */ "TAG_UPDATE_TELEM_SPEED",
    /*  6 */ "TAG_VCORE",
    /*  7 */ "TAG_TDP",
    /*  8 */ "TAG_TDC",
    /*  9 */ "TAG_VDD_LIMITS",
    /* 10 */ "TAG_THM_LIMIT_SHUTDOWN",
    /* 11 */ "TAG_ASIC_TEMPERATURE",
    /* 12 */ "TAG_VREG_TEMPERATURE",
    /* 13 */ "TAG_BOARD_TEMPERATURE",
    /* 14 */ "TAG_AICLK",
    /* 15 */ "TAG_AXICLK",
    /* 16 */ "TAG_ARCCLK",
    /* 17 */ "TAG_L2CPUCLK0",
    /* 18 */ "TAG_L2CPUCLK1",
    /* 19 */ "TAG_L2CPUCLK2",
    /* 20 */ "TAG_L2CPUCLK3",
    /* 21 */ "TAG_ETH_LIVE_STATUS",
    /* 22 */ "TAG_GDDR_STATUS",
    /* 23 */ "TAG_GDDR_SPEED",
    /* 24 */ "TAG_ETH_FW_VERSION",
    /* 25 */ "TAG_GDDR_FW_VERSION",
    /* 26 */ "TAG_BM_APP_FW_VERSION",
    /* 27 */ "TAG_BM_BL_FW_VERSION",
    /* 28 */ "TAG_FLASH_BUNDLE_VERSION",
    /* 29 */ "TAG_CM_FW_VERSION",
    /* 30 */ "TAG_L2CPU_FW_VERSION",
    /* 31 */ "TAG_FAN_SPEED",
    /* 32 */ "TAG_TIMER_HEARTBEAT",
    /* 33 */ "TAG_TELEM_ENUM_COUNT",
    /* 34 */ "TAG_ENABLED_TENSIX_COL",
    /* 35 */ "TAG_ENABLED_ETH",
    /* 36 */ "TAG_ENABLED_GDDR",
    /* 37 */ "TAG_ENABLED_L2CPU",
    /* 38 */ "TAG_PCIE_USAGE",
    /* 39 */ "TAG_INPUT_CURRENT",
    /* 40 */ "TAG_NOC_TRANSLATION",
    /* 41 */ "TAG_FAN_RPM",
    /* 42 */ "TAG_GDDR_0_1_TEMP",
    /* 43 */ "TAG_GDDR_2_3_TEMP",
    /* 44 */ "TAG_GDDR_4_5_TEMP",
    /* 45 */ "TAG_GDDR_6_7_TEMP",
    /* 46 */ "TAG_GDDR_0_1_CORR_ERRS",
    /* 47 */ "TAG_GDDR_2_3_CORR_ERRS",
    /* 48 */ "TAG_GDDR_4_5_CORR_ERRS",
    /* 49 */ "TAG_GDDR_6_7_CORR_ERRS",
    /* 50 */ "TAG_GDDR_UNCORR_ERRS",
    /* 51 */ "TAG_MAX_GDDR_TEMP",
    /* 52 */ "TAG_ASIC_LOCATION",
    /* 53 */ "TAG_BOARD_POWER_LIMIT",
    /* 54 */ "TAG_INPUT_POWER",
    /* 55 */ "TAG_TDC_LIMIT_MAX",
    /* 56 */ "TAG_THM_LIMIT_THROTTLE",
    /* 57 */ "TAG_FW_BUILD_DATE",
    /* 58 */ "TAG_TT_FLASH_VERSION",
    /* 59 */ "TAG_ENABLED_TENSIX_ROW",
    /* 60 */ "TAG_THERM_TRIP_COUNT",
    /* 61 */ "TAG_ASIC_ID_HIGH",
    /* 62 */ "TAG_ASIC_ID_LOW",
    /* 63 */ "TAG_AICLK_LIMIT_MAX",
    /* 64 */ "TAG_TDP_LIMIT_MAX",
    /* 65 */ nullptr,
    /* 66 */ nullptr,
    /* 67 */ nullptr,
    /* 68 */ nullptr,
    /* 69 */ nullptr,
    /* 70 */ nullptr,
    /* 71 */ nullptr,
    /* 72 */ nullptr,
    /* 73 */ nullptr,
    /* 74 */ nullptr,
    /* 75 */ nullptr,
    /* 76 */ nullptr,
    /* 77 */ nullptr,
    // 65..77 are not Wormhole tags.
    /* 78 */ "TAG_FAN_2_SPEED",
    /* 79 */ "TAG_FAN_2_RPM",
    /* 80 */ "TAG_FAN_3_SPEED",
    /* 81 */ "TAG_FAN_3_RPM",
};

// The per-architecture telemetry layout.  The ARC coordinates and architecture
// name live in struct chip; this holds what is specific to the telemetry walk.
struct telemetry_arch {
    uint64_t table_ptr_addr;
    uint64_t data_ptr_addr;
    uint64_t pointer_fixup;
    const char* const* tag_names;
    unsigned tag_count;
    int is_blackhole;
};

static const struct telemetry_arch wormhole_telemetry = {
    .table_ptr_addr = WH_TELEMETRY_TABLE_PTR_ADDR,
    .data_ptr_addr = WH_TELEMETRY_DATA_PTR_ADDR,
    .pointer_fixup = WH_NOC_ADDR_HIGH_BIT,
    .tag_names = wh_tag_names,
    .tag_count = WH_TAG_COUNT,
    .is_blackhole = 0,
};

static const struct telemetry_arch blackhole_telemetry = {
    .table_ptr_addr = BH_TELEMETRY_TABLE_PTR_ADDR,
    .data_ptr_addr = BH_TELEMETRY_DATA_PTR_ADDR,
    .pointer_fixup = 0,
    .tag_names = bh_tag_names,
    .tag_count = BH_TAG_COUNT,
    .is_blackhole = 1,
};

// Result of a checked read.  The distinct all-ones status exists so that no
// call site can accidentally treat 0xFFFFFFFF as data.
enum read_status {
    READ_OK = 0,         // the read completed and the value is not all ones
    READ_ALL_ONES = -1,  // 0xFFFFFFFF came back, i.e. the read did not reach the chip
};

// Classify the bus idle pattern separately from data.  all_ones_ok suppresses
// the diagnostic for tag values whose encoding may legitimately be all ones.
static int noc_read32_checked(
    struct noc_window* win,
    uint16_t x,
    uint16_t y,
    uint64_t addr,
    const char* what,
    int all_ones_ok,
    uint32_t* out_value) {
    uint32_t value = noc_read32(win, x, y, addr);

    if (value == ALL_ONES) {
        if (!all_ones_ok) {
            fprintf(
                stderr,
                "\nError: read all ones from %s at NOC (%u, %u) 0x%llx.\n"
                "The read did not reach the chip, so every subsequent value would be\n"
                "garbage.  Stopping here rather than interacting with the chip further.\n",
                what,
                x,
                y,
                (unsigned long long)addr);
        }
        return READ_ALL_ONES;
    }

    *out_value = value;
    return READ_OK;
}

// A STRUCTURAL read that comes back all ones is fatal: either pointer register,
// the table header, or any directory entry.
//
// A tag VALUE that reads all ones is reported rather than aborting the dump.
// Blackhole firmware uses all ones for fan tags when fan control is disabled.
static int tag_all_ones_is_documented(const struct telemetry_arch* arch, unsigned tag) {
    return arch->is_blackhole && (tag == TAG_FAN_SPEED || tag == TAG_FAN_RPM);
}

// Wormhole firmware versions use incompatible encodings for these tags without
// distinguishing them through the telemetry format version.  Do not guess.
static int tag_encoding_is_firmware_dependent(const struct telemetry_arch* arch, unsigned tag) {
    if (arch->is_blackhole) {
        return 0;
    }

    return tag == TAG_FAN_SPEED || tag == TAG_FAN_RPM || tag == TAG_ETH_LIVE_STATUS || tag == TAG_FAN_2_SPEED ||
           tag == TAG_FAN_2_RPM;
}

// What to say in the decoded column for a tag we refuse to decode.
#define UNDECODABLE_NOTE "not decoded (encoding differs between firmware versions)"

// Whether this architecture assigns any meaning to the number at all.  The
// Wormhole tag space has a hole at 65..77, and those numbers are not tags that
// happen to be unpublished; they are not tags.
static int tag_is_defined(const struct telemetry_arch* arch, unsigned tag) {
    return tag < arch->tag_count && arch->tag_names[tag] != nullptr;
}

// Fills buf with the tag's symbolic name, or TAG_UNKNOWN_<n> for a tag a newer
// firmware added that we do not know about.
static void tag_name(const struct telemetry_arch* arch, unsigned tag, char* buf, size_t bufsz) {
    if (tag_is_defined(arch, tag)) {
        snprintf(buf, bufsz, "%s", arch->tag_names[tag]);
    } else {
        snprintf(buf, bufsz, "TAG_UNKNOWN_%u", tag);
    }
}

// Renders the decoded form of a tag value.  Tags whose encoding we do not know
// get the plain unsigned decimal, which is what the raw hex means anyway.
static void decode_blackhole_value(unsigned tag, uint32_t v, char* buf, size_t bufsz) {
    switch (tag) {
        case 4:   // TAG_HARVESTING_STATE
        case 12:  // TAG_VREG_TEMPERATURE
        case 13:  // TAG_BOARD_TEMPERATURE
        case 30:  // TAG_L2CPU_FW_VERSION
            // These zero placeholders are not temperature measurements.
            if (v == 0) {
                snprintf(buf, bufsz, "0 (firmware does not populate this tag)");
            } else {
                snprintf(buf, bufsz, "%u", v);
            }
            break;
        case 6:  // TAG_VCORE
            snprintf(buf, bufsz, "%u mV", v);
            break;
        case 7:   // TAG_TDP
        case 53:  // TAG_BOARD_POWER_LIMIT
        case 54:  // TAG_INPUT_POWER
        case 64:  // TAG_TDP_LIMIT_MAX
        case 73:  // TAG_GDDR_WEST_IO_POWER
        case 74:  // TAG_GDDR_EAST_IO_POWER
            snprintf(buf, bufsz, "%u W", v);
            break;
        case 8:   // TAG_TDC
        case 55:  // TAG_TDC_LIMIT_MAX
            snprintf(buf, bufsz, "%u A", v);
            break;
        case 9:  // TAG_VDD_LIMITS
            snprintf(buf, bufsz, "vdd_max %u mV, vdd_min %u mV", (v >> 16) & 0xFFFF, v & 0xFFFF);
            break;
        case 10:  // TAG_THM_LIMIT_SHUTDOWN
        case 51:  // TAG_MAX_GDDR_TEMP
        case 56:  // TAG_THM_LIMIT_THROTTLE
            snprintf(buf, bufsz, "%u C", v);
            break;
        case TAG_ASIC_TEMPERATURE:
            if (v == 0x80000000u) {
                snprintf(buf, bufsz, "error / not available");
            } else {
                snprintf(buf, bufsz, "%.2f C", (double)(int32_t)v / 65536.0);
            }
            break;
        case 14:  // TAG_AICLK
        case 15:  // TAG_AXICLK
        case 16:  // TAG_ARCCLK
        case 17:  // TAG_L2CPUCLK0
        case 18:  // TAG_L2CPUCLK1
        case 19:  // TAG_L2CPUCLK2
        case 20:  // TAG_L2CPUCLK3
        case 63:  // TAG_AICLK_LIMIT_MAX
            snprintf(buf, bufsz, "%u MHz", v);
            break;
        case 21:  // TAG_ETH_LIVE_STATUS
            snprintf(buf, bufsz, "heartbeat 0x%04x, link-up 0x%04x", v & 0xFFFF, (v >> 16) & 0xFFFF);
            break;
        case 22: {  // TAG_GDDR_STATUS
            // Four per-instance fields interleaved two bits at a time:
            // bit 2i training complete, bit 2i+1 error, bit 16+2i BIST complete,
            // bit 17+2i BIST failed, for GDDR instance i.  Repacking them into
            // per-instance masks makes them comparable with TAG_ENABLED_GDDR.
            unsigned trained = 0, error = 0, bist_done = 0, bist_fail = 0;

            for (unsigned i = 0; i < 8; ++i) {
                trained |= ((v >> (2 * i)) & 1) << i;
                error |= ((v >> (2 * i + 1)) & 1) << i;
                bist_done |= ((v >> (16 + 2 * i)) & 1) << i;
                bist_fail |= ((v >> (17 + 2 * i)) & 1) << i;
            }
            snprintf(
                buf,
                bufsz,
                "trained 0x%02x, error 0x%02x, BIST done 0x%02x, BIST failed 0x%02x",
                trained,
                error,
                bist_done,
                bist_fail);
            break;
        }
        case 23:  // TAG_GDDR_SPEED
            snprintf(buf, bufsz, "%u Mbps", v);
            break;
        case 25:  // TAG_GDDR_FW_VERSION
            snprintf(buf, bufsz, "%u.%u", (v >> 16) & 0xFFFF, v & 0xFFFF);
            break;
        case 26:  // TAG_DM_APP_FW_VERSION
        case 27:  // TAG_DM_BL_FW_VERSION
        case 28:  // TAG_FLASH_BUNDLE_VERSION
        case 29:  // TAG_CM_FW_VERSION
            // Zephyr APPVERSION packing.
            snprintf(buf, bufsz, "%u.%u.%u.%u", (v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF);
            break;
        case TAG_FAN_SPEED: snprintf(buf, bufsz, "%u %%", v); break;
        case 38:  // TAG_PCIE_USAGE
            snprintf(buf, bufsz, "pcie0 mode %u, pcie1 mode %u", v & 0x3, (v >> 2) & 0x3);
            break;
        case 40:  // TAG_NOC_TRANSLATION
            snprintf(buf, bufsz, "%s", v ? "true" : "false");
            break;
        case TAG_FAN_RPM: snprintf(buf, bufsz, "%u RPM", v); break;
        case 42:    // TAG_GDDR_0_1_TEMP
        case 43:    // TAG_GDDR_2_3_TEMP
        case 44:    // TAG_GDDR_4_5_TEMP
        case 45: {  // TAG_GDDR_6_7_TEMP
            unsigned n = (tag - 42) * 2;
            snprintf(
                buf,
                bufsz,
                "GDDR%u %u/%u C, GDDR%u %u/%u C (bottom/top)",
                n,
                v & 0xFF,
                (v >> 8) & 0xFF,
                n + 1,
                (v >> 16) & 0xFF,
                (v >> 24) & 0xFF);
            break;
        }
        case 46:    // TAG_GDDR_0_1_CORR_ERRS
        case 47:    // TAG_GDDR_2_3_CORR_ERRS
        case 48:    // TAG_GDDR_4_5_CORR_ERRS
        case 49: {  // TAG_GDDR_6_7_CORR_ERRS
            // Corrected EDC counts saturate at 255.
            unsigned n = (tag - 46) * 2;

            snprintf(
                buf,
                bufsz,
                "GDDR%u rd %u wr %u, GDDR%u rd %u wr %u",
                n,
                v & 0xFF,
                (v >> 8) & 0xFF,
                n + 1,
                (v >> 16) & 0xFF,
                (v >> 24) & 0xFF);
            break;
        }
        case 50: {  // TAG_GDDR_UNCORR_ERRS
            // bit 2i: GDDR i had an uncorrected read EDC error.
            // bit 2i+1: the same for writes.  Sticky, not counts.
            unsigned rd = 0, wr = 0;

            for (unsigned i = 0; i < 8; ++i) {
                rd |= ((v >> (2 * i)) & 1) << i;
                wr |= ((v >> (2 * i + 1)) & 1) << i;
            }
            if (rd == 0 && wr == 0) {
                snprintf(buf, bufsz, "none");
            } else {
                snprintf(buf, bufsz, "read 0x%02x, write 0x%02x", rd, wr);
            }
            break;
        }
        case 65:  // TAG_AICLK_ARB_MIN
        case 66:  // TAG_AICLK_ARB_MAX
            snprintf(buf, bufsz, "%u MHz, arbiter %u", v & 0xFFFF, (v >> 16) & 0xFFFF);
            break;
        case 69:  // TAG_AICLK_PPM_INFO
            // union aiclk_targ_freq_info in aiclk_ppm.h declares arbiter:16 first
            // and reason:16 second, so arbiter is the low half.  That is the
            // opposite order from TAG_AICLK_ARB_MIN and _MAX above, which put the
            // frequency low and the arbiter high.
            snprintf(buf, bufsz, "reason %u, arbiter %u", (v >> 16) & 0xFFFF, v & 0xFFFF);
            break;
        case 70:  // TAG_HOST_AICLK_LIMIT
            if (v == 0) {
                snprintf(buf, bufsz, "0 (host has not set a limit)");
            } else {
                snprintf(buf, bufsz, "%u MHz", v);
            }
            break;
        case 72: {  // TAG_GDDR_MRISC_NOC2AXI_PORT
            // One nibble per GDDR instance, instance 0 in the low nibble.  0xF
            // means that instance is disabled or harvested.
            char ports[80];
            size_t used = 0;
            for (unsigned i = 0; i < 8; ++i) {
                unsigned nibble = (v >> (4 * i)) & 0xF;
                char port[8];
                int n;

                if (nibble == 0xF) {
                    snprintf(port, sizeof(port), "off");
                } else {
                    snprintf(port, sizeof(port), "%u", nibble);
                }

                n = snprintf(ports + used, sizeof(ports) - used, "%sGDDR%u:%s", i == 0 ? "" : " ", i, port);
                if (n < 0 || (size_t)n >= sizeof(ports) - used) {
                    break;
                }
                used += (size_t)n;
            }
            snprintf(buf, bufsz, "noc2axi port %s", ports);
            break;
        }
        case 75:  // TAG_KERNEL_THROTTLER
            snprintf(
                buf,
                bufsz,
                "%s, stop-NOPs %u MHz%s",
                (v & 1) ? "enabled" : "disabled",
                (v >> 16) & 0xFFFF,
                ((v >> 16) & 0xFFFF) == 0 ? " (FW default)" : "");
            break;
        case 77:  // TAG_NOP_ON_DURATION
            snprintf(buf, bufsz, "%u ms", v);
            break;
        case 78:  // TAG_FW_CAPABILITIES_0
        case 79:  // TAG_FW_ACTIVE_CONFIG_0
            // Feature bitfield; only bit 0 (kernel NOP throttler) is assigned.
            snprintf(
                buf,
                bufsz,
                "kernel-NOP-throttler %s%s",
                tag == 78 ? (v & 1 ? "supported" : "not supported") : (v & 1 ? "enabled" : "disabled"),
                (v & ~1u) ? ", other bits set" : "");
            break;
        default: snprintf(buf, bufsz, "%u", v); break;
    }
}

static void decode_wormhole_value(unsigned tag, uint32_t v, char* buf, size_t bufsz) {
    switch (tag) {
        case 6:  // TAG_VCORE
            snprintf(buf, bufsz, "%u mV", v);
            break;
        case 7:   // TAG_TDP
        case 64:  // TAG_TDP_LIMIT_MAX
            snprintf(buf, bufsz, "%u W", v);
            break;
        case 8:   // TAG_TDC
        case 55:  // TAG_TDC_LIMIT_MAX
            snprintf(buf, bufsz, "%u A", v);
            break;
        case 9:  // TAG_VDD_LIMITS
            snprintf(buf, bufsz, "vdd_max %u mV, vdd_min %u mV", (v >> 16) & 0xFFFF, v & 0xFFFF);
            break;
        case 10:  // TAG_THM_LIMIT_SHUTDOWN
        case 56:  // TAG_THM_LIMIT_THROTTLE
            snprintf(buf, bufsz, "%u C", v);
            break;
        case TAG_ASIC_TEMPERATURE:
        case 13:  // TAG_BOARD_TEMPERATURE
            // New-telemetry table values are signed 16.16 on Wormhole, just as on
            // Blackhole.  The often-quoted low-16-bits / 16 encoding is only for
            // Wormhole's separate SMBus telemetry path.
            snprintf(buf, bufsz, "%.2f C", (double)(int32_t)v / 65536.0);
            break;
        case 14:  // TAG_AICLK
        case 15:  // TAG_AXICLK
        case 16:  // TAG_ARCCLK
        case 17:  // TAG_L2CPUCLK0
        case 18:  // TAG_L2CPUCLK1
        case 19:  // TAG_L2CPUCLK2
        case 20:  // TAG_L2CPUCLK3
        case 63:  // TAG_AICLK_LIMIT_MAX
            snprintf(buf, bufsz, "%u MHz", v);
            break;
        case 22: {  // TAG_GDDR_STATUS
            // Firmware maps pass to 1, fail to 3, and all other states to 0.
            // It does not emit 2.
            size_t used = 0;

            for (unsigned channel = 0; channel < 6; ++channel) {
                unsigned status = (v >> (2 * channel)) & 0x3;
                const char* state = status == 0   ? "not trained"
                                    : status == 1 ? "trained"
                                    : status == 3 ? "training failed"
                                                  : "2 (firmware emits no such value)";
                int n = snprintf(buf + used, bufsz - used, "%sch%u %s", channel == 0 ? "" : ", ", channel, state);

                if (n < 0 || (size_t)n >= bufsz - used) {
                    break;
                }
                used += (size_t)n;
            }
            break;
        }
        case 23:  // TAG_GDDR_SPEED
            // Firmware converts its speed enum to the data rate in thousands of
            // Mbps, same meaning as Blackhole's.
            snprintf(buf, bufsz, "%u Mbps", v);
            break;
        case 24:  // TAG_ETH_FW_VERSION
            snprintf(buf, bufsz, "%u.%u.%u", (v >> 16) & 0xFF, (v >> 12) & 0xF, v & 0xFFF);
            break;
        case 26:  // TAG_BM_APP_FW_VERSION
        case 27:  // TAG_BM_BL_FW_VERSION
        case 28:  // TAG_FLASH_BUNDLE_VERSION
        case 58:  // TAG_TT_FLASH_VERSION
            snprintf(buf, bufsz, "%u.%u.%u.%u", (v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF);
            break;
        case 29:  // TAG_CM_FW_VERSION
            snprintf(buf, bufsz, "%u.%u.%u", (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF);
            break;
        case 38:  // TAG_PCIE_USAGE
            snprintf(buf, bufsz, "pcie0 mode %u, pcie1 mode %u", v & 0x3, (v >> 2) & 0x3);
            break;
        case 40:  // TAG_NOC_TRANSLATION
            snprintf(buf, bufsz, "%s", v ? "true" : "false");
            break;
        case 57:  // TAG_FW_BUILD_DATE
            snprintf(
                buf,
                bufsz,
                "%04u-%02u-%02u %02u:%02u",
                2020 + ((v >> 28) & 0xF),
                (v >> 24) & 0xF,
                (v >> 16) & 0xFF,
                (v >> 8) & 0xFF,
                v & 0xFF);
            break;
        default:
            // Leave tags without a known decode as unsigned decimal; the adjacent
            // Raw column supplies the exact hexadecimal value.
            snprintf(buf, bufsz, "%u", v);
            break;
    }
}

static void telemetry_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [-r] [DEVICE]\n", prog);
    fprintf(stderr, "       %s [-r] -d DEVICE\n", prog);
    fprintf(stderr, "  -d PATH  Tenstorrent device to open, path or ordinal (default /dev/tenstorrent/0)\n");
    fprintf(stderr, "  -r       Raw values only, no decoded column\n");
    fprintf(stderr, "  -h       This help\n");
    fprintf(stderr, "Exit status:\n");
    fprintf(stderr, "  0 = success\n");
    fprintf(stderr, "  1 = usage or setup error\n");
    fprintf(stderr, "  2 = chip not answering\n");
    fprintf(stderr, "  3 = chip answers but telemetry is unpublished or malformed\n");
}

static int telemetry_main(int argc, char* argv[], const char* prog) {
    const char* device_path = "/dev/tenstorrent/0";
    int device_path_set = 0;
    int raw_only = 0;
    int rc = EXIT_SETUP_ERROR;
    int r;
    int opt;
    struct chip chip;
    struct noc_window win;
    const struct telemetry_arch* arch;
    uint32_t table_ptr_raw;
    uint32_t data_ptr_raw;
    uint64_t table_base;
    uint64_t data_base;
    uint32_t version;
    uint32_t entry_count;
    uint16_t dir_tag[MAX_ENTRY_COUNT];
    uint16_t dir_offset[MAX_ENTRY_COUNT];
    unsigned dir_count = 0;
    int tag_present[MAX_TAG_COUNT];
    uint16_t tag_offset[MAX_TAG_COUNT];
    int have_board_id_high = 0, have_board_id_low = 0;
    int have_asic_id_high = 0, have_asic_id_low = 0;
    uint32_t board_id_high = 0, board_id_low = 0;
    uint32_t asic_id_high = 0, asic_id_low = 0;

    memset(tag_present, 0, sizeof(tag_present));
    memset(tag_offset, 0, sizeof(tag_offset));

    while ((opt = getopt(argc, argv, "d:rh")) != -1) {
        switch (opt) {
            case 'd':
                device_path = device_path_arg(optarg);
                device_path_set = 1;
                break;
            case 'r': raw_only = 1; break;
            case 'h': telemetry_usage(prog); return EXIT_OK;
            default: telemetry_usage(prog); return EXIT_SETUP_ERROR;
        }
    }

    // Accept either -d or one positional device, but not both.
    if (optind < argc) {
        if (device_path_set) {
            fprintf(stderr, "Error: device given both as -d %s and as '%s'\n", device_path, argv[optind]);
            telemetry_usage(prog);
            return EXIT_SETUP_ERROR;
        }
        if (argc - optind > 1) {
            fprintf(stderr, "Error: unexpected extra argument '%s'\n", argv[optind + 1]);
            telemetry_usage(prog);
            return EXIT_SETUP_ERROR;
        }
        device_path = device_path_arg(argv[optind]);
    }

    chip_open(&chip, device_path);
    arch = chip.is_blackhole ? &blackhole_telemetry : &wormhole_telemetry;

    noc_window_open(&win, chip.fd);

    // Pointer, header, and directory reads are structural.
    if (noc_read32_checked(
            &win,
            chip.arc_x,
            chip.arc_y,
            arch->table_ptr_addr,
            "the telemetry table pointer register",
            0,
            &table_ptr_raw) == READ_ALL_ONES) {
        rc = EXIT_CHIP_SILENT;
        goto cleanup;
    }

    if (noc_read32_checked(
            &win,
            chip.arc_x,
            chip.arc_y,
            arch->data_ptr_addr,
            "the telemetry data pointer register",
            0,
            &data_ptr_raw) == READ_ALL_ONES) {
        rc = EXIT_CHIP_SILENT;
        goto cleanup;
    }

    // Zero pointers mean firmware has not published telemetry.
    if (table_ptr_raw == 0 || data_ptr_raw == 0) {
        fprintf(
            stderr,
            "Error: firmware has not published telemetry: table pointer 0x%08x, "
            "data pointer 0x%08x\n",
            table_ptr_raw,
            data_ptr_raw);
        rc = EXIT_BAD_TELEMETRY;
        goto cleanup;
    }

    if (!arch->is_blackhole && (table_ptr_raw < 0x10000000u || table_ptr_raw > 0x1007FFFFu ||
                                data_ptr_raw < 0x10000000u || data_ptr_raw > 0x1007FFFFu)) {
        fprintf(
            stderr,
            "Error: implausible Wormhole telemetry pointers: table 0x%08x, data 0x%08x; "
            "expected ARC-local CSM addresses in 0x10000000..0x1007ffff\n",
            table_ptr_raw,
            data_ptr_raw);
        rc = EXIT_BAD_TELEMETRY;
        goto cleanup;
    }

    table_base = (uint64_t)table_ptr_raw | arch->pointer_fixup;
    data_base = (uint64_t)data_ptr_raw | arch->pointer_fixup;

    if (noc_read32_checked(
            &win,
            chip.arc_x,
            chip.arc_y,
            table_base + TELEMETRY_VERSION_OFFSET,
            "the telemetry version word",
            0,
            &version) == READ_ALL_ONES) {
        rc = EXIT_CHIP_SILENT;
        goto cleanup;
    }

    if (noc_read32_checked(
            &win,
            chip.arc_x,
            chip.arc_y,
            table_base + TELEMETRY_ENTRY_COUNT_OFFSET,
            "the telemetry entry_count word",
            0,
            &entry_count) == READ_ALL_ONES) {
        rc = EXIT_CHIP_SILENT;
        goto cleanup;
    }

    if (version == 0) {
        fprintf(stderr, "Error: telemetry format version is zero\n");
        rc = EXIT_BAD_TELEMETRY;
        goto cleanup;
    }

    if (entry_count == 0 || entry_count > MAX_ENTRY_COUNT) {
        fprintf(
            stderr,
            "Error: implausible telemetry entry_count %u (table pointer 0x%llx); "
            "this does not look like a telemetry table\n",
            entry_count,
            (unsigned long long)table_base);
        rc = EXIT_BAD_TELEMETRY;
        goto cleanup;
    }

    if (!arch->is_blackhole && table_base + TELEMETRY_DIRECTORY_OFFSET + (uint64_t)entry_count * 4 > WH_CSM_NOC_END) {
        fprintf(
            stderr,
            "Error: telemetry directory (%u entries at 0x%llx) extends past the "
            "end of ARC CSM\n",
            entry_count,
            (unsigned long long)table_base);
        rc = EXIT_BAD_TELEMETRY;
        goto cleanup;
    }

    printf("Device:         %s\n", device_path);
    printf("Architecture:   %s (device_id=0x%04x)\n", chip.arch_name, chip.info.device_id);
    printf(
        "PCI location:   %04x:%02x:%02x.%u\n",
        chip.info.pci_domain,
        (chip.info.bus_dev_fn >> 8) & 0xFF,
        (chip.info.bus_dev_fn >> 3) & 0x1F,
        chip.info.bus_dev_fn & 0x7);
    printf("ARC NOC:        (%u, %u)\n", chip.arc_x, chip.arc_y);
    printf(
        "Table pointer:  0x%llx (read 0x%08x from 0x%llx)\n",
        (unsigned long long)table_base,
        table_ptr_raw,
        (unsigned long long)arch->table_ptr_addr);
    printf(
        "Data pointer:   0x%llx (read 0x%08x from 0x%llx)\n",
        (unsigned long long)data_base,
        data_ptr_raw,
        (unsigned long long)arch->data_ptr_addr);
    printf(
        "Format version: %u.%u.%u (raw 0x%08x)\n",
        (version >> 16) & 0xFF,
        (version >> 8) & 0xFF,
        version & 0xFF,
        version);
    printf("Entry count:    %u\n", entry_count);
    fflush(stdout);

    // Build tag -> offset.  Tag 0 denotes an unpopulated directory slot.
    for (unsigned i = 0; i < entry_count; ++i) {
        uint64_t addr = table_base + TELEMETRY_DIRECTORY_OFFSET + (uint64_t)i * sizeof(uint32_t);
        char what[64];
        uint32_t entry;
        unsigned tag;
        unsigned offset;

        snprintf(what, sizeof(what), "telemetry directory entry %u", i);
        if (noc_read32_checked(&win, chip.arc_x, chip.arc_y, addr, what, 0, &entry) == READ_ALL_ONES) {
            rc = EXIT_CHIP_SILENT;
            goto cleanup;
        }

        tag = entry & 0xFFFF;
        if (tag == 0) {
            continue;
        }

        offset = (entry >> 16) & 0xFFFF;
        if (offset > MAX_VALUE_OFFSET) {
            fprintf(
                stderr,
                "Warning: directory entry %u (tag %u) has implausible offset %u; "
                "skipping it rather than reading an unrelated address\n",
                i,
                tag,
                offset);
            continue;
        }

        if (!arch->is_blackhole && data_base + (uint64_t)offset * 4 + 4 > WH_CSM_NOC_END) {
            fprintf(
                stderr,
                "Warning: directory entry %u (tag %u) offset %u puts its value "
                "past the end of ARC CSM; skipping it\n",
                i,
                tag,
                offset);
            continue;
        }

        dir_tag[dir_count] = (uint16_t)tag;
        dir_offset[dir_count] = (uint16_t)offset;
        ++dir_count;

        if (tag < arch->tag_count) {
            tag_present[tag] = 1;
            tag_offset[tag] = (uint16_t)offset;
        }
    }

    printf("Directory:      %u populated entries, %u empty slots\n\n", dir_count, entry_count - dir_count);

    if (raw_only) {
        printf("Tag  Name                          Raw\n");
        printf("---  ----------------------------  ----------\n");
    } else {
        printf("Tag  Name                          Raw         Decoded\n");
        printf("---  ----------------------------  ----------  -------\n");
    }

    // Tags known to this build first, in tag order, then anything a newer
    // firmware published that we do not have a name for.
    for (unsigned pass = 0; pass < 2; ++pass) {
        unsigned limit = (pass == 0) ? arch->tag_count : dir_count;

        for (unsigned i = (pass == 0) ? 1 : 0; i < limit; ++i) {
            unsigned tag;
            uint16_t offset;
            char name[40];
            char decoded[192];
            uint32_t value;
            uint64_t addr;
            char what[80];

            if (pass == 0) {
                tag = i;
                offset = tag_offset[tag];
                if (!tag_present[tag]) {
                    if (!tag_is_defined(arch, tag)) {
                        continue;
                    }

                    tag_name(arch, tag, name, sizeof(name));
                    if (raw_only) {
                        printf("%3u  %-28s  %-10s\n", tag, name, "absent");
                    } else {
                        printf("%3u  %-28s  %-10s  %s\n", tag, name, "absent", "absent (not in firmware directory)");
                    }
                    continue;
                }
            } else {
                tag = dir_tag[i];
                offset = dir_offset[i];
                if (tag < arch->tag_count) {
                    continue;  // already printed in pass 0
                }
            }

            tag_name(arch, tag, name, sizeof(name));
            addr = data_base + (uint64_t)offset * sizeof(uint32_t);
            snprintf(what, sizeof(what), "the value of %s (tag %u)", name, tag);

            r = noc_read32_checked(&win, chip.arc_x, chip.arc_y, addr, what, 1, &value);
            if (r == READ_ALL_ONES) {
                const char* note = tag_encoding_is_firmware_dependent(arch, tag) ? UNDECODABLE_NOTE
                                   : tag_all_ones_is_documented(arch, tag)
                                       ? "n/a (fan control disabled)"
                                       : "all ones (chip answered, value not meaningful)";

                if (raw_only) {
                    printf("%3u  %-28s  0x%08x\n", tag, name, ALL_ONES);
                } else {
                    printf("%3u  %-28s  0x%08x  %s\n", tag, name, ALL_ONES, note);
                }
                continue;
            }

            if (raw_only) {
                printf("%3u  %-28s  0x%08x\n", tag, name, value);
            } else {
                if (tag_encoding_is_firmware_dependent(arch, tag)) {
                    snprintf(decoded, sizeof(decoded), "%s", UNDECODABLE_NOTE);
                } else if (arch->is_blackhole) {
                    decode_blackhole_value(tag, value, decoded, sizeof(decoded));
                } else {
                    decode_wormhole_value(tag, value, decoded, sizeof(decoded));
                }
                printf("%3u  %-28s  0x%08x  %s\n", tag, name, value, decoded);
            }

            switch (tag) {
                case TAG_BOARD_ID_HIGH:
                    board_id_high = value;
                    have_board_id_high = 1;
                    break;
                case TAG_BOARD_ID_LOW:
                    board_id_low = value;
                    have_board_id_low = 1;
                    break;
                case TAG_ASIC_ID_HIGH:
                    asic_id_high = value;
                    have_asic_id_high = 1;
                    break;
                case TAG_ASIC_ID_LOW:
                    asic_id_low = value;
                    have_asic_id_low = 1;
                    break;
                default: break;
            }
        }
    }

    // Print IDs only when both halves were read.
    if (have_board_id_high && have_board_id_low) {
        printf("\nBoard ID: 0x%016llx\n", ((unsigned long long)board_id_high << 32) | board_id_low);
    }
    if (have_asic_id_high && have_asic_id_low) {
        printf("ASIC ID:  0x%016llx\n", ((unsigned long long)asic_id_high << 32) | asic_id_low);
    }

    // Detect a chip that stopped answering during non-fatal value reads.
    {
        uint32_t version_recheck;

        if (noc_read32_checked(
                &win,
                chip.arc_x,
                chip.arc_y,
                table_base + TELEMETRY_VERSION_OFFSET,
                "the telemetry version word (liveness re-check)",
                0,
                &version_recheck) == READ_ALL_ONES) {
            fprintf(
                stderr,
                "Error: the chip stopped answering during the dump; "
                "values above may include garbage\n");
            rc = EXIT_CHIP_SILENT;
            goto cleanup;
        }
    }

    fflush(stdout);
    rc = EXIT_OK;

cleanup:
    noc_window_close(&win);
    chip_close(&chip);
    return rc;
}

// ============================================================================
// read32 / write32: one 32-bit NOC access at an explicit (x, y, addr)
// ============================================================================
//
// One unicast word on NOC0.  Writes are posted.

// Parses an unsigned number, accepting 0x-prefixed hex, 0-prefixed octal and
// decimal.  Returns 0 and stores the value, or -1 if @s is not a number, is
// negative, or exceeds @max.
static int parse_u64(const char* s, uint64_t max, uint64_t* out) {
    unsigned long long v;
    char* end;

    if (s[0] == '-') {
        return -1;
    }
    errno = 0;
    v = strtoull(s, &end, 0);
    if (errno != 0 || end == s || *end != '\0' || v > max) {
        return -1;
    }
    *out = v;
    return 0;
}

static void read32_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [-d PATH] X Y ADDR\n", prog);
    fprintf(stderr, "  -d PATH  device to open, path or ordinal (default /dev/tenstorrent/0)\n");
    fprintf(stderr, "  X Y      NOC0 coordinates of the target node\n");
    fprintf(stderr, "  ADDR     byte address at the target, 4-byte aligned\n");
    fprintf(stderr, "\nPrints the value alone, as 0x%%08x.\n");
    fprintf(stderr, "Exit status:\n");
    fprintf(stderr, "  0 = read performed\n");
    fprintf(stderr, "  1 = usage or setup error\n");
}

static int read32_main(int argc, char* argv[], const char* prog) {
    const char* device_path = "/dev/tenstorrent/0";
    struct chip chip;
    struct noc_window win;
    uint64_t x, y, addr;
    int opt;

    while ((opt = getopt(argc, argv, "d:h")) != -1) {
        switch (opt) {
            case 'd': device_path = device_path_arg(optarg); break;
            case 'h': read32_usage(prog); return EXIT_OK;
            default: read32_usage(prog); return EXIT_SETUP_ERROR;
        }
    }

    if (argc - optind != 3) {
        read32_usage(prog);
        return EXIT_SETUP_ERROR;
    }
    if (parse_u64(argv[optind], 0xFFFF, &x) != 0 || parse_u64(argv[optind + 1], 0xFFFF, &y) != 0) {
        fprintf(stderr, "X and Y must be numbers 0..65535\n");
        return EXIT_SETUP_ERROR;
    }
    if (parse_u64(argv[optind + 2], UINT64_MAX, &addr) != 0 || addr % 4 != 0) {
        fprintf(stderr, "ADDR must be a 4-byte-aligned number\n");
        return EXIT_SETUP_ERROR;
    }

    chip_open(&chip, device_path);
    noc_window_open(&win, chip.fd);

    printf("0x%08x\n", noc_read32(&win, x, y, addr));

    noc_window_close(&win);
    chip_close(&chip);
    return EXIT_OK;
}

static void write32_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [-d PATH] X Y ADDR VALUE\n", prog);
    fprintf(stderr, "  -d PATH  device to open, path or ordinal (default /dev/tenstorrent/0)\n");
    fprintf(stderr, "  X Y      NOC0 coordinates of the target node\n");
    fprintf(stderr, "  ADDR     byte address at the target, 4-byte aligned\n");
    fprintf(stderr, "  VALUE    32-bit value to write\n");
    fprintf(stderr, "\nPrints nothing on success.\n");
    fprintf(stderr, "Exit status:\n");
    fprintf(stderr, "  0 = write issued\n");
    fprintf(stderr, "  1 = usage or setup error\n");
}

static int write32_main(int argc, char* argv[], const char* prog) {
    const char* device_path = "/dev/tenstorrent/0";
    struct chip chip;
    struct noc_window win;
    uint64_t x, y, addr, value;
    int opt;

    while ((opt = getopt(argc, argv, "d:h")) != -1) {
        switch (opt) {
            case 'd': device_path = device_path_arg(optarg); break;
            case 'h': write32_usage(prog); return EXIT_OK;
            default: write32_usage(prog); return EXIT_SETUP_ERROR;
        }
    }

    if (argc - optind != 4) {
        write32_usage(prog);
        return EXIT_SETUP_ERROR;
    }
    if (parse_u64(argv[optind], 0xFFFF, &x) != 0 || parse_u64(argv[optind + 1], 0xFFFF, &y) != 0) {
        fprintf(stderr, "X and Y must be numbers 0..65535\n");
        return EXIT_SETUP_ERROR;
    }
    if (parse_u64(argv[optind + 2], UINT64_MAX, &addr) != 0 || addr % 4 != 0) {
        fprintf(stderr, "ADDR must be a 4-byte-aligned number\n");
        return EXIT_SETUP_ERROR;
    }
    if (parse_u64(argv[optind + 3], 0xFFFFFFFF, &value) != 0) {
        fprintf(stderr, "VALUE must be a number 0..0xFFFFFFFF\n");
        return EXIT_SETUP_ERROR;
    }

    chip_open(&chip, device_path);
    noc_window_open(&win, chip.fd);

    noc_write32(&win, x, y, addr, (uint32_t)value);

    noc_window_close(&win);
    chip_close(&chip);
    return EXIT_OK;
}

// ============================================================================
// hung: is the chip hung?  Config space, sysfs telemetry, heartbeat, then NOC
// ============================================================================
//
// Check PCI config space, live sysfs telemetry, heartbeat progress, then one
// NOC read to ARC.  One deadline bounds the entire check.
#define HUNG_DEADLINE_SECONDS 25

#define HUNG_HEARTBEAT_WAIT_NS 500000000L

static void hung_alarm(int sig) {
    static const char msg[] = "[FAIL] deadline expired; a device read is blocked, the chip is not answering\n";
    ssize_t w = write(STDOUT_FILENO, msg, sizeof(msg) - 1);

    (void)sig;
    (void)w;
    _exit(EXIT_CHIP_SILENT);
}

// Read a sysfs attribute and trim trailing whitespace.
static int read_sysfs_attr(const char* dir, const char* name, char* buf, size_t bufsz) {
    char path[256];
    int fd;
    ssize_t n;
    int saved_errno;

    snprintf(path, sizeof(path), "%s/%s", dir, name);
    fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return -1;
    }

    n = read(fd, buf, bufsz - 1);
    if (n < 0) {
        saved_errno = errno;
        close(fd);
        errno = saved_errno;
        return -1;
    }
    close(fd);

    while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == ' ')) {
        n--;
    }
    buf[n] = '\0';
    return 0;
}

// Whether a formatted sysfs value is what an all-ones telemetry read
// produces.  The driver (telemetry.c) has four formatters, each with one
// all-ones rendering: "4294967295" (u32 decimal), "FFFFFFFFFFFFFFFF" (u64
// hex), "255.255.255.255" (version), "255.15.4095" (the ETH version
// packing).  card_type's "unknown" is not matched: a live chip with an
// unrecognized card type reads the same.
static int sysfs_value_is_all_ones(const char* v) {
    static const char* const all_ones[] = {
        "4294967295",
        "FFFFFFFFFFFFFFFF",
        "255.255.255.255",
        "255.15.4095",
    };
    size_t i;

    for (i = 0; i < sizeof(all_ones) / sizeof(all_ones[0]); i++) {
        if (strcmp(v, all_ones[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

static int attr_name_cmp(const void* a, const void* b) { return strcmp((const char*)a, (const char*)b); }

static void hung_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [DEVICE]\n", prog);
    fprintf(stderr, "       %s -d DEVICE\n", prog);
    fprintf(stderr, "  -d PATH  Tenstorrent device to check, path or ordinal (default /dev/tenstorrent/0)\n");
    fprintf(stderr, "\nPrints one [PASS] or [FAIL] line last.\n");
    fprintf(stderr, "Exit status:\n");
    fprintf(stderr, "  0 = chip answers, heartbeat advances, and NOC responds\n");
    fprintf(stderr, "  1 = usage or setup error\n");
    fprintf(stderr, "  2 = PCIe endpoint, telemetry, or NOC not answering\n");
    fprintf(stderr, "  3 = chip answers but firmware is sick\n");
}

#define HUNG_MAX_ATTRS 64
#define HUNG_ATTR_NAME_LEN 40

static int hung_main(int argc, char* argv[], const char* prog) {
    const char* device_path = "/dev/tenstorrent/0";
    int device_path_set = 0;
    struct noc_window win;
    struct chip chip;
    struct stat st;
    struct sigaction sa;
    char sysdir[64];
    char link[256];
    char value[128];
    char hb1[128], hb2[128];
    static char attrs[HUNG_MAX_ATTRS][HUNG_ATTR_NAME_LEN];
    unsigned attr_count = 0;
    unsigned values_read = 0;
    unsigned read_errors = 0;
    int have_heartbeat = 0;
    uint32_t noc_node_id;
    uint32_t config_id;
    DIR* dir;
    struct dirent* de;
    ssize_t n;
    int fd;
    int opt;

    while ((opt = getopt(argc, argv, "d:h")) != -1) {
        switch (opt) {
            case 'd':
                device_path = device_path_arg(optarg);
                device_path_set = 1;
                break;
            case 'h': hung_usage(prog); return EXIT_OK;
            default: hung_usage(prog); return EXIT_SETUP_ERROR;
        }
    }
    if (optind < argc) {
        if (device_path_set) {
            fprintf(stderr, "Error: device given both as -d %s and as '%s'\n", device_path, argv[optind]);
            hung_usage(prog);
            return EXIT_SETUP_ERROR;
        }
        if (argc - optind > 1) {
            fprintf(stderr, "Error: unexpected extra argument '%s'\n", argv[optind + 1]);
            hung_usage(prog);
            return EXIT_SETUP_ERROR;
        }
        device_path = device_path_arg(argv[optind]);
    }

    if (stat(device_path, &st) != 0) {
        DIE("cannot stat %s", device_path);
    }
    if (!S_ISCHR(st.st_mode)) {
        DIEX("%s is not a character device", device_path);
    }
    snprintf(sysdir, sizeof(sysdir), "/sys/dev/char/%u:%u", major(st.st_rdev), minor(st.st_rdev));

    {
        char path[sizeof(sysdir) + 8];
        const char* bdf = "?";

        snprintf(path, sizeof(path), "%s/device", sysdir);
        n = readlink(path, link, sizeof(link) - 1);
        if (n > 0) {
            link[n] = '\0';
            bdf = strrchr(link, '/');
            bdf = bdf ? bdf + 1 : link;
        }
        printf("%s: PCI device %s, telemetry via %s\n", device_path, bdf, sysdir);
        fflush(stdout);
    }

    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = hung_alarm;
    sigaction(SIGALRM, &sa, nullptr);
    alarm(HUNG_DEADLINE_SECONDS);

    // The first four config bytes are vendor and device IDs.
    {
        char path[sizeof(sysdir) + 16];

        snprintf(path, sizeof(path), "%s/device/config", sysdir);
        fd = open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            DIE("cannot open %s", path);
        }
        n = read(fd, &config_id, sizeof(config_id));
        close(fd);
        if (n < 0) {
            DIE("reading %s failed", path);
        }
        if (n != (ssize_t)sizeof(config_id)) {
            DIEX("config space read returned %zd bytes", n);
        }
    }
    printf("config space: vendor:device %04x:%04x\n", config_id & 0xFFFF, (config_id >> 16) & 0xFFFF);
    fflush(stdout);
    if (config_id == ALL_ONES) {
        verdict(
            0,
            "config space reads all ones; the endpoint is off the PCIe bus "
            "and only a link reset can bring it back");
        return EXIT_CHIP_SILENT;
    }

    if (read_sysfs_attr(sysdir, "tt_serial", value, sizeof(value)) != 0) {
        if (errno == ENOENT) {
            verdict(
                0,
                "tt_serial does not exist under %s; the driver has no "
                "telemetry for this device",
                sysdir);
        } else {
            verdict(
                0,
                "reading tt_serial failed: %s; the driver cannot serve "
                "telemetry from the chip",
                strerror(errno));
        }
        return EXIT_FW_SICK;
    }
    if (sysfs_value_is_all_ones(value)) {
        verdict(0, "tt_serial reads %s; the chip is not answering telemetry reads", value);
        return EXIT_CHIP_SILENT;
    }
    printf("tt_serial: %s\n", value);
    fflush(stdout);

    // Read every exposed telemetry attribute in name order.
    dir = opendir(sysdir);
    if (dir == nullptr) {
        DIE("cannot enumerate %s", sysdir);
    }
    while ((de = readdir(dir)) != nullptr) {
        if (strncmp(de->d_name, "tt_", 3) != 0) {
            continue;
        }
        // No tt-kmd attribute name comes near this length; do not let a
        // stranger truncate into a name we would then read.
        if (strlen(de->d_name) >= HUNG_ATTR_NAME_LEN) {
            continue;
        }
        if (attr_count >= HUNG_MAX_ATTRS) {
            break;
        }
        strcpy(attrs[attr_count], de->d_name);
        attr_count++;
    }
    closedir(dir);
    qsort(attrs, attr_count, HUNG_ATTR_NAME_LEN, attr_name_cmp);

    printf("telemetry (%u attributes):\n", attr_count);
    for (unsigned i = 0; i < attr_count; i++) {
        if (read_sysfs_attr(sysdir, attrs[i], value, sizeof(value)) != 0) {
            printf("  %-22s (read failed: %s)\n", attrs[i], strerror(errno));
            read_errors++;
            continue;
        }
        printf("  %-22s %s\n", attrs[i], value);

        if (sysfs_value_is_all_ones(value)) {
            fflush(stdout);
            verdict(
                0,
                "%s reads all ones; the chip stopped answering "
                "partway through the check",
                attrs[i]);
            return EXIT_CHIP_SILENT;
        }

        values_read++;
        if (strcmp(attrs[i], "tt_heartbeat") == 0) {
            snprintf(hb1, sizeof(hb1), "%s", value);
            have_heartbeat = 1;
        }
    }
    fflush(stdout);

    if (!have_heartbeat) {
        verdict(
            0,
            "tt_heartbeat is not published, so liveness cannot be judged "
            "(%u values read, %u errors)",
            values_read,
            read_errors);
        return EXIT_FW_SICK;
    }

    // Second heartbeat sample.  Whether the counter advances is worth more
    // than its value: a chip can serve stale telemetry forever.
    {
        struct timespec ts = {0, HUNG_HEARTBEAT_WAIT_NS};

        nanosleep(&ts, nullptr);
    }
    if (read_sysfs_attr(sysdir, "tt_heartbeat", hb2, sizeof(hb2)) != 0) {
        verdict(0, "re-reading tt_heartbeat failed: %s", strerror(errno));
        return EXIT_FW_SICK;
    }
    if (sysfs_value_is_all_ones(hb2)) {
        verdict(
            0,
            "tt_heartbeat read all ones on the second sample; the chip "
            "stopped answering during the check");
        return EXIT_CHIP_SILENT;
    }
    printf("heartbeat: %s -> %s over 0.5s\n", hb1, hb2);
    fflush(stdout);

    if (strcmp(hb1, hb2) == 0) {
        verdict(
            0,
            "heartbeat did not advance over 0.5s (still %s); the chip "
            "answers but ARC looks stalled",
            hb1);
        return EXIT_FW_SICK;
    }

    if (read_errors) {
        verdict(
            0,
            "heartbeat advances but %u of %u telemetry attributes failed "
            "to read",
            read_errors,
            attr_count);
        return EXIT_FW_SICK;
    }

    chip_open(&chip, device_path);
    noc_window_open(&win, chip.fd);
    noc_node_id = noc_read32(&win, chip.arc_x, chip.arc_y, chip.is_blackhole ? BH_ARC_NOC_NODE_ID : WH_ARC_NOC_NODE_ID);
    noc_window_close(&win);
    chip_close(&chip);

    printf("NOC ARC node ID: (%u, %u)\n", noc_node_id & 0x3f, (noc_node_id >> 6) & 0x3f);
    fflush(stdout);
    if (noc_node_id == ALL_ONES) {
        verdict(0, "ARC heartbeat advances, but NOC read to ARC returned all ones");
        return EXIT_CHIP_SILENT;
    }
    if ((noc_node_id & 0x3f) != chip.arc_x || ((noc_node_id >> 6) & 0x3f) != chip.arc_y) {
        verdict(
            0,
            "ARC heartbeat advances, but NOC read to ARC returned "
            "node ID (%u, %u), expected (%u, %u)",
            noc_node_id & 0x3f,
            (noc_node_id >> 6) & 0x3f,
            chip.arc_x,
            chip.arc_y);
        return EXIT_CHIP_SILENT;
    }

    verdict(
        1,
        "chip answers, %u telemetry values read, heartbeat advancing, "
        "NOC responsive",
        values_read);
    return EXIT_OK;
}

// ============================================================================
// reset: reset one chip in place and prove it came back
// ============================================================================
//
// Reset sequence: SBR, ASIC_RESET, wait for the PCI command-register marker
// to clear, POST_RESET, verify firmware, NOC, and DMA, then request idle power.
// POST_RESET retries once on a fresh fd if hotplug invalidated the original.

#define RESET_MARKER_TIMEOUT_MS 15000

#define RESET_POLL_INTERVAL_MS 100

#define RESET_SETTLE_MS 500

#define RESET_KMD_FLOOR_MAJOR 2
#define RESET_KMD_FLOOR_MINOR 10
#define RESET_KMD_FLOOR_PATCH 0

#define RESET_MAX_DEVS 64

// All trays, all ASICs, assert and deassert reset.
#define RESET_GLX_IPMI_CMD "ipmitool raw 0x30 0x8b 0xf 0xff 0x0 0xf"

// Both architectures use 0xC0DE in the high half of the ARC post code.
#define FW_POST_CODE_PREFIX 0xC0DE

// Per-device state for single and parallel resets.
struct reset_dev {
    const char* path;      // device path as given on the command line
    char path_buf[32];     // backing for path when --all enumerates
    char bdf[16];          // dddd:bb:dd.f; the stable identity across the reset
    char config_path[64];  // the BDF's config file in sysfs
    struct chip chip;      // open handle; normally survives an in-place reset
    unsigned ordinal;      // for sorting an enumeration
    int sbr_only;          // --sbr: stop after the SBR, then verify
    int rc;                // per-device exit status
};

static int reset_health_checks(struct reset_dev* d);

static void format_bdf(char* buf, size_t bufsz, const struct tenstorrent_get_device_info_out* info) {
    snprintf(
        buf,
        bufsz,
        "%04x:%02x:%02x.%x",
        info->pci_domain,
        (info->bus_dev_fn >> 8) & 0xFF,
        (info->bus_dev_fn >> 3) & 0x1F,
        info->bus_dev_fn & 0x7);
}

static void reset_sleep_ms(unsigned ms) {
    struct timespec ts = {ms / 1000, (long)(ms % 1000) * 1000000L};

    nanosleep(&ts, nullptr);
}

static long long reset_now_ms() {
    struct timespec now;

    clock_gettime(CLOCK_MONOTONIC, &now);
    return (long long)now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

// A suffix on the driver version string ("2.10.1-pre") is ignored.
static void reset_check_kmd_version() {
    char value[64];
    unsigned major, minor, patch;

    if (read_sysfs_attr("/sys/module/tenstorrent", "version", value, sizeof(value)) != 0) {
        DIE("cannot read /sys/module/tenstorrent/version");
    }
    if (sscanf(value, "%u.%u.%u", &major, &minor, &patch) != 3) {
        DIEX("cannot parse tt-kmd version '%s'", value);
    }

    if (major * 1000000 + minor * 1000 + patch <
        RESET_KMD_FLOOR_MAJOR * 1000000 + RESET_KMD_FLOOR_MINOR * 1000 + RESET_KMD_FLOOR_PATCH) {
        DIEX(
            "tt-kmd %s is too old for reset; need %u.%u.%u or newer",
            value,
            RESET_KMD_FLOOR_MAJOR,
            RESET_KMD_FLOOR_MINOR,
            RESET_KMD_FLOOR_PATCH);
    }
}

// Issue one RESET_DEVICE ioctl, preserving verdict-last output ordering.
static int reset_step(struct reset_dev* d, uint32_t flags, const char* what) {
    struct tenstorrent_reset_device cmd;

    memset(&cmd, 0, sizeof(cmd));
    cmd.in.output_size_bytes = sizeof(cmd.out);
    cmd.in.flags = flags;

    if (ioctl(d->chip.fd, TENSTORRENT_IOCTL_RESET_DEVICE, &cmd) != 0) {
        DIE("%s: %s ioctl failed", d->bdf, what);
    }
    if (cmd.out.result != 0) {
        printf("%s: %s failed: driver result %u\n", d->bdf, what, cmd.out.result);
        fflush(stdout);
        return EXIT_RESET_FAILED;
    }
    printf("%s: %s ok\n", d->bdf, what);
    fflush(stdout);
    return 0;
}

// Wait for the command-register marker to clear.  If the device disappears,
// reappearance replaces marker clearance as the completion signal.
static int reset_wait_marker(struct reset_dev* d) {
    long long start = reset_now_ms();
    int vanished = 0;
    int read_ok = 0;
    uint16_t command = 0xFFFF;

    do {
        int fd = open(d->config_path, O_RDONLY | O_CLOEXEC);

        if (fd < 0) {
            if (errno == ENOENT && !vanished) {
                printf("%s: device dropped off the bus; waiting for it to return\n", d->bdf);
                fflush(stdout);
                vanished = 1;
            }
        } else {
            uint8_t bytes[2];
            ssize_t n = pread(fd, bytes, sizeof(bytes), 4);

            close(fd);
            if (n == (ssize_t)sizeof(bytes)) {
                read_ok = 1;
                command = bytes[0] | (uint16_t)bytes[1] << 8;
                if (vanished) {
                    printf("%s: device is back on the bus after %lld ms\n", d->bdf, reset_now_ms() - start);
                    fflush(stdout);
                    return 0;
                }
                if (command != 0xFFFF && (command & (1u << 6)) == 0) {
                    printf("%s: reset marker cleared\n", d->bdf);
                    fflush(stdout);
                    return 0;
                }
            }
        }

        reset_sleep_ms(RESET_POLL_INTERVAL_MS);
    } while (reset_now_ms() - start < RESET_MARKER_TIMEOUT_MS);

    if (vanished) {
        printf("%s: device did not return to the bus within %u ms\n", d->bdf, (unsigned)RESET_MARKER_TIMEOUT_MS);
    } else if (!read_ok) {
        printf("%s: could not read config space within %u ms\n", d->bdf, (unsigned)RESET_MARKER_TIMEOUT_MS);
    } else if (command == 0xFFFF) {
        printf("%s: config space still reads all ones after %u ms\n", d->bdf, (unsigned)RESET_MARKER_TIMEOUT_MS);
    } else {
        printf("%s: reset marker still set after %u ms\n", d->bdf, (unsigned)RESET_MARKER_TIMEOUT_MS);
    }
    fflush(stdout);
    return -1;
}

// Retry POST_RESET once on a fresh fd after hotplug invalidates the old one.
static int reset_post_reset(struct reset_dev* d) {
    struct tenstorrent_reset_device cmd;
    int retried = 0;

    for (;;) {
        memset(&cmd, 0, sizeof(cmd));
        cmd.in.output_size_bytes = sizeof(cmd.out);
        cmd.in.flags = TENSTORRENT_RESET_DEVICE_POST_RESET;

        if (ioctl(d->chip.fd, TENSTORRENT_IOCTL_RESET_DEVICE, &cmd) == 0) {
            break;
        }
        if (errno != ENODEV || retried) {
            DIE("%s: POST_RESET ioctl failed", d->bdf);
        }

        printf(
            "%s: POST_RESET returned ENODEV: the device was re-probed; "
            "reopening %s and retrying\n",
            d->bdf,
            d->path);
        fflush(stdout);
        chip_close(&d->chip);
        chip_open(&d->chip, d->path);
        retried = 1;
    }

    if (cmd.out.result != 0) {
        printf("%s: POST_RESET failed: driver result %u\n", d->bdf, cmd.out.result);
        fflush(stdout);
        return EXIT_RESET_FAILED;
    }
    printf("%s: POST_RESET ok\n", d->bdf);
    fflush(stdout);
    return 0;
}

static int reset_request_idle(struct reset_dev* d) {
    struct tenstorrent_power_state power_state;

    memset(&power_state, 0, sizeof(power_state));
    power_state.argsz = sizeof(power_state);
    power_state.validity = TT_POWER_VALIDITY(15, 14);
    if (ioctl(d->chip.fd, TENSTORRENT_IOCTL_SET_POWER_STATE, &power_state) != 0) {
        printf("%s: SET_POWER_STATE idle request failed: %s\n", d->bdf, strerror(errno));
        fflush(stdout);
        return EXIT_RESET_FAILED;
    }
    printf("%s: SET_POWER_STATE idle request ok\n", d->bdf);
    fflush(stdout);
    return EXIT_OK;
}

// Read the ARC post code over NOC; only the stable 0xC0DE prefix is judged.
static int reset_read_post_code(struct reset_dev* d) {
    const char* reg_name = d->chip.is_blackhole ? "STATUS_POST_CODE" : "POST_CODE";
    struct noc_window win;
    uint32_t post_code;

    noc_window_open(&win, d->chip.fd);
    post_code = noc_read32(&win, d->chip.arc_x, d->chip.arc_y, ARC_RESET_UNIT_BASE + ARC_SCRATCH_OFFSET);
    noc_window_close(&win);

    if (post_code == ALL_ONES) {
        printf("%s: %s reads all ones; the ARC tile is not answering\n", d->bdf, reg_name);
        fflush(stdout);
        return EXIT_CHIP_SILENT;
    }
    if ((post_code >> 16) != FW_POST_CODE_PREFIX) {
        printf("%s: %s is 0x%08x, which is not a 0xC0DE.... firmware post code\n", d->bdf, reg_name, post_code);
        fflush(stdout);
        return EXIT_RESET_FAILED;
    }
    printf("%s: %s is 0x%08x\n", d->bdf, reg_name, post_code);
    fflush(stdout);
    return EXIT_OK;
}

static void reset_open(struct reset_dev* d) {
    chip_open(&d->chip, d->path);

    format_bdf(d->bdf, sizeof(d->bdf), &d->chip.info);
    snprintf(d->config_path, sizeof(d->config_path), "/sys/bus/pci/devices/%s/config", d->bdf);
    printf("%s: %s is %s (device_id=0x%04x)\n", d->bdf, d->path, d->chip.arch_name, d->chip.info.device_id);
    fflush(stdout);
}

static int reset_one(struct reset_dev* d) {
    int rc;

    reset_open(d);

    rc = reset_step(d, TENSTORRENT_RESET_DEVICE_RESET_PCIE_LINK, "SBR (RESET_PCIE_LINK)");
    if (rc != 0) {
        goto out;
    }
    if (d->sbr_only) {
        rc = reset_read_post_code(d);
        if (rc == 0) {
            rc = reset_health_checks(d);
        }
        goto out;
    }

    rc = reset_step(d, TENSTORRENT_RESET_DEVICE_ASIC_RESET, "ASIC_RESET");
    if (rc != 0) {
        goto out;
    }

    // Allow reset to land before polling and ARC to boot before POST_RESET.
    reset_sleep_ms(RESET_SETTLE_MS);
    if (reset_wait_marker(d) != 0) {
        rc = EXIT_CHIP_SILENT;
        goto out;
    }
    reset_sleep_ms(RESET_SETTLE_MS);

    rc = reset_post_reset(d);
    if (rc != 0) {
        goto out;
    }

    rc = reset_read_post_code(d);
    if (rc == 0) {
        rc = reset_health_checks(d);
    }

out:
    chip_close(&d->chip);
    return rc;
}

static int reset_ordinal_cmp(const void* a, const void* b) {
    const struct reset_dev* da = static_cast<const struct reset_dev*>(a);
    const struct reset_dev* db = static_cast<const struct reset_dev*>(b);

    return (da->ordinal > db->ordinal) - (da->ordinal < db->ordinal);
}

// Fills devs with every device in /dev/tenstorrent, in ordinal order, and
// returns the count.  The numeric entries are the devices; by-id and
// by-bdf are symlink directories and fail the all-digits test.
static unsigned reset_enumerate(struct reset_dev* devs, unsigned max) {
    DIR* dir;
    struct dirent* de;
    unsigned count = 0;

    dir = opendir("/dev/tenstorrent");
    if (dir == nullptr) {
        DIE("cannot open /dev/tenstorrent");
    }
    while ((de = readdir(dir)) != nullptr) {
        const char* p = de->d_name;

        if (*p < '0' || *p > '9') {
            continue;
        }
        while (*p >= '0' && *p <= '9') {
            p++;
        }
        if (*p != '\0') {
            continue;
        }
        if (p - de->d_name > 10) {  // no plausible ordinal is longer
            continue;
        }
        if (count >= max) {
            break;
        }
        devs[count].ordinal = (unsigned)strtoul(de->d_name, nullptr, 10);
        count++;
    }
    closedir(dir);
    qsort(devs, count, sizeof(*devs), reset_ordinal_cmp);

    // The paths are formatted after the sort so that each one points into
    // its own element.
    for (unsigned i = 0; i < count; i++) {
        snprintf(devs[i].path_buf, sizeof(devs[i].path_buf), "/dev/tenstorrent/%u", devs[i].ordinal);
        devs[i].path = devs[i].path_buf;
    }
    return count;
}

static void* reset_thread(void* arg) {
    struct reset_dev* d = static_cast<struct reset_dev*>(arg);

    d->rc = reset_one(d);
    return nullptr;
}

// The aggregate verdict and exit status for a multi-device run: 0 only if
// every chip passed; otherwise a chip that stopped answering outranks one
// that answered wrongly.
static int reset_verdict_all(struct reset_dev* devs, unsigned count, int sbr_only) {
    unsigned ok = 0;
    int any_silent = 0;

    for (unsigned i = 0; i < count; i++) {
        if (devs[i].rc == EXIT_OK) {
            ok++;
        } else if (devs[i].rc == EXIT_CHIP_SILENT) {
            any_silent = 1;
        }
    }

    if (ok == count) {
        verdict(
            1,
            "%u/%u chips %s",
            ok,
            count,
            sbr_only ? "link reset and passed health checks" : "reset and passed health checks");
        return EXIT_OK;
    }
    verdict(0, "%u of %u chips failed %s verification", count - ok, count, sbr_only ? "link reset" : "reset");
    return any_silent ? EXIT_CHIP_SILENT : EXIT_RESET_FAILED;
}

// Run one reset thread per device.
static int reset_all(int sbr_only) {
    static struct reset_dev devs[RESET_MAX_DEVS];
    static pthread_t threads[RESET_MAX_DEVS];
    unsigned count;

    count = reset_enumerate(devs, RESET_MAX_DEVS);
    if (count == 0) {
        DIEX("no devices under /dev/tenstorrent");
    }

    for (unsigned i = 0; i < count; i++) {
        int err;

        devs[i].sbr_only = sbr_only;
        err = pthread_create(&threads[i], nullptr, reset_thread, &devs[i]);
        if (err != 0) {
            errno = err;
            DIE("pthread_create failed");
        }
    }
    for (unsigned i = 0; i < count; i++) {
        pthread_join(threads[i], nullptr);
    }

    return reset_verdict_all(devs, count, sbr_only);
}

// A chip that fails a step drops out of later steps; the tray reset proceeds.
static int reset_glx() {
    static struct reset_dev devs[RESET_MAX_DEVS];
    unsigned count;
    int st;

    count = reset_enumerate(devs, RESET_MAX_DEVS);
    if (count == 0) {
        DIEX("no devices under /dev/tenstorrent");
    }

    for (unsigned i = 0; i < count; i++) {
        reset_open(&devs[i]);
    }

    // USER_RESET arms driver bookkeeping; the BMC triggers the reset.
    for (unsigned i = 0; i < count; i++) {
        devs[i].rc = reset_step(&devs[i], TENSTORRENT_RESET_DEVICE_RESET_PCIE_LINK, "SBR (RESET_PCIE_LINK)");
    }
    for (unsigned i = 0; i < count; i++) {
        if (devs[i].rc == 0) {
            devs[i].rc = reset_step(&devs[i], TENSTORRENT_RESET_DEVICE_USER_RESET, "USER_RESET");
        }
    }

    printf("running: %s\n", RESET_GLX_IPMI_CMD);
    fflush(stdout);
    st = system(RESET_GLX_IPMI_CMD);
    if (st == -1 || !WIFEXITED(st) || WEXITSTATUS(st) != 0) {
        DIEX("'%s' failed", RESET_GLX_IPMI_CMD);
    }

    reset_sleep_ms(RESET_SETTLE_MS);
    for (unsigned i = 0; i < count; i++) {
        if (devs[i].rc == 0 && reset_wait_marker(&devs[i]) != 0) {
            devs[i].rc = EXIT_CHIP_SILENT;
        }
    }
    reset_sleep_ms(RESET_SETTLE_MS);

    for (unsigned i = 0; i < count; i++) {
        if (devs[i].rc == 0) {
            devs[i].rc = reset_post_reset(&devs[i]);
        }
    }
    for (unsigned i = 0; i < count; i++) {
        if (devs[i].rc == 0) {
            devs[i].rc = reset_read_post_code(&devs[i]);
        }
    }
    for (unsigned i = 0; i < count; i++) {
        if (devs[i].rc == 0) {
            devs[i].rc = reset_health_checks(&devs[i]);
        }
    }

    for (unsigned i = 0; i < count; i++) {
        chip_close(&devs[i].chip);
    }

    return reset_verdict_all(devs, count, 0);
}

static void reset_usage(const char* prog) {
    fprintf(stderr, "Usage: %s DEVICE [--sbr]\n", prog);
    fprintf(stderr, "       %s -d DEVICE [--sbr]\n", prog);
    fprintf(stderr, "       %s --all [--sbr]\n", prog);
    fprintf(stderr, "       %s --glx\n", prog);
    fprintf(stderr, "  -d PATH  Tenstorrent device to reset, path or ordinal (no default; this is destructive)\n");
    fprintf(stderr, "  --all    Every device in /dev/tenstorrent, in parallel\n");
    fprintf(stderr, "  --sbr    Secondary bus reset without ASIC reset, then run health checks\n");
    fprintf(stderr, "  --glx    Galaxy tray reset over IPMI; stands alone (needs ipmitool)\n");
    fprintf(stderr, "  -h       This help\n");
    fprintf(stderr, "\nResets the chip in place: SBR, ASIC_RESET, wait for the reset marker\n");
    fprintf(stderr, "to clear, POST_RESET, verify firmware, NOC, and DMA, then request idle power.\n");
    fprintf(
        stderr,
        "Requires tt-kmd %u.%u.%u or newer.\n",
        RESET_KMD_FLOOR_MAJOR,
        RESET_KMD_FLOOR_MINOR,
        RESET_KMD_FLOOR_PATCH);
    fprintf(stderr, "\nPrints one [PASS] or [FAIL] line last.\n");
    fprintf(stderr, "Exit status:\n");
    fprintf(stderr, "  0 = chip reset and passed post-code, NOC, and DMA checks\n");
    fprintf(stderr, "  1 = usage or setup error\n");
    fprintf(stderr, "  2 = chip not answering\n");
    fprintf(stderr, "  3 = chip answers but a reset step failed\n");
    fprintf(stderr, "  For --all/--glx: 0 if all chips passed, otherwise 2 if any\n");
    fprintf(stderr, "  chip is silent, otherwise 3.\n");
}

static int reset_main(int argc, char* argv[], const char* prog) {
    static const struct option longopts[] = {
        {"all", no_argument, nullptr, 'a'},
        {"sbr", no_argument, nullptr, 's'},
        {"glx", no_argument, nullptr, 'g'},
        {nullptr, 0, nullptr, 0},
    };
    struct reset_dev d;
    const char* ident;
    int all = 0, sbr = 0, glx = 0;
    int opt;
    int rc;

    memset(&d, 0, sizeof(d));

    while ((opt = getopt_long(argc, argv, "d:h", longopts, nullptr)) != -1) {
        switch (opt) {
            case 'a': all = 1; break;
            case 's': sbr = 1; break;
            case 'g': glx = 1; break;
            case 'd': d.path = device_path_arg(optarg); break;
            case 'h': reset_usage(prog); return EXIT_OK;
            default: reset_usage(prog); return EXIT_SETUP_ERROR;
        }
    }

    // Accept either -d or one positional device, but not both.
    if (optind < argc) {
        if (d.path != nullptr) {
            fprintf(stderr, "Error: device given both as -d %s and as '%s'\n", d.path, argv[optind]);
            reset_usage(prog);
            return EXIT_SETUP_ERROR;
        }
        if (argc - optind > 1) {
            fprintf(stderr, "Error: unexpected extra argument '%s'\n", argv[optind + 1]);
            reset_usage(prog);
            return EXIT_SETUP_ERROR;
        }
        d.path = device_path_arg(argv[optind]);
    }

    if (glx && (all || sbr || d.path != nullptr)) {
        fprintf(stderr, "Error: --glx stands alone\n");
        reset_usage(prog);
        return EXIT_SETUP_ERROR;
    }
    if (all && d.path != nullptr) {
        fprintf(stderr, "Error: --all and a device are mutually exclusive\n");
        reset_usage(prog);
        return EXIT_SETUP_ERROR;
    }
    if (!all && !glx && d.path == nullptr) {
        fprintf(stderr, "Error: no device given; reset is destructive and has no default\n");
        reset_usage(prog);
        return EXIT_SETUP_ERROR;
    }

    reset_check_kmd_version();

    if (glx) {
        return reset_glx();
    }
    if (all) {
        return reset_all(sbr);
    }

    d.sbr_only = sbr;
    rc = reset_one(&d);

    ident = d.bdf[0] ? d.bdf : d.path;
    switch (rc) {
        case EXIT_OK:
            verdict(1, sbr ? "%s link reset and passed health checks" : "%s reset and passed health checks", ident);
            break;
        case EXIT_CHIP_SILENT:
            verdict(0, sbr ? "%s not reachable after link reset" : "%s did not come back from reset", ident);
            break;
        case EXIT_RESET_FAILED: verdict(0, "%s answers but reset verification failed", ident); break;
        default: break;  // reset_one returns only the three above
    }
    return rc;
}

// ============================================================================
// nuke: kill every process holding the device open
// ============================================================================
//
// For clearing stale workloads before a reset.  SIGKILLs every pid in the
// driver's /proc/driver/tenstorrent/<N>/pids file, naming each victim, then
// waits for the file to read empty.  The device node itself is never opened
// -- the ordinal comes from /sys/dev/char -- so the tool cannot appear in
// its own victim list.
//
// A new process can open the device between the read and kill, so repeat until
// the file is empty or the pass limit is reached.
//
// Exit status: 0 nothing holds the device open (whether or not anything had
// to be killed), 1 usage or setup error, 2 holders remain after the pass
// limit.

#define NUKE_MAX_PASSES 10
#define NUKE_PASS_WAIT_NS 200000000L
#define NUKE_MAX_PIDS 256

// Reads the pids file, one pid per open fd, deduplicated.  Returns the count,
// or -1 with errno set if the file cannot be read.
static int read_holder_pids(const char* pids_path, pid_t* pids, unsigned max) {
    FILE* f;
    char line[64];
    unsigned count = 0;

    f = fopen(pids_path, "re");
    if (f == nullptr) {
        return -1;
    }

    while (fgets(line, sizeof(line), f) != nullptr && count < max) {
        pid_t pid = (pid_t)strtol(line, nullptr, 10);
        unsigned i;

        if (pid <= 1) {
            continue;
        }
        for (i = 0; i < count; i++) {
            if (pids[i] == pid) {
                break;
            }
        }
        if (i == count) {
            pids[count++] = pid;
        }
    }
    fclose(f);
    return (int)count;
}

// Process state from /proc/<pid>/stat; '?' if unavailable.
static char proc_state(pid_t pid) {
    char path[64];
    char buf[512];
    const char* p;
    int fd;
    ssize_t n;

    snprintf(path, sizeof(path), "/proc/%d/stat", pid);
    fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return '?';
    }
    n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) {
        return '?';
    }
    buf[n] = '\0';

    // The state field follows the comm, which is in parentheses and may
    // itself contain anything; parse from the last ')'.
    p = strrchr(buf, ')');
    if (p == nullptr || p[1] != ' ' || p[2] == '\0') {
        return '?';
    }
    return p[2];
}

#if defined(SYS_pidfd_open) && defined(SYS_pidfd_send_signal)
#define HAVE_PIDFD 1
#endif

// Start time in clock ticks since boot, field 22 of /proc/<pid>/stat.  0 if
// it cannot be read, which includes the process having already gone.
static unsigned long long proc_start_time(pid_t pid) {
    char path[64];
    char buf[512];
    const char* p;
    int fd;
    ssize_t n;

    snprintf(path, sizeof(path), "/proc/%d/stat", pid);
    fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return 0;
    }
    n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) {
        return 0;
    }
    buf[n] = '\0';

    // As in proc_state: the comm is parenthesised and may contain anything,
    // so the fields are counted from the last ')', which ends field 2.
    p = strrchr(buf, ')');
    if (p == nullptr) {
        return 0;
    }
    p++;
    for (int field = 3; field <= 21; field++) {
        while (*p == ' ') {
            p++;
        }
        while (*p != '\0' && *p != ' ') {
            p++;
        }
        if (*p == '\0') {
            return 0;
        }
    }
    return strtoull(p, nullptr, 10);
}

struct nuke_victim {
    pid_t pid;
    int fd;                    // pidfd; -1 when the kernel does not have them
    unsigned long long start;  // start time, the guard used when fd < 0
};

// Pins a holder so it can be signalled later without the number drifting to
// another process.  0 on success, -1 with errno set; ESRCH means it exited
// on its own, which needs no killing.
static int victim_pin(struct nuke_victim* v, pid_t pid) {
    v->pid = pid;
    v->fd = -1;
    v->start = 0;

#ifdef HAVE_PIDFD
    v->fd = (int)syscall(SYS_pidfd_open, pid, 0U);
    if (v->fd >= 0) {
        return 0;
    }
    if (errno != ENOSYS) {
        return -1;
    }
#endif

    v->start = proc_start_time(pid);
    if (v->start == 0) {
        errno = ESRCH;
        return -1;
    }
    return 0;
}

// SIGKILLs the pinned process.  0 on success, -1 with errno set.  ESRCH is
// the process having already gone -- or, on the fallback path, the number
// now belonging to a different process, which is deliberately left alone.
static int victim_kill(const struct nuke_victim* v) {
#ifdef HAVE_PIDFD
    if (v->fd >= 0) {
        return (int)syscall(SYS_pidfd_send_signal, v->fd, SIGKILL, (siginfo_t*)nullptr, 0U);
    }
#endif

    if (proc_start_time(v->pid) != v->start) {
        errno = ESRCH;
        return -1;
    }
    return kill(v->pid, SIGKILL);
}

static void victim_unpin(struct nuke_victim* v) {
    if (v->fd >= 0) {
        close(v->fd);
        v->fd = -1;
    }
}

static int pid_listed(pid_t pid, const pid_t* list, int count) {
    for (int i = 0; i < count; i++) {
        if (list[i] == pid) {
            return 1;
        }
    }
    return 0;
}

static void nuke_print_victim(const char* action, pid_t pid) {
    char dir[32];
    char comm[64];
    char cmdline[256];
    char path[64];
    int fd;
    ssize_t n = 0;

    snprintf(dir, sizeof(dir), "/proc/%d", pid);
    if (read_sysfs_attr(dir, "comm", comm, sizeof(comm)) != 0) {
        snprintf(comm, sizeof(comm), "?");
    }

    snprintf(path, sizeof(path), "%s/cmdline", dir);
    fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
        n = read(fd, cmdline, sizeof(cmdline) - 1);
        close(fd);
    }
    if (n < 0) {
        n = 0;
    }
    cmdline[n] = '\0';
    for (ssize_t i = 0; i < n; i++) {
        if (cmdline[i] == '\0') {
            cmdline[i] = ' ';
        }
    }

    printf("%s pid %d (%s)%s%s\n", action, pid, comm, cmdline[0] ? ": " : "", cmdline);
    fflush(stdout);
}

static void nuke_usage(const char* prog) {
    fprintf(stderr, "Usage: %s DEVICE\n", prog);
    fprintf(stderr, "       %s -d DEVICE\n", prog);
    fprintf(stderr, "  -d PATH  device whose holders to kill, path or ordinal; no default\n");
    fprintf(stderr, "\nSIGKILLs every process holding the device open, naming each one.\n");
    fprintf(stderr, "Prints one [PASS] or [FAIL] line last.\n");
    fprintf(stderr, "Exit status:\n");
    fprintf(stderr, "  0 = nothing holds the device open\n");
    fprintf(stderr, "  1 = usage or setup error\n");
    fprintf(stderr, "  2 = holders remain (likely blocked in the driver; reset anyway)\n");
}

static int nuke_main(int argc, char* argv[], const char* prog) {
    const char* device_path = nullptr;
    struct stat st;
    char syspath[64];
    char link[256];
    char pids_path[64];
    pid_t pids[NUKE_MAX_PIDS];
    const char* base;
    unsigned ordinal;
    unsigned killed = 0;
    unsigned pass;
    int count;
    ssize_t n;
    int opt;

    while ((opt = getopt(argc, argv, "d:h")) != -1) {
        switch (opt) {
            case 'd': device_path = device_path_arg(optarg); break;
            case 'h': nuke_usage(prog); return EXIT_OK;
            default: nuke_usage(prog); return EXIT_SETUP_ERROR;
        }
    }
    if (optind < argc) {
        if (device_path != nullptr) {
            fprintf(stderr, "Error: device given both as -d %s and as '%s'\n", device_path, argv[optind]);
            nuke_usage(prog);
            return EXIT_SETUP_ERROR;
        }
        if (argc - optind > 1) {
            fprintf(stderr, "Error: unexpected extra argument '%s'\n", argv[optind + 1]);
            nuke_usage(prog);
            return EXIT_SETUP_ERROR;
        }
        device_path = device_path_arg(argv[optind]);
    }

    if (device_path == nullptr) {
        fprintf(stderr, "Error: no device given; nuke kills processes and has no default\n");
        nuke_usage(prog);
        return EXIT_SETUP_ERROR;
    }

    // The ordinal names the driver's proc directory.  It comes from the
    // class device that /sys/dev/char points at, so any path to the right
    // character device works and the device is never opened.
    if (stat(device_path, &st) != 0) {
        DIE("cannot stat %s", device_path);
    }
    if (!S_ISCHR(st.st_mode)) {
        DIEX("%s is not a character device", device_path);
    }
    snprintf(syspath, sizeof(syspath), "/sys/dev/char/%u:%u", major(st.st_rdev), minor(st.st_rdev));
    n = readlink(syspath, link, sizeof(link) - 1);
    if (n <= 0) {
        DIE("cannot readlink %s", syspath);
    }
    link[n] = '\0';
    base = strrchr(link, '/');
    base = base ? base + 1 : link;
    if (sscanf(base, "tenstorrent!%u", &ordinal) != 1) {
        DIEX("%s is not a tenstorrent device (class device is '%s')", device_path, base);
    }
    snprintf(pids_path, sizeof(pids_path), "/proc/driver/tenstorrent/%u/pids", ordinal);

    printf("%s: device %u, holders per %s\n", device_path, ordinal, pids_path);
    fflush(stdout);

    for (pass = 0; pass < NUKE_MAX_PASSES; pass++) {
        struct timespec wait = {0, NUKE_PASS_WAIT_NS};
        struct nuke_victim victims[NUKE_MAX_PIDS];
        pid_t still[NUKE_MAX_PIDS];
        unsigned pinned = 0;
        int still_count;
        int i;

        count = read_holder_pids(pids_path, pids, NUKE_MAX_PIDS);
        if (count < 0) {
            DIE("cannot read %s", pids_path);
        }
        if (count == 0) {
            break;
        }

        for (i = 0; i < count; i++) {
            if (pids[i] == getpid()) {
                continue;
            }
            if (victim_pin(&victims[pinned], pids[i]) == 0) {
                pinned++;
            } else if (errno != ESRCH) {
                printf("  cannot take a handle on pid %d: %s\n", pids[i], strerror(errno));
            }
        }

        still_count = read_holder_pids(pids_path, still, NUKE_MAX_PIDS);
        if (still_count < 0) {
            DIE("cannot read %s", pids_path);
        }

        for (unsigned v = 0; v < pinned; v++) {
            if (pid_listed(victims[v].pid, still, still_count)) {
                nuke_print_victim("killing", victims[v].pid);
                if (victim_kill(&victims[v]) == 0) {
                    killed++;
                } else if (errno != ESRCH) {
                    printf("  kill pid %d failed: %s\n", victims[v].pid, strerror(errno));
                }
            }
            victim_unpin(&victims[v]);
        }
        fflush(stdout);

        nanosleep(&wait, nullptr);
    }

    count = read_holder_pids(pids_path, pids, NUKE_MAX_PIDS);
    if (count < 0) {
        DIE("cannot read %s", pids_path);
    }

    if (count > 0) {
        for (int i = 0; i < count; i++) {
            nuke_print_victim("still holding:", pids[i]);
            printf(
                "  state %c%s\n",
                proc_state(pids[i]),
                proc_state(pids[i]) == 'D'
                    ? " (blocked in the kernel; the SIGKILL takes effect once a reset unwedges the chip)"
                    : "");
        }
        verdict(0, "%d process(es) still hold %s open after SIGKILL", count, device_path);
        return EXIT_HOLDERS_REMAIN;
    }

    if (killed) {
        verdict(1, "killed %u process(es); nothing holds %s open", killed, device_path);
    } else {
        verdict(1, "nothing holds %s open", device_path);
    }
    return EXIT_OK;
}

// ============================================================================
// info: one-screen inventory of one device
// ============================================================================
//
// Stable "key: value" lines, with host-side facts available even when
// chip-side telemetry is not.

__attribute__((format(printf, 2, 3))) static void info_line(const char* key, const char* fmt, ...) {
    char keycol[32];
    va_list ap;

    snprintf(keycol, sizeof(keycol), "%s:", key);
    printf("%-16s ", keycol);
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
    fflush(stdout);
}

// Maps a sysfs *_link_speed string ("16.0 GT/s PCIe"; older kernels drop
// the suffix) to a PCIe generation.  0 if the string is not a defined rate.
static int pcie_gen_from_speed(const char* speed) {
    double gts = strtod(speed, nullptr);

    if (gts == 2.5) {
        return 1;
    }
    if (gts == 5.0) {
        return 2;
    }
    if (gts == 8.0) {
        return 3;
    }
    if (gts == 16.0) {
        return 4;
    }
    if (gts == 32.0) {
        return 5;
    }
    if (gts == 64.0) {
        return 6;
    }
    return 0;
}

static void info_pcie_speed_line(const char* pcidir, const char* file, const char* key) {
    char value[64];
    int gen;

    if (read_sysfs_attr(pcidir, file, value, sizeof(value)) != 0) {
        info_line(key, "unavailable");
        return;
    }
    gen = pcie_gen_from_speed(value);
    if (gen != 0) {
        info_line(key, "%d", gen);
    } else {
        info_line(key, "unknown (%s)", value);
    }
}

static void info_pcie_width_line(const char* pcidir, const char* file, const char* key) {
    char value[64];

    if (read_sysfs_attr(pcidir, file, value, sizeof(value)) != 0) {
        info_line(key, "unavailable");
    } else {
        info_line(key, "%s", value);
    }
}

// Board name from the UPI field of the board id, (board_id >> 36) & 0xFFFF.
// Same mapping as tt_sysfs_show_card_type() in tt-kmd telemetry.c.  NULL
// for a UPI we do not know.
static const char* info_board_name(unsigned upi) {
    switch (upi) {
        // Wormhole
        case 0x14: return "n300";
        case 0x18: return "n150";
        case 0x35: return "galaxy-wormhole";
        // Blackhole
        case 0x36: return "p100";
        case 0x40: return "p150a";
        case 0x41: return "p150b";
        case 0x42: return "p150c";
        case 0x43: return "p100a";
        case 0x44: return "p300b";
        case 0x45: return "p300a";
        case 0x46: return "p300c";
        case 0x47: return "galaxy-blackhole";
        default: return nullptr;
    }
}

// Galaxy physical location.
//
// A Galaxy is identified by subsystem device ID, and the PCI bus number
// encodes physical position: the high nibble selects the UBB, the low nibble
// is the 1-based chip index on that UBB.  See enumerate.c.
//
// host_side.sh carries the same tables in bash, because it must run with no
// dependency on this binary.  Keep the two in step.
#define PCI_SUBSYSTEM_ID_GALAXY_WH 0x0035
#define PCI_SUBSYSTEM_ID_GALAXY_BH 0x0047

static const unsigned wh_galaxy_ubb_prefix[4] = {0xC, 0x8, 0x0, 0x4};
static const unsigned bh_galaxy_ubb_prefix[4] = {0x0, 0x4, 0xC, 0x8};

// Fills buf with "u<ubb>c<chip>" and returns 1, or leaves buf empty and
// returns 0 for a part that is not in a Galaxy or a bus number that does not
// decode.
static int galaxy_loc(unsigned subsystem_id, unsigned bus, char* buf, size_t bufsz) {
    const unsigned* table;
    unsigned high = bus >> 4;
    unsigned low = bus & 0x0F;

    buf[0] = '\0';
    if (subsystem_id == PCI_SUBSYSTEM_ID_GALAXY_BH) {
        table = bh_galaxy_ubb_prefix;
    } else if (subsystem_id == PCI_SUBSYSTEM_ID_GALAXY_WH) {
        table = wh_galaxy_ubb_prefix;
    } else {
        return 0;
    }

    if (low < 1 || low > 8) {
        return 0;
    }

    for (unsigned ubb = 0; ubb < 4; ubb++) {
        if (table[ubb] == high) {
            snprintf(buf, bufsz, "u%uc%u", ubb + 1, low);
            return 1;
        }
    }
    return 0;
}

// The telemetry directory, walked once and kept.  status is EXIT_OK,
// EXIT_CHIP_SILENT or EXIT_BAD_TELEMETRY; on anything but EXIT_OK, reason
// says why the chip-side facts are unavailable.
struct info_telemetry {
    int status;
    char reason[160];
    uint64_t table_base;
    uint64_t data_base;
    int tag_present[MAX_TAG_COUNT];
    uint16_t tag_offset[MAX_TAG_COUNT];
};

// Walk telemetry without diagnostics; info reports failure through its output.
static void info_walk_telemetry(
    struct noc_window* win, const struct chip* chip, const struct telemetry_arch* arch, struct info_telemetry* t) {
    uint32_t table_ptr_raw, data_ptr_raw, version, entry_count;

    memset(t, 0, sizeof(*t));

    table_ptr_raw = noc_read32(win, chip->arc_x, chip->arc_y, arch->table_ptr_addr);
    data_ptr_raw = noc_read32(win, chip->arc_x, chip->arc_y, arch->data_ptr_addr);
    if (table_ptr_raw == ALL_ONES || data_ptr_raw == ALL_ONES) {
        t->status = EXIT_CHIP_SILENT;
        snprintf(t->reason, sizeof(t->reason), "chip not answering: a telemetry pointer register reads all ones");
        return;
    }

    if (table_ptr_raw == 0 || data_ptr_raw == 0) {
        t->status = EXIT_BAD_TELEMETRY;
        snprintf(
            t->reason,
            sizeof(t->reason),
            "firmware has not published telemetry (table 0x%08x, data 0x%08x)",
            table_ptr_raw,
            data_ptr_raw);
        return;
    }

    if (!arch->is_blackhole && (table_ptr_raw < 0x10000000u || table_ptr_raw > 0x1007FFFFu ||
                                data_ptr_raw < 0x10000000u || data_ptr_raw > 0x1007FFFFu)) {
        t->status = EXIT_BAD_TELEMETRY;
        snprintf(
            t->reason,
            sizeof(t->reason),
            "implausible telemetry pointers (table 0x%08x, data 0x%08x)",
            table_ptr_raw,
            data_ptr_raw);
        return;
    }

    t->table_base = (uint64_t)table_ptr_raw | arch->pointer_fixup;
    t->data_base = (uint64_t)data_ptr_raw | arch->pointer_fixup;

    version = noc_read32(win, chip->arc_x, chip->arc_y, t->table_base + TELEMETRY_VERSION_OFFSET);
    entry_count = noc_read32(win, chip->arc_x, chip->arc_y, t->table_base + TELEMETRY_ENTRY_COUNT_OFFSET);
    if (version == ALL_ONES || entry_count == ALL_ONES) {
        t->status = EXIT_CHIP_SILENT;
        snprintf(t->reason, sizeof(t->reason), "chip not answering: the telemetry table header reads all ones");
        return;
    }
    if (version == 0) {
        t->status = EXIT_BAD_TELEMETRY;
        snprintf(t->reason, sizeof(t->reason), "telemetry format version is zero");
        return;
    }
    if (entry_count == 0 || entry_count > MAX_ENTRY_COUNT) {
        t->status = EXIT_BAD_TELEMETRY;
        snprintf(t->reason, sizeof(t->reason), "implausible telemetry entry_count %u", entry_count);
        return;
    }
    if (!arch->is_blackhole &&
        t->table_base + TELEMETRY_DIRECTORY_OFFSET + (uint64_t)entry_count * 4 > WH_CSM_NOC_END) {
        t->status = EXIT_BAD_TELEMETRY;
        snprintf(t->reason, sizeof(t->reason), "telemetry directory extends past the end of ARC CSM");
        return;
    }

    for (unsigned i = 0; i < entry_count; ++i) {
        uint64_t addr = t->table_base + TELEMETRY_DIRECTORY_OFFSET + (uint64_t)i * sizeof(uint32_t);
        uint32_t entry = noc_read32(win, chip->arc_x, chip->arc_y, addr);
        unsigned tag;
        unsigned offset;

        if (entry == ALL_ONES) {
            t->status = EXIT_CHIP_SILENT;
            snprintf(t->reason, sizeof(t->reason), "chip stopped answering during the telemetry directory walk");
            return;
        }

        tag = entry & 0xFFFF;
        if (tag == 0) {
            continue;
        }

        offset = (entry >> 16) & 0xFFFF;
        if (offset > MAX_VALUE_OFFSET) {
            continue;
        }
        if (!arch->is_blackhole && t->data_base + (uint64_t)offset * 4 + 4 > WH_CSM_NOC_END) {
            continue;
        }

        if (tag < MAX_TAG_COUNT) {
            t->tag_present[tag] = 1;
            t->tag_offset[tag] = (uint16_t)offset;
        }
    }

    t->status = EXIT_OK;
}

// How info_read_tag() reports a tag firmware does not publish.
#define INFO_TAG_ABSENT (-2)

// Reads one tag's value: READ_OK with the value, READ_ALL_ONES if the chip
// returned the idle pattern, INFO_TAG_ABSENT if the tag is not in the
// firmware's directory.
static int info_read_tag(
    struct noc_window* win, const struct chip* chip, const struct info_telemetry* t, unsigned tag, uint32_t* out) {
    uint32_t v;

    if (!t->tag_present[tag]) {
        return INFO_TAG_ABSENT;
    }
    v = noc_read32(win, chip->arc_x, chip->arc_y, t->data_base + (uint64_t)t->tag_offset[tag] * sizeof(uint32_t));
    if (v == ALL_ONES) {
        return READ_ALL_ONES;
    }
    *out = v;
    return READ_OK;
}

static const char* info_tag_problem(int r) {
    return r == INFO_TAG_ABSENT ? "absent (not in firmware directory)" : "unavailable (reads all ones)";
}

static void info_tag_line(
    struct noc_window* win, const struct chip* chip, const struct info_telemetry* t, const char* key, unsigned tag) {
    uint32_t v;
    int r = info_read_tag(win, chip, t, tag, &v);

    if (r == READ_OK) {
        info_line(key, "%u", v);
    } else {
        info_line(key, "%s", info_tag_problem(r));
    }
}

static unsigned info_bit_count(uint32_t v) {
    unsigned n = 0;

    for (; v != 0; v &= v - 1) {
        n++;
    }
    return n;
}

// On Blackhole, judge trained/error bits only for enabled GDDR instances.
// Wormhole reports pass as 1, fail as 3, and all other states as 0.
static void info_dram_status(struct noc_window* win, const struct chip* chip, const struct info_telemetry* t) {
    uint32_t status;
    char buf[128];
    int r = info_read_tag(win, chip, t, TAG_GDDR_STATUS, &status);

    if (r != READ_OK) {
        info_line("dram_status", "%s", info_tag_problem(r));
        return;
    }

    if (chip->is_blackhole) {
        uint32_t enabled = 0xFF;
        unsigned trained = 0, error = 0;
        unsigned expect, not_trained;

        // Without TAG_ENABLED_GDDR every instance is expected; with it, a
        // harvested instance is exempt.
        (void)info_read_tag(win, chip, t, TAG_ENABLED_GDDR, &enabled);

        for (unsigned i = 0; i < 8; ++i) {
            trained |= ((status >> (2 * i)) & 1) << i;
            error |= ((status >> (2 * i + 1)) & 1) << i;
        }
        expect = enabled & 0xFF;
        not_trained = expect & ~trained;
        error &= expect;

        if (not_trained == 0 && error == 0) {
            snprintf(buf, sizeof(buf), "trained %u/%u", info_bit_count(trained & expect), info_bit_count(expect));
        } else {
            snprintf(
                buf,
                sizeof(buf),
                "trained %u/%u, error mask 0x%02x, untrained mask 0x%02x",
                info_bit_count(trained & expect),
                info_bit_count(expect),
                error,
                not_trained);
        }
    } else {
        unsigned ok = 0, failed = 0;

        for (unsigned ch = 0; ch < 6; ++ch) {
            unsigned s = (status >> (2 * ch)) & 0x3;

            if (s == 1) {
                ok++;
            } else if (s == 3) {
                failed++;
            }
        }
        if (ok == 6) {
            snprintf(buf, sizeof(buf), "trained 6/6");
        } else {
            snprintf(buf, sizeof(buf), "trained %u/6, %u failed, %u not trained", ok, failed, 6 - ok - failed);
        }
    }
    info_line("dram_status", "%s", buf);
}

static void info_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [DEVICE]\n", prog);
    fprintf(stderr, "       %s -d DEVICE\n", prog);
    fprintf(stderr, "  -d PATH  Tenstorrent device to open, path or ordinal (default /dev/tenstorrent/0)\n");
    fprintf(stderr, "  -h       This help\n");
    fprintf(stderr, "\nOne 'key: value' line per fact.  Host-side facts (PCI link, IOMMU) are\n");
    fprintf(stderr, "printed even when the chip is hung; chip-side facts then read unavailable.\n");
    fprintf(stderr, "Exit status:\n");
    fprintf(stderr, "  0 = success\n");
    fprintf(stderr, "  1 = usage or setup error\n");
    fprintf(stderr, "  2 = chip not answering\n");
    fprintf(stderr, "  3 = chip answers but telemetry is unpublished or malformed\n");
}

static int info_main(int argc, char* argv[], const char* prog) {
    const char* device_path = "/dev/tenstorrent/0";
    int device_path_set = 0;
    struct chip chip;
    struct noc_window win;
    struct info_telemetry telem;
    const struct telemetry_arch* arch;
    char bdf[16];
    char pcidir[80];
    char value[128];
    uint32_t v;
    int rc;
    int r;
    int opt;

    while ((opt = getopt(argc, argv, "d:h")) != -1) {
        switch (opt) {
            case 'd':
                device_path = device_path_arg(optarg);
                device_path_set = 1;
                break;
            case 'h': info_usage(prog); return EXIT_OK;
            default: info_usage(prog); return EXIT_SETUP_ERROR;
        }
    }

    // Accept either -d or one positional device, but not both.
    if (optind < argc) {
        if (device_path_set) {
            fprintf(stderr, "Error: device given both as -d %s and as '%s'\n", device_path, argv[optind]);
            info_usage(prog);
            return EXIT_SETUP_ERROR;
        }
        if (argc - optind > 1) {
            fprintf(stderr, "Error: unexpected extra argument '%s'\n", argv[optind + 1]);
            info_usage(prog);
            return EXIT_SETUP_ERROR;
        }
        device_path = device_path_arg(argv[optind]);
    }

    // GET_DEVICE_INFO and sysfs supply the host-side facts.
    chip_open(&chip, device_path);
    format_bdf(bdf, sizeof(bdf), &chip.info);
    snprintf(pcidir, sizeof(pcidir), "/sys/bus/pci/devices/%s", bdf);

    info_line("device", "%s", device_path);
    info_line("arch", "%s", chip.arch_name);
    info_line("pci", "%s", bdf);

    // Only a Galaxy part has a physical location to report; on anything else
    // the line is omitted rather than printed as "-".
    {
        char loc[16];

        if (galaxy_loc(chip.info.subsystem_id, (chip.info.bus_dev_fn >> 8) & 0xFF, loc, sizeof(loc))) {
            info_line("location", "%s", loc);
        }
    }

    info_pcie_speed_line(pcidir, "current_link_speed", "pcie_gen");
    info_pcie_width_line(pcidir, "current_link_width", "pcie_width");
    info_pcie_speed_line(pcidir, "max_link_speed", "pcie_max_gen");
    info_pcie_width_line(pcidir, "max_link_width", "pcie_max_width");

    // IOMMU mode: the iommu_group's type ("identity", "DMA", "DMA-FQ", ...).
    // No iommu_group at all means the device is not behind an IOMMU.
    {
        char iommu_dir[96];

        snprintf(iommu_dir, sizeof(iommu_dir), "%s/iommu_group", pcidir);
        if (read_sysfs_attr(iommu_dir, "type", value, sizeof(value)) == 0) {
            info_line("iommu", "%s", value);
        } else if (errno == ENOENT) {
            info_line("iommu", "none");
        } else {
            info_line("iommu", "unavailable (%s)", strerror(errno));
        }
    }

    arch = chip.is_blackhole ? &blackhole_telemetry : &wormhole_telemetry;
    noc_window_open(&win, chip.fd);
    info_walk_telemetry(&win, &chip, arch, &telem);

    if (telem.status != EXIT_OK) {
        static const char* const chip_keys[] = {
            "board_type",
            "board_id",
            "fw_bundle",
            "dram_status",
            "dram_speed_mbps",
            "vcore_mv",
            "current_a",
            "power_w",
            "aiclk_mhz",
            "asic_temp_c",
        };

        info_line("telemetry", "unavailable (%s)", telem.reason);
        for (const auto* chip_key : chip_keys) {
            info_line(chip_key, "unavailable");
        }
        rc = telem.status;
        goto out;
    }
    info_line("telemetry", "ok");

    {
        uint32_t hi = 0, lo = 0;
        int rhi = info_read_tag(&win, &chip, &telem, TAG_BOARD_ID_HIGH, &hi);
        int rlo = info_read_tag(&win, &chip, &telem, TAG_BOARD_ID_LOW, &lo);

        if (rhi != READ_OK) {
            info_line("board_type", "%s", info_tag_problem(rhi));
        } else {
            unsigned upi = (hi >> 4) & 0xFFFF;
            const char* name = info_board_name(upi);

            if (name != nullptr) {
                info_line("board_type", "%s", name);
            } else {
                info_line("board_type", "unknown (upi 0x%x)", upi);
            }
        }

        if (rhi == READ_OK && rlo == READ_OK) {
            info_line("board_id", "0x%016llx", ((unsigned long long)hi << 32) | lo);
        } else {
            info_line("board_id", "%s", info_tag_problem(rhi != READ_OK ? rhi : rlo));
        }
    }

    r = info_read_tag(&win, &chip, &telem, TAG_FLASH_BUNDLE_VERSION, &v);
    if (r == READ_OK) {
        info_line("fw_bundle", "%u.%u.%u.%u", (v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF);
    } else {
        info_line("fw_bundle", "%s", info_tag_problem(r));
    }

    info_dram_status(&win, &chip, &telem);
    info_tag_line(&win, &chip, &telem, "dram_speed_mbps", TAG_GDDR_SPEED);
    info_tag_line(&win, &chip, &telem, "vcore_mv", TAG_VCORE);
    info_tag_line(&win, &chip, &telem, "current_a", TAG_TDC);
    info_tag_line(&win, &chip, &telem, "power_w", TAG_TDP);
    info_tag_line(&win, &chip, &telem, "aiclk_mhz", TAG_AICLK);

    // ASIC temperature, signed 16.16 on both archs; Blackhole reports
    // 0x80000000 for a sensor error.
    r = info_read_tag(&win, &chip, &telem, TAG_ASIC_TEMPERATURE, &v);
    if (r != READ_OK) {
        info_line("asic_temp_c", "%s", info_tag_problem(r));
    } else if (chip.is_blackhole && v == 0x80000000u) {
        info_line("asic_temp_c", "unavailable (sensor error)");
    } else {
        info_line("asic_temp_c", "%.2f", (double)(int32_t)v / 65536.0);
    }

    // Liveness re-check, as in the telemetry subcommand: a chip that died
    // during the value reads above left "unavailable" lines, not garbage,
    // but the exit status should still say it is not answering.
    if (noc_read32(&win, chip.arc_x, chip.arc_y, telem.table_base + TELEMETRY_VERSION_OFFSET) == ALL_ONES) {
        info_line("liveness", "chip stopped answering during the dump");
        rc = EXIT_CHIP_SILENT;
        goto out;
    }

    rc = EXIT_OK;

out:
    noc_window_close(&win);
    chip_close(&chip);
    return rc;
}

// ============================================================================
// discover: enumerate every device, one line each, with a liveness verdict
// ============================================================================
//
// One line per numeric character device in /dev/tenstorrent:
//
//   DEVICE              ARCH PCI          GEN      WIDTH   STATUS
//   /dev/tenstorrent/0  BH   0000:02:00.0 Gen5/5   x16/16  ok
//
// On a Galaxy a LOC column appears, holding the chip's physical position as
// u<ubb>c<chip>, and the table is sorted by it rather than by ordinal: the
// ordinals come from driver probe order and carry no physical meaning, while
// eight failures on one UBB and eight spread across four are different faults.
//
// GEN and WIDTH are current/max for the endpoint.  The character devices are
// not opened.
//
// STATUS is one token:
//   ok                 the chip answers, its heartbeat advances, and the
//                      link trained to expectation
//   hung(off-bus)      config space reads all ones; the endpoint is gone
//   hung(silent)       telemetry reads all ones; the chip is not answering
//   hung(stalled)      answers but the heartbeat did not advance over 0.5s
//   hung(blocked)      a telemetry read hit its deadline
//   degraded(...)      the chip is fine but the link trained below
//                      expectation; the note says which half and by how
//                      much, e.g. degraded(gen3<5,x1<8)
//   unknown(no-telem)  the driver publishes no telemetry for the device
//   unknown(err)       sysfs did not cooperate; the other columns still
//                      show whatever was learned
// LINK EXPECTATION: a link trains to the lesser of the two ends' maxima,
// so degradation is judged against min(endpoint max, upstream port max),
// not the endpoint's capability alone.

#define DISCOVER_MAX_DEVS 64

// Bound each telemetry read so one device cannot prevent enumeration.
#define DISCOVER_READ_DEADLINE_S 5

#define DISCOVER_HEARTBEAT_WAIT_NS 500000000L

struct discover_dev {
    unsigned ordinal;
    char path[32];
    char sysdir[64];  // the /sys/dev/char/<maj>:<min> class device dir
    char bdf[16];
    char loc[16];      // Galaxy "u2c1", empty for a part that is not in one
    const char* arch;  // "WH", "BH", "??"
    int cur_gen;       // PCIe link facts; 0 for one we could not get
    int max_gen;
    int cur_width;
    int max_width;
    char link_note[24];  // non-empty if trained below expectation, e.g. "x1<8"
    const char* live;    // liveness ladder outcome
    char hb1[32];        // first heartbeat sample
    int need_hb2;        // got a first sample; judge after the shared wait
};

static sigjmp_buf discover_jmp;

static void discover_alarm(int sig) {
    (void)sig;
    siglongjmp(discover_jmp, 1);
}

// read_sysfs_attr() under the per-read deadline; timeout sets ETIMEDOUT.
static int discover_read_attr_deadline(const char* dir, const char* name, char* buf, size_t bufsz) {
    int r;

    if (sigsetjmp(discover_jmp, 1) != 0) {
        errno = ETIMEDOUT;
        return -1;
    }
    alarm(DISCOVER_READ_DEADLINE_S);
    r = read_sysfs_attr(dir, name, buf, bufsz);
    alarm(0);
    return r;
}

// One *_link_speed attribute as a generation number; 0 if unreadable or
// not a defined rate.
static int discover_link_gen(const char* dir, const char* name) {
    char value[64];

    if (read_sysfs_attr(dir, name, value, sizeof(value)) != 0) {
        return 0;
    }
    return pcie_gen_from_speed(value);
}

// One *_link_width attribute as a lane count; 0 if unreadable or not a
// number.
static int discover_link_width(const char* dir, const char* name) {
    char value[64];

    if (read_sysfs_attr(dir, name, value, sizeof(value)) != 0) {
        return 0;
    }
    if (value[0] == '\0' || strspn(value, "0123456789") != strlen(value)) {
        return 0;
    }
    return atoi(value);
}

// What the link should have trained to: the lesser of the two ends'
// maxima.  An unknown end (0) defers to the other; both unknown means no
// expectation and no degradation call.
static int discover_link_expectation(int ep_max, int upstream_max) {
    if (ep_max == 0) {
        return upstream_max;
    }
    if (upstream_max == 0) {
        return ep_max;
    }
    return ep_max < upstream_max ? ep_max : upstream_max;
}

// Gather host facts and the first heartbeat sample.
static void discover_probe(struct discover_dev* d) {
    struct stat st;
    char devdir[80];
    char link[256];
    char value[128];
    ssize_t n;

    d->arch = "??";
    snprintf(d->bdf, sizeof(d->bdf), "?");

    if (stat(d->path, &st) != 0 || !S_ISCHR(st.st_mode)) {
        d->live = "err";
        return;
    }
    snprintf(d->sysdir, sizeof(d->sysdir), "/sys/dev/char/%u:%u", major(st.st_rdev), minor(st.st_rdev));
    snprintf(devdir, sizeof(devdir), "%s/device", d->sysdir);

    n = readlink(devdir, link, sizeof(link) - 1);
    if (n > 0) {
        const char* base;

        link[n] = '\0';
        base = strrchr(link, '/');
        // A PCI location is 12 characters; the precision keeps a stranger
        // from overrunning the column (and quiets -Wformat-truncation).
        snprintf(d->bdf, sizeof(d->bdf), "%.12s", base ? base + 1 : link);
    }

    // Architecture from the PCI device id, so no fd is needed.
    if (read_sysfs_attr(devdir, "device", value, sizeof(value)) == 0) {
        unsigned long id = strtoul(value, nullptr, 0);

        if (id == WORMHOLE_PCI_DEVICE_ID) {
            d->arch = "WH";
        } else if (id == BLACKHOLE_PCI_DEVICE_ID) {
            d->arch = "BH";
        }
    }

    // Galaxy physical location, from the subsystem id and the bus number.
    // Both come from sysfs, so this needs no fd either.
    d->loc[0] = '\0';
    if (read_sysfs_attr(devdir, "subsystem_device", value, sizeof(value)) == 0) {
        unsigned long ssid = strtoul(value, nullptr, 0);
        unsigned bus = 0;

        // d->bdf is "dddd:bb:dd.f"; the bus is the two hex digits after the
        // first colon.
        if (strlen(d->bdf) >= 7) {
            bus = (unsigned)strtoul(d->bdf + 5, nullptr, 16);
        }
        galaxy_loc((unsigned)ssid, bus, d->loc, sizeof(d->loc));
    }

    // PCIe link generation and width, current and max, plus the upstream
    // port's maxima (the parent directory in the PCI hierarchy) so
    // degradation is judged against what this link could actually train
    // to.  See LINK EXPECTATION above.
    {
        char updir[96];
        size_t used = 0;
        int exp;

        d->cur_gen = discover_link_gen(devdir, "current_link_speed");
        d->max_gen = discover_link_gen(devdir, "max_link_speed");
        d->cur_width = discover_link_width(devdir, "current_link_width");
        d->max_width = discover_link_width(devdir, "max_link_width");

        snprintf(updir, sizeof(updir), "%s/..", devdir);
        exp = discover_link_expectation(d->max_gen, discover_link_gen(updir, "max_link_speed"));
        if (d->cur_gen != 0 && exp != 0 && d->cur_gen < exp) {
            used += (size_t)snprintf(d->link_note, sizeof(d->link_note), "gen%d<%d", d->cur_gen, exp);
        }

        exp = discover_link_expectation(d->max_width, discover_link_width(updir, "max_link_width"));
        if (d->cur_width != 0 && exp != 0 && d->cur_width < exp && used < sizeof(d->link_note)) {
            snprintf(
                d->link_note + used, sizeof(d->link_note) - used, "%sx%d<%d", used != 0 ? "," : "", d->cur_width, exp);
        }
    }

    // Liveness rung 1: config space.  All ones means the endpoint is off
    // the bus and nothing that follows would reach it.
    {
        char path[96];
        uint32_t id = 0;
        int fd;

        snprintf(path, sizeof(path), "%s/config", devdir);
        fd = open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            d->live = "err";
            return;
        }
        n = read(fd, &id, sizeof(id));
        close(fd);
        if (n != (ssize_t)sizeof(id)) {
            d->live = "err";
            return;
        }
        if (id == ALL_ONES) {
            d->live = "off-bus";
            return;
        }
    }

    // Rung 2: one telemetry value, the first read that crosses the chip.
    if (discover_read_attr_deadline(d->sysdir, "tt_serial", value, sizeof(value)) != 0) {
        d->live = errno == ETIMEDOUT ? "blocked" : errno == ENOENT ? "no-telem" : "err";
        return;
    }
    if (sysfs_value_is_all_ones(value)) {
        d->live = "silent";
        return;
    }

    // Rung 3: the first heartbeat sample; judged after the shared wait.
    if (discover_read_attr_deadline(d->sysdir, "tt_heartbeat", d->hb1, sizeof(d->hb1)) != 0) {
        d->live = errno == ETIMEDOUT ? "blocked" : errno == ENOENT ? "no-telem" : "err";
        return;
    }
    if (sysfs_value_is_all_ones(d->hb1)) {
        d->live = "silent";
        return;
    }
    d->need_hb2 = 1;
}

static void discover_judge_heartbeat(struct discover_dev* d) {
    char hb2[32];

    if (discover_read_attr_deadline(d->sysdir, "tt_heartbeat", hb2, sizeof(hb2)) != 0) {
        d->live = errno == ETIMEDOUT ? "blocked" : "err";
        return;
    }
    if (sysfs_value_is_all_ones(hb2)) {
        d->live = "silent";
        return;
    }
    d->live = strcmp(d->hb1, hb2) == 0 ? "stalled" : "ok";
}

// Outcomes that make the device unusable or prevent judging liveness.
static int discover_live_is_hung(const char* live) {
    return strcmp(live, "off-bus") == 0 || strcmp(live, "silent") == 0 || strcmp(live, "stalled") == 0 ||
           strcmp(live, "blocked") == 0;
}

// A current/max column half-pair, e.g. "Gen5/5" or "x1/8"; "?" for a half
// that could not be determined.
static void discover_pair(char* buf, size_t bufsz, const char* prefix, int cur, int max) {
    char cur_s[8] = "?", max_s[8] = "?";

    if (cur != 0) {
        snprintf(cur_s, sizeof(cur_s), "%d", cur);
    }
    if (max != 0) {
        snprintf(max_s, sizeof(max_s), "%d", max);
    }
    snprintf(buf, bufsz, "%s%s/%s", prefix, cur_s, max_s);
}

static int discover_ordinal_cmp(const void* a, const void* b) {
    const struct discover_dev* da = static_cast<const struct discover_dev*>(a);
    const struct discover_dev* db = static_cast<const struct discover_dev*>(b);

    return (da->ordinal > db->ordinal) - (da->ordinal < db->ordinal);
}

// "u2c1" sorts correctly as a plain string: both fields are one digit.
static int discover_loc_cmp(const void* a, const void* b) {
    const struct discover_dev* da = static_cast<const struct discover_dev*>(a);
    const struct discover_dev* db = static_cast<const struct discover_dev*>(b);

    return strcmp(da->loc, db->loc);
}

static void discover_usage(const char* prog) {
    fprintf(stderr, "Usage: %s\n", prog);
    fprintf(stderr, "\nA header, then one line per device in /dev/tenstorrent:\n");
    fprintf(stderr, "  DEVICE ARCH PCI GEN(cur/max) WIDTH(cur/max) STATUS\n");
    fprintf(stderr, "STATUS: ok, hung(off-bus|silent|stalled|blocked),\n");
    fprintf(stderr, "  degraded(link trained below min(endpoint max, upstream max)),\n");
    fprintf(stderr, "  unknown(no-telem|err)\n");
    fprintf(stderr, "Color (status token only) on a terminal unless NO_COLOR is set.\n");
    fprintf(stderr, "Exit status:\n");
    fprintf(stderr, "  0 = every chip ok\n");
    fprintf(stderr, "  1 = usage or setup error\n");
    fprintf(stderr, "  2 = some chip is hung\n");
    fprintf(stderr, "  3 = some chip is degraded or unknown\n");
}

static int discover_main(int argc, char* argv[], const char* prog) {
    static struct discover_dev devs[DISCOVER_MAX_DEVS];
    unsigned count = 0;
    struct sigaction sa;
    DIR* dir;
    struct dirent* de;
    int need_wait = 0;
    int any_hung = 0;
    int any_sick = 0;
    int opt;

    while ((opt = getopt(argc, argv, "h")) != -1) {
        switch (opt) {
            case 'h': discover_usage(prog); return EXIT_OK;
            default: discover_usage(prog); return EXIT_SETUP_ERROR;
        }
    }
    if (optind < argc) {
        fprintf(stderr, "Error: unexpected argument '%s'\n", argv[optind]);
        discover_usage(prog);
        return EXIT_SETUP_ERROR;
    }

    // The numeric entries are the devices; by-id and by-bdf are symlink
    // directories and fail the all-digits test.
    dir = opendir("/dev/tenstorrent");
    if (dir == nullptr) {
        DIE("cannot open /dev/tenstorrent");
    }
    while ((de = readdir(dir)) != nullptr) {
        const char* p = de->d_name;

        if (*p < '0' || *p > '9') {
            continue;
        }
        while (*p >= '0' && *p <= '9') {
            p++;
        }
        if (*p != '\0') {
            continue;
        }
        if (p - de->d_name > 10) {  // no plausible ordinal is longer
            continue;
        }
        if (count >= DISCOVER_MAX_DEVS) {
            break;
        }
        devs[count].ordinal = (unsigned)strtoul(de->d_name, nullptr, 10);
        snprintf(devs[count].path, sizeof(devs[count].path), "/dev/tenstorrent/%.10s", de->d_name);
        count++;
    }
    closedir(dir);

    if (count == 0) {
        fprintf(stderr, "no devices in /dev/tenstorrent\n");
        return EXIT_SETUP_ERROR;
    }
    qsort(devs, count, sizeof(devs[0]), discover_ordinal_cmp);

    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = discover_alarm;
    sigaction(SIGALRM, &sa, nullptr);

    for (unsigned i = 0; i < count; i++) {
        discover_probe(&devs[i]);
    }

    for (unsigned i = 0; i < count; i++) {
        need_wait |= devs[i].need_hb2;
    }
    if (need_wait) {
        struct timespec ts = {0, DISCOVER_HEARTBEAT_WAIT_NS};

        nanosleep(&ts, nullptr);
    }
    for (unsigned i = 0; i < count; i++) {
        if (devs[i].need_hb2) {
            discover_judge_heartbeat(&devs[i]);
        }
    }

    // On a Galaxy, group the table by physical position rather than by
    // ordinal: eight failures on one UBB and eight spread across four are
    // different faults, and only this ordering shows which you have.  The
    // ordinals carry no physical meaning -- they come from driver probe order.
    {
        unsigned with_loc = 0;

        for (unsigned i = 0; i < count; i++) {
            with_loc += devs[i].loc[0] != '\0';
        }
        if (with_loc == count) {
            qsort(devs, count, sizeof(devs[0]), discover_loc_cmp);
        }
    }

    {
        int use_color = isatty(STDOUT_FILENO) && getenv("NO_COLOR") == nullptr;
        const char* red = use_color ? "\033[31m" : "";
        const char* yellow = use_color ? "\033[33m" : "";
        const char* reset = use_color ? "\033[0m" : "";
        int any_loc = 0;

        for (unsigned i = 0; i < count; i++) {
            any_loc |= devs[i].loc[0] != '\0';
        }

        // The LOC column only appears when something can fill it, so a
        // single-card host does not get a column of dashes.
        if (any_loc) {
            printf("%-19s %-5s %-4s %-12s %-8s %-7s %s\n", "DEVICE", "LOC", "ARCH", "PCI", "GEN", "WIDTH", "STATUS");
        } else {
            printf("%-19s %-4s %-12s %-8s %-7s %s\n", "DEVICE", "ARCH", "PCI", "GEN", "WIDTH", "STATUS");
        }

        for (unsigned i = 0; i < count; i++) {
            const struct discover_dev* d = &devs[i];
            char gencol[16], widthcol[16], status[48];
            const char* color = "";

            discover_pair(gencol, sizeof(gencol), "Gen", d->cur_gen, d->max_gen);
            discover_pair(widthcol, sizeof(widthcol), "x", d->cur_width, d->max_width);

            if (discover_live_is_hung(d->live)) {
                snprintf(status, sizeof(status), "hung(%s)", d->live);
                color = red;
                any_hung = 1;
            } else if (d->link_note[0] != '\0') {
                snprintf(status, sizeof(status), "degraded(%s)", d->link_note);
                color = yellow;
                any_sick = 1;
            } else if (strcmp(d->live, "ok") != 0) {
                snprintf(status, sizeof(status), "unknown(%s)", d->live);
                any_sick = 1;
            } else {
                snprintf(status, sizeof(status), "ok");
            }

            if (any_loc) {
                printf(
                    "%-19s %-5s %-4s %-12s %-8s %-7s %s%s%s\n",
                    d->path,
                    d->loc[0] ? d->loc : "-",
                    d->arch,
                    d->bdf,
                    gencol,
                    widthcol,
                    color,
                    status,
                    color[0] != '\0' ? reset : "");
            } else {
                printf(
                    "%-19s %-4s %-12s %-8s %-7s %s%s%s\n",
                    d->path,
                    d->arch,
                    d->bdf,
                    gencol,
                    widthcol,
                    color,
                    status,
                    color[0] != '\0' ? reset : "");
            }
        }
    }
    fflush(stdout);

    return any_hung ? EXIT_CHIP_SILENT : any_sick ? EXIT_FW_SICK : EXIT_OK;
}

// ============================================================================
// test: named chip tests
// ============================================================================
//
// noc_sanity: ask every node on the NOC who it is, and check that it answers
// correctly.
//
// Every NOC interface unit (NIU) has a read-only NOC_NODE_ID register holding
// the coordinates of the node it belongs to.  Walking the grid and checking
// that the node at (x, y) agrees that it is (x, y) exercises the whole
// addressing path: TLB programming, the NOC2AXI bridge in the PCIe tile,
// routing, and the target NIU.  A chip whose NOC is confused will usually fail
// this before it fails anything more interesting.
//
// Every node is visited: Tensix, Ethernet, DRAM, ARC, PCIe, L2CPU, Security,
// and the "empty" nodes that contain nothing but a router and an NIU.  That is
// 120 nodes on Wormhole and 204 on Blackhole.
//
// WHY THIS IS MORE THAN A LOOP:
//   Three things vary, and getting any of them wrong reads from an address the
//   target does not implement, which is not a benign mistake (see HAZARD).
//
//   1. Where the register lives depends on the tile type.  Tensix and Ethernet
//      tiles map their NIU into the tile's own address space at a fixed
//      address.  ARC, PCIe, L2CPU and Security have no such map; they are
//      bridges, and the NIU appears as an aperture in the address space on the
//      far side.  Hence the wildly different bases in the tables below.
//
//   2. On Wormhole the DRAM address additionally depends on *which* node it is.
//      Three NOC nodes share one GDDR channel and one AXI bus, so the three
//      NIUs need three distinct apertures to be distinguishable.  Which of the
//      three a node is happens to be y % 3.
//
//   3. Blackhole applies coordinate translation, Wormhole effectively does not.
//      See COORDINATES.
//
// COORDINATES:
//   Firmware programs a translation table into every NIU, so the coordinates
//   handed to CONFIGURE_TLB are run through it before a packet is routed.
//
//   On Wormhole this is invisible to us: firmware programs entries 0..15 as an
//   identity map, and every real coordinate (x < 10, y < 12) falls inside that
//   range.  So we sweep raw coordinates and expect NOC_NODE_ID to match what we
//   asked for.
//
//   On Blackhole there is no identity range; translation is genuinely applied,
//   so the coordinates we program are *translated* coordinates and NOC_NODE_ID
//   reports something different from what we asked for.  With translation off
//   the addressed coordinate is the raw one and the grid is identical in both
//   spaces, so every check below still holds; the tool does not depend on
//   knowing which state the chip is in.
//
//   NOC_ID_LOGICAL is not the answer either, tempting as it looks.  Firmware
//   sets it to each node's *canonical* translated coordinate, and most nodes
//   have a canonical coordinate far away from the one we use to reach them:
//   the PCIe tile answers at (2, 0) but calls itself (19, 24), and the Ethernet
//   tiles answer along row 1 but call themselves (20..31, 25).  Comparing it
//   against the addressed coordinate only works for Tensix, which is why
//   Tensix-only checks get away with it.  We print it but do not judge it.
//
//   What we lean on instead is the shape of firmware's translation tables.
//   Sweeping translated x in 0..16 and y in 0..11 reaches all 204 nodes exactly
//   once, and within that box firmware leaves almost everything alone:
//     - The y table is the identity over 0..11, always.
//     - x is not translated at all on rows 0 and 1.
//     - x indices 0, 8 and 9 are never written, so the DRAM columns and the
//       column holding Security and the L2CPUs stay put.
//     - Only the Tensix columns permute, and only among themselves, which is
//       how harvesting renumbers them.
//   So every node must report the y we asked for, every non-Tensix node must
//   report the x we asked for, and a Tensix node must report some Tensix
//   column.  That holds on a harvested part, where a fixed expected-coordinate
//   table would not.
//
//   Two further checks make up for the looseness on Tensix x.  NOC_ENDPOINT_ID
//   says what kind of tile answered, so a node reached through the wrong base
//   is caught even if its coordinates look plausible.  And because the sweep
//   should be a bijection onto the raw grid, collecting every reported raw
//   coordinate and confirming the set is the whole grid with no duplicates
//   catches a translation table that aliases two translated coordinates onto
//   one node -- a failure where every node individually answers perfectly.
//
// HAZARD:
//   A NOC read to an address the target tile does not decode can take the whole
//   endpoint off the PCIe bus: config space reads back all ones and only a link
//   reset recovers it.  This was not theoretical; it is how the address tables
//   below got their final round of checking.  Consequences:
//     - Only ever issue an address that the tables say is right for that node's
//       type.  There is no probing and no trying the other base if the first
//       looks wrong.
//     - -l prints the entire plan, address by address, and issues no NOC reads.
//       Worth doing on a chip or a firmware you have not run this on before.
//     - A read of all ones means the target did not answer, and the sweep stops
//       there by default rather than walking the rest of the grid poking a
//       device that has stopped listening.  -k overrides.
//     - -s excludes coordinates you already know to be trouble.
//
//   A node that answers with the *wrong* coordinates is a different case and
//   the sweep continues: that node is alive, the next read costs a microsecond,
//   and the pattern across the grid is what names the root cause -- one node
//   wrong, one column wrong, and all Tensix wrong are three different faults.
//   --stop-first reverts to stopping at the first failure of any kind.
//
// SOURCES:
//   Layout and register bases are cross-checked between the ARC firmware node
//   table (syseng arc_fw/lib/tensix_utils.c), t6py's NOC node classes
//   (t6py/packages/tenstorrent/chip/tile.py), tt-umd's per-architecture
//   implementation headers and SoC descriptors, and tt-isa-documentation's NoC
//   chapters.  Where they disagreed, the firmware and tt-umd won.

// Register offsets from the NIU base.  Blackhole moved all three when its
// per-initiator register block grew (both architectures have four request
// initiators; Blackhole added registers to each).
#define WH_NODE_ID_OFFSET 0x2C
#define WH_ENDPOINT_ID_OFFSET 0x30
#define WH_ID_LOGICAL_OFFSET 0x138
#define BH_NODE_ID_OFFSET 0x44
#define BH_ENDPOINT_ID_OFFSET 0x48
#define BH_ID_LOGICAL_OFFSET 0x148

// NOC_NODE_ID and NOC_ID_LOGICAL pack coordinates the same way on both
// architectures.  NOC_NODE_ID additionally reports the grid dimensions and the
// routing dimension order, which every node on a given NOC should agree on.
#define ID_X(v) ((v) & 0x3F)
#define ID_Y(v) (((v) >> 6) & 0x3F)
#define ID_WIDTH(v) (((v) >> 12) & 0x7F)
#define ID_HEIGHT(v) (((v) >> 19) & 0x7F)
#define ID_DIM_ORDER(v) (((v) >> 28) & 0x1)  // 1 on NOC0, 0 on NOC1

// NOC_ENDPOINT_ID says what kind of tile answered.  Wormhole and Blackhole
// disagree on both the field position and the type encoding.
#define WH_ENDPOINT_TYPE(v) (((v) >> 16) & 0xFF)
#define BH_ENDPOINT_TYPE(v) (((v) >> 8) & 0xFFFF)
#define ENDPOINT_NOC(v) (((v) >> 24) & 0xFF)

// Sentinel for a tile whose NOC_ENDPOINT_ID type we cannot usefully check.
#define ENDPOINT_TYPE_ANY 0xFFFFFFFFu

// Tile types, spelled as the single characters used in the grid maps below.
#define TILE_TENSIX 'T'
#define TILE_ETH 'E'
#define TILE_DRAM 'D'
#define TILE_ARC 'A'
#define TILE_PCIE 'P'
#define TILE_L2CPU 'C'
#define TILE_SECURITY 'S'
#define TILE_EMPTY '.'

// Wormhole, 10x12, indexed [y][x] in raw NOC0 coordinates.  Transcribed from
// node_types[][] in syseng arc_fw/lib/tensix_utils.c, and independently equal
// to tt-umd's wormhole_b0_8x10.yaml and the tt-isa-documentation NoC diagram.
// Column 0 and column 5 hold everything that is not Tensix or Ethernet.
#define WH_GRID_X 10
#define WH_GRID_Y 12
static const char* const wh_grid[WH_GRID_Y] = {
    //        x=0123456789
    /* y= 0 */ "DEEEEDEEEE",
    /* y= 1 */ "DTTTTDTTTT",
    /* y= 2 */ ".TTTTDTTTT",
    /* y= 3 */ "PTTTTDTTTT",
    /* y= 4 */ ".TTTTDTTTT",
    /* y= 5 */ "DTTTTDTTTT",
    /* y= 6 */ "DEEEEDEEEE",
    /* y= 7 */ "DTTTTDTTTT",
    /* y= 8 */ ".TTTTDTTTT",
    /* y= 9 */ ".TTTTDTTTT",
    /* y=10 */ "ATTTTDTTTT",
    /* y=11 */ "DTTTTDTTTT",
};

// Blackhole, 17x12, indexed [y][x] in *translated* coordinates.  Rows 0 and 1
// pass x through untranslated, so there they are also raw coordinates.  For
// y >= 2 the translated column determines the type directly: 1..7 and 10..16
// are the Tensix columns, 0 and 9 are the two DRAM columns, and 8 is the column
// holding Security and the four L2CPUs.
//
// Equal to t6py's physical layout in chip/blackhole.py mapped through its
// _phys_to_noc0 permutations, to tt-umd's blackhole_140_arch.yaml, and to the
// tt-isa-documentation NoC diagram.
#define BH_GRID_X 17
#define BH_GRID_Y 12
static const char* const bh_grid[BH_GRID_Y] = {
    //        x=01234567890123456
    /* y= 0 */ "D.P.....AD.P.....",
    /* y= 1 */ "DEEEEEEE.DEEEEEEE",
    /* y= 2 */ "DTTTTTTTSDTTTTTTT",
    /* y= 3 */ "DTTTTTTTCDTTTTTTT",
    /* y= 4 */ "DTTTTTTT.DTTTTTTT",
    /* y= 5 */ "DTTTTTTTCDTTTTTTT",
    /* y= 6 */ "DTTTTTTT.DTTTTTTT",
    /* y= 7 */ "DTTTTTTTCDTTTTTTT",
    /* y= 8 */ "DTTTTTTT.DTTTTTTT",
    /* y= 9 */ "DTTTTTTTCDTTTTTTT",
    /* y=10 */ "DTTTTTTT.DTTTTTTT",
    /* y=11 */ "DTTTTTTT.DTTTTTTT",
};

#define MAX_GRID_X BH_GRID_X
#define MAX_GRID_Y BH_GRID_Y

struct noc_arch {
    const char* name;
    const char* const* grid;
    unsigned grid_x;
    unsigned grid_y;
    uint64_t node_id_offset;
    uint64_t endpoint_id_offset;
    uint64_t id_logical_offset;
    // Blackhole addresses nodes in translated space, so a Tensix node may
    // legitimately report an x other than the one we asked for.
    int translated;
};

static const struct noc_arch noc_arch_wh = {
    .name = "Wormhole",
    .grid = wh_grid,
    .grid_x = WH_GRID_X,
    .grid_y = WH_GRID_Y,
    .node_id_offset = WH_NODE_ID_OFFSET,
    .endpoint_id_offset = WH_ENDPOINT_ID_OFFSET,
    .id_logical_offset = WH_ID_LOGICAL_OFFSET,
    .translated = 0,
};

static const struct noc_arch noc_arch_bh = {
    .name = "Blackhole",
    .grid = bh_grid,
    .grid_x = BH_GRID_X,
    .grid_y = BH_GRID_Y,
    .node_id_offset = BH_NODE_ID_OFFSET,
    .endpoint_id_offset = BH_ENDPOINT_ID_OFFSET,
    .id_logical_offset = BH_ID_LOGICAL_OFFSET,
    .translated = 1,
};

// The NOC_ENDPOINT_ID tile type each tile should report, or ENDPOINT_TYPE_ANY
// where the encoding cannot distinguish what we care about.  Wormhole gives the
// empty router nodes the same type as PCIe, and Blackhole does not document a
// type for them at all, so in both cases the empty nodes go unchecked.
static uint32_t expected_endpoint_type(const struct noc_arch* arch, char tile) {
    if (arch == &noc_arch_bh) {
        switch (tile) {
            case TILE_TENSIX: return 0x0100;
            case TILE_ETH: return 0x0200;
            case TILE_PCIE: return 0x0300;
            case TILE_ARC: return 0x0500;
            case TILE_DRAM: return 0x0800;
            case TILE_L2CPU: return 0x0901;
            case TILE_SECURITY: return 0x0A00;
            default: return ENDPOINT_TYPE_ANY;
        }
    }

    switch (tile) {
        case TILE_TENSIX: return 1;
        case TILE_ETH: return 2;
        case TILE_PCIE: return 3;
        case TILE_ARC: return 5;
        case TILE_DRAM: return 8;
        default: return ENDPOINT_TYPE_ANY;
    }
}

static uint32_t endpoint_type(const struct noc_arch* arch, uint32_t endpoint_id) {
    return (arch == &noc_arch_bh) ? BH_ENDPOINT_TYPE(endpoint_id) : WH_ENDPOINT_TYPE(endpoint_id);
}

static const char* tile_type_name(char tile) {
    switch (tile) {
        case TILE_TENSIX: return "Tensix";
        case TILE_ETH: return "Ethernet";
        case TILE_DRAM: return "DRAM";
        case TILE_ARC: return "ARC";
        case TILE_PCIE: return "PCIe";
        case TILE_L2CPU: return "L2CPU";
        case TILE_SECURITY: return "Security";
        case TILE_EMPTY: return "empty";
        default: return "?";
    }
}

// NIU base for a Wormhole node, NOC0.  From get_noc_reg_bases() in syseng
// arc_fw/lib/tensix_utils.c.
//
// The three DRAM values are the 1st, 2nd and 3rd NIU apertures on a GDDR
// channel's shared APB bus.  All three are visible from all three of the
// channel's nodes, so the aperture has to match the node or it reports one of
// its neighbours.  y % 3 picks the right one: checked against every entry of
// tt-umd's hand-verified coordinate-to-port map in tests/api/test_noc.cpp.
//
// Note the DRAM groups in tt-umd's SoC descriptor YAML are *not* in port order;
// they are bank membership, chunked three at a time by the serializer.  Reading
// a port index out of them gives the wrong aperture for all 18 nodes.
static uint64_t wh_niu_base(char tile, unsigned y) {
    switch (tile) {
        case TILE_DRAM:
            switch (y % 3) {
                case 0: return 0x1000A0000ULL;
                case 1: return 0x100080000ULL;
                default: return 0x100090000ULL;
            }
        case TILE_ARC:
        case TILE_PCIE: return 0xFFFB20000ULL;
        default:
            // Tensix, Ethernet and the empty router nodes all map their NIU
            // into the tile's own address space at the same place.
            return 0x0FFB20000ULL;
    }
}

// NIU base for a Blackhole node, NOC0.  From the _noc_reg_base() methods of
// t6py's TensixNocNode / GddrNocNode / Noc2AxiNocNode / ExtraNocNode in
// chip/tile.py, matching tt-umd's NOC0_CONTROL_REG_ADDR_BASE_MAP.  Blackhole
// dropped Wormhole's per-DRAM-node apertures: all 24 DRAM nodes share one base.
static uint64_t bh_niu_base(char tile) {
    switch (tile) {
        case TILE_ARC:
        case TILE_PCIE:
        case TILE_L2CPU:
        case TILE_SECURITY:
            // Bridge tiles: the NIU is an aperture at the top of the address
            // space on the far side of the NOC2AXI bridge.
            return 0xFFFFFFFFFF000000ULL;
        case TILE_EMPTY: return 0x00000000FF000000ULL;
        default:
            // Tensix, Ethernet and DRAM.
            return 0x00000000FFB20000ULL;
    }
}

static uint64_t niu_base(const struct noc_arch* arch, char tile, unsigned y) {
    return (arch == &noc_arch_bh) ? bh_niu_base(tile) : wh_niu_base(tile, y);
}

// Per-node outcome, kept so the bijection check can run over the whole walk
// once it has finished.
#define RESULT_SKIPPED ' '
#define RESULT_PASS '.'
#define RESULT_MISMATCH 'X'
#define RESULT_NO_ANSWER '?'
#define RESULT_UNVISITED '-'

struct node_result {
    char status;
    char tile;
    uint32_t node_id;
    uint32_t endpoint_id;
    uint32_t id_logical;
};

#define NOC_MAX_SKIPS 64

// Options for one sweep.  Passed by the caller rather than held in file
// statics, because reset --all runs a sweep per chip on concurrent threads.
struct noc_sanity_opts {
    struct {
        unsigned x, y;
    } skips[NOC_MAX_SKIPS];
    unsigned skip_count;
    int keep_going;  // -k: walk on past a node that does not answer
    int stop_first;  // --stop-first: stop at the first failure of any kind
};

// What a sweep found.  summary is a ready-to-print phrase ("3 wrong, 1
// silent"); first is the first failing node, for a caller that wants one line.
struct noc_sanity_result {
    unsigned passed, mismatched, silent, skipped, total;
    int bijection_bad;
    char summary[128];
    char first[192];
};

// Failures are reported as they are found rather than collected and printed at
// the end, so that killing a sweep which has wedged the link still leaves the
// evidence behind.  A chip that fails everywhere would bury the log, hence the
// cap; the verdict line still carries the true count.
#define NOC_MAX_REPORTED_FAILURES 20

__attribute__((format(printf, 3, 4))) static void noc_report_failure(
    struct noc_sanity_result* res, unsigned* reported, const char* fmt, ...) {
    va_list ap;
    char line[192];

    va_start(ap, fmt);
    vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);

    if (res->first[0] == '\0') {
        snprintf(res->first, sizeof(res->first), "%s", line);
    }

    if (++*reported > NOC_MAX_REPORTED_FAILURES) {
        if (*reported == NOC_MAX_REPORTED_FAILURES + 1) {
            printf("  (further failures not shown)\n");
            fflush(stdout);
        }
        return;
    }

    printf("  %s\n", line);
    fflush(stdout);
}

static int noc_is_skipped(const struct noc_sanity_opts* opts, unsigned x, unsigned y) {
    for (unsigned i = 0; i < opts->skip_count; i++) {
        if (opts->skips[i].x == x && opts->skips[i].y == y) {
            return 1;
        }
    }
    return 0;
}

// Prints every address the walk would use, without opening the NOC.
static void noc_list_plan(const struct noc_arch* arch, const struct noc_sanity_opts* opts) {
    unsigned planned = 0, skips_hit = 0;

    printf(
        "%s: %u x %u, %u nodes, %s coordinates\n\n",
        arch->name,
        arch->grid_x,
        arch->grid_y,
        arch->grid_x * arch->grid_y,
        arch->translated ? "translated" : "raw");
    printf(
        "   x    y  tile      NIU base            reads at +0x%02llx +0x%02llx +0x%03llx\n",
        (unsigned long long)arch->node_id_offset,
        (unsigned long long)arch->endpoint_id_offset,
        (unsigned long long)arch->id_logical_offset);

    for (unsigned y = 0; y < arch->grid_y; y++) {
        for (unsigned x = 0; x < arch->grid_x; x++) {
            char tile = arch->grid[y][x];
            int skip = noc_is_skipped(opts, x, y);

            if (skip) {
                skips_hit++;
            } else {
                planned++;
            }
            printf(
                "  %2u   %2u  %-8s  0x%016llx%s\n",
                x,
                y,
                tile_type_name(tile),
                (unsigned long long)niu_base(arch, tile, y),
                skip ? "   (skipped)" : "");
        }
    }

    printf("\n%u nodes to read, %u skipped\n", planned, skips_hit);
}

// On Blackhole the sweep of translated coordinates should land on each raw node
// exactly once.  Anything else means the translation tables alias or omit
// nodes, which no individual node's answer would reveal.
static int noc_check_bijection(
    const struct noc_arch* arch,
    const struct node_result results[MAX_GRID_Y][MAX_GRID_X],
    struct noc_sanity_result* res,
    unsigned* reported) {
    unsigned char seen[MAX_GRID_Y][MAX_GRID_X];
    unsigned duplicates = 0;
    unsigned unreached = 0;
    unsigned checked = 0;

    memset(seen, 0, sizeof(seen));

    for (unsigned y = 0; y < arch->grid_y; y++) {
        for (unsigned x = 0; x < arch->grid_x; x++) {
            unsigned rx, ry;

            if (results[y][x].status != RESULT_PASS) {
                continue;
            }

            rx = ID_X(results[y][x].node_id);
            ry = ID_Y(results[y][x].node_id);
            if (rx >= arch->grid_x || ry >= arch->grid_y) {
                noc_report_failure(
                    res, reported, "(%u, %u) reports raw coordinate (%u, %u), off the grid", x, y, rx, ry);
                duplicates++;
                continue;
            }

            checked++;
            if (seen[ry][rx]++) {
                noc_report_failure(
                    res, reported, "raw node (%u, %u) is reached by more than one translated coordinate", rx, ry);
                duplicates++;
            }
        }
    }

    // Only meaningful if the whole grid was visited; a partial walk is
    // expected to leave holes.
    if (checked == arch->grid_x * arch->grid_y) {
        for (unsigned y = 0; y < arch->grid_y; y++) {
            for (unsigned x = 0; x < arch->grid_x; x++) {
                if (!seen[y][x]) {
                    noc_report_failure(
                        res, reported, "raw node (%u, %u) is not reached by any translated coordinate", x, y);
                    unreached++;
                }
            }
        }
    }

    return (duplicates || unreached) ? -1 : 0;
}

// Sweeps the grid.  Returns EXIT_OK, EXIT_TEST_FAILED if a node answered
// wrongly, EXIT_TEST_NO_ANSWER if a node did not answer at all, or
// EXIT_SETUP_ERROR if every node was skipped and nothing was checked.
static int noc_sanity_run(struct chip* chip, const struct noc_sanity_opts* opts, struct noc_sanity_result* res) {
    // Stack-local, not file-static: reset --all sweeps every chip on its own
    // thread, and file statics would have them scribbling over each other.
    struct node_result results[MAX_GRID_Y][MAX_GRID_X];
    const struct noc_arch* arch = chip->is_blackhole ? &noc_arch_bh : &noc_arch_wh;
    struct noc_window win;
    unsigned reported = 0;
    size_t used = 0;
    int rc;

    memset(res, 0, sizeof(*res));
    res->total = arch->grid_x * arch->grid_y;

    for (unsigned y = 0; y < arch->grid_y; y++) {
        for (unsigned x = 0; x < arch->grid_x; x++) {
            results[y][x].status = RESULT_UNVISITED;
        }
    }

    noc_window_open(&win, chip->fd);

    for (unsigned y = 0; y < arch->grid_y; y++) {
        for (unsigned x = 0; x < arch->grid_x; x++) {
            struct node_result* r = &results[y][x];
            char tile = arch->grid[y][x];
            uint64_t base = niu_base(arch, tile, y);
            uint32_t want_type, got_type;
            unsigned got_x, got_y;
            const char* problem;
            const char* silent_reg;

            r->tile = tile;

            if (noc_is_skipped(opts, x, y)) {
                r->status = RESULT_SKIPPED;
                res->skipped++;
                continue;
            }

            // Each read is tested for all ones before the next one is
            // issued: once a register reads all ones the node is not
            // answering, and issuing further reads to it is exactly what
            // the hazard policy says not to do.  All three registers get
            // the test, so a node that answers NOC_NODE_ID but nothing
            // else is still classified silent rather than misconfigured.
            silent_reg = nullptr;
            r->node_id = noc_read32(&win, x, y, base + arch->node_id_offset);
            if (r->node_id == ALL_ONES) {
                silent_reg = "NOC_NODE_ID";
            }

            if (!silent_reg) {
                r->endpoint_id = noc_read32(&win, x, y, base + arch->endpoint_id_offset);
                if (r->endpoint_id == ALL_ONES) {
                    silent_reg = "NOC_ENDPOINT_ID";
                }
            }

            if (!silent_reg) {
                r->id_logical = noc_read32(&win, x, y, base + arch->id_logical_offset);
                if (r->id_logical == ALL_ONES) {
                    silent_reg = "NOC_ID_LOGICAL";
                }
            }

            if (silent_reg) {
                r->status = RESULT_NO_ANSWER;
                res->silent++;
                noc_report_failure(
                    res, &reported, "(%u, %u) %s: no answer, %s read all ones", x, y, tile_type_name(tile), silent_reg);
                if (!opts->keep_going) {
                    printf("  stopping here; -k walks the rest anyway\n");
                    fflush(stdout);
                    goto summary;
                }
                continue;
            }

            got_x = ID_X(r->node_id);
            got_y = ID_Y(r->node_id);
            want_type = expected_endpoint_type(arch, tile);
            got_type = endpoint_type(arch, r->endpoint_id);
            problem = nullptr;

            // y is never translated within the swept box, so it must come
            // back exactly.  x is only translated for Tensix, and then only
            // onto another Tensix column.
            if (got_y != y) {
                problem = "wrong y";
            } else if (got_x >= arch->grid_x) {
                problem = "x off the grid";
            } else if (!arch->translated || tile != TILE_TENSIX) {
                if (got_x != x) {
                    problem = "wrong x";
                }
            } else if (arch->grid[y][got_x] != TILE_TENSIX) {
                problem = "translated to a column that is not Tensix";
            }

            if (!problem && want_type != ENDPOINT_TYPE_ANY && got_type != want_type) {
                problem = "wrong tile type";
            } else if (!problem && ENDPOINT_NOC(r->endpoint_id) != 0) {
                problem = "answered for the wrong NOC";
            } else if (!problem && (ID_WIDTH(r->node_id) != arch->grid_x || ID_HEIGHT(r->node_id) != arch->grid_y)) {
                problem = "wrong grid dimensions";
            } else if (!problem && ID_DIM_ORDER(r->node_id) != 1) {
                problem = "wrong routing dimension order for NOC0";
            }

            if (problem) {
                char want_str[16];

                if (want_type == ENDPOINT_TYPE_ANY) {
                    snprintf(want_str, sizeof(want_str), "unchecked");
                } else {
                    snprintf(want_str, sizeof(want_str), "0x%x", want_type);
                }

                r->status = RESULT_MISMATCH;
                res->mismatched++;
                noc_report_failure(
                    res,
                    &reported,
                    "(%u, %u) %s: %s\n"
                    "    NOC_NODE_ID     0x%08x -> (%u, %u), grid %ux%u, dim_order %u\n"
                    "    NOC_ENDPOINT_ID 0x%08x -> type 0x%x (expected %s), noc %u\n"
                    "    NOC_ID_LOGICAL  0x%08x -> (%u, %u)",
                    x,
                    y,
                    tile_type_name(tile),
                    problem,
                    r->node_id,
                    got_x,
                    got_y,
                    ID_WIDTH(r->node_id),
                    ID_HEIGHT(r->node_id),
                    ID_DIM_ORDER(r->node_id),
                    r->endpoint_id,
                    got_type,
                    want_str,
                    ENDPOINT_NOC(r->endpoint_id),
                    r->id_logical,
                    ID_X(r->id_logical),
                    ID_Y(r->id_logical));

                // A node that answered is alive, so continuing is neither
                // hazardous nor slow, and the pattern across the grid is
                // the diagnosis.  --stop-first opts out.
                if (opts->stop_first) {
                    printf("  stopping here; --stop-first was given\n");
                    fflush(stdout);
                    goto summary;
                }
                continue;
            }

            r->status = RESULT_PASS;
            res->passed++;
        }
    }

summary:
    noc_window_close(&win);

    // On Wormhole the addressed coordinate is the raw one, so the bijection is
    // what we already checked node by node and re-checking says nothing.  On
    // Blackhole the duplicate check is meaningful even over a partial walk, but
    // the completeness half needs every node to have answered, and
    // noc_check_bijection() only runs it then.
    if (arch->translated) {
        if (noc_check_bijection(arch, results, res, &reported) != 0) {
            res->bijection_bad = 1;
        }
        if (res->passed != res->total) {
            printf("  (aliasing checked over %u of %u nodes; completeness not verified)\n", res->passed, res->total);
            fflush(stdout);
        }
    }

    if (res->mismatched) {
        used += (size_t)snprintf(res->summary + used, sizeof(res->summary) - used, "%u wrong", res->mismatched);
    }
    if (res->silent) {
        used += (size_t)snprintf(
            res->summary + used, sizeof(res->summary) - used, "%s%u silent", used ? ", " : "", res->silent);
    }
    if (res->bijection_bad) {
        snprintf(res->summary + used, sizeof(res->summary) - used, "%stranslation aliasing", used ? ", " : "");
    }

    if (res->silent || res->mismatched || res->bijection_bad) {
        rc = res->silent ? EXIT_TEST_NO_ANSWER : EXIT_TEST_FAILED;
    } else if (res->skipped == res->total) {
        rc = EXIT_SETUP_ERROR;  // nothing was checked, so nothing is healthy
    } else {
        rc = EXIT_OK;
    }

    return rc;
}

static void noc_sanity_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [-d PATH] [-l] [-k] [--stop-first] [-s X,Y]\n", prog);
    fprintf(stderr, "  -d PATH       device to test, path or ordinal (default /dev/tenstorrent/0)\n");
    fprintf(stderr, "  -l            list the plan and exit; the device is opened only to\n");
    fprintf(stderr, "                identify the architecture, no NOC read is issued\n");
    fprintf(stderr, "  -k            keep going after a node fails to answer\n");
    fprintf(stderr, "  --stop-first  stop at the first failure of any kind\n");
    fprintf(stderr, "  -s X,Y        skip a coordinate; repeatable\n");
    fprintf(stderr, "\nAsks all 204 (Blackhole) or 120 (Wormhole) NOC0 nodes who they are, across\n");
    fprintf(stderr, "every tile type.  A node that does not answer stops the sweep; a node that\n");
    fprintf(stderr, "answers wrongly does not, because the pattern across the grid is the\n");
    fprintf(stderr, "diagnosis.  Prints one [PASS] or [FAIL] line last.\n");
    fprintf(stderr, "Exit status:\n");
    fprintf(stderr, "  0 = every node reported correct coordinates\n");
    fprintf(stderr, "  1 = usage or setup error\n");
    fprintf(stderr, "  2 = a node answered wrongly\n");
    fprintf(stderr, "  3 = a node did not answer at all\n");
}

static int noc_sanity_main(int argc, char* argv[], const char* prog) {
    static const struct option longopts[] = {
        {"stop-first", no_argument, nullptr, 'F'},
        {nullptr, 0, nullptr, 0},
    };
    const char* device_path = "/dev/tenstorrent/0";
    int device_path_set = 0;
    struct noc_sanity_opts opts;
    struct noc_sanity_result res;
    const struct noc_arch* arch;
    struct chip chip;
    int list_only = 0;
    int rc;
    int opt;

    memset(&opts, 0, sizeof(opts));

    while ((opt = getopt_long(argc, argv, "d:lks:h", longopts, nullptr)) != -1) {
        switch (opt) {
            case 'd':
                device_path = device_path_arg(optarg);
                device_path_set = 1;
                break;
            case 'l': list_only = 1; break;
            case 'k': opts.keep_going = 1; break;
            case 'F': opts.stop_first = 1; break;
            case 's': {
                unsigned x, y;
                char junk;

                if (sscanf(optarg, "%u,%u%c", &x, &y, &junk) != 2) {
                    fprintf(stderr, "-s wants X,Y, got \"%s\"\n", optarg);
                    return EXIT_SETUP_ERROR;
                }
                if (opts.skip_count >= NOC_MAX_SKIPS) {
                    fprintf(stderr, "too many -s options (max %u)\n", NOC_MAX_SKIPS);
                    return EXIT_SETUP_ERROR;
                }
                opts.skips[opts.skip_count].x = x;
                opts.skips[opts.skip_count].y = y;
                opts.skip_count++;
                break;
            }
            case 'h': noc_sanity_usage(prog); return EXIT_OK;
            default: noc_sanity_usage(prog); return EXIT_SETUP_ERROR;
        }
    }
    if (optind < argc) {
        if (device_path_set) {
            fprintf(stderr, "Error: device given both as -d %s and as '%s'\n", device_path, argv[optind]);
            noc_sanity_usage(prog);
            return EXIT_SETUP_ERROR;
        }
        if (argc - optind > 1) {
            fprintf(stderr, "Error: unexpected extra argument '%s'\n", argv[optind + 1]);
            noc_sanity_usage(prog);
            return EXIT_SETUP_ERROR;
        }
        device_path = device_path_arg(argv[optind]);
    }

    chip_open(&chip, device_path);
    arch = chip.is_blackhole ? &noc_arch_bh : &noc_arch_wh;

    // A skip outside the grid matches nothing; that is a typo, and it means a
    // coordinate the user thought was excluded is not.
    for (unsigned i = 0; i < opts.skip_count; i++) {
        if (opts.skips[i].x >= arch->grid_x || opts.skips[i].y >= arch->grid_y) {
            fprintf(
                stderr,
                "warning: -s %u,%u is outside the %ux%u grid and skips nothing\n",
                opts.skips[i].x,
                opts.skips[i].y,
                arch->grid_x,
                arch->grid_y);
        }
    }

    // Enough to tell one card in a machine from another in a log collected
    // weeks later.
    printf(
        "%s: %s at %04x:%02x:%02x.%u, %04x:%04x subsystem %04x:%04x, %u NOC0 nodes\n",
        device_path,
        arch->name,
        chip.info.pci_domain,
        (chip.info.bus_dev_fn >> 8) & 0xFF,
        (chip.info.bus_dev_fn >> 3) & 0x1F,
        chip.info.bus_dev_fn & 0x7,
        chip.info.vendor_id,
        chip.info.device_id,
        chip.info.subsystem_vendor_id,
        chip.info.subsystem_id,
        arch->grid_x * arch->grid_y);
    fflush(stdout);

    if (list_only) {
        noc_list_plan(arch, &opts);
        chip_close(&chip);
        return EXIT_OK;
    }

    rc = noc_sanity_run(&chip, &opts, &res);
    chip_close(&chip);

    if (rc == EXIT_SETUP_ERROR) {
        verdict(0, "all %u NOC0 nodes were skipped, nothing was checked", res.total);
    } else if (rc != EXIT_OK) {
        if (res.skipped) {
            verdict(0, "%s; %u of %u NOC0 nodes ok, %u skipped", res.summary, res.passed, res.total, res.skipped);
        } else {
            verdict(0, "%s; %u of %u NOC0 nodes ok", res.summary, res.passed, res.total);
        }
    } else if (res.skipped) {
        verdict(1, "%u of %u NOC0 nodes report correct coordinates, %u skipped", res.passed, res.total, res.skipped);
    } else {
        verdict(1, "all %u NOC0 nodes report correct coordinates", res.total);
    }

    return rc;
}

#define DMA_LOOPBACK_TRANSFER_SIZE (4ULL << 20)
#define DMA_LOOPBACK_1G_PAGE_SIZE (1ULL << 30)
#define DMA_LOOPBACK_2M_PAGE_SIZE (2ULL << 20)

struct dma_loopback_buffer {
    uint8_t* mem;
    size_t backing_size;
    size_t transfer_size;
    uint64_t noc_addr;
};

static int dma_loopback_pin(int fd, void* mem, size_t size, uint64_t* noc_addr) {
    struct tenstorrent_pin_pages pin;

    memset(&pin, 0, sizeof(pin));
    pin.in.output_size_bytes = sizeof(pin.out);
    pin.in.flags = TENSTORRENT_PIN_PAGES_NOC_DMA;
    pin.in.virtual_address = (uintptr_t)mem;
    pin.in.size = size;
    if (ioctl(fd, TENSTORRENT_IOCTL_PIN_PAGES, &pin) != 0) {
        return -1;
    }

    *noc_addr = pin.out.noc_address;
    return 0;
}

static int dma_loopback_has_translated_iommu(const struct chip* chip) {
    char iommu_dir[112];
    char pcidir[80];
    char value[64];
    char bdf[16];

    format_bdf(bdf, sizeof(bdf), &chip->info);
    snprintf(pcidir, sizeof(pcidir), "/sys/bus/pci/devices/%s", bdf);
    snprintf(iommu_dir, sizeof(iommu_dir), "%s/iommu_group", pcidir);
    if (read_sysfs_attr(iommu_dir, "type", value, sizeof(value)) != 0) {
        return 0;
    }
    return strcmp(value, "identity") != 0;
}

static int dma_loopback_try_buffer(
    int fd, struct dma_loopback_buffer* buf, size_t backing_size, size_t transfer_size, int mmap_flags) {
    buf->mem = static_cast<uint8_t*>(mmap(nullptr, backing_size, PROT_READ | PROT_WRITE, mmap_flags, -1, 0));
    if (buf->mem == MAP_FAILED) {
        return -1;
    }
    if (dma_loopback_pin(fd, buf->mem, backing_size, &buf->noc_addr) == 0) {
        buf->backing_size = backing_size;
        buf->transfer_size = transfer_size;
        return 0;
    }

    {
        int pin_errno = errno;

        if (munmap(buf->mem, backing_size) != 0) {
            DIE("munmap of unpinnable DMA destination failed");
        }
        buf->mem = static_cast<uint8_t*>(MAP_FAILED);
        errno = pin_errno;
    }
    return -1;
}

static void dma_loopback_hugepage_advice(const char* prefix) {
    printf("%sReserve a 1 GiB hugepage for the full 4 MiB test with:\n", prefix);
    printf(
        "%s  echo 1 | sudo tee "
        "/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages\n",
        prefix);
}

static void dma_loopback_alloc_buffer(struct chip* chip, struct dma_loopback_buffer* buf, const char* prefix) {
    long page_size = sysconf(_SC_PAGESIZE);
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;

    memset(buf, 0, sizeof(*buf));
    buf->mem = static_cast<uint8_t*>(MAP_FAILED);
    if (page_size <= 0) {
        DIE("cannot determine the host page size");
    }

    if (dma_loopback_has_translated_iommu(chip)) {
        if (dma_loopback_try_buffer(chip->fd, buf, DMA_LOOPBACK_TRANSFER_SIZE, DMA_LOOPBACK_TRANSFER_SIZE, flags) ==
            0) {
            return;
        }
        printf(
            "%s4 MiB DMA buffer could not be pinned (%s); "
            "reducing the test to one %ld KiB page\n",
            prefix,
            strerror(errno),
            page_size >> 10);
    } else {
        if (dma_loopback_try_buffer(
                chip->fd,
                buf,
                DMA_LOOPBACK_1G_PAGE_SIZE,
                DMA_LOOPBACK_TRANSFER_SIZE,
                flags | MAP_HUGETLB | MAP_HUGE_1GB) == 0) {
            return;
        }
        if (dma_loopback_try_buffer(
                chip->fd,
                buf,
                DMA_LOOPBACK_2M_PAGE_SIZE,
                DMA_LOOPBACK_2M_PAGE_SIZE,
                flags | MAP_HUGETLB | MAP_HUGE_2MB) == 0) {
            printf("%sUsing one 2 MiB hugepage; reducing the DMA test to 2 MiB\n", prefix);
            dma_loopback_hugepage_advice(prefix);
            return;
        }
        printf(
            "%sNo usable 1 GiB or 2 MiB hugepage; "
            "reducing the DMA test to one %ld KiB page\n",
            prefix,
            page_size >> 10);
        dma_loopback_hugepage_advice(prefix);
    }

    if (dma_loopback_try_buffer(chip->fd, buf, (size_t)page_size, (size_t)page_size, flags) != 0) {
        DIEX("PIN_PAGES of one %ld KiB page failed", page_size >> 10);
    }
}

static void dma_loopback_unpin(int fd, void* mem, size_t size) {
    struct tenstorrent_unpin_pages unpin;

    memset(&unpin, 0, sizeof(unpin));
    unpin.in.virtual_address = (uintptr_t)mem;
    unpin.in.size = size;
    if (ioctl(fd, TENSTORRENT_IOCTL_UNPIN_PAGES, &unpin) != 0) {
        DIE("ioctl UNPIN_PAGES failed");
    }
}

static void dma_loopback_fill_random(void* mem, size_t size) {
    uint8_t* p = static_cast<uint8_t*>(mem);

    while (size != 0) {
        ssize_t n = getrandom(p, size, 0);

        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            DIE("getrandom failed");
        }
        p += n;
        size -= n;
    }
}

static void dma_loopback_write(
    struct noc_window* win, unsigned x, unsigned y, uint64_t addr, const void* src, size_t size) {
    const uint8_t* src_bytes = static_cast<const uint8_t*>(src);

    while (size != 0) {
        size_t offset = addr & (TLB_SIZE - 1);
        size_t chunk = TLB_SIZE - offset;
        volatile uint32_t* dst;
        const uint32_t* src32;

        if (chunk > size) {
            chunk = size;
        }
        noc_window_aim(win, x, y, addr);
        dst = (volatile uint32_t*)(win->mmio + offset);
        src32 = (const uint32_t*)src_bytes;
        for (size_t i = 0; i < chunk / sizeof(*dst); i++) {
            dst[i] = src32[i];
        }
        __sync_synchronize();

        addr += chunk;
        src_bytes += chunk;
        size -= chunk;
    }
}

static int dma_loopback_run(struct chip* chip, const char* prefix, size_t* transfer_size) {
    struct dma_loopback_buffer dest;
    struct noc_window win;
    uint8_t* pattern;
    unsigned pcie_x;
    unsigned pcie_y;
    int mismatch;

    dma_loopback_alloc_buffer(chip, &dest, prefix);

    pattern = static_cast<uint8_t*>(
        mmap(nullptr, dest.transfer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (pattern == MAP_FAILED) {
        DIE("mmap of DMA source pattern failed");
    }
    dma_loopback_fill_random(pattern, dest.transfer_size);

    pcie_x = chip->is_blackhole ? 19 : 0;
    pcie_y = chip->is_blackhole ? 24 : 3;

    noc_window_open_wc(&win, chip->fd);
    dma_loopback_write(&win, pcie_x, pcie_y, dest.noc_addr, pattern, dest.transfer_size);
    noc_window_close(&win);

    noc_window_open(&win, chip->fd);
    (void)noc_read32(&win, pcie_x, pcie_y, dest.noc_addr);
    (void)noc_read32(&win, pcie_x, pcie_y, dest.noc_addr + dest.transfer_size - sizeof(uint32_t));
    noc_window_close(&win);

    mismatch = memcmp(dest.mem, pattern, dest.transfer_size) != 0;
    if (mismatch) {
        unsigned shown = 0;

        for (size_t i = 0; i < dest.transfer_size && shown < 12; i++) {
            if (dest.mem[i] == pattern[i]) {
                continue;
            }
            printf("%soffset 0x%zx: got 0x%02x, expected 0x%02x\n", prefix, i, dest.mem[i], pattern[i]);
            shown++;
        }
    }

    dma_loopback_unpin(chip->fd, dest.mem, dest.backing_size);
    if (munmap(dest.mem, dest.backing_size) != 0) {
        DIE("munmap of DMA destination failed");
    }
    if (munmap(pattern, dest.transfer_size) != 0) {
        DIE("munmap of DMA source pattern failed");
    }

    *transfer_size = dest.transfer_size;
    return mismatch;
}

static int reset_health_checks(struct reset_dev* d) {
    // Default options: stop at a node that does not answer, walk past one that
    // answers wrongly, skip nothing.
    struct noc_sanity_opts opts;
    struct noc_sanity_result res;
    char prefix[24];
    size_t transfer_size;
    int rc = EXIT_OK;

    memset(&opts, 0, sizeof(opts));
    if (noc_sanity_run(&d->chip, &opts, &res) != EXIT_OK) {
        printf("%s: NOC sweep failed: %s; %u of %u nodes ok\n", d->bdf, res.summary, res.passed, res.total);
        if (res.first[0] != '\0') {
            printf("%s: first failure: %s\n", d->bdf, res.first);
        }
        fflush(stdout);
        rc = EXIT_RESET_FAILED;
        goto idle;
    }
    printf("%s: NOC sanity ok, all %u nodes\n", d->bdf, res.total);
    fflush(stdout);

    snprintf(prefix, sizeof(prefix), "%s: ", d->bdf);
    if (dma_loopback_run(&d->chip, prefix, &transfer_size) != 0) {
        printf("%s: DMA loopback data mismatch\n", d->bdf);
        fflush(stdout);
        rc = EXIT_RESET_FAILED;
        goto idle;
    }
    if (transfer_size >= (1ULL << 20)) {
        printf("%s: DMA loopback ok, %zu MiB transferred\n", d->bdf, transfer_size >> 20);
    } else {
        printf("%s: DMA loopback ok, %zu KiB transferred\n", d->bdf, transfer_size >> 10);
    }
    fflush(stdout);

idle:
    if (reset_request_idle(d) != EXIT_OK) {
        rc = EXIT_RESET_FAILED;
    }
    return rc;
}

static void dma_loopback_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [DEVICE]\n", prog);
    fprintf(stderr, "       %s -d DEVICE\n", prog);
    fprintf(stderr, "  -d PATH  device to test, path or ordinal (default /dev/tenstorrent/0)\n");
    fprintf(stderr, "\nWrites up to 4 MiB through a WC TLB into NOC-visible host memory.\n");
    fprintf(stderr, "Without an IOMMU, one hugepage or base page backs the transfer.\n");
    fprintf(stderr, "Prints one [PASS] or [FAIL] line last.\n");
    fprintf(stderr, "Exit status:\n");
    fprintf(stderr, "  0 = test passed\n");
    fprintf(stderr, "  1 = usage or setup error\n");
    fprintf(stderr, "  2 = loopback data mismatch\n");
}

static int dma_loopback_main(int argc, char* argv[], const char* prog) {
    const char* device_path = "/dev/tenstorrent/0";
    int device_path_set = 0;
    struct chip chip;
    size_t transfer_size;
    int mismatch;
    int opt;

    while ((opt = getopt(argc, argv, "d:h")) != -1) {
        switch (opt) {
            case 'd':
                device_path = device_path_arg(optarg);
                device_path_set = 1;
                break;
            case 'h': dma_loopback_usage(prog); return EXIT_OK;
            default: dma_loopback_usage(prog); return EXIT_SETUP_ERROR;
        }
    }
    if (optind < argc) {
        if (device_path_set) {
            fprintf(stderr, "Error: device given both as -d %s and as '%s'\n", device_path, argv[optind]);
            dma_loopback_usage(prog);
            return EXIT_SETUP_ERROR;
        }
        if (argc - optind > 1) {
            fprintf(stderr, "Error: unexpected extra argument '%s'\n", argv[optind + 1]);
            dma_loopback_usage(prog);
            return EXIT_SETUP_ERROR;
        }
        device_path = device_path_arg(argv[optind]);
    }

    chip_open(&chip, device_path);
    mismatch = dma_loopback_run(&chip, "  ", &transfer_size);
    chip_close(&chip);

    if (mismatch) {
        verdict(0, "%s DMA loopback data mismatch", chip.arch_name);
        return EXIT_TEST_FAILED;
    }

    if (transfer_size >= (1ULL << 20)) {
        verdict(1, "%s DMA loopback, %zu MiB transferred", chip.arch_name, transfer_size >> 20);
    } else {
        verdict(1, "%s DMA loopback, %zu KiB transferred", chip.arch_name, transfer_size >> 10);
    }
    return EXIT_OK;
}

static const struct test_subcommand {
    const char* name;
    int (*run)(int argc, char* argv[], const char* prog);
    const char* summary;
} test_subcommands[] = {
    {"noc_sanity", noc_sanity_main, "check that NOC node IDs match their coordinates"},
    {"dma_loopback", dma_loopback_main, "DMA from the NOC into a device-visible host buffer"},
};

#define TEST_SUBCOMMAND_COUNT (sizeof(test_subcommands) / sizeof(test_subcommands[0]))

static void test_usage(const char* prog) {
    fprintf(stderr, "Usage: %s <test> [options]\n", prog);
    fprintf(stderr, "       %s --all [DEVICE]\n\n", prog);
    fprintf(stderr, "Tests:\n");
    fprintf(stderr, "  %-12s  run every test\n", "--all");
    for (const auto& test_subcommand : test_subcommands) {
        fprintf(stderr, "  %-12s  %s\n", test_subcommand.name, test_subcommand.summary);
    }
    fprintf(stderr, "\n%s <test> -h gives the test's own usage.\n", prog);
}

static int test_all_main(int argc, char* argv[], const char* prog) {
    const char* device_path = "/dev/tenstorrent/0";
    int device_path_set = 0;
    int opt;

    optind = 1;
    while ((opt = getopt(argc, argv, "d:h")) != -1) {
        switch (opt) {
            case 'd':
                device_path = device_path_arg(optarg);
                device_path_set = 1;
                break;
            case 'h':
                fprintf(stderr, "Usage: %s --all [DEVICE]\n", prog);
                fprintf(stderr, "       %s --all -d DEVICE\n", prog);
                return EXIT_OK;
            default: return EXIT_SETUP_ERROR;
        }
    }
    if (optind < argc) {
        if (device_path_set) {
            fprintf(stderr, "Error: device given both as -d %s and as '%s'\n", device_path, argv[optind]);
            return EXIT_SETUP_ERROR;
        }
        if (argc - optind > 1) {
            fprintf(stderr, "Error: unexpected extra argument '%s'\n", argv[optind + 1]);
            return EXIT_SETUP_ERROR;
        }
        device_path = device_path_arg(argv[optind]);
    }

    for (const auto& test_subcommand : test_subcommands) {
        char test_prog[96];
        char* test_argv[] = {
            (char*)test_subcommand.name,
            (char*)"-d",
            (char*)device_path,
            nullptr,
        };
        int rc;

        printf("=== %s ===\n", test_subcommand.name);
        snprintf(test_prog, sizeof(test_prog), "%s %s", prog, test_subcommand.name);
        optind = 0;
        rc = test_subcommand.run(3, test_argv, test_prog);
        if (rc != EXIT_OK) {
            return rc;
        }
    }

    verdict(1, "all %zu tests passed", TEST_SUBCOMMAND_COUNT);
    return EXIT_OK;
}

static int test_main(int argc, char* argv[], const char* prog) {
    char test_prog[96];

    if (argc < 2) {
        test_usage(prog);
        return EXIT_SETUP_ERROR;
    }
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        test_usage(prog);
        return EXIT_OK;
    }
    if (strcmp(argv[1], "--all") == 0) {
        return test_all_main(argc - 1, argv + 1, prog);
    }

    for (const auto& test_subcommand : test_subcommands) {
        if (strcmp(argv[1], test_subcommand.name) == 0) {
            snprintf(test_prog, sizeof(test_prog), "%s %s", prog, test_subcommand.name);
            return test_subcommand.run(argc - 1, argv + 1, test_prog);
        }
    }

    fprintf(stderr, "%s: unknown test '%s'\n\n", prog, argv[1]);
    test_usage(prog);
    return EXIT_SETUP_ERROR;
}

// ============================================================================
// Subcommand dispatch
// ============================================================================

static const struct subcommand {
    const char* name;
    int (*run)(int argc, char* argv[], const char* prog);
    const char* summary;
} subcommands[] = {
    {"discover", discover_main, "enumerate every device, one line each, with a liveness verdict"},
    {"info", info_main, "one-screen inventory: PCI link, IOMMU, board, DRAM, firmware, vitals"},
    {"scratch", scratch_main, "dump the ARC scratch register banks"},
    {"telemetry", telemetry_main, "dump every telemetry tag ARC firmware publishes"},
    {"read32", read32_main, "read one 32-bit word from a NOC endpoint"},
    {"write32", write32_main, "write one 32-bit word to a NOC endpoint"},
    {"test", test_main, "run one or all chip tests"},
    {"hung", hung_main, "check whether the chip is hung, via config space and sysfs"},
    {"reset", reset_main, "reset the chip in place and check it came back"},
    {"nuke", nuke_main, "kill every process holding the device open"},
};

#define SUBCOMMAND_COUNT (sizeof(subcommands) / sizeof(subcommands[0]))

static void multi_usage(const char* prog) {
    fprintf(stderr, "Usage: %s <subcommand> [options]\n\n", prog);
    fprintf(stderr, "Subcommands:\n");
    for (const auto& subcommand : subcommands) {
        fprintf(stderr, "  %-10s  %s\n", subcommand.name, subcommand.summary);
    }
    fprintf(stderr, "\n%s <subcommand> -h gives the subcommand's own usage.\n", prog);
}

int main(int argc, char* argv[]) {
    static char prog[64];
    const char* base;

    base = strrchr(argv[0], '/');
    base = base ? base + 1 : argv[0];

    if (argc < 2) {
        multi_usage(base);
        return EXIT_SETUP_ERROR;
    }

    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        multi_usage(base);
        return EXIT_OK;
    }

    for (const auto& subcommand : subcommands) {
        if (strcmp(argv[1], subcommand.name) == 0) {
            snprintf(prog, sizeof(prog), "%s %s", base, subcommand.name);
            return subcommand.run(argc - 1, argv + 1, prog);
        }
    }

    fprintf(stderr, "%s: unknown subcommand '%s'\n\n", base, argv[1]);
    multi_usage(base);
    return EXIT_SETUP_ERROR;
}
