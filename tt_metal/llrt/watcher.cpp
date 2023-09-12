// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <thread>
#include <unistd.h>
#include <chrono>
#include <ctime>
#include <unordered_map>
#include <memory>

#include "llrt.hpp"
#include "watcher.hpp"
#include "rtoptions.hpp"
#include "dev_mem_map.h"

// XXXX TODO(PGK): fix include paths so device can export interfaces
#include "tt_metal/src/firmware/riscv/common/debug_sanitize.h"

#include "noc/noc_parameters.h"
#include "noc/noc_overlay_parameters.h"

namespace tt {
namespace llrt {
namespace watcher {


// XXXX TODO(PGK) get these from soc
#define NOC_DRAM_ADDR_BASE 0
#define NOC_DRAM_ADDR_SIZE 1073741824
#define NOC_DRAM_ADDR_END (NOC_DRAM_ADDR_BASE + NOC_DRAM_ADDR_SIZE)

#define DEBUG_VALID_L1_ADDR(a, l) (((a) >= MEM_L1_BASE) && ((a) + (l) <= MEM_L1_BASE + MEM_L1_SIZE))

// TODO(PGK): remove soft reset when fw is downloaded at init
#define DEBUG_VALID_REG_ADDR(a)                                                        \
    ((((a) >= NOC_OVERLAY_START_ADDR) &&                                               \
      ((a) < NOC_OVERLAY_START_ADDR + NOC_STREAM_REG_SPACE_SIZE * NOC_NUM_STREAMS)) || \
     ((a) == RISCV_DEBUG_REG_SOFT_RESET_0))
#define DEBUG_VALID_WORKER_ADDR(a, l) (DEBUG_VALID_L1_ADDR(a, l) || (DEBUG_VALID_REG_ADDR(a) && (l) == 4))
#define DEBUG_VALID_DRAM_ADDR(a, l) (((a) >= NOC_DRAM_ADDR_BASE) && ((a) + (l) <= NOC_DRAM_ADDR_END))

#define DEBUG_VALID_ETH_ADDR(a, l) (((a) >= MEM_ETH_BASE) && ((a) + (l) < MEM_ETH_BASE + MEM_ETH_SIZE))


class WatcherDevice {
  public:
    tt_cluster *cluster_;
    int device_id_;
    std::function<CoreCoord ()>get_grid_size_;
    std::function<CoreCoord (CoreCoord)>worker_from_logical_;

    WatcherDevice(tt_cluster *cluster, int device_id, std::function<CoreCoord ()>get_grid_size, std::function<CoreCoord (CoreCoord)>worker_from_logical);
};

constexpr uint64_t DEBUG_SANITIZE_NOC_SENTINEL_OK_64 = 0xbadabadabadabada;
constexpr uint32_t DEBUG_SANITIZE_NOC_SENTINEL_OK_32 = 0xbadabada;
constexpr uint16_t DEBUG_SANITIZE_NOC_SENTINEL_OK_16 = 0xbada;
constexpr uint32_t N_NOCS = 2;

static bool enabled = false;
static std::mutex watch_mutex;
static std::unordered_map<void *, std::shared_ptr<WatcherDevice>> devices;
static FILE *logfile = nullptr;
static std::chrono::time_point start_time = std::chrono::system_clock::now();

WatcherDevice::WatcherDevice(tt_cluster *cluster, int device_id, std::function<CoreCoord ()>get_grid_size, std::function<CoreCoord (CoreCoord)>worker_from_logical) : cluster_(cluster), device_id_(device_id), get_grid_size_(get_grid_size), worker_from_logical_(worker_from_logical) {
}

static uint32_t get_elapsed_secs() {
    std::chrono::time_point now_time = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_secs = now_time - start_time;

    return (uint32_t)elapsed_secs.count();
}

static FILE * create_file(const string& log_path) {

    FILE *f;

    const char *fmode = getenv("TT_METAL_WATCHER_APPEND") ? "a" : "w";
    string fname = log_path + "watcher.log";
    if ((f = fopen(fname.c_str(), fmode)) == nullptr) {
        log_fatal(LogLLRuntime, "Watcher failed to create log file\n");
        exit(1);
    }
    fprintf(f, "At %ds starting\n", watcher::get_elapsed_secs());
    fprintf(f, "Legend:\n");
    fprintf(f, "\tComma separated list specifices waypoint for BRISC,NCRISC,TRISC0,TRISC1,TRISC2\n");
    fprintf(f, "\tI=initialization sequence\n");
    fprintf(f, "\tW=wait (top of spin loop)\n");
    fprintf(f, "\tR=run (entering kernel)\n");
    fprintf(f, "\tD=done (finished spin loop)\n");
    fprintf(f, "\tX=host written value prior to fw launch\n");
    fprintf(f, "\tU=unexpected value\n");
    fprintf(f, "\n");
    fprintf(f, "\tA single character status is in the FW, other characters clarify where, eg:\n");
    fprintf(f, "\t\tNRW is \"noc read wait\"\n");
    fprintf(f, "\t\tNWD is \"noc write done\"\n");
    fprintf(f, "\tnoc<n>:<risc>{a, l}=an L1 address used by NOC<n> by <riscv> (eg, local src address)\n");
    fprintf(f, "\tnoc<n>:<riscv>{(x,y), a, l}=NOC<n> unicast address used by <riscv>\n");
    fprintf(f, "\tnoc<n>:<riscv>{(x1,y1)-(x2,y2), a, l}=NOC<n> multicast address used by <riscv>\n");
    fprintf(f, "\n");

    return f;
}

static void dump_l1_status(FILE *f, WatcherDevice *wdev, CoreCoord core) {

    // Read L1 address 0, looking for memory corruption
    std::vector<uint32_t> data;
    data = read_hex_vec_from_core(wdev->cluster_, wdev->device_id_, core, MEM_L1_BASE, sizeof(uint32_t));
    // XXXX TODO(pgk): get this const from llrt (jump to fw insn)
    if (data[0] != 0x2010006f) {
        fprintf(f, "L1[0]=bad 0x%08x ", data[0]);
    }
}

static const char * get_sanity_riscv_name(uint32_t type)
{
    switch (type) {
    case DebugSanitizeBrisc:
        return "brisc";
    case DebugSanitizeNCrisc:
        return "ncrisc";
    case DebugSanitizeTrisc0:
        return "trisc0";
    case DebugSanitizeTrisc1:
        return "trisc1";
    case DebugSanitizeTrisc2:
        return "trisc2";
    default:
        log_fatal(LogLLRuntime, "Watcher unexpected riscv type {}", type);
        exit(-1);
    }
}

static void dump_noc_sanity_status(FILE *f, int noc, const debug_sanitize_noc_addr_t* san) {

    switch (san->invalid) {
    case DebugSanitizeNocInvalidOK:
        if (san->addr != DEBUG_SANITIZE_NOC_SENTINEL_OK_64 ||
            san->len != DEBUG_SANITIZE_NOC_SENTINEL_OK_32 ||
            san->which != DEBUG_SANITIZE_NOC_SENTINEL_OK_16) {
            log_fatal(LogLLRuntime, "Watcher unexpected noc debug state, reported valid got (addr,len,which)=({},{},{})", san->addr, san->len, san->which);
            exit(-1);
        }
        break;
    case DebugSanitizeNocInvalidL1:
        fprintf(f, "noc%d:%s{0x%08lx, %d} ", noc, get_sanity_riscv_name(san->which), san->addr, san->len);
        break;
    case DebugSanitizeNocInvalidUnicast:
        fprintf(f, "noc%d:%s{(%02ld,%02ld) 0x%08lx, %d} ",
                noc,
                get_sanity_riscv_name(san->which),
                NOC_UNICAST_ADDR_X(san->addr),
                NOC_UNICAST_ADDR_Y(san->addr),
                NOC_LOCAL_ADDR_OFFSET(san->addr), san->len);
        break;
    case DebugSanitizeNocInvalidMulticast:
        fprintf(f, "noc%d:%s{(%02ld,%02ld)-(%02ld,%02ld) 0x%08lx, %d} ",
                noc,
                get_sanity_riscv_name(san->which),
                NOC_MCAST_ADDR_START_X(san->addr),
                NOC_MCAST_ADDR_START_Y(san->addr),
                NOC_MCAST_ADDR_END_X(san->addr),
                NOC_MCAST_ADDR_END_Y(san->addr),
                NOC_LOCAL_ADDR_OFFSET(san->addr), san->len);
        break;
    default:
        log_fatal(LogLLRuntime, "Watcher unexpected noc debug state: {}\n", san->invalid);
        exit(-1);
    }
}

static void dump_noc_sanity_status(FILE *f, WatcherDevice *wdev, CoreCoord core) {

    // Read L1 address 0, looking for memory corruption
    std::vector<uint32_t> data;
    data = read_hex_vec_from_core(wdev->cluster_, wdev->device_id_, core, MEM_DEBUG_SANITIZE_NOC_MAILBOX_ADDRESS, N_NOCS * sizeof(debug_sanitize_noc_addr_t));
    debug_sanitize_noc_addr_t *san = reinterpret_cast<debug_sanitize_noc_addr_t *>(&data[0]);

    for (uint32_t noc = 0; noc < N_NOCS; noc++) {
        dump_noc_sanity_status(f, noc, &san[noc]);
    }
}

static void dump_run_mailboxes(FILE *f, WatcherDevice *wdev, CoreCoord core) {

    std::vector<uint32_t> data;
    data = read_hex_vec_from_core(wdev->cluster_, wdev->device_id_, core, MEM_RUN_MAILBOX_ADDRESS, sizeof(uint32_t));
    char code = 'U';
    if (data[0] == INIT_VALUE) code = 'I';
    if (data[0] == DONE_VALUE) code = 'D';
    if (code == 'U') {
        fprintf(f, " rmb:U(%d) ", data[0]);
    } else {
        fprintf(f, " rmb:%c ", code);
    }
}

static void dump_debug_status(FILE *f, WatcherDevice *wdev, CoreCoord core) {

    // Currently, debug status is redundant to go status for non-brisc
    // Just print brisc status

    std::vector<uint32_t> data;
    data = read_hex_vec_from_core(wdev->cluster_, wdev->device_id_, core, MEM_DEBUG_STATUS_MAILBOX_START_ADDRESS, MEM_DEBUG_STATUS_MAILBOX_END_ADDRESS - MEM_DEBUG_STATUS_MAILBOX_START_ADDRESS);
    constexpr int num_riscv_per_core = 5;
    constexpr int num_status_bytes_per_riscv = 4;
    static_assert(MEM_DEBUG_STATUS_MAILBOX_END_ADDRESS - MEM_DEBUG_STATUS_MAILBOX_START_ADDRESS == num_riscv_per_core * num_status_bytes_per_riscv);

    for (int cpu = 0; cpu < num_riscv_per_core; cpu++) {
        for (int byte = 0; byte < num_status_bytes_per_riscv; byte++) {
            char v = ((char *)&data[cpu])[byte];
            if (v == 0) break;
            if (isprint(v)) {
                fprintf(f, "%c", v);
            } else {
                fprintf(f, "U(%d)", v);
            }
        }
        if (cpu != num_riscv_per_core - 1) fprintf(f, ",");
    }

    fprintf(f, " ");
}

static void dump_cb_state(FILE *f, WatcherDevice *wdev, CoreCoord core) {

    // Read back all of the stream state, most of it is unused
    std::vector<uint32_t> data;
    for (uint32_t operand = 0; operand < NUM_CIRCULAR_BUFFERS; operand++) {
        // XXXX TODO(PGK) get this from device
        const uint32_t OPERAND_START_STREAM = 8;
        uint32_t base = NOC_OVERLAY_START_ADDR + (OPERAND_START_STREAM + operand) * NOC_STREAM_REG_SPACE_SIZE;

        uint32_t rcvd_addr = base + STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX * sizeof(uint32_t);
        data = read_hex_vec_from_core(wdev->cluster_, wdev->device_id_, core, rcvd_addr, sizeof(uint32_t));
        uint32_t rcvd = data[0];

        uint32_t ackd_addr = base + STREAM_REMOTE_DEST_BUF_START_REG_INDEX * sizeof(uint32_t);
        data = read_hex_vec_from_core(wdev->cluster_, wdev->device_id_, core, ackd_addr, sizeof(uint32_t));
        uint32_t ackd = data[0];

        if (rcvd != ackd) {
            fprintf(f, "cb[%d](rcv %d!=ack %d) ", operand, rcvd, ackd);
        }
    }
}

static void dump_core(FILE *f, WatcherDevice *wdev, CoreCoord core) {

    // Core (x, y): L1[0]=ok  R:RRRR
    fprintf(f, "Core %s: \t", core.str().c_str());

    if (watcher::enabled) {
        // Dump state only gathered if device is compiled w/ watcher
        dump_debug_status(f, wdev, core);
        dump_l1_status(f, wdev, core);
        dump_noc_sanity_status(f, wdev, core);
    }

    // Dump state always available
    dump_run_mailboxes(f, wdev, core);
    dump_cb_state(f, wdev, core);

    fprintf(f, "\n");

    fflush(f);
}

static void  __attribute__((noinline)) dump(FILE *f) {
    for (auto const& dev_pair : devices) {
        std::shared_ptr<WatcherDevice>wdev = dev_pair.second;

        if (f != stdout && f != stderr) {
            log_info(LogLLRuntime, "Watcher checking device {}", wdev->device_id_);
        }

        CoreCoord grid_size = wdev->get_grid_size_();
        for (uint32_t y = 0; y < grid_size.y; y++) {
            for (uint32_t x = 0; x < grid_size.x; x++) {
                CoreCoord logical_core(x, y);
                CoreCoord worker_core = wdev->worker_from_logical_(logical_core);
                if (worker_core.y != 11 || worker_core.x == 1) {
                    dump_core(f, wdev.get(), worker_core);
                }
            }
        }
    }
}

static void watcher_loop(int sleep_usecs) {
    int count = 0;

    log_info(LogLLRuntime, "Watcher thread watching...");

    while (true) {
        // Odds are this thread will be killed during the usleep
        usleep(sleep_usecs);
        count++;

        {
            const std::lock_guard<std::mutex> lock(watch_mutex);

            fprintf(logfile, "-----\n");
            fprintf(logfile, "Dump #%d at %ds\n", count, watcher::get_elapsed_secs());

            if (devices.size() == 0) {
                fprintf(logfile, "No active devices\n");
            }

            dump(logfile);

            fprintf(logfile, "\n");
        }
    }
}

} // namespace watcher

static void init_device(tt_cluster *cluster,
                        int device_id,
                        std::function<CoreCoord ()>get_grid_size,
                        std::function<CoreCoord (CoreCoord)>worker_from_logical) {

    // Initialize debug status values to "unknown"
    std::vector<uint32_t> debug_status_init_val = { 'X', 'X', 'X', 'X', 'X' };

    // Initialize debug sanity L1/NOC addresses to sentinel "all ok"
    std::vector<uint32_t> debug_sanity_init_val;
    debug_sanity_init_val.resize(watcher::N_NOCS * sizeof(debug_sanitize_noc_addr_t) / sizeof(uint32_t));
    static_assert(sizeof(debug_sanitize_noc_addr_t) % sizeof(uint32_t) == 0);
    debug_sanitize_noc_addr_t *data = reinterpret_cast<debug_sanitize_noc_addr_t *>(&(debug_sanity_init_val[0]));
    for (int i = 0; i < watcher::N_NOCS; i++) {
        data[i].addr = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_64;
        data[i].len = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_32;
        data[i].which = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_16;;
        data[i].invalid = DebugSanitizeNocInvalidOK;
    }

    CoreCoord grid_size = get_grid_size();
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core = worker_from_logical(logical_core);
            tt::llrt::write_hex_vec_to_core(cluster, device_id, worker_core, debug_status_init_val, MEM_DEBUG_STATUS_MAILBOX_START_ADDRESS);
            tt::llrt::write_hex_vec_to_core(cluster, device_id, worker_core, debug_sanity_init_val, MEM_DEBUG_SANITIZE_NOC_MAILBOX_ADDRESS);
        }
    }
}

static bool coord_found_p(vector<CoreCoord>coords, CoreCoord core) {
    for (CoreCoord item : coords) {
        if (item == core) return true;
    }
    return false;
}

static bool coord_found_p(CoreCoord range, CoreCoord core) {
    return
        core.x >= 1 && core.x <= range.x &&
        core.y >= 1 && core.y <= range.y;
}

static string noc_address(CoreCoord core, uint64_t a, uint32_t l) {
    std::stringstream ss;
    ss << "noc{" << core.str() << ", 0x" << std::setfill('0') << std::setw(8) << std::hex << a << ", " << std::dec << l << "}";
    return ss.str();
}

static void print_stack_trace (void) {
    void *array[15];

    int size = backtrace (array, 15);
    char **strings = backtrace_symbols(array, size);
    if (strings != NULL) {
        fprintf(stderr, "Obtained %d stack frames.\n", size);
        for (int i = 0; i < size; i++) fprintf(stderr, "%s\n", strings[i]);
    }

    free (strings);
}

static void watcher_sanitize_host_noc(const char* what,
                                      metal_SocDescriptor soc_d,
                                      CoreCoord core,
                                      uint64_t addr,
                                      uint32_t lbytes) {

    if (coord_found_p(soc_d.get_pcie_cores(), core)) {
        log_fatal(LogLLRuntime, "Host watcher: bad {} NOC coord {}", what, core.str());
    } else if (coord_found_p(soc_d.get_dram_cores(), core)) {
        if (!DEBUG_VALID_DRAM_ADDR(addr, lbytes)) {
            print_stack_trace();
            log_fatal(LogLLRuntime, "Host watcher: bad {} dram address {}", what, noc_address(core, addr, lbytes));
        }
    } else if (coord_found_p(soc_d.get_ethernet_cores(), core)) {
        if (!DEBUG_VALID_ETH_ADDR(addr, lbytes)) {
            print_stack_trace();
            log_fatal(LogLLRuntime, "Host watcher: bad {} eth address {}", what, noc_address(core, addr, lbytes));
        }
    } else if (coord_found_p(soc_d.grid_size, core)) {
        if (!DEBUG_VALID_WORKER_ADDR(addr, lbytes)) {
            print_stack_trace();
            log_fatal(LogLLRuntime, "Host watcher: bad {} worker address {}", what, noc_address(core, addr, lbytes));
        }
    } else {
        // Bad COORD
        print_stack_trace();
        log_fatal(LogLLRuntime, "Host watcher: bad {} NOC coord {}", core.str());
    }
}

void watcher_attach(void *dev,
                    tt_cluster *cluster,
                    int device_id,
                    const std::function<CoreCoord ()>& get_grid_size,
                    const std::function<CoreCoord (CoreCoord)>& worker_from_logical,
                    const string& log_path) {

    const std::lock_guard<std::mutex> lock(watcher::watch_mutex);

    if (!watcher::enabled && OptionsG.get_watcher_enabled()) {

        watcher::logfile = watcher::create_file(log_path);

        int sleep_usecs = OptionsG.get_watcher_interval() * 1000 * 1000;
        std::thread watcher_thread = std::thread(&watcher::watcher_loop, sleep_usecs);
        watcher_thread.detach();

        watcher::enabled = true;
    }

    if (llrt::watcher::logfile != nullptr) {
        fprintf(watcher::logfile, "At %ds attach device %d\n", watcher::get_elapsed_secs(), device_id);
    }

    if (watcher::enabled) {
        init_device(cluster, device_id, get_grid_size, worker_from_logical);
    }

    // Always register the device w/ watcher, even if disabled
    // This allows dump() to be called from debugger
    std::shared_ptr<watcher::WatcherDevice> wdev(new watcher::WatcherDevice(cluster, device_id, get_grid_size, worker_from_logical));
    watcher::devices.insert(pair<void *, std::shared_ptr<watcher::WatcherDevice>>(dev, wdev));
}

void watcher_detach(void *old) {

    const std::lock_guard<std::mutex> lock(watcher::watch_mutex);

    auto pair = watcher::devices.find(old);
    TT_ASSERT(pair != watcher::devices.end());
    if (watcher::logfile != nullptr) {
        fprintf(watcher::logfile, "At %ds detach device %d\n", watcher::get_elapsed_secs(), pair->second->device_id_);
    }
    watcher::devices.erase(old);
}

void watcher_sanitize_host_noc_read(metal_SocDescriptor soc_d, CoreCoord core, uint64_t addr, uint32_t lbytes) {
    watcher_sanitize_host_noc("read", soc_d, core, addr, lbytes);
}

void watcher_sanitize_host_noc_write(metal_SocDescriptor soc_d, CoreCoord core, uint64_t addr, uint32_t lbytes) {
    watcher_sanitize_host_noc("write", soc_d, core, addr, lbytes);
}

} // namespace llrt
} // namespace tt
