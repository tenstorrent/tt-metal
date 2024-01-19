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
#include "dev_msgs.h"

#include "noc/noc_parameters.h"
#include "noc/noc_overlay_parameters.h"

#include "debug/sanitize_noc.h"

namespace tt {
namespace llrt {
namespace watcher {


#define DEBUG_VALID_L1_ADDR(a, l) (((a) >= MEM_L1_BASE) && ((a) + (l) <= MEM_L1_BASE + MEM_L1_SIZE))

// what's the size of the NOC<n> address space?  using 0x1000 for now
#define DEBUG_VALID_REG_ADDR(a)                                                        \
    (                                                                                  \
     (((a) >= NOC_OVERLAY_START_ADDR) && ((a) < NOC_OVERLAY_START_ADDR + NOC_STREAM_REG_SPACE_SIZE * NOC_NUM_STREAMS)) || \
     (((a) >= NOC0_REGS_START_ADDR) && ((a) < NOC0_REGS_START_ADDR + 0x1000)) || \
     (((a) >= NOC1_REGS_START_ADDR) && ((a) < NOC1_REGS_START_ADDR + 0x1000)) || \
     ((a) == RISCV_DEBUG_REG_SOFT_RESET_0))
#define DEBUG_VALID_WORKER_ADDR(a, l) (DEBUG_VALID_L1_ADDR(a, l) || (DEBUG_VALID_REG_ADDR(a) && (l) == 4))
#define DEBUG_VALID_DRAM_ADDR(a, l, b, e) (((a) >= b) && ((a) + (l) <= e))

#define DEBUG_VALID_ETH_ADDR(a, l) (((a) >= MEM_ETH_BASE) && ((a) + (l) < MEM_ETH_BASE + MEM_ETH_SIZE))


class WatcherDevice {
  public:
    int device_id_;
    std::function<CoreCoord ()>get_grid_size_;
    std::function<CoreCoord (CoreCoord)>worker_from_logical_;
    std::function<const std::set<CoreCoord> &()> storage_only_cores_;

    WatcherDevice(int device_id, std::function<CoreCoord ()>get_grid_size, std::function<CoreCoord (CoreCoord)>worker_from_logical, std::function<const std::set<CoreCoord> &()>storage_only_cores);
};

constexpr uint64_t DEBUG_SANITIZE_NOC_SENTINEL_OK_64 = 0xbadabadabadabada;
constexpr uint32_t DEBUG_SANITIZE_NOC_SENTINEL_OK_32 = 0xbadabada;
constexpr uint16_t DEBUG_SANITIZE_NOC_SENTINEL_OK_16 = 0xbada;

static bool enabled = false;
static std::mutex watch_mutex;
static std::unordered_map<void *, std::shared_ptr<WatcherDevice>> devices;
static string logfile_path = "";
static FILE *logfile = nullptr;
static std::chrono::time_point start_time = std::chrono::system_clock::now();
static std::vector<string> kernel_names;

WatcherDevice::WatcherDevice(int device_id, std::function<CoreCoord ()>get_grid_size, std::function<CoreCoord (CoreCoord)>worker_from_logical, std::function<const std::set<CoreCoord> &()> storage_only_cores) : device_id_(device_id), get_grid_size_(get_grid_size), worker_from_logical_(worker_from_logical), storage_only_cores_(storage_only_cores) {
}

static uint32_t get_elapsed_secs() {
    std::chrono::time_point now_time = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_secs = now_time - start_time;

    return (uint32_t)elapsed_secs.count();
}

static FILE * create_file(const string& log_path) {

    FILE *f;

    const char *fmode = OptionsG.get_watcher_append()? "a" : "w";
    string fname = log_path + "watcher.log";
    if ((f = fopen(fname.c_str(), fmode)) == nullptr) {
        TT_THROW("Watcher failed to create log file\n");
    }
    log_info(LogLLRuntime, "Watcher log file: {}", fname);

    fprintf(f, "At %ds starting\n", watcher::get_elapsed_secs());
    fprintf(f, "Legend:\n");
    fprintf(f, "\tComma separated list specifices waypoint for BRISC,NCRISC,TRISC0,TRISC1,TRISC2\n");
    fprintf(f, "\tI=initialization sequence\n");
    fprintf(f, "\tW=wait (top of spin loop)\n");
    fprintf(f, "\tR=run (entering kernel)\n");
    fprintf(f, "\tD=done (finished spin loop)\n");
    fprintf(f, "\tX=host written value prior to fw launch\n");
    fprintf(f, "\n");
    fprintf(f, "\tA single character status is in the FW, other characters clarify where, eg:\n");
    fprintf(f, "\t\tNRW is \"noc read wait\"\n");
    fprintf(f, "\t\tNWD is \"noc write done\"\n");
    fprintf(f, "\tnoc<n>:<risc>{a, l}=an L1 address used by NOC<n> by <riscv> (eg, local src address)\n");
    fprintf(f, "\tnoc<n>:<riscv>{(x,y), a, l}=NOC<n> unicast address used by <riscv>\n");
    fprintf(f, "\tnoc<n>:<riscv>{(x1,y1)-(x2,y2), a, l}=NOC<n> multicast address used by <riscv>\n");
    fprintf(f, "\trmsg:<c>=brisc host run message, D/H device/host dispatch; brisc NOC ID; I/G/D init/go/done; | separator; B/b enable/disable brisc; N/n enable/disable ncrisc; T/t enable/disable TRISC\n");
    fprintf(f, "\tsmsg:<c>=slave run message, I/G/D for NCRISC, TRISC0, TRISC1, TRISC2\n");
    fprintf(f, "\tk_ids:<brisc id>|<ncrisc id>|<trisc id> (ID map to file at end of section)\n");
    fprintf(f, "\n");

    return f;
}

static void log_running_kernels(const launch_msg_t *launch_msg) {
    log_info("While running kernels:");
    log_info(" brisc : {}", kernel_names[launch_msg->brisc_watcher_kernel_id]);
    log_info(" ncrisc: {}", kernel_names[launch_msg->ncrisc_watcher_kernel_id]);
    log_info(" triscs: {}", kernel_names[launch_msg->triscs_watcher_kernel_id]);
}

static void dump_l1_status(FILE *f, WatcherDevice *wdev, CoreCoord core, const launch_msg_t *launch_msg) {

    // Read L1 address 0, looking for memory corruption
    std::vector<uint32_t> data;
    data = read_hex_vec_from_core(wdev->device_id_, core, MEM_L1_BASE, sizeof(uint32_t));
    // XXXX TODO(pgk): get this const from llrt (jump to fw insn)
    if (data[0] != 0x2010006f) {
        log_running_kernels(launch_msg);
        TT_THROW("Watcher found corruption at L1[0] on core {}: read {}", core.str(), data[0]);
    }
}

static const char * get_sanity_riscv_name(CoreCoord core, const launch_msg_t *launch_msg, uint32_t type)
{
    switch (type) {
    case DebugSanitizeBrisc:
        return "brisc";
    case DebugSanitizeNCrisc:
        return "ncrisc";
    case DebugSanitizeErisc:
        return "erisc";
    case DebugSanitizeTrisc0:
        return "trisc0";
    case DebugSanitizeTrisc1:
        return "trisc1";
    case DebugSanitizeTrisc2:
        return "trisc2";
    default:
        log_running_kernels(launch_msg);
        TT_THROW("Watcher data corrupted, unexpected riscv type on core {}: {}", core.str(), type);
    }
    return nullptr;
}

static string get_debug_status(CoreCoord core, const launch_msg_t *launch_msg, const debug_status_msg_t *debug_status) {

    string out;

    for (int cpu = 0; cpu < num_riscv_per_core; cpu++) {
        for (int byte = 0; byte < num_status_bytes_per_riscv; byte++) {
            char v = ((char *)&debug_status[cpu])[byte];
            if (v == 0) break;
            if (isprint(v)) {
                out += v;
            } else {
                log_running_kernels(launch_msg);
                TT_THROW("Watcher data corrupted, unexpected debug status on core {}, unprintable character {}",
                          core.str(), (int)v);
            }
        }
        if (cpu != num_riscv_per_core - 1) out += ',';
    }

    out += " ";
    return out;
}

static void log_waypoint(CoreCoord core, const launch_msg_t *launch_msg, const debug_status_msg_t *debug_status) {
    string out = get_debug_status(core, launch_msg, debug_status);
    out = string("Last waypoint: ") + out;
    log_info(out.c_str());
}

static void dump_noc_sanity_status(FILE *f,
                                   CoreCoord core,
                                   const launch_msg_t *launch_msg,
                                   int noc,
                                   const debug_sanitize_noc_addr_msg_t* san,
                                   const debug_status_msg_t *debug_status) {

    char buf[256];

    switch (san->invalid) {
    case DebugSanitizeNocInvalidOK:
        if (san->addr != DEBUG_SANITIZE_NOC_SENTINEL_OK_64 ||
            san->len != DEBUG_SANITIZE_NOC_SENTINEL_OK_32 ||
            san->which != DEBUG_SANITIZE_NOC_SENTINEL_OK_16) {
            log_running_kernels(launch_msg);
            log_waypoint(core, launch_msg, debug_status);
            snprintf(buf,sizeof(buf),
                     "Watcher unexpected noc debug state on core %s, reported valid got noc%d{0x%08lx, %d}",
                     core.str().c_str(), san->which, san->addr, san->len);
            TT_THROW(buf);
        }
        break;
    case DebugSanitizeNocInvalidL1:
        fprintf(f, "%s using noc%d reading L1[addr=0x%08lx,len=%d]\n", get_sanity_riscv_name(core, launch_msg, san->which), noc, san->addr, san->len);
        fflush(f);
        log_running_kernels(launch_msg);
        log_info("Watcher stopped the device due to bad NOC L1/reg address");
        log_waypoint(core, launch_msg, debug_status);
        snprintf(buf, sizeof(buf), "On core %s: %s using noc%d reading L1[addr=0x%08lx,len=%d]",
                 core.str().c_str(), get_sanity_riscv_name(core, launch_msg, san->which), noc, san->addr, san->len);
        TT_THROW(buf);
        break;
    case DebugSanitizeNocInvalidUnicast:
        fprintf(f, "%s using noc%d tried to access core (%02ld,%02ld) L1[addr=0x%08lx,len=%d]\n",
                get_sanity_riscv_name(core, launch_msg, san->which),
                noc,
                NOC_UNICAST_ADDR_X(san->addr),
                NOC_UNICAST_ADDR_Y(san->addr),
                NOC_LOCAL_ADDR_OFFSET(san->addr), san->len);
        fflush(f);
        log_info("Watcher stopped the device due to bad NOC unicast transaction");
        log_running_kernels(launch_msg);
        log_waypoint(core, launch_msg, debug_status);
        snprintf(buf, sizeof(buf), "On core %s: %s using noc%d tried to accesss core (%02ld,%02ld) L1[addr=0x%08lx,len=%d]",
                 core.str().c_str(),
                 get_sanity_riscv_name(core, launch_msg, san->which),
                 noc,
                 NOC_UNICAST_ADDR_X(san->addr),
                 NOC_UNICAST_ADDR_Y(san->addr),
                 NOC_LOCAL_ADDR_OFFSET(san->addr), san->len);
        TT_THROW(buf);
        break;
    case DebugSanitizeNocInvalidMulticast:
        fprintf(f, "%s using noc%d tried to access core range (%02ld,%02ld)-(%02ld,%02ld) L1[addr=0x%08lx,len=%d]\n",
                get_sanity_riscv_name(core, launch_msg, san->which),
                noc,
                NOC_MCAST_ADDR_START_X(san->addr),
                NOC_MCAST_ADDR_START_Y(san->addr),
                NOC_MCAST_ADDR_END_X(san->addr),
                NOC_MCAST_ADDR_END_Y(san->addr),
                NOC_LOCAL_ADDR_OFFSET(san->addr), san->len);
        fflush(f);
        log_info("Watcher stopped the device due to bad NOC multicast transaction");
        log_running_kernels(launch_msg);
        log_waypoint(core, launch_msg, debug_status);
        snprintf(buf, sizeof(buf), "On core %s: %s using noc%d tried to access core range (%02ld,%02ld)-(%02ld,%02ld) L1[addr=0x%08lx,len=%d]}",
                 core.str().c_str(),
                 get_sanity_riscv_name(core, launch_msg, san->which),
                 noc,
                 NOC_MCAST_ADDR_START_X(san->addr),
                 NOC_MCAST_ADDR_START_Y(san->addr),
                 NOC_MCAST_ADDR_END_X(san->addr),
                 NOC_MCAST_ADDR_END_Y(san->addr),
                 NOC_LOCAL_ADDR_OFFSET(san->addr), san->len);
        TT_THROW(buf);
        break;
    default:
        log_running_kernels(launch_msg);
        TT_THROW("Watcher unexpected data corruption, noc debug state on core {}, unknown failure code: {}\n",
                  core.str(), san->invalid);
    }
}

static void dump_noc_sanity_status(FILE *f,
                                   CoreCoord core,
                                   const launch_msg_t *launch_msg,
                                   const debug_sanitize_noc_addr_msg_t *san,
                                   const debug_status_msg_t *debug_status) {

    for (uint32_t noc = 0; noc < NUM_NOCS; noc++) {
        dump_noc_sanity_status(f, core, launch_msg, noc, &san[noc], debug_status);
    }
}

static void dump_run_state(FILE *f, CoreCoord core, const launch_msg_t *launch_msg, uint32_t state) {
    char code = 'U';
    if (state == RUN_MSG_INIT) code = 'I';
    else if (state == RUN_MSG_GO) code = 'G';
    else if (state == RUN_MSG_DONE) code = 'D';
    if (code == 'U') {
        log_running_kernels(launch_msg);
        TT_THROW("Watcher data corruption, unexpected run state on core{}: {} (expected {} or {} or {})",
                  core.str(), state, RUN_MSG_INIT, RUN_MSG_GO, RUN_MSG_DONE);
    } else {
        fprintf(f, "%c", code);
    }
}

static void dump_run_mailboxes(FILE *f,
                               CoreCoord core,
                               const launch_msg_t *launch_msg,
                               const slave_sync_msg_t *slave_sync) {

    fprintf(f, "rmsg:");

    if (launch_msg->mode == DISPATCH_MODE_DEV) {
        fprintf(f, "D");
    } else if (launch_msg->mode == DISPATCH_MODE_HOST) {
        fprintf(f, "H");
    } else {
        log_running_kernels(launch_msg);
        TT_THROW("Watcher data corruption, unexpected launch mode on core {}: {} (expected {} or {})",
                  core.str(), launch_msg->mode, DISPATCH_MODE_DEV, DISPATCH_MODE_HOST);
    }

    if (launch_msg->brisc_noc_id == 0 || launch_msg->brisc_noc_id == 1) {
        fprintf(f, "%d", launch_msg->brisc_noc_id);
    } else {
        log_running_kernels(launch_msg);
        TT_THROW("Watcher data corruption, unexpected brisc noc_id on core {}: {} (expected 0 or 1)",
                  core.str(), launch_msg->brisc_noc_id);
    }

    dump_run_state(f, core, launch_msg, launch_msg->run);

    fprintf(f, "|");

    if (launch_msg->enable_brisc == 1) {
        fprintf(f, "B");
    } else if (launch_msg->enable_brisc == 0) {
        fprintf(f, "b");
    } else {
        log_running_kernels(launch_msg);
        TT_THROW("Watcher data corruption, unexpected brisc enable on core {}: {} (expected 0 or 1)",
                  core.str(),
                  launch_msg->enable_brisc);
    }

    if (launch_msg->enable_ncrisc == 1) {
        fprintf(f, "N");
    } else if (launch_msg->enable_ncrisc == 0) {
        fprintf(f, "n");
    } else {
        log_running_kernels(launch_msg);
        TT_THROW("Watcher data corruption, unexpected ncrisc enable on core {}: {} (expected 0 or 1)",
                  core.str(),
                  launch_msg->enable_ncrisc);
    }

    if (launch_msg->enable_triscs == 1) {
        fprintf(f, "T");
    } else if (launch_msg->enable_triscs == 0) {
        fprintf(f, "t");
    } else {
        log_running_kernels(launch_msg);
        TT_THROW("Watcher data corruption, unexpected trisc enable on core {}: {} (expected 0 or 1)",
                  core.str(),
                  launch_msg->enable_triscs);
    }

    fprintf(f, " ");

    fprintf(f, "smsg:");
    dump_run_state(f, core, launch_msg, slave_sync->ncrisc);
    dump_run_state(f, core, launch_msg, slave_sync->trisc0);
    dump_run_state(f, core, launch_msg, slave_sync->trisc1);
    dump_run_state(f, core, launch_msg, slave_sync->trisc2);

    fprintf(f, " ");
}

static void dump_debug_status(FILE *f, CoreCoord core, const launch_msg_t *launch_msg, const debug_status_msg_t *debug_status) {

    string out = get_debug_status(core, launch_msg, debug_status);
    fprintf(f, "%s ", out.c_str());
}

static void dump_sync_regs(FILE *f, WatcherDevice *wdev, CoreCoord core) {

    // Read back all of the stream state, most of it is unused
    std::vector<uint32_t> data;
    for (uint32_t operand = 0; operand < NUM_CIRCULAR_BUFFERS; operand++) {
        // XXXX TODO(PGK) get this from device
        const uint32_t OPERAND_START_STREAM = 8;
        uint32_t base = NOC_OVERLAY_START_ADDR + (OPERAND_START_STREAM + operand) * NOC_STREAM_REG_SPACE_SIZE;

        uint32_t rcvd_addr = base + STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX * sizeof(uint32_t);
        data = read_hex_vec_from_core(wdev->device_id_, core, rcvd_addr, sizeof(uint32_t));
        uint32_t rcvd = data[0];

        uint32_t ackd_addr = base + STREAM_REMOTE_DEST_BUF_START_REG_INDEX * sizeof(uint32_t);
        data = read_hex_vec_from_core(wdev->device_id_, core, ackd_addr, sizeof(uint32_t));
        uint32_t ackd = data[0];

        if (rcvd != ackd) {
            fprintf(f, "cb[%d](rcv %d!=ack %d) ", operand, rcvd, ackd);
        }
    }
}

static void validate_kernel_ids(FILE *f,
                                std::map<int, bool>& used_kernel_names,
                                CoreCoord core,
                                const launch_msg_t *launch) {

    if (launch->brisc_watcher_kernel_id >= kernel_names.size()) {
        TT_THROW("Watcher data corruption, unexpected brisc kernel id on core {}: {} (last valid {})",
                  core.str(), launch->brisc_watcher_kernel_id, kernel_names.size());
    }
    used_kernel_names[launch->brisc_watcher_kernel_id] = true;

    if (launch->ncrisc_watcher_kernel_id >= kernel_names.size()) {
        TT_THROW("Watcher data corruption, unexpected ncrisc kernel id on core {}: {} (last valid {})",
                  core.str(), launch->ncrisc_watcher_kernel_id, kernel_names.size());
    }
    used_kernel_names[launch->ncrisc_watcher_kernel_id] = true;

    if (launch->triscs_watcher_kernel_id >= kernel_names.size()) {
        TT_THROW("Watcher data corruption, unexpected trisc kernel id on core {}: {} (last valid {})",
                  core.str(), launch->triscs_watcher_kernel_id, kernel_names.size());
    }
    used_kernel_names[launch->triscs_watcher_kernel_id] = true;
}

static void dump_core(FILE *f, std::map<int, bool>& used_kernel_names, WatcherDevice *wdev, CoreCoord core, bool dump_all) {

    string pad(11 - core.str().length(), ' ');
    fprintf(f, "Device %i, ", wdev->device_id_);
    fprintf(f, "Core %s:%s  ", core.str().c_str(), pad.c_str());

    std::vector<uint32_t> data;
    data = read_hex_vec_from_core(wdev->device_id_, core, MEM_MAILBOX_BASE, sizeof(mailboxes_t));
    mailboxes_t *mbox_data = (mailboxes_t *)(&data[0]);

    // Validate these first since they are used in diagnostic messages below
    validate_kernel_ids(f, used_kernel_names, core, &mbox_data->launch);

    if (watcher::enabled) {
        // Dump state only gathered if device is compiled w/ watcher
        dump_debug_status(f, core, &mbox_data->launch, mbox_data->debug_status);
        dump_l1_status(f, wdev, core,  &mbox_data->launch);
        dump_noc_sanity_status(f, core, &mbox_data->launch, mbox_data->sanitize_noc, mbox_data->debug_status);
    }

    // Dump state always available
    dump_run_mailboxes(f, core, &mbox_data->launch, &mbox_data->slave_sync);
    if (dump_all || OptionsG.get_watcher_dump_all()) {
        // Reading registers while running can cause hangs, only read if
        // requested explicitly
        dump_sync_regs(f, wdev, core);
    }

    fprintf(f, "k_ids:%d|%d|%d",
            mbox_data->launch.brisc_watcher_kernel_id,
            mbox_data->launch.ncrisc_watcher_kernel_id,
            mbox_data->launch.triscs_watcher_kernel_id);

    fprintf(f, "\n");

    fflush(f);
}

// noinline so that this fn exists to be called from dgb
static void  __attribute__((noinline)) dump(FILE *f, bool dump_all) {
    for (auto const& dev_pair : devices) {
        std::shared_ptr<WatcherDevice>wdev = dev_pair.second;

        if (f != stdout && f != stderr) {
            log_info(LogLLRuntime, "Watcher checking device {}", wdev->device_id_);
        }

        std::map<int, bool> used_kernel_names;
        CoreCoord grid_size = wdev->get_grid_size_();
        for (uint32_t y = 0; y < grid_size.y; y++) {
            for (uint32_t x = 0; x < grid_size.x; x++) {
                CoreCoord logical_core(x, y);
                CoreCoord worker_core = wdev->worker_from_logical_(logical_core);
                if (wdev->storage_only_cores_().find(logical_core) == wdev->storage_only_cores_().end()) {
                    dump_core(f, used_kernel_names, wdev.get(), worker_core, dump_all);
                }
            }
        }

        for (auto k_id : used_kernel_names) {
            fprintf(f, "k_id[%d]: %s\n", k_id.first, kernel_names[k_id.first].c_str());
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

            dump(logfile, false);

            fprintf(logfile, "\n");
        }
    }
}

} // namespace watcher

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
                                      const metal_SocDescriptor& soc_d,
                                      CoreCoord core,
                                      uint64_t addr,
                                      uint32_t lbytes) {

    if (coord_found_p(soc_d.get_pcie_cores(), core)) {
        TT_THROW("Host watcher: bad {} NOC coord {}", what, core.str());
    } else if (coord_found_p(soc_d.get_dram_cores(), core)) {
        uint64_t dram_addr_base = 0;
        uint64_t dram_addr_size = soc_d.dram_core_size;
        uint64_t dram_addr_end = dram_addr_size - dram_addr_base;
        if (!DEBUG_VALID_DRAM_ADDR(addr, lbytes, dram_addr_base, dram_addr_end)) {
            print_stack_trace();
            TT_THROW("Host watcher: bad {} dram address {}", what, noc_address(core, addr, lbytes));
        }
    } else if (coord_found_p(soc_d.get_physical_ethernet_cores(), core)) {
        if (!DEBUG_VALID_ETH_ADDR(addr, lbytes)) {
            print_stack_trace();
            TT_THROW("Host watcher: bad {} eth address {}", what, noc_address(core, addr, lbytes));
        }
    } else if (coord_found_p(soc_d.grid_size, core)) {
        if (!DEBUG_VALID_WORKER_ADDR(addr, lbytes)) {
            print_stack_trace();
            TT_THROW("Host watcher: bad {} worker address {}", what, noc_address(core, addr, lbytes));
        }
    } else {
        // Bad COORD
        print_stack_trace();
        TT_THROW("Host watcher: bad {} NOC coord {}", core.str());
    }
}

void watcher_init(int device_id,
                  std::function<CoreCoord ()>get_grid_size,
                  std::function<CoreCoord (CoreCoord)>worker_from_logical) {

    // Initialize debug status values to "unknown"
    std::vector<uint32_t> debug_status_init_val = { 'X', 'X', 'X', 'X', 'X' };

    // Initialize debug sanity L1/NOC addresses to sentinel "all ok"
    std::vector<uint32_t> debug_sanity_init_val;
    debug_sanity_init_val.resize(NUM_NOCS * sizeof(debug_sanitize_noc_addr_msg_t) / sizeof(uint32_t));
    static_assert(sizeof(debug_sanitize_noc_addr_msg_t) % sizeof(uint32_t) == 0);
    debug_sanitize_noc_addr_msg_t *data = reinterpret_cast<debug_sanitize_noc_addr_msg_t *>(&(debug_sanity_init_val[0]));
    for (int i = 0; i < NUM_NOCS; i++) {
        data[i].addr = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_64;
        data[i].len = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_32;
        data[i].which = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_16;
        data[i].invalid = DebugSanitizeNocInvalidOK;
    }

    CoreCoord grid_size = get_grid_size();
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core = worker_from_logical(logical_core);
            tt::llrt::write_hex_vec_to_core(device_id, worker_core, debug_status_init_val, GET_MAILBOX_ADDRESS_HOST(debug_status));
            tt::llrt::write_hex_vec_to_core(device_id, worker_core, debug_sanity_init_val, GET_MAILBOX_ADDRESS_HOST(sanitize_noc));
        }
    }

    log_debug(LogLLRuntime, "Watcher initialized device {}", device_id);
}

void watcher_attach(void *dev,
                    int device_id,
                    const std::function<CoreCoord ()>& get_grid_size,
                    const std::function<CoreCoord (CoreCoord)>& worker_from_logical,
                    const std::function<const std::set<CoreCoord> &()>& storage_only_cores,
                    const string& log_path) {

    const std::lock_guard<std::mutex> lock(watcher::watch_mutex);

    if (!watcher::enabled && OptionsG.get_watcher_enabled()) {

        watcher::logfile_path = log_path;
        watcher::logfile = watcher::create_file(log_path);

        int sleep_usecs = OptionsG.get_watcher_interval() * 1000;
        std::thread watcher_thread = std::thread(&watcher::watcher_loop, sleep_usecs);
        watcher_thread.detach();

        watcher::kernel_names.push_back("blank");

        watcher::enabled = true;
    }

    if (llrt::watcher::logfile != nullptr) {
        fprintf(watcher::logfile, "At %ds attach device %d\n", watcher::get_elapsed_secs(), device_id);
    }

    if (watcher::enabled) {
        log_info(LogLLRuntime, "Watcher attached device {}", device_id);
    }

    // Always register the device w/ watcher, even if disabled
    // This allows dump() to be called from debugger
    std::shared_ptr<watcher::WatcherDevice> wdev(new watcher::WatcherDevice(device_id, get_grid_size, worker_from_logical, storage_only_cores));
    watcher::devices.insert(pair<void *, std::shared_ptr<watcher::WatcherDevice>>(dev, wdev));
}

void watcher_detach(void *old) {

    const std::lock_guard<std::mutex> lock(watcher::watch_mutex);

    auto pair = watcher::devices.find(old);
    TT_ASSERT(pair != watcher::devices.end());
    if (watcher::logfile != nullptr) {
        log_info(LogLLRuntime, "Watcher detached device {}", pair->second->device_id_);
        fprintf(watcher::logfile, "At %ds detach device %d\n", watcher::get_elapsed_secs(), pair->second->device_id_);
    }
    watcher::devices.erase(old);
}

void watcher_sanitize_host_noc_read(const metal_SocDescriptor& soc_d, CoreCoord core, uint64_t addr, uint32_t lbytes) {
    watcher_sanitize_host_noc("read", soc_d, core, addr, lbytes);
}

void watcher_sanitize_host_noc_write(const metal_SocDescriptor& soc_d, CoreCoord core, uint64_t addr, uint32_t lbytes) {
    watcher_sanitize_host_noc("write", soc_d, core, addr, lbytes);
}

int watcher_register_kernel(const string& name) {
    const std::lock_guard<std::mutex> lock(watcher::watch_mutex);

    watcher::kernel_names.push_back(name);

    return watcher::kernel_names.size() - 1;
}

void watcher_clear_log() {
    watcher::logfile = watcher::create_file(watcher::logfile_path);
}

} // namespace llrt
} // namespace tt
