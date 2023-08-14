#include <thread>
#include <unistd.h>
#include <chrono>
#include <ctime>

#include "llrt.hpp"
#include "watcher.hpp"
#include "dev_mem_map.h"

namespace tt {
namespace llrt {
namespace watcher {

constexpr int default_sleep_secs = 2 * 60;

static bool enabled = false;
static std::mutex watch_mutex;
static vector<tt_metal::Device *> devices;
static FILE *f = nullptr;
static std::chrono::time_point start_time = std::chrono::system_clock::now();

static uint32_t get_elapsed_secs() {
    std::chrono::time_point now_time = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_secs = now_time - start_time;

    return (uint32_t)elapsed_secs.count();
}

static FILE * create_file() {

    const char *fmode = getenv("TT_METAL_WATCHER_APPEND") ? "a" : "w";
    if ((f = fopen("/tmp/metal_watcher.txt", fmode)) == nullptr) {
        log_fatal(LogLLRuntime, "Watcher failed to create log file\n");
        exit(1);
    }
    fprintf(f, "At %ds starting\n", watcher::get_elapsed_secs());
    fprintf(f, "Legend:\n");
    fprintf(f, "\tBRISC,NCRISC,TRISC0,TRISC1,TRISC2\n");
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
    fprintf(f, "\n");

    return f;
}

static void print_l1_status(FILE *f, tt_metal::Device *dev, CoreCoord core) {

    // Read L1 address 0, looking for memory corruption
    std::vector<uint32_t> data;
    data = read_hex_vec_from_core(dev->cluster(), dev->pcie_slot(), core, 0, sizeof(uint32_t));
    // XXXX TODO(pgk): get this const from llrt (jump to fw insn)
    if (data[0] == 0x2010006f) {
        fprintf(f, "L1[0]=ok ");
    } else {
        fprintf(f, "L1[0]=bad 0x%08x ", data[0]);
    }
}

static void print_debug_status(FILE *f, tt_metal::Device *dev, CoreCoord core) {

    // Currently, debug status is redundant to go status for non-brisc
    // Just print brisc status

    std::vector<uint32_t> data;
    data = read_hex_vec_from_core(dev->cluster(), dev->pcie_slot(), core, MEM_DEBUG_STATUS_MAILBOX_START_ADDRESS, MEM_DEBUG_STATUS_MAILBOX_END_ADDRESS - MEM_DEBUG_STATUS_MAILBOX_START_ADDRESS);
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
}

static void process_core(FILE *f, tt_metal::Device *dev, CoreCoord core) {

    // Core (x, y): L1[0]=ok  R:RRRR
    fprintf(f, "Core %s: \t", core.str().c_str());

    print_l1_status(f, dev, core);
    print_debug_status(f, dev, core);

    fprintf(f, "\n");

    fflush(f);
}

static void watcher_loop(int sleep_usecs) {
    int count = 0;

    log_debug(LogLLRuntime, "Watcher thread watching...");

    while (true) {
        // Odds are this thread will be killed during the usleep
        usleep(sleep_usecs);
        count++;

        {
            const std::lock_guard<std::mutex> lock(watch_mutex);

            fprintf(f, "-----\n");
            fprintf(f, "Dump #%d at %ds\n", count, watcher::get_elapsed_secs());

            if (devices.size() == 0) {
                fprintf(f, "No active devices\n");
            }
            for (auto const& dev : devices) {
                log_debug(LogLLRuntime, "Watcher checking device {}", dev->pcie_slot());

                CoreCoord grid_size = dev->logical_grid_size();
                for (uint32_t y = 0; y < grid_size.y; y++) {
                    for (uint32_t x = 0; x < grid_size.x; x++) {
                        CoreCoord logical_core(x, y);
                        CoreCoord worker_core = dev->worker_core_from_logical_core(logical_core);
                        if (worker_core.y != 11 || worker_core.x == 1) {
                            process_core(f, dev, worker_core);
                        }
                    }
                }
            }

            fprintf(f, "\n");
        }
    }
}

} // namespace watcher

void watcher_attach(tt_metal::Device *dev) {
    const std::lock_guard<std::mutex> lock(watcher::watch_mutex);

    const char *enable_str = getenv("TT_METAL_WATCHER");
    if (!watcher::enabled && enable_str != nullptr) {
        watcher::f = watcher::create_file();

        int sleep_secs = 0;
        sscanf(enable_str, "%d", &sleep_secs);
        int sleep_usecs = ((sleep_secs == 0) ? watcher::default_sleep_secs : sleep_secs) * 1000 * 1000;

        std::thread watcher = std::thread(&watcher::watcher_loop, sleep_usecs);
        watcher.detach();

        watcher::enabled = true;
        log_debug(LogLLRuntime, "Watcher thread spawned");
    }

    if (watcher::enabled) {
        if (llrt::watcher::f != nullptr) {
            // Hmmm, do we want to always open the file so we always get this?
            fprintf(watcher::f, "At %ds attach device %d\n", watcher::get_elapsed_secs(), dev->pcie_slot());
        }
        watcher::devices.push_back(dev);
    }
}

void watcher_detach(tt_metal::Device *old) {
    const std::lock_guard<std::mutex> lock(watcher::watch_mutex);

    if (watcher::enabled) {
        fprintf(llrt::watcher::f, "At %ds detach device %d\n", watcher::get_elapsed_secs(), old->pcie_slot());

        for (vector<tt_metal::Device *>::iterator iter = watcher::devices.begin();
             iter < watcher::devices.end();
             iter++) {

            if (*iter == old) {
                watcher::devices.erase(iter, iter + 1);
            }
        }
    }
}

void watcher_init(tt_metal::Device *dev) {
    if (getenv("TT_METAL_WATCHER_DOPE_L1") != nullptr) {
        uint32_t value = atoi(getenv("TT_METAL_WATCHER_DOPE_L1"));

        log_debug(LogLLRuntime, "Watcher doping L1 with {}", value);

        std::vector<uint32_t> dope;
        dope.resize(MEM_L1_SIZE / sizeof(uint32_t));
        for (unsigned int index = 0; index < dope.size(); index++) {
            dope[index] = value;
        }

        CoreCoord grid_size = dev->logical_grid_size();
        for (uint32_t y = 0; y < grid_size.y; y++) {
            for (uint32_t x = 0; x < grid_size.x; x++) {
                CoreCoord logical_core(x, y);
                CoreCoord worker_core = dev->worker_core_from_logical_core(logical_core);
                tt::llrt::write_hex_vec_to_core(dev->cluster(), dev->pcie_slot(), worker_core, dope, MEM_L1_BASE);
            }
        }
    }

    CoreCoord grid_size = dev->logical_grid_size();
    std::vector<uint32_t> debug_status_init_val = { 'X', 'X', 'X', 'X', 'X' };
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core = dev->worker_core_from_logical_core(logical_core);
            tt::llrt::write_hex_vec_to_core(dev->cluster(), dev->pcie_slot(), worker_core, debug_status_init_val, MEM_DEBUG_STATUS_MAILBOX_START_ADDRESS);
        }
    }
}

} // namespace llrt
} // namespace tt
