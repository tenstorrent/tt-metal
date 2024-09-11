// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "watcher_server.hpp"

#include <unistd.h>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "llrt/hal.hpp"
#include "dev_msgs.h"
#include "llrt/llrt.hpp"
#include "llrt/rtoptions.hpp"
#include "noc/noc_overlay_parameters.h"
#include "noc/noc_parameters.h"
#include "debug/ring_buffer.h"
#include "watcher_device_reader.hpp"

namespace tt {
namespace watcher {

#define GET_WATCHER_TENSIX_DEV_ADDR()                                   \
    hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalMemAddrType::WATCHER)

#define GET_WATCHER_ERISC_DEV_ADDR()                                    \
    hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalMemAddrType::WATCHER)

#define GET_WATCHER_IERISC_DEV_ADDR()                                   \
    hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalMemAddrType::WATCHER)

static std::atomic<bool> enabled = false;
static std::atomic<bool> server_running = false;
static std::atomic<int> dump_count = 0;
static std::mutex watch_mutex;
static std::map<Device *, watcher::WatcherDeviceReader> devices;
static string logfile_path = "generated/watcher/";
static string logfile_name = "watcher.log";
static FILE *logfile = nullptr;
static std::chrono::time_point start_time = std::chrono::system_clock::now();
static std::vector<string> kernel_names;
static FILE *kernel_file = nullptr;
static string kernel_file_name = "kernel_names.txt";

// Flag to signal whether the watcher server has been killed due to a thrown exception.
static std::atomic<bool> watcher_killed_due_to_error = false;

static std::mutex watcher_exception_message_mutex;

// Function to get the static string
static std::string& watcher_exception_message() {
    static std::string message = "";
    return message;
}

// Function to set the static string
static void set_watcher_exception_message(const std::string& message) {
    std::lock_guard<std::mutex> lock(watcher_exception_message_mutex);
    watcher_exception_message() = message;
}

static double get_elapsed_secs() {
    std::chrono::time_point now_time = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_secs = now_time - start_time;

    return elapsed_secs.count();
}

void create_log_file() {
    FILE *f;

    const char *fmode = tt::llrt::OptionsG.get_watcher_append() ? "a" : "w";
    std::filesystem::path output_dir(tt::llrt::OptionsG.get_root_dir() + watcher::logfile_path);
    std::filesystem::create_directories(output_dir);
    string fname = output_dir.string() + watcher::logfile_name;
    if ((f = fopen(fname.c_str(), fmode)) == nullptr) {
        TT_THROW("Watcher failed to create log file\n");
    }
    log_info(LogLLRuntime, "Watcher log file: {}", fname);

    fprintf(f, "At %.3lfs starting\n", watcher::get_elapsed_secs());
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
    fprintf(
        f,
        "\trmsg:<c>=brisc host run message, D/H device/host dispatch; brisc NOC ID; I/G/D init/go/done; | separator; "
        "B/b enable/disable brisc; N/n enable/disable ncrisc; T/t enable/disable TRISC\n");
    fprintf(f, "\tsmsg:<c>=slave run message, I/G/D for NCRISC, TRISC0, TRISC1, TRISC2\n");
    fprintf(f, "\tk_ids:<brisc id>|<ncrisc id>|<trisc id> (ID map to file at end of section)\n");
    fprintf(f, "\n");
    fflush(f);

    watcher::logfile = f;
}

void create_kernel_file() {
    FILE *f;
    const char *fmode = tt::llrt::OptionsG.get_watcher_append() ? "a" : "w";
    std::filesystem::path output_dir(tt::llrt::OptionsG.get_root_dir() + watcher::logfile_path);
    std::filesystem::create_directories(output_dir);
    string fname = output_dir.string() + watcher::kernel_file_name;
    if ((f = fopen(fname.c_str(), fmode)) == nullptr) {
        TT_THROW("Watcher failed to create kernel name file\n");
    }
    watcher::kernel_names.clear();
    watcher::kernel_names.push_back("blank");
    fprintf(f, "0: blank\n");
    fflush(f);

    watcher::kernel_file = f;
}

// noinline so that this fn exists to be called from dgb
static void __attribute__((noinline)) dump(FILE *f) {
    for (auto &device_and_reader : devices) {
        device_and_reader.second.Dump(f);
    }
}

static void watcher_loop(int sleep_usecs) {
    TT_ASSERT(watcher::server_running == false);
    watcher::server_running = true;
    watcher::dump_count = 1;

    // Print to the user which features are disabled via env vars.
    string disabled_features = "";
    auto &disabled_features_set = tt::llrt::OptionsG.get_watcher_disabled_features();
    if (!disabled_features_set.empty()) {
        for (auto &feature : disabled_features_set) {
            disabled_features += feature + ",";
        }
        disabled_features.pop_back();
    } else {
        disabled_features = "None";
    }
    log_info(LogLLRuntime, "Watcher server initialized, disabled features: {}", disabled_features);

    while (true) {
        // Delay the amount of time specified by the user. Don't include watcher polling time to avoid the case where
        // watcher dominates the communication links due to heavy traffic.
        double last_elapsed_time = watcher::get_elapsed_secs();
        while ((watcher::get_elapsed_secs() - last_elapsed_time) < ((double) sleep_usecs) / 1000000.) {
            // Odds are this thread will be killed during the usleep, the kill signal is
            // watcher::enabled = false from the main thread.
            if (!watcher::enabled)
                break;
            usleep(1);
        }

        {
            const std::lock_guard<std::mutex> lock(watch_mutex);

            // If all devices are detached, we can turn off the server, it will be turned back on
            // when a new device is attached.
            if (!watcher::enabled)
                break;

            fprintf(logfile, "-----\n");
            fprintf(logfile, "Dump #%d at %.3lfs\n", watcher::dump_count.load(), watcher::get_elapsed_secs());

            if (devices.size() == 0) {
                fprintf(logfile, "No active devices\n");
            }

            try {
                dump(logfile);
            } catch (std::runtime_error &e) {
                // Depending on whether test mode is enabled, catch and stop server, or re-throw.
                if (tt::llrt::OptionsG.get_test_mode_enabled()) {
                    watcher::watcher_killed_due_to_error = true;
                    watcher::enabled = false;
                    break;
                } else {
                    throw e;
                }
            }

            fprintf(logfile, "Dump #%d completed at %.3lfs\n", watcher::dump_count.load(), watcher::get_elapsed_secs());
        }
        fflush(logfile);
        watcher::dump_count++;
    }

    log_info(LogLLRuntime, "Watcher thread stopped watching...");
    watcher::server_running = false;
}

}  // namespace watcher

void watcher_init(Device *device) {
    std::vector<uint32_t> watcher_init_val;
    watcher_init_val.resize(sizeof(watcher_msg_t) / sizeof(uint32_t), 0);
    watcher_msg_t *data = reinterpret_cast<watcher_msg_t *>(&(watcher_init_val[0]));

    // Initialize watcher enable flag according to user setting.
    data->enable = (tt::llrt::OptionsG.get_watcher_enabled())? WatcherEnabled : WatcherDisabled;

    // Initialize debug status values to "unknown"
    for (int idx = 0; idx < MAX_RISCV_PER_CORE; idx++)
        data->debug_waypoint[idx].waypoint[0] = 'X';

    // Initialize debug sanity L1/NOC addresses to sentinel "all ok"
    for (int i = 0; i < NUM_NOCS; i++) {
        data->sanitize_noc[i].noc_addr = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_64;
        data->sanitize_noc[i].l1_addr = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_32;
        data->sanitize_noc[i].len = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_32;
        data->sanitize_noc[i].which = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_16;
        data->sanitize_noc[i].multicast = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_16;
        data->sanitize_noc[i].invalid = DebugSanitizeNocInvalidOK;
    }

    // Initialize debug asserts to not tripped.
    data->assert_status.line_num = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_16;
    data->assert_status.tripped = DebugAssertOK;
    data->assert_status.which = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_8;

    // Initialize pause flags to 0
    for (int idx = 0; idx < DebugNumUniqueRiscs; idx++)
        data->pause_status.flags[idx] = 0;

    // Initialize stack usage data to unset
    for (int idx = 0; idx < DebugNumUniqueRiscs; idx++)
        data->stack_usage.max_usage[idx] = watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_16;

    // Initialize debug ring buffer to a known init val, we'll check against this to see if any
    // data has been written.
    std::vector<uint32_t> debug_ring_buf_init_val(sizeof(debug_ring_buf_msg_t) / sizeof(uint32_t), 0);
    debug_ring_buf_msg_t *ring_buf_data = reinterpret_cast<debug_ring_buf_msg_t *>(&(debug_ring_buf_init_val[0]));
    data->debug_ring_buf.current_ptr = DEBUG_RING_BUFFER_STARTING_INDEX;
    data->debug_ring_buf.wrapped = 0;

    // Initialize Debug Delay feature
    std::map<CoreCoord, debug_insert_delays_msg_t> debug_delays_val;
    for (tt::llrt::RunTimeDebugFeatures delay_feature = tt::llrt::RunTimeDebugFeatureReadDebugDelay;
         (int)delay_feature <= tt::llrt::RunTimeDebugFeatureAtomicDebugDelay;
         delay_feature = (tt::llrt::RunTimeDebugFeatures)((int)delay_feature + 1)) {
        vector<chip_id_t> chip_ids = tt::llrt::OptionsG.get_feature_chip_ids(delay_feature);
        bool this_chip_enabled = tt::llrt::OptionsG.get_feature_all_chips(delay_feature) ||
                                 std::find(chip_ids.begin(), chip_ids.end(), device->id()) != chip_ids.end();
        if (this_chip_enabled) {
            static_assert(sizeof(debug_sanitize_noc_addr_msg_t) % sizeof(uint32_t) == 0);
            debug_insert_delays_msg_t delay_setup;

            // Create the mask based on the feature
            uint32_t hart_mask = tt::llrt::OptionsG.get_feature_riscv_mask(delay_feature);
            switch (delay_feature) {
                case tt::llrt::RunTimeDebugFeatureReadDebugDelay: delay_setup.read_delay_riscv_mask = hart_mask; break;
                case tt::llrt::RunTimeDebugFeatureWriteDebugDelay:
                    delay_setup.write_delay_riscv_mask = hart_mask;
                    break;
                case tt::llrt::RunTimeDebugFeatureAtomicDebugDelay:
                    delay_setup.atomic_delay_riscv_mask = hart_mask;
                    break;
                default: break;
            }

            for (CoreType core_type : {CoreType::WORKER, CoreType::ETH}) {
                vector<CoreCoord> delayed_cores = tt::llrt::OptionsG.get_feature_cores(delay_feature)[core_type];
                for (tt_xy_pair logical_core : delayed_cores) {
                    CoreCoord phys_core;
                    bool valid_logical_core = true;
                    try {
                        phys_core = device->physical_core_from_logical_core(logical_core, core_type);
                    } catch (std::runtime_error &error) {
                        valid_logical_core = false;
                    }
                    if (valid_logical_core) {
                        // Update the masks for the core
                        if (debug_delays_val.find(phys_core) != debug_delays_val.end()) {
                            debug_delays_val[phys_core].read_delay_riscv_mask |= delay_setup.read_delay_riscv_mask;
                            debug_delays_val[phys_core].write_delay_riscv_mask |= delay_setup.write_delay_riscv_mask;
                            debug_delays_val[phys_core].atomic_delay_riscv_mask |= delay_setup.atomic_delay_riscv_mask;
                        } else {
                            debug_delays_val.insert({phys_core, delay_setup});
                        }
                    } else {
                        log_warning(
                            tt::LogMetal,
                            "TT_METAL_{}_CORES included {} core with logical coordinates {} (physical coordinates {}), which is not a valid core on device {}. This coordinate will be ignored by {} feature.",
                            tt::llrt::RunTimeDebugFeatureNames[delay_feature],
                            tt::llrt::get_core_type_name(core_type),
                            logical_core.str(),
                            valid_logical_core? phys_core.str() : "INVALID",
                            device->id(),
                            tt::llrt::RunTimeDebugFeatureNames[delay_feature]
                        );
                    }
                }
            }
        }
    }

    // Iterate over debug_delays_val and print what got configured where
    for (auto &delay : debug_delays_val) {
        log_info(
            tt::LogMetal,
            "Configured Watcher debug delays for device {}, core {}: read_delay_cores_mask=0x{:x}, "
            "write_delay_cores_mask=0x{:x}, atomic_delay_cores_mask=0x{:x}. Delay cycles: {}",
            device->id(),
            delay.first.str().c_str(),
            delay.second.read_delay_riscv_mask,
            delay.second.write_delay_riscv_mask,
            delay.second.atomic_delay_riscv_mask,
            tt::llrt::OptionsG.get_watcher_debug_delay()
            );
    }

    debug_insert_delays_msg_t debug_delays_val_zero = {0, 0, 0, 0};

    // TODO: hal needs more work as of 8/6/24, but eventually loop over dispatch_core_types and get
    // cores from that to consolidate the loops below

    // Initialize worker cores debug values
    CoreCoord grid_size = device->logical_grid_size();
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core = device->worker_core_from_logical_core(logical_core);
            if (debug_delays_val.find(worker_core) != debug_delays_val.end()) {
                data->debug_insert_delays = debug_delays_val[worker_core];
            } else {
                data->debug_insert_delays = debug_delays_val_zero;
            }
            tt::llrt::write_hex_vec_to_core(device->id(), worker_core, watcher_init_val, GET_WATCHER_TENSIX_DEV_ADDR());
        }
    }

    // Initialize ethernet cores debug values
    for (const CoreCoord &eth_core : device->ethernet_cores()) {
        // Mailbox address is different depending on active vs inactive eth cores.
        bool is_active_eth_core;
        if (device->is_active_ethernet_core(eth_core)) {
            is_active_eth_core = true;
        } else if (device->is_inactive_ethernet_core(eth_core)) {
            is_active_eth_core = false;
        } else {
            continue;
        }
        CoreCoord physical_core = device->ethernet_core_from_logical_core(eth_core);
        if (debug_delays_val.find(physical_core) != debug_delays_val.end()) {
            data->debug_insert_delays = debug_delays_val[physical_core];
        } else {
            data->debug_insert_delays = debug_delays_val_zero;
        }
        tt::llrt::write_hex_vec_to_core(
            device->id(),
            physical_core,
            watcher_init_val,
            is_active_eth_core ? GET_WATCHER_ERISC_DEV_ADDR() : GET_WATCHER_IERISC_DEV_ADDR());
    }

    log_debug(LogLLRuntime, "Watcher initialized device {}", device->id());
}

void watcher_attach(Device *device) {
    const std::lock_guard<std::mutex> lock(watcher::watch_mutex);

    if (!watcher::enabled && tt::llrt::OptionsG.get_watcher_enabled()) {
        watcher::create_log_file();
        if (!watcher::kernel_file) {
            watcher::create_kernel_file();
        }
        watcher::watcher_killed_due_to_error = false;
        watcher::set_watcher_exception_message("");

        watcher::enabled = true;

        int sleep_usecs = tt::llrt::OptionsG.get_watcher_interval() * 1000;
        std::thread watcher_thread = std::thread(&watcher::watcher_loop, sleep_usecs);
        watcher_thread.detach();
    }

    if (watcher::logfile != nullptr) {
        fprintf(watcher::logfile, "At %.3lfs attach device %d\n", watcher::get_elapsed_secs(), device->id());
    }

    if (watcher::enabled) {
        log_info(LogLLRuntime, "Watcher attached device {}", device->id());
    }

    // Always register the device w/ watcher, even if disabled
    // This allows dump() to be called from debugger
    watcher::devices.emplace(
        device,
        watcher::WatcherDeviceReader(
            watcher::logfile, device, watcher::kernel_names, &watcher::set_watcher_exception_message));
}

void watcher_detach(Device *old) {
    {
        const std::lock_guard<std::mutex> lock(watcher::watch_mutex);

        TT_ASSERT(watcher::devices.find(old) != watcher::devices.end());
        if (watcher::enabled && watcher::logfile != nullptr) {
            log_info(LogLLRuntime, "Watcher detached device {}", old->id());
            fprintf(watcher::logfile, "At %.3lfs detach device %d\n", watcher::get_elapsed_secs(), old->id());
        }
        watcher::devices.erase(old);
        if (watcher::enabled && watcher::devices.empty()) {
            // If no devices remain, shut down the watcher server.
            watcher::enabled = false;
            if (watcher::logfile != nullptr) {
                std::fclose(watcher::logfile);
                watcher::logfile = nullptr;
            }
            if (watcher::kernel_file != nullptr) {
                std::fclose(watcher::kernel_file);
                watcher::kernel_file = nullptr;
            }
        }
    }

    // If we shut down the watcher server, wait until it finishes up. Do this without holding the
    // lock because the watcher server may be waiting on it before it does its exit check.
    if (watcher::devices.empty())
        while (watcher::server_running) {
            ;
        }
}

int watcher_register_kernel(const string &name) {
    const std::lock_guard<std::mutex> lock(watcher::watch_mutex);

    if (!watcher::kernel_file)
        watcher::create_kernel_file();
    int k_id = watcher::kernel_names.size();
    watcher::kernel_names.push_back(name);
    fprintf(watcher::kernel_file, "%d: %s\n", k_id, name.c_str());
    fflush(watcher::kernel_file);

    return k_id;
}

bool watcher_server_killed_due_to_error() { return watcher::watcher_killed_due_to_error; }

void watcher_server_set_error_flag(bool val) { watcher::watcher_killed_due_to_error = val; }

void watcher_clear_log() { watcher::create_log_file(); }

string watcher_get_log_file_name() {
    return tt::llrt::OptionsG.get_root_dir() + watcher::logfile_path + watcher::logfile_name;
}

int watcher_get_dump_count() { return watcher::dump_count; }

void watcher_dump() {
    if (!watcher::logfile)
        watcher::create_log_file();
    watcher::dump(watcher::logfile);
}

void watcher_read_kernel_ids_from_file() {
    std::filesystem::path output_dir(tt::llrt::OptionsG.get_root_dir() + watcher::logfile_path);
    string fname = output_dir.string() + watcher::kernel_file_name;
    FILE *f;
    if ((f = fopen(fname.c_str(), "r")) == nullptr) {
        TT_THROW("Watcher failed to open kernel name file: {}\n", fname);
    }

    char *line = nullptr;
    size_t len;
    while (getline(&line, &len, f) != -1) {
        string s(line);
        s = s.substr(0, s.length() - 1);            // Strip newline
        int k_id = stoi(s.substr(0, s.find(":")));  // Format is {k_id}: {kernel}
        watcher::kernel_names.push_back(s.substr(s.find(":") + 2));
    }
}

// Function to get the static string value
std::string get_watcher_exception_message() {
    std::lock_guard<std::mutex> lock(watcher::watcher_exception_message_mutex);
    return watcher::watcher_exception_message();
}

}  // namespace tt
