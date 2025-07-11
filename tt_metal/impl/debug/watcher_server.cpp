// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc
//
// SPDX-License-Identifier: Apache-2.0

#include "watcher_server.hpp"

#include "dev_msgs.h"
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <condition_variable>
#include <filesystem>
#include <future>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "assert.hpp"
#include "core_coord.hpp"
#include "debug/ring_buffer.h"
#include "debug_helpers.hpp"
#include "hal_types.hpp"
#include "llrt.hpp"
#include <tt-logger/tt-logger.hpp>
#include "metal_soc_descriptor.h"
#include <tt_stl/span.hpp>
#include "impl/context/metal_context.hpp"
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/tt_xy_pair.h>
#include <umd/device/types/cluster_descriptor_types.h>
#include <umd/device/types/xy_pair.h>
#include "watcher_device_reader.hpp"

using namespace tt::tt_metal;

namespace tt::tt_metal {
class WatcherServer::Impl {
public:
    // Implementation of WatcherServer public functions
    void init_devices();
    void attach_devices();
    void detach_devices();
    void __attribute__((noinline)) dump(FILE* f);  // noinline so that this fn exists to be called from gdb
    void dump() { dump(logfile_); }
    void isolated_dump(std::vector<chip_id_t>& device_ids);
    void clear_log() {
        const std::lock_guard<std::mutex> lock(watch_mutex_);
        create_log_file();
    }
    std::string log_file_name();
    int register_kernel(const std::string& name);
    void register_kernel_elf_paths(int id, std::vector<std::string>& paths);
    void read_kernel_ids_from_file();
    bool killed_due_to_error() { return server_killed_due_to_error_; }
    void set_killed_due_to_error_flag(bool val) { server_killed_due_to_error_ = val; }
    std::string exception_message();
    void set_exception_message(const std::string& message);
    int dump_count() { return dump_count_.load(); }
    std::unique_lock<std::mutex> get_lock() { return std::unique_lock<std::mutex>(watch_mutex_); }

private:
    double get_elapsed_secs();
    void create_log_file();
    void create_kernel_file();
    void create_kernel_elf_file();
    void init_device(chip_id_t device_id);
    void poll_watcher_data();

    std::atomic<bool> stop_server_ = false;
    std::condition_variable stop_server_cv_;
    std::atomic<bool> server_running_ = false;
    std::atomic<bool> server_killed_due_to_error_ = false;
    std::atomic<int> dump_count_ = 0;

    std::map<chip_id_t, WatcherDeviceReader> device_id_to_reader_;
    std::vector<std::string> kernel_names_;
    inline static std::chrono::time_point start_time = std::chrono::system_clock::now();
    std::mutex watch_mutex_;  // Guards server internal state + logfile + device watcher mailbox

    std::thread* server_thread_;

    FILE* logfile_ = nullptr;
    FILE* kernel_file_ = nullptr;
    FILE* kernel_elf_file_ = nullptr;

    std::string exception_message_ = "";
    std::mutex exception_message_mutex_;

    inline static const std::string LOG_FILE_PATH = "generated/watcher/";
    inline static const std::string LOG_FILE_NAME = "watcher.log";
    inline static const std::string KERNEL_FILE_NAME = "kernel_names.txt";
    inline static const std::string KERNEL_ELF_FILE_NAME = "kernel_elf_paths.txt";
};

#define GET_WATCHER_TENSIX_DEV_ADDR() \
    MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::WATCHER)

#define GET_WATCHER_ERISC_DEV_ADDR() \
    MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::WATCHER)

#define GET_WATCHER_IERISC_DEV_ADDR() \
    MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::WATCHER)

void WatcherServer::Impl::init_devices() {
    auto all_devices = MetalContext::instance().get_cluster().all_chip_ids();
    for (chip_id_t device_id : all_devices) {
        init_device(device_id);
    }
}

void WatcherServer::Impl::attach_devices() {
    auto& rtoptions = tt_metal::MetalContext::instance().rtoptions();
    if (!rtoptions.get_watcher_enabled()) {
        return;
    }

    {
        const std::lock_guard<std::mutex> lock(watch_mutex_);
        create_log_file();
        create_kernel_file();
        auto all_devices = MetalContext::instance().get_cluster().all_chip_ids();
        for (chip_id_t device_id : all_devices) {
            device_id_to_reader_.try_emplace(device_id, logfile_, device_id, kernel_names_);
            log_info(LogLLRuntime, "Watcher attached device {}", device_id);
            fprintf(logfile_, "At %.3lfs attach device %d\n", get_elapsed_secs(), device_id);
        }

        // Since dma library is not thread-safe, disable it when watcher runs.
        rtoptions.set_disable_dma_ops(true);
    }

    // Spin off thread to run the server.
    server_thread_ = new std::thread(&WatcherServer::Impl::poll_watcher_data, this);
}

void WatcherServer::Impl::detach_devices() {
    // If server isn't running, and wasn't killed due to an error, nothing to do here.
    if (!server_thread_ and !server_killed_due_to_error_) {
        return;
    }

    if (server_thread_) {
        // Signal the server thread to finish
        stop_server_ = true;
        stop_server_cv_.notify_all();

        // Wait for the thread to end, with a timeout
        auto future = std::async(std::launch::async, &std::thread::join, server_thread_);
        if (future.wait_for(std::chrono::seconds(2)) == std::future_status::timeout) {
            log_fatal(tt::LogMetal, "Timed out waiting on watcher server thread to terminate.");
        }
        delete server_thread_;
        server_thread_ = nullptr;
    }

    // Detach all devices
    {
        const std::lock_guard<std::mutex> lock(watch_mutex_);
        auto all_devices = MetalContext::instance().get_cluster().all_chip_ids();
        for (chip_id_t device_id : all_devices) {
            TT_ASSERT(device_id_to_reader_.count(device_id) > 0);
            device_id_to_reader_.erase(device_id);
            log_info(LogLLRuntime, "Watcher detached device {}", device_id);
            fprintf(logfile_, "At %.3lfs detach device %d\n", get_elapsed_secs(), device_id);
        }

        // Watcher server closed, can use dma library again.
        MetalContext::instance().rtoptions().set_disable_dma_ops(false);

        // Close files
        std::fclose(logfile_);
        logfile_ = nullptr;
    }
}

void WatcherServer::Impl::dump(FILE* f) {
    for (auto& device_id_and_reader : device_id_to_reader_) {
        device_id_and_reader.second.Dump(f);
    }
}

void WatcherServer::Impl::isolated_dump(std::vector<chip_id_t>& device_ids) {
    // No init, so we don't clear mailboxes
    clear_log();
    read_kernel_ids_from_file();
    for (chip_id_t device_id : device_ids) {
        device_id_to_reader_.try_emplace(device_id, logfile_, device_id, kernel_names_);
        log_info(LogLLRuntime, "Watcher attached device {}", device_id);
        fprintf(logfile_, "At %.3lfs attach device %d\n", get_elapsed_secs(), device_id);
    }
    dump();
    device_id_to_reader_.clear();
}

std::string WatcherServer::Impl::log_file_name() {
    return tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir() + LOG_FILE_PATH + LOG_FILE_NAME;
}

int WatcherServer::Impl::register_kernel(const std::string& name) {
    const std::lock_guard<std::mutex> lock(watch_mutex_);

    if (!kernel_file_) {
        create_kernel_file();
    }
    int k_id = kernel_names_.size();
    kernel_names_.push_back(name);
    fprintf(kernel_file_, "%d: %s\n", k_id, name.c_str());
    fflush(kernel_file_);

    return k_id;
}

void WatcherServer::Impl::register_kernel_elf_paths(int id, std::vector<std::string>& paths) {
    const std::lock_guard<std::mutex> lock(watch_mutex_);
    if (!kernel_elf_file_) {
        create_kernel_elf_file();
    }
    std::string combined_paths = paths[0];
    for (int i = 1; i < paths.size(); i++) {
        combined_paths += ":" + paths[i];
    }
    fprintf(kernel_elf_file_, "%d: %s\n", id, combined_paths.c_str());
    fflush(kernel_elf_file_);
}

void WatcherServer::Impl::read_kernel_ids_from_file() {
    std::filesystem::path output_dir(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir() + LOG_FILE_PATH);
    std::string fname = output_dir.string() + KERNEL_FILE_NAME;
    FILE* f;
    if ((f = fopen(fname.c_str(), "r")) == nullptr) {
        TT_THROW("Watcher failed to open kernel name file: {}\n", fname);
    }

    char* line = nullptr;
    size_t len;
    while (getline(&line, &len, f) != -1) {
        std::string s(line);
        s = s.substr(0, s.length() - 1);            // Strip newline
        int k_id = stoi(s.substr(0, s.find(":")));  // Format is {k_id}: {kernel}
        kernel_names_.push_back(s.substr(s.find(":") + 2));
    }
}

std::string WatcherServer::Impl::exception_message() {
    std::lock_guard<std::mutex> lock(exception_message_mutex_);
    return exception_message_;
}

void WatcherServer::Impl::set_exception_message(const std::string& message) {
    std::lock_guard<std::mutex> lock(exception_message_mutex_);
    exception_message_ = message;
}

double WatcherServer::Impl::get_elapsed_secs() {
    std::chrono::time_point now_time = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_secs = now_time - start_time;
    return elapsed_secs.count();
}

void WatcherServer::Impl::create_log_file() {
    FILE* f;

    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    const char* fmode = rtoptions.get_watcher_append() ? "a" : "w";
    std::filesystem::path output_dir(rtoptions.get_root_dir() + LOG_FILE_PATH);
    std::filesystem::create_directories(output_dir);
    std::string fname = output_dir.string() + LOG_FILE_NAME;
    if (rtoptions.get_watcher_skip_logging()) {
        fname = "/dev/null";
    }
    if ((f = fopen(fname.c_str(), fmode)) == nullptr) {
        TT_THROW("Watcher failed to create log file\n");
    }
    log_info(LogLLRuntime, "Watcher log file: {}", fname);

    fprintf(f, "At %.3lfs starting\n", get_elapsed_secs());
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
    fprintf(
        f,
        "\trmsg(brisc host run message): D/H device/host dispatch; brisc NOC ID; I/G/D init/go/done; | separator; "
        "B/b enable/disable brisc; N/n enable/disable ncrisc; T/t enable/disable TRISC\n");
    fprintf(f, "\tsmsg(subordinate run message): I/G/D for NCRISC, TRISC0, TRISC1, TRISC2\n");
    fprintf(f, "\tk_ids:<brisc id>|<ncrisc id>|<trisc id> (ID map to file at end of section)\n");
    fprintf(f, "\n");
    fflush(f);

    logfile_ = f;
}

void WatcherServer::Impl::create_kernel_file() {
    FILE* f;
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    const char* fmode = rtoptions.get_watcher_append() ? "a" : "w";
    std::filesystem::path output_dir(rtoptions.get_root_dir() + LOG_FILE_PATH);
    std::filesystem::create_directories(output_dir);
    std::string fname = output_dir.string() + KERNEL_FILE_NAME;
    if ((f = fopen(fname.c_str(), fmode)) == nullptr) {
        TT_THROW("Watcher failed to create kernel name file\n");
    }
    kernel_names_.clear();
    kernel_names_.push_back("blank");
    fprintf(f, "0: blank\n");
    fflush(f);

    kernel_file_ = f;
}

void WatcherServer::Impl::create_kernel_elf_file() {
    FILE* f;
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    std::filesystem::path output_dir(rtoptions.get_root_dir() + LOG_FILE_PATH);
    std::filesystem::create_directories(output_dir);
    std::string fname = output_dir.string() + KERNEL_ELF_FILE_NAME;
    if ((f = fopen(fname.c_str(), "w")) == nullptr) {
        TT_THROW("Watcher failed to create kernel ELF file\n");
    }
    kernel_elf_file_ = f;
    fprintf(f, "0: blank\n");
    fflush(f);
}

void WatcherServer::Impl::init_device(chip_id_t device_id) {
    const std::lock_guard<std::mutex> lock(watch_mutex_);
    std::vector<uint32_t> watcher_init_val;
    watcher_init_val.resize(sizeof(watcher_msg_t) / sizeof(uint32_t), 0);
    watcher_msg_t* data = reinterpret_cast<watcher_msg_t*>(&(watcher_init_val[0]));
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();

    // Initialize watcher enable flag according to user setting.
    data->enable = (rtoptions.get_watcher_enabled()) ? WatcherEnabled : WatcherDisabled;

    // Initialize debug status values to "unknown"
    for (int idx = 0; idx < MAX_RISCV_PER_CORE; idx++) {
        data->debug_waypoint[idx].waypoint[0] = 'X';
    }

    // Initialize debug sanity L1/NOC addresses to sentinel "all ok"
    const auto NUM_NOCS = tt::tt_metal::MetalContext::instance().hal().get_num_nocs();
    for (int i = 0; i < NUM_NOCS; i++) {
        data->sanitize_noc[i].noc_addr = DEBUG_SANITIZE_NOC_SENTINEL_OK_64;
        data->sanitize_noc[i].l1_addr = DEBUG_SANITIZE_NOC_SENTINEL_OK_32;
        data->sanitize_noc[i].len = DEBUG_SANITIZE_NOC_SENTINEL_OK_32;
        data->sanitize_noc[i].which_risc = DEBUG_SANITIZE_NOC_SENTINEL_OK_16;
        data->sanitize_noc[i].return_code = DebugSanitizeNocOK;
        data->sanitize_noc[i].is_multicast = DEBUG_SANITIZE_NOC_SENTINEL_OK_8;
        data->sanitize_noc[i].is_write = DEBUG_SANITIZE_NOC_SENTINEL_OK_8;
        data->sanitize_noc[i].is_target = DEBUG_SANITIZE_NOC_SENTINEL_OK_8;
    }

    // Initialize debug asserts to not tripped.
    data->assert_status.line_num = DEBUG_SANITIZE_NOC_SENTINEL_OK_16;
    data->assert_status.tripped = DebugAssertOK;
    data->assert_status.which = DEBUG_SANITIZE_NOC_SENTINEL_OK_8;

    // Initialize pause flags to 0
    for (int idx = 0; idx < DebugNumUniqueRiscs; idx++) {
        data->pause_status.flags[idx] = 0;
    }

    // Initialize stack usage data to unset
    for (int idx = 0; idx < DebugNumUniqueRiscs; idx++) {
        data->stack_usage.cpu[idx].min_free = 0;
    }

    // Initialize debug ring buffer to a known init val, we'll check against this to see if any
    // data has been written.
    std::vector<uint32_t> debug_ring_buf_init_val(sizeof(debug_ring_buf_msg_t) / sizeof(uint32_t), 0);
    data->debug_ring_buf.current_ptr = DEBUG_RING_BUFFER_STARTING_INDEX;
    data->debug_ring_buf.wrapped = 0;

    // Initialize Debug Delay feature
    std::map<CoreCoord, debug_insert_delays_msg_t> debug_delays_val;
    for (tt::llrt::RunTimeDebugFeatures delay_feature = tt::llrt::RunTimeDebugFeatureReadDebugDelay;
         (int)delay_feature <= tt::llrt::RunTimeDebugFeatureAtomicDebugDelay;
         delay_feature = (tt::llrt::RunTimeDebugFeatures)((int)delay_feature + 1)) {
        std::vector<chip_id_t> chip_ids = rtoptions.get_feature_chip_ids(delay_feature);
        bool this_chip_enabled = rtoptions.get_feature_all_chips(delay_feature) ||
                                 std::find(chip_ids.begin(), chip_ids.end(), device_id) != chip_ids.end();
        if (this_chip_enabled) {
            static_assert(sizeof(debug_sanitize_noc_addr_msg_t) % sizeof(uint32_t) == 0);
            debug_insert_delays_msg_t delay_setup;

            // Create the mask based on the feature
            uint32_t hart_mask = rtoptions.get_feature_riscv_mask(delay_feature);
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
                const auto& delayed_cores = rtoptions.get_feature_cores(delay_feature);
                if (delayed_cores.count(core_type) == 0) {
                    continue;
                }
                for (tt_xy_pair logical_core : delayed_cores.at(core_type)) {
                    CoreCoord virtual_core;
                    bool valid_logical_core = true;
                    try {
                        virtual_core =
                            tt::tt_metal::MetalContext::instance()
                                .get_cluster()
                                .get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, core_type);
                    } catch (std::runtime_error& error) {
                        valid_logical_core = false;
                    }
                    if (valid_logical_core) {
                        // Update the masks for the core
                        if (debug_delays_val.find(virtual_core) != debug_delays_val.end()) {
                            debug_delays_val[virtual_core].read_delay_riscv_mask |= delay_setup.read_delay_riscv_mask;
                            debug_delays_val[virtual_core].write_delay_riscv_mask |= delay_setup.write_delay_riscv_mask;
                            debug_delays_val[virtual_core].atomic_delay_riscv_mask |=
                                delay_setup.atomic_delay_riscv_mask;
                        } else {
                            debug_delays_val.insert({virtual_core, delay_setup});
                        }
                    } else {
                        log_warning(
                            tt::LogMetal,
                            "TT_METAL_{}_CORES included {} core with logical coordinates {} (virtual coordinates {}), "
                            "which is not a valid core on device {}. This coordinate will be ignored by {} feature.",
                            tt::llrt::RunTimeDebugFeatureNames[delay_feature],
                            tt::tt_metal::get_core_type_name(core_type),
                            logical_core.str(),
                            valid_logical_core ? virtual_core.str() : "INVALID",
                            device_id,
                            tt::llrt::RunTimeDebugFeatureNames[delay_feature]);
                    }
                }
            }
        }
    }

    // Iterate over debug_delays_val and print what got configured where
    for (auto& delay : debug_delays_val) {
        log_info(
            tt::LogMetal,
            "Configured Watcher debug delays for device {}, core {}: read_delay_cores_mask=0x{:x}, "
            "write_delay_cores_mask=0x{:x}, atomic_delay_cores_mask=0x{:x}. Delay cycles: {}",
            device_id,
            delay.first.str().c_str(),
            delay.second.read_delay_riscv_mask,
            delay.second.write_delay_riscv_mask,
            delay.second.atomic_delay_riscv_mask,
            rtoptions.get_watcher_debug_delay());
    }

    debug_insert_delays_msg_t debug_delays_val_zero = {0, 0, 0, 0};

    // TODO: hal needs more work as of 8/6/24, but eventually loop over dispatch_core_types and get
    // cores from that to consolidate the loops below

    // Initialize worker cores debug values
    CoreCoord grid_size =
        tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core =
                tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
                    device_id, logical_core, CoreType::WORKER);
            if (debug_delays_val.find(worker_core) != debug_delays_val.end()) {
                data->debug_insert_delays = debug_delays_val[worker_core];
            } else {
                data->debug_insert_delays = debug_delays_val_zero;
            }
            tt::llrt::write_hex_vec_to_core(
                device_id,
                worker_core,
                tt::stl::Span<const uint32_t>(watcher_init_val.data(), watcher_init_val.size()),
                GET_WATCHER_TENSIX_DEV_ADDR());
        }
    }

    // Initialize ethernet cores debug values
    auto init_eth_debug_values = [&](const CoreCoord& eth_core, bool is_active_eth_core) {
        CoreCoord virtual_core =
            tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
                device_id, eth_core, CoreType::ETH);
        if (debug_delays_val.find(virtual_core) != debug_delays_val.end()) {
            data->debug_insert_delays = debug_delays_val[virtual_core];
        } else {
            data->debug_insert_delays = debug_delays_val_zero;
        }
        tt::llrt::write_hex_vec_to_core(
            device_id,
            virtual_core,
            watcher_init_val,
            is_active_eth_core ? GET_WATCHER_ERISC_DEV_ADDR() : GET_WATCHER_IERISC_DEV_ADDR());
    };
    for (const CoreCoord& active_eth_core :
         tt::tt_metal::MetalContext::instance().get_control_plane().get_active_ethernet_cores(device_id)) {
        init_eth_debug_values(active_eth_core, true);
    }
    for (const CoreCoord& inactive_eth_core :
         tt::tt_metal::MetalContext::instance().get_control_plane().get_inactive_ethernet_cores(device_id)) {
        init_eth_debug_values(inactive_eth_core, false);
    }

    log_debug(LogLLRuntime, "Watcher initialized device {}", device_id);
}

void WatcherServer::Impl::poll_watcher_data() {
    TT_ASSERT(server_running_ == false);
    server_running_ = true;
    dump_count_ = 1;
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    auto sleep_duration = std::chrono::milliseconds(rtoptions.get_watcher_interval());

    // Print to the user which features are disabled via env vars.
    std::string disabled_features = "";
    auto& disabled_features_set = rtoptions.get_watcher_disabled_features();
    if (!disabled_features_set.empty()) {
        for (auto& feature : disabled_features_set) {
            disabled_features += feature + ",";
        }
        disabled_features.pop_back();
    } else {
        disabled_features = "None";
    }
    log_info(LogLLRuntime, "Watcher server initialized, disabled features: {}", disabled_features);

    while (true) {
        std::unique_lock<std::mutex> lock(watch_mutex_);
        if (stop_server_cv_.wait_for(lock, sleep_duration, [&] { return stop_server_.load(); })) {
            break;
        }

        fprintf(logfile_, "-----\n");
        fprintf(logfile_, "Dump #%d at %.3lfs\n", dump_count_.load(), get_elapsed_secs());

        if (device_id_to_reader_.size() == 0) {
            fprintf(logfile_, "No active devices\n");
        }

        try {
            dump();
        } catch (std::runtime_error& e) {
            // Depending on whether test mode is enabled, catch and stop server, or re-throw.
            if (rtoptions.get_test_mode_enabled()) {
                server_killed_due_to_error_ = true;
                break;
            } else {
                throw e;
            }
        }

        fprintf(logfile_, "Dump #%d completed at %.3lfs\n", dump_count_.load(), get_elapsed_secs());
        fflush(logfile_);
        dump_count_++;
    }

    log_info(LogLLRuntime, "Watcher thread stopped watching...");
    server_running_ = false;
    stop_server_ = false;
}

// Wrapper class functions
WatcherServer::WatcherServer() : impl_(std::make_unique<Impl>()) {};
WatcherServer::~WatcherServer() = default;
void WatcherServer::init_devices() { impl_->init_devices(); }
void WatcherServer::attach_devices() { impl_->attach_devices(); }
void WatcherServer::detach_devices() { impl_->detach_devices(); }
void WatcherServer::clear_log() { impl_->clear_log(); }
std::string WatcherServer::log_file_name() { return impl_->log_file_name(); }
int WatcherServer::register_kernel(const std::string& name) { return impl_->register_kernel(name); }
void WatcherServer::register_kernel_elf_paths(int id, std::vector<std::string>& paths) {
    impl_->register_kernel_elf_paths(id, paths);
}
bool WatcherServer::killed_due_to_error() { return impl_->killed_due_to_error(); }
void WatcherServer::set_killed_due_to_error_flag(bool val) { impl_->set_killed_due_to_error_flag(val); }
std::string WatcherServer::exception_message() { return impl_->exception_message(); }
void WatcherServer::set_exception_message(const std::string& msg) { impl_->set_exception_message(msg); }
int WatcherServer::dump_count() { return impl_->dump_count(); }
std::unique_lock<std::mutex> WatcherServer::get_lock() { return impl_->get_lock(); }
void WatcherServer::isolated_dump(std::vector<chip_id_t>& device_ids) { impl_->isolated_dump(device_ids); }
}  // namespace tt::tt_metal
