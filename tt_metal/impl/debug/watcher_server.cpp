// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc
//
// SPDX-License-Identifier: Apache-2.0

#include "watcher_server.hpp"

#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include <tt_stl/assert.hpp>
#include "core_coord.hpp"
#include "api/debug/ring_buffer.h"
#include "debug_helpers.hpp"
#include "hal_types.hpp"
#include "llrt/hal.hpp"
#include <tt-logger/tt-logger.hpp>
#include "metal_soc_descriptor.h"
#include <tt_stl/span.hpp>
#include "impl/context/metal_context.hpp"
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/types/xy_pair.hpp>
#include "rtoptions.hpp"
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
    void isolated_dump(std::vector<ChipId>& device_ids);
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
    void init_device(ChipId device_id);
    void poll_watcher_data();

    std::atomic<bool> stop_server_ = false;
    std::condition_variable stop_server_cv_;
    std::atomic<bool> server_running_ = false;
    std::atomic<bool> server_killed_due_to_error_ = false;
    std::atomic<int> dump_count_ = 0;

    std::map<ChipId, WatcherDeviceReader> device_id_to_reader_;
    std::vector<std::string> kernel_names_;
    inline static std::chrono::time_point start_time = std::chrono::system_clock::now();
    std::mutex watch_mutex_;  // Guards server internal state + logfile + device watcher mailbox

    std::thread* server_thread_{};

    FILE* logfile_ = nullptr;
    FILE* kernel_file_ = nullptr;
    FILE* kernel_elf_file_ = nullptr;

    std::string exception_message_;
    std::mutex exception_message_mutex_;

    inline static const std::string LOG_FILE_PATH = "generated/watcher/";
    inline static const std::string LOG_FILE_NAME = "watcher.log";
    inline static const std::string KERNEL_FILE_NAME = "kernel_names.txt";
    inline static const std::string KERNEL_ELF_FILE_NAME = "kernel_elf_paths.txt";
};

void WatcherServer::Impl::init_devices() {
    auto all_devices = MetalContext::instance().get_cluster().all_chip_ids();
    for (ChipId device_id : all_devices) {
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
        for (ChipId device_id : all_devices) {
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
    auto close_file = [](FILE*& file) {
        if (file != nullptr) {
            std::fclose(file);
            file = nullptr;
        }
    };
    if (!server_thread_ and !server_killed_due_to_error_) {
        close_file(logfile_);
        close_file(kernel_file_);
        close_file(kernel_elf_file_);
        return;
    }

    if (server_thread_) {
        // Let one full watcher dump happen so we can catch anything between the last scheduled dump and teardown.
        // Don't do this in test mode, to keep the tests running quickly.
        if (!MetalContext::instance().rtoptions().get_test_mode_enabled() and !server_killed_due_to_error_) {
            int target_count = dump_count() + 1;
            while (dump_count() < target_count) {
                ;
            }
        }
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
        for (ChipId device_id : all_devices) {
            TT_ASSERT(device_id_to_reader_.contains(device_id));
            device_id_to_reader_.erase(device_id);
            log_info(LogLLRuntime, "Watcher detached device {}", device_id);
            fprintf(logfile_, "At %.3lfs detach device %d\n", get_elapsed_secs(), device_id);
        }

        // Watcher server closed, can use dma library again.
        MetalContext::instance().rtoptions().set_disable_dma_ops(false);
        close_file(logfile_);
        close_file(kernel_file_);
        close_file(kernel_elf_file_);
    }
}

void WatcherServer::Impl::dump(FILE* f) {
    for (auto& device_id_and_reader : device_id_to_reader_) {
        device_id_and_reader.second.Dump(f);
    }
}

void WatcherServer::Impl::isolated_dump(std::vector<ChipId>& device_ids) {
    // No init, so we don't clear mailboxes
    clear_log();
    read_kernel_ids_from_file();
    for (ChipId device_id : device_ids) {
        device_id_to_reader_.try_emplace(device_id, logfile_, device_id, kernel_names_);
        log_info(LogLLRuntime, "Watcher attached device {}", device_id);
        fprintf(logfile_, "At %.3lfs attach device %d\n", get_elapsed_secs(), device_id);
    }
    dump();
    device_id_to_reader_.clear();
}

std::string WatcherServer::Impl::log_file_name() {
    return tt::tt_metal::MetalContext::instance().rtoptions().get_logs_dir() + LOG_FILE_PATH + LOG_FILE_NAME;
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
    std::filesystem::path output_dir(tt::tt_metal::MetalContext::instance().rtoptions().get_logs_dir() + LOG_FILE_PATH);
    std::string fname = output_dir.string() + KERNEL_FILE_NAME;
    std::ifstream f(fname);
    if (!f) {
        TT_THROW("Watcher failed to open kernel name file: {}\n", fname);
    }

    std::string line;
    while (std::getline(f, line)) {
        kernel_names_.push_back(line.substr(line.find(':') + 2));
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
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    const char* fmode = rtoptions.get_watcher_append() ? "a" : "w";
    std::filesystem::path output_dir(rtoptions.get_logs_dir() + LOG_FILE_PATH);
    std::filesystem::create_directories(output_dir);
    std::string fname = output_dir.string() + LOG_FILE_NAME;
    if (rtoptions.get_watcher_skip_logging()) {
        fname = "/dev/null";
    }
    FILE* f = fopen(fname.c_str(), fmode);
    if (!f) {
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
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    const char* fmode = rtoptions.get_watcher_append() ? "a" : "w";
    std::filesystem::path output_dir(rtoptions.get_logs_dir() + LOG_FILE_PATH);
    std::filesystem::create_directories(output_dir);
    std::string fname = output_dir.string() + KERNEL_FILE_NAME;
    FILE* f = fopen(fname.c_str(), fmode);
    if (!f) {
        TT_THROW("Watcher failed to create kernel name file\n");
    }
    kernel_names_.clear();
    kernel_names_.push_back("blank");
    fprintf(f, "0: blank\n");
    fflush(f);

    kernel_file_ = f;
}

void WatcherServer::Impl::create_kernel_elf_file() {
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    std::filesystem::path output_dir(rtoptions.get_logs_dir() + LOG_FILE_PATH);
    std::filesystem::create_directories(output_dir);
    std::string fname = output_dir.string() + KERNEL_ELF_FILE_NAME;
    FILE* f = fopen(fname.c_str(), "w");
    if (!f) {
        TT_THROW("Watcher failed to create kernel ELF file\n");
    }
    kernel_elf_file_ = f;
    fprintf(f, "0: blank\n");
    fflush(f);
}

void WatcherServer::Impl::init_device(ChipId device_id) {
    const std::lock_guard<std::mutex> lock(watch_mutex_);
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& hal = MetalContext::instance().hal();
    std::vector<dev_msgs::watcher_msg_t> watcher_init_val;
    watcher_init_val.reserve(NumHalProgrammableCoreTypes);

    for (int programmable_core_type_index = 0; programmable_core_type_index < NumHalProgrammableCoreTypes;
         programmable_core_type_index++) {
        HalProgrammableCoreType programmable_core_type = hal.get_programmable_core_type(programmable_core_type_index);
        auto factory = hal.get_dev_msgs_factory(programmable_core_type);
        watcher_init_val.push_back(factory.create<dev_msgs::watcher_msg_t>());
        auto data = watcher_init_val.back().view();
        // Initialize watcher enable flag according to user setting.
        data.enable() = (rtoptions.get_watcher_enabled()) ? dev_msgs::WatcherEnabled : dev_msgs::WatcherDisabled;
        // Initialize debug status values to "unknown"
        for (auto debug_waypoint : data.debug_waypoint()) {
            debug_waypoint.waypoint()[0] = 'X';
        }

        // Initialize debug sanity L1/NOC addresses to sentinel "all ok"
        for (auto sanitize : data.sanitize()) {
            sanitize.noc_addr() = DEBUG_SANITIZE_SENTINEL_OK_64;
            sanitize.l1_addr() = DEBUG_SANITIZE_SENTINEL_OK_32;
            sanitize.len() = DEBUG_SANITIZE_SENTINEL_OK_32;
            sanitize.which_risc() = DEBUG_SANITIZE_SENTINEL_OK_16;
            sanitize.return_code() = dev_msgs::DebugSanitizeOK;
            sanitize.is_multicast() = DEBUG_SANITIZE_SENTINEL_OK_8;
            sanitize.is_write() = DEBUG_SANITIZE_SENTINEL_OK_8;
            sanitize.is_target() = DEBUG_SANITIZE_SENTINEL_OK_8;
        }

        // Initialize debug asserts to not tripped.
        data.assert_status().line_num() = DEBUG_SANITIZE_SENTINEL_OK_16;
        data.assert_status().tripped() = dev_msgs::DebugAssertOK;
        data.assert_status().which() = DEBUG_SANITIZE_SENTINEL_OK_8;

        // Initialize debug ring buffer to a known init val, we'll check against this to see if any
        // data has been written.
        data.debug_ring_buf().current_ptr() = DEBUG_RING_BUFFER_STARTING_INDEX;
        data.debug_ring_buf().wrapped() = 0;
    }

    // Initialize Debug Delay feature
    std::map<CoreCoord, dev_msgs::debug_insert_delays_msg_t> debug_delays_val;
    constexpr tt::llrt::RunTimeDebugFeatures debug_delay_features[] = {
        tt::llrt::RunTimeDebugFeatureReadDebugDelay,
        tt::llrt::RunTimeDebugFeatureWriteDebugDelay,
        tt::llrt::RunTimeDebugFeatureAtomicDebugDelay};
    for (auto delay_feature : debug_delay_features) {
        const std::vector<ChipId>& chip_ids = rtoptions.get_feature_chip_ids(delay_feature);
        bool this_chip_enabled =
            rtoptions.get_feature_all_chips(delay_feature) || std::ranges::find(chip_ids, device_id) != chip_ids.end();
        if (this_chip_enabled) {
            for (CoreType core_type : {CoreType::WORKER, CoreType::ETH}) {
                const auto& delayed_cores = rtoptions.get_feature_cores(delay_feature);
                if (!delayed_cores.contains(core_type)) {
                    continue;
                }
                for (tt_xy_pair logical_core : delayed_cores.at(core_type)) {
                    CoreCoord virtual_core;
                    bool valid_logical_core = true;
                    try {
                        virtual_core =
                            cluster.get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, core_type);
                    } catch (std::runtime_error& error) {
                        valid_logical_core = false;
                    }
                    if (valid_logical_core) {
                        auto programmable_core_type = llrt::get_core_type(device_id, virtual_core);
                        // Create the mask based on the feature
                        uint32_t processor_mask =
                            rtoptions.get_feature_processors(delay_feature).get_processor_mask(programmable_core_type);
                        auto factory = hal.get_dev_msgs_factory(programmable_core_type);
                        // Update the masks for the core
                        auto iter = debug_delays_val.find(virtual_core);
                        if (iter == debug_delays_val.end()) {
                            iter = debug_delays_val
                                       .emplace(virtual_core, factory.create<dev_msgs::debug_insert_delays_msg_t>())
                                       .first;
                        }
                        auto delay_setup = iter->second.view();
                        switch (delay_feature) {
                            case tt::llrt::RunTimeDebugFeatureReadDebugDelay:
                                delay_setup.read_delay_processor_mask() |= processor_mask;
                                break;
                            case tt::llrt::RunTimeDebugFeatureWriteDebugDelay:
                                delay_setup.write_delay_processor_mask() |= processor_mask;
                                break;
                            case tt::llrt::RunTimeDebugFeatureAtomicDebugDelay:
                                delay_setup.atomic_delay_processor_mask() |= processor_mask;
                                break;
                            default: TT_THROW("Unexpected debug delay feature");
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
    for (auto& [core, delay_setup] : debug_delays_val) {
        log_info(
            tt::LogMetal,
            "Configured Watcher debug delays for device {}, core {}: read_delay_cores_mask=0x{:x}, "
            "write_delay_cores_mask=0x{:x}, atomic_delay_cores_mask=0x{:x}. Delay cycles: {}",
            device_id,
            core.str().c_str(),
            delay_setup.view().read_delay_processor_mask(),
            delay_setup.view().write_delay_processor_mask(),
            delay_setup.view().atomic_delay_processor_mask(),
            rtoptions.get_watcher_debug_delay());
    }

    auto write_watcher_init_val = [&](const CoreCoord& logical_core, HalProgrammableCoreType programmable_core_type) {
        auto programmable_core_type_index = hal.get_programmable_core_type_index(programmable_core_type);
        CoreCoord virtual_core = cluster.get_virtual_coordinate_from_logical_coordinates(
            device_id, logical_core, hal.get_core_type(programmable_core_type_index));
        auto data = watcher_init_val[programmable_core_type_index].view();
        if (auto iter = debug_delays_val.find(virtual_core); iter != debug_delays_val.end()) {
            std::copy_n(iter->second.data(), iter->second.size(), data.debug_insert_delays().data());
        } else {
            std::fill_n(data.debug_insert_delays().data(), data.debug_insert_delays().size(), std::byte{0});
        }
        auto addr = hal.get_dev_addr(programmable_core_type, HalL1MemAddrType::WATCHER);
        cluster.write_core(data.data(), data.size(), {static_cast<size_t>(device_id), virtual_core}, addr);
    };

    // Initialize worker cores debug values
    CoreCoord grid_size = cluster.get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            write_watcher_init_val({x, y}, HalProgrammableCoreType::TENSIX);
        }
    }

    // Initialize ethernet cores debug values
    for (const CoreCoord& active_eth_core :
         tt::tt_metal::MetalContext::instance().get_control_plane().get_active_ethernet_cores(device_id)) {
        write_watcher_init_val(active_eth_core, HalProgrammableCoreType::ACTIVE_ETH);
    }
    for (const CoreCoord& inactive_eth_core :
         tt::tt_metal::MetalContext::instance().get_control_plane().get_inactive_ethernet_cores(device_id)) {
        write_watcher_init_val(inactive_eth_core, HalProgrammableCoreType::IDLE_ETH);
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
    std::string disabled_features;
    const auto& disabled_features_set = rtoptions.get_watcher_disabled_features();
    if (!disabled_features_set.empty()) {
        for (const auto& feature : disabled_features_set) {
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

        if (device_id_to_reader_.empty()) {
            fprintf(logfile_, "No active devices\n");
        }

        try {
            dump();
        } catch (std::runtime_error& e) {
            // Depending on whether test mode is enabled, catch and stop server, or re-throw.
            if (rtoptions.get_test_mode_enabled()) {
                server_killed_due_to_error_ = true;
                break;
            }
            throw e;
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
void WatcherServer::isolated_dump(std::vector<ChipId>& device_ids) { impl_->isolated_dump(device_ids); }
}  // namespace tt::tt_metal
