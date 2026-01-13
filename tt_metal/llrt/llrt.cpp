// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/assert.hpp>
#include <fmt/base.h>
#include <fmt/ranges.h>
#include <tt-logger/tt-logger.hpp>
#include <unistd.h>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "hal.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "hal_types.hpp"
#include "llrt.hpp"
#include <umd/device/driver_atomics.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <llrt/tt_cluster.hpp>

namespace {
void print_aerisc_training_status(tt::ChipId device_id, const CoreCoord& virtual_core) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    if (!hal.get_dispatch_feature_enabled(tt::tt_metal::DispatchFeature::ETH_MAILBOX_API)) {
        return;
    }
    const auto port_status_addr = hal.get_eth_fw_mailbox_val(tt::tt_metal::FWMailboxMsg::PORT_STATUS);
    const auto retrain_count_addr = hal.get_eth_fw_mailbox_val(tt::tt_metal::FWMailboxMsg::RETRAIN_COUNT);
    const auto rx_link_up_addr = hal.get_eth_fw_mailbox_val(tt::tt_metal::FWMailboxMsg::RX_LINK_UP);
    uint32_t port_status = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        device_id, virtual_core, port_status_addr, sizeof(uint32_t))[0];
    uint32_t retrain_count = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        device_id, virtual_core, retrain_count_addr, sizeof(uint32_t))[0];
    uint32_t rx_link_up = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        device_id, virtual_core, rx_link_up_addr, sizeof(uint32_t))[0];
    log_critical(
        tt::LogMetal,
        "Device {}: Virtual core {}, Port status: {:#x}, Retrain count: {:#x}, Rx link up: {:#x}",
        device_id,
        virtual_core.str(),
        port_status,
        retrain_count,
        rx_link_up);
}
}  // namespace

// llrt = lower-level runtime
namespace tt::llrt {

using std::uint16_t;
using std::uint32_t;
using std::uint64_t;

const ll_api::memory& get_risc_binary(
    const std::string& path,
    ll_api::memory::Loading loading,
    const std::function<void(ll_api::memory&)>& update_callback) {
    static struct {
        std::unordered_map<std::string, std::unique_ptr<const ll_api::memory>> map;
        std::mutex mutex;
        std::condition_variable cvar;
    } cache;

    std::unique_lock lock(cache.mutex);
    auto [slot, inserted] = cache.map.try_emplace(path);
    const ll_api::memory* ptr = nullptr;
    if (inserted) {
        // We're the first with PATH. Create and insert.
        lock.unlock();
        ll_api::memory* mutable_ptr = new ll_api::memory(path, loading);
        if (update_callback) {
            update_callback(*mutable_ptr);
        }

        lock.lock();
        // maps have iterator stability, so SLOT is still valid.
        slot->second = decltype(slot->second)(mutable_ptr);
        ptr = mutable_ptr;
        // We can't wake just those waiting on this slot, so wake them
        // all. Should be a rare event anyway.
        cache.cvar.notify_all();
    } else {
        if (!slot->second) {
            // Someone else is creating the initial entry, wait for them.
            cache.cvar.wait(lock, [=] { return bool(slot->second); });
        }
        ptr = slot->second.get();
        TT_ASSERT(ptr->get_loading() == loading);
    }

    return *ptr;
}

// CoreCoord core --> NOC coordinates ("functional workers" from the SOC descriptor)
// NOC coord is also synonymous to routing / physical coord
// dram_channel id (0..7) for GS is also mapped to NOC coords in the SOC descriptor
CoreCoord logical_core_from_ethernet_core(tt::ChipId chip_id, const CoreCoord& ethernet_core) {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_logical_ethernet_core_from_virtual(
        chip_id, ethernet_core);
}

tt_metal::HalProgrammableCoreType get_core_type(tt::ChipId chip_id, const CoreCoord& virtual_core) {
    bool is_eth_core = tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_core(virtual_core, chip_id);
    bool is_active_eth_core = false;
    bool is_inactive_eth_core = false;

    // Determine whether an ethernet core is active or idle. Their host handshake interfaces are different.
    if (is_eth_core) {
        auto active_eth_cores =
            tt::tt_metal::MetalContext::instance().get_control_plane().get_active_ethernet_cores(chip_id);
        auto inactive_eth_cores =
            tt::tt_metal::MetalContext::instance().get_control_plane().get_inactive_ethernet_cores(chip_id);
        is_active_eth_core = active_eth_cores.contains(logical_core_from_ethernet_core(chip_id, virtual_core));
        is_inactive_eth_core = inactive_eth_cores.contains(logical_core_from_ethernet_core(chip_id, virtual_core));
        // we should not be operating on any reserved cores here.
        TT_ASSERT(is_active_eth_core or is_inactive_eth_core);
    }

    if (is_active_eth_core) {
        return tt_metal::HalProgrammableCoreType::ACTIVE_ETH;
    }
    if (is_inactive_eth_core) {
        return tt_metal::HalProgrammableCoreType::IDLE_ETH;
    }
    return tt_metal::HalProgrammableCoreType::TENSIX;
}

void send_reset_go_signal(tt::ChipId chip, const CoreCoord& virtual_core) {
    tt_metal::HalProgrammableCoreType dispatch_core_type = get_core_type(chip, virtual_core);
    const auto& hal = tt_metal::MetalContext::instance().hal();
    const auto& cluster = tt_metal::MetalContext::instance().get_cluster();
    uint64_t go_signal_adrr = hal.get_dev_addr(dispatch_core_type, tt_metal::HalL1MemAddrType::GO_MSG);
    auto reset_msg = hal.get_dev_msgs_factory(dispatch_core_type).create<tt_metal::dev_msgs::go_msg_t>();

    reset_msg.view().signal() = tt_metal::dev_msgs::RUN_MSG_RESET_READ_PTR_FROM_HOST;
    cluster.write_core_immediate(
        reset_msg.data(), reset_msg.size(), {static_cast<size_t>(chip), virtual_core}, go_signal_adrr);
    cluster.l1_barrier(chip);
    uint32_t go_message_index_addr = hal.get_dev_addr(dispatch_core_type, tt_metal::HalL1MemAddrType::GO_MSG_INDEX);
    uint32_t zero = 0;
    cluster.write_core_immediate(
        &zero, sizeof(uint32_t), {static_cast<size_t>(chip), virtual_core}, go_message_index_addr);
}

void write_launch_msg_to_core(
    tt::ChipId chip,
    CoreCoord core,
    tt_metal::dev_msgs::launch_msg_t::View msg,
    tt_metal::dev_msgs::go_msg_t::ConstView go_msg,
    bool send_go) {
    tt_metal::HalProgrammableCoreType dispatch_core_type = get_core_type(chip, core);
    const auto& hal = tt_metal::MetalContext::instance().hal();
    const auto& cluster = tt_metal::MetalContext::instance().get_cluster();

    msg.kernel_config().mode() = tt_metal::dev_msgs::DISPATCH_MODE_HOST;

    uint64_t launch_addr = hal.get_dev_addr(dispatch_core_type, tt_metal::HalL1MemAddrType::LAUNCH);
    uint64_t go_addr = hal.get_dev_addr(dispatch_core_type, tt_metal::HalL1MemAddrType::GO_MSG);

    cluster.write_core_immediate(msg.data(), msg.size(), {static_cast<size_t>(chip), core}, launch_addr);
    tt_driver_atomics::sfence();
    if (send_go) {
        cluster.write_core_immediate(go_msg.data(), go_msg.size(), {static_cast<size_t>(chip), core}, go_addr);
    }
}

ll_api::memory read_mem_from_core(
    tt::ChipId chip, const CoreCoord& core, const ll_api::memory& mem, uint64_t local_init_addr) {
    ll_api::memory read_mem;
    read_mem.fill_from_mem_template(mem, [&](std::vector<uint32_t>::iterator mem_ptr, uint64_t addr, uint32_t len) {
        uint64_t relo_addr = tt::tt_metal::MetalContext::instance().hal().relocate_dev_addr(addr, local_init_addr);
        tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            &*mem_ptr, len * sizeof(uint32_t), tt_cxy_pair(chip, core), relo_addr);
    });
    return read_mem;
}

bool test_load_write_read_risc_binary(
    const ll_api::memory& mem,
    tt::ChipId chip_id,
    const CoreCoord& core,
    uint32_t core_type_idx,
    uint32_t processor_class_idx,
    uint32_t processor_type_idx) {
    TT_ASSERT(
        tt::tt_metal::MetalContext::instance().get_cluster().is_worker_core(core, chip_id) or
        tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_core(core, chip_id));

    uint64_t local_init_addr = tt::tt_metal::MetalContext::instance()
                                   .hal()
                                   .get_jit_build_config(core_type_idx, processor_class_idx, processor_type_idx)
                                   .local_init_addr;

    auto core_type = get_core_type(chip_id, core);
    // Depending on the arch, active ethernet may be shared local memory with the base firmware
    // Primary risc is shared
    // TODO: Move this query into the HAL
    bool local_mem_offset = processor_type_idx == 0 && core_type == tt_metal::HalProgrammableCoreType::ACTIVE_ETH;
    log_debug(tt::LogLLRuntime, "hex_vec size = {}, size_in_bytes = {}", mem.size(), mem.size() * sizeof(uint32_t));
    mem.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t addr, uint32_t len_words) {
        uint64_t relo_addr =
            tt::tt_metal::MetalContext::instance().hal().relocate_dev_addr(addr, local_init_addr, local_mem_offset);

        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            &*mem_ptr, len_words * sizeof(uint32_t), tt_cxy_pair(chip_id, core), relo_addr);
    });

    log_debug(tt::LogLLRuntime, "wrote hex to core {}", core.str().c_str());

    if (std::getenv("TT_METAL_KERNEL_READBACK_ENABLE") != nullptr) {
        tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(chip_id);
        ll_api::memory read_mem = read_mem_from_core(chip_id, core, mem, local_init_addr);
        log_debug(tt::LogLLRuntime, "read hex back from the core");
        return mem == read_mem;
    }

    return true;
}

bool test_load_multicast_write_risc_binary(
    const ll_api::memory& mem,
    tt::ChipId chip_id,
    const CoreCoord& start_core,
    const CoreCoord& end_core,
    uint32_t core_type_idx,
    uint32_t processor_class_idx,
    uint32_t processor_type_idx) {
    TT_ASSERT(
        tt::tt_metal::MetalContext::instance().get_cluster().is_worker_core(start_core, chip_id) and
        tt::tt_metal::MetalContext::instance().get_cluster().is_worker_core(end_core, chip_id));

    uint64_t local_init_addr = tt::tt_metal::MetalContext::instance()
                                   .hal()
                                   .get_jit_build_config(core_type_idx, processor_class_idx, processor_type_idx)
                                   .local_init_addr;

    log_debug(tt::LogLLRuntime, "hex_vec size = {}, size_in_bytes = {}", mem.size(), mem.size() * sizeof(uint32_t));
    mem.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t addr, uint32_t len_words) {
        uint64_t relo_addr =
            tt::tt_metal::MetalContext::instance().hal().relocate_dev_addr(addr, local_init_addr, false);

        tt::tt_metal::MetalContext::instance().get_cluster().noc_multicast_write(
            &*mem_ptr, len_words * sizeof(uint32_t), chip_id, start_core, end_core, relo_addr);
    });

    log_debug(tt::LogLLRuntime, "multicast hex to cores {} - {}", start_core.str().c_str(), end_core.str().c_str());

    if (std::getenv("TT_METAL_KERNEL_READBACK_ENABLE")) {
        log_info(tt::LogLLRuntime, "WARNING: Readback after multicast write is not yet supported");
    }

    return true;
}

void write_binary_to_address(const ll_api::memory& mem, tt::ChipId chip_id, const CoreCoord& core, uint32_t address) {
    log_debug(tt::LogLLRuntime, "vec size = {}, size_in_bytes = {}", mem.size(), mem.size() * sizeof(uint32_t));
    mem.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t /*addr*/, uint32_t len_words) {
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            &*mem_ptr, len_words * sizeof(uint32_t), tt_cxy_pair(chip_id, core), address);
    });
}

namespace internal_ {

bool is_active_eth_core(tt::ChipId chip_id, const CoreCoord& core) {
    auto active_eth_cores =
        tt::tt_metal::MetalContext::instance().get_control_plane().get_active_ethernet_cores(chip_id);
    return active_eth_cores.contains(logical_core_from_ethernet_core(chip_id, core));
}

namespace {

bool check_if_riscs_on_specified_core_done(tt::ChipId chip_id, const CoreCoord& core, int run_state) {
    tt_metal::HalProgrammableCoreType dispatch_core_type = get_core_type(chip_id, core);
    const auto& hal = tt_metal::MetalContext::instance().hal();
    auto dev_msgs_factory = hal.get_dev_msgs_factory(dispatch_core_type);

    uint64_t go_msg_addr = hal.get_dev_addr(dispatch_core_type, tt_metal::HalL1MemAddrType::GO_MSG);

    auto get_mailbox_is_done = [&](uint64_t go_msg_addr) {
        auto core_status = dev_msgs_factory.create<tt_metal::dev_msgs::go_msg_t>();
        tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            core_status.data(), core_status.size(), {static_cast<size_t>(chip_id), core}, go_msg_addr & ~0x3);
        uint8_t run = core_status.view().signal();
        if (run != run_state && run != tt_metal::dev_msgs::RUN_MSG_DONE) {
            fprintf(
                stderr,
                "Read unexpected run_mailbox value: 0x%x (expected 0x%x or 0x%x)\n",
                run,
                run_state,
                tt_metal::dev_msgs::RUN_MSG_DONE);
            TT_FATAL(
                run == run_state || run == tt_metal::dev_msgs::RUN_MSG_DONE,
                "Read unexpected run_mailbox value from core {}",
                core.str());
        }

        return run == tt_metal::dev_msgs::RUN_MSG_DONE;
    };
    return get_mailbox_is_done(go_msg_addr);
}

}  // namespace

void wait_until_cores_done(
    tt::ChipId device_id, int run_state, std::unordered_set<CoreCoord>& not_done_phys_cores, int timeout_ms) {
    // poll the cores until the set of not done cores is empty
    [[maybe_unused]] int loop_count = 1;
    auto start = std::chrono::high_resolution_clock::now();
    const auto& rtoptions = tt_metal::MetalContext::instance().rtoptions();
    bool is_simulator = rtoptions.get_simulator_enabled();
    // For simulators, always disable timeout (infinite wait). For non-simulators, a 0
    // timeout means: use the configured timeout for operations.
    if (is_simulator) {
        timeout_ms = 0;
    } else if (timeout_ms == 0) {
        timeout_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(rtoptions.get_timeout_duration_for_operations())
                .count();
    }
    while (!not_done_phys_cores.empty()) {
        if (timeout_ms > 0) {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            if (elapsed > timeout_ms) {
                for (const auto& core : not_done_phys_cores) {
                    if (internal_::is_active_eth_core(device_id, core)) {
                        print_aerisc_training_status(device_id, core);
                    }
                }
                std::string cores = fmt::format("{}", fmt::join(not_done_phys_cores, ", "));

                tt::tt_metal::MetalContext::instance().on_dispatch_timeout_detected();

                TT_THROW(
                    "Device {}: Timeout ({} ms) waiting for physical cores to finish: {}.",
                    device_id,
                    timeout_ms,
                    cores);
            }
        }

#ifdef DEBUG
        // Print not-done cores
        if (loop_count % 1000 == 0) {
            log_debug(
                tt::LogMetal, "Device {}: Not done phys cores: {}", device_id, fmt::join(not_done_phys_cores, " "));
            usleep(100000);
        }
#endif

        for (auto it = not_done_phys_cores.begin(); it != not_done_phys_cores.end();) {
            const auto& phys_core = *it;

            bool is_done = llrt::internal_::check_if_riscs_on_specified_core_done(device_id, phys_core, run_state);
            if (is_done) {
                log_debug(tt::LogMetal, "Device {}: Phys cores just done: {}", device_id, phys_core.str());
                it = not_done_phys_cores.erase(it);
            } else {
                ++it;
            }
        }
        loop_count++;

        // Continuously polling cores here can cause other host-driven noc transactions (dprint, watcher) to drastically
        // slow down for remote devices. So when debugging with these features, add a small delay to allow other
        // host-driven transactions through.
        if (rtoptions.get_watcher_enabled() || rtoptions.get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
}

void send_msg_to_eth_mailbox(
    tt::ChipId device_id,
    const CoreCoord& virtual_core,
    tt_metal::FWMailboxMsg msg_type,
    int mailbox_index,
    std::vector<uint32_t> args,
    bool wait_for_ack,
    int timeout_ms) {
    constexpr auto k_sleep_time = std::chrono::nanoseconds{5};
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    if (!hal.get_dispatch_feature_enabled(tt::tt_metal::DispatchFeature::ETH_MAILBOX_API)) {
        TT_THROW("Ethernet mailbox API not supported on device {}", device_id);
    }

    bool is_eth_core = internal_::is_active_eth_core(device_id, virtual_core);
    TT_ASSERT(
        is_eth_core,
        "target core for send_msg_to_eth_mailbox {} (virtual) must be an active ethernet core",
        virtual_core.str());

    const auto max_args = hal.get_eth_fw_mailbox_arg_count();
    const auto mailbox_addr = hal.get_eth_fw_mailbox_address(mailbox_index);
    const auto status_mask = hal.get_eth_fw_mailbox_val(tt_metal::FWMailboxMsg::ETH_MSG_STATUS_MASK);
    const auto call = hal.get_eth_fw_mailbox_val(tt_metal::FWMailboxMsg::ETH_MSG_CALL);
    const auto done_message = hal.get_eth_fw_mailbox_val(tt_metal::FWMailboxMsg::ETH_MSG_DONE);

    // Check mailbox is empty/ready
    uint32_t msg_status = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                              device_id, virtual_core, mailbox_addr, sizeof(uint32_t))[0] &
                          status_mask;
    {
        const auto start_time = std::chrono::steady_clock::now();
        while (msg_status != done_message && msg_status != 0) {
            uint32_t mailbox_val = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                device_id, virtual_core, mailbox_addr, sizeof(uint32_t))[0];
            msg_status = mailbox_val & status_mask;
            const auto timenow = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(timenow - start_time).count();
            if (elapsed > timeout_ms) {
                log_debug(
                    tt::LogMetal,
                    "Device {}: Timed out while waiting for ack when trying to launch Metal ethernet firmware on "
                    "ethernet core {}. Last message status: {:#x}",
                    device_id,
                    virtual_core.str(),
                    mailbox_val);

                TT_THROW("Device {} Firmware update is required. Minimum tt-firmware verison is 18.10.0", device_id);
            }
            std::this_thread::sleep_for(k_sleep_time);
        }
    }

    // Must write args first.
    // Pad remaining args to zero
    args.resize(max_args, 0);
    uint32_t first_arg_addr = hal.get_eth_fw_mailbox_arg_addr(mailbox_index, 0);
    tt::tt_metal::MetalContext::instance().get_cluster().write_reg(
        args.data(), tt_cxy_pair(device_id, virtual_core), first_arg_addr);
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device_id);

    const auto msg_val = hal.get_eth_fw_mailbox_val(msg_type);
    const uint32_t msg = call | msg_val;
    log_debug(
        tt::LogLLRuntime,
        "Device {}: Eth {} Mailbox {:#x} Command {:#x}, {}",
        device_id,
        virtual_core.str(),
        mailbox_addr,
        msg,
        fmt::join(args, ", "));
    tt::tt_metal::MetalContext::instance().get_cluster().write_reg(
        std::vector<uint32_t>{msg}.data(), tt_cxy_pair(device_id, virtual_core), mailbox_addr);
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device_id);

    // Wait for ack
    tt_cxy_pair target{static_cast<size_t>(device_id), virtual_core};
    if (wait_for_ack) {
        const auto start_time = std::chrono::steady_clock::now();
        do {
            uint32_t mailbox_val = 0;
            tt::tt_metal::MetalContext::instance().get_cluster().read_reg(&mailbox_val, target, mailbox_addr);
            msg_status = mailbox_val & status_mask;
            const auto timenow = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(timenow - start_time).count();
            if (elapsed > timeout_ms) {
                log_debug(
                    tt::LogMetal,
                    "Device {}: Timed out while waiting for ack when trying to launch Metal ethernet firmware on "
                    "ethernet core {}. Last message status: {:#x}",
                    device_id,
                    virtual_core.str(),
                    mailbox_val);

                TT_THROW("Device {} Firmware update is required. Minimum tt-firmware verison is 18.10.0", device_id);
            }
            std::this_thread::sleep_for(k_sleep_time);
        } while (msg_status != done_message);
    }
}

void return_to_base_firmware_and_wait_for_heartbeat(
    tt::ChipId device_id, const CoreCoord& virtual_core, int timeout_ms) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    if (!hal.get_dispatch_feature_enabled(tt::tt_metal::DispatchFeature::ETH_MAILBOX_API)) {
        TT_THROW("Ethernet mailbox API not supported on device {}", device_id);
    }

    tt_cxy_pair target{static_cast<size_t>(device_id), virtual_core};
    const auto heartbeat_addr = hal.get_eth_fw_mailbox_val(tt_metal::FWMailboxMsg::HEARTBEAT);

    uint32_t heartbeat_val = 0;
    tt::tt_metal::MetalContext::instance().get_cluster().read_reg(&heartbeat_val, target, heartbeat_addr);

    constexpr auto k_sleep_time = std::chrono::nanoseconds{5};
    std::this_thread::sleep_for(k_sleep_time);

    uint32_t previous_heartbeat_val = 0;
    tt::tt_metal::MetalContext::instance().get_cluster().read_reg(&previous_heartbeat_val, target, heartbeat_addr);

    const auto start = std::chrono::steady_clock::now();

    // Below steps can be skipped if we already have a heartbeat from the base firmware
    while (heartbeat_val == previous_heartbeat_val) {
        std::this_thread::sleep_for(k_sleep_time);
        // Try sending the stop message again
        tt::llrt::internal_::set_metal_eth_fw_run_flag(device_id, virtual_core, false);
        previous_heartbeat_val = heartbeat_val;
        tt::tt_metal::MetalContext::instance().get_cluster().read_reg(&heartbeat_val, target, heartbeat_addr);
        if (timeout_ms > 0) {
            const auto now = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            if (elapsed > timeout_ms) {
                print_aerisc_training_status(device_id, virtual_core);
                TT_THROW(
                    "Device {}: Timed out while waiting for active ethernet core {} to become active again. "
                    "Try resetting the board. Minimum tt-firmware version is 18.10.0",
                    device_id,
                    virtual_core.str());
            }
        }
    }
}

void set_metal_eth_fw_run_flag(tt::ChipId device_id, const CoreCoord& virtual_core, bool enable) {
    constexpr auto k_CoreType = tt_metal::HalProgrammableCoreType::ACTIVE_ETH;
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    if (!hal.get_dispatch_feature_enabled(tt::tt_metal::DispatchFeature::ETH_MAILBOX_API)) {
        TT_THROW("Ethernet mailbox API not supported on device {}", device_id);
    }
    tt::tt_metal::DeviceAddr mailbox_addr = hal.get_dev_addr(k_CoreType, tt::tt_metal::HalL1MemAddrType::MAILBOX);
    tt::tt_metal::DeviceAddr run_flag_addr =
        mailbox_addr + hal.get_dev_msgs_factory(k_CoreType)
                           .offset_of<tt::tt_metal::dev_msgs::mailboxes_t>(
                               tt::tt_metal::dev_msgs::mailboxes_t::Field::aerisc_run_flag);
    std::vector<uint32_t> en = {enable};
    tt::tt_metal::MetalContext::instance().get_cluster().write_reg(
        en.data(), tt_cxy_pair(device_id, virtual_core), run_flag_addr);
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device_id);
}

}  // namespace internal_

}  // namespace tt::llrt
