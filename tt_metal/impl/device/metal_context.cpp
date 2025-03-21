// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/metal_context.hpp>
#include <tt-metalium/dispatch_settings.hpp>

#include "tt_metal/impl/dispatch/topology.hpp"
#include "tt_metal/impl/dispatch/dispatch_query_manager.hpp"
#include "tt_metal/impl/debug/dprint_server.hpp"
#include "tt_metal/impl/debug/noc_logging.hpp"
#include "tt_metal/impl/debug/watcher_server.hpp"
#include "tt_metal/jit_build/build_env_manager.hpp"
#include "tt_metal/llrt/llrt.hpp"

namespace tt::tt_metal {

void clear_l1_state(chip_id_t device_id) {
    log_debug(tt::LogMetal, "Clearing L1 for device {}", device_id);
    // Clear all clearable Tensix and Eth L1
    CoreCoord logical_grid_size = tt::Cluster::instance().get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    uint32_t l1_size_per_core = tt::Cluster::instance().get_soc_desc(device_id).worker_l1_size;
    TT_ASSERT(l1_size_per_core % sizeof(uint32_t) == 0);
    std::vector<uint32_t> zero_vec(l1_size_per_core / sizeof(uint32_t), 0);
    constexpr uint32_t start_address = 0;
    for (uint32_t x = 0; x < logical_grid_size.x; x++) {
        for (uint32_t y = 0; y < logical_grid_size.y; y++) {
            CoreCoord logical_core(x, y);
            auto virtual_core = tt::Cluster::instance().get_virtual_coordinate_from_logical_coordinates(
                device_id, logical_core, CoreType::WORKER);
            tt::llrt::write_hex_vec_to_core(device_id, virtual_core, zero_vec, start_address);
        }
    }

    // These L1 ranges are restricted becase UMD base routing FW uses L1 below FIRMWARE_BASE and
    // between TILE_HEADER_BUFFER_BASE to COMMAND_Q_BASE
    // Clear erisc sync info
    for (const auto& eth_core : tt::Cluster::instance().get_active_ethernet_cores(device_id)) {
        static const uint32_t max_l1_loading_size =
            hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED) +
            hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);

        static uint32_t zero_vec_size = max_l1_loading_size;
        auto zero_vec_addr = HalL1MemAddrType::UNRESERVED;
        if (hal.get_eth_fw_is_cooperative()) {
            zero_vec_size -=
                hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::TILE_HEADER_BUFFER);
            zero_vec_addr = HalL1MemAddrType::TILE_HEADER_BUFFER;
        }

        static std::vector<uint32_t> zero_vec(zero_vec_size / sizeof(uint32_t), 0);

        CoreCoord virtual_core =
            tt::Cluster::instance().get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);
        tt::llrt::write_hex_vec_to_core(
            device_id, virtual_core, zero_vec, hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, zero_vec_addr));
    }
    // TODO: clear idle eriscs as well
    tt::Cluster::instance().l1_barrier(device_id);
}

// TODO: Rename? This doesn't initialize a cluster
void initialize_cluster(chip_id_t device_id) {
    ZoneScoped;
    if (llrt::RunTimeOptions::get_instance().get_clear_l1()) {
        clear_l1_state(device_id);
    }
    int ai_clk = tt::Cluster::instance().get_device_aiclk(device_id);
    log_info(tt::LogMetal, "AI CLK for device {} is:   {} MHz", device_id, ai_clk);
}

void reset_cores(chip_id_t device_id) {
    ZoneScoped;

    auto erisc_app_still_running = [device_id](CoreCoord virtual_core) {
        // Check if the kernel/erisc_app is still running on a ethernet core with context switching enabled
        // The LAUNCH_ERISC_APP_FLAG is reset to 0 after reset/reboot, and set to 1 when Metal runtime launches erisc
        // app FW Only applicable to WORMHOLE ethernet cores today, but could in theory extend to other cores, remove
        // assert if so
        TT_ASSERT(
            (tt::Cluster::instance().arch() == ARCH::WORMHOLE_B0) and
                (tt::Cluster::instance().is_ethernet_core(virtual_core, device_id)),
            "Invalid core type for context switch check");
        auto core_type_idx = hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH);
        std::uint32_t launch_erisc_addr = tt::tt_metal::hal.get_jit_build_config(core_type_idx, 0, 0).fw_launch_addr;
        auto data = tt::llrt::read_hex_vec_from_core(device_id, virtual_core, launch_erisc_addr, sizeof(std::uint32_t));
        return (data[0] != 0);
    };

    // Check dispatch kernels for kernels that were left running previously, and send them an early exit signal.
    auto dispatch_cores = get_logical_dispatch_cores(device_id);
    // Assert worker cores + dispatch cores, in case they were in a bad state from before.
    std::unordered_set<CoreCoord> early_exit_cores;
    CoreType dispatch_core_type = tt::tt_metal::dispatch_core_manager::instance().get_dispatch_core_type();
    go_msg_t go_msg = {0};
    for (const auto& logical_core : dispatch_cores) {
        // Only need to manually exit active ethernet cores, since we reset all tensix + idle eth cores below.
        if (logical_core.type != CoreType::ETH) {
            continue;
        }

        bool is_active_eth =
            (tt::Cluster::instance().get_active_ethernet_cores(device_id).count(logical_core.coord) != 0);
        if (!is_active_eth) {
            continue;
        }

        CoreCoord virtual_core = tt::Cluster::instance().get_virtual_coordinate_from_logical_coordinates(
            device_id, logical_core.coord, logical_core.type);
        if (erisc_app_still_running(virtual_core)) {
            // Detected a still-running kernel on an ethernet dispatch core, signal it to exit.
            log_info(
                tt::LogMetal,
                "While initializing Device {}, ethernet dispatch core {} detected as still running, "
                "issuing exit signal.",
                device_id,
                virtual_core.str());
            DeviceAddr launch_addr = hal.get_dev_addr(
                is_active_eth ? HalProgrammableCoreType::ACTIVE_ETH : HalProgrammableCoreType::ACTIVE_ETH,
                HalL1MemAddrType::LAUNCH);

            std::vector<uint32_t> data(sizeof(launch_msg_t) / sizeof(uint32_t));
            data = tt::llrt::read_hex_vec_from_core(device_id, virtual_core, launch_addr, sizeof(launch_msg_t));
            launch_msg_t* launch_msg = (launch_msg_t*)(&data[0]);
            launch_msg->kernel_config.exit_erisc_kernel = 1;
            llrt::write_launch_msg_to_core(device_id, virtual_core, launch_msg, &go_msg, launch_addr, false);
            early_exit_cores.insert(virtual_core);  // Track cores we sent an early exit signal to verify later.
        }
    }

    // Early exiting dispatch cores should show RUN_MSG_DONE when they exit.
    const int timeout_ms = 10000;  // 10 seconds for now
    try {
        llrt::internal_::wait_until_cores_done(device_id, RUN_MSG_GO, early_exit_cores, timeout_ms);
    } catch (std::runtime_error& e) {
        log_warning(
            "Detected dispatch kernels still running but failed to complete an early exit. This may happen "
            "from time to time following a reset, continuing to FW intialization...");
    }

    // Ignore storage-only cores
    std::unordered_set<CoreCoord> storage_only_cores;
    uint8_t num_hw_cqs = tt::tt_metal::dispatch_core_manager::instance().get_num_hw_cqs();
    DispatchCoreConfig dispatch_core_config =
        tt::tt_metal::dispatch_core_manager::instance().get_dispatch_core_config();
    for (auto core_coord : tt::get_logical_storage_cores(device_id, num_hw_cqs, dispatch_core_config)) {
        storage_only_cores.insert(core_coord);
    }

    // Reset Tensix cores
    // TODO: reset BH eth cores as well
    CoreCoord grid_size = tt::Cluster::instance().get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core = tt::Cluster::instance().get_virtual_coordinate_from_logical_coordinates(
                device_id, logical_core, CoreType::WORKER);

            if (storage_only_cores.find(logical_core) == storage_only_cores.end()) {
                tt::Cluster::instance().assert_risc_reset_at_core(tt_cxy_pair(device_id, worker_core));
            }
        }
    }

    // Reset idle ethernet cores
    for (CoreCoord logical_core : tt::Cluster::instance().get_inactive_ethernet_cores(device_id)) {
        CoreCoord virtual_core = tt::Cluster::instance().get_virtual_coordinate_from_logical_coordinates(
            device_id, logical_core, CoreType::ETH);
        tt::Cluster::instance().assert_risc_reset_at_core(tt_cxy_pair(device_id, virtual_core));
    }
}

MetalContext* MetalContext::_inst = nullptr;

void MetalContext::initialize(const DispatchCoreConfig& dispatch_core_config, uint8_t num_hw_cqs) noexcept {
    log_debug(tt::LogMetal, "DevicePool initialize");
    if (_inst == nullptr) {
        static MetalContext MetalContext(dispatch_core_config, num_hw_cqs);
        _inst = &MetalContext;
    } else if (_inst->dispatch_core_config_ != dispatch_core_config or num_hw_cqs != _inst->num_hw_cqs_) {
        TT_THROW("No re-init allowed.");
    }
}

MetalContext& MetalContext::instance() {
    TT_ASSERT(MetalContext::_inst != nullptr, "Trying to get MetalContext without initializing it");
    return *MetalContext::_inst;
}

MetalContext::MetalContext(const DispatchCoreConfig& dispatch_core_config, uint8_t num_hw_cqs) {
    log_warning("MetalContext Init!");
    dispatch_core_config_ = dispatch_core_config;
    num_hw_cqs_ = num_hw_cqs;

    // Initialize dispatch core manager, query manager, settings. TODO: could these all be pulled under this context?
    tt::tt_metal::dispatch_core_manager::initialize(dispatch_core_config, num_hw_cqs);
    tt_metal::DispatchQueryManager::initialize(num_hw_cqs);
    tt_metal::DispatchSettings::initialize(tt::Cluster::instance());

    auto all_devices = tt::Cluster::instance().all_chip_ids();
    for (chip_id_t device_id : all_devices) {
        initialize_cluster(device_id);

        // Create build env for this device, and build FW if it's not built already
        BuildEnvManager::get_instance().add_build_env(device_id, num_hw_cqs_);
        uint32_t fw_build_key = BuildEnvManager::get_instance().get_device_build_env(device_id).build_key;
        if (!firmware_built_keys_.contains(fw_build_key)) {
            BuildEnvManager::get_instance().build_firmware(device_id);
            firmware_built_keys_.insert(fw_build_key);
        }
    }

    // Populate FD topology across all devices
    if (tt::llrt::RunTimeOptions::get_instance().get_fast_dispatch()) {
        std::set<chip_id_t> all_devices_set(all_devices.begin(), all_devices.end());
        populate_fd_kernels(all_devices_set, num_hw_cqs_);
    }

    // Initialize debug tools, reset cores, init FW
    for (chip_id_t device_id : all_devices) {
        ClearNocData(device_id);
        DprintServerAttach(device_id);
        watcher_init(device_id);

        // TODO: as optimization, investigate removing all this call for already initialized devivces
        if (!llrt::RunTimeOptions::get_instance().get_skip_reset_cores_on_init()) {
            reset_cores(device_id);
        }

        // Watcher needs to init before FW since FW needs watcher mailboxes to be set up, and needs to attach after FW
        // starts since it also writes to watcher mailboxes.
    }
}

MetalContext::~MetalContext() { log_warning("MetalContext Teardown!"); }

}  // namespace tt::tt_metal
