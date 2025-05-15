// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "metal_context.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "tt_metal/impl/allocator/l1_banking_allocator.hpp"
#include "tt_metal/impl/dispatch/topology.hpp"
#include "tt_metal/impl/debug/dprint_server.hpp"
#include "tt_metal/impl/debug/noc_logging.hpp"
#include "tt_metal/impl/debug/watcher_server.hpp"
#include "tt_metal/impl/debug/debug_helpers.hpp"
#include "tt_metal/jit_build/build_env_manager.hpp"
#include "tt_metal/llrt/llrt.hpp"
#include "tt_metal/llrt/get_platform_architecture.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_metal.hpp>

namespace tt::tt_metal {

void MetalContext::initialize(
    const DispatchCoreConfig& dispatch_core_config,
    uint8_t num_hw_cqs,
    const BankMapping& l1_bank_remap,
    bool minimal) {
    // Settings that affect FW build can also trigger a re-initialization
    auto fw_compile_hash = std::hash<std::string>{}(rtoptions_.get_compile_hash_string());
    if (initialized_) {
        if (this->dispatch_core_config_ != dispatch_core_config or num_hw_cqs != this->num_hw_cqs_ or
            l1_bank_remap != this->l1_bank_remap_ or fw_compile_hash != this->fw_compile_hash_) {
            log_warning("Closing and re-initializing MetalContext with new parameters.");
            teardown();
        } else {
            // Re-init request with the same parameters, do nothing
            return;
        }
    }

    initialized_ = true;
    dispatch_core_config_ = dispatch_core_config;
    num_hw_cqs_ = num_hw_cqs;
    l1_bank_remap_ = l1_bank_remap;
    fw_compile_hash_ = fw_compile_hash;

    // Initialize dispatch state
    dispatch_core_manager_ = std::make_unique<dispatch_core_manager>(dispatch_core_config, num_hw_cqs);
    dispatch_query_manager_ = std::make_unique<DispatchQueryManager>(num_hw_cqs);
    // Need DispatchMemMap for both dispatch core types
    tt_metal::DispatchSettings::initialize(*cluster_);
    dispatch_mem_map_[magic_enum::enum_integer(CoreType::WORKER)] =
        std::make_unique<DispatchMemMap>(CoreType::WORKER, num_hw_cqs);
    dispatch_mem_map_[magic_enum::enum_integer(CoreType::ETH)] =
        std::make_unique<DispatchMemMap>(CoreType::ETH, num_hw_cqs);

    // Minimal setup, don't initialize FW/Dispatch/etc.
    if (minimal) {
        return;
    }

    // TODO: Move FW, fabric, dispatch init here
    auto all_devices = cluster_->all_chip_ids();
    for (chip_id_t device_id : all_devices) {
        // Clear L1/DRAM if requested
        if (rtoptions_.get_clear_l1()) {
            clear_l1_state(device_id);
        }
        if (rtoptions_.get_clear_dram()) {
            clear_dram_state(device_id);
        }
        int ai_clk = cluster_->get_device_aiclk(device_id);
        log_info(tt::LogMetal, "AI CLK for device {} is:   {} MHz", device_id, ai_clk);

        // Create build env for this device, and build FW if it's not built already
        BuildEnvManager::get_instance().add_build_env(device_id, num_hw_cqs_);
        uint32_t fw_build_key = BuildEnvManager::get_instance().get_device_build_env(device_id).build_key;
        if (!firmware_built_keys_.contains(fw_build_key)) {
            BuildEnvManager::get_instance().build_firmware(device_id);
            firmware_built_keys_.insert(fw_build_key);
        }

        // Clear the entire launch message ring buffer on ethernet cores before application firmware is activated.
        // This is required since ethernet cores context switch between application and routing firmware.
        // If ERISC application firmware is activated before the launch messages are cleared, it can enter an undefined
        // state by reading a corrupted launch message. Routing firmware will never run in this case, causing UMD issued
        // transactions to hang.
        // TODO: run this alongside FW init
        // clear_launch_messages_on_eth_cores(device_id);
    }

    // Populate FD topology across all devices
    if (rtoptions_.get_fast_dispatch()) {
        std::set<chip_id_t> all_devices_set(all_devices.begin(), all_devices.end());
        // TODO: enable this when dispatch init/teardown moves to MetalContext
        // populate_fd_kernels(all_devices_set, num_hw_cqs);
    }

    // Initialize debug tools, reset cores, init FW
    for (chip_id_t device_id : all_devices) {
        // Init debug tools
        ClearNocData(device_id);
        DprintServerAttach(device_id);
        watcher_init(device_id);

        // TODO: as optimization, investigate removing all this call for already initialized devivces
        if (!tt_metal::MetalContext::instance().rtoptions().get_skip_reset_cores_on_init()) {
            // reset_cores();
        }

        // initialize_and_launch_firmware(device_id);

        // Watcher needs to init before FW since FW needs watcher mailboxes to be set up, and needs to attach after FW
        // starts since it also writes to watcher mailboxes.
        // watcher_attach(device_id);
    }

    // Register teardown function, but only once.
    if (not teardown_registered_) {
        std::atexit([]() { MetalContext::instance().teardown(); });
        teardown_registered_ = true;
    }
}

void MetalContext::teardown() {
    initialized_ = false;

    auto all_devices = cluster_->all_chip_ids();
    for (chip_id_t device_id : all_devices) {
        DprintServerDetach(device_id);
        // watcher_detach(device_id);
        // Assert cores

        cluster_->l1_barrier(device_id);
    }

    for (auto& mem_map : dispatch_mem_map_) {
        if (mem_map) {
            mem_map.reset();
        }
    }
    dispatch_query_manager_.reset();
    dispatch_core_manager_.reset();
}

MetalContext& MetalContext::instance() {
    static tt::stl::Indestructible<MetalContext> inst;
    return inst.get();
}

MetalContext::MetalContext() {
    bool is_base_routing_fw_enabled =
        Cluster::is_base_routing_fw_enabled(Cluster::get_cluster_type_from_cluster_desc(rtoptions_));
    hal_ = std::make_unique<Hal>(get_platform_architecture(rtoptions_), is_base_routing_fw_enabled);
    cluster_ = std::make_unique<Cluster>(rtoptions_, *hal_);
}

MetalContext::~MetalContext() {
    cluster_.reset();
    hal_.reset();
}

llrt::RunTimeOptions& MetalContext::rtoptions() { return rtoptions_; }

Cluster& MetalContext::get_cluster() {
    TT_FATAL(cluster_, "Trying to get cluster before intializing it.");
    return *cluster_;
}

const llrt::RunTimeOptions& MetalContext::rtoptions() const { return rtoptions_; }

const Cluster& MetalContext::get_cluster() const {
    TT_FATAL(cluster_, "Trying to get cluster before intializing it.");
    return *cluster_;
}

const Hal& MetalContext::hal() const {
    TT_FATAL(hal_, "Trying to get hal before intializing it.");
    return *hal_;
}

dispatch_core_manager& MetalContext::get_dispatch_core_manager() {
    TT_FATAL(dispatch_core_manager_, "Trying to get dispatch_core_manager before intializing it.");
    return *dispatch_core_manager_;
}

DispatchQueryManager& MetalContext::get_dispatch_query_manager() {
    TT_FATAL(dispatch_query_manager_, "Trying to get dispatch_query_manager before intializing it.");
    return *dispatch_query_manager_;
}

const DispatchMemMap& MetalContext::dispatch_mem_map() const {
    return dispatch_mem_map(dispatch_core_config_.get_core_type());
}

const DispatchMemMap& MetalContext::dispatch_mem_map(const CoreType& core_type) const {
    auto& mem_map = dispatch_mem_map_[magic_enum::enum_integer(core_type)];
    TT_FATAL(mem_map, "Tried to get dispatch_mem_map for {} before intializing it.", core_type);
    return *mem_map;
}

void MetalContext::clear_l1_state(chip_id_t device_id) {
    log_debug(tt::LogMetal, "Clearing L1 for device {}", device_id);
    // Clear all clearable Tensix and Eth L1
    CoreCoord logical_grid_size = cluster_->get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    uint32_t l1_size_per_core = cluster_->get_soc_desc(device_id).worker_l1_size;
    TT_ASSERT(l1_size_per_core % sizeof(uint32_t) == 0);
    std::vector<uint32_t> zero_vec(l1_size_per_core / sizeof(uint32_t), 0);
    constexpr uint32_t start_address = 0;
    for (uint32_t x = 0; x < logical_grid_size.x; x++) {
        for (uint32_t y = 0; y < logical_grid_size.y; y++) {
            CoreCoord logical_core(x, y);
            auto virtual_core =
                cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::WORKER);
            tt::llrt::write_hex_vec_to_core(device_id, virtual_core, zero_vec, start_address);
        }
    }

    // Clear erisc unreserved L1
    for (const auto& eth_core : cluster_->get_active_ethernet_cores(device_id)) {
        static uint32_t zero_vec_size = tt::tt_metal::hal::get_erisc_l1_unreserved_size();
        auto zero_vec_addr = tt::tt_metal::hal::get_erisc_l1_unreserved_base();

        static std::vector<uint32_t> zero_vec(zero_vec_size / sizeof(uint32_t), 0);

        CoreCoord virtual_core =
            cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);
        llrt::write_hex_vec_to_core(device_id, virtual_core, zero_vec, zero_vec_addr);
    }
    // TODO: clear idle eriscs as well
    cluster_->l1_barrier(device_id);
}

void MetalContext::clear_dram_state(chip_id_t device_id) {
    log_debug(tt::LogMetal, "Clearing DRAM for device {}", device_id);

    auto dram_size_per_channel = cluster_->get_soc_desc(device_id).dram_view_size;
    auto num_dram_channels = cluster_->get_soc_desc(device_id).get_num_dram_views();
    TT_ASSERT(dram_size_per_channel % sizeof(uint32_t) == 0);
    constexpr uint32_t start_address = 0;
    std::vector<uint32_t> zero_vec(dram_size_per_channel / sizeof(uint32_t), 0);
    for (int channel = 0; channel < num_dram_channels; ++channel) {
        tt::tt_metal::MetalContext::instance().get_cluster().write_dram_vec(
            zero_vec, device_id, channel, start_address);

        cluster_->dram_barrier(device_id);
    }
}

void MetalContext::clear_launch_messages_on_eth_cores(chip_id_t device_id) {
    launch_msg_t launch_msg;
    go_msg_t go_msg;
    go_msg.signal = RUN_MSG_INIT;
    std::memset(&launch_msg, 0, sizeof(launch_msg_t));
    std::vector<launch_msg_t> init_launch_msg_data(launch_msg_buffer_num_entries, launch_msg);

    auto clear_ethernet_core = [&](const CoreCoord& logical_eth_core) {
        CoreCoord virtual_eth_core =
            cluster_->get_virtual_coordinate_from_logical_coordinates(device_id, logical_eth_core, CoreType::ETH);
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            init_launch_msg_data.data(),
            launch_msg_buffer_num_entries * sizeof(launch_msg_t),
            tt_cxy_pair(device_id, virtual_eth_core),
            hal_->get_dev_addr(get_programmable_core_type(virtual_eth_core, device_id), HalL1MemAddrType::LAUNCH));
        uint32_t go_addr =
            hal_->get_dev_addr(get_programmable_core_type(virtual_eth_core, device_id), HalL1MemAddrType::GO_MSG);
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            &go_msg, sizeof(go_msg_t), tt_cxy_pair(device_id, virtual_eth_core), go_addr);
    };

    for (const auto& eth_core : cluster_->get_active_ethernet_cores(device_id)) {
        clear_ethernet_core(eth_core);
    }
    for (const auto& eth_core : cluster_->get_inactive_ethernet_cores(device_id)) {
        clear_ethernet_core(eth_core);
    }
}

}  // namespace tt::tt_metal
