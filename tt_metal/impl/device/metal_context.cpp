// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/metal_context.hpp>
#include <tt-metalium/dispatch_settings.hpp>

#include "tt_metal/impl/allocator/l1_banking_allocator.hpp"
#include "tt_metal/impl/dispatch/topology.hpp"
#include "tt_metal/impl/dispatch/dispatch_query_manager.hpp"
#include "tt_metal/impl/debug/dprint_server.hpp"
#include "tt_metal/impl/debug/noc_logging.hpp"
#include "tt_metal/impl/debug/watcher_server.hpp"
#include "tt_metal/impl/debug/debug_helpers.hpp"
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

std::unordered_set<CoreCoord> get_storage_only_cores(chip_id_t device_id, uint8_t num_hw_cqs) {
    std::unordered_set<CoreCoord> storage_only_cores;
    DispatchCoreConfig dispatch_core_config =
        tt::tt_metal::dispatch_core_manager::instance().get_dispatch_core_config();
    for (auto core_coord : tt::get_logical_storage_cores(device_id, num_hw_cqs, dispatch_core_config)) {
        storage_only_cores.insert(core_coord);
    }
    return storage_only_cores;
}

void reset_cores(chip_id_t device_id, uint8_t num_hw_cqs) {
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
    auto storage_only_cores = get_storage_only_cores(device_id, num_hw_cqs);

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

// TODO: Move this to general utils?
std::vector<CoreCoord> worker_cores_from_logical_cores(
    chip_id_t device_id, const std::vector<CoreCoord>& logical_cores) {
    std::vector<CoreCoord> worker_cores(logical_cores.size());
    for (std::size_t idx = 0; idx < logical_cores.size(); idx++) {
        worker_cores[idx] = tt::Cluster::instance().get_virtual_coordinate_from_logical_coordinates(
            device_id, logical_cores[idx], CoreType::WORKER);
    }

    return worker_cores;
}

CoreCoord virtual_noc0_coordinate(chip_id_t device_id, uint8_t noc_index, CoreCoord coord) {
    const auto& grid_size = tt::Cluster::instance().get_soc_desc(device_id).grid_size;
    if (coord.x >= grid_size.x || coord.y >= grid_size.y) {
        // Coordinate already in virtual space: NOC0 and NOC1 are the same
        return coord;
    } else {
        // Coordinate in Physical NOC0 Space. Convert to Virtual.
        coord = tt::Cluster::instance().get_virtual_coordinate_from_physical_coordinates(device_id, coord);
        // Derive virtual coord in noc_index space.
        CoreCoord virtual_coord = {
            hal.noc_coordinate(noc_index, grid_size.x, coord.x), hal.noc_coordinate(noc_index, grid_size.y, coord.y)};
        return virtual_coord;
    }
}

void MetalContext::generate_device_bank_to_noc_tables(chip_id_t device_id) {
    // Create a dummy allocator to generate the bank/noc tables. Specifically, these depend on l1_bank_remap.
    auto config = L1BankingAllocator::generate_config(
        device_id,
        num_hw_cqs_,
        DEFAULT_L1_SMALL_SIZE,      // Not required for noc table gen
        DEFAULT_TRACE_REGION_SIZE,  // Not required for noc table gen
        l1_bank_remap_);
    const auto allocator = L1BankingAllocator(config);
    const metal_SocDescriptor& soc_d = tt::Cluster::instance().get_soc_desc(device_id);
    const size_t num_dram_banks = allocator.get_num_banks(BufferType::DRAM);
    std::vector<CoreCoord> dram_noc_coord_per_bank(num_dram_banks);
    dram_bank_offset_map_[device_id].clear();
    dram_bank_offset_map_[device_id].resize(num_dram_banks);
    for (unsigned bank_id = 0; bank_id < num_dram_banks; bank_id++) {
        dram_noc_coord_per_bank[bank_id] =
            soc_d.get_preferred_worker_core_for_dram_view(allocator.get_dram_channel_from_bank_id(bank_id));
        dram_bank_offset_map_[device_id][bank_id] = allocator.get_bank_offset(BufferType::DRAM, bank_id);
    }
    const size_t num_l1_banks = allocator.get_num_banks(BufferType::L1);
    std::vector<CoreCoord> l1_noc_coord_per_bank(num_l1_banks);
    l1_bank_offset_map_[device_id].clear();
    l1_bank_offset_map_[device_id].resize(num_l1_banks);
    for (unsigned bank_id = 0; bank_id < num_l1_banks; bank_id++) {
        l1_noc_coord_per_bank[bank_id] = tt::Cluster::instance().get_virtual_coordinate_from_logical_coordinates(
            device_id, allocator.get_logical_core_from_bank_id(bank_id), CoreType::WORKER);
        l1_bank_offset_map_[device_id][bank_id] = allocator.get_bank_offset(BufferType::L1, bank_id);
    }

    dram_bank_to_noc_xy_[device_id].clear();
    dram_bank_to_noc_xy_[device_id].reserve(tt::tt_metal::hal.get_num_nocs() * dram_noc_coord_per_bank.size());
    for (unsigned int noc = 0; noc < tt::tt_metal::hal.get_num_nocs(); noc++) {
        for (unsigned int bank_id = 0; bank_id < dram_noc_coord_per_bank.size(); bank_id++) {
            uint16_t noc_x =
                tt::tt_metal::hal.noc_coordinate(noc, soc_d.grid_size.x, dram_noc_coord_per_bank[bank_id].x);
            uint16_t noc_y =
                tt::tt_metal::hal.noc_coordinate(noc, soc_d.grid_size.y, dram_noc_coord_per_bank[bank_id].y);
            uint16_t xy = ((noc_y << tt::tt_metal::hal.get_noc_addr_node_id_bits()) | noc_x)
                          << tt::tt_metal::hal.get_noc_coord_reg_offset();
            dram_bank_to_noc_xy_[device_id].push_back(xy);
        }
    }

    l1_bank_to_noc_xy_[device_id].clear();
    l1_bank_to_noc_xy_[device_id].reserve(tt::tt_metal::hal.get_num_nocs() * l1_noc_coord_per_bank.size());
    for (unsigned int noc = 0; noc < tt::tt_metal::hal.get_num_nocs(); noc++) {
        for (unsigned int bank_id = 0; bank_id < l1_noc_coord_per_bank.size(); bank_id++) {
            auto l1_noc_coords = virtual_noc0_coordinate(device_id, noc, l1_noc_coord_per_bank[bank_id]);
            uint16_t noc_x = l1_noc_coords.x;
            uint16_t noc_y = l1_noc_coords.y;
            uint16_t xy = ((noc_y << tt::tt_metal::hal.get_noc_addr_node_id_bits()) | noc_x)
                          << tt::tt_metal::hal.get_noc_coord_reg_offset();
            l1_bank_to_noc_xy_[device_id].push_back(xy);
        }
    }
}

void MetalContext::initialize_device_bank_to_noc_tables(
    chip_id_t device_id, const HalProgrammableCoreType& core_type, CoreCoord virtual_core) {
    const uint32_t dram_to_noc_sz_in_bytes = dram_bank_to_noc_xy_[device_id].size() * sizeof(uint16_t);
    const uint32_t l1_to_noc_sz_in_bytes = l1_bank_to_noc_xy_[device_id].size() * sizeof(uint16_t);
    const uint32_t dram_offset_sz_in_bytes = dram_bank_offset_map_[device_id].size() * sizeof(int32_t);
    const uint32_t l1_offset_sz_in_bytes = l1_bank_offset_map_[device_id].size() * sizeof(int32_t);

    const uint64_t mem_bank_to_noc_addr = hal.get_dev_addr(core_type, HalL1MemAddrType::BANK_TO_NOC_SCRATCH);
    const uint32_t mem_bank_to_noc_size = hal.get_dev_size(core_type, HalL1MemAddrType::BANK_TO_NOC_SCRATCH);

    TT_ASSERT(
        (dram_to_noc_sz_in_bytes + l1_to_noc_sz_in_bytes + dram_offset_sz_in_bytes + l1_offset_sz_in_bytes) <=
            mem_bank_to_noc_size,
        "Size of bank_to_noc table is greater than available space");

    tt::Cluster::instance().write_core(
        &dram_bank_to_noc_xy_[device_id][0],
        dram_to_noc_sz_in_bytes,
        tt_cxy_pair(device_id, virtual_core),
        mem_bank_to_noc_addr);
    uint64_t l1_noc_addr = mem_bank_to_noc_addr + dram_to_noc_sz_in_bytes;
    tt::Cluster::instance().write_core(
        &l1_bank_to_noc_xy_[device_id][0], l1_to_noc_sz_in_bytes, tt_cxy_pair(device_id, virtual_core), l1_noc_addr);

    uint64_t dram_offset_addr = l1_noc_addr + l1_to_noc_sz_in_bytes;
    tt::Cluster::instance().write_core(
        &dram_bank_offset_map_[device_id][0],
        dram_offset_sz_in_bytes,
        tt_cxy_pair(device_id, virtual_core),
        dram_offset_addr);
    uint64_t l1_offset_addr = dram_offset_addr + dram_offset_sz_in_bytes;
    tt::Cluster::instance().write_core(
        &l1_bank_offset_map_[device_id][0],
        l1_offset_sz_in_bytes,
        tt_cxy_pair(device_id, virtual_core),
        l1_offset_addr);
}

void MetalContext::initialize_firmware(
    chip_id_t device_id,
    const HalProgrammableCoreType& core_type,
    CoreCoord virtual_core,
    launch_msg_t* launch_msg,
    go_msg_t* go_msg) {
    ZoneScoped;

    initialize_device_bank_to_noc_tables(device_id, core_type, virtual_core);
    uint32_t core_type_idx = hal.get_programmable_core_type_index(core_type);
    uint32_t processor_class_count = hal.get_processor_classes_count(core_type);
    auto jit_build_config =
        hal.get_jit_build_config(core_type_idx, 0, 0);  // Only the first risc needs to be programmed

    switch (core_type) {
        case HalProgrammableCoreType::TENSIX: {
            for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
                auto [build_idx, num_build_states] =
                    BuildEnvManager::get_instance().get_build_index_and_state_count(core_type_idx, processor_class);
                for (uint32_t riscv_id = 0; riscv_id < num_build_states; riscv_id++) {
                    auto fw_path = BuildEnvManager::get_instance()
                                       .get_firmware_build_state(device_id, core_type_idx, processor_class, riscv_id)
                                       .get_target_out_path("");
                    const ll_api::memory& binary_mem = llrt::get_risc_binary(fw_path);
                    uint32_t fw_size = binary_mem.get_text_size();
                    if (riscv_id + build_idx == 1) {  // TODO: clean up how brisc/ncrisc are handled
                        // In this context, ncrisc_kernel_size16 is the size of the fw
                        launch_msg->kernel_config.ncrisc_kernel_size16 = (fw_size + 15) >> 4;
                    }
                    log_debug(LogDevice, "RISC {} fw binary size: {} in bytes", riscv_id, fw_size);

                    if (not llrt::RunTimeOptions::get_instance().get_skip_loading_fw()) {
                        llrt::test_load_write_read_risc_binary(
                            binary_mem, device_id, virtual_core, core_type_idx, processor_class, riscv_id);
                    }
                }
            }

            if (!tt::llrt::RunTimeOptions::get_instance().get_fast_dispatch()) {
                // Host always writes launch messages
                launch_msg->kernel_config.mode = DISPATCH_MODE_HOST;
            } else {
                std::vector<CoreCoord> physical_dispatch_cores = {};
                if (dispatch_core_manager::instance().get_dispatch_core_type() == CoreType::WORKER) {
                    physical_dispatch_cores = worker_cores_from_logical_cores(
                        device_id, dispatch_core_manager::instance().get_all_logical_dispatch_cores(device_id));
                }
                if (std::find(physical_dispatch_cores.begin(), physical_dispatch_cores.end(), virtual_core) !=
                    physical_dispatch_cores.end()) {
                    // Dispatch cores - Host writes launch messages
                    launch_msg->kernel_config.mode = DISPATCH_MODE_HOST;
                } else {
                    // Worker cores - Dispatcher will write launch messages
                    launch_msg->kernel_config.mode = DISPATCH_MODE_DEV;
                }
            }

            break;
        }
        case HalProgrammableCoreType::ACTIVE_ETH:
        case HalProgrammableCoreType::IDLE_ETH: {
            bool is_idle_eth = core_type == HalProgrammableCoreType::IDLE_ETH;
            TensixSoftResetOptions reset_val = TENSIX_ASSERT_SOFT_RESET;
            if (not is_idle_eth) {
                reset_val =
                    reset_val & static_cast<TensixSoftResetOptions>(
                                    ~std::underlying_type<TensixSoftResetOptions>::type(TensixSoftResetOptions::BRISC));
            }
            if (is_idle_eth or !hal.get_eth_fw_is_cooperative()) {
                tt::Cluster::instance().assert_risc_reset_at_core(tt_cxy_pair(device_id, virtual_core), reset_val);
            }
            if (not llrt::RunTimeOptions::get_instance().get_skip_loading_fw()) {
                for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
                    auto num_build_states = hal.get_processor_types_count(core_type_idx, processor_class);
                    for (uint32_t eriscv_id = 0; eriscv_id < num_build_states; eriscv_id++) {
                        auto fw_path =
                            BuildEnvManager::get_instance()
                                .get_firmware_build_state(device_id, core_type_idx, processor_class, eriscv_id)
                                .get_target_out_path("");
                        const ll_api::memory& binary_mem = llrt::get_risc_binary(fw_path);
                        uint32_t fw_size = binary_mem.get_text_size();
                        log_debug(LogDevice, "ERISC fw binary size: {} in bytes", fw_size);
                        llrt::test_load_write_read_risc_binary(
                            binary_mem, device_id, virtual_core, core_type_idx, processor_class, eriscv_id);
                    }
                }
            }
            // Ethernet worker core. Launch messages will be sent by FD infra if it's enabled
            // Idle ethernet core. Used by FD infra. Host will write launch messages during init.
            launch_msg->kernel_config.mode =
                (!tt::llrt::RunTimeOptions::get_instance().get_fast_dispatch() or is_idle_eth) ? DISPATCH_MODE_HOST
                                                                                               : DISPATCH_MODE_DEV;
            break;
        }
        default:
            TT_THROW(
                "Unsupported programable core type {} to initialize build states", magic_enum::enum_name(core_type));
    }

    tt::Cluster::instance().write_core(
        &jit_build_config.fw_launch_addr_value,
        sizeof(uint32_t),
        tt_cxy_pair(device_id, virtual_core),
        jit_build_config.fw_launch_addr);

    // Initialize each entry in the launch_msg ring buffer with the correct dispatch mode - Cores that don't get a valid
    // launch_message during program execution need to at least have the correct dispatch mode.
    // When using Fast Dispatch on Tensix:
    // dispatch cores (Tensix) configured with DISPATCH_MODE_HOST
    // worker cores (Tensix and active eth) configured with DISPATCH_MODE_DEV
    // Idle Eth cores configured with DISPATCH_MODE_HOST but not used
    // When using Fast Dispatch on Idle Eth:
    // dispatch cores (Idle Eth) configured with DISPATCH_MODE_HOST
    // worker cores (Tensix and active eth) configured with DISPATCH_MODE_DEV
    // When using Slow Dispatch, all cores initialized with DISPATCH_MODE_HOST
    std::vector<launch_msg_t> init_launch_msg_data(launch_msg_buffer_num_entries, *launch_msg);
    tt::Cluster::instance().write_core(
        init_launch_msg_data.data(),
        launch_msg_buffer_num_entries * sizeof(launch_msg_t),
        tt_cxy_pair(device_id, virtual_core),
        hal.get_dev_addr(get_programmable_core_type(virtual_core, device_id), HalL1MemAddrType::LAUNCH));
    uint32_t go_addr = hal.get_dev_addr(get_programmable_core_type(virtual_core, device_id), HalL1MemAddrType::GO_MSG);
    tt::Cluster::instance().write_core(go_msg, sizeof(go_msg_t), tt_cxy_pair(device_id, virtual_core), go_addr);
    uint64_t launch_msg_buffer_read_ptr_addr = hal.get_dev_addr(
        get_programmable_core_type(virtual_core, device_id), HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR);
    uint32_t zero = 0;
    tt::Cluster::instance().write_core(
        &zero, sizeof(uint32_t), tt_cxy_pair(device_id, virtual_core), launch_msg_buffer_read_ptr_addr);
}

void MetalContext::initialize_and_launch_firmware(chip_id_t device_id) {
    ZoneScoped;

    launch_msg_t launch_msg;
    go_msg_t go_msg;
    std::memset(&launch_msg, 0, sizeof(launch_msg_t));
    go_msg.signal = RUN_MSG_INIT;

    // Populate core info, which will be written to device
    std::vector<uint32_t> core_info_vec(sizeof(core_info_msg_t) / sizeof(uint32_t));
    core_info_msg_t* core_info = (core_info_msg_t*)core_info_vec.data();

    const metal_SocDescriptor& soc_d = tt::Cluster::instance().get_soc_desc(device_id);
    uint64_t pcie_chan_base_addr = tt::Cluster::instance().get_pcie_base_addr_from_device(device_id);
    uint32_t num_host_channels = tt::Cluster::instance().get_num_host_channels(device_id);
    uint64_t pcie_chan_end_addr = pcie_chan_base_addr;
    for (int pcie_chan = 0; pcie_chan < num_host_channels; pcie_chan++) {
        pcie_chan_end_addr += tt::Cluster::instance().get_host_channel_size(device_id, pcie_chan);
    }
    core_info->noc_pcie_addr_base = pcie_chan_base_addr;
    core_info->noc_pcie_addr_end = pcie_chan_end_addr;
    core_info->noc_dram_addr_base = 0;
    core_info->noc_dram_addr_end = soc_d.dram_core_size;

    const std::vector<tt::umd::CoreCoord>& pcie_cores = soc_d.get_cores(CoreType::PCIE, soc_d.get_umd_coord_system());
    const std::vector<tt::umd::CoreCoord>& dram_cores = soc_d.get_cores(CoreType::DRAM, soc_d.get_umd_coord_system());
    const std::vector<tt::umd::CoreCoord>& eth_cores = soc_d.get_cores(CoreType::ETH, CoordSystem::PHYSICAL);
    // The SOC descriptor can list a dram core multiple times, depending on how GDDR is assigned to banks
    // Get a list of unique DRAM cores.
    std::unordered_set<CoreCoord> unique_dram_cores(dram_cores.begin(), dram_cores.end());
    TT_ASSERT(
        pcie_cores.size() + dram_cores.size() + eth_cores.size() <= MAX_NON_WORKER_CORES,
        "Detected more pcie/dram/eth cores than fit in the device mailbox.");
    TT_ASSERT(
        eth_cores.size() <= MAX_VIRTUAL_NON_WORKER_CORES,
        "Detected more eth cores (virtual non-workers) than can fit in device mailbox.");
    for (int idx = 0; idx < MAX_NON_WORKER_CORES; idx++) {
        core_info->non_worker_cores[idx] = {CORE_COORD_INVALID, CORE_COORD_INVALID, AddressableCoreType::UNKNOWN};
    }
    for (int idx = 0; idx < MAX_VIRTUAL_NON_WORKER_CORES; idx++) {
        core_info->virtual_non_worker_cores[idx] = {
            CORE_COORD_INVALID, CORE_COORD_INVALID, AddressableCoreType::UNKNOWN};
    }

    int non_worker_cores_idx = 0;
    for (const tt::umd::CoreCoord& core : pcie_cores) {
        core_info->non_worker_cores[non_worker_cores_idx++] = {core.x, core.y, AddressableCoreType::PCIE};
    }
    for (const tt::umd::CoreCoord& core : dram_cores) {
        core_info->non_worker_cores[non_worker_cores_idx++] = {core.x, core.y, AddressableCoreType::DRAM};
    }
    for (const tt::umd::CoreCoord& core : eth_cores) {
        core_info->non_worker_cores[non_worker_cores_idx++] = {core.x, core.y, AddressableCoreType::ETH};
    }
    if (hal.is_coordinate_virtualization_enabled()) {
        // Track Virtual Non Worker Cores (In this case only Eth) separately
        uint32_t virtual_non_worker_cores_idx = 0;
        for (const tt::umd::CoreCoord& core : eth_cores) {
            auto virtual_core =
                tt::Cluster::instance().get_virtual_coordinate_from_physical_coordinates(device_id, {core.x, core.y});
            core_info->virtual_non_worker_cores[virtual_non_worker_cores_idx++] = {
                virtual_core.x, virtual_core.y, AddressableCoreType::ETH};
        }
    }

    // Determine which noc-coords are harvested
    // TODO(PGK/Almeet): fix this w/ new UMD
    std::vector<uint32_t> harvested_rows;
    CoreCoord logical_grid_size = tt::Cluster::instance().get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    uint32_t harvested_noc_rows = CoordinateManager::shuffle_tensix_harvesting_mask_to_noc0_coords(
        tt::Cluster::instance().get_soc_desc(device_id).arch, tt::Cluster::instance().get_harvesting_mask(device_id));
    for (uint32_t y = 0; y < soc_d.grid_size.y; y++) {
        bool row_harvested = (harvested_noc_rows >> y) & 0x1;
        if (row_harvested) {
            harvested_rows.push_back(y);
        }
    }
    TT_ASSERT(harvested_rows.size() <= MAX_HARVESTED_ROWS, "Detected more harvested rows than fit in mailbox.");
    for (int idx = 0; idx < MAX_HARVESTED_ROWS; idx++) {
        core_info->harvested_y[idx] = (idx < harvested_rows.size()) ? harvested_rows[idx] : CORE_COORD_INVALID;
        // Populate harvested rows in virtual coordinate space if virtualization is supported by HW.
        // Harvested rows in the virtual space are placed at the end of the worker grid,
        if (hal.is_coordinate_virtualization_enabled() and idx < harvested_rows.size()) {
            core_info->virtual_harvested_y[idx] =
                (hal.get_virtual_worker_start_y() + logical_grid_size.y + harvested_rows.size() - (idx + 1));
        } else {
            core_info->virtual_harvested_y[idx] = CORE_COORD_INVALID;
        }
    }

    core_info->noc_size_x = soc_d.grid_size.x;
    core_info->noc_size_y = soc_d.grid_size.y;
    core_info->worker_grid_size_x = logical_grid_size.x;  // Grid size as virtual coords see it (workers only)
    core_info->worker_grid_size_y = logical_grid_size.y;

    // Download to worker cores
    log_debug("Initializing firmware");
    std::unordered_set<CoreCoord> not_done_cores;

    auto storage_only_cores = get_storage_only_cores(device_id, num_hw_cqs_);
    for (uint32_t y = 0; y < logical_grid_size.y; y++) {
        for (uint32_t x = 0; x < logical_grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            if (!storage_only_cores.count(logical_core)) {
                CoreCoord worker_core = tt::Cluster::instance().get_virtual_coordinate_from_logical_coordinates(
                    device_id, logical_core, CoreType::WORKER);
                // Setup the absolute logical coordinates of this worker which are relative to true origin. not the sub
                // device. When running the user kernel, which potentially is on a sub device, send that info using the
                // launch message using dispatch.
                core_info->absolute_logical_x = logical_core.x;
                core_info->absolute_logical_y = logical_core.y;
                // Must write to core before starting it
                tt::llrt::write_hex_vec_to_core(
                    device_id,
                    worker_core,
                    core_info_vec,
                    hal.get_dev_addr(get_programmable_core_type(worker_core, device_id), HalL1MemAddrType::CORE_INFO));
                initialize_firmware(device_id, HalProgrammableCoreType::TENSIX, worker_core, &launch_msg, &go_msg);
                not_done_cores.insert(worker_core);
            }
        }
    }

    // Clear erisc sync info
    for (const auto& eth_core : tt::Cluster::instance().get_active_ethernet_cores(device_id)) {
        static std::vector<uint32_t> zero_vec_erisc_init(
            hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::APP_SYNC_INFO) / sizeof(uint32_t),
            0);

        CoreCoord virtual_core =
            tt::Cluster::instance().get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);

        llrt::write_hex_vec_to_core(
            device_id,
            virtual_core,
            zero_vec_erisc_init,
            hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::APP_SYNC_INFO));
    }

    // Load erisc app base FW to eth cores on WH and active_erisc FW on second risc of BH active eth cores
    std::unordered_set<CoreCoord> active_eth_cores;
    for (const auto& eth_core : tt::Cluster::instance().get_active_ethernet_cores(device_id)) {
        CoreCoord virtual_core =
            tt::Cluster::instance().get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);
        core_info->absolute_logical_x = eth_core.x;
        core_info->absolute_logical_y = eth_core.y;
        tt::llrt::write_hex_vec_to_core(
            device_id,
            virtual_core,
            core_info_vec,
            hal.get_dev_addr(get_programmable_core_type(virtual_core, device_id), HalL1MemAddrType::CORE_INFO));
        initialize_firmware(device_id, HalProgrammableCoreType::ACTIVE_ETH, virtual_core, &launch_msg, &go_msg);
        if (!hal.get_eth_fw_is_cooperative()) {
            active_eth_cores.insert(virtual_core);
            not_done_cores.insert(virtual_core);
        }
    }

    for (const auto& eth_core : tt::Cluster::instance().get_inactive_ethernet_cores(device_id)) {
        CoreCoord virtual_core =
            tt::Cluster::instance().get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);
        core_info->absolute_logical_x = eth_core.x;
        core_info->absolute_logical_y = eth_core.y;
        tt::llrt::write_hex_vec_to_core(
            device_id,
            virtual_core,
            core_info_vec,
            hal.get_dev_addr(get_programmable_core_type(virtual_core, device_id), HalL1MemAddrType::CORE_INFO));
        initialize_firmware(device_id, HalProgrammableCoreType::IDLE_ETH, virtual_core, &launch_msg, &go_msg);
        not_done_cores.insert(virtual_core);
    }

    // Barrier between L1 writes above and deassert below
    tt::Cluster::instance().l1_barrier(device_id);

    // Deassert worker cores
    TensixSoftResetOptions reset_val;
    for (const auto& worker_core : not_done_cores) {
        if (active_eth_cores.find(worker_core) != active_eth_cores.end()) {
            // bit 12 needs to be deasserted to run second erisc on BH
            reset_val = TENSIX_DEASSERT_SOFT_RESET &
                        static_cast<TensixSoftResetOptions>(
                            ~std::underlying_type<TensixSoftResetOptions>::type(TensixSoftResetOptions::TRISC0));
        } else {
            reset_val = TENSIX_DEASSERT_SOFT_RESET;
        }
        tt::Cluster::instance().deassert_risc_reset_at_core(tt_cxy_pair(device_id, worker_core), reset_val);
    }

    // Wait until fw init is done, ensures the next launch msg doesn't get
    // written while fw is still in init
    log_debug("Waiting for firmware init complete");
    const int timeout_ms = 10000;  // 10 seconds for now
    try {
        llrt::internal_::wait_until_cores_done(device_id, RUN_MSG_INIT, not_done_cores, timeout_ms);
    } catch (std::runtime_error& e) {
        TT_THROW("Device {} init: failed to initialize FW! Try resetting the board.", device_id);
    }
    log_debug("Firmware init complete");
}

MetalContext* MetalContext::_inst = nullptr;

void MetalContext::initialize(
    const DispatchCoreConfig& dispatch_core_config, uint8_t num_hw_cqs, BankMapping l1_bank_remap) noexcept {
    log_debug(tt::LogMetal, "DevicePool initialize");
    if (_inst == nullptr) {
        static MetalContext MetalContext(dispatch_core_config, num_hw_cqs, l1_bank_remap);
        _inst = &MetalContext;
    } else if (
        _inst->dispatch_core_config_ != dispatch_core_config or num_hw_cqs != _inst->num_hw_cqs_ or
        l1_bank_remap != _inst->l1_bank_remap_) {
        TT_THROW("No re-init allowed.");
    }
}

MetalContext& MetalContext::instance() {
    TT_ASSERT(MetalContext::_inst != nullptr, "Trying to get MetalContext without initializing it");
    return *MetalContext::_inst;
}

MetalContext::MetalContext(
    const DispatchCoreConfig& dispatch_core_config, uint8_t num_hw_cqs, BankMapping l1_bank_remap) {
    log_warning("MetalContext Init!");
    dispatch_core_config_ = dispatch_core_config;
    num_hw_cqs_ = num_hw_cqs;
    l1_bank_remap_ = l1_bank_remap;

    // Initialize dispatch core manager, query manager, settings. TODO: could these all be pulled under this context?
    tt::tt_metal::dispatch_core_manager::initialize(dispatch_core_config, num_hw_cqs);
    tt_metal::DispatchQueryManager::initialize(num_hw_cqs);
    tt_metal::DispatchSettings::initialize(tt::Cluster::instance());

    auto all_devices = tt::Cluster::instance().all_chip_ids();
    for (chip_id_t device_id : all_devices) {
        initialize_cluster(device_id);
        generate_device_bank_to_noc_tables(device_id);

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
            reset_cores(device_id, num_hw_cqs_);
        }

        initialize_and_launch_firmware(device_id);

        // Watcher needs to init before FW since FW needs watcher mailboxes to be set up, and needs to attach after FW
        // starts since it also writes to watcher mailboxes.
        watcher_attach(device_id);
    }
}

MetalContext::~MetalContext() { log_warning("MetalContext Teardown!"); }

}  // namespace tt::tt_metal
