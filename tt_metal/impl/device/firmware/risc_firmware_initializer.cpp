// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "risc_firmware_initializer.hpp"

#include <algorithm>
#include <future>
#include <set>

#include <enchantum/enchantum.hpp>
#include <tracy/Tracy.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "impl/context/context_descriptor.hpp"
#include "core_coord.hpp"
#include "hal.hpp"
#include "hal_types.hpp"
#include "allocator/l1_banking_allocator.hpp"
#include "debug/noc_logging.hpp"
#include "dispatch/dispatch_core_common.hpp"
#include "dispatch/dispatch_core_manager.hpp"
#include "dispatch/topology.hpp"
#include "jit_build/build.hpp"
#include "jit_build/build_env_manager.hpp"
#include "llrt/llrt.hpp"
#include "common/executor.hpp"
#include <experimental/fabric/control_plane.hpp>
#include <experimental/fabric/fabric_types.hpp>
#include "hostdevcommon/common_values.hpp"
#include "tt_align.hpp"
#include <umd/device/types/xy_pair.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::tt_metal {

RiscFirmwareInitializer::RiscFirmwareInitializer(
    std::shared_ptr<const ContextDescriptor> descriptor,
    const GetControlPlaneFn& get_control_plane,
    dispatch_core_manager& dispatch_core_manager,
    std::optional<GetDispatchIgnoreCoresFn> get_dispatch_ignore_cores) :
    FirmwareInitializer(std::move(descriptor)),
    get_control_plane_(get_control_plane),
    dispatch_core_manager_(dispatch_core_manager),
    get_dispatch_ignore_cores_(std::move(get_dispatch_ignore_cores)),
    num_hw_cqs_(static_cast<uint8_t>(descriptor_->num_cqs())) {
    const Hal& hal = descriptor_->hal();
    size_t worker_l1_size = descriptor_->worker_l1_size();
    std::uint32_t max_alignment = std::max(hal.get_alignment(HalMemType::DRAM), hal.get_alignment(HalMemType::L1));
    worker_l1_unreserved_start_ = tt::align(
        hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) +
            hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) - worker_l1_size,
        max_alignment);
}

RiscFirmwareInitializer::~RiscFirmwareInitializer() = default;

void RiscFirmwareInitializer::init(
    const std::vector<Device*>& /*devices*/, const std::unordered_set<InitializerKey>& /*init_done*/) {
    TT_THROW(
        "RiscFirmwareInitializer::init is not implemented. Use run_async_build_phase and run_launch_phase instead.");
}

void RiscFirmwareInitializer::run_async_build_phase(const std::set<tt::ChipId>& device_ids) {
    ZoneScopedN("FW builds and Device Inits");

    std::vector<std::shared_future<void>> futures;
    futures.reserve(device_ids.size());

    // Reserve tables per device ID (single-threaded)
    dram_bank_offset_map_.reserve(device_ids.size());
    l1_bank_offset_map_.reserve(device_ids.size());
    dram_bank_to_noc_xy_.reserve(device_ids.size());
    l1_bank_to_noc_xy_.reserve(device_ids.size());
    worker_logical_col_to_virtual_col_.reserve(device_ids.size());
    worker_logical_row_to_virtual_row_.reserve(device_ids.size());
    for (tt::ChipId device_id : device_ids) {
        dram_bank_offset_map_[device_id].reserve(num_hw_cqs_);
        l1_bank_offset_map_[device_id].reserve(num_hw_cqs_);
        dram_bank_to_noc_xy_[device_id].reserve(num_hw_cqs_);
        l1_bank_to_noc_xy_[device_id].reserve(num_hw_cqs_);
        worker_logical_col_to_virtual_col_[device_id].reserve(num_hw_cqs_);
        worker_logical_row_to_virtual_row_[device_id].reserve(num_hw_cqs_);
    }

    // Set pointers to directly access the tables for each device as
    // multi threaded unordered_map access is not thread-safe
    struct PerDeviceTableRefs {
        tt::ChipId device_id;
        std::vector<int32_t>* dram_bank_offset_map;
        std::vector<int32_t>* l1_bank_offset_map;
        std::vector<uint16_t>* dram_bank_to_noc_xy;
        std::vector<uint16_t>* l1_bank_to_noc_xy;
        std::vector<uint8_t>* worker_logical_col_to_virtual_col;
        std::vector<uint8_t>* worker_logical_row_to_virtual_row;
    };
    std::vector<PerDeviceTableRefs> table_refs;
    table_refs.reserve(device_ids.size());
    for (tt::ChipId device_id : device_ids) {
        table_refs.push_back(
            {device_id,
             &dram_bank_offset_map_[device_id],
             &l1_bank_offset_map_[device_id],
             &dram_bank_to_noc_xy_[device_id],
             &l1_bank_to_noc_xy_[device_id],
             &worker_logical_col_to_virtual_col_[device_id],
             &worker_logical_row_to_virtual_row_[device_id]});
    }

    for (auto refs : table_refs) {
        futures.emplace_back(detail::async([this, refs]() {
            tt::ChipId device_id = refs.device_id;
            // Clear L1/DRAM if requested - skip for mock devices
            if (cluster_.get_target_device_type() != tt::TargetDevice::Mock) {
                if (rtoptions_.get_clear_l1()) {
                    clear_l1_state(device_id);
                }
                if (rtoptions_.get_clear_dram()) {
                    clear_dram_state(device_id);
                }
            }
            [[maybe_unused]] int ai_clk = cluster_.get_device_aiclk(device_id);
            log_debug(tt::LogMetal, "AI CLK for device {} is:   {} MHz", device_id, ai_clk);
            generate_device_bank_to_noc_tables(
                device_id,
                *refs.dram_bank_offset_map,
                *refs.l1_bank_offset_map,
                *refs.dram_bank_to_noc_xy,
                *refs.l1_bank_to_noc_xy);
            generate_worker_logical_to_virtual_map(
                device_id, *refs.worker_logical_col_to_virtual_col, *refs.worker_logical_row_to_virtual_row);

            // Skip firmware building for mock devices
            if (cluster_.get_target_device_type() != tt::TargetDevice::Mock) {
                // Create build env for this device, and build FW if it's not built already.
                // build_firmware ensures that the FW is built only once for a given build key
                // (which captures the fw_compile_hash).
                BuildEnvManager::get_instance().add_build_env(device_id, num_hw_cqs_);
                BuildEnvManager::get_instance().build_firmware(device_id);
                // Clear the entire launch message ring buffer on ethernet cores before application firmware is
                // activated. This is required since ethernet cores context switch between application and routing
                // firmware. If ERISC application firmware is activated before the launch messages are cleared, it
                // can enter an undefined state by reading a corrupted launch message. Routing firmware will never
                // run in this case, causing UMD issued transactions to hang.
                clear_launch_messages_on_eth_cores(device_id);
            }
        }));
    }

    for (auto& fut : futures) {
        fut.get();
    }
}

void RiscFirmwareInitializer::run_launch_phase(const std::set<tt::ChipId>& device_ids) {
    // Launch FW on each device sequentially, since a multithreaded launch leads to initialization hangs.
    // See https://github.com/tenstorrent/tt-metal/issues/35701
    ZoneScopedN("Resets and FW Launch");
    for (tt::ChipId device_id : device_ids) {
        if (cluster_.get_target_device_type() != tt::TargetDevice::Mock) {
            ClearNocData(device_id);
            reset_cores(device_id);
            initialize_and_launch_firmware(device_id);
        }
    }
    initialized_ = true;
}

void RiscFirmwareInitializer::configure() {}

void RiscFirmwareInitializer::teardown_simulator_ethernet_cores() {
    // If simulator is enabled, force a teardown of active ethernet cores for WH
    if (rtoptions_.get_simulator_enabled()) {
        if (hal_.get_eth_fw_is_cooperative()) {
            auto all_devices = cluster_.all_chip_ids();
            for (tt::ChipId device_id : all_devices) {
                for (const auto& logical_core : this->get_control_plane_().get_active_ethernet_cores(device_id)) {
                    CoreCoord virtual_core = cluster_.get_virtual_coordinate_from_logical_coordinates(
                        device_id, logical_core, CoreType::ETH);
                    erisc_send_exit_signal(device_id, virtual_core, false);
                    while (erisc_app_still_running(device_id, virtual_core)) {
                    }
                }
            }
        }
    }
}

void RiscFirmwareInitializer::teardown(std::unordered_set<InitializerKey>& /*init_done*/) {
    auto all_devices = cluster_.all_chip_ids();

    teardown_simulator_ethernet_cores();

    if (cluster_.get_target_device_type() != tt::TargetDevice::Mock) {
        for (tt::ChipId device_id : all_devices) {
            std::unordered_set<CoreCoord> ignore_cores;
            if (get_dispatch_ignore_cores_) {
                ignore_cores = (*get_dispatch_ignore_cores_)(device_id);
            }
            assert_cores(device_id, ignore_cores);
            cluster_.l1_barrier(device_id);
        }
        // Set internal routing to false to exit active ethernet FW & go back to base FW
        // Must be last
        if (get_control_plane_) {
            cluster_.set_internal_routing_info_for_ethernet_cores(this->get_control_plane_(), false);
        }
    }

    initialized_ = false;
}

bool RiscFirmwareInitializer::is_initialized() const { return initialized_; }

void RiscFirmwareInitializer::clear_l1_state(tt::ChipId device_id) {
    log_debug(tt::LogMetal, "Clearing L1 for device {}", device_id);
    CoreCoord logical_grid_size = cluster_.get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    uint32_t l1_size_per_core = cluster_.get_soc_desc(device_id).worker_l1_size;
    TT_ASSERT(l1_size_per_core % sizeof(uint32_t) == 0);
    std::vector<uint32_t> zero_vec(l1_size_per_core / sizeof(uint32_t), 0);
    constexpr uint32_t start_address = 0;
    for (uint32_t x = 0; x < logical_grid_size.x; x++) {
        for (uint32_t y = 0; y < logical_grid_size.y; y++) {
            CoreCoord logical_core(x, y);
            auto virtual_core =
                cluster_.get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::WORKER);
            cluster_.write_core(device_id, virtual_core, zero_vec, start_address);
        }
    }

    for (const auto& eth_core : this->get_control_plane_().get_active_ethernet_cores(device_id)) {
        static uint32_t zero_vec_size = hal::get_erisc_l1_unreserved_size();
        auto zero_vec_addr = hal::get_erisc_l1_unreserved_base();
        static std::vector<uint32_t> zero_vec(zero_vec_size / sizeof(uint32_t), 0);
        CoreCoord virtual_core =
            cluster_.get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);
        cluster_.write_core(device_id, virtual_core, zero_vec, zero_vec_addr);
    }
    cluster_.l1_barrier(device_id);
}

void RiscFirmwareInitializer::clear_dram_state(tt::ChipId device_id) {
    log_debug(tt::LogMetal, "Clearing DRAM for device {}", device_id);
    auto dram_size_per_channel = cluster_.get_soc_desc(device_id).dram_view_size;
    auto num_dram_channels = cluster_.get_soc_desc(device_id).get_num_dram_views();
    constexpr uint32_t start_address = 0;
    std::vector<uint8_t> zero_vec(dram_size_per_channel, 0);
    for (int channel = 0; channel < num_dram_channels; ++channel) {
        cluster_.write_dram_vec(zero_vec.data(), zero_vec.size(), device_id, channel, start_address);
        cluster_.dram_barrier(device_id);
    }
}

void RiscFirmwareInitializer::clear_launch_messages_on_eth_cores(tt::ChipId device_id) {
    auto clear_ethernet_core = [&](const CoreCoord& logical_eth_core, HalProgrammableCoreType programmable_core_type) {
        auto factory = hal_.get_dev_msgs_factory(programmable_core_type);
        std::vector<std::byte> init_launch_msg_data(
            dev_msgs::launch_msg_buffer_num_entries * factory.size_of<dev_msgs::launch_msg_t>(), std::byte{0});
        dev_msgs::go_msg_t go_msg = factory.create<dev_msgs::go_msg_t>();
        go_msg.view().signal() = dev_msgs::RUN_MSG_INIT;

        CoreCoord virtual_eth_core =
            cluster_.get_virtual_coordinate_from_logical_coordinates(device_id, logical_eth_core, CoreType::ETH);
        cluster_.write_core(
            init_launch_msg_data.data(),
            init_launch_msg_data.size(),
            tt_cxy_pair(device_id, virtual_eth_core),
            hal_.get_dev_addr(programmable_core_type, HalL1MemAddrType::LAUNCH));
        cluster_.write_core(
            go_msg.data(),
            go_msg.size(),
            {static_cast<size_t>(device_id), virtual_eth_core},
            hal_.get_dev_addr(programmable_core_type, HalL1MemAddrType::GO_MSG));
    };

    if (!has_flag(descriptor_->fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
        return;
    }
    for (const auto& eth_core : this->get_control_plane_().get_active_ethernet_cores(device_id)) {
        clear_ethernet_core(eth_core, HalProgrammableCoreType::ACTIVE_ETH);
    }
    for (const auto& eth_core : this->get_control_plane_().get_inactive_ethernet_cores(device_id)) {
        clear_ethernet_core(eth_core, HalProgrammableCoreType::IDLE_ETH);
    }
    cluster_.l1_barrier(device_id);
}

void RiscFirmwareInitializer::assert_active_ethernet_cores_to_reset(tt::ChipId device_id) {
    for (const auto& logical_core : this->get_control_plane_().get_active_ethernet_cores(device_id)) {
        CoreCoord virtual_core =
            cluster_.get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::ETH);
        if (rtoptions_.get_enable_2_erisc_mode()) {
            llrt::internal_::return_to_base_firmware_and_wait_for_heartbeat(device_id, virtual_core);
        }
        tt::umd::RiscType reset_val = tt::umd::RiscType::ALL_TENSIX & ~tt::umd::RiscType::ERISC0;
        cluster_.assert_risc_reset_at_core(tt_cxy_pair(device_id, virtual_core), reset_val);
    }
}

void RiscFirmwareInitializer::assert_tensix_workers_impl(
    tt::ChipId device_id, const std::unordered_set<CoreCoord>* ignore_virtual_cores) {
    CoreCoord grid_size = cluster_.get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    const bool teardown_mode = (ignore_virtual_cores != nullptr);
    const std::unordered_set<CoreCoord>& active_eth_logical =
        teardown_mode ? this->get_control_plane_().get_active_ethernet_cores(device_id, false)
                      : std::unordered_set<CoreCoord>{};
    const bool skip_active_eth_workers = teardown_mode && !hal_.get_eth_fw_is_cooperative();

    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core =
                cluster_.get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::WORKER);

            if (teardown_mode && ignore_virtual_cores->contains(worker_core)) {
                log_debug(tt::LogMetal, "{} will not be Reset when closing Device {}", worker_core.str(), device_id);
                continue;
            }
            if (skip_active_eth_workers && active_eth_logical.contains(logical_core)) {
                continue;  // Cannot put these cores into reset; they are running base FW (handled below)
            }
            cluster_.assert_risc_reset_at_core(tt_cxy_pair(device_id, worker_core), tt::umd::RiscType::ALL);
        }
    }
}

void RiscFirmwareInitializer::assert_inactive_ethernet_cores(tt::ChipId device_id) {
    for (const auto& logical_core : this->get_control_plane_().get_inactive_ethernet_cores(device_id)) {
        CoreCoord virtual_core =
            cluster_.get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::ETH);
        cluster_.assert_risc_reset_at_core(tt_cxy_pair(device_id, virtual_core), tt::umd::RiscType::ALL);
    }
}

void RiscFirmwareInitializer::reset_cores(tt::ChipId device_id) {
    ZoneScoped;
    std::unordered_map<tt::ChipId, std::unordered_set<CoreCoord>> device_to_early_exit_cores;

    if (has_flag(descriptor_->fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
        if (hal_.get_eth_fw_is_cooperative()) {
            for (const auto& logical_core : this->get_control_plane_().get_active_ethernet_cores(device_id)) {
                CoreCoord virtual_core =
                    cluster_.get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::ETH);
                if (erisc_app_still_running(device_id, virtual_core)) {
                    log_info(
                        tt::LogMetal,
                        "While initializing device {}, active ethernet dispatch core {} detected as still "
                        "running, issuing exit signal.",
                        device_id,
                        virtual_core.str());
                    erisc_send_exit_signal(device_id, virtual_core, false);
                    device_to_early_exit_cores[device_id].insert(virtual_core);
                }
            }
        } else {
            assert_active_ethernet_cores_to_reset(device_id);
        }
    }

    for (auto& id_and_cores : device_to_early_exit_cores) {
        const int timeout_ms = 10000;
        if (!id_and_cores.second.empty()) {
            try {
                llrt::internal_::wait_until_cores_done(
                    id_and_cores.first, dev_msgs::RUN_MSG_GO, id_and_cores.second, timeout_ms);
            } catch (std::runtime_error&) {
                log_warning(
                    tt::LogAlways,
                    "Detected dispatch kernels still running but failed to complete an early exit. This may happen "
                    "from time to time following a reset, continuing to FW initialization...");
            }
        }
    }

    assert_tensix_workers_impl(device_id, nullptr);

    if (has_flag(descriptor_->fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
        assert_inactive_ethernet_cores(device_id);
    }
    cluster_.l1_barrier(device_id);
}

void RiscFirmwareInitializer::assert_cores(tt::ChipId device_id, std::unordered_set<CoreCoord>& ignore_virtual_cores) {
    assert_tensix_workers_impl(device_id, &ignore_virtual_cores);
    if (!hal_.get_eth_fw_is_cooperative()) {
        assert_active_ethernet_cores_to_reset(device_id);
    }
}

CoreCoord RiscFirmwareInitializer::virtual_noc0_coordinate(tt::ChipId device_id, uint8_t noc_index, CoreCoord coord) {
    const auto& grid_size = cluster_.get_soc_desc(device_id).grid_size;
    if (coord.x >= grid_size.x || coord.y >= grid_size.y || cluster_.arch() == ARCH::BLACKHOLE) {
        return coord;
    }
    coord = cluster_.get_virtual_coordinate_from_physical_coordinates(device_id, coord);
    CoreCoord virtual_coord = {
        hal_.noc_coordinate(noc_index, grid_size.x, coord.x), hal_.noc_coordinate(noc_index, grid_size.y, coord.y)};
    return virtual_coord;
}

void RiscFirmwareInitializer::generate_device_bank_to_noc_tables(tt::ChipId device_id) {
    generate_device_bank_to_noc_tables(
        device_id,
        dram_bank_offset_map_[device_id],
        l1_bank_offset_map_[device_id],
        dram_bank_to_noc_xy_[device_id],
        l1_bank_to_noc_xy_[device_id]);
}

void RiscFirmwareInitializer::generate_device_bank_to_noc_tables(
    tt::ChipId device_id,
    std::vector<int32_t>& dram_bank_offset_map,
    std::vector<int32_t>& l1_bank_offset_map,
    std::vector<uint16_t>& dram_bank_to_noc_xy,
    std::vector<uint16_t>& l1_bank_to_noc_xy) {
    BankMapping l1_bank_remap(descriptor_->l1_bank_remap().begin(), descriptor_->l1_bank_remap().end());
    auto config = L1BankingAllocator::generate_config(
        device_id,
        num_hw_cqs_,
        DEFAULT_L1_SMALL_SIZE,      // Not required for noc table gen
        DEFAULT_TRACE_REGION_SIZE,  // Not required for noc table gen
        worker_l1_unreserved_start_,
        l1_bank_remap);
    const auto allocator = L1BankingAllocator(config);
    const auto& soc_d = cluster_.get_soc_desc(device_id);
    const size_t num_dram_banks = allocator.get_num_banks(BufferType::DRAM);
    dram_bank_offset_map.clear();
    dram_bank_offset_map.resize(num_dram_banks);
    for (unsigned bank_id = 0; bank_id < num_dram_banks; bank_id++) {
        dram_bank_offset_map[bank_id] = allocator.get_bank_offset(BufferType::DRAM, bank_id);
    }
    const size_t num_l1_banks = allocator.get_num_banks(BufferType::L1);
    std::vector<CoreCoord> l1_noc_coord_per_bank(num_l1_banks);
    l1_bank_offset_map.clear();
    l1_bank_offset_map.resize(num_l1_banks);
    for (unsigned bank_id = 0; bank_id < num_l1_banks; bank_id++) {
        l1_noc_coord_per_bank[bank_id] = cluster_.get_virtual_coordinate_from_logical_coordinates(
            device_id, allocator.get_logical_core_from_bank_id(bank_id), CoreType::WORKER);
        l1_bank_offset_map[bank_id] = allocator.get_bank_offset(BufferType::L1, bank_id);
    }

    dram_bank_to_noc_xy.clear();
    dram_bank_to_noc_xy.reserve(hal_.get_num_nocs() * num_dram_banks);
    bool noc_translation_enabled = cluster_.get_target_device_type() != tt::TargetDevice::Mock &&
                                   cluster_.get_cluster_desc()->get_noc_translation_table_en().at(device_id);
    bool dram_is_virtualized =
        noc_translation_enabled && (hal_.get_virtualized_core_types().contains(dev_msgs::AddressableCoreType::DRAM));
    for (unsigned int noc = 0; noc < hal_.get_num_nocs(); noc++) {
        for (unsigned int bank_id = 0; bank_id < num_dram_banks; bank_id++) {
            CoreCoord dram_noc_coord =
                soc_d.get_preferred_worker_core_for_dram_view(allocator.get_dram_channel_from_bank_id(bank_id), noc);
            uint16_t noc_x, noc_y;
            if (dram_is_virtualized) {
                noc_x = dram_noc_coord.x;
                noc_y = dram_noc_coord.y;
            } else {
                noc_x = hal_.noc_coordinate(noc, soc_d.grid_size.x, dram_noc_coord.x);
                noc_y = hal_.noc_coordinate(noc, soc_d.grid_size.y, dram_noc_coord.y);
            }
            uint16_t xy = ((noc_y << hal_.get_noc_addr_node_id_bits()) | noc_x) << hal_.get_noc_coord_reg_offset();
            dram_bank_to_noc_xy.push_back(xy);
        }
    }

    l1_bank_to_noc_xy.clear();
    l1_bank_to_noc_xy.reserve(hal_.get_num_nocs() * l1_noc_coord_per_bank.size());
    for (unsigned int noc = 0; noc < hal_.get_num_nocs(); noc++) {
        for (const auto& noc_coord : l1_noc_coord_per_bank) {
            auto l1_noc_coords = virtual_noc0_coordinate(device_id, noc, noc_coord);
            uint16_t noc_x = l1_noc_coords.x;
            uint16_t noc_y = l1_noc_coords.y;
            uint16_t xy = ((noc_y << hal_.get_noc_addr_node_id_bits()) | noc_x) << hal_.get_noc_coord_reg_offset();
            l1_bank_to_noc_xy.push_back(xy);
        }
    }
}

void RiscFirmwareInitializer::generate_worker_logical_to_virtual_map(tt::ChipId device_id) {
    generate_worker_logical_to_virtual_map(
        device_id, worker_logical_col_to_virtual_col_[device_id], worker_logical_row_to_virtual_row_[device_id]);
}

void RiscFirmwareInitializer::generate_worker_logical_to_virtual_map(
    tt::ChipId device_id,
    std::vector<uint8_t>& worker_logical_col_to_virtual_col,
    std::vector<uint8_t>& worker_logical_row_to_virtual_row) {
    const auto& soc_desc = cluster_.get_soc_desc(device_id);
    auto tensix_grid_size = soc_desc.get_grid_size(CoreType::TENSIX);

    worker_logical_col_to_virtual_col.clear();
    worker_logical_row_to_virtual_row.clear();
    worker_logical_col_to_virtual_col.reserve(tensix_grid_size.x);
    worker_logical_row_to_virtual_row.reserve(tensix_grid_size.y);

    for (size_t x = 0; x < tensix_grid_size.x; x++) {
        worker_logical_col_to_virtual_col.push_back(
            soc_desc
                .translate_coord_to({tt_xy_pair{x, 0}, CoreType::TENSIX, CoordSystem::LOGICAL}, CoordSystem::TRANSLATED)
                .x);
    }
    for (size_t y = 0; y < tensix_grid_size.y; y++) {
        worker_logical_row_to_virtual_row.push_back(
            soc_desc
                .translate_coord_to({tt_xy_pair{0, y}, CoreType::TENSIX, CoordSystem::LOGICAL}, CoordSystem::TRANSLATED)
                .y);
    }
}

void RiscFirmwareInitializer::initialize_device_bank_to_noc_tables(
    tt::ChipId device_id,
    const HalProgrammableCoreType& core_type,
    CoreCoord virtual_core,
    std::optional<CoreCoord> end_core) {
    const uint32_t dram_to_noc_sz_in_bytes = dram_bank_to_noc_xy_[device_id].size() * sizeof(uint16_t);
    const uint32_t l1_to_noc_sz_in_bytes = l1_bank_to_noc_xy_[device_id].size() * sizeof(uint16_t);
    const uint32_t dram_offset_sz_in_bytes = dram_bank_offset_map_[device_id].size() * sizeof(int32_t);
    const uint32_t l1_offset_sz_in_bytes = l1_bank_offset_map_[device_id].size() * sizeof(int32_t);

    const uint64_t mem_bank_to_noc_addr = hal_.get_dev_addr(core_type, HalL1MemAddrType::BANK_TO_NOC_SCRATCH);
    const uint32_t mem_bank_to_noc_size = hal_.get_dev_size(core_type, HalL1MemAddrType::BANK_TO_NOC_SCRATCH);

    TT_ASSERT(
        (dram_to_noc_sz_in_bytes + l1_to_noc_sz_in_bytes + dram_offset_sz_in_bytes + l1_offset_sz_in_bytes) <=
            mem_bank_to_noc_size,
        "Size of bank_to_noc table is greater than available space");

    if (end_core.has_value()) {
        auto start_core = virtual_core;
        cluster_.noc_multicast_write(
            dram_bank_to_noc_xy_[device_id].data(),
            dram_to_noc_sz_in_bytes,
            device_id,
            start_core,
            end_core.value(),
            mem_bank_to_noc_addr);

        uint64_t l1_noc_addr = mem_bank_to_noc_addr + dram_to_noc_sz_in_bytes;
        cluster_.noc_multicast_write(
            l1_bank_to_noc_xy_[device_id].data(),
            l1_to_noc_sz_in_bytes,
            device_id,
            start_core,
            end_core.value(),
            l1_noc_addr);

        uint64_t dram_offset_addr = l1_noc_addr + l1_to_noc_sz_in_bytes;
        cluster_.noc_multicast_write(
            dram_bank_offset_map_[device_id].data(),
            dram_offset_sz_in_bytes,
            device_id,
            start_core,
            end_core.value(),
            dram_offset_addr);

        uint64_t l1_offset_addr = dram_offset_addr + dram_offset_sz_in_bytes;
        cluster_.noc_multicast_write(
            l1_bank_offset_map_[device_id].data(),
            l1_offset_sz_in_bytes,
            device_id,
            start_core,
            end_core.value(),
            l1_offset_addr);
    } else {
        cluster_.write_core(
            dram_bank_to_noc_xy_[device_id].data(),
            dram_to_noc_sz_in_bytes,
            tt_cxy_pair(device_id, virtual_core),
            mem_bank_to_noc_addr);

        uint64_t l1_noc_addr = mem_bank_to_noc_addr + dram_to_noc_sz_in_bytes;
        cluster_.write_core(
            l1_bank_to_noc_xy_[device_id].data(),
            l1_to_noc_sz_in_bytes,
            tt_cxy_pair(device_id, virtual_core),
            l1_noc_addr);

        uint64_t dram_offset_addr = l1_noc_addr + l1_to_noc_sz_in_bytes;
        cluster_.write_core(
            dram_bank_offset_map_[device_id].data(),
            dram_offset_sz_in_bytes,
            tt_cxy_pair(device_id, virtual_core),
            dram_offset_addr);

        uint64_t l1_offset_addr = dram_offset_addr + dram_offset_sz_in_bytes;
        cluster_.write_core(
            l1_bank_offset_map_[device_id].data(),
            l1_offset_sz_in_bytes,
            tt_cxy_pair(device_id, virtual_core),
            l1_offset_addr);
    }
}

void RiscFirmwareInitializer::initialize_worker_logical_to_virtual_tables(
    tt::ChipId device_id, const HalProgrammableCoreType& core_type, CoreCoord start_core, CoreCoord end_core) {
    const auto& soc_desc = cluster_.get_soc_desc(device_id);
    const uint32_t logical_col_to_virtual_col_sz_in_bytes =
        worker_logical_col_to_virtual_col_[device_id].size() * sizeof(uint8_t);
    const uint8_t firmware_grid_size_x = tt::round_up(soc_desc.grid_size.x, 4);
    const uint32_t logical_row_to_virtual_row_sz_in_bytes =
        worker_logical_row_to_virtual_row_[device_id].size() * sizeof(uint8_t);
    const uint64_t logical_to_virtual_map_addr =
        hal_.get_dev_addr(core_type, HalL1MemAddrType::LOGICAL_TO_VIRTUAL_SCRATCH);
    const uint32_t logical_to_virtual_map_size =
        hal_.get_dev_size(core_type, HalL1MemAddrType::LOGICAL_TO_VIRTUAL_SCRATCH);

    TT_ASSERT(
        (firmware_grid_size_x + logical_row_to_virtual_row_sz_in_bytes) <= logical_to_virtual_map_size,
        "Size of logical to virtual map is greater than available space");

    uint64_t logical_col_to_virtual_col_addr = logical_to_virtual_map_addr;
    cluster_.noc_multicast_write(
        worker_logical_col_to_virtual_col_[device_id].data(),
        logical_col_to_virtual_col_sz_in_bytes,
        device_id,
        start_core,
        end_core,
        logical_col_to_virtual_col_addr);

    uint64_t logical_row_to_virtual_row_addr = logical_to_virtual_map_addr + (firmware_grid_size_x * sizeof(uint8_t));
    cluster_.noc_multicast_write(
        worker_logical_row_to_virtual_row_[device_id].data(),
        logical_row_to_virtual_row_sz_in_bytes,
        device_id,
        start_core,
        end_core,
        logical_row_to_virtual_row_addr);
}

uint32_t RiscFirmwareInitializer::get_active_erisc_launch_flag_addr() {
    auto core_type_idx = hal_.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH);
    return hal_.get_jit_build_config(core_type_idx, 0, 0).fw_launch_addr;
}

bool RiscFirmwareInitializer::erisc_app_still_running(tt::ChipId device_id, CoreCoord virtual_core) {
    if (cluster_.arch() != ARCH::WORMHOLE_B0) {
        return false;
    }
    TT_ASSERT(
        cluster_.is_ethernet_core(virtual_core, device_id),
        "Invalid core {} for context switch check",
        virtual_core.str());
    std::uint32_t launch_erisc_addr = get_active_erisc_launch_flag_addr();
    auto data = cluster_.read_core(device_id, virtual_core, launch_erisc_addr, sizeof(std::uint32_t));
    return (data[0] != 0);
}

void RiscFirmwareInitializer::erisc_send_exit_signal(tt::ChipId device_id, CoreCoord virtual_core, bool is_idle_eth) {
    HalProgrammableCoreType programmable_core_type =
        is_idle_eth ? HalProgrammableCoreType::IDLE_ETH : HalProgrammableCoreType::ACTIVE_ETH;
    auto dev_msgs_factory = hal_.get_dev_msgs_factory(programmable_core_type);
    auto launch_msg = dev_msgs_factory.create<dev_msgs::launch_msg_t>();
    auto go_msg = dev_msgs_factory.create<dev_msgs::go_msg_t>();
    DeviceAddr launch_addr = hal_.get_dev_addr(programmable_core_type, HalL1MemAddrType::LAUNCH);

    cluster_.read_core(
        launch_msg.data(), launch_msg.size(), {static_cast<size_t>(device_id), virtual_core}, launch_addr);

    launch_msg.view().kernel_config().exit_erisc_kernel() = 1;
    llrt::write_launch_msg_to_core(device_id, virtual_core, launch_msg.view(), go_msg.view(), false);

    if (!is_idle_eth) {
        std::vector<uint32_t> clear_flag_data = {0};
        cluster_.write_core_immediate(device_id, virtual_core, clear_flag_data, get_active_erisc_launch_flag_addr());
    }
}

dev_msgs::core_info_msg_t RiscFirmwareInitializer::populate_core_info_msg(
    tt::ChipId device_id, HalProgrammableCoreType programmable_core_type) const {
    const metal_SocDescriptor& soc_d = cluster_.get_soc_desc(device_id);
    auto factory = hal_.get_dev_msgs_factory(programmable_core_type);
    dev_msgs::core_info_msg_t buffer = factory.create<dev_msgs::core_info_msg_t>();
    auto core_info = buffer.view();
    core_info.noc_pcie_addr_base() = hal_.get_pcie_addr_lower_bound();
    core_info.noc_pcie_addr_end() = hal_.get_pcie_addr_upper_bound();
    core_info.noc_dram_addr_base() = 0;
    core_info.noc_dram_addr_end() = soc_d.dram_core_size;
    core_info.l1_unreserved_start() = align(worker_l1_unreserved_start_, hal_.get_alignment(HalMemType::DRAM));
    if (programmable_core_type == HalProgrammableCoreType::TENSIX) {
        core_info.core_magic_number() = dev_msgs::CoreMagicNumber::WORKER;
    } else if (programmable_core_type == HalProgrammableCoreType::ACTIVE_ETH) {
        core_info.core_magic_number() = dev_msgs::CoreMagicNumber::ACTIVE_ETH;
    } else {
        core_info.core_magic_number() = dev_msgs::CoreMagicNumber::IDLE_ETH;
    }
    const std::vector<tt::umd::CoreCoord>& pcie_cores = soc_d.get_cores(CoreType::PCIE, CoordSystem::NOC0);
    std::unordered_set<tt::umd::CoreCoord> dram_cores;
    auto num_dram_channels = cluster_.get_soc_desc(device_id).get_num_dram_views();
    for (uint32_t dram_channel = 0; dram_channel < num_dram_channels; dram_channel++) {
        for (uint32_t noc = 0; noc < hal_.get_num_nocs(); noc++) {
            auto worker_dram_ep = soc_d.get_preferred_worker_core_for_dram_view(dram_channel, noc);
            auto eth_dram_ep = soc_d.get_preferred_eth_core_for_dram_view(dram_channel, noc);
            auto physical_worker_dram_ep =
                soc_d.translate_coord_to(worker_dram_ep, CoordSystem::TRANSLATED, CoordSystem::NOC0);
            auto physical_eth_dram_ep =
                soc_d.translate_coord_to(eth_dram_ep, CoordSystem::TRANSLATED, CoordSystem::NOC0);
            dram_cores.insert(physical_worker_dram_ep);
            dram_cores.insert(physical_eth_dram_ep);
        }
    }

    const std::vector<tt::umd::CoreCoord>& eth_cores = soc_d.get_cores(CoreType::ETH, CoordSystem::NOC0);

    TT_ASSERT(
        pcie_cores.size() + dram_cores.size() + eth_cores.size() <= core_info.non_worker_cores().size(),
        "Detected more pcie/dram/eth cores than fit in the device mailbox.");
    TT_ASSERT(
        eth_cores.size() <= core_info.virtual_non_worker_cores().size(),
        "Detected more eth cores (virtual non-workers) than can fit in device mailbox.");
    auto set_addressable_core =
        [](dev_msgs::addressable_core_t::View core, const CoreCoord& core_coord, dev_msgs::AddressableCoreType type) {
            core.x() = core_coord.x;
            core.y() = core_coord.y;
            core.type() = type;
        };
    for (auto non_worker_core : core_info.non_worker_cores()) {
        set_addressable_core(
            non_worker_core,
            {dev_msgs::CORE_COORD_INVALID, dev_msgs::CORE_COORD_INVALID},
            dev_msgs::AddressableCoreType::UNKNOWN);
    }
    for (auto virtual_non_worker_core : core_info.virtual_non_worker_cores()) {
        set_addressable_core(
            virtual_non_worker_core,
            {dev_msgs::CORE_COORD_INVALID, dev_msgs::CORE_COORD_INVALID},
            dev_msgs::AddressableCoreType::UNKNOWN);
    }
    int non_worker_cores_idx = 0;
    bool skip_physical = cluster_.arch() == ARCH::BLACKHOLE and hal_.is_coordinate_virtualization_enabled();
    if (not skip_physical) {
        for (tt::umd::CoreCoord core : pcie_cores) {
            set_addressable_core(
                core_info.non_worker_cores()[non_worker_cores_idx++], core, dev_msgs::AddressableCoreType::PCIE);
        }
        for (tt::umd::CoreCoord core : dram_cores) {
            set_addressable_core(
                core_info.non_worker_cores()[non_worker_cores_idx++], core, dev_msgs::AddressableCoreType::DRAM);
        }
        for (tt::umd::CoreCoord core : eth_cores) {
            set_addressable_core(
                core_info.non_worker_cores()[non_worker_cores_idx++], core, dev_msgs::AddressableCoreType::ETH);
        }
    }

    if (hal_.is_coordinate_virtualization_enabled()) {
        uint32_t virtual_non_worker_cores_idx = 0;
        for (tt::umd::CoreCoord core : eth_cores) {
            auto virtual_core = cluster_.get_virtual_coordinate_from_physical_coordinates(device_id, {core.x, core.y});
            set_addressable_core(
                core_info.virtual_non_worker_cores()[virtual_non_worker_cores_idx++],
                virtual_core,
                dev_msgs::AddressableCoreType::ETH);
        }

        if (cluster_.arch() == ARCH::BLACKHOLE) {
            for (const CoreCoord& core : pcie_cores) {
                auto virtual_core =
                    cluster_.get_virtual_coordinate_from_physical_coordinates(device_id, {core.x, core.y});
                set_addressable_core(
                    core_info.virtual_non_worker_cores()[virtual_non_worker_cores_idx++],
                    virtual_core,
                    dev_msgs::AddressableCoreType::PCIE);
            }

            for (const CoreCoord& core : dram_cores) {
                auto virtual_core =
                    cluster_.get_virtual_coordinate_from_physical_coordinates(device_id, {core.x, core.y});
                set_addressable_core(
                    core_info.virtual_non_worker_cores()[virtual_non_worker_cores_idx++],
                    virtual_core,
                    dev_msgs::AddressableCoreType::DRAM);
            }
        }
    }

    std::vector<uint32_t> harvested_axis_coord;
    CoreCoord logical_grid_size = cluster_.get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    uint32_t harvested_noc_coords = umd::CoordinateManager::shuffle_tensix_harvesting_mask_to_noc0_coords(
        cluster_.get_soc_desc(device_id).arch, cluster_.get_harvesting_mask(device_id));
    uint32_t max_along_axis =
        hal_.get_tensix_harvest_axis() == HalTensixHarvestAxis::ROW ? soc_d.grid_size.y : soc_d.grid_size.x;
    for (uint32_t idx = 0; idx < max_along_axis; idx++) {
        bool harvested_axis = (harvested_noc_coords >> idx) & 0x1;
        if (harvested_axis) {
            harvested_axis_coord.push_back(idx);
        }
    }
    TT_ASSERT(
        harvested_axis_coord.size() <= core_info.harvested_coords().size(),
        "Detected more harvested rows than fit in mailbox.");
    for (size_t idx = 0; idx < core_info.harvested_coords().size(); idx++) {
        core_info.harvested_coords()[idx] =
            (idx < harvested_axis_coord.size()) ? harvested_axis_coord[idx] : dev_msgs::CORE_COORD_INVALID;
        if (hal_.is_coordinate_virtualization_enabled() and idx < harvested_axis_coord.size()) {
            uint32_t end_virtual_grid;
            if (hal_.get_tensix_harvest_axis() == HalTensixHarvestAxis::ROW) {
                end_virtual_grid = hal_.get_virtual_worker_start_y() + logical_grid_size.y;
            } else if (cluster_.arch() == ARCH::BLACKHOLE) {
                end_virtual_grid = max_along_axis - 1;
            } else {
                end_virtual_grid = hal_.get_virtual_worker_start_x() + logical_grid_size.x;
            }
            core_info.virtual_harvested_coords()[idx] = end_virtual_grid + harvested_axis_coord.size() - (idx + 1);
        } else {
            core_info.virtual_harvested_coords()[idx] = dev_msgs::CORE_COORD_INVALID;
        }
    }

    core_info.noc_size_x() = soc_d.grid_size.x;
    core_info.noc_size_y() = soc_d.grid_size.y;
    core_info.worker_grid_size_x() = logical_grid_size.x;
    core_info.worker_grid_size_y() = logical_grid_size.y;

    return buffer;
}

void RiscFirmwareInitializer::initialize_firmware(
    tt::ChipId device_id,
    const HalProgrammableCoreType& core_type,
    CoreCoord virtual_core,
    dev_msgs::launch_msg_t::View launch_msg,
    dev_msgs::go_msg_t::ConstView go_msg,
    std::optional<CoreCoord> end_core) {
    ZoneScoped;

    TT_FATAL(
        core_type != HalProgrammableCoreType::TENSIX or end_core.has_value(),
        "Tensix cores require end_core to be specified for bank to noc table initialization.");

    initialize_device_bank_to_noc_tables(device_id, core_type, virtual_core, end_core);
    if (core_type == HalProgrammableCoreType::TENSIX) {
        initialize_worker_logical_to_virtual_tables(device_id, core_type, virtual_core, end_core.value());
    }

    uint32_t core_type_idx = hal_.get_programmable_core_type_index(core_type);
    uint32_t processor_class_count = hal_.get_processor_classes_count(core_type);
    auto jit_build_config = hal_.get_jit_build_config(core_type_idx, 0, 0);

    const auto start_core = virtual_core;

    size_t launch_msg_size = launch_msg.size();
    std::vector<std::byte> init_launch_msg_data(
        dev_msgs::launch_msg_buffer_num_entries * launch_msg_size, std::byte{0});
    auto prepare_initial_launch_msg = [&]() {
        for (size_t i = 0; i < dev_msgs::launch_msg_buffer_num_entries; ++i) {
            std::copy(
                launch_msg.data(),
                launch_msg.data() + launch_msg_size,
                init_launch_msg_data.data() + (i * launch_msg_size));
        }
    };
    const auto write_initial_go_launch_msg = [&]() {
        auto programmable_core_type = llrt::get_core_type(device_id, virtual_core);
        uint32_t launch_addr = hal_.get_dev_addr(programmable_core_type, HalL1MemAddrType::LAUNCH);
        uint32_t go_addr = hal_.get_dev_addr(programmable_core_type, HalL1MemAddrType::GO_MSG);
        uint64_t launch_msg_buffer_read_ptr_addr =
            hal_.get_dev_addr(programmable_core_type, HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR);
        uint32_t go_message_index_addr = hal_.get_dev_addr(programmable_core_type, HalL1MemAddrType::GO_MSG_INDEX);
        if (core_type != HalProgrammableCoreType::TENSIX) {
            cluster_.write_core(
                init_launch_msg_data.data(),
                init_launch_msg_data.size(),
                tt_cxy_pair(device_id, virtual_core),
                launch_addr);
            cluster_.write_core(go_msg.data(), go_msg.size(), tt_cxy_pair(device_id, virtual_core), go_addr);
            uint32_t zero = 0;
            cluster_.write_core(
                &zero, sizeof(uint32_t), tt_cxy_pair(device_id, virtual_core), launch_msg_buffer_read_ptr_addr);
            cluster_.write_core(&zero, sizeof(uint32_t), tt_cxy_pair(device_id, virtual_core), go_message_index_addr);
        } else {
            cluster_.noc_multicast_write(
                init_launch_msg_data.data(),
                init_launch_msg_data.size(),
                device_id,
                start_core,
                end_core.value(),
                launch_addr);
            cluster_.noc_multicast_write(
                go_msg.data(), go_msg.size(), device_id, start_core, end_core.value(), go_addr);
            uint32_t zero = 0;
            cluster_.noc_multicast_write(
                &zero, sizeof(uint32_t), device_id, start_core, end_core.value(), launch_msg_buffer_read_ptr_addr);
            cluster_.noc_multicast_write(
                &zero, sizeof(uint32_t), device_id, start_core, end_core.value(), go_message_index_addr);
        }
    };

    switch (core_type) {
        case HalProgrammableCoreType::TENSIX: {
            for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
                auto [_, num_build_states] = BuildEnvManager::get_instance().get_build_index_and_state_count(
                    core_type_idx, processor_class, true);
                for (uint32_t riscv_id = 0; riscv_id < num_build_states; riscv_id++) {
                    auto fw_path = BuildEnvManager::get_instance().get_firmware_binary_path(
                        device_id, core_type_idx, processor_class, riscv_id);
                    const ll_api::memory& binary_mem = llrt::get_risc_binary(fw_path);
                    uint32_t fw_size = binary_mem.get_text_size();
                    hal_.set_iram_text_size(
                        launch_msg, core_type, static_cast<HalProcessorClassType>(processor_class), riscv_id, fw_size);

                    if (not rtoptions_.get_skip_loading_fw()) {
                        llrt::test_load_multicast_write_risc_binary(
                            binary_mem,
                            device_id,
                            start_core,
                            end_core.value(),
                            core_type_idx,
                            processor_class,
                            riscv_id);
                    }
                }
            }

            if (!rtoptions_.get_fast_dispatch()) {
                launch_msg.kernel_config().mode() = dev_msgs::DISPATCH_MODE_HOST;
            } else {
                launch_msg.kernel_config().mode() = dev_msgs::DISPATCH_MODE_DEV;
            }
            prepare_initial_launch_msg();
            write_initial_go_launch_msg();
            if (rtoptions_.get_fast_dispatch() && dispatch_core_manager_.get_dispatch_core_type() == CoreType::WORKER) {
                launch_msg.kernel_config().mode() = dev_msgs::DISPATCH_MODE_HOST;
                prepare_initial_launch_msg();
                for (const auto& logical_core : dispatch_core_manager_.get_all_logical_dispatch_cores(device_id)) {
                    auto virtual_dispatch_core = cluster_.get_virtual_coordinate_from_logical_coordinates(
                        device_id, logical_core, CoreType::WORKER);
                    auto programmable_core_type = llrt::get_core_type(device_id, virtual_dispatch_core);
                    cluster_.write_core(
                        init_launch_msg_data.data(),
                        init_launch_msg_data.size(),
                        tt_cxy_pair(device_id, virtual_dispatch_core),
                        hal_.get_dev_addr(programmable_core_type, HalL1MemAddrType::LAUNCH));
                }
            }

            cluster_.noc_multicast_write(
                &jit_build_config.fw_launch_addr_value,
                sizeof(uint32_t),
                device_id,
                start_core,
                end_core.value(),
                jit_build_config.fw_launch_addr);

            break;
        }
        case HalProgrammableCoreType::ACTIVE_ETH:
        case HalProgrammableCoreType::IDLE_ETH: {
            if (!has_flag(descriptor_->fabric_manager(), tt_fabric::FabricManagerMode::INIT_FABRIC)) {
                break;
            }
            const bool is_idle_eth = core_type == HalProgrammableCoreType::IDLE_ETH;
            const bool is_active_eth = !is_idle_eth;
            tt::umd::RiscType reset_val = tt::umd::RiscType::ALL_TENSIX;
            if (is_active_eth) {
                reset_val &= ~tt::umd::RiscType::ERISC0;
            }
            if (is_idle_eth or !hal_.get_eth_fw_is_cooperative()) {
                cluster_.assert_risc_reset_at_core(tt_cxy_pair(device_id, virtual_core), reset_val);
            }
            if (not rtoptions_.get_skip_loading_fw()) {
                for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
                    auto num_build_states = hal_.get_processor_types_count(core_type_idx, processor_class);
                    for (uint32_t eriscv_id = 0; eriscv_id < num_build_states; eriscv_id++) {
                        auto fw_path = BuildEnvManager::get_instance().get_firmware_binary_path(
                            device_id, core_type_idx, processor_class, eriscv_id);
                        const ll_api::memory& binary_mem = llrt::get_risc_binary(fw_path);
                        llrt::test_load_write_read_risc_binary(
                            binary_mem, device_id, virtual_core, core_type_idx, processor_class, eriscv_id);
                    }
                }
            }
            launch_msg.kernel_config().mode() = (!rtoptions_.get_fast_dispatch() or is_idle_eth)
                                                    ? dev_msgs::DISPATCH_MODE_HOST
                                                    : dev_msgs::DISPATCH_MODE_DEV;
            prepare_initial_launch_msg();
            write_initial_go_launch_msg();
            if (core_type == HalProgrammableCoreType::ACTIVE_ETH) {
                DeviceAddr mailbox_addr = hal_.get_dev_addr(core_type, HalL1MemAddrType::MAILBOX);
                auto factory = hal_.get_dev_msgs_factory(core_type);
                DeviceAddr ncrisc_halt_addr =
                    mailbox_addr + factory.offset_of<dev_msgs::mailboxes_t>(dev_msgs::mailboxes_t::Field::ncrisc_halt);
                std::vector<uint8_t> data(factory.size_of<dev_msgs::ncrisc_halt_msg_t>(), 0);
                cluster_.write_core(data.data(), data.size(), tt_cxy_pair(device_id, virtual_core), ncrisc_halt_addr);
            }

            if (hal_.get_eth_fw_is_cooperative() || core_type != HalProgrammableCoreType::ACTIVE_ETH ||
                !rtoptions_.get_enable_2_erisc_mode()) {
                cluster_.write_core(
                    &jit_build_config.fw_launch_addr_value,
                    sizeof(uint32_t),
                    tt_cxy_pair(device_id, virtual_core),
                    jit_build_config.fw_launch_addr);
            } else {
                constexpr uint32_t mailbox_index = 0;
                tt::llrt::internal_::send_msg_to_eth_mailbox(
                    device_id,
                    virtual_core,
                    tt_metal::FWMailboxMsg::ETH_MSG_RELEASE_CORE,
                    mailbox_index,
                    {/*l1 addr to exec*/ jit_build_config.fw_launch_addr_value},
                    false);
            }

            break;
        }
        default:
            TT_THROW(
                "Unsupported programable core type {} to initialize build states", enchantum::to_string(core_type));
    }
}

void RiscFirmwareInitializer::initialize_and_launch_firmware(tt::ChipId device_id) {
    ZoneScoped;

    log_debug(tt::LogMetal, "Initializing worker cores");
    std::unordered_set<CoreCoord> not_done_cores;
    CoreCoord logical_grid_size = cluster_.get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);

    auto dev_msgs_factory = hal_.get_dev_msgs_factory(HalProgrammableCoreType::TENSIX);
    auto core_info = populate_core_info_msg(device_id, HalProgrammableCoreType::TENSIX);
    auto launch_msg = dev_msgs_factory.create<dev_msgs::launch_msg_t>();
    auto go_msg = dev_msgs_factory.create<dev_msgs::go_msg_t>();
    go_msg.view().signal() = dev_msgs::RUN_MSG_INIT;

    for (uint32_t y = 0; y < logical_grid_size.y; y++) {
        for (uint32_t x = 0; x < logical_grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core =
                cluster_.get_virtual_coordinate_from_logical_coordinates(device_id, logical_core, CoreType::WORKER);
            core_info.view().absolute_logical_x() = logical_core.x;
            core_info.view().absolute_logical_y() = logical_core.y;
            cluster_.write_core_immediate(
                core_info.data(),
                core_info.size(),
                {static_cast<size_t>(device_id), worker_core},
                hal_.get_dev_addr(llrt::get_core_type(device_id, worker_core), HalL1MemAddrType::CORE_INFO));
            not_done_cores.insert(worker_core);
        }
    }
    CoreCoord start_core =
        cluster_.get_virtual_coordinate_from_logical_coordinates(device_id, CoreCoord(0, 0), CoreType::WORKER);
    CoreCoord end_core = cluster_.get_virtual_coordinate_from_logical_coordinates(
        device_id, CoreCoord(logical_grid_size.x - 1, logical_grid_size.y - 1), CoreType::WORKER);
    initialize_firmware(
        device_id, HalProgrammableCoreType::TENSIX, start_core, launch_msg.view(), go_msg.view(), end_core);

    for (const auto& eth_core : this->get_control_plane_().get_active_ethernet_cores(device_id)) {
        static std::vector<uint32_t> zero_vec_erisc_init(
            hal_.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::APP_SYNC_INFO) / sizeof(uint32_t),
            0);

        CoreCoord virtual_core =
            cluster_.get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);

        cluster_.write_core_immediate(
            device_id,
            virtual_core,
            zero_vec_erisc_init,
            hal_.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::APP_SYNC_INFO));
    }

    log_debug(tt::LogMetal, "Initializing active ethernet cores");
    dev_msgs_factory = hal_.get_dev_msgs_factory(HalProgrammableCoreType::ACTIVE_ETH);
    core_info = populate_core_info_msg(device_id, HalProgrammableCoreType::ACTIVE_ETH);
    launch_msg = dev_msgs_factory.create<dev_msgs::launch_msg_t>();
    go_msg = dev_msgs_factory.create<dev_msgs::go_msg_t>();
    go_msg.view().signal() = dev_msgs::RUN_MSG_INIT;

    std::unordered_set<CoreCoord> multi_risc_active_eth_cores;
    for (const auto& eth_core : this->get_control_plane_().get_active_ethernet_cores(device_id)) {
        CoreCoord virtual_core =
            cluster_.get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);
        core_info.view().absolute_logical_x() = eth_core.x;
        core_info.view().absolute_logical_y() = eth_core.y;
        cluster_.write_core_immediate(
            core_info.data(),
            core_info.size(),
            {static_cast<size_t>(device_id), virtual_core},
            hal_.get_dev_addr(llrt::get_core_type(device_id, virtual_core), HalL1MemAddrType::CORE_INFO));
        initialize_firmware(
            device_id, HalProgrammableCoreType::ACTIVE_ETH, virtual_core, launch_msg.view(), go_msg.view());
        if (!hal_.get_eth_fw_is_cooperative()) {
            multi_risc_active_eth_cores.insert(virtual_core);
            not_done_cores.insert(virtual_core);
        }
    }

    log_debug(tt::LogMetal, "Initializing idle ethernet cores");
    dev_msgs_factory = hal_.get_dev_msgs_factory(HalProgrammableCoreType::IDLE_ETH);
    core_info = populate_core_info_msg(device_id, HalProgrammableCoreType::IDLE_ETH);
    launch_msg = dev_msgs_factory.create<dev_msgs::launch_msg_t>();
    go_msg = dev_msgs_factory.create<dev_msgs::go_msg_t>();
    go_msg.view().signal() = dev_msgs::RUN_MSG_INIT;
    for (const auto& eth_core : this->get_control_plane_().get_inactive_ethernet_cores(device_id)) {
        CoreCoord virtual_core =
            cluster_.get_virtual_coordinate_from_logical_coordinates(device_id, eth_core, CoreType::ETH);
        core_info.view().absolute_logical_x() = eth_core.x;
        core_info.view().absolute_logical_y() = eth_core.y;
        cluster_.write_core_immediate(
            core_info.data(),
            core_info.size(),
            {static_cast<size_t>(device_id), virtual_core},
            hal_.get_dev_addr(llrt::get_core_type(device_id, virtual_core), HalL1MemAddrType::CORE_INFO));
        initialize_firmware(
            device_id, HalProgrammableCoreType::IDLE_ETH, virtual_core, launch_msg.view(), go_msg.view());
        not_done_cores.insert(virtual_core);
    }

    cluster_.l1_barrier(device_id);

    for (const auto& worker_core : not_done_cores) {
        if (multi_risc_active_eth_cores.contains(worker_core) && rtoptions_.get_enable_2_erisc_mode()) {
            continue;
        }

        tt::umd::RiscType reset_val;
        if (cluster_.arch() == ARCH::QUASAR) {
            reset_val = tt::umd::RiscType::ALL_NEO_DMS;
        } else {
            reset_val = tt::umd::RiscType::BRISC;
            if (multi_risc_active_eth_cores.contains(worker_core)) {
                reset_val |= tt::umd::RiscType::ERISC1;
            }
        }
        cluster_.deassert_risc_reset_at_core(tt_cxy_pair(device_id, worker_core), reset_val);
    }

    log_debug(LogDevice, "Waiting for firmware init complete");
    const int timeout_ms = 10000;
    try {
        llrt::internal_::wait_until_cores_done(device_id, dev_msgs::RUN_MSG_INIT, not_done_cores, timeout_ms);
    } catch (std::runtime_error&) {
        TT_THROW("Device {} init: failed to initialize FW! Try resetting the board.", device_id);
    }
    log_debug(LogDevice, "Firmware init complete");
}

}  // namespace tt::tt_metal
