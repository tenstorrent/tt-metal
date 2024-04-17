// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/trace/trace.hpp"
#include "tt_metal/common/core_descriptor.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "impl/debug/dprint_server.hpp"
#include "impl/debug/watcher_server.hpp"
#include "tt_metal/third_party/umd/device/util.hpp"

#include "common/env_lib.hpp"
#include "common/utils.hpp"
#include "llrt/llrt.hpp"
#include "dev_msgs.h"

namespace tt {

namespace tt_metal {

void ::detail::ProgramDeleter::operator()(Program *p) {
    delete p;
}

ActiveDevices Device::active_devices_;

ActiveDevices::ActiveDevices() {
}

ActiveDevices::~ActiveDevices() {
    for (size_t i = 0; i < active_devices_.size(); i++) {
        if (active_devices_[i] == ActiveState::ACTIVE) {
            TT_THROW("Process tear down with device {} still active", i);
        }
    }
}

bool ActiveDevices::activate_device(chip_id_t id) {
    bool already_initialized;
    const std::lock_guard<std::mutex> lock(lock_);
    if (this->active_devices_.size() < id + 1) {
        this->active_devices_.resize(id + 1);
        already_initialized = false;
    } else if (this->active_devices_[id] == ActiveState::ACTIVE) {
        TT_THROW("Cannot re-initialize device {}, must first call close()", id);
    } else {
        already_initialized = (this->active_devices_[id] == ActiveState::INACTIVE) ? true : false;
    }
    this->active_devices_[id] = ActiveState::ACTIVE;

    return already_initialized;
}

void ActiveDevices::deactivate_device(chip_id_t id) {
    const std::lock_guard<std::mutex> lock(lock_);
    this->active_devices_[id] = ActiveState::INACTIVE;
}

Device::Device(
    chip_id_t device_id, const uint8_t num_hw_cqs, size_t l1_small_size, const std::vector<uint32_t> &l1_bank_remap) :
    id_(device_id), num_hw_cqs_(num_hw_cqs), work_executor(device_id) {
    ZoneScoped;
    TT_ASSERT(num_hw_cqs > 0 and num_hw_cqs < 3, "num_hw_cqs can be between 1 and 2");
    this->initialize(l1_small_size, l1_bank_remap);
}

void Device::initialize_cluster() {
    ZoneScoped;
    if (llrt::OptionsG.get_clear_l1()) {
        this->clear_l1_state();
    }
#ifdef TT_METAL_VERSIM_DISABLED
    int ai_clk = tt::Cluster::instance().get_device_aiclk(this->id_);
    log_info(tt::LogMetal, "AI CLK for device {} is:   {} MHz", this->id_, ai_clk);
#endif
}

void Device::initialize_allocator(size_t l1_small_size, const std::vector<uint32_t> &l1_bank_remap) {
    ZoneScoped;
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    // Construct allocator config from soc_desc
    AllocatorConfig config(
        {.num_dram_channels = static_cast<size_t>(soc_desc.get_num_dram_channels()),
         .dram_bank_size = soc_desc.dram_bank_size,
         .dram_bank_offsets = {},
         .worker_grid_size = this->logical_grid_size(),
         .worker_l1_size = static_cast<size_t>(soc_desc.worker_l1_size),
         .l1_bank_size = static_cast<size_t>(get_storage_core_bank_size(this->id_, this->num_hw_cqs_)),
         .l1_small_size = l1_small_size,
         .core_type_from_noc_coord_table = {},  // Populated later
         .worker_log_to_physical_routing_x = soc_desc.worker_log_to_physical_routing_x,
         .worker_log_to_physical_routing_y = soc_desc.worker_log_to_physical_routing_y,
         .l1_bank_remap = l1_bank_remap,
         .compute_grid_size = this->compute_with_storage_grid_size()});
    TT_FATAL(config.l1_small_size < config.l1_bank_size, "Reserved size must be less than bank size");
    TT_FATAL(
        config.l1_small_size % ADDRESS_ALIGNMENT == 0,
        "Reserved size must be aligned to ADDRESS_ALIGNMENT",
        ADDRESS_ALIGNMENT);
    // Initialize dram_offsets from soc_descriptor
    for (auto channel = 0; channel < soc_desc.get_num_dram_channels(); channel++) {
        config.dram_bank_offsets.push_back(soc_desc.get_address_offset(channel));
    }
    // Initialize core_type_from_noc_coord_table table
    for (const auto& core: soc_desc.physical_cores) {
        config.core_type_from_noc_coord_table.insert({core.first, AllocCoreType::Invalid});
    }

    for (const CoreCoord& core : tt::get_logical_compute_cores(id_, num_hw_cqs_)) {
        this->compute_cores_.insert(core);
        const auto noc_coord = this->worker_core_from_logical_core(core);
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::ComputeAndStore;
    }
    for (const CoreCoord& core : tt::get_logical_storage_cores(id_, num_hw_cqs_)) {
        this->storage_only_cores_.insert(core);
        const auto noc_coord = this->worker_core_from_logical_core(core);
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::StorageOnly;
    }
    for (const CoreCoord& core : tt::get_logical_dispatch_cores(id_, num_hw_cqs_)) {
        CoreType dispatch_core_type = tt::get_dispatch_core_type(id_, num_hw_cqs_);
        const auto noc_coord = this->physical_core_from_logical_core(core, dispatch_core_type);
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::Dispatch;
    }
    for (const auto &core : soc_desc.get_logical_ethernet_cores()) {
        this->ethernet_cores_.insert(core);
    }

    // L1_BANKING scheme creates 1 bank per DRAM core and splits up L1 such that there are power 2 num L1 banks
    // This is the only allocator scheme supported because kernel APIs assume num L1 banks are power of 2
    TT_ASSERT(this->allocator_scheme_ == MemoryAllocator::L1_BANKING);
    this->allocator_ = std::make_unique<L1BankingAllocator>(config);
}

void Device::initialize_build() {
    ZoneScoped;

    this->build_env_.init(this->id(), this->arch());

    auto init_helper = [this] (bool is_fw) -> JitBuildStateSet {
        std::vector<std::shared_ptr<JitBuildState>> build_states;

        build_states.resize(arch() == tt::ARCH::GRAYSKULL ? 5 : 7);

        build_states[build_processor_type_to_index(JitBuildProcessorType::DATA_MOVEMENT).first + 0] =
            std::make_shared<JitBuildDataMovement>(this->build_env_, 0, is_fw);
        build_states[build_processor_type_to_index(JitBuildProcessorType::DATA_MOVEMENT).first + 1] =
            std::make_shared<JitBuildDataMovement>(this->build_env_, 1, is_fw);
        build_states[build_processor_type_to_index(JitBuildProcessorType::COMPUTE).first + 0] =
            std::make_shared<JitBuildCompute>(this->build_env_, 0, is_fw);
        build_states[build_processor_type_to_index(JitBuildProcessorType::COMPUTE).first + 1] =
            std::make_shared<JitBuildCompute>(this->build_env_, 1, is_fw);
        build_states[build_processor_type_to_index(JitBuildProcessorType::COMPUTE).first + 2] =
            std::make_shared<JitBuildCompute>(this->build_env_, 2, is_fw);

        if (arch() != tt::ARCH::GRAYSKULL) {
            build_states[build_processor_type_to_index(JitBuildProcessorType::ETHERNET).first + 0] =
                std::make_shared<JitBuildEthernet>(this->build_env_, 0, is_fw);
            build_states[build_processor_type_to_index(JitBuildProcessorType::ETHERNET).first + 1] =
                std::make_shared<JitBuildEthernet>(this->build_env_, 1, is_fw);
        }

       return build_states;
    };

    this->firmware_build_states_ = init_helper(true);
    this->kernel_build_states_ = init_helper(false);
}

void Device::build_firmware() {
    ZoneScoped;

    detail::GenerateDeviceHeaders(this, this->build_env_.get_out_firmware_root_path());
    jit_build_set(this->firmware_build_states_, nullptr, "");
}

void Device::initialize_firmware(CoreCoord phys_core, launch_msg_t *launch_msg) {
    ZoneScoped;

    if (llrt::is_ethernet_core(phys_core, this->id())) {
        //ethernet core.
        //Determine if its a connected or unconnected ethernet core.
        //Unconnected ethernet cores will get idle_erisc fw.
        auto active_eth_cores = this->get_active_ethernet_cores();

        if (active_eth_cores.find(logical_core_from_ethernet_core(phys_core)) != active_eth_cores.end()) {
            int eriscv_id = build_processor_type_to_index(JitBuildProcessorType::ETHERNET).first + 0;
            ll_api::memory binary_mem = llrt::get_risc_binary(firmware_build_states_[eriscv_id]->get_target_out_path(""));
            uint32_t kernel_size16 = llrt::get_binary_code_size16(binary_mem, eriscv_id);
            log_debug(LogDevice, "ERISC fw binary size: {} in bytes", kernel_size16 * 16);
            llrt::test_load_write_read_risc_binary(binary_mem, this->id(), phys_core, eriscv_id);
            llrt::launch_erisc_app_fw_on_core(this->id(), phys_core);
        } else {
            tt::Cluster::instance().assert_risc_reset_at_core(tt_cxy_pair(this->id(), phys_core));
            int eriscv_id = build_processor_type_to_index(JitBuildProcessorType::ETHERNET).first + 1;
            ll_api::memory binary_mem = llrt::get_risc_binary(firmware_build_states_[eriscv_id]->get_target_out_path(""));
            uint32_t kernel_size16 = llrt::get_binary_code_size16(binary_mem, eriscv_id);
            log_debug(LogDevice, "ERISC fw binary size: {} in bytes", kernel_size16 * 16);
            llrt::test_load_write_read_risc_binary(binary_mem, this->id(), phys_core, eriscv_id);
            llrt::program_risc_startup_addr(this->id(), phys_core);
        }
    } else {
        llrt::program_risc_startup_addr(this->id(), phys_core);
        for (int riscv_id = 0; riscv_id < 5; riscv_id++) {
            ll_api::memory binary_mem =
                llrt::get_risc_binary(firmware_build_states_[riscv_id]->get_target_out_path(""));
            uint32_t kernel_size16 = llrt::get_binary_code_size16(binary_mem, riscv_id);
            if (riscv_id == 1) {
                launch_msg->ncrisc_kernel_size16 = kernel_size16;
            }
            log_debug(LogDevice, "RISC {} fw binary size: {} in bytes", riscv_id, kernel_size16 * 16);
            llrt::test_load_write_read_risc_binary(binary_mem, this->id(), phys_core, riscv_id);
        }
    }
    //This is an initialization launch message.
    //Clears launch message fields to 0 in target core L1.
    //Sets launch.run to RUN_MSG_INIT.
    llrt::write_launch_msg_to_core(this->id(), phys_core, launch_msg);
}

void Device::initialize_and_launch_firmware() {
    ZoneScoped;

    launch_msg_t launch_msg = {
        .brisc_watcher_kernel_id = 0,
        .ncrisc_watcher_kernel_id = 0,
        .triscs_watcher_kernel_id = 0,
        .ncrisc_kernel_size16 = 0,
        .mode = DISPATCH_MODE_HOST,
        .brisc_noc_id = 0,
        .enable_brisc = 0,
        .enable_ncrisc = 0,
        .enable_triscs = 0,
        .enable_erisc = 0,
        .run = RUN_MSG_INIT,
    };

    // Download to worker cores
    log_debug("Initializing firmware");
    CoreCoord grid_size = this->logical_grid_size();
    std::unordered_set<CoreCoord> not_done_cores;


    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            if (!this->storage_only_cores_.count(logical_core)) {
                CoreCoord worker_core = this->worker_core_from_logical_core(logical_core);
                this->initialize_firmware(worker_core, &launch_msg);
                not_done_cores.insert(worker_core);
            }
        }
    }

    // Load erisc app base FW to eth cores
    for (const auto &eth_core : this->get_active_ethernet_cores()) {
        CoreCoord phys_eth_core = this->ethernet_core_from_logical_core(eth_core);
        this->initialize_firmware(phys_eth_core, &launch_msg);
    }

    for (const auto &eth_core : this->get_inactive_ethernet_cores()) {
        CoreCoord phys_eth_core = this->ethernet_core_from_logical_core(eth_core);
        this->initialize_firmware(phys_eth_core, &launch_msg);
        not_done_cores.insert(phys_eth_core);
    }

    // Barrier between L1 writes above and deassert below
    tt::Cluster::instance().l1_barrier(this->id());

    // Deassert worker cores
    for(const auto& worker_core : not_done_cores)
        tt::Cluster::instance().deassert_risc_reset_at_core(tt_cxy_pair(this->id(), worker_core));

    // Wait until fw init is done, ensures the next launch msg doesn't get
    // written while fw is still in init
    log_debug("Waiting for firmware init complete");
    llrt::internal_::wait_until_cores_done(this->id(), RUN_MSG_INIT, not_done_cores);
    log_debug("Firmware init complete");
}

void Device::clear_l1_state() {
    CoreCoord logical_grid_size = this->logical_grid_size();
    TT_ASSERT(this->l1_size_per_core() % sizeof(uint32_t) == 0);
    std::vector<uint32_t> zero_vec(this->l1_size_per_core() / sizeof(uint32_t), 0);
    constexpr uint32_t start_address = 0;
    for (uint32_t x = 0; x < logical_grid_size.x; x++) {
        for (uint32_t y = 0; y < logical_grid_size.y; y++) {
            CoreCoord logical_core(x, y);
            detail::WriteToDeviceL1(this, logical_core, start_address, zero_vec);
        }
    }

    // Clear erisc sync info
    for (const auto &eth_core : this->get_active_ethernet_cores()) {
        CoreCoord physical_core = this->ethernet_core_from_logical_core(eth_core);
        // These L1 ranges are restricted becase UMD base routing FW uses L1 below FIRMWARE_BASE and
        // between TILE_HEADER_BUFFER_BASE to COMMAND_Q_BASE
        std::vector<uint32_t> zero_vec_above_tile_header_buffer(
            (eth_l1_mem::address_map::MAX_L1_LOADING_SIZE - eth_l1_mem::address_map::TILE_HEADER_BUFFER_BASE) /
                sizeof(uint32_t),
            0);

        llrt::write_hex_vec_to_core(
            this->id(),
            physical_core,
            zero_vec_above_tile_header_buffer,
            eth_l1_mem::address_map::TILE_HEADER_BUFFER_BASE);

        /* TODO: removing this section of code fixes the n300 hangs, what's the proper fix?
        std::vector<uint32_t> zero_vec_below_command_q_base(
            (eth_l1_mem::address_map::COMMAND_Q_BASE - eth_l1_mem::address_map::FIRMWARE_BASE) / sizeof(uint32_t), 0);

        llrt::write_hex_vec_to_core(
            this->id(), physical_core, zero_vec_below_command_q_base, eth_l1_mem::address_map::FIRMWARE_BASE);
        */
    }
    // TODO: clear idle eriscs as well
}

// TODO (abhullar): Refactor this with #2593 to allow each target fast dispatch (FD) device to program their associated FD cores regardless of whether they are on the target device or not.
// Currently we have to program FD cores for the remote device when initializing the MMIO device because completion queue cores are on MMIO device
//  and we don't have handle on MMIO device when initializing the remote device
void Device::compile_command_queue_programs() {
    ZoneScoped;
    unique_ptr<Program, detail::ProgramDeleter> command_queue_program_ptr(new Program);

    std::string prefetch_kernel_path = "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp";
    std::string dispatch_kernel_path = "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp";

    // TODO: These are semaphore IDs, remove these when CreateSemaphore returns ID rather than address
    constexpr uint32_t prefetch_sync_sem = 0;
    constexpr uint32_t prefetch_downstream_cb_sem = 1;
    constexpr uint32_t dispatch_sync_sem = 0;
    constexpr uint32_t dispatch_cb_sem = 1;

    constexpr uint32_t prefetch_d_upstream_cb_sem = 1;
    constexpr uint32_t prefetch_d_downstream_cb_sem = 2;
    constexpr uint32_t prefetch_h_exec_buf_sem = 2;

    if (this->is_mmio_capable()) {
        for (const chip_id_t &device_id : tt::Cluster::instance().get_devices_controlled_by_mmio_device(this->id())) {
            if (device_id != this->id()) {
                continue; // REMOVE WHEN R CHIP IS SUPPORTED
            }
            // TODO (abhullar): allow for multiple cqs on remote device, atm device initialization asserts one cq for the remote device
            uint8_t num_hw_cqs = device_id == this->id() ? this->num_hw_cqs() : 1;
            uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
            uint32_t cq_size = this->sysmem_manager().get_cq_size();

            for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
                CoreType dispatch_core_type = dispatch_core_manager::get(num_hw_cqs).get_dispatch_core_type(device_id);

                tt_cxy_pair prefetch_location = dispatch_core_manager::get(num_hw_cqs).prefetcher_core(device_id, channel, cq_id);
                tt_cxy_pair completion_q_writer_location = dispatch_core_manager::get(num_hw_cqs).completion_queue_writer_core(device_id, channel, cq_id);
                tt_cxy_pair dispatch_location = dispatch_core_manager::get(num_hw_cqs).dispatcher_core(device_id, channel, cq_id);

                TT_ASSERT(prefetch_location.chip == this->id() and completion_q_writer_location.chip == this->id(),
                    "Issue queue interface is on device {} and completion queue interface is on device {} but they are expected to be on device {}", prefetch_location.chip, completion_q_writer_location.chip, this->id());

                CoreCoord prefetch_physical_core = get_physical_core_coordinate(prefetch_location, dispatch_core_type);
                CoreCoord completion_q_physical_core = get_physical_core_coordinate(completion_q_writer_location, dispatch_core_type);
                CoreCoord dispatch_physical_core = get_physical_core_coordinate(dispatch_location, dispatch_core_type);

                log_debug(LogDevice, "Dispatching out of {} cores",  magic_enum::enum_name(dispatch_core_type));
                log_debug(LogDevice, "Prefetch HD logical location: {} physical core: {}", prefetch_location.str(), prefetch_physical_core.str());
                log_debug(LogDevice, "Dispatch HD logical location: {} physical core {}", dispatch_location.str(), dispatch_physical_core.str());

                uint32_t command_queue_start_addr = get_absolute_cq_offset(channel, cq_id, cq_size);
                uint32_t issue_queue_start_addr = command_queue_start_addr + CQ_START;
                uint32_t issue_queue_size = this->sysmem_manager_->get_issue_queue_size(cq_id);
                uint32_t completion_queue_start_addr = issue_queue_start_addr + issue_queue_size;
                uint32_t completion_queue_size = this->sysmem_manager_->get_completion_queue_size(cq_id);

                TT_ASSERT(tt::Cluster::instance().get_soc_desc(this->id()).pcie_cores.size() == 1);
                CoreCoord pcie_physical_core = tt::Cluster::instance().get_soc_desc(this->id()).pcie_cores.at(0);

                std::map<string, string> prefetch_defines = {
                    {"DISPATCH_KERNEL", "1"},
                    {"MY_NOC_X", std::to_string(prefetch_physical_core.x)},
                    {"MY_NOC_Y", std::to_string(prefetch_physical_core.y)},
                    {"UPSTREAM_NOC_X", std::to_string(0)},
                    {"UPSTREAM_NOC_Y", std::to_string(0)},
                    {"DOWNSTREAM_NOC_X", std::to_string(dispatch_physical_core.x)},
                    {"DOWNSTREAM_NOC_Y", std::to_string(dispatch_physical_core.y)},
                };

                std::vector<uint32_t> prefetch_compile_args = {
                    dispatch_constants::DISPATCH_BUFFER_BASE,
                    dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE,
                    dispatch_constants::get(dispatch_core_type).dispatch_buffer_pages(),
                    prefetch_downstream_cb_sem,
                    dispatch_cb_sem,
                    issue_queue_start_addr,
                    issue_queue_size,
                    dispatch_constants::PREFETCH_Q_BASE,
                    dispatch_constants::PREFETCH_Q_SIZE,
                    CQ_PREFETCH_Q_RD_PTR,
                    dispatch_constants::CMDDAT_Q_BASE,
                    dispatch_constants::get(dispatch_core_type).cmddat_q_size(),
                    dispatch_constants::get(dispatch_core_type).scratch_db_base(),
                    dispatch_constants::get(dispatch_core_type).scratch_db_size(),
                    prefetch_sync_sem,
                    dispatch_constants::PREFETCH_D_BUFFER_PAGES, // prefetch_d only
                    prefetch_d_upstream_cb_sem, // prefetch_d only
                    prefetch_downstream_cb_sem, // prefetch_d only
                    dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE,
                    dispatch_constants::PREFETCH_D_BUFFER_BLOCKS, // prefetch_d only
                    prefetch_h_exec_buf_sem,
                    true,   // is_dram_variant
                    true    // is_host_variant
                };

                if (dispatch_core_type == CoreType::WORKER) {
                    tt::tt_metal::CreateKernel(
                        *command_queue_program_ptr, prefetch_kernel_path, prefetch_location,
                        DataMovementConfig{
                            .processor = DataMovementProcessor::RISCV_1,
                            .noc = NOC::NOC_0,
                            .compile_args = prefetch_compile_args,
                            .defines = prefetch_defines});
                } else {
                    tt::tt_metal::CreateKernel(
                        *command_queue_program_ptr, prefetch_kernel_path, prefetch_location,
                        EthernetConfig{
                            .eth_mode = Eth::IDLE,
                            .noc = NOC::NOC_0,
                            .compile_args = prefetch_compile_args,
                            .defines = prefetch_defines});
                }

                tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, prefetch_location, 0, dispatch_core_type);
                tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, prefetch_location, dispatch_constants::get(dispatch_core_type).dispatch_buffer_pages(), dispatch_core_type);
                tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, prefetch_location, 0, dispatch_core_type);

                if (device_id == this->id()) {
                    std::map<string, string> dispatch_defines = {
                        {"DISPATCH_KERNEL", "1"},
                        {"MY_NOC_X", std::to_string(dispatch_physical_core.x)},
                        {"MY_NOC_Y", std::to_string(dispatch_physical_core.y)},
                        {"UPSTREAM_NOC_X", std::to_string(prefetch_physical_core.x)},
                        {"UPSTREAM_NOC_Y", std::to_string(prefetch_physical_core.y)},
                        {"DOWNSTREAM_NOC_X", std::to_string(0)},
                        {"DOWNSTREAM_NOC_Y", std::to_string(0)},
                    };
                    std::vector<uint32_t> dispatch_compile_args = {
                        dispatch_constants::DISPATCH_BUFFER_BASE,
                        dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE,
                        dispatch_constants::get(dispatch_core_type).dispatch_buffer_pages(),
                        dispatch_cb_sem,
                        prefetch_downstream_cb_sem,
                        dispatch_constants::DISPATCH_BUFFER_SIZE_BLOCKS,
                        prefetch_sync_sem,
                        command_queue_start_addr,
                        completion_queue_start_addr,
                        completion_queue_size,
                        dispatch_constants::DISPATCH_BUFFER_BASE,
                        dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE * dispatch_constants::get(dispatch_core_type).dispatch_buffer_pages(),
                        0, // unused on hd, filled in below for h and d
                        0, // unused on hd, filled in below for h and d
                        0, // unused unless tunneler is between h and d
                        true,   // is_dram_variant
                        true    // is_host_variant
                    };

                    if (dispatch_core_type == CoreType::WORKER) {
                        tt::tt_metal::CreateKernel(
                            *command_queue_program_ptr, dispatch_kernel_path, dispatch_location,
                            DataMovementConfig{
                                .processor = DataMovementProcessor::RISCV_1,
                                .noc = NOC::NOC_0,
                                .compile_args = dispatch_compile_args,
                                .defines = dispatch_defines});
                    } else {
                        tt::tt_metal::CreateKernel(
                            *command_queue_program_ptr, dispatch_kernel_path, dispatch_location,
                            EthernetConfig{
                                .eth_mode = Eth::IDLE,
                                .noc = NOC::NOC_0,
                                .compile_args = dispatch_compile_args,
                                .defines = dispatch_defines});
                    }

                    tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, dispatch_location, 0, dispatch_core_type);
                    tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, dispatch_location, 0, dispatch_core_type);
                    tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, dispatch_location, dispatch_constants::get(dispatch_core_type).dispatch_buffer_pages(), dispatch_core_type);

                } else {
                    TT_THROW("FD2.0 does not support R chip yet");
                }
            }
        }
    } else {
        TT_THROW("FD2.0 does not support R chip yet");
    }
    detail::CompileProgram(this, *command_queue_program_ptr);
    this->command_queue_programs.push_back(std::move(command_queue_program_ptr));
}

// Writes issue and completion queue pointers to device and in sysmem and loads fast dispatch program onto dispatch cores
void Device::configure_command_queue_programs() {
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->id());

    std::vector<uint32_t> zero = {0x0}; // Reset state in case L1 Clear is disabled.
    std::vector<uint32_t> pointers(CQ_START / sizeof(uint32_t), 0);
    uint32_t cq_size = this->sysmem_manager().get_cq_size();

    TT_ASSERT(this->command_queue_programs.size() == 1);
    Program& command_queue_program = *this->command_queue_programs[0];

    for (uint8_t cq_id = 0; cq_id < this->num_hw_cqs(); cq_id++) {
        // Reset the host manager's pointer for this command queue
        this->sysmem_manager_->reset(cq_id);

        pointers[HOST_CQ_ISSUE_READ_PTR / sizeof(uint32_t)] = (CQ_START + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4;
        pointers[HOST_CQ_COMPLETION_WRITE_PTR / sizeof(uint32_t)] = (CQ_START + this->sysmem_manager_->get_issue_queue_size(cq_id) + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4;

        tt::Cluster::instance().write_sysmem(pointers.data(), pointers.size() * sizeof(uint32_t), cq_id * cq_size, mmio_device_id, channel);
    }

    if (this->is_mmio_capable()) {
        for (const chip_id_t &device_id : tt::Cluster::instance().get_devices_controlled_by_mmio_device(this->id())) {
            if (device_id != this->id()) {
                continue; // UPDATE THIS FOR R CHIP SUPPORT!
            }

            uint8_t curr_num_hw_cqs = device_id == this->id() ? this->num_hw_cqs() : 1;
            uint16_t curr_channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
            uint32_t curr_cq_size = this->sysmem_manager().get_cq_size();

            for (uint8_t cq_id = 0; cq_id < curr_num_hw_cqs; cq_id++) {
                tt_cxy_pair prefetch_location = dispatch_core_manager::get(curr_num_hw_cqs).prefetcher_core(device_id, curr_channel, cq_id);
                tt_cxy_pair completion_q_writer_location = dispatch_core_manager::get(curr_num_hw_cqs).completion_queue_writer_core(device_id, curr_channel, cq_id);
                tt_cxy_pair dispatch_location = dispatch_core_manager::get(curr_num_hw_cqs).dispatcher_core(device_id, channel, cq_id);
                CoreType dispatch_core_type = dispatch_core_manager::get(curr_num_hw_cqs).get_dispatch_core_type(device_id);

                TT_ASSERT(prefetch_location.chip == this->id() and completion_q_writer_location.chip == this->id(),
                    "Issue queue interface is on device {} and completion queue interface is on device {} but they are expected to be on device {}", prefetch_location.chip, completion_q_writer_location.chip, this->id());

                // Initialize the FetchQ
                std::vector<uint32_t> prefetch_q(dispatch_constants::PREFETCH_Q_ENTRIES, 0);
                std::vector<uint32_t> prefetch_q_rd_ptr_addr_data = {
                    (uint32_t)(dispatch_constants::PREFETCH_Q_BASE + dispatch_constants::PREFETCH_Q_SIZE)
                };
                detail::WriteToDeviceL1(this, prefetch_location, CQ_PREFETCH_Q_RD_PTR, prefetch_q_rd_ptr_addr_data, dispatch_core_type);
                detail::WriteToDeviceL1(this, prefetch_location, dispatch_constants::PREFETCH_Q_BASE, prefetch_q, dispatch_core_type);

                // Initialize completion queue write pointer and read pointer copy
                uint32_t issue_queue_size = this->sysmem_manager_->get_issue_queue_size(cq_id);
                uint32_t completion_queue_start_addr = CQ_START + issue_queue_size + get_absolute_cq_offset(curr_channel, cq_id, curr_cq_size);
                uint32_t completion_queue_start_addr_16B = completion_queue_start_addr >> 4;
                vector<uint32_t> completion_queue_wr_ptr = {completion_queue_start_addr_16B};
                detail::WriteToDeviceL1(this, completion_q_writer_location, CQ_COMPLETION_READ_PTR, completion_queue_wr_ptr, dispatch_core_type);
                detail::WriteToDeviceL1(this, completion_q_writer_location, CQ_COMPLETION_WRITE_PTR, completion_queue_wr_ptr, dispatch_core_type);
                detail::WriteToDeviceL1(this, completion_q_writer_location, CQ0_COMPLETION_LAST_EVENT, zero, dispatch_core_type);
                detail::WriteToDeviceL1(this, completion_q_writer_location, CQ1_COMPLETION_LAST_EVENT, zero, dispatch_core_type);

                // Initialize address where workers signal to completion to dispatch core
                // This value is always increasing
                detail::WriteToDeviceL1(this, dispatch_location, DISPATCH_MESSAGE_ADDR, zero, dispatch_core_type);
            }
        }
    }
    detail::ConfigureDeviceWithProgram(this, command_queue_program, true);
    tt::Cluster::instance().l1_barrier(this->id());
}

void Device::initialize_command_queue() {
    TT_ASSERT(this->is_mmio_capable() or (not this->is_mmio_capable() and this->num_hw_cqs() == 1), "Only support one hardware command queue for fast dispatch on remote device");
    using_fast_dispatch = true;
    this->sysmem_manager_ = std::make_unique<SystemMemoryManager>(this->id_, this->num_hw_cqs());
    hw_command_queues_.resize(num_hw_cqs());
    for (size_t cq_id = 0; cq_id < num_hw_cqs(); cq_id++) {
        hw_command_queues_[cq_id] = std::make_unique<HWCommandQueue>(this, cq_id);
        // Need to do this since CommandQueue constructor is private
        sw_command_queues_.push_back(std::unique_ptr<CommandQueue>(new CommandQueue(this, cq_id)));
    }

    this->compile_command_queue_programs();
    TT_ASSERT(this->command_queue_programs.size() == 1);
    this->configure_command_queue_programs();
    Program& command_queue_program = *this->command_queue_programs[0];

    for (uint8_t cq_id = 0; cq_id < this->num_hw_cqs(); cq_id++) {
        for (const auto &[core_type, logical_dispatch_cores] : command_queue_program.logical_cores()) {
            for (const CoreCoord &logical_dispatch_core : logical_dispatch_cores) {
                launch_msg_t msg = command_queue_program.kernels_on_core(logical_dispatch_core, core_type)->launch_msg;
                tt::llrt::write_launch_msg_to_core(this->id(), this->physical_core_from_logical_core(logical_dispatch_core, core_type), &msg);
            }
        }
    }
    // Added this for safety while debugging hangs with FD v1.3 tunnel to R, should experiment with removing it
    // tt::Cluster::instance().l1_barrier(this->id());
}

void Device::initialize_synchronous_sw_cmd_queue() {
    // Initialize a single Software Command Queue for SD, using passthrough mode.
    // This queue is used for all host bound functions using the Software CQ in SD mode.
    for (size_t cq_id = 0; cq_id < num_hw_cqs(); cq_id++) {
        // Need to do this since CommandQueue constructor is private
        sw_command_queues_.push_back(std::unique_ptr<CommandQueue>(new CommandQueue(this, cq_id)));
        sw_command_queues_[cq_id]->set_mode(CommandQueue::CommandQueueMode::PASSTHROUGH);
    }
}

bool Device::initialize(size_t l1_small_size, const std::vector<uint32_t> &l1_bank_remap) {
    ZoneScoped;
    log_info(tt::LogMetal, "Initializing device {}. Program cache is NOT enabled", this->id_);
    bool already_initialized = this->active_devices_.activate_device(this->id_);
    this->initialize_cluster();
    this->initialize_allocator(l1_small_size, l1_bank_remap);
    this->initialize_build();
    if (!already_initialized) {
        this->build_firmware();
    }

    DprintServerAttach(this);
    watcher_init(this);

    this->initialize_and_launch_firmware();

    watcher_attach(this);

    // Mark initialized before compiling and sending dispatch kernels to device because compilation expects device to be initialized
    this->initialized_ = true;

    // Create system memory writer for this device to have an associated interface to hardware command queue (i.e. hugepage)
    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr) {
        detail::DispatchStateCheck(true);
        this->initialize_command_queue();
    } else {
        detail::DispatchStateCheck(false);
        this->initialize_synchronous_sw_cmd_queue();
        TT_ASSERT(this->num_hw_cqs() == 1, "num_hw_cqs must be 1 in slow dispatch");
    }

    return true;
}

bool Device::close() {
    log_info(tt::LogMetal, "Closing device {}", this->id_);
    if (not this->initialized_) {
        TT_THROW("Cannot close device {} that has not been initialized!", this->id_);
    }
    this->deallocate_buffers();
    watcher_detach(this);
    DprintServerDetach(this);

    for (const std::unique_ptr<HWCommandQueue> &hw_command_queue : hw_command_queues_) {
        hw_command_queue->terminate();
    }

    // Assert worker cores
    CoreCoord grid_size = this->logical_grid_size();
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core = this->worker_core_from_logical_core(logical_core);

            if (this->storage_only_cores_.find(logical_core) == this->storage_only_cores_.end()) {
                tt::Cluster::instance().assert_risc_reset_at_core(tt_cxy_pair(this->id(), worker_core));
            }
        }
    }

    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);

    if (llrt::OptionsG.get_clear_l1()) {
        this->clear_l1_state();
    }
    tt::Cluster::instance().l1_barrier(id_);
    allocator::clear(*this->allocator_);

    this->active_devices_.deactivate_device(this->id_);
    this->disable_and_clear_program_cache();
    this->sw_command_queues_.clear();
    this->hw_command_queues_.clear();

    this->initialized_ = false;

    return true;
}

Device::~Device() {
    if (this->initialized_) {
        this->close();
    }
}

tt::ARCH Device::arch() const {
    return tt::Cluster::instance().arch();
}

int Device::num_dram_channels() const {
    return tt::Cluster::instance().get_soc_desc(id_).get_num_dram_channels();
}

uint32_t Device::l1_size_per_core() const {
    return tt::Cluster::instance().get_soc_desc(id_).worker_l1_size;
}
uint32_t Device::dram_size_per_channel() const {
    return tt::Cluster::instance().get_soc_desc(id_).dram_bank_size;
}

CoreCoord Device::logical_grid_size() const {
    return tt::Cluster::instance().get_soc_desc(id_).worker_grid_size;
}

CoreCoord Device::compute_with_storage_grid_size() const {
    return tt::get_compute_grid_size(id_, num_hw_cqs_);
}

CoreCoord Device::physical_core_from_logical_core(const CoreCoord &logical_coord, const CoreType &core_type) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    return soc_desc.get_physical_core_from_logical_core(logical_coord, core_type);
}

CoreType Device::core_type_from_physical_core(const CoreCoord &physical_coord) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    if (soc_desc.physical_cores.find(physical_coord) == soc_desc.physical_cores.end())
        TT_THROW("Physical core {} doesn't exist in metal_SocDescriptor.", physical_coord);

    return soc_desc.physical_cores.at(physical_coord).type;
}

CoreCoord Device::worker_core_from_logical_core(const CoreCoord &logical_core) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    return soc_desc.get_physical_tensix_core_from_logical(logical_core);
}

std::vector<CoreCoord> Device::worker_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const {
    std::vector<CoreCoord> worker_cores(logical_cores.size());
    for (std::size_t idx = 0; idx < logical_cores.size(); idx++)
        worker_cores[idx] = worker_core_from_logical_core(logical_cores[idx]);

    return worker_cores;
}

CoreCoord Device::ethernet_core_from_logical_core(const CoreCoord &logical_core) const {
    return tt::Cluster::instance().ethernet_core_from_logical_core(id_, logical_core);
}

CoreCoord Device::logical_core_from_ethernet_core(const CoreCoord &physical_core) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    return soc_desc.get_logical_ethernet_core_from_physical(physical_core);
}

std::vector<CoreCoord> Device::ethernet_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const {
    std::vector<CoreCoord> ethernet_cores(logical_cores.size());

    for (std::size_t idx = 0; idx < logical_cores.size(); idx++)
        ethernet_cores[idx] = ethernet_core_from_logical_core(logical_cores[idx]);
    return ethernet_cores;
}

void Device::check_allocator_is_initialized() const {
    if (this->allocator_ == nullptr) {
        TT_THROW("No memory allocator! Device has not been initialized, did you forget to call InitializeDevice?");
    }
}

uint32_t Device::num_banks(const BufferType &buffer_type) const {
    this->check_allocator_is_initialized();
    return allocator::num_banks(*this->allocator_, buffer_type);
}

uint32_t Device::bank_size(const BufferType &buffer_type) const {
    this->check_allocator_is_initialized();
    return allocator::bank_size(*this->allocator_, buffer_type);
}

uint32_t Device::dram_channel_from_bank_id(uint32_t bank_id) const {
    this->check_allocator_is_initialized();
    return allocator::dram_channel_from_bank_id(*this->allocator_, bank_id);
}

CoreCoord Device::core_from_dram_channel(uint32_t dram_channel) const {
    TT_ASSERT(
        dram_channel < this->num_dram_channels(),
        "Bounds-Error -- dram_channel={} is outside of num_dram_channels={}",
        dram_channel,
        this->num_dram_channels()
    );
    return tt::Cluster::instance().get_soc_desc(id_).get_preferred_worker_core_for_dram_channel(dram_channel);
}

int32_t Device::bank_offset(BufferType buffer_type, uint32_t bank_id) const {
    this->check_allocator_is_initialized();
    return allocator::bank_offset(*this->allocator_, buffer_type, bank_id);
}

CoreCoord Device::logical_core_from_bank_id(uint32_t bank_id) const {
    this->check_allocator_is_initialized();
    return allocator::logical_core_from_bank_id(*this->allocator_, bank_id);
}

const std::vector<uint32_t> &Device::bank_ids_from_dram_channel(uint32_t dram_channel) const {
    this->check_allocator_is_initialized();
    return allocator::bank_ids_from_dram_channel(*this->allocator_, dram_channel);
}

const std::vector<uint32_t> &Device::bank_ids_from_logical_core(
    BufferType buffer_type, const CoreCoord &logical_core) const {
    this->check_allocator_is_initialized();
    return allocator::bank_ids_from_logical_core(*this->allocator_, buffer_type, logical_core);
}

allocator::Statistics Device::get_memory_allocation_statistics(const BufferType &buffer_type) const {
    this->check_allocator_is_initialized();
    return allocator::get_statistics(*this->allocator_, buffer_type);
}

size_t Device::get_l1_small_size() const {
    this->check_allocator_is_initialized();
    return this->allocator_->config.l1_small_size;
}

void Device::dump_memory_blocks(const BufferType &buffer_type, std::ofstream &out) const {
    this->check_allocator_is_initialized();
    return allocator::dump_memory_blocks(*this->allocator_, buffer_type, out);
}

void Device::deallocate_buffers(){
    allocator::deallocate_buffers(*allocator_);
}

float Device::sfpu_eps() const {

  float value = std::numeric_limits<float>::epsilon();
  if( arch() == tt::ARCH::GRAYSKULL  ) {
    value = tt::tt_metal::EPS_GS;
  } else if( arch() == tt::ARCH::WORMHOLE_B0 ) {
    value = tt::tt_metal::EPS_WHB0;
  }

  return value;
}

pair<int, int> Device::build_processor_type_to_index(JitBuildProcessorType t) const {
    constexpr int DataMovementBuildCount = 2;
    constexpr int ComputeBuildCount = 3;
    constexpr int EthernetBuildCount = 2;

    switch (t) {
    case JitBuildProcessorType::DATA_MOVEMENT: return pair<int, int>(0, DataMovementBuildCount);
    case JitBuildProcessorType::COMPUTE: return pair<int, int>(DataMovementBuildCount, ComputeBuildCount);
    case JitBuildProcessorType::ETHERNET: return pair<int, int>(DataMovementBuildCount + ComputeBuildCount, EthernetBuildCount);
    default: TT_ASSERT("Bad processor type: {}", static_cast<std::underlying_type<JitBuildProcessorType>::type>(t));
    }

    // shh the warnings
    return pair<int, int>(0, 0);
}

// Ideally the firmware getter would be private to the device, however, tests look for this
const JitBuildState& Device::build_firmware_state(JitBuildProcessorType t, int i) const {
    return *(this->firmware_build_states_[build_processor_type_to_index(t).first + i]);
}

const JitBuildState& Device::build_kernel_state(JitBuildProcessorType t, int i) const {
    return *(this->kernel_build_states_[build_processor_type_to_index(t).first + i]);
}

const JitBuildStateSubset Device::build_kernel_states(JitBuildProcessorType t) const {
    pair<int, int> bptti = build_processor_type_to_index(t);
    JitBuildStateSubset subset = {
        &this->kernel_build_states_[bptti.first],
        bptti.second
    };
    return subset;
}

const string Device::build_firmware_target_path(JitBuildProcessorType t, int i) const {
    const JitBuildState& bs = build_firmware_state(t, i);
    return bs.get_target_out_path("");
}

const string Device::build_kernel_target_path(JitBuildProcessorType t, int i, const string& kernel_name) const {
    const JitBuildState& bs = build_kernel_state(t, i);
    return bs.get_target_out_path(kernel_name);
}

HWCommandQueue& Device::hw_command_queue(size_t cq_id) {
    detail::DispatchStateCheck(true);
    TT_ASSERT( cq_id < hw_command_queues_.size(), "cq_id {} is out of range", cq_id );
    TT_FATAL(this->is_initialized(), "Device has not been initialized, did you forget to call InitializeDevice?");
    return *hw_command_queues_[cq_id];
}

CommandQueue &Device::command_queue(size_t cq_id) {
    detail::DispatchStateCheck(using_fast_dispatch);
    TT_ASSERT( cq_id < sw_command_queues_.size(), "cq_id {} is out of range", cq_id );
    TT_FATAL(this->is_initialized(), "Device has not been initialized, did you forget to call InitializeDevice?");
    return *sw_command_queues_[cq_id];
}

void Device::push_work(std::function<void()>&& work, bool blocking) {
    this->work_executor.push_work(work, blocking);
}

void Device::synchronize() {
    this->work_executor.synchronize();
}

void Device::set_worker_mode(const WorkExecutorMode& mode) {
    this->work_executor.set_worker_mode(mode);
}

void Device::enable_async(bool enable) {
    auto mode = enable ? WorkExecutorMode::ASYNCHRONOUS : WorkExecutorMode::SYNCHRONOUS;
    this->set_worker_mode(mode);
}

bool Device::using_slow_dispatch() const {
    return not (this->using_fast_dispatch);
}

void Device::begin_trace() {
    this->trace_contexts_.clear();
    for (size_t cq_id = 0; cq_id < num_hw_cqs(); cq_id++) {
        trace_contexts_.push_back(std::make_shared<detail::TraceDescriptor>());
        hw_command_queues_[cq_id]->record_begin(trace_contexts_.at(cq_id));
    }
}

void Device::end_trace() {
    this->trace_insts_.clear();
    this->release_last_trace();
    for (size_t cq_id = 0; cq_id < num_hw_cqs(); cq_id++) {
        hw_command_queues_[cq_id]->record_end();
        uint32_t tid = Trace::instantiate(
            this->command_queue(cq_id), trace_contexts_.at(cq_id), std::move(this->sysmem_manager().get_bypass_data()));
        trace_insts_.push_back(tid);
    }
}

void Device::execute_last_trace(bool blocking) {
    for (size_t cq_id = 0; cq_id < num_hw_cqs(); cq_id++) {
        if (this->trace_insts_.at(cq_id).has_value()) {
            uint32_t tid = this->trace_insts_.at(cq_id).value();
            TT_FATAL(Trace::has_instance(tid), "Trace instance " + std::to_string(tid) + " must exist on device");
            this->command_queue(cq_id).run_command(CommandInterface{
                .type = EnqueueCommandType::ENQUEUE_TRACE,
                .blocking = blocking,
                .trace_id = tid
            });
        }
    }
}

void Device::release_last_trace() {
    for (size_t cq_id = 0; cq_id < num_hw_cqs(); cq_id++) {
        if (this->trace_insts_.size() > cq_id) {
            if (this->trace_insts_.at(cq_id).has_value()) {
                uint32_t tid = this->trace_insts_.at(cq_id).value();
                if (Trace::has_instance(tid)) {
                    Trace::remove_instance(tid);
                }
            }
        }
    }
}

}  // namespace tt_metal

}  // namespace tt
