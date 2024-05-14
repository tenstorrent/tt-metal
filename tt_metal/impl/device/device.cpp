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
#include "common/env_lib.hpp"
#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"
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

bool ActiveDevices::is_device_active(chip_id_t id) {
    if (this->active_devices_.size() < id + 1) {
        return false;
    } else {
        return this->active_devices_[id] == ActiveState::ACTIVE;
    }
}

Device::Device(
    chip_id_t device_id, const uint8_t num_hw_cqs, size_t l1_small_size, const std::vector<uint32_t> &l1_bank_remap, bool minimal) :
    id_(device_id), num_hw_cqs_(num_hw_cqs), work_executor(device_id) {
    ZoneScoped;
    TT_ASSERT(num_hw_cqs > 0 and num_hw_cqs < 3, "num_hw_cqs can be between 1 and 2");
    this->build_key_ = tt::Cluster::instance().get_harvesting_mask(device_id);
    this->initialize(l1_small_size, l1_bank_remap, minimal);
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

    this->build_env_.init(this->build_key(), this->arch());

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

    for (const auto &eth_core : this->get_inactive_ethernet_cores()) {
        CoreCoord physical_core = this->ethernet_core_from_logical_core(eth_core);
        std::vector<uint32_t> zero_vec_mailbox(128 / sizeof(uint32_t), 0);
        llrt::write_hex_vec_to_core(this->id(), physical_core, zero_vec_mailbox, MEM_IERISC_MAILBOX_BASE);
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
    unique_ptr<Program, detail::ProgramDeleter> mmio_command_queue_program_ptr(new Program);

    std::string prefetch_kernel_path = "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp";
    std::string dispatch_kernel_path = "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp";

    // TODO: These are semaphore IDs, remove these when CreateSemaphore returns ID rather than address
    constexpr uint32_t prefetch_sync_sem = 0;
    constexpr uint32_t prefetch_downstream_cb_sem = 1;
    constexpr uint32_t dispatch_sync_sem = 0;
    constexpr uint32_t dispatch_cb_sem = 1;

    constexpr uint32_t prefetch_d_sync_sem = 0;
    constexpr uint32_t prefetch_d_upstream_cb_sem = 1;
    constexpr uint32_t prefetch_d_downstream_cb_sem = 2;
    constexpr uint32_t prefetch_h_exec_buf_sem = 2;
    constexpr uint32_t mux_upstream_cb_sem = 1;
    constexpr uint32_t demux_downstream_cb_sem = 1;
    constexpr uint32_t dispatch_downstream_cb_sem = 2;

    if (this->is_mmio_capable()) {
        auto device_id = this->id();
        uint8_t num_hw_cqs = this->num_hw_cqs();
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
        uint32_t cq_size = this->sysmem_manager().get_cq_size();

        for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
            CoreType dispatch_core_type = dispatch_core_manager::get(num_hw_cqs).get_dispatch_core_type(device_id);
            //add apis for dispatch_h/d prefetch_h
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
                dispatch_constants::get(dispatch_core_type).prefetch_q_size(),
                CQ_PREFETCH_Q_RD_PTR,
                dispatch_constants::get(dispatch_core_type).cmddat_q_base(),
                dispatch_constants::get(dispatch_core_type).cmddat_q_size(),
                dispatch_constants::get(dispatch_core_type).scratch_db_base(),
                dispatch_constants::get(dispatch_core_type).scratch_db_size(),
                prefetch_sync_sem,
                dispatch_constants::get(dispatch_core_type).prefetch_d_buffer_pages(), // prefetch_d only
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
                (1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE) * dispatch_constants::get(dispatch_core_type).dispatch_buffer_pages(),
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
        }
        detail::CompileProgram(this, *command_queue_program_ptr);
        this->command_queue_programs.push_back(std::move(command_queue_program_ptr));
    } else {
        /////////////////Following section is for mmio device serving Remote Device
        uint8_t num_hw_cqs = 1;
        uint32_t cq_id = 0;
        chip_id_t device_id = this->id();
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        Device *mmio_device = tt::tt_metal::detail::GetDeviceHandle(mmio_device_id);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
        uint32_t cq_size = mmio_device->sysmem_manager().get_cq_size();


        CoreType dispatch_core_type = dispatch_core_manager::get(num_hw_cqs).get_dispatch_core_type(mmio_device_id);
        tt_cxy_pair prefetch_location = dispatch_core_manager::get(num_hw_cqs).prefetcher_core(device_id, channel, cq_id);
        tt_cxy_pair dispatch_location = dispatch_core_manager::get(num_hw_cqs).dispatcher_core(device_id, channel, cq_id);
        bool dispatch_on_eth = dispatch_core_type == CoreType::ETH;

        TT_ASSERT(prefetch_location.chip == mmio_device_id and dispatch_location.chip == mmio_device_id,
            "Prefetcher is on device {} and Dispatcher is on device {} but they are expected to be on device {}", prefetch_location.chip, dispatch_location.chip, mmio_device_id);

        CoreCoord prefetch_physical_core = get_physical_core_coordinate(prefetch_location, dispatch_core_type);
        CoreCoord dispatch_physical_core = get_physical_core_coordinate(dispatch_location, dispatch_core_type);

        log_debug(LogDevice, "Dispatching out of {} cores",  magic_enum::enum_name(dispatch_core_type));
        log_debug(LogDevice, "Prefetch H logical location: {} physical core: {}", prefetch_location.str(), prefetch_physical_core.str());
        log_debug(LogDevice, "Dispatch H logical location: {} physical core {}", dispatch_location.str(), dispatch_physical_core.str());

        uint32_t command_queue_start_addr = get_absolute_cq_offset(channel, cq_id, cq_size);
        uint32_t issue_queue_start_addr = command_queue_start_addr + CQ_START;
        uint32_t issue_queue_size = mmio_device->sysmem_manager_->get_issue_queue_size(cq_id);
        uint32_t completion_queue_start_addr = issue_queue_start_addr + issue_queue_size;
        uint32_t completion_queue_size = mmio_device->sysmem_manager_->get_completion_queue_size(cq_id);

        tt_cxy_pair mux_location = dispatch_core_manager::get(num_hw_cqs).mux_core(device_id, channel, cq_id);
        tt_cxy_pair demux_location = dispatch_core_manager::get(num_hw_cqs).demux_core(device_id, channel, cq_id);
        tt_cxy_pair tunneler_location = dispatch_core_manager::get(num_hw_cqs).tunneler_core(device_id, channel, cq_id);
        CoreCoord tunneler_logical_core = CoreCoord(tunneler_location.x, tunneler_location.y);
        TT_ASSERT(tunneler_location.chip == mmio_device_id,
            "Tunneler is on device {} but it is expected to be on device {}", tunneler_location.chip, mmio_device_id);
        CoreCoord r_tunneler_logical_core = std::get<1>(tt::Cluster::instance().get_connected_ethernet_core(std::make_tuple(tunneler_location.chip, tunneler_logical_core)));
        CoreCoord r_tunneler_physical_core = this->ethernet_core_from_logical_core(r_tunneler_logical_core);

        CoreCoord tunneler_physical_core = mmio_device->ethernet_core_from_logical_core(tunneler_location);
        CoreCoord mux_physical_core = get_physical_core_coordinate(mux_location, dispatch_core_type);
        CoreCoord demux_physical_core = get_physical_core_coordinate(demux_location, dispatch_core_type);

        uint32_t tunneler_queue_start_addr = 0x19000;
        uint32_t tunneler_queue_size_bytes = 0x10000;
        uint32_t tunneler_test_results_addr = 0x39000;
        uint32_t tunneler_test_results_size = 0x7000;
        constexpr uint32_t packetized_path_test_results_addr = BRISC_L1_RESULT_BASE;
        constexpr uint32_t packetized_path_test_results_size = 1024;

        // Packetized path buffer, can be at any available address.
        constexpr uint32_t relay_demux_queue_start_addr = L1_UNRESERVED_BASE;
        constexpr uint32_t relay_demux_queue_size_bytes = 0x10000;
        constexpr uint32_t src_endpoint_start_id = 0xaa;
        constexpr uint32_t dest_endpoint_start_id = 0xbb;

        tt::tt_metal::CreateSemaphore(*mmio_command_queue_program_ptr, prefetch_location, 0, dispatch_core_type); // prefetch_sync_sem
        tt::tt_metal::CreateSemaphore(*mmio_command_queue_program_ptr, prefetch_location, dispatch_constants::get(dispatch_core_type).prefetch_d_buffer_pages(), dispatch_core_type); // prefetch_downstream_cb_sem
        tt::tt_metal::CreateSemaphore(*mmio_command_queue_program_ptr, prefetch_location, 0, dispatch_core_type);

        tt::tt_metal::CreateSemaphore(*mmio_command_queue_program_ptr, mux_location, 0, dispatch_core_type); // unused mux semaphore
        tt::tt_metal::CreateSemaphore(*mmio_command_queue_program_ptr, mux_location, 0, dispatch_core_type); // mux_upstream_cb_sem

        tt_metal::CreateSemaphore(*mmio_command_queue_program_ptr, demux_location, 0, dispatch_core_type); // unused
        tt_metal::CreateSemaphore(*mmio_command_queue_program_ptr, demux_location, 0, dispatch_core_type); // unused
        // for the unpacketize stage, we use rptr/wptr for flow control, and poll semaphore
        // value only to update the rptr:
        tt_metal::CreateSemaphore(*mmio_command_queue_program_ptr, demux_location, 0, dispatch_core_type);

        constexpr uint32_t dispatch_h_cb_sem = 0;
        tt_metal::CreateSemaphore(*mmio_command_queue_program_ptr, dispatch_location, 0, dispatch_core_type);

        std::map<string, string> prefetch_defines = {
            {"DISPATCH_KERNEL", "1"},
            {"MY_NOC_X", std::to_string(prefetch_physical_core.x)},
            {"MY_NOC_Y", std::to_string(prefetch_physical_core.y)},
            {"UPSTREAM_NOC_X", std::to_string(0)},
            {"UPSTREAM_NOC_Y", std::to_string(0)},
            {"DOWNSTREAM_NOC_X", std::to_string(mux_physical_core.x)},
            {"DOWNSTREAM_NOC_Y", std::to_string(mux_physical_core.y)},
        };

        std::vector<uint32_t> prefetch_compile_args = {
            dispatch_constants::DISPATCH_BUFFER_BASE,
            dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE,
            dispatch_constants::get(dispatch_core_type).prefetch_d_buffer_pages(),
            prefetch_downstream_cb_sem,
            mux_upstream_cb_sem,
            issue_queue_start_addr,
            issue_queue_size,
            dispatch_constants::PREFETCH_Q_BASE,
            dispatch_constants::get(dispatch_core_type).prefetch_q_size(),
            CQ_PREFETCH_Q_RD_PTR,
            dispatch_constants::get(dispatch_core_type).cmddat_q_base(),
            dispatch_constants::get(dispatch_core_type).cmddat_q_size(),
            dispatch_constants::get(dispatch_core_type).scratch_db_base(), // unused for prefetch_h
            dispatch_constants::get(dispatch_core_type).scratch_db_size(), // unused for prefetch_h
            prefetch_sync_sem, // unused for prefetch_h
            dispatch_constants::get(dispatch_core_type).prefetch_d_buffer_pages(), // prefetch_d only
            prefetch_d_upstream_cb_sem, // prefetch_d only
            prefetch_downstream_cb_sem, // prefetch_d only
            dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE,
            dispatch_constants::PREFETCH_D_BUFFER_BLOCKS, // prefetch_d only
            prefetch_h_exec_buf_sem,
            false,   // is_dram_variant
            true    // is_host_variant
        };

        if (dispatch_on_eth) {
            tt::tt_metal::CreateKernel(
                *mmio_command_queue_program_ptr,
                "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp",
                prefetch_location,
                EthernetConfig{
                    .eth_mode = Eth::IDLE,
                    .noc = NOC::NOC_0,
                    .compile_args = prefetch_compile_args,
                    .defines = prefetch_defines});
        } else {
        tt::tt_metal::CreateKernel(
            *mmio_command_queue_program_ptr,
            "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp", // update this for remote device
            prefetch_location,
            tt::tt_metal::DataMovementConfig {
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = prefetch_compile_args,
                .defines = prefetch_defines});
        }
        log_debug(LogDevice, "run prefetch_h {}", prefetch_location.str());

        uint32_t relay_mux_queue_start_addr = dispatch_constants::DISPATCH_BUFFER_BASE;
        uint32_t relay_mux_queue_size_bytes = dispatch_constants::get(dispatch_core_type).prefetch_d_buffer_size();
        uint32_t timeout_mcycles = 0;
        std::vector<uint32_t> mux_compile_args =
        {
            0, // 0: reserved
            (relay_mux_queue_start_addr >> 4), // 1: rx_queue_start_addr_words
            (relay_mux_queue_size_bytes >> 4), // 2: rx_queue_size_words
            1, // 3: mux_fan_in
            packet_switch_4B_pack((uint32_t)prefetch_physical_core.x,
                                (uint32_t)prefetch_physical_core.y,
                                1,
                                (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: src 0 info
            packet_switch_4B_pack(0,
                                0,
                                1,
                                (uint32_t)DispatchRemoteNetworkType::NOC0), // 5: src 1 info
            packet_switch_4B_pack(0,
                                0,
                                1,
                                (uint32_t)DispatchRemoteNetworkType::NOC0), // 6: src 2 info
            packet_switch_4B_pack(0,
                                0,
                                1,
                                (uint32_t)DispatchRemoteNetworkType::NOC0), // 7: src 3 info
            (tunneler_queue_start_addr >> 4), // 8: remote_tx_queue_start_addr_words
            (tunneler_queue_size_bytes >> 4), // 9: remote_tx_queue_size_words
            (uint32_t)tunneler_physical_core.x, // 10: remote_tx_x
            (uint32_t)tunneler_physical_core.y, // 11: remote_tx_y
            0, // 12: remote_tx_queue_id
            (uint32_t)DispatchRemoteNetworkType::NOC0, // 13: tx_network_type
            packetized_path_test_results_addr, // 14: test_results_addr
            packetized_path_test_results_size, // 15: test_results_size
            timeout_mcycles * 1000 * 1000, // 16: timeout_cycles
            0x0,// 17: output_depacketize
            0x0,// 18: output_depacketize info
            // 19: input 0 packetize info:
            packet_switch_4B_pack(0x1,
                                dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE,
                                mux_upstream_cb_sem, // local sem
                                prefetch_downstream_cb_sem), // upstream sem
            packet_switch_4B_pack(0, 0, 0, 0), // 20: input 1 packetize info
            packet_switch_4B_pack(0, 0, 0, 0), // 21: input 2 packetize info
            packet_switch_4B_pack(0, 0, 0, 0), // 22: input 3 packetize info
            packet_switch_4B_pack(src_endpoint_start_id, 0, 0, 0), // 23: packetized input src id
            packet_switch_4B_pack(dest_endpoint_start_id, 0, 0, 0), // 24: packetized input dest id
        };

        log_debug(LogDevice, "run mux at {}", mux_location.str());
        if (dispatch_on_eth) {
            tt::tt_metal::CreateKernel(
                *mmio_command_queue_program_ptr,
                "tt_metal/impl/dispatch/kernels/packet_mux.cpp",
                mux_location,
                EthernetConfig{
                    .eth_mode = Eth::IDLE,
                    .noc = NOC::NOC_0,
                    .compile_args = mux_compile_args,
                    .defines = {}
                }
            );
        } else {
        tt_metal::CreateKernel(
            *mmio_command_queue_program_ptr,
            "tt_metal/impl/dispatch/kernels/packet_mux.cpp",
            mux_location,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = mux_compile_args,
                .defines = {}
            }
        );
        }

        std::vector<uint32_t> tunneler_l_compile_args =
        {
            dest_endpoint_start_id, // 0: endpoint_id_start_index
            2, // tunnel_lanes. 1 => Unidirectional. 2 => Bidirectional.
            (tunneler_queue_start_addr >> 4), // 2: rx_queue_start_addr_words
            (tunneler_queue_size_bytes >> 4), // 3: rx_queue_size_words
            packet_switch_4B_pack(r_tunneler_physical_core.x,
                                r_tunneler_physical_core.y,
                                0,
                                (uint32_t)DispatchRemoteNetworkType::ETH), // 4: remote_receiver_0_info
            packet_switch_4B_pack(demux_physical_core.x,
                                demux_physical_core.y,
                                1,//num_dest_endpoints,
                                (uint32_t)DispatchRemoteNetworkType::NOC0), // 5: remote_receiver_1_info
            tunneler_queue_start_addr >> 4, // 6: remote_receiver_queue_start_addr_words 0
            tunneler_queue_size_bytes >> 4, // 7: remote_receiver_queue_size_words 0
            (relay_demux_queue_start_addr >> 4), // 8: remote_receiver_queue_start_addr_words 1
            (relay_demux_queue_size_bytes >> 4), // 9: remote_receiver_queue_size_words 1
            packet_switch_4B_pack(mux_physical_core.x,
                                mux_physical_core.y,
                                1,//num_dest_endpoints,
                                (uint32_t)DispatchRemoteNetworkType::NOC0), // 10: remote_sender_0_info
            packet_switch_4B_pack(r_tunneler_physical_core.x,
                                r_tunneler_physical_core.y,
                                3,
                                (uint32_t)DispatchRemoteNetworkType::ETH), // 11: remote_sender_1_info
            tunneler_test_results_addr, // 12: test_results_addr
            tunneler_test_results_size, // 13: test_results_size
            timeout_mcycles * 1000 * 1000 * 4, // 14: timeout_cycles
        };

        tt_metal::CreateKernel(
            *mmio_command_queue_program_ptr,
            "tt_metal/impl/dispatch/kernels/eth_tunneler.cpp",
            tunneler_logical_core,
            tt_metal::EthernetConfig{
                .noc = tt_metal::NOC::NOC_0,
                .compile_args = tunneler_l_compile_args
            }
        );
        log_debug(LogDevice, "run tunneler at {}", tunneler_location.str());

        uint32_t dest_map_array[4] = {0, 1, 2, 3};
        uint64_t dest_endpoint_output_map = packet_switch_dest_pack(dest_map_array, 4);
        std::vector<uint32_t> demux_compile_args =
        {
            dest_endpoint_start_id, // 0: endpoint_id_start_index
            (relay_demux_queue_start_addr >> 4), // 1: rx_queue_start_addr_words
            (relay_demux_queue_size_bytes >> 4), // 2: rx_queue_size_words
            1, // 3: demux_fan_out
            packet_switch_4B_pack(dispatch_physical_core.x,
                                    dispatch_physical_core.y,
                                    0,
                                    (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: remote_tx_0_info
            packet_switch_4B_pack(0,
                                    0,
                                    0,
                                    (uint32_t)DispatchRemoteNetworkType::NOC0), // 5: remote_tx_1_info
            packet_switch_4B_pack(0,
                                    0,
                                    0,
                                    (uint32_t)DispatchRemoteNetworkType::NOC0), // 6: remote_tx_2_info
            packet_switch_4B_pack(0,
                                    0,
                                    0,
                                    (uint32_t)DispatchRemoteNetworkType::NOC0), // 7: remote_tx_3_info
            (dispatch_constants::DISPATCH_BUFFER_BASE >> 4), // 8: remote_tx_queue_start_addr_words 0
            ((1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE)*dispatch_constants::get(dispatch_core_type).dispatch_buffer_pages()) >> 4, // 9: remote_tx_queue_size_words 0
            0, // 10: remote_tx_queue_start_addr_words 1
            0, // 11: remote_tx_queue_size_words 1
            0, // 12: remote_tx_queue_start_addr_words 2
            0, // 13: remote_tx_queue_size_words 2
            0, // 14: remote_tx_queue_start_addr_words 3
            0, // 15: remote_tx_queue_size_words 3
            //(uint32_t)phys_dispatch_relay_mux_core.x, // 16: remote_rx_x
            //(uint32_t)phys_dispatch_relay_mux_core.y, // 17: remote_rx_y
            //num_dest_endpoints, // 18: remote_rx_queue_id
            (uint32_t)tunneler_physical_core.x, // 16: remote_rx_x
            (uint32_t)tunneler_physical_core.y, // 17: remote_rx_y
            3, // 18: remote_rx_queue_id
            (uint32_t)DispatchRemoteNetworkType::NOC0, // 19: tx_network_type
            (uint32_t)(dest_endpoint_output_map >> 32), // 20: dest_endpoint_output_map_hi
            (uint32_t)(dest_endpoint_output_map & 0xFFFFFFFF), // 21: dest_endpoint_output_map_lo
            packetized_path_test_results_addr, // 22: test_results_addr
            packetized_path_test_results_size, // 23: test_results_size
            timeout_mcycles * 1000 * 1000, // 24: timeout_cycles
            0x1, // 25: output_depacketize_mask
            // 26: output 0 packetize info:
            packet_switch_4B_pack(dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE,
                                    dispatch_h_cb_sem, // downstream sem
                                    dispatch_downstream_cb_sem, // local sem
                                    1), // remove header
            packet_switch_4B_pack(0, 0, 0, 0), // 27: output 1 packetize info
            packet_switch_4B_pack(0, 0, 0, 0), // 28: output 2 packetize info
            packet_switch_4B_pack(0, 0, 0, 0), // 29: output 3 packetize info
        };

        log_debug(LogDevice, "run dispatch demux at {}", demux_location.str());

        if (dispatch_on_eth) {
            tt::tt_metal::CreateKernel(
                *mmio_command_queue_program_ptr,
                "tt_metal/impl/dispatch/kernels/packet_demux.cpp",
                demux_location,
                EthernetConfig{
                    .eth_mode = Eth::IDLE,
                    .noc = NOC::NOC_0,
                    .compile_args = demux_compile_args,
                    .defines = {}
                }
            );
        } else {
        tt_metal::CreateKernel(
            *mmio_command_queue_program_ptr,
            "tt_metal/impl/dispatch/kernels/packet_demux.cpp",
            {demux_location},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = demux_compile_args,
                .defines = {}
            }
        );
        }

        std::vector<uint32_t> dispatch_compile_args = {
            dispatch_constants::DISPATCH_BUFFER_BASE,
            dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE,
            dispatch_constants::get(dispatch_core_type).dispatch_buffer_pages(),
            dispatch_h_cb_sem, // overridden below for h
            prefetch_d_downstream_cb_sem,
            dispatch_constants::DISPATCH_BUFFER_SIZE_BLOCKS,
            prefetch_sync_sem,
            command_queue_start_addr,
            completion_queue_start_addr,
            completion_queue_size,
            dispatch_constants::DISPATCH_BUFFER_BASE,
            (1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE) * dispatch_constants::get(dispatch_core_type).dispatch_buffer_pages(),
            dispatch_h_cb_sem, // unused on hd, filled in below for h and d
            dispatch_downstream_cb_sem, // unused on hd, filled in below for h and d
            0, // preamble size. unused unless tunneler is between h and d
            false,   // is_dram_variant
            true     // is_host_variant
        };

        std::map<string, string> dispatch_defines = {
            {"DISPATCH_KERNEL", "1"},
            {"MY_NOC_X", std::to_string(dispatch_physical_core.x)},
            {"MY_NOC_Y", std::to_string(dispatch_physical_core.y)},
            {"UPSTREAM_NOC_X", std::to_string(demux_physical_core.x)},
            {"UPSTREAM_NOC_Y", std::to_string(demux_physical_core.y)},
            {"DOWNSTREAM_NOC_X", std::to_string(0xffffffff)},
            {"DOWNSTREAM_NOC_Y", std::to_string(0xffffffff)},
        };

        log_debug(LogDevice, "run dispatch_h at {}", dispatch_location.str());

        if (dispatch_on_eth) {
            tt::tt_metal::CreateKernel(
                *mmio_command_queue_program_ptr,
                "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",
                dispatch_location,
                EthernetConfig{
                    .eth_mode = Eth::IDLE,
                    .noc = NOC::NOC_0,
                    .compile_args = dispatch_compile_args,
                    .defines = dispatch_defines
                }
            );
        } else {
        tt::tt_metal::CreateKernel(
            *mmio_command_queue_program_ptr,
            "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",
            dispatch_location,
            tt::tt_metal::DataMovementConfig {
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = dispatch_compile_args,
                .defines = dispatch_defines});
        }

        /////////////////Following section is for Remote Device
        //auto device_id = this->id();
        //uint8_t num_hw_cqs = 1;
        //uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
        dispatch_core_type = dispatch_core_manager::get(num_hw_cqs).get_dispatch_core_type(device_id);
        dispatch_on_eth = dispatch_core_type == CoreType::ETH;

        uint32_t dispatch_buffer_pages = dispatch_constants::get(dispatch_core_type).dispatch_buffer_pages();
        uint32_t mux_queue_start_addr = dispatch_constants::DISPATCH_BUFFER_BASE;
        uint32_t mux_queue_size_bytes = (1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE)*dispatch_buffer_pages;
        // Packetized path buffer, can be at any available address.
        constexpr uint32_t demux_queue_start_addr = L1_UNRESERVED_BASE;
        constexpr uint32_t demux_queue_size_bytes = 0x10000;

        //uint32_t tunneler_queue_start_addr = 0x19000;
        //uint32_t tunneler_queue_size_bytes = 0x10000;
        //uint32_t tunneler_test_results_addr = 0x39000;
        //uint32_t tunneler_test_results_size = 0x7000;
        //constexpr uint32_t packetized_path_test_results_addr = BRISC_L1_RESULT_BASE;
        //constexpr uint32_t packetized_path_test_results_size = 1024;

        // For tests with checkers enabled, packetized path may time out and
        // cause the test to fail.
        // To save inner loop cycles, presently the packetized components have
        // a 32-bit timeout cycle counter so 4K cycles is the maximum timeout.
        // Setting this to 0 disables the timeout.
        //uint32_t timeout_mcycles = 0;

        // These could start from 0, but we assign values that are easy to
        // identify for debug.
        //constexpr uint32_t src_endpoint_start_id = 0xaa;
        //constexpr uint32_t dest_endpoint_start_id = 0xbb;

        //uint32_t cq_id = num_hw_cqs - 1;
        //tt_cxy_pair tunneler_location = dispatch_core_manager::get(num_hw_cqs).tunneler_core(device_id, channel, cq_id);
        //CoreCoord tunneler_logical_core = CoreCoord(tunneler_location.x, tunneler_location.y);
        //CoreCoord tunneler_physical_core = tt::Cluster::instance().ethernet_core_from_logical_core(tunneler_location.chip, tunneler_logical_core);

        //std::tuple<chip_id_t, CoreCoord> connected_eth_core = tt::Cluster::instance().get_connected_ethernet_core(std::make_tuple(tunneler_location.chip, tunneler_logical_core));

        //CoreCoord r_tunneler_logical_core = std::get<1>(connected_eth_core);
        //CoreCoord r_tunneler_physical_core = this->ethernet_core_from_logical_core(r_tunneler_logical_core);

        tt_cxy_pair mux_d_location = dispatch_core_manager::get(num_hw_cqs).mux_d_core(device_id, channel, cq_id);
        CoreCoord mux_d_physical_core = get_physical_core_coordinate(mux_d_location, dispatch_core_type);
        tt_cxy_pair demux_d_location = dispatch_core_manager::get(num_hw_cqs).demux_d_core(device_id, channel, cq_id);
        CoreCoord demux_d_physical_core = get_physical_core_coordinate(demux_d_location, dispatch_core_type);

        tt_cxy_pair prefetch_d_location = dispatch_core_manager::get(num_hw_cqs).prefetcher_d_core(device_id, channel, cq_id);
        CoreCoord prefetch_d_physical_core = get_physical_core_coordinate(prefetch_d_location, dispatch_core_type);

        //tt_cxy_pair dispatch_location = dispatch_core_manager::get(num_hw_cqs).dispatcher_d_core(device_id, channel, cq_id);
        //CoreCoord dispatch_physical_core = get_physical_core_coordinate(dispatch_location, dispatch_core_type);
        dispatch_location = dispatch_core_manager::get(num_hw_cqs).dispatcher_d_core(device_id, channel, cq_id);
        dispatch_physical_core = get_physical_core_coordinate(dispatch_location, dispatch_core_type);

        tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, prefetch_d_location, 0, dispatch_core_type); // prefetch_d_sync_sem
        tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, prefetch_d_location, 0, dispatch_core_type); // prefetch_d_upstream_cb_sem
        tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, prefetch_d_location, dispatch_buffer_pages, dispatch_core_type); // prefetch_d_downstream_cb_sem

        tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, {demux_d_location}, 0, dispatch_core_type); // unused demux semaphore
        tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, {demux_d_location}, 0, dispatch_core_type); // demux_downstream_cb_sem

        tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, dispatch_location, 0, dispatch_core_type); // dispatch_sync_sem
        tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, dispatch_location, 0, dispatch_core_type); // dispatch_cb_sem
        tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, dispatch_location, dispatch_buffer_pages, dispatch_core_type); // dispatch_downstream_cb_sem

        //constexpr uint32_t dispatch_h_cb_sem = 0;
        tt_metal::CreateSemaphore(*command_queue_program_ptr, mux_d_location, 0, dispatch_core_type);

        uint32_t prefetch_d_buffer_base = dispatch_constants::DISPATCH_BUFFER_BASE;

        std::vector<uint32_t> tunneler_r_compile_args =
        {
            dest_endpoint_start_id, // 0: endpoint_id_start_index
            2,  // tunnel_lanes. 1 => Unidirectional. 2 => Bidirectional.
            (tunneler_queue_start_addr >> 4), // 2: rx_queue_start_addr_words
            (tunneler_queue_size_bytes >> 4), // 3: rx_queue_size_words
            packet_switch_4B_pack(demux_d_physical_core.x,
                                    demux_d_physical_core.y,
                                    1, //num_dest_endpoints,
                                    (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: remote_receiver_0_info
            packet_switch_4B_pack(tunneler_physical_core.x,
                                    tunneler_physical_core.y,
                                    1,
                                    (uint32_t)DispatchRemoteNetworkType::ETH), // 5: remote_receiver_1_info
            (demux_queue_start_addr >> 4), // 6: remote_receiver_queue_start_addr_words 0
            (demux_queue_size_bytes >> 4), // 7: remote_receiver_queue_size_words 0
            (tunneler_queue_start_addr + tunneler_queue_size_bytes) >> 4, // 8: remote_receiver_queue_start_addr_words 1
            tunneler_queue_size_bytes >> 4, // 9: remote_receiver_queue_size_words 1
            packet_switch_4B_pack(tunneler_physical_core.x,
                                tunneler_physical_core.y,
                                2,
                                (uint32_t)DispatchRemoteNetworkType::ETH), // 10: remote_sender_0_info
            packet_switch_4B_pack(mux_d_physical_core.x,
                                mux_d_physical_core.y,
                                1, //num_dest_endpoints,
                                (uint32_t)DispatchRemoteNetworkType::NOC0), // 11: remote_sender_1_info
            tunneler_test_results_addr, // 12: test_results_addr
            tunneler_test_results_size, // 13: test_results_size
            timeout_mcycles * 1000 * 1000 * 4, // 14: timeout_cycles
        };

        tt_metal::CreateKernel(
            *command_queue_program_ptr,
            "tt_metal/impl/dispatch/kernels/eth_tunneler.cpp",
            r_tunneler_logical_core,
            tt_metal::EthernetConfig{
                .noc = tt_metal::NOC::NOC_0,
                .compile_args = tunneler_r_compile_args
            }
        );
        log_debug(LogDevice, "run tunneler at device {} Core {}", this->id(), r_tunneler_logical_core.str());

        //uint32_t dest_map_array[4] = {0, 1, 2, 3};
        //uint64_t dest_endpoint_output_map = packet_switch_dest_pack(dest_map_array, 4);
        std::vector<uint32_t> demux_d_compile_args =
        {
            dest_endpoint_start_id, // 0: endpoint_id_start_index
            (demux_queue_start_addr >> 4), // 1: rx_queue_start_addr_words
            (demux_queue_size_bytes >> 4), // 2: rx_queue_size_words
            1, // 3: demux_fan_out
            packet_switch_4B_pack(prefetch_d_physical_core.x,
                                prefetch_d_physical_core.y,
                                0,
                                (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: remote_tx_0_info
            packet_switch_4B_pack(0,
                                0,
                                0,
                                (uint32_t)DispatchRemoteNetworkType::NOC0), // 5: remote_tx_1_info
            packet_switch_4B_pack(0,
                                0,
                                0,
                                (uint32_t)DispatchRemoteNetworkType::NOC0), // 6: remote_tx_2_info
            packet_switch_4B_pack(0,
                                0,
                                0,
                                (uint32_t)DispatchRemoteNetworkType::NOC0), // 7: remote_tx_3_info
            (prefetch_d_buffer_base >> 4), // 8: remote_tx_queue_start_addr_words 0
            dispatch_constants::get(dispatch_core_type).prefetch_d_buffer_size() >> 4, // 9: remote_tx_queue_size_words 0
            0, // 10: remote_tx_queue_start_addr_words 1
            0, // 11: remote_tx_queue_size_words 1
            0, // 12: remote_tx_queue_start_addr_words 2
            0, // 13: remote_tx_queue_size_words 2
            0, // 14: remote_tx_queue_start_addr_words 3
            0, // 15: remote_tx_queue_size_words 3
            (uint32_t)r_tunneler_physical_core.x, // 16: remote_rx_x
            (uint32_t)r_tunneler_physical_core.y, // 17: remote_rx_y
            2, // 18: remote_rx_queue_id
            (uint32_t)DispatchRemoteNetworkType::NOC0, // 19: tx_network_type
            (uint32_t)(dest_endpoint_output_map >> 32), // 20: dest_endpoint_output_map_hi
            (uint32_t)(dest_endpoint_output_map & 0xFFFFFFFF), // 21: dest_endpoint_output_map_lo
            packetized_path_test_results_addr, // 22: test_results_addr
            packetized_path_test_results_size, // 23: test_results_size
            timeout_mcycles * 1000 * 1000, // 24: timeout_cycles
            0x1, // 25: output_depacketize_mask
            // 26: output 0 packetize info:
            packet_switch_4B_pack(dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE,
                                demux_downstream_cb_sem, // local sem
                                prefetch_d_upstream_cb_sem, // downstream sem
                                0),
            packet_switch_4B_pack(0, 0, 0, 0), // 27: output 1 packetize info
            packet_switch_4B_pack(0, 0, 0, 0), // 28: output 2 packetize info
            packet_switch_4B_pack(0, 0, 0, 0), // 29: output 3 packetize info
        };

        log_debug(LogDevice, "run demux at {}", demux_d_location.str());

        if (dispatch_on_eth) {
            tt::tt_metal::CreateKernel(
                *command_queue_program_ptr,
                "tt_metal/impl/dispatch/kernels/packet_demux.cpp",
                demux_d_location,
                EthernetConfig{
                    .eth_mode = Eth::IDLE,
                    .noc = NOC::NOC_0,
                    .compile_args = demux_d_compile_args,
                    .defines = {}
                }
            );
        } else {
        tt_metal::CreateKernel(
            *command_queue_program_ptr,
            "tt_metal/impl/dispatch/kernels/packet_demux.cpp",
            demux_d_location,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = demux_d_compile_args,
                .defines = {}
            }
        );
        }

        // prefetch_d
        uint32_t scratch_db_base = (prefetch_d_buffer_base + dispatch_constants::get(dispatch_core_type).prefetch_d_buffer_size()
                                    + PCIE_ALIGNMENT - 1) & (~(PCIE_ALIGNMENT - 1));
        uint32_t scratch_db_size = dispatch_constants::get(dispatch_core_type).scratch_db_size();
        const uint32_t l1_size = dispatch_core_type == CoreType::WORKER ? MEM_L1_SIZE : MEM_ETH_SIZE;

        TT_ASSERT(scratch_db_base + scratch_db_size <= l1_size);

        std::map<string, string> prefetch_d_defines = {
            {"DISPATCH_KERNEL", "1"},
            {"MY_NOC_X", std::to_string(prefetch_d_physical_core.x)},
            {"MY_NOC_Y", std::to_string(prefetch_d_physical_core.y)},
            {"UPSTREAM_NOC_X", std::to_string(demux_d_physical_core.x)},
            {"UPSTREAM_NOC_Y", std::to_string(demux_d_physical_core.y)},
            {"DOWNSTREAM_NOC_X", std::to_string(dispatch_physical_core.x)},
            {"DOWNSTREAM_NOC_Y", std::to_string(dispatch_physical_core.y)},
        };

        std::vector<uint32_t> prefetch_d_compile_args = {
            dispatch_constants::DISPATCH_BUFFER_BASE, // overridden below for prefetch_h
            dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE, // overridden below for prefetch_h
            dispatch_buffer_pages, // overridden below for prefetch_h
            prefetch_d_downstream_cb_sem, // overridden below for prefetch_d
            dispatch_cb_sem, // overridden below for prefetch_h
            0, //issue_queue_start_addr,
            0, //issue_queue_size,
            0, //prefetch_q_base,
            dispatch_constants::get(dispatch_core_type).prefetch_q_size(),
            CQ_PREFETCH_Q_RD_PTR,
            prefetch_d_buffer_base, // overridden for split below
            dispatch_constants::get(dispatch_core_type).prefetch_d_buffer_size(), // overridden for split below
            scratch_db_base, // scratch_db_base filled in below if used
            scratch_db_size,
            prefetch_sync_sem,
            dispatch_constants::get(dispatch_core_type).prefetch_d_buffer_pages(), // prefetch_d only
            prefetch_d_upstream_cb_sem, // prefetch_d only my upstream
            demux_downstream_cb_sem, // prefetch_d only upstream
            dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE,
            dispatch_constants::PREFETCH_D_BUFFER_BLOCKS, // prefetch_d only
            prefetch_h_exec_buf_sem,
            true,
            false
        };

        if (dispatch_on_eth) {
            tt::tt_metal::CreateKernel(
                *command_queue_program_ptr,
                "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp",
                prefetch_d_location,
                EthernetConfig{
                    .eth_mode = Eth::IDLE,
                    .noc = NOC::NOC_0,
                    .compile_args = prefetch_d_compile_args,
                    .defines = prefetch_d_defines
                }
            );
        } else {
        tt::tt_metal::CreateKernel(
            *command_queue_program_ptr,
            "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp", // update this for remote device
            prefetch_d_location,
            tt::tt_metal::DataMovementConfig {
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = prefetch_d_compile_args,
                .defines = prefetch_d_defines});
        }

        log_debug(LogDevice, "run prefertch_d at {}", prefetch_d_location.str());


        std::map<string, string> dispatch_d_defines = {
            {"DISPATCH_KERNEL", "1"},
            {"MY_NOC_X", std::to_string(dispatch_physical_core.x)},
            {"MY_NOC_Y", std::to_string(dispatch_physical_core.y)},
            {"UPSTREAM_NOC_X", std::to_string(prefetch_d_physical_core.x)},
            {"UPSTREAM_NOC_Y", std::to_string(prefetch_d_physical_core.y)},
            {"DOWNSTREAM_NOC_X", std::to_string(mux_d_physical_core.x)},
            {"DOWNSTREAM_NOC_Y", std::to_string(mux_d_physical_core.y)},
        };
        std::vector<uint32_t> dispatch_d_compile_args = {
            dispatch_constants::DISPATCH_BUFFER_BASE,
            dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE,
            dispatch_buffer_pages,
            dispatch_cb_sem,
            prefetch_d_downstream_cb_sem,
            dispatch_constants::DISPATCH_BUFFER_SIZE_BLOCKS,
            dispatch_sync_sem,
            128,
            128 + 256 * 1024 * 1024,
            256 * 1024 * 1024,
            dispatch_constants::DISPATCH_BUFFER_BASE,
            (1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE) * dispatch_buffer_pages,
            dispatch_downstream_cb_sem, // unused on hd, filled in below for h and d
            dispatch_h_cb_sem, // unused on hd, filled in below for h and d
            sizeof(dispatch_packet_header_t), // unused unless tunneler is between h and d
            true,   // is_dram_variant
            false    // is_host_variant
        };

        if (dispatch_on_eth) {
            tt::tt_metal::CreateKernel(
                *command_queue_program_ptr,
                "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",
                dispatch_location,
                EthernetConfig{
                    .eth_mode = Eth::IDLE,
                    .noc = NOC::NOC_0,
                    .compile_args = dispatch_d_compile_args,
                    .defines = dispatch_d_defines
                }
            );
        } else {
        tt::tt_metal::CreateKernel(
            *command_queue_program_ptr,
            "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",
            dispatch_location,
            tt::tt_metal::DataMovementConfig {
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = dispatch_d_compile_args,
                .defines = dispatch_d_defines});
        }

        log_debug(LogDevice, "run dispatch at {}", dispatch_location.str());

        std::vector<uint32_t> mux_d_compile_args =
        {
            0, // 0: reserved
            (mux_queue_start_addr >> 4), // 1: rx_queue_start_addr_words
            (mux_queue_size_bytes >> 4), // 2: rx_queue_size_words
            1, // 3: mux_fan_in
            packet_switch_4B_pack((uint32_t)dispatch_physical_core.x,
                                    (uint32_t)dispatch_physical_core.y,
                                    1,
                                    (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: src 0 info
            packet_switch_4B_pack(0,
                                    0,
                                    1,
                                    (uint32_t)DispatchRemoteNetworkType::NOC0), // 5: src 1 info
            packet_switch_4B_pack(0,
                                    0,
                                    1,
                                    (uint32_t)DispatchRemoteNetworkType::NOC0), // 6: src 2 info
            packet_switch_4B_pack(0,
                                    0,
                                    1,
                                    (uint32_t)DispatchRemoteNetworkType::NOC0), // 7: src 3 info
            ((tunneler_queue_start_addr + tunneler_queue_size_bytes) >> 4), // 8: remote_tx_queue_start_addr_words
            (tunneler_queue_size_bytes >> 4), // 9: remote_tx_queue_size_words
            (uint32_t)r_tunneler_physical_core.x, // 10: remote_tx_x
            (uint32_t)r_tunneler_physical_core.y, // 11: remote_tx_y
            1, // 12: remote_tx_queue_id
            (uint32_t)DispatchRemoteNetworkType::NOC0, // 13: tx_network_type
            packetized_path_test_results_addr, // 14: test_results_addr
            packetized_path_test_results_size, // 15: test_results_size
            timeout_mcycles * 1000 * 1000, // 16: timeout_cycles
            0x0,// 17: output_depacketize
            0x0,// 18: output_depacketize info
            // 19: input 0 packetize info:
            packet_switch_4B_pack(0x1,
                                    dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE,
                                    dispatch_downstream_cb_sem, // upstream sem
                                    dispatch_h_cb_sem), // local sem
            packet_switch_4B_pack(0, 0, 0, 0), // 20: input 1 packetize info
            packet_switch_4B_pack(0, 0, 0, 0), // 21: input 2 packetize info
            packet_switch_4B_pack(0, 0, 0, 0), // 22: input 3 packetize info
            packet_switch_4B_pack(src_endpoint_start_id, 0, 0, 0), // 23: packetized input src id
            packet_switch_4B_pack(dest_endpoint_start_id, 0, 0, 0), // 24: packetized input dest id
        };

        log_debug(LogDevice, "run mux at {}", mux_d_location.str());

        if (dispatch_on_eth) {
            tt::tt_metal::CreateKernel(
                *command_queue_program_ptr,
                "tt_metal/impl/dispatch/kernels/packet_mux.cpp",
                mux_d_location,
                EthernetConfig{
                    .eth_mode = Eth::IDLE,
                    .noc = NOC::NOC_0,
                    .compile_args = mux_d_compile_args,
                    .defines = {}
                }
            );
        } else {
        tt_metal::CreateKernel(
            *command_queue_program_ptr,
            "tt_metal/impl/dispatch/kernels/packet_mux.cpp",
            mux_d_location,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = mux_d_compile_args,
                .defines = {}
            }
        );
        }

        detail::CompileProgram(this, *command_queue_program_ptr);
        this->command_queue_programs.push_back(std::move(command_queue_program_ptr));
        detail::CompileProgram(mmio_device, *mmio_command_queue_program_ptr);
        this->command_queue_programs.push_back(std::move(mmio_command_queue_program_ptr));
    }
}

// Writes issue and completion queue pointers to device and in sysmem and loads fast dispatch program onto dispatch cores
void Device::configure_command_queue_programs() {
    chip_id_t device_id = this->id();
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
    Device *mmio_device = tt::tt_metal::detail::GetDeviceHandle(mmio_device_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);

    std::vector<uint32_t> zero = {0x0}; // Reset state in case L1 Clear is disabled.
    std::vector<uint32_t> pointers(CQ_START / sizeof(uint32_t), 0);
    uint32_t cq_size = this->sysmem_manager().get_cq_size();

    if (this->is_mmio_capable()) {
        TT_ASSERT(this->command_queue_programs.size() == 1);
    } else {
        TT_ASSERT(this->command_queue_programs.size() == 2);
    }

    Program& command_queue_program = *this->command_queue_programs[0];

    for (uint8_t cq_id = 0; cq_id < this->num_hw_cqs(); cq_id++) {
        // Reset the host manager's pointer for this command queue
        this->sysmem_manager_->reset(cq_id);

        pointers[HOST_CQ_ISSUE_READ_PTR / sizeof(uint32_t)] = (CQ_START + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4;
        pointers[HOST_CQ_COMPLETION_WRITE_PTR / sizeof(uint32_t)] = (CQ_START + this->sysmem_manager_->get_issue_queue_size(cq_id) + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4;

        tt::Cluster::instance().write_sysmem(pointers.data(), pointers.size() * sizeof(uint32_t), cq_id * cq_size, mmio_device_id, channel);
    }

    uint8_t num_hw_cqs = device_id == mmio_device_id ? this->num_hw_cqs() : 1;
    for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
        tt_cxy_pair prefetch_location = dispatch_core_manager::get(num_hw_cqs).prefetcher_core(device_id, channel, cq_id);
        tt_cxy_pair completion_q_writer_location = dispatch_core_manager::get(num_hw_cqs).completion_queue_writer_core(device_id, channel, cq_id);
        tt_cxy_pair dispatch_location = dispatch_core_manager::get(num_hw_cqs).dispatcher_core(device_id, channel, cq_id);
        CoreType dispatch_core_type = dispatch_core_manager::get(num_hw_cqs).get_dispatch_core_type(mmio_device_id);

        TT_ASSERT(prefetch_location.chip == mmio_device_id and completion_q_writer_location.chip == mmio_device_id,
            "Issue queue interface is on device {} and completion queue interface is on device {} but they are expected to be on device {}", prefetch_location.chip, completion_q_writer_location.chip, mmio_device_id);

        // Initialize the FetchQ
        std::vector<uint32_t> prefetch_q(dispatch_constants::get(dispatch_core_type).prefetch_q_entries(), 0);
        std::vector<uint32_t> prefetch_q_rd_ptr_addr_data = {
            (uint32_t)(dispatch_constants::PREFETCH_Q_BASE + dispatch_constants::get(dispatch_core_type).prefetch_q_size())
        };
        detail::WriteToDeviceL1(mmio_device, prefetch_location, CQ_PREFETCH_Q_RD_PTR, prefetch_q_rd_ptr_addr_data, dispatch_core_type);
        detail::WriteToDeviceL1(mmio_device, prefetch_location, dispatch_constants::PREFETCH_Q_BASE, prefetch_q, dispatch_core_type);

        // Initialize completion queue write pointer and read pointer copy
        uint32_t issue_queue_size = mmio_device->sysmem_manager_->get_issue_queue_size(cq_id);
        uint32_t completion_queue_start_addr = CQ_START + issue_queue_size + get_absolute_cq_offset(channel, cq_id, cq_size);
        uint32_t completion_queue_start_addr_16B = completion_queue_start_addr >> 4;
        vector<uint32_t> completion_queue_wr_ptr = {completion_queue_start_addr_16B};
        detail::WriteToDeviceL1(mmio_device, completion_q_writer_location, CQ_COMPLETION_READ_PTR, completion_queue_wr_ptr, dispatch_core_type);
        detail::WriteToDeviceL1(mmio_device, completion_q_writer_location, CQ_COMPLETION_WRITE_PTR, completion_queue_wr_ptr, dispatch_core_type);
        detail::WriteToDeviceL1(mmio_device, completion_q_writer_location, CQ0_COMPLETION_LAST_EVENT, zero, dispatch_core_type);
        detail::WriteToDeviceL1(mmio_device, completion_q_writer_location, CQ1_COMPLETION_LAST_EVENT, zero, dispatch_core_type);

        // Initialize address where workers signal to completion to dispatch core
        // This value is always increasing
        detail::WriteToDeviceL1(mmio_device, dispatch_location, DISPATCH_MESSAGE_ADDR, zero, dispatch_core_type);
        if (device_id != mmio_device_id) {
            tt_cxy_pair dispatch_d_location = dispatch_core_manager::get(num_hw_cqs).dispatcher_d_core(device_id, channel, cq_id);
            dispatch_core_type = dispatch_core_manager::get(num_hw_cqs).get_dispatch_core_type(device_id);
            detail::WriteToDeviceL1(this, dispatch_d_location, DISPATCH_MESSAGE_ADDR, zero, dispatch_core_type);
        }
    }

    detail::ConfigureDeviceWithProgram(this, command_queue_program, true);
    tt::Cluster::instance().l1_barrier(this->id());
    if (device_id != mmio_device_id) {
        Program& mmio_command_queue_program = *this->command_queue_programs[1];
        detail::ConfigureDeviceWithProgram(mmio_device, mmio_command_queue_program, true);
        tt::Cluster::instance().l1_barrier(mmio_device_id);
    }
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
    if (this->is_mmio_capable()) {
        TT_ASSERT(this->command_queue_programs.size() == 1);
    } else {
        TT_ASSERT(this->command_queue_programs.size() == 2);
    }
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

    if (!this->is_mmio_capable()) {
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->id());
        Device *mmio_device = tt::tt_metal::detail::GetDeviceHandle(mmio_device_id);
        Program& mmio_command_queue_program = *this->command_queue_programs[1];
        for (const auto &[core_type, logical_dispatch_cores] : mmio_command_queue_program.logical_cores()) {
            for (const CoreCoord &logical_dispatch_core : logical_dispatch_cores) {
                launch_msg_t msg = mmio_command_queue_program.kernels_on_core(logical_dispatch_core, core_type)->launch_msg;
                tt::llrt::write_launch_msg_to_core(mmio_device_id, mmio_device->physical_core_from_logical_core(logical_dispatch_core, core_type), &msg);
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

bool Device::initialize(size_t l1_small_size, const std::vector<uint32_t> &l1_bank_remap, bool minimal) {
    ZoneScoped;
    log_info(tt::LogMetal, "Initializing device {}. Program cache is {}enabled", this->id_, this->program_cache.is_enabled() ? "": "NOT ");
    this->initialize_cluster();
    this->initialize_allocator(l1_small_size, l1_bank_remap);
    this->initialize_build();
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    tt::tt_metal::device_pool::devices.resize(num_devices, nullptr);
    TT_ASSERT(id_ < num_devices);
    tt::tt_metal::device_pool::devices[id_] = this;
    // For minimal setup, don't initialize FW, watcher, dprint. They won't work if we're attaching to a hung chip.
    if (minimal)
        return true;

    bool already_initialized = this->active_devices_.activate_device(this->id_);
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

    for (const std::unique_ptr<HWCommandQueue> &hw_command_queue : hw_command_queues_) {
        hw_command_queue->terminate();
    }

    std::unordered_set<CoreCoord> not_done_dispatch_cores;
    std::unordered_set<CoreCoord> cores_to_skip;


    if (this->is_mmio_capable()) {
        for (const chip_id_t &device_id : tt::Cluster::instance().get_devices_controlled_by_mmio_device(this->id_)) {
            uint8_t curr_num_hw_cqs = device_id == this->id_ ? this->num_hw_cqs() : 1;
            uint16_t curr_channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
            CoreType dispatch_core_type = dispatch_core_manager::get(curr_num_hw_cqs).get_dispatch_core_type(device_id);
            for (uint8_t cq_id = 0; cq_id < curr_num_hw_cqs; cq_id++) {
                if (device_id == this->id_) {
                    //mmio device.
                    if (dispatch_core_manager::get(curr_num_hw_cqs).is_dispatcher_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair dispatch_location = dispatch_core_manager::get(curr_num_hw_cqs).dispatcher_core(device_id, curr_channel, cq_id);
                        CoreCoord phys_core = get_physical_core_coordinate(dispatch_location, dispatch_core_type);
                        not_done_dispatch_cores.insert(phys_core);
                        log_debug(tt::LogMetal, "MMIO Device Dispatch core: Logical: {} - Physical: {}", dispatch_location.str(), phys_core.str());
                    }
                    if (dispatch_core_manager::get(curr_num_hw_cqs).is_prefetcher_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair prefetch_location = dispatch_core_manager::get(curr_num_hw_cqs).prefetcher_core(device_id, curr_channel, cq_id);
                        CoreCoord phys_core = get_physical_core_coordinate(prefetch_location, dispatch_core_type);
                        not_done_dispatch_cores.insert(phys_core);
                        log_debug(tt::LogMetal, "MMIO Device Prefetch core: Logical: {} - Physical: {}", prefetch_location.str(), phys_core.str());
                    }
                } else if (this->active_devices_.is_device_active(device_id)) {
                    //non mmio devices serviced by this mmio capable device.
                    //skip remote dispatch cores only if respective remote device is active.
                    if (dispatch_core_manager::get(curr_num_hw_cqs).is_dispatcher_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair dispatch_location = dispatch_core_manager::get(curr_num_hw_cqs).dispatcher_core(device_id, curr_channel, cq_id);
                        CoreCoord phys_core = get_physical_core_coordinate(dispatch_location, dispatch_core_type);
                        cores_to_skip.insert(phys_core);
                        log_debug(tt::LogMetal, "Remote Device Dispatch core: Logical: {} - Physical: {} will keep running on MMIO Device.", dispatch_location.str(), phys_core.str());
                    }
                    if (dispatch_core_manager::get(curr_num_hw_cqs).is_prefetcher_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair prefetch_location = dispatch_core_manager::get(curr_num_hw_cqs).prefetcher_core(device_id, curr_channel, cq_id);
                        CoreCoord phys_core = get_physical_core_coordinate(prefetch_location, dispatch_core_type);
                        cores_to_skip.insert(phys_core);
                        log_debug(tt::LogMetal, "Remote Device Prefetch core: Logical: {} - Physical: {} will keep running on MMIO Device.", prefetch_location.str(), phys_core.str());
                    }
                    if (dispatch_core_manager::get(curr_num_hw_cqs).is_mux_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair mux_location = dispatch_core_manager::get(curr_num_hw_cqs).mux_core(device_id, curr_channel, cq_id);
                        CoreCoord phys_core = get_physical_core_coordinate(mux_location, dispatch_core_type);
                        cores_to_skip.insert(phys_core);
                        log_debug(tt::LogMetal, "Remote Device Mux core: Logical: {} - Physical: {} will keep running on MMIO Device.", mux_location.str(), phys_core.str());
                    }
                    if (dispatch_core_manager::get(curr_num_hw_cqs).is_demux_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair demux_location = dispatch_core_manager::get(curr_num_hw_cqs).demux_core(device_id, curr_channel, cq_id);
                        CoreCoord phys_core = get_physical_core_coordinate(demux_location, dispatch_core_type);
                        cores_to_skip.insert(phys_core);
                        log_debug(tt::LogMetal, "Remote Device Demux core: Logical: {} - Physical: {} will keep running on MMIO Device.", demux_location.str(), phys_core.str());
                    }
                    /*
                    tt_cxy_pair dispatch_location = dispatch_core_manager::get(curr_num_hw_cqs).dispatcher_core(device_id, curr_channel, cq_id);
                    tt_cxy_pair prefetch_location = dispatch_core_manager::get(curr_num_hw_cqs).prefetcher_core(device_id, curr_channel, cq_id);
                    tt_cxy_pair mux_location = dispatch_core_manager::get(curr_num_hw_cqs).mux_core(device_id, curr_channel, cq_id);
                    tt_cxy_pair demux_location = dispatch_core_manager::get(curr_num_hw_cqs).demux_core(device_id, curr_channel, cq_id);
                    cores_to_skip.insert(get_physical_core_coordinate(dispatch_location, dispatch_core_type));
                    cores_to_skip.insert(get_physical_core_coordinate(prefetch_location, dispatch_core_type));
                    cores_to_skip.insert(get_physical_core_coordinate(mux_location, dispatch_core_type));
                    cores_to_skip.insert(get_physical_core_coordinate(demux_location, dispatch_core_type));
                    log_debug(tt::LogMetal, "Remote Device dispatch cores: {} : {} : {} : {} will keep running on MMIO Device.", dispatch_location.str(), prefetch_location.str(), mux_location.str(), demux_location.str());
                    */
                }
            }
        }
    } else {
        //remote device that is active
        uint8_t curr_num_hw_cqs = 1;
        auto device_id = this->id_;
        uint16_t curr_channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
        CoreType dispatch_core_type = dispatch_core_manager::get(curr_num_hw_cqs).get_dispatch_core_type(device_id);
        for (uint8_t cq_id = 0; cq_id < curr_num_hw_cqs; cq_id++) {
            if (dispatch_core_manager::get(curr_num_hw_cqs).is_dispatcher_core_allocated(device_id, curr_channel, cq_id)) {
                tt_cxy_pair dispatch_location = dispatch_core_manager::get(curr_num_hw_cqs).dispatcher_core(device_id, curr_channel, cq_id);
                CoreCoord phys_core = get_physical_core_coordinate(dispatch_location, dispatch_core_type);
                not_done_dispatch_cores.insert(phys_core);
                log_debug(tt::LogMetal, "Remote Device Dispatch core: Logical: {} - Physical: {} will be reset on MMIO Device.", dispatch_location.str(), phys_core.str());
            }
            if (dispatch_core_manager::get(curr_num_hw_cqs).is_prefetcher_core_allocated(device_id, curr_channel, cq_id)) {
                tt_cxy_pair prefetch_location = dispatch_core_manager::get(curr_num_hw_cqs).prefetcher_core(device_id, curr_channel, cq_id);
                CoreCoord phys_core = get_physical_core_coordinate(prefetch_location, dispatch_core_type);
                not_done_dispatch_cores.insert(phys_core);
                log_debug(tt::LogMetal, "Remote Device Prefetch core: Logical: {} - Physical: {} will be reset on MMIO Device.", prefetch_location.str(), phys_core.str());
            }
            if (dispatch_core_manager::get(curr_num_hw_cqs).is_mux_core_allocated(device_id, curr_channel, cq_id)) {
                tt_cxy_pair mux_location = dispatch_core_manager::get(curr_num_hw_cqs).mux_core(device_id, curr_channel, cq_id);
                CoreCoord phys_core = get_physical_core_coordinate(mux_location, dispatch_core_type);
                not_done_dispatch_cores.insert(phys_core);
                log_debug(tt::LogMetal, "Remote Device Mux core: Logical: {} - Physical: {} will be reset on MMIO Device.", mux_location.str(), phys_core.str());
            }
            if (dispatch_core_manager::get(curr_num_hw_cqs).is_demux_core_allocated(device_id, curr_channel, cq_id)) {
                tt_cxy_pair demux_location = dispatch_core_manager::get(curr_num_hw_cqs).demux_core(device_id, curr_channel, cq_id);
                CoreCoord phys_core = get_physical_core_coordinate(demux_location, dispatch_core_type);
                not_done_dispatch_cores.insert(phys_core);
                log_debug(tt::LogMetal, "Remote Device Demux core: Logical: {} - Physical: {} will be reset on MMIO Device.", demux_location.str(), phys_core.str());
            }
            /*
            tt_cxy_pair dispatch_location = dispatch_core_manager::get(curr_num_hw_cqs).dispatcher_core(device_id, curr_channel, cq_id);
            tt_cxy_pair prefetch_location = dispatch_core_manager::get(curr_num_hw_cqs).prefetcher_core(device_id, curr_channel, cq_id);
            tt_cxy_pair mux_location = dispatch_core_manager::get(curr_num_hw_cqs).mux_core(device_id, curr_channel, cq_id);
            tt_cxy_pair demux_location = dispatch_core_manager::get(curr_num_hw_cqs).demux_core(device_id, curr_channel, cq_id);
            not_done_dispatch_cores.insert(get_physical_core_coordinate(dispatch_location, dispatch_core_type));
            not_done_dispatch_cores.insert(get_physical_core_coordinate(prefetch_location, dispatch_core_type));
            not_done_dispatch_cores.insert(get_physical_core_coordinate(mux_location, dispatch_core_type));
            not_done_dispatch_cores.insert(get_physical_core_coordinate(demux_location, dispatch_core_type));
            log_debug(tt::LogMetal, "Remote Device dispatch cores {} : {} : {} : {} will be reset on MMIO Device.", dispatch_location.str(), prefetch_location.str(), mux_location.str(), demux_location.str());
            */
        }
    }

    auto mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->id_);
    std::unordered_set<CoreCoord> wait_for_cores = not_done_dispatch_cores;

    llrt::internal_::wait_until_cores_done(mmio_device_id, RUN_MSG_GO, wait_for_cores);

    DprintServerDetach(this);

    // Assert worker cores
    CoreCoord grid_size = this->logical_grid_size();
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core = this->worker_core_from_logical_core(logical_core);

            if (cores_to_skip.find(worker_core) == cores_to_skip.end()) {
                if (this->storage_only_cores_.find(logical_core) == this->storage_only_cores_.end()) {
                    tt::Cluster::instance().assert_risc_reset_at_core(tt_cxy_pair(this->id(), worker_core));
                }
            } else {
                log_debug(tt::LogMetal, "{} will not be Reset when closing Device {}", worker_core.str(), this->id());
            }
        }
    }

    if (this->id_ != mmio_device_id) {
        for (auto it = not_done_dispatch_cores.begin(); it != not_done_dispatch_cores.end(); it++) {
            const auto &phys_core = *it;
            if(llrt::is_ethernet_core(phys_core, this->id_)) {
                log_debug(tt::LogMetal, "Ethernet dispatch core {} on Device {} is idle. Closing Device {}", phys_core.str(), mmio_device_id, this->id());
            } else {
                log_debug(tt::LogMetal, "Resetting core {} on Device {} when closing Device {}", phys_core.str(), mmio_device_id, this->id());
                tt::Cluster::instance().assert_risc_reset_at_core(tt_cxy_pair(mmio_device_id, phys_core));
            }
        }
    }

    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);

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
    TT_FATAL(
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
    TT_FATAL( cq_id < hw_command_queues_.size(), "cq_id {} is out of range", cq_id );
    TT_FATAL(this->is_initialized(), "Device has not been initialized, did you forget to call InitializeDevice?");
    return *hw_command_queues_[cq_id];
}

CommandQueue &Device::command_queue(size_t cq_id) {
    detail::DispatchStateCheck(using_fast_dispatch);
    TT_FATAL( cq_id < sw_command_queues_.size(), "cq_id {} is out of range", cq_id );
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

    // Currently only supports one trace at a time per CQ, so release last trace
    // before instantiating new ones.
    this->release_last_trace();

    for (size_t cq_id = 0; cq_id < num_hw_cqs(); cq_id++) {
        hw_command_queues_[cq_id]->record_end();
        trace_contexts_.at(cq_id)->data = std::move(this->sysmem_manager().get_bypass_data());
        uint32_t tid = Trace::instantiate(this->command_queue(cq_id), trace_contexts_.at(cq_id));
        trace_insts_.push_back(tid);
    }
}

void Device::execute_last_trace(bool blocking) {
    constexpr bool check = false;
    for (size_t cq_id = 0; cq_id < num_hw_cqs(); cq_id++) {
        if (this->trace_insts_.at(cq_id).has_value()) {
            uint32_t tid = this->trace_insts_.at(cq_id).value();
            TT_FATAL(Trace::has_instance(tid), "Trace instance " + std::to_string(tid) + " must exist on device");
            if constexpr (check) {
                Trace::validate_instance(tid);
            }
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
    this->trace_insts_.clear();
}

}  // namespace tt_metal

}  // namespace tt
