// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/common/core_descriptor.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "impl/debug/dprint_server.hpp"
#include "impl/debug/watcher_server.hpp"
#include "tt_metal/third_party/umd/device/util.hpp"


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

Device::Device(chip_id_t device_id, const uint8_t num_hw_cqs, const std::vector<uint32_t>& l1_bank_remap) : id_(device_id), num_hw_cqs_(num_hw_cqs)
{
    ZoneScoped;
    TT_ASSERT(num_hw_cqs > 0 and num_hw_cqs < 3, "num_hw_cqs can be between 1 and 2");
    this->initialize(l1_bank_remap);
    if (this->worker_queue_mode == WorkerQueueMode::ASYNCHRONOUS) {
        this->worker_queue.parent_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        this->start_worker();
    }
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

void Device::initialize_allocator(const std::vector<uint32_t>& l1_bank_remap) {
    ZoneScoped;
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    // Construct allocator config from soc_desc
    AllocatorConfig config({
        .num_dram_channels = static_cast<size_t>(soc_desc.get_num_dram_channels()),
        .dram_bank_size = soc_desc.dram_bank_size,
        .dram_bank_offsets = {},
        .worker_grid_size = this->logical_grid_size(),
        .worker_l1_size = static_cast<size_t>(soc_desc.worker_l1_size),
        .l1_bank_size = static_cast<size_t>(get_storage_core_bank_size(this->id_, this->num_hw_cqs_)),
        .core_type_from_noc_coord_table = {}, // Populated later
        .worker_log_to_physical_routing_x=soc_desc.worker_log_to_physical_routing_x,
        .worker_log_to_physical_routing_y=soc_desc.worker_log_to_physical_routing_y,
        .l1_bank_remap = l1_bank_remap,
        .compute_grid_size = this->compute_with_storage_grid_size()
    });
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

// TODO: This will be removed once FD v1.3 is backported to Grayskull
void Device::compile_command_queue_programs_for_grayskull() {
    ZoneScoped;
    unique_ptr<Program, detail::ProgramDeleter> command_queue_program_ptr(new Program);

    const uint32_t num_tensix_command_slots = 2;
    uint32_t cmd_start_tensix = get_command_start_l1_address(false);
    uint32_t data_section_addr_tensix = get_data_section_l1_address(false, false);
    uint32_t producer_data_buffer_size_tensix = get_cq_data_buffer_size(false, false);
    uint32_t consumer_data_buffer_size_tensix = get_consumer_data_buffer_size();

    uint8_t num_hw_cqs = this->num_hw_cqs();
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->id());
    uint32_t cq_size = this->sysmem_manager().get_cq_size();

    for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
        tt_cxy_pair issue_q_reader_location = dispatch_core_manager::get(num_hw_cqs).issue_queue_reader_core(this->id(), channel, cq_id);
        tt_cxy_pair completion_q_writer_location = dispatch_core_manager::get(num_hw_cqs).completion_queue_writer_core(this->id(), channel, cq_id);

        TT_ASSERT(issue_q_reader_location.chip == this->id() and completion_q_writer_location.chip == this->id(),
            "Issue queue interface is on device {} and completion queue interface is on device {} but they are expected to be on device {}", issue_q_reader_location.chip, completion_q_writer_location.chip, this->id());

        CoreCoord issue_q_logical_core(issue_q_reader_location.x, issue_q_reader_location.y);
        CoreCoord completion_q_logical_core(completion_q_writer_location.x, completion_q_writer_location.y);
        CoreCoord issue_q_physical_core = get_physical_core_coordinate(issue_q_reader_location, CoreType::WORKER);
        CoreCoord completion_q_physical_core = get_physical_core_coordinate(completion_q_writer_location, CoreType::WORKER);

        std::map<string, string> producer_defines = {
            {"DISPATCH_KERNEL", "1"},
            {"CONSUMER_NOC_X", std::to_string(completion_q_physical_core.x)},
            {"CONSUMER_NOC_Y", std::to_string(completion_q_physical_core.y)},
        };
        std::map<string, string> consumer_defines = {
            {"DISPATCH_KERNEL", "1"},
            {"PRODUCER_NOC_X", std::to_string(issue_q_physical_core.x)},
            {"PRODUCER_NOC_Y", std::to_string(issue_q_physical_core.y)},
        };

        // Address in sysmem for CQ to write back its read ptr to
        uint32_t host_issue_queue_read_ptr_addr = HOST_CQ_ISSUE_READ_PTR + get_absolute_cq_offset(channel, cq_id, cq_size);
        uint32_t issue_queue_start_addr = CQ_START + get_absolute_cq_offset(channel, cq_id, cq_size);
        uint32_t issue_queue_size = tt::round_up((cq_size - CQ_START) * SystemMemoryCQInterface::default_issue_queue_split, 32);

        uint32_t consumer_cmd_base_addr = cmd_start_tensix;
        uint32_t consumer_data_buff_size = consumer_data_buffer_size_tensix;

        std::vector<uint32_t> producer_compile_args = {
            host_issue_queue_read_ptr_addr,
            issue_queue_start_addr,
            issue_queue_size,
            cmd_start_tensix,
            data_section_addr_tensix,
            producer_data_buffer_size_tensix,
            consumer_cmd_base_addr,
            consumer_data_buff_size};

        uint32_t host_completion_queue_write_ptr_addr = HOST_CQ_COMPLETION_WRITE_PTR + get_absolute_cq_offset(channel, cq_id, cq_size);
        uint32_t completion_queue_start_addr = CQ_START + issue_queue_size + get_absolute_cq_offset(channel, cq_id, cq_size);
        uint32_t completion_queue_size = (cq_size - CQ_START) - issue_queue_size;
        uint32_t host_finish_addr = HOST_CQ_FINISH_PTR + get_absolute_cq_offset(channel, cq_id, cq_size);
        std::vector<uint32_t> consumer_compile_args = {host_completion_queue_write_ptr_addr, completion_queue_start_addr, completion_queue_size, host_finish_addr, consumer_cmd_base_addr, consumer_data_buff_size};

        tt::tt_metal::CreateKernel(
            *command_queue_program_ptr,
            "tt_metal/impl/dispatch/kernels/command_queue_producer.cpp",
            issue_q_logical_core,
            tt::tt_metal::DataMovementConfig {
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = producer_compile_args,
                .defines = producer_defines});

        tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, issue_q_logical_core, num_tensix_command_slots);

        tt::tt_metal::CreateKernel(
            *command_queue_program_ptr,
            "tt_metal/impl/dispatch/kernels/command_queue_consumer.cpp",
            completion_q_logical_core,
            tt::tt_metal::DataMovementConfig {
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = consumer_compile_args,
                .defines = consumer_defines});

        tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, completion_q_logical_core, 0);
    }
    detail::CompileProgram(this, *command_queue_program_ptr);
    this->command_queue_programs.push_back(std::move(command_queue_program_ptr));
}

// TODO (abhullar): Refactor this with #2593 to allow each target fast dispatch (FD) device to program their associated FD cores regardless of whether they are on the target device or not.
// Currently we have to program FD cores for the remote device when initializing the MMIO device because completion queue cores are on MMIO device
//  and we don't have handle on MMIO device when initializing the remote device
void Device::compile_command_queue_programs() {
    ZoneScoped;
    unique_ptr<Program, detail::ProgramDeleter> command_queue_program_ptr(new Program);

    // Currently we only double buffer commands in tensix cores
    const uint32_t num_tensix_command_slots = 2;
    const uint32_t num_eth_command_slots = 1;
    const uint32_t accept_cmd_sem_value = 0;

    uint32_t cmd_start_tensix = get_command_start_l1_address(false);
    uint32_t data_section_addr_tensix = get_data_section_l1_address(false, false);
    uint32_t producer_data_buffer_size_tensix = get_cq_data_buffer_size(false, false);
    uint32_t consumer_data_buffer_size_tensix = get_cq_data_buffer_size(false, false);

    // Idle erisc dispatch
    uint32_t cmd_start_eth_dispatch = get_command_start_l1_address(true);
    uint32_t consumer_data_buffer_size_eth_dispatch = get_cq_data_buffer_size(true, true);

    uint32_t producer_data_buffer_size_eth = get_cq_data_buffer_size(true, false);
    uint32_t consumer_data_buffer_size_eth = get_cq_data_buffer_size(true, false);

    // Eth tunneller kernel
    uint32_t issue_path_cmd_start_eth = get_eth_command_start_l1_address(SyncCBConfigRegion::ROUTER_ISSUE);
    uint32_t completion_path_cmd_start_eth = get_eth_command_start_l1_address(SyncCBConfigRegion::ROUTER_COMPLETION);

    if (this->is_mmio_capable()) {
        for (const chip_id_t &device_id : tt::Cluster::instance().get_devices_controlled_by_mmio_device(this->id())) {
            // TODO (abhullar): allow for multiple cqs on remote device, atm device initialization asserts one cq for the remote device
            uint8_t num_hw_cqs = device_id == this->id() ? this->num_hw_cqs() : 1;
            uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
            uint32_t cq_size = this->sysmem_manager().get_cq_size();

            for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
                tt_cxy_pair issue_q_reader_location = dispatch_core_manager::get(num_hw_cqs).issue_queue_reader_core(device_id, channel, cq_id);
                tt_cxy_pair completion_q_writer_location = dispatch_core_manager::get(num_hw_cqs).completion_queue_writer_core(device_id, channel, cq_id);
                tt_cxy_pair dispatch_location = dispatch_core_manager::get(num_hw_cqs).command_dispatcher_core(device_id, channel, cq_id);
                CoreType dispatch_core_type = dispatch_core_manager::get(num_hw_cqs).get_dispatch_core_type(device_id);

                TT_ASSERT(issue_q_reader_location.chip == this->id() and completion_q_writer_location.chip == this->id(),
                    "Issue queue interface is on device {} and completion queue interface is on device {} but they are expected to be on device {}", issue_q_reader_location.chip, completion_q_writer_location.chip, this->id());

                CoreCoord issue_q_physical_core = get_physical_core_coordinate(issue_q_reader_location, dispatch_core_type);
                CoreCoord completion_q_physical_core = get_physical_core_coordinate(completion_q_writer_location, dispatch_core_type);
                CoreCoord dispatch_physical_core = get_physical_core_coordinate(dispatch_location, dispatch_core_type);

                CoreCoord consumer_physical_core = completion_q_physical_core;
                CoreCoord producer_physical_core = issue_q_physical_core;
                if (device_id != this->id()) {
                    // This means the issue queue and completion queue interfaces that service a remote device are being set up
                    // the issue queue interface needs to send fast dispatch packets to the "src" ethernet core and
                    // the completion queue writer receives packets from the "dst" ethernet core
                    tt_cxy_pair logical_eth_router_src = tt::Cluster::instance().get_eth_core_for_dispatch_core(
                        issue_q_reader_location, EthRouterMode::BI_DIR_TUNNELING, device_id);
                    consumer_physical_core = this->ethernet_core_from_logical_core(logical_eth_router_src);

                    // remote_issue_q writing to eth SRC, semaphore 0
                    tt::Cluster::instance().write_core(&accept_cmd_sem_value, sizeof(uint32_t), tt_cxy_pair(this->id(), consumer_physical_core), eth_l1_mem::address_map::SEMAPHORE_BASE);

                    tt_cxy_pair logical_eth_router_dst = tt::Cluster::instance().get_eth_core_for_dispatch_core(
                        completion_q_writer_location, EthRouterMode::BI_DIR_TUNNELING, device_id);
                    producer_physical_core = this->ethernet_core_from_logical_core(logical_eth_router_dst);

                    // remote_command_processor receiving from eth DST, semaphore 1
                    tt::Cluster::instance().write_core(&num_eth_command_slots, sizeof(uint32_t), tt_cxy_pair(this->id(), producer_physical_core), eth_l1_mem::address_map::SEMAPHORE_BASE + L1_ALIGNMENT);

                    // Setup eth core for bidirectional tunneling
                    std::map<string, string> eth_tunneller_defines = {
                        {"DISPATCH_KERNEL", "1"}, //TODO: do we need this?
                        {"CONSUMER_NOC_X", std::to_string(completion_q_physical_core.x)},
                        {"CONSUMER_NOC_Y", std::to_string(completion_q_physical_core.y)},
                        {"PRODUCER_NOC_X", std::to_string(issue_q_physical_core.x)},
                        {"PRODUCER_NOC_Y", std::to_string(issue_q_physical_core.y)},
                    };
                    std::vector<uint32_t> eth_tunneller_compile_args = {true, num_eth_command_slots};
                    std::string command_q_tunneller_kernel = "tt_metal/impl/dispatch/kernels/command_queue_bidirectional_tunneller.cpp";
                    tt::tt_metal::CreateKernel(
                        *command_queue_program_ptr,
                        command_q_tunneller_kernel,
                        logical_eth_router_src,
                        tt::tt_metal::EthernetConfig {
                            .noc = tt::tt_metal::NOC::RISCV_0_default,
                            .compile_args = eth_tunneller_compile_args,
                            .defines = eth_tunneller_defines});
                }

                TT_ASSERT(tt::Cluster::instance().get_soc_desc(this->id()).pcie_cores.size() == 1);
                CoreCoord pcie_physical_core = tt::Cluster::instance().get_soc_desc(this->id()).pcie_cores.at(0);

                std::map<string, string> producer_defines = {
                    {"DISPATCH_KERNEL", "1"},
                    {"PULL_NOC_X", std::to_string(pcie_physical_core.x)},
                    {"PULL_NOC_Y", std::to_string(pcie_physical_core.y)},
                    {"PUSH_NOC_X", std::to_string(consumer_physical_core.x)},
                    {"PUSH_NOC_Y", std::to_string(consumer_physical_core.y)},
                    {"DISPATCH_NOC_X", std::to_string(dispatch_physical_core.x)},
                    {"DISPATCH_NOC_Y", std::to_string(dispatch_physical_core.y)},
                };
                std::map<string, string> consumer_defines = {
                    {"DISPATCH_KERNEL", "1"},
                    {"PRODUCER_NOC_X", std::to_string(producer_physical_core.x)},
                    {"PRODUCER_NOC_Y", std::to_string(producer_physical_core.y)},
                };

                // Address in sysmem for CQ to write back its read ptr to
                bool eth_core = dispatch_core_type == CoreType::ETH;
                uint32_t host_issue_queue_read_ptr_addr = HOST_CQ_ISSUE_READ_PTR + get_absolute_cq_offset(channel, cq_id, cq_size);
                uint32_t issue_queue_start_addr = CQ_START + get_absolute_cq_offset(channel, cq_id, cq_size);
                uint32_t issue_queue_size = tt::round_up((cq_size - CQ_START) * SystemMemoryCQInterface::default_issue_queue_split, 32);

                uint32_t host_completion_queue_write_ptr_addr = HOST_CQ_COMPLETION_WRITE_PTR + get_absolute_cq_offset(channel, cq_id, cq_size);
                uint32_t completion_queue_start_addr = CQ_START + issue_queue_size + get_absolute_cq_offset(channel, cq_id, cq_size);
                uint32_t completion_queue_size = (cq_size - CQ_START) - issue_queue_size;
                uint32_t host_finish_addr = HOST_CQ_FINISH_PTR + get_absolute_cq_offset(channel, cq_id, cq_size);

                uint32_t consumer_cmd_base_addr =  (device_id != this->id()) ? issue_path_cmd_start_eth : eth_core ? cmd_start_eth_dispatch : cmd_start_tensix; // device is MMIO capable but current device_id being set up is remote
                uint32_t consumer_data_buff_size = (device_id != this->id()) ? consumer_data_buffer_size_eth : eth_core ? consumer_data_buffer_size_eth_dispatch : consumer_data_buffer_size_tensix; // device is MMIO capable but current device_id being set up is remote

                uint32_t cmd_start_producer = eth_core ? cmd_start_eth_dispatch : cmd_start_tensix;
                uint32_t data_section_addr_producer = eth_core ? get_data_section_l1_address(true, true) : data_section_addr_tensix;
                uint32_t producer_data_buffer_size = eth_core ? get_cq_data_buffer_size(true, true) : producer_data_buffer_size_tensix;

                tt::PullAndPushConfig pull_and_push_config = (device_id != this->id()) ? tt::PullAndPushConfig::PUSH_TO_REMOTE : tt::PullAndPushConfig::LOCAL;
                std::vector<uint32_t> producer_compile_args = {
                    host_issue_queue_read_ptr_addr,
                    issue_queue_start_addr,
                    issue_queue_size,
                    host_completion_queue_write_ptr_addr,
                    completion_queue_start_addr,
                    completion_queue_size,
                    host_finish_addr,
                    cmd_start_producer,
                    data_section_addr_producer,
                    producer_data_buffer_size,
                    consumer_cmd_base_addr,
                    consumer_data_buff_size,
                    (uint32_t)pull_and_push_config
                };

                std::string pull_and_push_kernel = "tt_metal/impl/dispatch/kernels/cq_prefetcher.cpp";
                if (dispatch_core_type != CoreType::ETH) {
                    tt::tt_metal::CreateKernel(
                        *command_queue_program_ptr,
                        pull_and_push_kernel,
                        issue_q_reader_location,
                        tt::tt_metal::DataMovementConfig {
                            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                            .noc = tt::tt_metal::NOC::RISCV_0_default,
                            .compile_args = producer_compile_args,
                            .defines = producer_defines});
                } else {
                    tt::tt_metal::CreateKernel(
                        *command_queue_program_ptr,
                        pull_and_push_kernel,
                        issue_q_reader_location,
                        tt::tt_metal::EthernetConfig {
                            .eth_mode = Eth::IDLE,
                            .noc = tt::tt_metal::NOC::RISCV_0_default,
                            .compile_args = producer_compile_args,
                            .defines = producer_defines});
                }

                uint32_t num_command_slots = (device_id == this->id()) ? num_tensix_command_slots : num_eth_command_slots;
                tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, issue_q_reader_location, num_command_slots, dispatch_core_type);
                tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, issue_q_reader_location, 0, dispatch_core_type);
                tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, issue_q_reader_location, num_tensix_command_slots, dispatch_core_type); // semaphore between push&pull kernel and dispatch kernel

                if (device_id == this->id()) {
                    uint32_t cmd_start_consumer = eth_core ? cmd_start_eth_dispatch : cmd_start_tensix;
                    uint32_t consumer_data_buffer_size = eth_core ? consumer_data_buffer_size_eth_dispatch : consumer_data_buffer_size_tensix;
                    std::vector<uint32_t> consumer_compile_args = {cmd_start_consumer, consumer_data_buffer_size};

                    if (dispatch_core_type != CoreType::ETH) {
                        tt::tt_metal::CreateKernel(
                            *command_queue_program_ptr,
                            "tt_metal/impl/dispatch/kernels/cq_dispatcher.cpp",
                            dispatch_location,
                            tt::tt_metal::DataMovementConfig {
                                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                                .noc = tt::tt_metal::NOC::RISCV_0_default,
                                .compile_args = consumer_compile_args,
                                .defines = consumer_defines});
                    } else {
                        tt::tt_metal::CreateKernel(
                            *command_queue_program_ptr,
                            "tt_metal/impl/dispatch/kernels/cq_dispatcher.cpp",
                            dispatch_location,
                            tt::tt_metal::EthernetConfig {
                                .eth_mode = Eth::IDLE,
                                .noc = tt::tt_metal::NOC::RISCV_0_default,
                                .compile_args = consumer_compile_args,
                                .defines = consumer_defines});
                    }

                    tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, dispatch_location, 0, dispatch_core_type);
                } else {
                    // program the completion queue writer for the remote command queue

                    std::map<string, string> completion_q_defines = {
                        {"DISPATCH_KERNEL", "1"},
                        {"PULL_NOC_X", std::to_string(producer_physical_core.x)},
                        {"PULL_NOC_Y", std::to_string(producer_physical_core.y)},
                        {"PUSH_NOC_X", std::to_string(pcie_physical_core.x)},
                        {"PUSH_NOC_Y", std::to_string(pcie_physical_core.y)},
                        {"DISPATCH_NOC_X", std::to_string(pcie_physical_core.x)},   // this is unused by completion queue writer
                        {"DISPATCH_NOC_Y", std::to_string(pcie_physical_core.y)},   // this is unused by completion queue writer
                    };

                    std::vector<uint32_t> completion_q_writer_args = {
                        host_issue_queue_read_ptr_addr,
                        issue_queue_start_addr,
                        issue_queue_size,
                        host_completion_queue_write_ptr_addr,
                        completion_queue_start_addr,
                        completion_queue_size,
                        host_finish_addr,
                        cmd_start_tensix,
                        data_section_addr_tensix,
                        producer_data_buffer_size_tensix,
                        consumer_cmd_base_addr,
                        consumer_data_buff_size,
                        (uint32_t)tt::PullAndPushConfig::PULL_FROM_REMOTE
                    };

                    if (dispatch_core_type != CoreType::ETH) {
                        tt::tt_metal::CreateKernel(
                            *command_queue_program_ptr,
                            pull_and_push_kernel,
                            completion_q_writer_location,
                            tt::tt_metal::DataMovementConfig{
                                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                                .noc = tt::tt_metal::NOC::RISCV_0_default,
                                .compile_args = completion_q_writer_args,
                                .defines = completion_q_defines});
                    } else {
                        tt::tt_metal::CreateKernel(
                            *command_queue_program_ptr,
                            pull_and_push_kernel,
                            completion_q_writer_location,
                            tt::tt_metal::EthernetConfig{
                                .eth_mode = Eth::IDLE,
                                .noc = tt::tt_metal::NOC::RISCV_0_default,
                                .compile_args = completion_q_writer_args,
                                .defines = completion_q_defines});
                    }

                    tt::tt_metal::CreateSemaphore(
                        *command_queue_program_ptr,
                        completion_q_writer_location,
                        num_eth_command_slots,
                        dispatch_core_type);  // push semaphore
                    tt::tt_metal::CreateSemaphore(
                        *command_queue_program_ptr,
                        completion_q_writer_location,
                        0,
                        dispatch_core_type);  // pull semaphore
                    tt::tt_metal::CreateSemaphore(
                        *command_queue_program_ptr,
                        completion_q_writer_location,
                        num_tensix_command_slots,
                        dispatch_core_type);  // semaphore between push&pull kernel and dispatch kernel
                }
            }
        }
    } else {
        TT_ASSERT(this->num_hw_cqs() == 1, "Currently can only support one command queue for remote device");
        uint8_t num_hw_cqs = this->num_hw_cqs();
        const uint8_t cq_id = 0;
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->id());
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->id());
        uint32_t cq_size = this->sysmem_manager().get_cq_size();

        tt_cxy_pair remote_processor_location = dispatch_core_manager::get(num_hw_cqs).remote_push_and_pull_core(this->id(), channel, cq_id);
        tt_cxy_pair dispatch_location = dispatch_core_manager::get(num_hw_cqs).command_dispatcher_core(this->id(), channel, cq_id);
        CoreType dispatch_core_type = dispatch_core_manager::get(num_hw_cqs).get_dispatch_core_type(this->id());

        CoreCoord remote_processor_physical_core =
            get_physical_core_coordinate(remote_processor_location, dispatch_core_type);
        CoreCoord dispatch_physical_core = get_physical_core_coordinate(dispatch_location, dispatch_core_type);

        // Set up the dst router to receive fast dispatch packets
        tt_cxy_pair logical_eth_router_remote_dst = tt::Cluster::instance().get_eth_core_for_dispatch_core(remote_processor_location, EthRouterMode::BI_DIR_TUNNELING, mmio_device_id);
        CoreCoord physical_eth_router_remote_dst = this->ethernet_core_from_logical_core(logical_eth_router_remote_dst);

        // TODO (abhullar / aliu): there is no API to configure ethernet semaphores used for FD so manually write initial semaphore value
        // remote_completion_writer receiving from eth DST, semaphore 1
        tt::Cluster::instance().write_core(&num_eth_command_slots, sizeof(uint32_t), tt_cxy_pair(this->id(), physical_eth_router_remote_dst), eth_l1_mem::address_map::SEMAPHORE_BASE + L1_ALIGNMENT);

        // Set up the src router on remote device to send fast dispatch packets on the return path to MMIO device
        CoreCoord logical_eth_router_remote_src = tt::Cluster::instance().get_eth_core_for_dispatch_core(
            remote_processor_location, EthRouterMode::BI_DIR_TUNNELING, mmio_device_id);

        // remote_signaller writing to eth SRC, semaphore 0
        CoreCoord physical_eth_router_remote_src = this->ethernet_core_from_logical_core(logical_eth_router_remote_src);
        tt::Cluster::instance().write_core(&accept_cmd_sem_value, sizeof(uint32_t), tt_cxy_pair(this->id(), physical_eth_router_remote_src), eth_l1_mem::address_map::SEMAPHORE_BASE);
        // TODO: aliu add more bidirection tunneling kernels for multihop dispatch
          // Setup eth core for bidirectional tunneling
            std::map<string, string> eth_tunneller_defines = {
                {"DISPATCH_KERNEL", "1"}, //TODO: do we need this?
                {"CONSUMER_NOC_X", std::to_string(remote_processor_physical_core.x)},
                {"CONSUMER_NOC_Y", std::to_string(remote_processor_physical_core.y)},
                {"PRODUCER_NOC_X", std::to_string(remote_processor_physical_core.x)},
                {"PRODUCER_NOC_Y", std::to_string(remote_processor_physical_core.y)},
            };
            std::vector<uint32_t> eth_tunneller_compile_args = {false, num_eth_command_slots}; // SENDER is ISSUE
            std::string command_q_tunneller_kernel = "tt_metal/impl/dispatch/kernels/command_queue_bidirectional_tunneller.cpp";
            tt::tt_metal::CreateKernel(
                *command_queue_program_ptr,
                command_q_tunneller_kernel,
                logical_eth_router_remote_src,
                tt::tt_metal::EthernetConfig {
                    .noc = tt::tt_metal::NOC::RISCV_0_default,
                    .compile_args = eth_tunneller_compile_args,
                    .defines = eth_tunneller_defines});

        bool eth_core = dispatch_core_type == CoreType::ETH;
        uint32_t cmd_start_consumer = eth_core ? cmd_start_eth_dispatch : cmd_start_tensix;
        uint32_t consumer_data_buffer_size = eth_core ? consumer_data_buffer_size_eth_dispatch : consumer_data_buffer_size_tensix;
        uint32_t data_section_addr_producer = eth_core ? get_data_section_l1_address(true, true) : data_section_addr_tensix;
        uint32_t producer_data_buffer_size = eth_core ? get_cq_data_buffer_size(true, true) : producer_data_buffer_size_tensix;
        std::vector<uint32_t> remote_pull_and_push_compile_args = {
            0, // host_issue_queue_read_ptr_addr,
            0, // issue_queue_start_addr,
            0, // issue_queue_size,
            0, // host_completion_queue_write_ptr_addr,
            0, // completion_queue_start_addr,
            0, // completion_queue_size,
            0, // host_finish_addr
            cmd_start_consumer,
            data_section_addr_producer,
            producer_data_buffer_size,
            cmd_start_consumer,
            consumer_data_buffer_size,
            (uint32_t)tt::PullAndPushConfig::REMOTE_PULL_AND_PUSH
        };

        std::map<string, string> remote_pull_and_push_defines = {
            {"DISPATCH_KERNEL", "1"},
            {"PULL_NOC_X", std::to_string(physical_eth_router_remote_dst.x)},
            {"PULL_NOC_Y", std::to_string(physical_eth_router_remote_dst.y)},
            {"PUSH_NOC_X", std::to_string(physical_eth_router_remote_src.x)},
            {"PUSH_NOC_Y", std::to_string(physical_eth_router_remote_src.y)},
            {"DISPATCH_NOC_X", std::to_string(dispatch_physical_core.x)},
            {"DISPATCH_NOC_Y", std::to_string(dispatch_physical_core.y)},
        };

        if (dispatch_core_type != CoreType::ETH) {
            tt::tt_metal::CreateKernel(
                *command_queue_program_ptr,
                "tt_metal/impl/dispatch/kernels/cq_prefetcher.cpp",
                remote_processor_location,
                tt::tt_metal::DataMovementConfig{
                    .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt::tt_metal::NOC::RISCV_0_default,
                    .compile_args = remote_pull_and_push_compile_args,
                    .defines = remote_pull_and_push_defines});
        } else {
            tt::tt_metal::CreateKernel(
                *command_queue_program_ptr,
                "tt_metal/impl/dispatch/kernels/cq_prefetcher.cpp",
                remote_processor_location,
                tt::tt_metal::EthernetConfig{
                    .eth_mode = Eth::IDLE,
                    .noc = tt::tt_metal::NOC::RISCV_0_default,
                    .compile_args = remote_pull_and_push_compile_args,
                    .defines = remote_pull_and_push_defines});
        }

        // first semaphore is between pull_and_relay and pusher
        tt::tt_metal::CreateSemaphore(
            *command_queue_program_ptr, remote_processor_location, num_eth_command_slots, dispatch_core_type);
        // second semaphore is between processor and dispatcher to detect whether dispatcher can accept commands
        tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, remote_processor_location, 0, dispatch_core_type);
        tt::tt_metal::CreateSemaphore(
            *command_queue_program_ptr,
            remote_processor_location,
            num_tensix_command_slots,
            dispatch_core_type);  // semaphore between push&pull kernel and dispatch kernel

        std::vector<uint32_t> dispatch_compile_args = {cmd_start_consumer, consumer_data_buffer_size};

        std::map<string, string> remote_dispatch_defines = {
            {"DISPATCH_KERNEL", "1"},
            {"PRODUCER_NOC_X", std::to_string(remote_processor_physical_core.x)},
            {"PRODUCER_NOC_Y", std::to_string(remote_processor_physical_core.y)},
        };

        if (dispatch_core_type != CoreType::ETH) {
            tt::tt_metal::CreateKernel(
                *command_queue_program_ptr,
                "tt_metal/impl/dispatch/kernels/cq_dispatcher.cpp",
                dispatch_location,
                tt::tt_metal::DataMovementConfig{
                    .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt::tt_metal::NOC::RISCV_0_default,
                    .compile_args = dispatch_compile_args,
                    .defines = remote_dispatch_defines});
        } else {
            tt::tt_metal::CreateKernel(
                *command_queue_program_ptr,
                "tt_metal/impl/dispatch/kernels/cq_dispatcher.cpp",
                dispatch_location,
                tt::tt_metal::EthernetConfig{
                    .eth_mode = Eth::IDLE,
                    .noc = tt::tt_metal::NOC::RISCV_0_default,
                    .compile_args = dispatch_compile_args,
                    .defines = remote_dispatch_defines});
        }

        tt::tt_metal::CreateSemaphore(*command_queue_program_ptr, dispatch_location, 0, dispatch_core_type);
    }
    detail::CompileProgram(this, *command_queue_program_ptr);
    this->command_queue_programs.push_back(std::move(command_queue_program_ptr));
}

// Writes issue and completion queue pointers to device and in sysmem and loads fast dispatch program onto dispatch cores
void Device::configure_command_queue_programs() {
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->id());

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
            uint8_t curr_num_hw_cqs = device_id == this->id() ? this->num_hw_cqs() : 1;
            uint16_t curr_channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
            uint32_t curr_cq_size = this->sysmem_manager().get_cq_size();

            for (uint8_t cq_id = 0; cq_id < curr_num_hw_cqs; cq_id++) {
                tt_cxy_pair issue_q_reader_location = dispatch_core_manager::get(curr_num_hw_cqs).issue_queue_reader_core(device_id, curr_channel, cq_id);
                tt_cxy_pair completion_q_writer_location = dispatch_core_manager::get(curr_num_hw_cqs).completion_queue_writer_core(device_id, curr_channel, cq_id);
                CoreType dispatch_core_type = dispatch_core_manager::get(curr_num_hw_cqs).get_dispatch_core_type(device_id);

                TT_ASSERT(issue_q_reader_location.chip == this->id() and completion_q_writer_location.chip == this->id(),
                    "Issue queue interface is on device {} and completion queue interface is on device {} but they are expected to be on device {}", issue_q_reader_location.chip, completion_q_writer_location.chip, this->id());

                // Re-start the pointers
                uint32_t issue_queue_start_addr = CQ_START + get_absolute_cq_offset(curr_channel, cq_id, curr_cq_size);
                uint32_t issue_queue_size = tt::round_up((cq_size - CQ_START) * SystemMemoryCQInterface::default_issue_queue_split, 32);
                uint32_t issue_queue_start_addr_16B = issue_queue_start_addr >> 4;
                vector<uint32_t> issue_queue_read_ptr = {issue_queue_start_addr_16B};
                detail::WriteToDeviceL1(this, issue_q_reader_location, CQ_ISSUE_READ_PTR, issue_queue_read_ptr, dispatch_core_type);
                detail::WriteToDeviceL1(this, issue_q_reader_location, CQ_ISSUE_WRITE_PTR, issue_queue_read_ptr, dispatch_core_type);

                uint32_t completion_queue_start_addr = CQ_START + issue_queue_size + get_absolute_cq_offset(curr_channel, cq_id, curr_cq_size);
                uint32_t completion_queue_start_addr_16B = completion_queue_start_addr >> 4;
                vector<uint32_t> completion_queue_wr_ptr = {completion_queue_start_addr_16B};
                vector<uint32_t> completion_queue_last_event = {0x0}; // Reset state in case L1 Clear is disabled.
                detail::WriteToDeviceL1(this, completion_q_writer_location, CQ_COMPLETION_READ_PTR, completion_queue_wr_ptr, dispatch_core_type);
                detail::WriteToDeviceL1(this, completion_q_writer_location, CQ_COMPLETION_WRITE_PTR, completion_queue_wr_ptr, dispatch_core_type);
                detail::WriteToDeviceL1(this, completion_q_writer_location, CQ_COMPLETION_LAST_EVENT, completion_queue_last_event, dispatch_core_type);
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
    if (tt::Cluster::instance().arch() == tt::ARCH::GRAYSKULL) {
        this->compile_command_queue_programs_for_grayskull();
    } else {
        this->compile_command_queue_programs();
    }
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
    tt::Cluster::instance().l1_barrier(this->id());
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

bool Device::initialize(const std::vector<uint32_t>& l1_bank_remap) {
    ZoneScoped;
    log_info(tt::LogMetal, "Initializing device {}", this->id_);
    bool already_initialized = this->active_devices_.activate_device(this->id_);
    this->initialize_cluster();
    this->initialize_allocator(l1_bank_remap);
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
    if (this->worker_queue_mode == WorkerQueueMode::ASYNCHRONOUS) {
        stop_worker();
    }
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

int32_t Device::l1_bank_offset_from_bank_id(uint32_t bank_id) const {
    this->check_allocator_is_initialized();
    return allocator::l1_bank_offset_from_bank_id(*this->allocator_, bank_id);
}

int32_t Device::dram_bank_offset_from_bank_id(uint32_t bank_id) const {
    this->check_allocator_is_initialized();
    return allocator::dram_bank_offset_from_bank_id(*this->allocator_, bank_id);
}

CoreCoord Device::logical_core_from_bank_id(uint32_t bank_id) const {
    this->check_allocator_is_initialized();
    return allocator::logical_core_from_bank_id(*this->allocator_, bank_id);
}

const std::vector<uint32_t> &Device::bank_ids_from_dram_channel(uint32_t dram_channel) const {
    this->check_allocator_is_initialized();
    return allocator::bank_ids_from_dram_channel(*this->allocator_, dram_channel);
}

const std::vector<uint32_t> &Device::bank_ids_from_logical_core(const CoreCoord &logical_core) const {
    this->check_allocator_is_initialized();
    return allocator::bank_ids_from_logical_core(*this->allocator_, logical_core);
}

allocator::Statistics Device::get_memory_allocation_statistics(const BufferType &buffer_type) const {
    this->check_allocator_is_initialized();
    return allocator::get_statistics(*this->allocator_, buffer_type);
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

CommandQueue& Device::command_queue(size_t cq_id) {
    detail::DispatchStateCheck(using_fast_dispatch);
    TT_ASSERT( cq_id < sw_command_queues_.size(), "cq_id {} is out of range", cq_id );
    TT_FATAL(this->is_initialized(), "Device has not been initialized, did you forget to call InitializeDevice?");
    return *sw_command_queues_[cq_id];
}

void Device::push_work(std::function<void()> work_executor, bool blocking) {
    if (this->worker_queue_mode == WorkerQueueMode::ASYNCHRONOUS) {
        if (std::hash<std::thread::id>{}(std::this_thread::get_id()) == worker_queue.parent_thread_id.load()) {
            // Push function executor to worker queue
            this->worker_queue.push(work_executor);
            if (blocking) {
                this->synchronize();
            }
        } else {
            TT_ASSERT(std::hash<std::thread::id>{}(std::this_thread::get_id()) == worker_queue.worker_thread_id.load(), "Only main thread or worker thread can push to device worker queue.");
            work_executor();
        }
    } else {
        // Synchronous execution: Run function right away.
        work_executor();
    }
}

void Device::start_worker() {
    this->worker_state = WorkerState::RUNNING;
    this->worker_thread = std::thread(&Device::run_worker, this);
}

void Device::run_worker() {
    worker_queue.worker_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    while (true) {
        if(this->worker_queue.empty()) {
            if (this->worker_state == WorkerState::TERMINATE) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        else {
            auto func = this->worker_queue.pop();
            (*func)();
        }
    }
}

void Device::stop_worker() {
    if (this->worker_state == WorkerState::IDLE) {
        return;
    }
    this->worker_state = WorkerState::TERMINATE;
    this->worker_thread.join();
    this->worker_state = WorkerState::IDLE;
}

void Device::synchronize() {
    if (this->worker_queue_mode == WorkerQueueMode::ASYNCHRONOUS) {
        // Blocking = wait for queue flushed
        this->worker_queue.push([](){}); // Send flush command (i.e. empty function)
        // Wait for queue empty, i.e. flush command picked up
        while(not this->worker_queue.empty()) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        };
    }
}

void Device::set_worker_mode(const WorkerQueueMode& mode) {
    if (this->worker_queue_mode == mode) {
        return;
    }
    this->worker_queue_mode = mode;
    if (this->worker_queue_mode == WorkerQueueMode::ASYNCHRONOUS) {
        this->worker_queue.parent_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        this->start_worker();
    } else if (this->worker_queue_mode == WorkerQueueMode::SYNCHRONOUS) {
        this->synchronize();
        this->stop_worker();
    }
}

void Device::enable_async(bool enable) {
    auto mode = enable ? WorkerQueueMode::ASYNCHRONOUS : WorkerQueueMode::SYNCHRONOUS;
    this->set_worker_mode(mode);
}

bool Device::using_slow_dispatch() const {
    return not (this->using_fast_dispatch);
}
}  // namespace tt_metal

}  // namespace tt
