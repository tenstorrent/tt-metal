// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "impl/debug/dprint_server.hpp"
#include "tt_metal/third_party/umd/device/util.hpp"

#include "common/utils.hpp"
#include "llrt/llrt.hpp"
#include "dev_msgs.h"

namespace tt {

namespace tt_metal {

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
        already_initialized = true;
    }
    this->active_devices_[id] = ActiveState::ACTIVE;

    return already_initialized;
}

void ActiveDevices::deactivate_device(chip_id_t id) {
    const std::lock_guard<std::mutex> lock(lock_);
    this->active_devices_[id] = ActiveState::INACTIVE;
}

Device::Device(chip_id_t device_id, const std::vector<uint32_t>& l1_bank_remap) : id_(device_id)
{
    ZoneScoped;
    this->initialize(l1_bank_remap);
}

size_t Device::detect_num_available_devices() {
#ifdef TT_METAL_VERSIM_DISABLED
    return tt::Cluster::instance().number_of_devices();
#else
    return 1;
#endif
}

size_t Device::detect_num_pci_devices() {
#ifdef TT_METAL_VERSIM_DISABLED
    return tt::Cluster::instance().number_of_pci_devices();
#else
    return 1;
#endif
}

void Device::initialize_cluster() {
    ZoneScoped;
    this->clear_l1_state();
#ifdef TT_METAL_VERSIM_DISABLED
    int ai_clk = tt::Cluster::instance().get_device_aiclk(this->id_);
    log_info(tt::LogMetal, "AI CLK for device {} is:   {} MHz", this->id_, ai_clk);
#endif
    tt::Cluster::instance().assert_risc_reset(this->id_);
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
        .l1_bank_size = static_cast<size_t>(soc_desc.l1_bank_size),
        .core_type_from_noc_coord_table = {}, // Populated later
        .worker_log_to_physical_routing_x=soc_desc.worker_log_to_physical_routing_x,
        .worker_log_to_physical_routing_y=soc_desc.worker_log_to_physical_routing_y,
        .l1_bank_remap = l1_bank_remap,
    });
    // Initialize dram_offsets from soc_descriptor
    for (auto channel = 0; channel < soc_desc.get_num_dram_channels(); channel++) {
        config.dram_bank_offsets.push_back(soc_desc.get_address_offset(channel));
    }
    // Initialize core_type_from_noc_coord_table table
    for (const auto& core: soc_desc.physical_cores) {
        config.core_type_from_noc_coord_table.insert({core.first, AllocCoreType::Invalid});
    }
    for (const auto& core : soc_desc.compute_with_storage_cores) {
        const auto logical_coord = get_core_coord_from_relative(core, this->logical_grid_size());
        this->compute_cores.insert(logical_coord);
        const auto noc_coord = this->worker_core_from_logical_core(logical_coord);
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::ComputeAndStore;
    }
    for (const auto& core : soc_desc.storage_cores) {
        const auto logical_coord = get_core_coord_from_relative(core, this->logical_grid_size());
        this->storage_only_cores_.insert(logical_coord);
        const auto noc_coord = this->worker_core_from_logical_core(logical_coord);
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::StorageOnly;
    }
    for (const auto& core : soc_desc.dispatch_cores) {
        const auto logical_coord = get_core_coord_from_relative(core, this->logical_grid_size());
        this->dispatch_cores_.insert(logical_coord);
        const auto noc_coord = this->worker_core_from_logical_core(logical_coord);
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::Dispatch;
    }
    for (const auto &core : soc_desc.get_logical_ethernet_cores()) {
        this->ethernet_cores_.insert(core);
    }

    // L1_BANKING scheme creates 1 bank per DRAM core and splits up L1 such that there are power 2 num L1 banks
    // This is the only allocator scheme supported because kernel APIs assume num L1 banks are power of 2
    static_assert(this->allocator_scheme_ == MemoryAllocator::L1_BANKING);
    this->allocator_ = std::make_unique<L1BankingAllocator>(config);
}

void Device::initialize_build() {
    ZoneScoped;
    build_kernel_for_riscv_options_t build_options(this->id());
    detail::GenerateDeviceHeaders(this, &build_options, "");
    tt::ARCH arch = this->arch();
    generate_binaries_params_t default_params;
    generate_binaries_all_riscs(&build_options,
                                "",
                                arch,
                                default_params);
}

void Device::initialize_firmware(CoreCoord phys_core, launch_msg_t *launch_msg) {
    ZoneScoped;
    for (int riscv_id = 0; riscv_id < 5; riscv_id++) {
        string fname;
        switch (riscv_id) {
        case 0:
            fname = "brisc/brisc.hex";
            llrt::program_brisc_startup_addr(this->id(), phys_core);
            break;
        case 1: fname = "ncrisc/ncrisc.hex"; break;
        case 2: fname = "tensix_thread0/tensix_thread0.hex"; break;
        case 3: fname = "tensix_thread1/tensix_thread1.hex"; break;
        case 4: fname = "tensix_thread2/tensix_thread2.hex"; break;
        }

        ll_api::memory binary_mem = llrt::get_risc_binary(fname, this->id(), true);
        uint32_t kernel_size16 = llrt::get_binary_code_size16(binary_mem, riscv_id);
        if (riscv_id == 1) {
            launch_msg->ncrisc_kernel_size16 = kernel_size16;
        }
        log_debug(LogDevice, "RISC {} fw binary size: {} in bytes", riscv_id, kernel_size16 * 16);
        llrt::test_load_write_read_risc_binary(binary_mem, this->id(), phys_core, riscv_id);
    }

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
        .run = RUN_MSG_INIT,
    };

    // Download to worker cores
    log_debug("Initializing firmware");
    CoreCoord grid_size = this->logical_grid_size();
    std::unordered_set<CoreCoord> not_done_cores;
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core = this->worker_core_from_logical_core(logical_core);

            if (this->storage_only_cores_.find(logical_core) == this->storage_only_cores_.end()) {
                this->initialize_firmware(worker_core, &launch_msg);
                not_done_cores.insert(worker_core);
            }
        }
    }

    // Barrier between L1 writes above and deassert below
    tt::Cluster::instance().l1_barrier(this->id());

    // Deassert worker cores
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core = this->worker_core_from_logical_core(logical_core);

            if (this->storage_only_cores_.find(logical_core) == this->storage_only_cores_.end()) {
                tt::Cluster::instance().deassert_risc_reset_at_core(tt_cxy_pair(this->id(), worker_core));
            }
        }
    }

    // Wait until fw init is done, ensures the next launch msg doesn't get
    // written while fw is still in init
    log_debug("Waiting for firmware init complete");
    llrt::internal_::wait_until_cores_done(this->id(), RUN_MSG_INIT, not_done_cores);
    log_debug("Firmware init complete");

    // Clear erisc sync info
    for (const auto &eth_core : this->ethernet_cores()) {
        CoreCoord physical_core = this->ethernet_core_from_logical_core(eth_core);
        std::vector<uint32_t> zero_vec(eth_l1_mem::address_map::ERISC_APP_SYNC_INFO_SIZE / sizeof(uint32_t), 0);

        llrt::write_hex_vec_to_core(
            this->id(), physical_core, zero_vec, eth_l1_mem::address_map::ERISC_APP_SYNC_INFO_BASE);
    }
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
}

bool Device::initialize(const std::vector<uint32_t>& l1_bank_remap) {
    ZoneScoped;
    log_info(tt::LogMetal, "Initializing device {}", this->id_);
    bool already_initialized = this->active_devices_.activate_device(this->id_);
    this->initialize_cluster();
    this->initialize_allocator(l1_bank_remap);
    if (!already_initialized) {
        this->initialize_build();
    }

    tt_start_debug_print_server(
        [&, this]() { return this->logical_grid_size(); },
        [&, this](CoreCoord core) { return this->worker_core_from_logical_core(core); }
    );
    llrt::watcher_init(this->id(),
                       [&, this]() { return this->logical_grid_size(); },
                       [&, this](CoreCoord core) { return this->worker_core_from_logical_core(core); }
                       );

    this->initialize_and_launch_firmware();

    llrt::watcher_attach(this, this->id(),
                         [&, this]() { return this->logical_grid_size(); },
                         [&, this](CoreCoord core) { return this->worker_core_from_logical_core(core); },
                         [&, this]() -> const std::set<CoreCoord>& { return this->storage_only_cores(); },
                         get_compile_outpath()
                         );

    this->initialized_ = true;
    return true;
}

bool Device::close() {
    log_info(tt::LogMetal, "Closing device {}", this->id_);
    if (not this->initialized_) {
        TT_THROW("Cannot close device {} that has not been initialized!", this->id_);
    }
    this->deallocate_buffers();
    llrt::watcher_detach(this);
    tt_stop_debug_print_server();
    tt::Cluster::instance().assert_risc_reset(id_);
    this->clear_l1_state();
    tt::Cluster::instance().l1_barrier(id_);
    allocator::clear(*this->allocator_);

    this->active_devices_.deactivate_device(this->id_);

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
    return tt::Cluster::instance().get_soc_desc(id_).compute_with_storage_grid_size;
}

CoreCoord Device::physical_core_from_logical_core(const CoreCoord &logical_coord, const CoreType &core_type) const {
    switch (core_type) {
        case CoreType::ETH: return this->ethernet_core_from_logical_core(logical_coord);
        case CoreType::WORKER: return this->worker_core_from_logical_core(logical_coord);
        default: TT_THROW("Undefined conversion for core type.");
    }
}

CoreCoord Device::worker_core_from_logical_core(const CoreCoord &logical_core) const {
    CoreCoord logical_grid_size = this->logical_grid_size();
    TT_ASSERT(
        (logical_core.x < logical_grid_size.x) and
        (logical_core.y < logical_grid_size.y),
        "Bounds-Error -- Logical_core={} is outside of logical_grid_size={}",
        logical_core.str(),
        logical_grid_size.str()
    );
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    CoreCoord worker_core({
            .x = static_cast<size_t>(soc_desc.worker_log_to_physical_routing_x.at(logical_core.x)),
            .y = static_cast<size_t>(soc_desc.worker_log_to_physical_routing_y.at(logical_core.y)),
    });
    return worker_core;
}

std::vector<CoreCoord> Device::worker_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const {
    std::vector<CoreCoord> worker_cores;
    for (auto logical_core : logical_cores) {
        worker_cores.push_back(worker_core_from_logical_core(logical_core));
    }
    return worker_cores;
}

CoreCoord Device::ethernet_core_from_logical_core(const CoreCoord &logical_core) const {
    const auto &eth_cores = tt::Cluster::instance().get_soc_desc(id_).get_physical_ethernet_cores();
    const auto &eth_chan_map = tt::Cluster::instance().get_soc_desc(id_).logical_eth_core_to_chan_map;
    TT_ASSERT(
        (eth_chan_map.find(logical_core) != eth_chan_map.end()),
        "Bounds-Error -- Logical_core={} is outside of ethernet logical grid",
        logical_core.str());
    return eth_cores.at(eth_chan_map.at(logical_core));
}

std::vector<CoreCoord> Device::ethernet_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const {
    std::vector<CoreCoord> ethernet_cores;
    for (auto logical_core : logical_cores) {
        ethernet_cores.emplace_back(ethernet_core_from_logical_core(logical_core));
    }
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

}  // namespace tt_metal

}  // namespace tt
