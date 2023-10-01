// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_cluster.hpp"

#include <immintrin.h>

#include <iomanip>
#include <iostream>
#include <string>

#include "device_data.hpp"
#include "hostdevcommon/dprint_common.h"
#include "rtoptions.hpp"
#include "third_party/umd/device/tt_silicon_driver_common.hpp"
#include "tools/profiler/profiler.hpp"
#include "tt_metal/third_party/umd/device/util.hpp"
#include "watcher.hpp"

#ifdef ARCH_GRAYSKULL
static constexpr uint32_t DYNAMIC_TLB_COUNT = 16;
static constexpr unsigned int MEM_SMALL_READ_WRITE_TLB = DEVICE_DATA.TLB_BASE_INDEX_2M + 1;
static constexpr unsigned int DYNAMIC_TLB_BASE_INDEX = DEVICE_DATA.MEM_LARGE_READ_TLB + 1;

#else
static constexpr uint32_t DYNAMIC_TLB_COUNT = 16;
static constexpr unsigned int MEM_SMALL_READ_WRITE_TLB = DEVICE_DATA.TLB_BASE_INDEX_2M + 1;
static constexpr uint32_t DYNAMIC_TLB_BASE_INDEX = DEVICE_DATA.MEM_LARGE_READ_TLB + 1;
#endif

namespace tt {

Cluster &Cluster::instance() {
    static Cluster inst;
    return inst;
}

Cluster::Cluster() {
    ZoneScoped;
    log_info(tt::LogDevice, "Opening device driver");

#ifdef TT_METAL_VERSIM_DISABLED
    this->target_type_ = TargetDevice::Silicon;
    std::vector<chip_id_t> physical_mmio_device_ids = tt_SiliconDevice::detect_available_device_ids(true, false);
    this->arch_ = detect_arch(physical_mmio_device_ids.at(0));
    for (int dev_index = 1; dev_index < physical_mmio_device_ids.size(); dev_index++) {
        chip_id_t device_id = physical_mmio_device_ids.at(dev_index);
        tt::ARCH detected_arch = detect_arch(device_id);
        TT_FATAL(
            this->arch_ == detected_arch,
            "Expected all devices to be {} but device {} is {}",
            get_arch_str(this->arch_),
            device_id,
            get_arch_str(detected_arch));
    }
    const std::string sdesc_file = get_soc_description_file(this->arch_, this->target_type_);
    this->cluster_desc_path_ = (this->arch_ == tt::ARCH::WORMHOLE_B0) ? GetClusterDescYAML().string() : "";
#else
    this->target_type_ = TargetDevice::Versim;
    auto arch_env = getenv("ARCH_NAME");
    TT_FATAL(arch_env, "arch_env needs to be set for versim (ARCH_NAME=)");
    this->arch_ = tt::get_arch_from_string(arch_env);
    const std::string sdesc_file = get_soc_description_file(this->arch_, this->target_type_);
    this->cluster_desc_path_ = "";
#endif

    if (this->arch_ == tt::ARCH::GRAYSKULL) {
        // Cannot use tt_SiliconDevice::detect_available_device_ids because that returns physical device IDs
        std::set<chip_id_t> logical_mmio_device_ids;
        for (chip_id_t logical_mmio_device_id = 0; logical_mmio_device_id < physical_mmio_device_ids.size(); logical_mmio_device_id++) {
            logical_mmio_device_ids.insert(logical_mmio_device_id);
        }
        this->cluster_desc_ = tt_ClusterDescriptor::create_for_grayskull_cluster(logical_mmio_device_ids);
    } else {
        this->cluster_desc_ = tt_ClusterDescriptor::create_from_yaml(this->cluster_desc_path_);
    }

    // Map MMIO device id to all devices on the same card (including the MMIO device)
    if (this->target_type_ == TargetDevice::Versim) {
        std::set<chip_id_t> dummy_versim_card = {0};
        this->devices_grouped_by_assoc_mmio_device_[0] = dummy_versim_card;
        this->device_to_mmio_device_[0] = 0;
    } else {
        for (chip_id_t device_id : this->cluster_desc_->get_all_chips()) {
            chip_id_t closest_mmio_device_id = this->cluster_desc_->get_closest_mmio_capable_chip(device_id);
            std::set<chip_id_t> &device_ids = this->devices_grouped_by_assoc_mmio_device_[closest_mmio_device_id];
            device_ids.insert(device_id);
            this->device_to_mmio_device_[device_id] = closest_mmio_device_id;
        }
    }
}

void Cluster::initialize_device_driver(chip_id_t device_id) {
    chip_id_t assoc_mmio_device_id = this->device_to_mmio_device_.at(device_id);
    if (this->mmio_device_id_to_driver_.count(assoc_mmio_device_id) and this->mmio_device_id_to_driver_.at(assoc_mmio_device_id) != nullptr) {
        TT_FATAL(this->target_device_ids_.find(device_id) != this->target_device_ids_.end(), "Expected UMD containing device {} to be initialized with group for MMIO device {}!", device_id, assoc_mmio_device_id);
        // Already initialized UMD that includes the current device
        return;
    }

    this->open_device(device_id);

    tt_device_params default_params;
    if (getenv("TT_METAL_VERSIM_DUMP_CORES")) {
        std::string dump_cores_string = getenv("TT_METAL_VERSIM_DUMP_CORES");
        default_params.vcd_dump_cores = tt::utils::strsplit(dump_cores_string, ',');
    }
    this->start_device(device_id, default_params);
}

void Cluster::get_metal_desc_from_tt_desc(
    const std::unordered_map<chip_id_t, tt_SocDescriptor> &input,
    const std::unordered_map<chip_id_t, uint32_t> &per_chip_id_harvesting_masks) {
    for (const auto it : input) {
        chip_id_t id = it.first;
        this->sdesc_per_chip_.emplace(id, metal_SocDescriptor(it.second, per_chip_id_harvesting_masks.at(id)));
    }
}

void Cluster::open_device(chip_id_t device_id, const bool &skip_driver_allocs) {
#ifdef ARCH_GRAYSKULL
    TT_FATAL(
        this->arch_ == tt::ARCH::GRAYSKULL,
        "Arch={} doesn't match compile-time build for GRAYSKULL",
        get_string(this->arch_));
#endif
#ifdef ARCH_WORMHOLE
    TT_FATAL(
        (this->arch_ == tt::ARCH::WORMHOLE_B0) || (this->arch_ == tt::ARCH::WORMHOLE),
        "Arch={} doesn't match compile-time build for WORMHOLE",
        get_string(this->arch_));
#endif
    TT_FATAL(this->target_type_ == TargetDevice::Versim or this->target_type_ == TargetDevice::Silicon);
    if (this->target_type_ == TargetDevice::Versim and device_id != 0) {
        TT_FATAL("Versim can only target device 0");
    }

    chip_id_t assoc_mmio_device_id = this->device_to_mmio_device_.at(device_id);
    std::set<chip_id_t> device_ids = this->devices_grouped_by_assoc_mmio_device_.at(assoc_mmio_device_id);
    this->target_device_ids_.insert(device_ids.begin(), device_ids.end());

    const std::string sdesc_path = get_soc_description_file(this->arch_, this->target_type_);

    std::unique_ptr<tt_device> device_driver;
    if (this->target_type_ == TargetDevice::Silicon) {
        // This is the target/desired number of mem channels per arch/device. Silicon driver will attempt to open
        // this many hugepages as channels, and assert if workload uses more than available.
        uint32_t num_host_mem_ch_per_mmio_device = 1;
        std::unordered_map<std::string, std::int32_t> dynamic_tlb_config = {};
        dynamic_tlb_config["REG_TLB"] = DEVICE_DATA.REG_TLB;
        // This will remove harvested rows from the soc descriptor
        const bool perform_harvesting = true;

        device_driver = std::make_unique<tt_SiliconDevice>(
            sdesc_path,
            this->cluster_desc_path_,
            device_ids,
            num_host_mem_ch_per_mmio_device,
            dynamic_tlb_config,
            skip_driver_allocs,
            perform_harvesting);

        device_driver->clean_system_resources();
        device_driver->set_driver_host_address_params(host_address_params);
        device_driver->set_driver_eth_interface_params(eth_interface_params);

        // Adding this check is a workaround for current UMD bug that only uses this getter to populate private metadata that is later expected to be populated by unrelated APIs
        TT_FATAL(device_driver->get_target_mmio_device_ids().size() == 1);
    } else if (this->target_type_ == TargetDevice::Versim) {
        device_driver = std::make_unique<tt_VersimDevice>(sdesc_path, this->cluster_desc_path_);
    }
    device_driver->set_device_dram_address_params(dram_address_params);
    device_driver->set_device_l1_address_params(l1_address_params);

    this->get_metal_desc_from_tt_desc(device_driver->get_virtual_soc_descriptors(), device_driver->get_harvesting_masks_for_soc_descriptors());
    this->mmio_device_id_to_driver_[assoc_mmio_device_id] = std::move(device_driver);
}

#ifdef ARCH_WORMHOLE
std::int32_t get_static_tlb_index(CoreCoord target) {
    bool is_eth_location =
        std::find(std::cbegin(DEVICE_DATA.ETH_LOCATIONS), std::cend(DEVICE_DATA.ETH_LOCATIONS), target) !=
        std::cend(DEVICE_DATA.ETH_LOCATIONS);
    bool is_tensix_location =
        std::find(std::cbegin(DEVICE_DATA.T6_X_LOCATIONS), std::cend(DEVICE_DATA.T6_X_LOCATIONS), target.x) !=
            std::cend(DEVICE_DATA.T6_X_LOCATIONS) &&
        std::find(std::cbegin(DEVICE_DATA.T6_Y_LOCATIONS), std::cend(DEVICE_DATA.T6_Y_LOCATIONS), target.y) !=
            std::cend(DEVICE_DATA.T6_Y_LOCATIONS);
    // implementation migrated from wormhole.py in `src/t6ifc/t6py/packages/tenstorrent/chip/wormhole.py` from tensix
    // repo (t6py-wormhole-bringup branch)

    // Special handling for DRAM TLBs : return a 2MB TLB pointing to the start of the Epoch Cmd Queue Table
    // The default 1MB TLB is not used for DRAM cores
    // auto DRAM_TLB_IDX = std::find(DEVICE_DATA.DRAM_LOCATIONS.begin(), DEVICE_DATA.DRAM_LOCATIONS.end(), target);
    // if (DRAM_TLB_IDX != DEVICE_DATA.DRAM_LOCATIONS.end()) {
    //     return EPOCH_CMD_QUEUE_TLBS.at(DRAM_TLB_IDX - DEVICE_DATA.DRAM_LOCATIONS.begin());
    // }

    if (is_eth_location) {
        if (target.y == 6) {
            target.y = 1;
        }

        if (target.x >= 5) {
            target.x -= 1;
        }
        target.x -= 1;

        int flat_index = target.y * 8 + target.x;
        int tlb_index = flat_index;
        return tlb_index;

    } else if (is_tensix_location) {
        if (target.x >= 5) {
            target.x -= 1;
        }
        target.x -= 1;

        if (target.y >= 6) {
            target.y -= 1;
        }
        target.y -= 1;

        int flat_index = target.y * 8 + target.x;

        // All 80 get single 1MB TLB.
        int tlb_index = DEVICE_DATA.ETH_LOCATIONS.size() + flat_index;

        return tlb_index;
    } else {
        return -1;
    }
}
#endif

#ifdef ARCH_GRAYSKULL
std::int32_t get_static_tlb_index(CoreCoord target) {
    // Special handling for DRAM TLBs : return a 2MB TLB pointing to the start of the Epoch Cmd Queue Table
    // The default 1MB TLB is not used for DRAM cores
    // auto DRAM_TLB_IDX = std::find(DEVICE_DATA.DRAM_LOCATIONS.begin(), DEVICE_DATA.DRAM_LOCATIONS.end(), target);
    // if (DRAM_TLB_IDX != DEVICE_DATA.DRAM_LOCATIONS.end()) {
    //     return EPOCH_CMD_QUEUE_TLBS.at(DRAM_TLB_IDX - DEVICE_DATA.DRAM_LOCATIONS.begin());
    // }
    int flat_index = target.y * DEVICE_DATA.GRID_SIZE_X + target.x;
    if (flat_index == 0) {
        return -1;
    }
    return flat_index;
}
#endif

void Cluster::configure_static_tlbs(chip_id_t mmio_device_id) {
    auto sdesc = get_soc_desc(mmio_device_id);
    auto statically_mapped_cores = sdesc.workers;
    statically_mapped_cores.insert(
        statically_mapped_cores.end(), sdesc.ethernet_cores.begin(), sdesc.ethernet_cores.end());
    std::int32_t address = 0;

    // Setup static TLBs for all worker cores
    for (auto &core : statically_mapped_cores) {
        auto tlb_index = get_static_tlb_index(core);
        this->get_driver(mmio_device_id).configure_tlb(mmio_device_id, core, tlb_index, address);
    }
    // Setup static TLBs for MMIO mapped data space
    uint64_t peer_dram_offset = DEVICE_DATA.DRAM_CHANNEL_0_PEER2PEER_REGION_START;
    for (uint32_t tlb_id = DYNAMIC_TLB_BASE_INDEX; tlb_id < DYNAMIC_TLB_BASE_INDEX + DYNAMIC_TLB_COUNT; tlb_id++) {
        this->get_driver(mmio_device_id).configure_tlb(
            mmio_device_id, CoreCoord(DEVICE_DATA.DRAM_CHANNEL_0_X, DEVICE_DATA.DRAM_CHANNEL_0_Y), tlb_id, peer_dram_offset);
        // Align address space of 16MB TLB to 16MB boundary
        peer_dram_offset += DEVICE_DATA.DYNAMIC_TLB_16M_SIZE;
    }
    this->get_driver(mmio_device_id).setup_core_to_tlb_map([](CoreCoord core) { return get_static_tlb_index(core); });
}

void Cluster::start_device(chip_id_t device_id, tt_device_params &device_params) {
    chip_id_t mmio_device_id = this->device_to_mmio_device_.at(device_id);
    device_params.init_device = true;

    TT_FATAL(this->sdesc_per_chip_.size(), "Descriptor must be loaded. Try open_device()");

    if (this->target_type_ == TargetDevice::Silicon && device_params.init_device) {
        configure_static_tlbs(mmio_device_id);
    }

    this->mmio_device_id_to_driver_.at(mmio_device_id)->start_device(device_params);
}

void Cluster::close_device_driver(chip_id_t device_id) {
    log_info(tt::LogDevice, "Closing device driver");

    chip_id_t mmio_device_id = this->device_to_mmio_device_.at(device_id);
    bool is_mmio_device = (device_id == mmio_device_id);

    // There is one device driver per MMIO device.
    // Driver needs to remain open if any remote device is still open
    if (is_mmio_device) {
        bool all_devices_on_card_closed = true;
        for (const chip_id_t &device_id_on_card : this->devices_grouped_by_assoc_mmio_device_.at(mmio_device_id)) {
            if (device_id_on_card == mmio_device_id) { continue; }
            if (this->target_device_ids_.find(device_id_on_card) != this->target_device_ids_.end()) {
                all_devices_on_card_closed = false;
                break;
            }
        }

        if (all_devices_on_card_closed) {
            this->get_driver(mmio_device_id).close_device();
            this->mmio_device_id_to_driver_.at(mmio_device_id).reset();
        }
    }

    // For both MMIO and remote devices we remove it from sdesc map and target device IDs collection to indicate that device has been closed
    this->sdesc_per_chip_.erase(device_id);
    this->target_device_ids_.erase(device_id);
}

Cluster::~Cluster() {
    for (chip_id_t device_id : this->target_device_ids_) {
        this->close_device_driver(device_id);
    }
    this->sdesc_per_chip_.clear();
}

tt_device &Cluster::get_driver(chip_id_t device_id) const {
    if (this->target_device_ids_.find(device_id) == this->target_device_ids_.end()) {
        TT_FATAL("Cannot access driver for device ID {} before it is initialized! Call initialize_device_driver({}) first", device_id, device_id);
    }
    chip_id_t mmio_device_id = this->device_to_mmio_device_.at(device_id);
    return *(this->mmio_device_id_to_driver_.at(mmio_device_id));
}

const metal_SocDescriptor &Cluster::get_soc_desc(chip_id_t chip) const {
    if (this->sdesc_per_chip_.find(chip) == this->sdesc_per_chip_.end()) {
        TT_FATAL("Cannot access soc descriptor for {} before device driver is initialized! Call initialize_device_driver({}) first", chip, chip);
    }
    return this->sdesc_per_chip_.at(chip);
}

uint32_t Cluster::get_harvested_rows(chip_id_t chip) const {
    if (this->target_type_ == TargetDevice::Versim) {
        return 0;
    } else {
        return this->get_driver(chip).harvested_rows_per_target.at(chip);
    }
}

// clean up bad system resource state that may be carried over
void Cluster::clean_system_resources(chip_id_t device_id) const {
    this->get_driver(device_id).clean_system_resources();
}

void Cluster::verify_eth_fw() const {
    for (const chip_id_t &chip : this->target_device_ids_) {
        std::vector<uint32_t> fw_versions;
        for (const CoreCoord &eth_core : get_soc_desc(chip).ethernet_cores) {
            uint32_t val;
            read_core(&val, sizeof(uint32_t), tt_cxy_pair(chip, eth_core), eth_l1_mem::address_map::FW_VERSION_ADDR);
            fw_versions.push_back(val);
        }
        verify_sw_fw_versions(chip, SW_VERSION, fw_versions);
    }
}

int Cluster::get_device_aiclk(const chip_id_t &chip_id) const {
    if (this->target_device_ids_.find(chip_id) != this->target_device_ids_.end()) {
        // get_clocks returns MMIO device ID -> clock frequency
        // There is one driver per MMIO device, so we use that to index returned map
        chip_id_t mmio_device_id = this->device_to_mmio_device_.at(chip_id);
        return this->get_driver(chip_id).get_clocks().at(mmio_device_id);
    }
    TT_THROW("Cannot get frequency for device {} that is not initialized!", chip_id);
    return 0;
}

void Cluster::reset_debug_print_server_buffers() const {
    for (const int device_id : this->target_device_ids_) {
        auto workers = get_soc_desc(device_id).workers;
        for (const CoreCoord &core : workers)
            for (int hart_id = 0; hart_id < 5; hart_id++) {  // TODO(AP): must match DPRINT_NHARTS, magic
                // compute the buffer address for the requested hart
                uint32_t base_addr = PRINT_BUFFER_NC + hart_id * PRINT_BUFFER_SIZE;

                // The way this works is we first initialize all dprint buffers
                // for all cores and all harts to this magic number.
                // This way the device DPRINT will know that this core hasn't been initialized
                // from tt_start_debug_print_server
                // If we didn't have this mechanism them DPRINT in the kernel would have no way of knowing
                // Whether the host is polling it's buffer and flushing it.
                // If kernel code (in debug_print.h) detects that this magic value is present
                // Then it simply skips the print entirely. It prevents the kernel code
                // from hanging in a stall waiting for the host to flush the buffer
                // and removes the requirement that the host must listen on device's buffer.
                vector<uint32_t> initbuf = {uint32_t(DEBUG_PRINT_SERVER_DISABLED_MAGIC)};
                write_core(initbuf.data(), initbuf.size() * sizeof(uint32_t), {uint32_t(device_id), core}, base_addr);
            }
    }
}

void Cluster::assert_risc_reset(const chip_id_t &chip) const {  this->get_driver(chip).assert_risc_reset(chip); }

void Cluster::deassert_risc_reset_at_core(const tt_cxy_pair &physical_chip_coord) const {
    const metal_SocDescriptor &soc_desc = this->get_soc_desc(physical_chip_coord.chip);
    tt_cxy_pair virtual_chip_coord = soc_desc.convert_to_umd_coordinates(physical_chip_coord);
    this->get_driver(virtual_chip_coord.chip).deassert_risc_reset_at_core(virtual_chip_coord);
}

void Cluster::deassert_risc_reset(const chip_id_t &target_device_id, bool start_stagger) const {
    if (this->target_type_ == TargetDevice::Versim) {
        // Not running silicon multichip test
        this->get_driver(target_device_id).deassert_risc_reset(*this->target_device_ids_.begin());
    } else if (this->target_type_ == TargetDevice::Silicon) {
        log_debug(tt::LogLLRuntime, "Stagger start : {}", start_stagger);
        TT_ASSERT(not start_stagger, "UMD currently does not support staggered deassert of RISC reset");
        this->get_driver(target_device_id).deassert_risc_reset(target_device_id);
    }
}

inline uint64_t get_sys_addr(uint32_t chip_x, uint32_t chip_y, uint32_t noc_x, uint32_t noc_y, uint64_t offset) {
    uint64_t result = chip_y;
    uint64_t noc_addr_local_bits_mask = (1UL << NOC_ADDR_LOCAL_BITS) - 1;
    result <<= NOC_ADDR_NODE_ID_BITS;
    result |= chip_x;
    result <<= NOC_ADDR_NODE_ID_BITS;
    result |= noc_y;
    result <<= NOC_ADDR_NODE_ID_BITS;
    result |= noc_x;
    result <<= NOC_ADDR_LOCAL_BITS;
    result |= (noc_addr_local_bits_mask & offset);
    return result;
}

void Cluster::write_dram_vec(vector<uint32_t> &vec, tt_target_dram dram, uint64_t addr, bool small_access) const {
    int chip_id, d_chan, d_subchannel;
    std::tie(chip_id, d_chan, d_subchannel) = dram;
    const metal_SocDescriptor &desc_to_use = get_soc_desc(chip_id);
    TT_ASSERT(
        d_chan < desc_to_use.dram_cores.size(),
        "Bounds-Error -- dram_channel={} is outside of num_dram_channels={}",
        d_chan,
        desc_to_use.dram_cores.size());
    TT_ASSERT(
        d_subchannel < desc_to_use.dram_cores.at(d_chan).size(),
        "Trying to address dram sub channel that doesnt exist in the device descriptor");
    tt_cxy_pair dram_core = tt_cxy_pair(chip_id, desc_to_use.get_core_for_dram_channel(d_chan, d_subchannel));
    size_t offset = desc_to_use.get_address_offset(d_chan);
        write_core(vec.data(), vec.size() * sizeof(uint32_t), dram_core, addr + offset, small_access);
}

void Cluster::read_dram_vec(
    vector<uint32_t> &vec, uint32_t sz_in_bytes, tt_target_dram dram, uint64_t addr,  bool small_access) const {
    int chip_id, d_chan, d_subchannel;
    std::tie(chip_id, d_chan, d_subchannel) = dram;
    const metal_SocDescriptor &desc_to_use = get_soc_desc(chip_id);
    TT_ASSERT(
        d_chan < desc_to_use.dram_cores.size(),
        "Bounds-Error -- dram_channel={} is outside of num_dram_channels={}",
        d_chan,
        desc_to_use.dram_cores.size());
    TT_ASSERT(
        d_subchannel < desc_to_use.dram_cores.at(d_chan).size(),
        "Trying to address dram sub channel that doesnt exist in the device descriptor");
    tt_cxy_pair dram_core = tt_cxy_pair(chip_id, desc_to_use.get_core_for_dram_channel(d_chan, d_subchannel));
    size_t offset = desc_to_use.get_address_offset(d_chan);
    read_core(vec, sz_in_bytes, dram_core, addr + offset, small_access);
}

void Cluster::write_core(
    const void *mem_ptr, uint32_t sz_in_bytes, tt_cxy_pair core, uint64_t addr, bool small_access) const {
    chip_id_t chip_id = core.chip;
    const metal_SocDescriptor &soc_desc = this->get_soc_desc(chip_id);
    if (tt::llrt::OptionsG.get_watcher_enabled()) {
        tt::llrt::watcher_sanitize_host_noc_write(
            soc_desc, {core.x, core.y}, addr, sz_in_bytes);
    }
    tt_cxy_pair virtual_core = soc_desc.convert_to_umd_coordinates(core);
    this->get_driver(chip_id).write_to_device(mem_ptr, sz_in_bytes, virtual_core, addr, "LARGE_WRITE_TLB");
    if (this->get_driver(chip_id).get_target_remote_device_ids().find(virtual_core.chip) !=
        this->get_driver(chip_id).get_target_remote_device_ids().end()) {
            this->get_driver(chip_id).wait_for_non_mmio_flush();
        }
}

void Cluster::read_core(
    void *mem_ptr, uint32_t size_in_bytes, tt_cxy_pair core, uint64_t addr, bool small_access) const {
    int chip_id = core.chip;
    const metal_SocDescriptor &soc_desc = this->get_soc_desc(chip_id);

    if (tt::llrt::OptionsG.get_watcher_enabled()) {
        tt::llrt::watcher_sanitize_host_noc_read(
            soc_desc, {core.x, core.y}, addr, size_in_bytes);
    }

    tt_cxy_pair virtual_core = soc_desc.convert_to_umd_coordinates(core);
    this->get_driver(chip_id).read_from_device(mem_ptr, virtual_core, addr, size_in_bytes, "LARGE_READ_TLB");
}

void Cluster::read_core(
    vector<uint32_t>& data, uint32_t size_in_bytes, tt_cxy_pair core, uint64_t addr, bool small_access) const {
    data.resize(size_in_bytes / sizeof(uint32_t));
    read_core(data.data(), size_in_bytes, core, addr, small_access);
}

void Cluster::write_reg(const std::uint32_t *mem_ptr, tt_cxy_pair target, uint64_t addr) const {
    const unsigned int size_in_bytes = sizeof(uint32_t);
    int chip_id = target.chip;
    const metal_SocDescriptor &soc_desc = this->get_soc_desc(chip_id);

    if (tt::llrt::OptionsG.get_watcher_enabled()) {
        tt::llrt::watcher_sanitize_host_noc_write(soc_desc, {target.x, target.y}, addr, size_in_bytes);
    }
    tt_cxy_pair virtual_target = soc_desc.convert_to_umd_coordinates(target);
    this->get_driver(chip_id).write_to_device(mem_ptr, size_in_bytes, virtual_target, addr, "REG_TLB");
    if (this->get_driver(chip_id).get_target_remote_device_ids().find(virtual_target.chip) !=
        this->get_driver(chip_id).get_target_remote_device_ids().end()) {
        this->get_driver(chip_id).wait_for_non_mmio_flush();
    }
}

void Cluster::read_reg(std::uint32_t *mem_ptr, tt_cxy_pair target, uint64_t addr) const {
    const unsigned int size_in_bytes = sizeof(uint32_t);
    int chip_id = target.chip;
    const metal_SocDescriptor &soc_desc = this->get_soc_desc(chip_id);

    if (tt::llrt::OptionsG.get_watcher_enabled()) {
        tt::llrt::watcher_sanitize_host_noc_read(soc_desc, {target.x, target.y}, addr, size_in_bytes);
    }
    tt_cxy_pair virtual_target = soc_desc.convert_to_umd_coordinates(target);
    this->get_driver(chip_id).read_from_device(mem_ptr, virtual_target, addr, size_in_bytes, "REG_TLB");
}

void Cluster::write_sysmem(const void* vec, uint32_t size_in_bytes, uint64_t addr, chip_id_t src_device_id) const {
    constexpr uint16_t channel = 0;
    this->get_driver(src_device_id).write_to_sysmem(vec, size_in_bytes, addr, channel, src_device_id);
}

void Cluster::read_sysmem(void *vec, uint32_t size_in_bytes, uint64_t addr, chip_id_t src_device_id) const {
    // TODO: Uplift
    constexpr uint16_t channel = 0;
    this->get_driver(src_device_id).read_from_sysmem(vec, addr, channel, size_in_bytes, src_device_id);
}

void Cluster::verify_sw_fw_versions(
    int device_id, std::uint32_t sw_version, std::vector<std::uint32_t> &fw_versions) const {
    tt_version sw(sw_version), fw_first_eth_core(fw_versions.at(0));
    tt::log_info(
        tt::LogDevice,
        "Software version {}, Ethernet FW version {} (Device {})",
        sw.str(),
        fw_first_eth_core.str(),
        device_id);
    for (std::uint32_t &fw_version : fw_versions) {
        tt_version fw(fw_version);

        TT_FATAL(fw == fw_first_eth_core, "FW versions are not the same across different ethernet cores");
        TT_FATAL(sw.major == fw.major, "SW/FW major version number out of sync");
        TT_FATAL(sw.minor <= fw.minor, "SW version is newer than FW version");
    }
}

// DRAM barrier is used to implement host-to-device synchronization and should be used when all previous writes to DRAM
// need to be flushed This is needed because writes to device are not blocking unless strict TLB ordering is used
// (default ordering is posted) This barrier is intended to prevent races caused by out of order writes, specifically to
// ensure metadata and data to compute on are committed before launching kernels
void Cluster::dram_barrier(chip_id_t chip_id) const {
    std::unordered_set<uint32_t> dram_channels;
    for (uint32_t channel = 0; channel < this->get_soc_desc(chip_id).get_num_dram_channels(); channel++) {
        dram_channels.insert(channel);
    }
    this->get_driver(chip_id).dram_membar(chip_id, "LARGE_WRITE_TLB", dram_channels);
}

// L1 barrier is used to implement host-to-device synchronization and should be used when all previous writes to L1 need
// to be flushed This is needed because writes to device are not blocking unless strict TLB ordering is used (default
// ordering is posted) This barrier is intended to prevent races caused by out of order writes, specifically to ensure
// binaries, metadata, and data to compute on are committed before launching kernels
void Cluster::l1_barrier(chip_id_t chip_id) const {
    // Sets and resets L1 barrier of all tensix cores and ethernet cores
    this->get_driver(chip_id).l1_membar(chip_id, "LARGE_WRITE_TLB");
}

uint32_t Cluster::get_num_host_channels(chip_id_t device_id) const {
    bool mmio_capable = this->cluster_desc_->is_chip_mmio_capable(device_id);
    return mmio_capable ? this->get_driver(device_id).get_num_host_channels(device_id) : 0;
}

uint32_t Cluster::get_host_channel_size(chip_id_t device_id, uint32_t channel) const {
    TT_ASSERT(this->cluster_desc_->is_chip_mmio_capable(device_id));
    return this->get_driver(device_id).get_host_channel_size(device_id, channel);
}

void *Cluster::host_dma_address(uint64_t offset, chip_id_t src_device_id, uint16_t channel) const {
    TT_ASSERT(this->cluster_desc_->is_chip_mmio_capable(src_device_id));
    return this->get_driver(src_device_id).host_dma_address(offset, src_device_id, channel);
}

uint64_t Cluster::get_pcie_base_addr_from_device(chip_id_t chip_id) const {
    return this->get_driver(chip_id).get_pcie_base_addr_from_device();
}

// Ethernet cluster api
std::unordered_set<chip_id_t> Cluster::get_ethernet_connected_chip_ids(chip_id_t chip_id) const {
    std::unordered_set<chip_id_t> connected_chips;
    const auto &all_eth_connections = this->cluster_desc_->get_ethernet_connections();
    if (all_eth_connections.find(chip_id) == all_eth_connections.end()) {
        return {};
    }
    for (const auto &[eth_chan, connected_chip_chan] : all_eth_connections.at(chip_id)) {
        connected_chips.insert(std::get<0>(connected_chip_chan));
    }
    return connected_chips;
}

std::unordered_set<CoreCoord> Cluster::get_active_ethernet_cores(chip_id_t chip_id) const {
    std::unordered_set<CoreCoord> active_ethernet_cores;
    const auto &connected_chips = this->get_ethernet_connected_chip_ids(chip_id);
    for (auto &other_chip_id : connected_chips) {
        for (const auto &channel_pair :
             this->cluster_desc_->get_directly_connected_ethernet_channels_between_chips(chip_id, other_chip_id)) {
            ethernet_channel_t local_chip_chan = std::get<0>(channel_pair);
            active_ethernet_cores.insert(get_soc_desc(chip_id).chan_to_logical_eth_core_map.at(local_chip_chan));
        }
    }
    return active_ethernet_cores;
}

std::unordered_set<CoreCoord> Cluster::get_inactive_ethernet_cores(chip_id_t chip_id) const {
    std::unordered_set<CoreCoord> active_ethernet_cores = this->get_active_ethernet_cores(chip_id);
    std::unordered_set<CoreCoord> inactive_ethernet_cores;
    for (const auto &[eth_core, chan] : get_soc_desc(chip_id).logical_eth_core_to_chan_map) {
        if (active_ethernet_cores.find(eth_core) == active_ethernet_cores.end()) {
            inactive_ethernet_cores.insert(eth_core);
        }
    }
    return inactive_ethernet_cores;
}

std::tuple<chip_id_t, CoreCoord> Cluster::get_connected_ethernet_core(std::tuple<chip_id_t, CoreCoord> eth_core) const {
    const auto &soc_desc = get_soc_desc(std::get<0>(eth_core));
    ethernet_channel_t eth_chan = soc_desc.logical_eth_core_to_chan_map.at(std::get<1>(eth_core));
    TT_ASSERT(
        (this->cluster_desc_->ethernet_core_has_active_ethernet_link(std::get<0>(eth_core), eth_chan)),
        "Logical eth core {} is not an active eth core on chip {}.",
        std::get<1>(eth_core).str(),
        std::get<0>(eth_core));
    auto connected_eth_core =
        this->cluster_desc_->get_chip_and_channel_of_remote_ethernet_core(std::get<0>(eth_core), eth_chan);
    return std::make_tuple(
        std::get<0>(connected_eth_core), soc_desc.chan_to_logical_eth_core_map.at(std::get<1>(connected_eth_core)));
}

}  // namespace tt

std::ostream &operator<<(std::ostream &os, tt_target_dram const &dram) {
    os << "Target DRAM chip = " << std::get<0>(dram) << ", chan = " << std::get<1>(dram)
       << ", subchan = " << std::get<2>(dram);
    return os;
}
