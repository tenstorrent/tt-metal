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
#include "tt_metal/impl/debug/sanitize_noc_host.hpp"

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

const Cluster &Cluster::instance() {
    static Cluster inst;
    return inst;
}

Cluster::Cluster() {
    ZoneScoped;
    log_info(tt::LogDevice, "Opening user mode device driver");

    this->detect_arch_and_target();

    this->generate_cluster_descriptor();

    this->initialize_device_drivers();

    this->reserve_ethernet_cores_for_tunneling();

    this->initialize_ethernet_sockets();

    this->assert_risc_reset();
}

void Cluster::detect_arch_and_target() {
#ifdef TT_METAL_VERSIM_DISABLED
    this->target_type_ = TargetDevice::Silicon;
    std::vector<chip_id_t> physical_mmio_device_ids = tt_SiliconDevice::detect_available_device_ids();
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
#else
    this->target_type_ = TargetDevice::Versim;
    auto arch_env = getenv("ARCH_NAME");
    TT_FATAL(arch_env, "arch_env needs to be set for versim (ARCH_NAME=)");
    this->arch_ = tt::get_arch_from_string(arch_env);
#endif

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
}

void Cluster::generate_cluster_descriptor() {
    this->cluster_desc_path_ = (this->target_type_ == TargetDevice::Silicon and this->arch_ == tt::ARCH::WORMHOLE_B0) ? GetClusterDescYAML().string() : "";

    if (this->arch_ == tt::ARCH::GRAYSKULL) {
        // Cannot use tt_SiliconDevice::detect_available_device_ids because that returns physical device IDs
        std::vector<chip_id_t> physical_mmio_device_ids = tt_SiliconDevice::detect_available_device_ids();
        std::set<chip_id_t> logical_mmio_device_ids;
        for (chip_id_t logical_mmio_device_id = 0; logical_mmio_device_id < physical_mmio_device_ids.size(); logical_mmio_device_id++) {
            logical_mmio_device_ids.insert(logical_mmio_device_id);
        }
        this->cluster_desc_ =
            tt_ClusterDescriptor::create_for_grayskull_cluster(logical_mmio_device_ids, physical_mmio_device_ids);
    } else {
        this->cluster_desc_ = tt_ClusterDescriptor::create_from_yaml(this->cluster_desc_path_);
    }

    // Use cluster descriptor to map MMIO device id to all devices on the same card (including the MMIO device)
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

    uint32_t total_num_hugepages = get_num_hugepages();
    TT_FATAL(total_num_hugepages >= this->cluster_desc_->get_all_chips().size(),
        "Machine setup error: Insufficient number of hugepages available, expected one per device ({}) but have {}. Increase number of hugepages!", this->cluster_desc_->get_all_chips().size(), total_num_hugepages);

}

void Cluster::initialize_device_drivers() {
    for (const auto &[mmio_device_id, controlled_devices] : this->devices_grouped_by_assoc_mmio_device_) {
        this->assign_mem_channels_to_devices(mmio_device_id, controlled_devices);

        this->open_driver(mmio_device_id, controlled_devices);

        tt_device_params default_params;
        if (getenv("TT_METAL_VERSIM_DUMP_CORES")) {
            std::string dump_cores_string = getenv("TT_METAL_VERSIM_DUMP_CORES");
            default_params.vcd_dump_cores = tt::utils::strsplit(dump_cores_string, ',');
        }
        this->start_driver(mmio_device_id, default_params);
    }
}

void Cluster::assert_risc_reset() {
    for (const auto &[mmio_device_id, controlled_devices] : this->devices_grouped_by_assoc_mmio_device_) {
        this->get_driver(mmio_device_id).assert_risc_reset();
    }
}

void Cluster::assign_mem_channels_to_devices(chip_id_t mmio_device_id, const std::set<chip_id_t> &controlled_device_ids) {
    // g_MAX_HOST_MEM_CHANNELS (4) is defined in tt_SiliconDevice and denotes the max number of host memory channels per MMIO device
    // Metal currently assigns 1 channel per device. See https://github.com/tenstorrent-metal/tt-metal/issues/4087
    TT_ASSERT(controlled_device_ids.size() <= 4, "Unable to assign each device to its own host memory channel!");
    uint16_t channel = 0;
    this->device_to_host_mem_channel_[mmio_device_id] = channel++;
    for (const chip_id_t &device_id : controlled_device_ids) {
        if (device_id == mmio_device_id) {
            continue;
        }
        this->device_to_host_mem_channel_[device_id] = channel++;
    }
}

void Cluster::get_metal_desc_from_tt_desc(
    const std::unordered_map<chip_id_t, tt_SocDescriptor> &input,
    const std::unordered_map<chip_id_t, uint32_t> &per_chip_id_harvesting_masks) {
    for (const auto it : input) {
        chip_id_t id = it.first;
        this->sdesc_per_chip_.emplace(id, metal_SocDescriptor(it.second, per_chip_id_harvesting_masks.at(id)));
    }
}

void Cluster::open_driver(chip_id_t mmio_device_id, const std::set<chip_id_t> &controlled_device_ids, const bool &skip_driver_allocs) {
    const std::string sdesc_path = get_soc_description_file(this->arch_, this->target_type_);

    std::unique_ptr<tt_device> device_driver;
    if (this->target_type_ == TargetDevice::Silicon) {
        // This is the target/desired number of mem channels per arch/device.
        // Silicon driver will attempt to open this many hugepages as channels, and assert if workload uses more than available.
        // Metal currently uses assigns 1 channel per device
        uint32_t num_host_mem_ch_per_mmio_device = controlled_device_ids.size();
        std::unordered_map<std::string, std::int32_t> dynamic_tlb_config = {};
        dynamic_tlb_config["REG_TLB"] = DEVICE_DATA.REG_TLB;
        // This will remove harvested rows from the soc descriptor
        const bool perform_harvesting = true;
        const bool clean_system_resources = true;
        device_driver = std::make_unique<tt_SiliconDevice>(
            sdesc_path,
            this->cluster_desc_path_,
            controlled_device_ids,
            num_host_mem_ch_per_mmio_device,
            dynamic_tlb_config,
            skip_driver_allocs,
            clean_system_resources,
            perform_harvesting);

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
    this->mmio_device_id_to_driver_[mmio_device_id] = std::move(device_driver);
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

void Cluster::configure_static_tlbs(chip_id_t mmio_device_id) const {
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

void Cluster::start_driver(chip_id_t mmio_device_id, tt_device_params &device_params) const {
    device_params.init_device = true;

    TT_FATAL(this->sdesc_per_chip_.size(), "Descriptor must be loaded. Try open_driver()");

    if (this->target_type_ == TargetDevice::Silicon && device_params.init_device) {
        configure_static_tlbs(mmio_device_id);
    }

    this->mmio_device_id_to_driver_.at(mmio_device_id)->start_device(device_params);
}

Cluster::~Cluster() {
    log_info(tt::LogDevice, "Closing user mode device drivers");

    for (const auto &[mmio_device_id, device_driver] : this->mmio_device_id_to_driver_) {
        device_driver->close_device();
    }

    this->mmio_device_id_to_driver_.clear();
    this->sdesc_per_chip_.clear();
}

tt_device &Cluster::get_driver(chip_id_t device_id) const {
    chip_id_t mmio_device_id = this->device_to_mmio_device_.at(device_id);
    return *(this->mmio_device_id_to_driver_.at(mmio_device_id));
}

const metal_SocDescriptor &Cluster::get_soc_desc(chip_id_t chip) const {
    if (this->sdesc_per_chip_.find(chip) == this->sdesc_per_chip_.end()) {
        TT_THROW("Cannot access soc descriptor for {} before device driver is initialized! Call initialize_device_driver({}) first", chip, chip);
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

void Cluster::verify_eth_fw() const {
    for (const auto &[chip, mmio_device_id] : this->device_to_mmio_device_) {
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
    if (this->device_to_mmio_device_.find(chip_id) != this->device_to_mmio_device_.end()) {
        // get_clocks returns MMIO device ID -> clock frequency
        // There is one driver per MMIO device, so we use that to index returned map
        chip_id_t mmio_device_id = this->device_to_mmio_device_.at(chip_id);
        return this->get_driver(chip_id).get_clocks().at(mmio_device_id);
    }
    TT_THROW("Cannot get frequency for device {} that is not initialized!", chip_id);
    return 0;
}

void Cluster::deassert_risc_reset_at_core(const tt_cxy_pair &physical_chip_coord) const {
    const metal_SocDescriptor &soc_desc = this->get_soc_desc(physical_chip_coord.chip);
    tt_cxy_pair virtual_chip_coord = soc_desc.convert_to_umd_coordinates(physical_chip_coord);
    this->get_driver(virtual_chip_coord.chip).deassert_risc_reset_at_core(virtual_chip_coord);
}

void Cluster::assert_risc_reset_at_core(const tt_cxy_pair &physical_chip_coord) const {
    const metal_SocDescriptor &soc_desc = this->get_soc_desc(physical_chip_coord.chip);
    tt_cxy_pair virtual_chip_coord = soc_desc.convert_to_umd_coordinates(physical_chip_coord);
    this->get_driver(virtual_chip_coord.chip).assert_risc_reset_at_core(virtual_chip_coord);
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
        tt::watcher_sanitize_host_noc_write(
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
        tt::watcher_sanitize_host_noc_read(
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
        tt::watcher_sanitize_host_noc_write(soc_desc, {target.x, target.y}, addr, size_in_bytes);
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
        tt::watcher_sanitize_host_noc_read(soc_desc, {target.x, target.y}, addr, size_in_bytes);
    }
    tt_cxy_pair virtual_target = soc_desc.convert_to_umd_coordinates(target);
    this->get_driver(chip_id).read_from_device(mem_ptr, virtual_target, addr, size_in_bytes, "REG_TLB");
}

void Cluster::write_sysmem(const void* vec, uint32_t size_in_bytes, uint64_t addr, chip_id_t src_device_id, uint16_t channel) const {
    TT_ASSERT(this->cluster_desc_->is_chip_mmio_capable(src_device_id));
    this->get_driver(src_device_id).write_to_sysmem(vec, size_in_bytes, addr, channel, src_device_id);
}

void Cluster::read_sysmem(void *vec, uint32_t size_in_bytes, uint64_t addr, chip_id_t src_device_id, uint16_t channel) const {
    TT_ASSERT(this->cluster_desc_->is_chip_mmio_capable(src_device_id));
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

std::unordered_map<chip_id_t, std::vector<CoreCoord>> Cluster::get_ethernet_cores_grouped_by_connected_chips(
    chip_id_t chip_id) const {
    const auto &soc_desc = get_soc_desc(chip_id);
    std::unordered_map<chip_id_t, std::vector<CoreCoord>> connected_chips;
    const auto &all_eth_connections = this->cluster_desc_->get_ethernet_connections();
    if (all_eth_connections.find(chip_id) == all_eth_connections.end()) {
        return {};
    }
    for (const auto &[eth_chan, connected_chip_chan] : all_eth_connections.at(chip_id)) {
        const auto &other_chip_id = std::get<0>(connected_chip_chan);
        if (connected_chips.find(other_chip_id) == connected_chips.end()) {
            std::vector<CoreCoord> active_ethernet_cores;

            for (const auto &channel_pair :
                 this->cluster_desc_->get_directly_connected_ethernet_channels_between_chips(chip_id, other_chip_id)) {
                ethernet_channel_t local_chip_chan = std::get<0>(channel_pair);
                active_ethernet_cores.emplace_back(
                    get_soc_desc(chip_id).chan_to_logical_eth_core_map.at(local_chip_chan));
            }
            connected_chips.insert({other_chip_id, active_ethernet_cores});
        } else {
            continue;
        }
    }
    return connected_chips;
}

// Ethernet cluster api
void Cluster::initialize_ethernet_sockets() {
    for (const auto &chip_id : this->cluster_desc_->get_all_chips()) {
        if (this->ethernet_sockets_.find(chip_id) == this->ethernet_sockets_.end()) {
            this->ethernet_sockets_.insert({chip_id, {}});
        }
        for (const auto &[connected_chip_id, eth_cores] :
             this->get_ethernet_cores_grouped_by_connected_chips(chip_id)) {
            if (this->ethernet_sockets_.at(chip_id).find(connected_chip_id) ==
                this->ethernet_sockets_.at(chip_id).end()) {
                this->ethernet_sockets_.at(chip_id).insert({connected_chip_id, {}});
            }
            if (this->ethernet_sockets_.find(connected_chip_id) == this->ethernet_sockets_.end()) {
                this->ethernet_sockets_.insert({connected_chip_id, {}});
            }
            if (this->ethernet_sockets_.at(connected_chip_id).find(chip_id) ==
                this->ethernet_sockets_.at(connected_chip_id).end()) {
                this->ethernet_sockets_.at(connected_chip_id).insert({chip_id, {}});
            } else {
                continue;
            }
            for (const auto &eth_core : eth_cores) {
              if(this->device_eth_routing_info_.at(chip_id).at(eth_core) == EthRouterMode::IDLE) {
                this->ethernet_sockets_.at(chip_id).at(connected_chip_id).emplace_back(eth_core);
                this->ethernet_sockets_.at(connected_chip_id)
                    .at(chip_id)
                    .emplace_back(std::get<1>(this->get_connected_ethernet_core(std::make_tuple(chip_id, eth_core))));
              }
            }
        }
    }
}

std::unordered_set<chip_id_t> Cluster::get_ethernet_connected_device_ids(chip_id_t chip_id) const {
    std::unordered_set<chip_id_t> device_ids;
    const auto &connected_chips = this->get_ethernet_cores_grouped_by_connected_chips(chip_id);
    for (const auto &[other_chip_id, eth_cores] : connected_chips) {
      for (const auto &eth_core: eth_cores) {
        if(this->device_eth_routing_info_.at(chip_id).at(eth_core) == EthRouterMode::IDLE) {
            device_ids.insert(other_chip_id);
        }
      }
    }
    return device_ids;
}

std::unordered_set<CoreCoord> Cluster::get_active_ethernet_cores(chip_id_t chip_id, bool skip_reserved_tunnel_cores) const {
    std::unordered_set<CoreCoord> active_ethernet_cores;
    const auto &connected_chips = this->get_ethernet_cores_grouped_by_connected_chips(chip_id);
    for (const auto &[other_chip_id, eth_cores] : connected_chips) {
        for (const auto &eth_core : eth_cores) {
            if(this->device_eth_routing_info_.at(chip_id).at(eth_core) == EthRouterMode::BI_DIR_TUNNELING and skip_reserved_tunnel_cores) {
                continue;
            }
            active_ethernet_cores.insert(eth_core);
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

std::vector<CoreCoord> Cluster::get_ethernet_sockets(chip_id_t local_chip, chip_id_t remote_chip) const {
    const auto &local_ethernet_sockets = this->ethernet_sockets_.at(local_chip);
    TT_ASSERT(
        local_ethernet_sockets.find(remote_chip) != local_ethernet_sockets.end(),
        "Device {} is not connected to Device {}",
        local_chip,
        remote_chip);
    return local_ethernet_sockets.at(remote_chip);
}

CoreCoord Cluster::ethernet_core_from_logical_core(chip_id_t chip_id, const CoreCoord &logical_core) const {
    const metal_SocDescriptor &soc_desc = get_soc_desc(chip_id);
    return soc_desc.get_physical_ethernet_core_from_logical(logical_core);
}

void Cluster::reserve_ethernet_cores_for_tunneling() {
    const char *TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    const uint32_t routing_info_addr = eth_l1_mem::address_map::ERISC_APP_ROUTING_INFO_BASE;
    for (const auto &[assoc_mmio_device, devices] : this->devices_grouped_by_assoc_mmio_device_) {
        for (const auto &chip_id : devices) {
            if (this->device_eth_routing_info_.find(chip_id) == this->device_eth_routing_info_.end()) {
                this->device_eth_routing_info_.insert({chip_id, {}});
            }
        }
        std::map<std::tuple<chip_id_t, chip_id_t>, bool> reserved_chip_connections = {};
        for (const auto &chip_id : devices) {
            if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
                for (const auto &[connected_chip_id, active_eth_cores] :
                     this->get_ethernet_cores_grouped_by_connected_chips(chip_id)) {
                    for (const auto &eth_core : active_eth_cores) {
                        const auto connected_eth_core =
                            std::get<1>(this->get_connected_ethernet_core(std::make_tuple(chip_id, eth_core)));
                        if (this->device_eth_routing_info_.at(chip_id).find(eth_core) ==
                            this->device_eth_routing_info_.at(chip_id).end()) {
                            tt_cxy_pair this_phys_core(chip_id, ethernet_core_from_logical_core(chip_id, eth_core));
                            if (devices.find(connected_chip_id) != devices.end() && reserved_chip_connections.find(std::make_tuple(chip_id, connected_chip_id)) == reserved_chip_connections.end()) {
                                // only setup fd tunneling for devices grouped with same mmio device and if no bi dir tunnel found between the two chips
                                    this->device_eth_routing_info_.at(chip_id).insert({eth_core, EthRouterMode::BI_DIR_TUNNELING});
                                    this->device_eth_routing_info_.at(connected_chip_id)
                                        .insert({connected_eth_core, EthRouterMode::BI_DIR_TUNNELING});
                                    reserved_chip_connections.insert({std::make_tuple(chip_id, connected_chip_id), true});
                                    reserved_chip_connections.insert({std::make_tuple(connected_chip_id, chip_id), true});
                            } else {
                                this->device_eth_routing_info_.at(chip_id).insert({eth_core, EthRouterMode::IDLE});
                            }
                        }
                    }
                }
            } else {
                // Slow dispatch mode
                for (const auto &[connected_chip_id, active_eth_cores] :
                     this->get_ethernet_cores_grouped_by_connected_chips(chip_id)) {
                    for (const auto &eth_core : active_eth_cores) {
                        this->device_eth_routing_info_.at(chip_id).insert({eth_core, EthRouterMode::IDLE});
                    }
                }
            }
        }
    }
}


tt_cxy_pair Cluster::get_eth_core_for_dispatch_core(
    tt_cxy_pair logical_dispatch_core, EthRouterMode mode, chip_id_t connected_chip_id) const {
    const auto &local_chip_id = logical_dispatch_core.chip;
    for (const auto &[eth_core, router_mode] : this->device_eth_routing_info_.at(local_chip_id)) {

      // Check for connected chip id since one chip can be bi directional tunneling to multiple chips
        const auto connected_tunnel_chip_id =
            std::get<0>(this->get_connected_ethernet_core(std::make_tuple(local_chip_id, eth_core)));
        if (router_mode == mode and connected_tunnel_chip_id == connected_chip_id) {
            return tt_cxy_pair(local_chip_id, eth_core);
        }
    }
    TT_ASSERT(false, "Cluster does not contain requested eth routing core");
    return {};
}

// TODO: ALLAN Can change to write one bit
void Cluster::set_internal_routing_info_for_ethernet_cores(bool enable_internal_routing) const {
    const uint32_t routing_info_addr = eth_l1_mem::address_map::ERISC_APP_ROUTING_INFO_BASE;
    // TODO: initialize devices if user does not
    // Must initialize remote chips first, then mmio chips since once mmio chips are doing fd routing
    // we do not always context switch to base FW
    const routing_info_t routing_info_disabled = {
        .routing_enabled = 0,
        .src_sent_valid_cmd = 0,
        .dst_acked_valid_cmd = 0,
      };
    const routing_info_t routing_info_enabled = {
        .routing_enabled = 1,
        .src_sent_valid_cmd = 0,
        .dst_acked_valid_cmd = 0,
      };
    for (const auto &[assoc_mmio_device, devices] : this->devices_grouped_by_assoc_mmio_device_) {
        for (const auto &chip_id : devices) {
            for (const auto &[eth_core, routing_info] : this->device_eth_routing_info_.at(chip_id)) {
                tt_cxy_pair eth_phys_core(chip_id, ethernet_core_from_logical_core(chip_id, eth_core));
                if (chip_id == assoc_mmio_device and not enable_internal_routing) {
                    // Disable internal ethernet routing for mmio devices
                    write_core(
                        (void *)&routing_info_disabled,
                        sizeof(routing_info_t),
                        eth_phys_core,
                        routing_info_addr,
                        false);
                } else if (chip_id != assoc_mmio_device and enable_internal_routing) {
                    // Enable internal ethernet routing for non-mmio devices
                    write_core(
                        (void *)&routing_info_enabled,
                        sizeof(routing_info_t),
                        eth_phys_core,
                        routing_info_addr,
                        false);

                } else {
                    continue;
                }
            }
        }
        for (const auto &chip_id : devices) {
            for (const auto &[eth_core, routing_info] : this->device_eth_routing_info_.at(chip_id)) {
                tt_cxy_pair eth_phys_core(chip_id, ethernet_core_from_logical_core(chip_id, eth_core));
                if (chip_id != assoc_mmio_device and not enable_internal_routing) {
                    // Disable internal ethernet routing for non-mmio devices
                    write_core(
                        (void *)&routing_info_disabled,
                        sizeof(routing_info_t),
                        eth_phys_core,
                        routing_info_addr,
                        false);
                } else if (chip_id == assoc_mmio_device and enable_internal_routing) {
                    // Enable internal ethernet routing for mmio devices
                    write_core(
                        (void *)&routing_info_enabled,
                        sizeof(routing_info_t),
                        eth_phys_core,
                        routing_info_addr,
                        false);
                } else {
                    continue;
                }
            }
        }
    }
}

uint32_t Cluster::get_tensix_soft_reset_addr() const {
    return DEVICE_DATA.TENSIX_SOFT_RESET_ADDR;
}

}  // namespace tt

std::ostream &operator<<(std::ostream &os, tt_target_dram const &dram) {
    os << "Target DRAM chip = " << std::get<0>(dram) << ", chan = " << std::get<1>(dram)
       << ", subchan = " << std::get<2>(dram);
    return os;
}
