// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctype.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <assert.hpp>
#include <circular_buffer_constants.h>  // For NUM_CIRCULAR_BUFFERS
#include <core_coord.hpp>
#include <fmt/base.h>
#include <metal_soc_descriptor.h>
#include <tt-logger/tt-logger.hpp>
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/types/arch.h>
#include <umd/device/types/cluster_descriptor_types.h>
#include <umd/device/types/xy_pair.h>

#include "control_plane.hpp"
#include "core_descriptor.hpp"
#include "debug_helpers.hpp"
#include "dev_msgs.h"
#include "dispatch_core_common.hpp"
#include "hal_types.hpp"
#include "hw/inc/debug/ring_buffer.h"
#include "impl/context/metal_context.hpp"
#include "llrt.hpp"
#include "watcher_device_reader.hpp"

using namespace tt::tt_metal;
using std::string;

#define NOC_MCAST_ADDR_START_X(addr) (MetalContext::instance().hal().get_noc_mcast_addr_start_x(addr))
#define NOC_MCAST_ADDR_START_Y(addr) (MetalContext::instance().hal().get_noc_mcast_addr_start_y(addr))
#define NOC_MCAST_ADDR_END_X(addr) (MetalContext::instance().hal().get_noc_mcast_addr_end_x(addr))
#define NOC_MCAST_ADDR_END_Y(addr) (MetalContext::instance().hal().get_noc_mcast_addr_end_y(addr))
#define NOC_UNICAST_ADDR_X(addr) (MetalContext::instance().hal().get_noc_ucast_addr_x(addr))
#define NOC_UNICAST_ADDR_Y(addr) (MetalContext::instance().hal().get_noc_ucast_addr_y(addr))
#define NOC_LOCAL_ADDR(addr) (MetalContext::instance().hal().get_noc_local_addr(addr))
#define NOC_OVERLAY_START_ADDR (MetalContext::instance().hal().get_noc_overlay_start_addr())
#define NOC_STREAM_REG_SPACE_SIZE (MetalContext::instance().hal().get_noc_stream_reg_space_size())
#define STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX \
    (MetalContext::instance().hal().get_noc_stream_remote_dest_buf_size_reg_index())
#define STREAM_REMOTE_DEST_BUF_START_REG_INDEX \
    (MetalContext::instance().hal().get_noc_stream_remote_dest_buf_start_reg_index())

namespace {  // Helper functions

// Helper function to get string rep of riscv type
// TODO: Remove this and switch to HAL's generic names (such as TENSIX_DM_0),
// or move it to HAL and make it arch-dependent.
const char* get_riscv_name(HalProgrammableCoreType core_type, uint32_t processor_index) {
    switch (core_type) {
        case HalProgrammableCoreType::TENSIX: {
            static const char* const names[] = {
                " brisc",
                "ncrisc",
                "trisc0",
                "trisc1",
                "trisc2",
            };
            TT_FATAL(
                processor_index < 5,
                "Watcher data corrupted, unexpected processor index {} on core {}",
                processor_index,
                core_type);
            return names[processor_index];
        }
        case HalProgrammableCoreType::ACTIVE_ETH: {
            static const char* const names[] = {"erisc", "subordinate_erisc"};
            TT_FATAL(
                processor_index < 2,
                "Watcher data corrupted, unexpected processor index {} on core {}",
                processor_index,
                core_type);
            return names[processor_index];
        }
        case HalProgrammableCoreType::IDLE_ETH: {
            static const char* const names[] = {"ierisc", "subordinate_ierisc"};
            TT_FATAL(
                processor_index < 2,
                "Watcher data corrupted, unexpected processor index {} on core {}",
                processor_index,
                core_type);
            return names[processor_index];
        }
        case HalProgrammableCoreType::COUNT: TT_THROW("unsupported core type");
    }
    TT_THROW("unreachable");
}

// Helper function to determine core type from virtual coord. TODO: Remove this once we fix code types.
CoreType core_type_from_virtual_core(chip_id_t device_id, const CoreCoord& virtual_coord) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().is_worker_core(virtual_coord, device_id)) {
        return CoreType::WORKER;
    } else if (tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_core(virtual_coord, device_id)) {
        return CoreType::ETH;
    }

    const metal_SocDescriptor& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id);

    const std::vector<tt::umd::CoreCoord>& translated_dram_cores =
        soc_desc.get_cores(CoreType::DRAM, CoordSystem::TRANSLATED);
    if (std::find(translated_dram_cores.begin(), translated_dram_cores.end(), virtual_coord) !=
        translated_dram_cores.end()) {
        return CoreType::DRAM;
    }

    CoreType core_type = soc_desc.translate_coord_to(virtual_coord, CoordSystem::NOC0, CoordSystem::NOC0).core_type;
    if (core_type == CoreType::TENSIX) {
        core_type = CoreType::WORKER;
    }
    return core_type;
}

// Helper function to convert noc coord -> virtual coord. TODO: Remove this once we fix code types.
CoreCoord virtual_noc_coordinate(chip_id_t device_id, uint8_t noc_index, CoreCoord coord) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == tt::ARCH::BLACKHOLE) {
        return coord;
    }
    auto grid_size = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id).grid_size;
    if (coord.x >= grid_size.x || coord.y >= grid_size.y) {
        // Coordinate already in virtual space: NOC0 and NOC1 are the same
        return coord;
    } else {
        // Coordinate passed in can be NOC0 or NOC1. The noc_index corresponds to
        // the system this coordinate belongs to.
        // Use this to convert to NOC0 coordinates and then derive Virtual Coords from it.
        CoreCoord physical_coord = {
            MetalContext::instance().hal().noc_coordinate(noc_index, grid_size.x, coord.x),
            MetalContext::instance().hal().noc_coordinate(noc_index, grid_size.y, coord.y)};
        return tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_physical_coordinates(
            device_id, physical_coord);
    }
}

// Helper function to get string rep of noc target.
string get_noc_target_str(
    chip_id_t device_id,
    CoreCoord virtual_coord,
    HalProgrammableCoreType programmable_core_type,
    int noc,
    const debug_sanitize_noc_addr_msg_t* san) {
    auto get_core_and_mem_type = [](chip_id_t device_id, CoreCoord& noc_coord, int noc) -> std::pair<string, string> {
        // Get the virtual coord from the noc coord
        CoreCoord virtual_core = virtual_noc_coordinate(device_id, noc, noc_coord);
        CoreType core_type;
        try {
            core_type = core_type_from_virtual_core(device_id, virtual_core);
        } catch (std::runtime_error& e) {
            // We may not be able to get a core type if the virtual coords are bad.
            return {"Unknown", ""};
        }
        switch (core_type) {
            case CoreType::DRAM: return {"DRAM", "DRAM"};
            case CoreType::ETH: return {"Ethernet", "L1"};
            case CoreType::PCIE: return {"PCIe", "PCIE"};
            case CoreType::WORKER: return {"Tensix", "L1"};
            default: return {"Unknown", ""};
        }
    };
    string out = fmt::format(
        "{} using noc{} tried to {} {} {} bytes {} local L1[{:#08x}] {} ",
        get_riscv_name(programmable_core_type, san->which_risc),
        noc,
        san->is_multicast ? "multicast" : "unicast",
        san->is_write ? "write" : "read",
        san->len,
        san->is_write ? "from" : "to",
        san->l1_addr,
        san->is_write ? "to" : "from");

    if (san->is_multicast) {
        CoreCoord target_virtual_noc_core_start = {
            NOC_MCAST_ADDR_START_X(san->noc_addr), NOC_MCAST_ADDR_START_Y(san->noc_addr)};
        CoreCoord target_virtual_noc_core_end = {
            NOC_MCAST_ADDR_END_X(san->noc_addr), NOC_MCAST_ADDR_END_Y(san->noc_addr)};
        auto type_and_mem = get_core_and_mem_type(device_id, target_virtual_noc_core_start, noc);
        out += fmt::format(
            "{} core range w/ virtual coords {}-{} {}",
            type_and_mem.first,
            target_virtual_noc_core_start.str(),
            target_virtual_noc_core_end.str(),
            type_and_mem.second);
    } else {
        CoreCoord target_virtual_noc_core = {NOC_UNICAST_ADDR_X(san->noc_addr), NOC_UNICAST_ADDR_Y(san->noc_addr)};
        auto type_and_mem = get_core_and_mem_type(device_id, target_virtual_noc_core, noc);
        out += fmt::format(
            "{} core w/ virtual coords {} {}", type_and_mem.first, target_virtual_noc_core.str(), type_and_mem.second);
    }

    out += fmt::format("[addr=0x{:08x}]", NOC_LOCAL_ADDR(san->noc_addr));
    return out;
}

const launch_msg_t* get_valid_launch_message(const mailboxes_t* mbox_data) {
    uint32_t launch_msg_read_ptr = mbox_data->launch_msg_rd_ptr;
    if (mbox_data->launch[launch_msg_read_ptr].kernel_config.enables == 0) {
        launch_msg_read_ptr = (launch_msg_read_ptr - 1 + launch_msg_buffer_num_entries) % launch_msg_buffer_num_entries;
    }
    return &mbox_data->launch[launch_msg_read_ptr];
}

}  // anonymous namespace

namespace tt::tt_metal {

// Struct containing relevant info for stack usage
struct stack_usage_info_t {
    CoreCoord virtual_coord;
    uint16_t stack_free = uint16_t(~0);
    uint16_t kernel_id{};
};

struct PausedCoreInfo {
    CoreCoord virtual_coord;
    uint32_t processor_index{};

    bool operator<(const PausedCoreInfo& other) const {
        return std::tie(virtual_coord, processor_index) < std::tie(other.virtual_coord, other.processor_index);
    }
};

// Information that needs to be kept around on a per-dump basis, shared per-core
struct WatcherDeviceReader::DumpData {
    std::set<PausedCoreInfo> paused_cores;
    std::map<HalProcessorIdentifier, stack_usage_info_t> highest_stack_usage;
    std::map<int, bool> used_kernel_names;
};

class WatcherDeviceReader::Core {
private:
    CoreCoord virtual_coord_;
    HalProgrammableCoreType programmable_core_type_;
    std::string core_str_;
    std::vector<uint32_t> l1_read_buf_;
    const mailboxes_t* mbox_data_;
    const launch_msg_t* launch_msg_;
    const WatcherDeviceReader& reader_;
    DumpData& dump_data_;

    void DumpL1Status() const;
    void DumpNocSanitizeStatus(int noc) const;
    void DumpAssertStatus() const;
    void DumpPauseStatus() const;
    void DumpEthLinkStatus() const;
    void DumpRingBuffer(bool to_stdout = false) const;
    void DumpRunState(uint32_t state) const;
    void DumpLaunchMessage() const;
    void DumpWaypoints(bool to_stdout = false) const;
    void DumpSyncRegs() const;
    void DumpStackUsage() const;
    void LogRunningKernels() const;
    const std::string& GetKernelName(uint32_t processor_index) const;
    void ValidateKernelIDs() const;

public:
    Core(
        CoreCoord logical_coord,
        HalProgrammableCoreType programmable_core_type,
        const WatcherDeviceReader& reader,
        DumpData& dump_data);

    void Dump() const;
};

WatcherDeviceReader::WatcherDeviceReader(FILE* f, chip_id_t device_id, const std::vector<string>& kernel_names) :
    f(f), device_id(device_id), kernel_names(kernel_names) {
    // On init, read out eth link retraining register so that we can see if retraining has occurred. WH only for now.
    if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == ARCH::WORMHOLE_B0 &&
        tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_enabled()) {
        std::vector<uint32_t> read_data;
        for (const CoreCoord& eth_core :
             tt::tt_metal::MetalContext::instance().get_control_plane().get_active_ethernet_cores(device_id)) {
            CoreCoord virtual_core =
                tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
                    device_id, eth_core, CoreType::ETH);
            read_data = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                device_id,
                virtual_core,
                MetalContext::instance().hal().get_dev_addr(
                    HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::RETRAIN_COUNT),
                sizeof(uint32_t));
            logical_core_to_eth_link_retraining_count[eth_core] = read_data[0];
        }
    }

    num_erisc_cores = tt::tt_metal::MetalContext::instance().hal().get_processor_classes_count(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
}

WatcherDeviceReader::~WatcherDeviceReader() {
    // On close, read out eth link retraining register so that we can see if retraining has occurred.
    if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == ARCH::WORMHOLE_B0 &&
        tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_enabled()) {
        std::vector<uint32_t> read_data;
        for (const CoreCoord& eth_core :
             tt::tt_metal::MetalContext::instance().get_control_plane().get_active_ethernet_cores(device_id)) {
            CoreCoord virtual_core =
                tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
                    device_id, eth_core, CoreType::ETH);
            read_data = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                device_id,
                virtual_core,
                MetalContext::instance().hal().get_dev_addr(
                    HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::RETRAIN_COUNT),
                sizeof(uint32_t));
            uint32_t num_events = read_data[0] - logical_core_to_eth_link_retraining_count[eth_core];
            if (num_events > 0) {
                log_warning(
                    tt::LogMetal,
                    "Device {} virtual ethernet core {}: Watcher detected {} link retraining events.",
                    device_id,
                    virtual_core,
                    num_events);
            }
            if (f) {
                fprintf(
                    f,
                    "%s\n",
                    fmt::format(
                        "\tDevice {} Ethernet Core {} retraining events: {}", device_id, virtual_core, num_events)
                        .c_str());
            }
        }
    }
}

void WatcherDeviceReader::Dump(FILE* file) {
    // If specified, override the existing file destination
    if (file != nullptr) {
        this->f = file;
    }

    // At this point, file should be valid.
    TT_ASSERT(this->f != nullptr);

    if (f != stdout && f != stderr) {
        log_info(tt::LogMetal, "Watcher checking device {}", device_id);
    }

    DumpData dump_data;

    // Ignore storage-only cores
    std::unordered_set<CoreCoord> storage_only_cores;
    uint8_t num_hw_cqs = tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().get_num_hw_cqs();
    DispatchCoreConfig dispatch_core_config =
        tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
    for (auto core_coord : tt::get_logical_storage_cores(device_id, num_hw_cqs, dispatch_core_config)) {
        storage_only_cores.insert(core_coord);
    }

    // Dump worker cores
    CoreCoord grid_size =
        tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord coord = {x, y};
            if (storage_only_cores.find(coord) == storage_only_cores.end()) {
                Core(coord, HalProgrammableCoreType::TENSIX, *this, dump_data).Dump();
            }
        }
    }

    // Dump eth cores
    for (const CoreCoord& eth_core :
         tt::tt_metal::MetalContext::instance().get_control_plane().get_active_ethernet_cores(device_id)) {
        Core(eth_core, HalProgrammableCoreType::ACTIVE_ETH, *this, dump_data).Dump();
    }
    for (const CoreCoord& eth_core :
         tt::tt_metal::MetalContext::instance().get_control_plane().get_inactive_ethernet_cores(device_id)) {
        Core(eth_core, HalProgrammableCoreType::IDLE_ETH, *this, dump_data).Dump();
    }

    for (auto k_id : dump_data.used_kernel_names) {
        fprintf(f, "k_id[%3d]: %s\n", k_id.first, kernel_names[k_id.first].c_str());
    }

    const auto& hal = MetalContext::instance().hal();
    // Print stack usage report for this device/dump
    if (!dump_data.highest_stack_usage.empty()) {
        fprintf(f, "Stack usage summary:");
        for (auto& [processor, info] : dump_data.highest_stack_usage) {
            auto processor_name = get_riscv_name(
                processor.core_type,
                hal.get_processor_index(processor.core_type, processor.processor_class, processor.processor_type));
            // Threshold of free space for warning.
            constexpr uint32_t min_threshold = 64;
            fprintf(
                f,
                "\n\t%s highest stack usage: %u bytes free, on core %s, running kernel %s",
                processor_name,
                info.stack_free,
                info.virtual_coord.str().c_str(),
                kernel_names[info.kernel_id].c_str());
            if (info.stack_free == 0) {
                // We had no free stack, this probably means we
                // overflowed, but it could be a remarkable coincidence.
                fprintf(f, " (OVERFLOW)");
                log_fatal(
                    tt::LogMetal,
                    "Watcher detected stack overflow on Device {} Core {}: "
                    "{}! Kernel {} uses (at least) all of the stack.",
                    device_id,
                    info.virtual_coord.str(),
                    processor_name,
                    kernel_names[info.kernel_id].c_str());
            } else if (info.stack_free < min_threshold) {
                fprintf(f, " (Close to overflow)");
                log_warning(
                    tt::LogMetal,
                    "Watcher detected stack had fewer than {} bytes free on Device {} Core {}: "
                    "{}! Kernel {} leaves {} bytes unused.",
                    min_threshold,
                    device_id,
                    info.virtual_coord.str(),
                    processor_name,
                    kernel_names[info.kernel_id].c_str(),
                    info.stack_free);
            }
        }
        fprintf(f, "\n");
    }

    // Handle any paused cores, wait for user input.
    if (!dump_data.paused_cores.empty()) {
        string paused_cores_str = "Paused cores: ";
        for (auto& [virtual_core, processor_index] : dump_data.paused_cores) {
            paused_cores_str += fmt::format(
                "{}:{}, ",
                virtual_core.str(),
                get_riscv_name(get_programmable_core_type(virtual_core, device_id), processor_index));
        }
        paused_cores_str += "\n";
        fprintf(f, "%s", paused_cores_str.c_str());
        log_info(tt::LogMetal, "{}Press ENTER to unpause core(s) and continue...", paused_cores_str);
        if (!tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_auto_unpause()) {
            while (std::cin.get() != '\n') {
                ;
            }
        }

        // Clear all pause flags
        for (auto& [virtual_core, processor_index] : dump_data.paused_cores) {
            uint64_t addr =
                hal.get_dev_addr(get_programmable_core_type(virtual_core, device_id), HalL1MemAddrType::WATCHER) +
                offsetof(watcher_msg_t, pause_status);

            // Clear only the one flag that we saved, in case another one was raised on device
            auto pause_data = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                device_id, virtual_core, addr, sizeof(debug_pause_msg_t));
            auto pause_msg = reinterpret_cast<debug_pause_msg_t*>(&(pause_data[0]));
            pause_msg->flags[processor_index] = 0;
            tt::tt_metal::MetalContext::instance().get_cluster().write_core(device_id, virtual_core, pause_data, addr);
        }
    }
    fflush(f);
}

WatcherDeviceReader::Core::Core(
    CoreCoord logical_coord,
    HalProgrammableCoreType programmable_core_type,
    const WatcherDeviceReader& reader,
    DumpData& dump_data) :
    programmable_core_type_(programmable_core_type), reader_(reader), dump_data_(dump_data) {
    const auto& rtoptions = tt_metal::MetalContext::instance().rtoptions();
    const auto& hal = MetalContext::instance().hal();
    CoreType core_type = hal.get_core_type(hal.get_programmable_core_type_index(programmable_core_type));
    virtual_coord_ =
        tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
            reader_.device_id, logical_coord, core_type);

    // Print device id, core coords (logical)
    string core_type_str = programmable_core_type == HalProgrammableCoreType::ACTIVE_ETH ? "acteth"
                           : programmable_core_type == HalProgrammableCoreType::IDLE_ETH ? "idleth"
                                                                                         : "worker";
    string core_coord_str = fmt::format(
        "core(x={:2},y={:2}) virtual(x={:2},y={:2})",
        logical_coord.x,
        logical_coord.y,
        virtual_coord_.x,
        virtual_coord_.y);
    if (rtoptions.get_watcher_phys_coords()) {
        CoreCoord phys_core =
            tt::tt_metal::MetalContext::instance().get_cluster().get_physical_coordinate_from_logical_coordinates(
                reader_.device_id, logical_coord, core_type, true);
        core_coord_str += fmt::format(" phys(x={:2},y={:2})", phys_core.x, phys_core.y);
    }
    core_str_ = fmt::format("Device {} {} {}", reader_.device_id, core_type_str, core_coord_str);
    fprintf(reader_.f, "%s: ", core_str_.c_str());

    uint64_t mailbox_addr =
        MetalContext::instance().hal().get_dev_addr(programmable_core_type, HalL1MemAddrType::MAILBOX);

    constexpr uint32_t mailbox_read_size = offsetof(mailboxes_t, watcher) + sizeof(watcher_msg_t);
    l1_read_buf_ = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        reader_.device_id, virtual_coord_, mailbox_addr, mailbox_read_size);
    mbox_data_ = reinterpret_cast<mailboxes_t*>(l1_read_buf_.data());
    launch_msg_ = get_valid_launch_message(mbox_data_);
}

void WatcherDeviceReader::Core::Dump() const {
    const auto& rtoptions = tt_metal::MetalContext::instance().rtoptions();
    bool is_eth_core =
        (programmable_core_type_ == HalProgrammableCoreType::ACTIVE_ETH ||
         programmable_core_type_ == HalProgrammableCoreType::IDLE_ETH);

    ValidateKernelIDs();

    // Whether or not watcher data is available depends on a flag set on the device.
    if (mbox_data_->watcher.enable != WatcherEnabled and mbox_data_->watcher.enable != WatcherDisabled) {
        TT_THROW(
            "Watcher read invalid watcher.enable on {}. Read {}, valid values are {} and {}.",
            core_str_,
            mbox_data_->watcher.enable,
            WatcherEnabled,
            WatcherDisabled);
    }
    bool enabled = (mbox_data_->watcher.enable == WatcherEnabled);

    if (enabled) {
        // Dump state only gathered if device is compiled w/ watcher
        if (!rtoptions.watcher_status_disabled()) {
            DumpWaypoints();
        }
        // Ethernet cores have firmware that starts at address 0, so no need to check it for a
        // magic value.
        if (!is_eth_core) {
            DumpL1Status();
        }
        if (!rtoptions.watcher_noc_sanitize_disabled()) {
            const auto NUM_NOCS_ = tt::tt_metal::MetalContext::instance().hal().get_num_nocs();
            for (uint32_t noc = 0; noc < NUM_NOCS_; noc++) {
                DumpNocSanitizeStatus(noc);
            }
        }
        if (!rtoptions.watcher_assert_disabled()) {
            DumpAssertStatus();
        }
        if (!rtoptions.watcher_pause_disabled()) {
            DumpPauseStatus();
        }

        if (is_eth_core && !rtoptions.watcher_eth_link_status_disabled()) {
            DumpEthLinkStatus();
        }
    }

    // Dump state always available
    DumpLaunchMessage();
    // Ethernet cores don't use the sync reg
    if (!is_eth_core && rtoptions.get_watcher_dump_all()) {
        // Reading registers while running can cause hangs, only read if
        // requested explicitly
        DumpSyncRegs();
    }

    // Eth core only reports erisc kernel id, uses the brisc field
    if (is_eth_core) {
        fprintf(reader_.f, "k_id:%3d", launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM0]);
        if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == ARCH::BLACKHOLE) {
            fprintf(reader_.f, "|%3d", launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM1]);
        }
    } else {
        fprintf(
            reader_.f,
            "k_ids:%3d|%3d|%3d",
            launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM0],
            launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM1],
            launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE]);

        if (rtoptions.get_watcher_text_start()) {
            uint32_t kernel_config_base = launch_msg_->kernel_config.kernel_config_base[0];
            fprintf(reader_.f, " text_start:");
            for (size_t i = 0; i < NUM_PROCESSORS_PER_CORE_TYPE; i++) {
                const char* separator = (i > 0) ? "|" : "";
                fprintf(
                    reader_.f,
                    "%s0x%x",
                    separator,
                    kernel_config_base + launch_msg_->kernel_config.kernel_text_offset[i]);
            }
        }
    }

    // Ring buffer at the end because it can print a bunch of data, same for stack usage
    if (enabled) {
        if (!rtoptions.watcher_stack_usage_disabled()) {
            DumpStackUsage();
        }
        if (!rtoptions.watcher_ring_buffer_disabled()) {
            DumpRingBuffer();
        }
    }

    fprintf(reader_.f, "\n");

    fflush(reader_.f);
}

void WatcherDeviceReader::Core::DumpL1Status() const {
    // Read L1 address 0, looking for memory corruption
    std::vector<uint32_t> data;
    data = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        reader_.device_id, virtual_coord_, HAL_MEM_L1_BASE, sizeof(uint32_t));
    TT_ASSERT(programmable_core_type_ == HalProgrammableCoreType::TENSIX);
    uint32_t core_type_idx =
        MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    auto fw_launch_value =
        MetalContext::instance().hal().get_jit_build_config(core_type_idx, 0, 0).fw_launch_addr_value;
    if (data[0] != fw_launch_value) {
        LogRunningKernels();
        TT_THROW("Watcher found corruption at L1[0] on core {}: read {}", virtual_coord_.str(), data[0]);
    }
}

void WatcherDeviceReader::Core::DumpNocSanitizeStatus(int noc) const {
    const debug_sanitize_noc_addr_msg_t* san = &mbox_data_->watcher.sanitize_noc[noc];
    string error_msg;
    string error_reason;

    switch (san->return_code) {
        case DebugSanitizeNocOK:
            if (san->noc_addr != DEBUG_SANITIZE_NOC_SENTINEL_OK_64 ||
                san->l1_addr != DEBUG_SANITIZE_NOC_SENTINEL_OK_32 || san->len != DEBUG_SANITIZE_NOC_SENTINEL_OK_32 ||
                san->which_risc != DEBUG_SANITIZE_NOC_SENTINEL_OK_16 ||
                san->is_multicast != DEBUG_SANITIZE_NOC_SENTINEL_OK_8 ||
                san->is_write != DEBUG_SANITIZE_NOC_SENTINEL_OK_8 ||
                san->is_target != DEBUG_SANITIZE_NOC_SENTINEL_OK_8) {
                error_msg = fmt::format(
                    "Watcher unexpected noc debug state on core {}, reported valid got noc{}{{0x{:08x}, {} }}",
                    virtual_coord_.str(),
                    san->which_risc,
                    san->noc_addr,
                    san->len);
                error_msg += " (corrupted noc sanitization state - sanitization memory overwritten)";
            }
            break;
        case DebugSanitizeNocAddrUnderflow:
            error_msg = get_noc_target_str(reader_.device_id, virtual_coord_, programmable_core_type_, noc, san);
            error_msg += string(san->is_target ? " (NOC target" : " (Local L1") + " address underflow).";
            break;
        case DebugSanitizeNocAddrOverflow:
            error_msg = get_noc_target_str(reader_.device_id, virtual_coord_, programmable_core_type_, noc, san);
            error_msg += string(san->is_target ? " (NOC target" : " (Local L1") + " address overflow).";
            break;
        case DebugSanitizeNocAddrZeroLength:
            error_msg = get_noc_target_str(reader_.device_id, virtual_coord_, programmable_core_type_, noc, san);
            error_msg += " (zero length transaction).";
            break;
        case DebugSanitizeNocTargetInvalidXY:
            error_msg = get_noc_target_str(reader_.device_id, virtual_coord_, programmable_core_type_, noc, san);
            error_msg += " (NOC target address did not map to any known Tensix/Ethernet/DRAM/PCIE core).";
            break;
        case DebugSanitizeNocMulticastNonWorker:
            error_msg = get_noc_target_str(reader_.device_id, virtual_coord_, programmable_core_type_, noc, san);
            error_msg += " (multicast to non-worker core).";
            break;
        case DebugSanitizeNocMulticastInvalidRange:
            error_msg = get_noc_target_str(reader_.device_id, virtual_coord_, programmable_core_type_, noc, san);
            error_msg += " (multicast invalid range).";
            break;
        case DebugSanitizeNocAlignment:
            error_msg = get_noc_target_str(reader_.device_id, virtual_coord_, programmable_core_type_, noc, san);
            error_msg += " (invalid address alignment in NOC transaction).";
            break;
        case DebugSanitizeNocMixedVirtualandPhysical:
            error_msg = get_noc_target_str(reader_.device_id, virtual_coord_, programmable_core_type_, noc, san);
            error_msg += " (mixing virtual and virtual coordinates in Mcast).";
            break;
        case DebugSanitizeInlineWriteDramUnsupported:
            error_msg = get_noc_target_str(reader_.device_id, virtual_coord_, programmable_core_type_, noc, san);
            error_msg += " (inline dw writes do not support DRAM destination addresses).";
            break;
        case DebugSanitizeNocAddrMailbox:
            error_msg = get_noc_target_str(reader_.device_id, virtual_coord_, programmable_core_type_, noc, san);
            error_msg += string(san->is_target ? " (NOC target" : " (Local L1") + " overwrites mailboxes).";
            break;
        case DebugSanitizeNocLinkedTransactionViolation:
            error_msg = get_noc_target_str(reader_.device_id, virtual_coord_, programmable_core_type_, noc, san);
            error_msg += fmt::format(" (submitting a non-mcast transaction when there's a linked transaction).");
            break;
        default:
            error_msg = fmt::format(
                "Watcher unexpected data corruption, noc debug state on core {}, unknown failure code: {}",
                virtual_coord_.str(),
                san->return_code);
            error_msg += " (corrupted noc sanitization state - unknown failure code).";
    }

    // If we logged an error, print to stdout and throw.
    if (!error_msg.empty()) {
        log_warning(tt::LogMetal, "Watcher detected NOC error and stopped device:");
        log_warning(tt::LogMetal, "{}: {}", core_str_, error_msg);
        DumpWaypoints(true);
        DumpRingBuffer(true);
        LogRunningKernels();
        // Save the error string for checking later in unit tests.
        MetalContext::instance().watcher_server()->set_exception_message(fmt::format("{}: {}", core_str_, error_msg));
        TT_THROW("{}: {}", core_str_, error_msg);
    }
}

void WatcherDeviceReader::Core::DumpAssertStatus() const {
    const debug_assert_msg_t* assert_status = &mbox_data_->watcher.assert_status;
    if (assert_status->tripped == DebugAssertOK) {
        if (assert_status->line_num != DEBUG_SANITIZE_NOC_SENTINEL_OK_16 ||
            assert_status->which != DEBUG_SANITIZE_NOC_SENTINEL_OK_8) {
            TT_THROW(
                "Watcher unexpected assert state on core {}, reported OK but got processor {}, line {}.",
                virtual_coord_.str(),
                assert_status->which,
                assert_status->line_num);
        }
        return;  // no assert tripped, nothing to do
    }
    std::string error_msg =
        fmt::format("{}: {} ", core_str_, get_riscv_name(programmable_core_type_, assert_status->which));
    switch (assert_status->tripped) {
        case DebugAssertTripped: {
            error_msg += fmt::format("tripped an assert on line {}.", assert_status->line_num);
            // TODO: Get rid of this once #6098 is implemented.
            error_msg +=
                " Note that file name reporting is not yet implemented, and the reported line number for the assert "
                "may be from a different file.";
            break;
        }
        case DebugAssertNCriscNOCReadsFlushedTripped: {
            error_msg +=
                "detected an inter-kernel data race due to kernel completing with pending NOC transactions (missing "
                "NOC reads flushed barrier).";
            break;
        }
        case DebugAssertNCriscNOCNonpostedWritesSentTripped: {
            error_msg +=
                "detected an inter-kernel data race due to kernel completing with pending NOC transactions (missing "
                "NOC non-posted writes sent barrier).";
            break;
        }
        case DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped: {
            error_msg +=
                "detected an inter-kernel data race due to kernel completing with pending NOC transactions (missing "
                "NOC non-posted atomics flushed barrier).";
            break;
        }
        case DebugAssertNCriscNOCPostedWritesSentTripped: {
            error_msg +=
                "detected an inter-kernel data race due to kernel completing with pending NOC transactions (missing "
                "NOC posted writes sent barrier).";
            break;
        }
        default:
            LogRunningKernels();
            TT_THROW(
                "Watcher data corruption, noc assert state on core {} unknown failure code: {}.\n",
                virtual_coord_.str(),
                assert_status->tripped);
    }
    error_msg += fmt::format(" Current kernel: {}.", GetKernelName(assert_status->which));
    log_warning(tt::LogMetal, "Watcher stopped the device due to tripped assert, see watcher log for more details");
    log_warning(tt::LogMetal, "{}", error_msg);
    DumpWaypoints(true);
    DumpRingBuffer(true);
    LogRunningKernels();
    MetalContext::instance().watcher_server()->set_exception_message(error_msg);
    TT_THROW("Watcher detected tripped assert and stopped device.");
}

void WatcherDeviceReader::Core::DumpPauseStatus() const {
    const debug_pause_msg_t* pause_status = &mbox_data_->watcher.pause_status;
    const auto& hal = MetalContext::instance().hal();
    // Just record which cores are paused, printing handled at the end.
    auto num_processors = hal.get_num_risc_processors(programmable_core_type_);
    for (uint32_t processor_index = 0; processor_index < num_processors; processor_index++) {
        auto pause = pause_status->flags[processor_index];
        if (pause == 1) {
            dump_data_.paused_cores.insert({virtual_coord_, processor_index});
        } else if (pause > 1) {
            string error_reason = fmt::format(
                "Watcher data corruption, pause state on core {} unknown code: {}.\n", virtual_coord_.str(), pause);
            log_warning(tt::LogMetal, "{}: {}", core_str_, error_reason);
            DumpWaypoints(true);
            DumpRingBuffer(true);
            LogRunningKernels();
            // Save the error string for checking later in unit tests.
            MetalContext::instance().watcher_server()->set_exception_message(
                fmt::format("{}: {}", core_str_, error_reason));
            TT_THROW("{}", error_reason);
        }
    }
}

void WatcherDeviceReader::Core::DumpEthLinkStatus() const {
    const debug_eth_link_t* eth_link_status = &mbox_data_->watcher.eth_status;
    if (eth_link_status->link_down == 0) {
        return;
    }
    auto noc0_core = tt::tt_metal::MetalContext::instance()
                         .get_cluster()
                         .get_soc_desc(reader_.device_id)
                         .translate_coord_to(virtual_coord_, CoordSystem::TRANSLATED, CoordSystem::NOC0);
    string error_msg = fmt::format(
        "Watcher detected that active eth link on virtual core {} (noc0 core: {}) went down after training.\n",
        virtual_coord_.str(),
        noc0_core.str());
    log_warning(tt::LogMetal, "{}", error_msg);
    DumpWaypoints();
    DumpRingBuffer();
    LogRunningKernels();
    MetalContext::instance().watcher_server()->set_exception_message(fmt::format("{}: {}", core_str_, error_msg));
    TT_THROW("{}: {}", core_str_, error_msg);
}

void WatcherDeviceReader::Core::DumpRingBuffer(bool to_stdout) const {
    const debug_ring_buf_msg_t* ring_buf_data = &mbox_data_->watcher.debug_ring_buf;
    string out = "";
    if (ring_buf_data->current_ptr != DEBUG_RING_BUFFER_STARTING_INDEX) {
        // Latest written idx is one less than the index read out of L1.
        out += "\n\tdebug_ring_buffer=\n\t[";
        int curr_idx = ring_buf_data->current_ptr;
        for (int count = 1; count <= DEBUG_RING_BUFFER_ELEMENTS; count++) {
            out += fmt::format("0x{:08x},", ring_buf_data->data[curr_idx]);
            if (count % 8 == 0) {
                out += "\n\t ";
            }
            if (curr_idx == 0) {
                if (ring_buf_data->wrapped == 0) {
                    break;  // No wrapping, so no extra data available
                } else {
                    curr_idx = DEBUG_RING_BUFFER_ELEMENTS - 1;  // Loop
                }
            } else {
                curr_idx--;
            }
        }
        // Remove the last comma
        out.pop_back();
        out += "]";
    }

    // This function can either dump to stdout or the log file.
    if (to_stdout) {
        if (!out.empty()) {
            out = string("Last ring buffer status: ") + out;
            log_info(tt::LogMetal, "{}", out);
        }
    } else {
        fprintf(reader_.f, "%s", out.c_str());
    }
}

void WatcherDeviceReader::Core::DumpRunState(uint32_t state) const {
    char code = 'U';
    if (state == RUN_MSG_INIT) {
        code = 'I';
    } else if (state == RUN_MSG_GO) {
        code = 'G';
    } else if (state == RUN_MSG_DONE) {
        code = 'D';
    } else if (state == RUN_MSG_RESET_READ_PTR) {
        code = 'R';
    } else if (state == RUN_SYNC_MSG_LOAD) {
        code = 'L';
    } else if (state == RUN_SYNC_MSG_WAITING_FOR_RESET) {
        code = 'W';
    } else if (state == RUN_SYNC_MSG_INIT_SYNC_REGISTERS) {
        code = 'S';
    }
    if (code == 'U') {
        LogRunningKernels();
        TT_THROW(
            "Watcher data corruption, unexpected run state on core{}: {} (expected {}, {}, {}, {}, or {})",
            virtual_coord_.str(),
            state,
            RUN_MSG_INIT,
            RUN_MSG_GO,
            RUN_MSG_DONE,
            RUN_SYNC_MSG_LOAD,
            RUN_SYNC_MSG_WAITING_FOR_RESET);
    } else {
        fprintf(reader_.f, "%c", code);
    }
}

void WatcherDeviceReader::Core::DumpLaunchMessage() const {
    const subordinate_sync_msg_t* subordinate_sync = &mbox_data_->subordinate_sync;
    fprintf(reader_.f, "rmsg:");
    if (launch_msg_->kernel_config.mode == DISPATCH_MODE_DEV) {
        fprintf(reader_.f, "D");
    } else if (launch_msg_->kernel_config.mode == DISPATCH_MODE_HOST) {
        fprintf(reader_.f, "H");
    } else {
        LogRunningKernels();
        TT_THROW(
            "Watcher data corruption, unexpected launch mode on core {}: {} (expected {} or {})",
            virtual_coord_.str(),
            launch_msg_->kernel_config.mode,
            DISPATCH_MODE_DEV,
            DISPATCH_MODE_HOST);
    }

    if (launch_msg_->kernel_config.brisc_noc_id == 0 || launch_msg_->kernel_config.brisc_noc_id == 1) {
        fprintf(reader_.f, "%d", launch_msg_->kernel_config.brisc_noc_id);
    } else {
        LogRunningKernels();
        TT_THROW(
            "Watcher data corruption, unexpected brisc noc_id on core {}: {} (expected 0 or 1)",
            virtual_coord_.str(),
            launch_msg_->kernel_config.brisc_noc_id);
    }
    if (mbox_data_->go_message_index < go_message_num_entries) {
        DumpRunState(mbox_data_->go_messages[mbox_data_->go_message_index].signal);
    } else {
        LogRunningKernels();
        TT_THROW(
            "Watcher data corruption, unexpected go message index on core {}: {} (expected < {})",
            virtual_coord_.str(),
            mbox_data_->go_message_index,
            go_message_num_entries);
    }

    fprintf(reader_.f, "|");
    if (launch_msg_->kernel_config.enables &
        ~(DISPATCH_CLASS_MASK_TENSIX_ENABLE_DM0 | DISPATCH_CLASS_MASK_TENSIX_ENABLE_DM1 | DISPATCH_CLASS_MASK_ETH_DM0 |
          DISPATCH_CLASS_MASK_ETH_DM1 | DISPATCH_CLASS_MASK_TENSIX_ENABLE_COMPUTE)) {
        LogRunningKernels();
        TT_THROW(
            "Watcher data corruption, unexpected kernel enable on core {}: {} (expected only low bits set)",
            virtual_coord_.str(),
            launch_msg_->kernel_config.enables);
    }

    // TODO(#17275): Generalize and pull risc data out of HAL
    if (programmable_core_type_ == HalProgrammableCoreType::TENSIX) {
        if (launch_msg_->kernel_config.enables & DISPATCH_CLASS_MASK_TENSIX_ENABLE_DM0) {
            fprintf(reader_.f, "B");
        } else {
            fprintf(reader_.f, "b");
        }

        if (launch_msg_->kernel_config.enables & DISPATCH_CLASS_MASK_TENSIX_ENABLE_DM1) {
            fprintf(reader_.f, "N");
        } else {
            fprintf(reader_.f, "n");
        }

        if (launch_msg_->kernel_config.enables & DISPATCH_CLASS_MASK_TENSIX_ENABLE_COMPUTE) {
            fprintf(reader_.f, "T");
        } else {
            fprintf(reader_.f, "t");
        }
    } else {
        if (launch_msg_->kernel_config.enables & DISPATCH_CLASS_MASK_ETH_DM0) {
            fprintf(reader_.f, "E");
        } else {
            fprintf(reader_.f, "e");
        }
        if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == ARCH::BLACKHOLE) {
            if (launch_msg_->kernel_config.enables & DISPATCH_CLASS_MASK_ETH_DM1) {
                fprintf(reader_.f, "E");
            } else {
                fprintf(reader_.f, "e");
            }
        }
    }

    fprintf(reader_.f, " h_id:%3d ", launch_msg_->kernel_config.host_assigned_id);

    if (programmable_core_type_ == HalProgrammableCoreType::TENSIX) {
        fprintf(reader_.f, "smsg:");
        DumpRunState(subordinate_sync->dm1);
        DumpRunState(subordinate_sync->trisc0);
        DumpRunState(subordinate_sync->trisc1);
        DumpRunState(subordinate_sync->trisc2);
        fprintf(reader_.f, " ");
    } else if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == ARCH::BLACKHOLE) {
        fprintf(reader_.f, "smsg:");
        DumpRunState(subordinate_sync->dm1);
        fprintf(reader_.f, " ");
    }
}

void WatcherDeviceReader::Core::DumpWaypoints(bool to_stdout) const {
    const debug_waypoint_msg_t* debug_waypoint = mbox_data_->watcher.debug_waypoint;
    string out;

    for (int cpu = 0; cpu < MAX_RISCV_PER_CORE; cpu++) {
        string risc_status;
        for (int byte = 0; byte < num_waypoint_bytes_per_riscv; byte++) {
            char v = ((char*)&debug_waypoint[cpu])[byte];
            if (v == 0) {
                break;
            }
            if (isprint(v)) {
                risc_status += v;
            } else {
                LogRunningKernels();
                TT_THROW(
                    "Watcher data corrupted, unexpected debug status on core {}, unprintable character {}",
                    virtual_coord_.str(),
                    (int)v);
            }
        }
        // Pad risc status to 4 chars for alignment
        string pad(4 - risc_status.length(), ' ');
        out += (pad + risc_status);
        if (cpu != MAX_RISCV_PER_CORE - 1) {
            out += ',';
        }
    }

    out += " ";

    // This function can either log the waypoint to the log or stdout.
    if (to_stdout) {
        out = string("Last waypoint: ") + out;
        log_info(tt::LogMetal, "{}", out);
    } else {
        fprintf(reader_.f, "%s ", out.c_str());
    }
}

void WatcherDeviceReader::Core::DumpSyncRegs() const {
    // Read back all of the stream state, most of it is unused
    std::vector<uint32_t> data;
    for (uint32_t operand = 0; operand < NUM_CIRCULAR_BUFFERS; operand++) {
        // XXXX TODO(PGK) get this from device
        const uint32_t OPERAND_START_STREAM = 8;
        uint32_t base = NOC_OVERLAY_START_ADDR + (OPERAND_START_STREAM + operand) * NOC_STREAM_REG_SPACE_SIZE;

        uint32_t rcvd_addr = base + STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX * sizeof(uint32_t);
        data = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            reader_.device_id, virtual_coord_, rcvd_addr, sizeof(uint32_t));
        uint32_t rcvd = data[0];

        uint32_t ackd_addr = base + STREAM_REMOTE_DEST_BUF_START_REG_INDEX * sizeof(uint32_t);
        data = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            reader_.device_id, virtual_coord_, ackd_addr, sizeof(uint32_t));
        uint32_t ackd = data[0];

        if (rcvd != ackd) {
            fprintf(reader_.f, "cb[%d](rcv %d!=ack %d) ", operand, rcvd, ackd);
        }
    }
}

void WatcherDeviceReader::Core::DumpStackUsage() const {
    const debug_stack_usage_t* stack_usage_mbox = &mbox_data_->watcher.stack_usage;
    const auto& hal = MetalContext::instance().hal();
    auto num_processors = hal.get_num_risc_processors(programmable_core_type_);
    for (uint32_t processor_index = 0; processor_index < num_processors; processor_index++) {
        const auto& usage = stack_usage_mbox->cpu[processor_index];
        if (usage.min_free) {
            auto [processor_class, processor_type] =
                hal.get_processor_class_and_type_from_index(programmable_core_type_, processor_index);
            HalProcessorIdentifier processor = {programmable_core_type_, processor_class, processor_type};
            auto& slot = dump_data_.highest_stack_usage[processor];
            if (usage.min_free <= slot.stack_free) {
                slot = {virtual_coord_, usage.min_free - 1, usage.watcher_kernel_id};
            }
        }
    }
}

void WatcherDeviceReader::Core::ValidateKernelIDs() const {
    if (programmable_core_type_ == HalProgrammableCoreType::ACTIVE_ETH ||
        programmable_core_type_ == HalProgrammableCoreType::IDLE_ETH) {
        if (launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM0] >= reader_.kernel_names.size()) {
            uint16_t watcher_kernel_id = launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM0];
            TT_THROW(
                "Watcher data corruption, unexpected erisc0 kernel id on Device {} core {}: {} (last valid {})",
                reader_.device_id,
                virtual_coord_.str(),
                watcher_kernel_id,
                reader_.kernel_names.size());
        }
        dump_data_.used_kernel_names[launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM0]] = true;

        if (launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM1] >= reader_.kernel_names.size()) {
            uint16_t watcher_kernel_id = launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM1];
            TT_THROW(
                "Watcher data corruption, unexpected erisc1 kernel id on Device {} core {}: {} (last valid {})",
                reader_.device_id,
                virtual_coord_.str(),
                watcher_kernel_id,
                reader_.kernel_names.size());
        }
        dump_data_.used_kernel_names[launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM1]] = true;
    } else {
        if (launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM0] >= reader_.kernel_names.size()) {
            uint16_t watcher_kernel_id = launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM0];
            TT_THROW(
                "Watcher data corruption, unexpected brisc kernel id on Device {} core {}: {} (last valid {})",
                reader_.device_id,
                virtual_coord_.str(),
                watcher_kernel_id,
                reader_.kernel_names.size());
        }
        dump_data_.used_kernel_names[launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM0]] = true;

        if (launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM1] >= reader_.kernel_names.size()) {
            uint16_t watcher_kernel_id = launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM1];
            TT_THROW(
                "Watcher data corruption, unexpected ncrisc kernel id on Device {} core {}: {} (last valid {})",
                reader_.device_id,
                virtual_coord_.str(),
                watcher_kernel_id,
                reader_.kernel_names.size());
        }
        dump_data_.used_kernel_names[launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM1]] = true;

        if (launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE] >=
            reader_.kernel_names.size()) {
            uint16_t watcher_kernel_id = launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE];
            TT_THROW(
                "Watcher data corruption, unexpected trisc kernel id on Device {} core {}: {} (last valid {})",
                reader_.device_id,
                virtual_coord_.str(),
                watcher_kernel_id,
                reader_.kernel_names.size());
        }
        dump_data_.used_kernel_names[launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE]] =
            true;
    }
}

void WatcherDeviceReader::Core::LogRunningKernels() const {
    log_info(tt::LogMetal, "While running kernels:");
    if (programmable_core_type_ == HalProgrammableCoreType::ACTIVE_ETH ||
        programmable_core_type_ == HalProgrammableCoreType::IDLE_ETH) {
        log_info(
            tt::LogMetal,
            " erisc : {}",
            reader_.kernel_names[launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM0]]);
        if (reader_.num_erisc_cores > 1) {
            log_info(
                tt::LogMetal,
                " erisc1 : {}",
                reader_.kernel_names[launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM1]]);
        }
    } else {
        log_info(
            tt::LogMetal,
            " brisc : {}",
            reader_.kernel_names[launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM0]]);
        log_info(
            tt::LogMetal,
            " ncrisc: {}",
            reader_.kernel_names[launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM1]]);
        log_info(
            tt::LogMetal,
            " triscs: {}",
            reader_.kernel_names[launch_msg_->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE]]);
    }
}

const std::string& WatcherDeviceReader::Core::GetKernelName(uint32_t processor_index) const {
    uint32_t dispatch_class;
    // TODO: Revisit when dispatch class is removed, then this can be made arch-independent
    // (just use processor index to index watcher_kernel_ids).
    auto [processor_class, processor_type] = MetalContext::instance().hal().get_processor_class_and_type_from_index(
        programmable_core_type_, processor_index);
    switch (programmable_core_type_) {
        case HalProgrammableCoreType::ACTIVE_ETH:
        case HalProgrammableCoreType::IDLE_ETH: dispatch_class = processor_type; break;
        case HalProgrammableCoreType::TENSIX: {
            dispatch_class =
                processor_class == HalProcessorClassType::DM ? processor_type : DISPATCH_CLASS_TENSIX_COMPUTE;
            break;
        }
        default: TT_THROW("Unexpected programmable core type");
    }
    TT_FATAL(
        dispatch_class < DISPATCH_CLASS_MAX,
        "invalid dispatch class for processor {} on {} {}",
        processor_index,
        programmable_core_type_,
        processor_class);
    return reader_.kernel_names[launch_msg_->kernel_config.watcher_kernel_ids[dispatch_class]];
}

}  // namespace tt::tt_metal
