// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cctype>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt_stl/assert.hpp>
#include <circular_buffer_constants.h>  // For NUM_CIRCULAR_BUFFERS
#include <core_coord.hpp>
#include <fmt/base.h>
#include <fmt/ranges.h>
#include "llrt/metal_soc_descriptor.hpp"
#include <tt-logger/tt-logger.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/arch.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/types/xy_pair.hpp>

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "core_descriptor.hpp"
#include "llrt.hpp"
#include "llrt/hal.hpp"
#include "dispatch_core_common.hpp"
#include "hal_types.hpp"
#include "api/debug/ring_buffer.h"
#include "impl/context/metal_context.hpp"
#include "watcher_device_reader.hpp"
#include <impl/debug/watcher_server.hpp>
#include <llrt/tt_cluster.hpp>

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
tt::CoreType core_type_from_virtual_core(tt::ChipId device_id, const CoreCoord& virtual_coord) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().is_worker_core(virtual_coord, device_id)) {
        return tt::CoreType::WORKER;
    }
    if (tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_core(virtual_coord, device_id)) {
        return tt::CoreType::ETH;
    }

    const metal_SocDescriptor& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id);

    const std::vector<tt::umd::CoreCoord>& translated_dram_cores =
        soc_desc.get_cores(tt::CoreType::DRAM, tt::CoordSystem::TRANSLATED);
    if (std::find(translated_dram_cores.begin(), translated_dram_cores.end(), virtual_coord) !=
        translated_dram_cores.end()) {
        return tt::CoreType::DRAM;
    }

    tt::CoreType core_type =
        soc_desc.translate_coord_to(virtual_coord, tt::CoordSystem::NOC0, tt::CoordSystem::NOC0).core_type;
    if (core_type == tt::CoreType::TENSIX) {
        core_type = tt::CoreType::WORKER;
    }
    return core_type;
}

// Helper function to convert noc coord -> virtual coord. TODO: Remove this once we fix code types.
CoreCoord virtual_noc_coordinate(tt::ChipId device_id, uint8_t noc_index, CoreCoord coord) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == tt::ARCH::BLACKHOLE) {
        return coord;
    }
    auto grid_size = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id).grid_size;
    if (coord.x >= grid_size.x || coord.y >= grid_size.y) {
        // Coordinate already in virtual space: NOC0 and NOC1 are the same
        return coord;
    }  // Coordinate passed in can be NOC0 or NOC1. The noc_index corresponds to
    // the system this coordinate belongs to.
    // Use this to convert to NOC0 coordinates and then derive Virtual Coords from it.
    CoreCoord physical_coord = {
        MetalContext::instance().hal().noc_coordinate(noc_index, grid_size.x, coord.x),
        MetalContext::instance().hal().noc_coordinate(noc_index, grid_size.y, coord.y)};
    return tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_physical_coordinates(
        device_id, physical_coord);
}

// Helper function to get string rep of noc target.
string get_noc_target_str(
    tt::ChipId device_id,
    HalProgrammableCoreType programmable_core_type,
    int noc,
    dev_msgs::debug_sanitize_addr_msg_t::ConstView san) {
    auto get_core_and_mem_type = [](tt::ChipId device_id, CoreCoord& noc_coord, int noc) -> std::pair<string, string> {
        // Get the virtual coord from the noc coord
        CoreCoord virtual_core = virtual_noc_coordinate(device_id, noc, noc_coord);
        tt::CoreType core_type;
        try {
            core_type = core_type_from_virtual_core(device_id, virtual_core);
        } catch (std::runtime_error& e) {
            // We may not be able to get a core type if the virtual coords are bad.
            return {"Unknown", ""};
        }
        switch (core_type) {
            case tt::CoreType::DRAM: return {"DRAM", "DRAM"};
            case tt::CoreType::ETH: return {"Ethernet", "L1"};
            case tt::CoreType::PCIE: return {"PCIe", "PCIE"};
            case tt::CoreType::WORKER: return {"Tensix", "L1"};
            default: return {"Unknown", ""};
        }
    };
    string out = fmt::format(
        "{} using noc{} tried to {} {} {} bytes {} local L1[{:#08x}] {} ",
        get_riscv_name(programmable_core_type, san.which_risc()),
        noc,
        san.is_multicast() ? "multicast" : "unicast",
        san.is_write() ? "write" : "read",
        san.len(),
        san.is_write() ? "from" : "to",
        san.l1_addr(),
        san.is_write() ? "to" : "from");

    if (san.is_multicast()) {
        CoreCoord target_virtual_noc_core_start = {
            NOC_MCAST_ADDR_START_X(san.noc_addr()), NOC_MCAST_ADDR_START_Y(san.noc_addr())};
        CoreCoord target_virtual_noc_core_end = {
            NOC_MCAST_ADDR_END_X(san.noc_addr()), NOC_MCAST_ADDR_END_Y(san.noc_addr())};
        auto type_and_mem = get_core_and_mem_type(device_id, target_virtual_noc_core_start, noc);
        out += fmt::format(
            "{} core range w/ virtual coords {}-{} {}",
            type_and_mem.first,
            target_virtual_noc_core_start.str(),
            target_virtual_noc_core_end.str(),
            type_and_mem.second);
    } else {
        CoreCoord target_virtual_noc_core = {NOC_UNICAST_ADDR_X(san.noc_addr()), NOC_UNICAST_ADDR_Y(san.noc_addr())};
        auto type_and_mem = get_core_and_mem_type(device_id, target_virtual_noc_core, noc);
        out += fmt::format(
            "{} core w/ virtual coords {} {}", type_and_mem.first, target_virtual_noc_core.str(), type_and_mem.second);
    }

    out += fmt::format("[addr=0x{:08x}]", NOC_LOCAL_ADDR(san.noc_addr()));
    return out;
}

string get_l1_target_str(
    HalProgrammableCoreType programmable_core_type, dev_msgs::debug_sanitize_addr_msg_t::ConstView san) {
    string out = fmt::format(
        "{} core overflowed L1 with access to {:#x} of length {}",
        get_riscv_name(programmable_core_type, san.which_risc()),
        san.l1_addr(),
        san.len());
    return out;
}

dev_msgs::launch_msg_t::ConstView get_valid_launch_message(dev_msgs::mailboxes_t::ConstView mbox_data) {
    uint32_t launch_msg_read_ptr = mbox_data.launch_msg_rd_ptr();
    if (mbox_data.launch()[launch_msg_read_ptr].kernel_config().enables() == 0) {
        launch_msg_read_ptr = (launch_msg_read_ptr - 1 + dev_msgs::launch_msg_buffer_num_entries) %
                              dev_msgs::launch_msg_buffer_num_entries;
    }
    return mbox_data.launch()[launch_msg_read_ptr];
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
    std::vector<std::byte> l1_read_buf_;
    dev_msgs::mailboxes_t::ConstView mbox_data_;
    dev_msgs::launch_msg_t::ConstView launch_msg_;
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

    Core(
        CoreCoord virtual_coord,
        HalProgrammableCoreType programmable_core_type,
        std::string core_str,
        std::vector<std::byte> l1_read_buf,
        dev_msgs::Factory dev_msgs_factory,
        const WatcherDeviceReader& reader,
        DumpData& dump_data) :
        virtual_coord_(virtual_coord),
        programmable_core_type_(programmable_core_type),
        core_str_(std::move(core_str)),
        l1_read_buf_(std::move(l1_read_buf)),
        mbox_data_(dev_msgs_factory.create_view<dev_msgs::mailboxes_t>(l1_read_buf_.data())),
        launch_msg_(get_valid_launch_message(mbox_data_)),
        reader_(reader),
        dump_data_(dump_data) {}

public:
    static Core Create(
        CoreCoord logical_coord,
        HalProgrammableCoreType programmable_core_type,
        const WatcherDeviceReader& reader,
        DumpData& dump_data);

    void Dump() const;
};

WatcherDeviceReader::WatcherDeviceReader(FILE* f, ChipId device_id, const std::vector<string>& kernel_names) :
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

    // Dump worker cores
    CoreCoord grid_size =
        tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord coord = {x, y};
            Core::Create(coord, HalProgrammableCoreType::TENSIX, *this, dump_data).Dump();
        }
    }

    // Dump eth cores
    for (const CoreCoord& eth_core :
         tt::tt_metal::MetalContext::instance().get_control_plane().get_active_ethernet_cores(device_id)) {
        Core::Create(eth_core, HalProgrammableCoreType::ACTIVE_ETH, *this, dump_data).Dump();
    }
    for (const CoreCoord& eth_core :
         tt::tt_metal::MetalContext::instance().get_control_plane().get_inactive_ethernet_cores(device_id)) {
        Core::Create(eth_core, HalProgrammableCoreType::IDLE_ETH, *this, dump_data).Dump();
    }

    for (auto k_id : dump_data.used_kernel_names) {
        fprintf(f, "k_id[%3d]: %s\n", k_id.first, kernel_names[k_id.first].c_str());
    }

    const auto& hal = MetalContext::instance().hal();
    // Print stack usage report for this device/dump
    if (!dump_data.highest_stack_usage.empty()) {
        fprintf(f, "Stack usage summary:");
        for (auto& [processor, info] : dump_data.highest_stack_usage) {
            const auto* processor_name = get_riscv_name(
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
        for (const auto& [virtual_core, processor_index] : dump_data.paused_cores) {
            paused_cores_str += fmt::format(
                "{}:{}, ",
                virtual_core.str(),
                get_riscv_name(llrt::get_core_type(device_id, virtual_core), processor_index));
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
        for (const auto& [virtual_core, processor_index] : dump_data.paused_cores) {
            auto programmable_core_type = llrt::get_core_type(device_id, virtual_core);
            auto dev_msgs_factory = hal.get_dev_msgs_factory(programmable_core_type);
            auto pause_data = dev_msgs_factory.create<dev_msgs::debug_pause_msg_t>();
            uint64_t addr =
                hal.get_dev_addr(programmable_core_type, HalL1MemAddrType::WATCHER) +
                dev_msgs_factory.offset_of<dev_msgs::watcher_msg_t>(dev_msgs::watcher_msg_t::Field::pause_status);

            // Clear only the one flag that we saved, in case another one was raised on device
            tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                pause_data.data(), pause_data.size(), {static_cast<size_t>(device_id), virtual_core}, addr);
            pause_data.view().flags()[processor_index] = 0;
            tt::tt_metal::MetalContext::instance().get_cluster().write_core(
                pause_data.data(), pause_data.size(), {static_cast<size_t>(device_id), virtual_core}, addr);
        }
    }
    fflush(f);
}

WatcherDeviceReader::Core WatcherDeviceReader::Core::Create(
    CoreCoord logical_coord,
    HalProgrammableCoreType programmable_core_type,
    const WatcherDeviceReader& reader,
    DumpData& dump_data) {
    const auto& rtoptions = tt_metal::MetalContext::instance().rtoptions();
    const auto& hal = MetalContext::instance().hal();
    CoreType core_type = hal.get_core_type(hal.get_programmable_core_type_index(programmable_core_type));
    auto virtual_coord =
        tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
            reader.device_id, logical_coord, core_type);

    // Print device id, core coords (logical)
    string core_type_str;
    if (programmable_core_type == HalProgrammableCoreType::ACTIVE_ETH) {
        core_type_str = "acteth";
    } else if (programmable_core_type == HalProgrammableCoreType::IDLE_ETH) {
        core_type_str = "idleth";
    } else {
        core_type_str = "worker";
    }
    string core_coord_str = fmt::format(
        "core(x={:2},y={:2}) virtual(x={:2},y={:2})",
        logical_coord.x,
        logical_coord.y,
        virtual_coord.x,
        virtual_coord.y);
    if (rtoptions.get_watcher_phys_coords()) {
        CoreCoord phys_core =
            tt::tt_metal::MetalContext::instance().get_cluster().get_physical_coordinate_from_logical_coordinates(
                reader.device_id, logical_coord, core_type, true);
        core_coord_str += fmt::format(" phys(x={:2},y={:2})", phys_core.x, phys_core.y);
    }
    auto core_str = fmt::format("Device {} {} {}", reader.device_id, core_type_str, core_coord_str);
    fprintf(reader.f, "%s: ", core_str.c_str());

    uint64_t mailbox_addr =
        MetalContext::instance().hal().get_dev_addr(programmable_core_type, HalL1MemAddrType::MAILBOX);

    auto dev_msgs_factory = hal.get_dev_msgs_factory(programmable_core_type);
    uint32_t mailbox_read_size =
        dev_msgs_factory.offset_of<dev_msgs::mailboxes_t>(dev_msgs::mailboxes_t::Field::watcher) +
        dev_msgs_factory.size_of<dev_msgs::watcher_msg_t>();
    // Watcher only reads the mailbox up to the end of the watcher struct.
    // Should be ok if we never access past that.
    std::vector<std::byte> l1_read_buf(mailbox_read_size);
    tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        l1_read_buf.data(), l1_read_buf.size(), {static_cast<size_t>(reader.device_id), virtual_coord}, mailbox_addr);
    return Core(
        virtual_coord,
        programmable_core_type,
        std::move(core_str),
        std::move(l1_read_buf),
        dev_msgs_factory,
        reader,
        dump_data);
}

void WatcherDeviceReader::Core::Dump() const {
    const auto& rtoptions = tt_metal::MetalContext::instance().rtoptions();
    bool is_eth_core =
        (programmable_core_type_ == HalProgrammableCoreType::ACTIVE_ETH ||
         programmable_core_type_ == HalProgrammableCoreType::IDLE_ETH);

    ValidateKernelIDs();

    // Whether or not watcher data is available depends on a flag set on the device.
    if (mbox_data_.watcher().enable() != dev_msgs::WatcherEnabled and
        mbox_data_.watcher().enable() != dev_msgs::WatcherDisabled) {
        TT_THROW(
            "Watcher read invalid watcher.enable on {}. Read {}, valid values are {} and {}.",
            core_str_,
            mbox_data_.watcher().enable(),
            dev_msgs::WatcherEnabled,
            dev_msgs::WatcherDisabled);
    }
    bool enabled = (mbox_data_.watcher().enable() == dev_msgs::WatcherEnabled);

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

    auto kernel_config = launch_msg_.kernel_config();
    fprintf(reader_.f, "k_ids:");
    auto num_processors = MetalContext::instance().hal().get_num_risc_processors(programmable_core_type_);
    for (size_t i = 0; i < num_processors; i++) {
        const char* separator = (i > 0) ? "|" : "";
        fprintf(reader_.f, "%s%3d", separator, kernel_config.watcher_kernel_ids()[i]);
    }
    if (!is_eth_core && rtoptions.get_watcher_text_start()) {
        uint32_t kernel_config_base = kernel_config.kernel_config_base()[0];
        fprintf(reader_.f, " text_start:");
        for (size_t i = 0; i < num_processors; i++) {
            const char* separator = (i > 0) ? "|" : "";
            fprintf(reader_.f, "%s0x%x", separator, kernel_config_base + kernel_config.kernel_text_offset()[i]);
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
    auto san = mbox_data_.watcher().sanitize()[noc];
    string error_msg;

    switch (san.return_code()) {
        case dev_msgs::DebugSanitizeOK:
            if (san.noc_addr() != DEBUG_SANITIZE_SENTINEL_OK_64 || san.l1_addr() != DEBUG_SANITIZE_SENTINEL_OK_32 ||
                san.len() != DEBUG_SANITIZE_SENTINEL_OK_32 || san.which_risc() != DEBUG_SANITIZE_SENTINEL_OK_16 ||
                san.is_multicast() != DEBUG_SANITIZE_SENTINEL_OK_8 || san.is_write() != DEBUG_SANITIZE_SENTINEL_OK_8 ||
                san.is_target() != DEBUG_SANITIZE_SENTINEL_OK_8) {
                error_msg = fmt::format(
                    "Watcher unexpected noc debug state on core {}, reported valid got noc{}{{0x{:08x}, {} }}",
                    virtual_coord_.str(),
                    san.which_risc(),
                    san.noc_addr(),
                    san.len());
                error_msg += " (corrupted noc sanitization state - sanitization memory overwritten)";
            }
            break;
        case dev_msgs::DebugSanitizeNocAddrUnderflow:
            error_msg = get_noc_target_str(reader_.device_id, programmable_core_type_, noc, san);
            error_msg += string(san.is_target() ? " (NOC target" : " (Local L1") + " address underflow).";
            break;
        case dev_msgs::DebugSanitizeNocAddrOverflow:
            error_msg = get_noc_target_str(reader_.device_id, programmable_core_type_, noc, san);
            error_msg += string(san.is_target() ? " (NOC target" : " (Local L1") + " address overflow).";
            break;
        case dev_msgs::DebugSanitizeNocAddrZeroLength:
            error_msg = get_noc_target_str(reader_.device_id, programmable_core_type_, noc, san);
            error_msg += " (zero length transaction).";
            break;
        case dev_msgs::DebugSanitizeNocTargetInvalidXY:
            error_msg = get_noc_target_str(reader_.device_id, programmable_core_type_, noc, san);
            error_msg += " (NOC target address did not map to any known Tensix/Ethernet/DRAM/PCIE core).";
            break;
        case dev_msgs::DebugSanitizeNocMulticastNonWorker:
            error_msg = get_noc_target_str(reader_.device_id, programmable_core_type_, noc, san);
            error_msg += " (multicast to non-worker core).";
            break;
        case dev_msgs::DebugSanitizeNocMulticastInvalidRange:
            error_msg = get_noc_target_str(reader_.device_id, programmable_core_type_, noc, san);
            error_msg += " (multicast invalid range).";
            break;
        case dev_msgs::DebugSanitizeNocAlignment:
            error_msg = get_noc_target_str(reader_.device_id, programmable_core_type_, noc, san);
            error_msg += " (invalid address alignment in NOC transaction).";
            break;
        case dev_msgs::DebugSanitizeNocMixedVirtualandPhysical:
            error_msg = get_noc_target_str(reader_.device_id, programmable_core_type_, noc, san);
            error_msg += " (mixing virtual and virtual coordinates in Mcast).";
            break;
        case dev_msgs::DebugSanitizeInlineWriteDramUnsupported:
            error_msg = get_noc_target_str(reader_.device_id, programmable_core_type_, noc, san);
            error_msg += " (inline dw writes do not support DRAM destination addresses).";
            break;
        case dev_msgs::DebugSanitizeNocAddrMailbox:
            error_msg = get_noc_target_str(reader_.device_id, programmable_core_type_, noc, san);
            error_msg += string(san.is_target() ? " (NOC target" : " (Local L1") + " overwrites mailboxes).";
            break;
        case dev_msgs::DebugSanitizeNocLinkedTransactionViolation:
            error_msg = get_noc_target_str(reader_.device_id, programmable_core_type_, noc, san);
            error_msg += fmt::format(" (submitting a non-mcast transaction when there's a linked transaction).");
            break;
        case dev_msgs::DebugSanitizeL1AddrOverflow:
            error_msg = get_l1_target_str(programmable_core_type_, san);
            error_msg += " (read or write past the end of local memory).";
            break;
        case dev_msgs::DebugSanitizeEthDestL1AddrOverflow:
            error_msg = get_l1_target_str(programmable_core_type_, san);
            error_msg += " (ethernet send to core with L1 destination overflow).";
            break;
        case dev_msgs::DebugSanitizeEthSrcL1AddrOverflow:
            error_msg = get_l1_target_str(programmable_core_type_, san);
            error_msg += " (ethernet send with L1 source overflow).";
            break;
        default:
            error_msg = fmt::format(
                "Watcher unexpected data corruption, noc debug state on core {}, unknown failure code: {}",
                virtual_coord_.str(),
                san.return_code());
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
    auto assert_status = mbox_data_.watcher().assert_status();
    if (assert_status.tripped() == dev_msgs::DebugAssertOK) {
        if (assert_status.line_num() != DEBUG_SANITIZE_SENTINEL_OK_16 ||
            assert_status.which() != DEBUG_SANITIZE_SENTINEL_OK_8) {
            TT_THROW(
                "Watcher unexpected assert state on core {}, reported OK but got processor {}, line {}.",
                virtual_coord_.str(),
                assert_status.which(),
                assert_status.line_num());
        }
        return;  // no assert tripped, nothing to do
    }
    std::string error_msg =
        fmt::format("{}: {} ", core_str_, get_riscv_name(programmable_core_type_, assert_status.which()));
    switch (assert_status.tripped()) {
        case dev_msgs::DebugAssertTripped: {
            error_msg += fmt::format("tripped an assert on line {}.", assert_status.line_num());
            // TODO: Get rid of this once #6098 is implemented.
            error_msg +=
                " Note that file name reporting is not yet implemented, and the reported line number for the assert "
                "may be from a different file.";
            break;
        }
        case dev_msgs::DebugAssertNCriscNOCReadsFlushedTripped: {
            error_msg +=
                "detected an inter-kernel data race due to kernel completing with pending NOC transactions (missing "
                "NOC reads flushed barrier).";
            break;
        }
        case dev_msgs::DebugAssertNCriscNOCNonpostedWritesSentTripped: {
            error_msg +=
                "detected an inter-kernel data race due to kernel completing with pending NOC transactions (missing "
                "NOC non-posted writes sent barrier).";
            break;
        }
        case dev_msgs::DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped: {
            error_msg +=
                "detected an inter-kernel data race due to kernel completing with pending NOC transactions (missing "
                "NOC non-posted atomics flushed barrier).";
            break;
        }
        case dev_msgs::DebugAssertNCriscNOCPostedWritesSentTripped: {
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
                assert_status.tripped());
    }
    error_msg += fmt::format(" Current kernel: {}.", GetKernelName(assert_status.which()));
    log_warning(tt::LogMetal, "Watcher stopped the device due to tripped assert, see watcher log for more details");
    log_warning(tt::LogMetal, "{}", error_msg);
    DumpWaypoints(true);
    DumpRingBuffer(true);
    LogRunningKernels();
    MetalContext::instance().watcher_server()->set_exception_message(error_msg);
    TT_THROW("Watcher detected tripped assert and stopped device.");
}

void WatcherDeviceReader::Core::DumpPauseStatus() const {
    auto pause_status = mbox_data_.watcher().pause_status();
    const auto& hal = MetalContext::instance().hal();
    // Just record which cores are paused, printing handled at the end.
    auto num_processors = hal.get_num_risc_processors(programmable_core_type_);
    for (uint32_t processor_index = 0; processor_index < num_processors; processor_index++) {
        auto pause = pause_status.flags()[processor_index];
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
    auto eth_link_status = mbox_data_.watcher().eth_status();
    if (eth_link_status.link_down() == 0) {
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
    auto ring_buf_data = mbox_data_.watcher().debug_ring_buf();
    string out;
    if (ring_buf_data.current_ptr() != DEBUG_RING_BUFFER_STARTING_INDEX) {
        // Latest written idx is one less than the index read out of L1.
        out += "\n\tdebug_ring_buffer=\n\t[";
        int curr_idx = ring_buf_data.current_ptr();
        size_t ring_buffer_elements = ring_buf_data.data().size();
        for (int count = 1; count <= ring_buffer_elements; count++) {
            out += fmt::format("0x{:08x},", ring_buf_data.data()[curr_idx]);
            if (count % 8 == 0) {
                out += "\n\t ";
            }
            if (curr_idx == 0) {
                if (ring_buf_data.wrapped() == 0) {
                    break;  // No wrapping, so no extra data available
                }
                curr_idx = ring_buffer_elements - 1;  // Loop

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
    if (state == dev_msgs::RUN_MSG_INIT) {
        code = 'I';
    } else if (state == dev_msgs::RUN_MSG_GO) {
        code = 'G';
    } else if (state == dev_msgs::RUN_MSG_DONE) {
        code = 'D';
    } else if (state == dev_msgs::RUN_MSG_RESET_READ_PTR) {
        code = 'R';
    } else if (state == dev_msgs::RUN_SYNC_MSG_LOAD) {
        code = 'L';
    } else if (state == dev_msgs::RUN_SYNC_MSG_WAITING_FOR_RESET) {
        code = 'W';
    } else if (state == dev_msgs::RUN_SYNC_MSG_INIT_SYNC_REGISTERS) {
        code = 'S';
    }
    if (code == 'U') {
        LogRunningKernels();
        TT_THROW(
            "Watcher data corruption, unexpected run state on core{}: {} (expected {}, {}, {}, {}, or {})",
            virtual_coord_.str(),
            state,
            dev_msgs::RUN_MSG_INIT,
            dev_msgs::RUN_MSG_GO,
            dev_msgs::RUN_MSG_DONE,
            dev_msgs::RUN_SYNC_MSG_LOAD,
            dev_msgs::RUN_SYNC_MSG_WAITING_FOR_RESET);
    } else {
        fprintf(reader_.f, "%c", code);
    }
}

void WatcherDeviceReader::Core::DumpLaunchMessage() const {
    auto subordinate_sync = mbox_data_.subordinate_sync();
    const auto& hal = MetalContext::instance().hal();
    fprintf(reader_.f, "rmsg:");
    if (launch_msg_.kernel_config().mode() == dev_msgs::DISPATCH_MODE_DEV) {
        fprintf(reader_.f, "D");
    } else if (launch_msg_.kernel_config().mode() == dev_msgs::DISPATCH_MODE_HOST) {
        fprintf(reader_.f, "H");
    } else {
        LogRunningKernels();
        TT_THROW(
            "Watcher data corruption, unexpected launch mode on core {}: {} (expected {} or {})",
            virtual_coord_.str(),
            launch_msg_.kernel_config().mode(),
            dev_msgs::DISPATCH_MODE_DEV,
            dev_msgs::DISPATCH_MODE_HOST);
    }

    if (launch_msg_.kernel_config().brisc_noc_id() == 0 || launch_msg_.kernel_config().brisc_noc_id() == 1) {
        fprintf(reader_.f, "%d", launch_msg_.kernel_config().brisc_noc_id());
    } else {
        LogRunningKernels();
        TT_THROW(
            "Watcher data corruption, unexpected brisc noc_id on core {}: {} (expected 0 or 1)",
            virtual_coord_.str(),
            launch_msg_.kernel_config().brisc_noc_id());
    }
    if (mbox_data_.go_message_index() < dev_msgs::go_message_num_entries) {
        DumpRunState(mbox_data_.go_messages()[mbox_data_.go_message_index()].signal());
    } else {
        LogRunningKernels();
        TT_THROW(
            "Watcher data corruption, unexpected go message index on core {}: {} (expected < {})",
            virtual_coord_.str(),
            mbox_data_.go_message_index(),
            dev_msgs::go_message_num_entries);
    }

    fprintf(reader_.f, "|");
    auto num_processors = hal.get_num_risc_processors(programmable_core_type_);
    uint32_t all_enable_mask = (1u << num_processors) - 1;
    uint32_t enables = launch_msg_.kernel_config().enables();
    if (enables & ~all_enable_mask) {
        LogRunningKernels();
        TT_THROW(
            "Watcher data corruption, unexpected kernel enable on core {}: {} (expected only low bits set)",
            virtual_coord_.str(),
            enables);
    }

    // TODO(#17275): Generalize and pull risc data out of HAL
    std::string_view symbols;
    if (programmable_core_type_ == HalProgrammableCoreType::TENSIX) {
        symbols = "BNT";
    } else {
        if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == ARCH::BLACKHOLE) {
            symbols = "EE";
        } else {
            symbols = "E";
        }
    }
    for (size_t i = 0; i < symbols.size(); i++) {
        char c = symbols[i];
        if ((enables & (1u << i)) == 0) {
            c = tolower(c);
        }
        fputc(c, reader_.f);
    }

    fprintf(reader_.f, " h_id:%3d ", launch_msg_.kernel_config().host_assigned_id());

    if (programmable_core_type_ == HalProgrammableCoreType::TENSIX) {
        fprintf(reader_.f, "smsg:");
        // TODO once we have triscs running on Quasar, just loop over all RISC cores
        DumpRunState(subordinate_sync.map()[0]);
        if (tt::tt_metal::MetalContext::instance().get_cluster().arch() != ARCH::QUASAR) {
            DumpRunState(subordinate_sync.map()[1]);
            DumpRunState(subordinate_sync.map()[2]);
            DumpRunState(subordinate_sync.map()[3]);
        }
        fprintf(reader_.f, " ");
    } else if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == ARCH::BLACKHOLE) {
        fprintf(reader_.f, "smsg:");
        DumpRunState(subordinate_sync.map()[0]);
        fprintf(reader_.f, " ");
    }
}

void WatcherDeviceReader::Core::DumpWaypoints(bool to_stdout) const {
    auto debug_waypoint = mbox_data_.watcher().debug_waypoint();
    std::vector<std::string> risc_status;

    for (auto cpu : debug_waypoint) {
        auto& status = risc_status.emplace_back();
        for (char v : cpu.waypoint()) {
            if (v == 0) {
                break;
            }
            if (isprint(v)) {
                status += v;
            } else {
                LogRunningKernels();
                TT_THROW(
                    "Watcher data corrupted, unexpected debug status on core {}, unprintable character {}",
                    virtual_coord_.str(),
                    (int)v);
            }
        }
    }
    // Pad riscv status to 4 chars for alignment
    if (to_stdout) {
        log_info(tt::LogMetal, "Last waypoint: {:>4}", fmt::join(risc_status, ","));
    } else {
        fmt::print(reader_.f, "{:>4}  ", fmt::join(risc_status, ","));
    }
}

void WatcherDeviceReader::Core::DumpSyncRegs() const {
    // Read back all of the stream state, most of it is unused
    std::vector<uint32_t> data;
    for (uint32_t operand = 0; operand < NUM_CIRCULAR_BUFFERS; operand++) {
        // XXXX TODO(PGK) get this from device
        const uint32_t OPERAND_START_STREAM = 8;
        uint32_t base = NOC_OVERLAY_START_ADDR + ((OPERAND_START_STREAM + operand) * NOC_STREAM_REG_SPACE_SIZE);

        uint32_t rcvd_addr = base + (STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX * sizeof(uint32_t));
        data = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            reader_.device_id, virtual_coord_, rcvd_addr, sizeof(uint32_t));
        uint32_t rcvd = data[0];

        uint32_t ackd_addr = base + (STREAM_REMOTE_DEST_BUF_START_REG_INDEX * sizeof(uint32_t));
        data = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            reader_.device_id, virtual_coord_, ackd_addr, sizeof(uint32_t));
        uint32_t ackd = data[0];

        if (rcvd != ackd) {
            fprintf(reader_.f, "cb[%d](rcv %d!=ack %d) ", operand, rcvd, ackd);
        }
    }
}

void WatcherDeviceReader::Core::DumpStackUsage() const {
    auto stack_usage_mbox = mbox_data_.watcher().stack_usage();
    const auto& hal = MetalContext::instance().hal();
    auto num_processors = hal.get_num_risc_processors(programmable_core_type_);
    for (uint32_t processor_index = 0; processor_index < num_processors; processor_index++) {
        const auto& usage = stack_usage_mbox.cpu()[processor_index];
        if (usage.min_free()) {
            auto [processor_class, processor_type] =
                hal.get_processor_class_and_type_from_index(programmable_core_type_, processor_index);
            HalProcessorIdentifier processor = {
                programmable_core_type_, processor_class, static_cast<int>(processor_type)};
            auto& slot = dump_data_.highest_stack_usage[processor];
            if (usage.min_free() <= slot.stack_free) {
                slot = {virtual_coord_, static_cast<uint16_t>(usage.min_free() - 1), usage.watcher_kernel_id()};
            }
        }
    }
}

void WatcherDeviceReader::Core::ValidateKernelIDs() const {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    auto num_processors = hal.get_num_risc_processors(programmable_core_type_);
    for (size_t i = 0; i < num_processors; i++) {
        uint16_t watcher_kernel_id = launch_msg_.kernel_config().watcher_kernel_ids()[i];
        if (watcher_kernel_id >= reader_.kernel_names.size()) {
            TT_THROW(
                "Watcher data corruption, unexpected {} kernel id on Device {} core {}: {} (last valid {})",
                get_riscv_name(programmable_core_type_, i),
                reader_.device_id,
                virtual_coord_.str(),
                watcher_kernel_id,
                reader_.kernel_names.size());
        }
        dump_data_.used_kernel_names[watcher_kernel_id] = true;
    }
}

void WatcherDeviceReader::Core::LogRunningKernels() const {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    auto num_processors = hal.get_num_risc_processors(programmable_core_type_);
    log_info(tt::LogMetal, "While running kernels:");
    for (size_t i = 0; i < num_processors; i++) {
        log_info(tt::LogMetal, " {}: {}", get_riscv_name(programmable_core_type_, i), GetKernelName(i));
    }
}

const std::string& WatcherDeviceReader::Core::GetKernelName(uint32_t processor_index) const {
    TT_FATAL(processor_index < launch_msg_.kernel_config().watcher_kernel_ids().size(), "processor_index out of range");
    return reader_.kernel_names[launch_msg_.kernel_config().watcher_kernel_ids()[processor_index]];
}

}  // namespace tt::tt_metal
