// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <string>

#include "common/core_coord.h"
#include "hw/inc/debug/ring_buffer.h"
#include "hw/inc/dev_msgs.h"
#include "impl/device/device.hpp"
#include "llrt/rtoptions.hpp"
#include "noc/noc_overlay_parameters.h"
#include "noc/noc_parameters.h"

#include "watcher_device_reader.hpp"

using std::string;

namespace tt::watcher {

// Helper function to get string rep of riscv type
static const char *get_riscv_name(CoreCoord core, uint32_t type) {
    switch (type) {
        case DebugBrisc: return "brisc";
        case DebugNCrisc: return "ncrisc";
        case DebugErisc: return "erisc";
        case DebugIErisc: return "ierisc";
        case DebugTrisc0: return "trisc0";
        case DebugTrisc1: return "trisc1";
        case DebugTrisc2: return "trisc2";
        default: TT_THROW("Watcher data corrupted, unexpected riscv type on core {}: {}", core.str(), type);
    }
    return nullptr;
}

// Helper function to get stack size by riscv core type
uint32_t get_riscv_stack_size(CoreCoord core, uint32_t type) {
    switch (type) {
        case DebugBrisc: return MEM_BRISC_STACK_SIZE;
        case DebugNCrisc: return MEM_NCRISC_STACK_SIZE;
        case DebugErisc: return 0; // Not managed/checked by us.
        case DebugIErisc: return MEM_BRISC_STACK_SIZE;
        case DebugTrisc0: return MEM_TRISC0_STACK_SIZE;
        case DebugTrisc1: return MEM_TRISC1_STACK_SIZE;
        case DebugTrisc2: return MEM_TRISC2_STACK_SIZE;
        default: TT_THROW("Watcher data corrupted, unexpected riscv type on core {}: {}", core.str(), type);
    }
    return 0;
}

// Helper function to get string rep of noc target.
static string get_noc_target_str(Device *device, CoreCoord &core, int noc, const debug_sanitize_noc_addr_msg_t *san) {
    auto get_core_and_mem_type = [](Device *device, CoreCoord &noc_coord, int noc) -> std::pair<string, string> {
        // Get the physical coord from the noc coord
        const metal_SocDescriptor &soc_d = tt::Cluster::instance().get_soc_desc(device->id());
        CoreCoord phys_core = {
            NOC_0_X(noc, soc_d.grid_size.x, noc_coord.x), NOC_0_Y(noc, soc_d.grid_size.y, noc_coord.y)};

        CoreType core_type;
        try {
            core_type = device->core_type_from_physical_core(phys_core);
        } catch (std::runtime_error &e) {
            // We may not be able to get a core type if the physical coords are bad.
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
    string out = fmt::format("{} using noc{} tried to access ", get_riscv_name(core, san->which), noc);
    if (san->multicast) {
        CoreCoord target_phys_noc_core_start = {
            NOC_MCAST_ADDR_START_X(san->noc_addr), NOC_MCAST_ADDR_START_Y(san->noc_addr)};
        CoreCoord target_phys_noc_core_end = {NOC_MCAST_ADDR_END_X(san->noc_addr), NOC_MCAST_ADDR_END_Y(san->noc_addr)};
        auto type_and_mem = get_core_and_mem_type(device, target_phys_noc_core_start, noc);
        out += fmt::format(
            "{} core range w/ physical coords {}-{} {}",
            type_and_mem.first,
            target_phys_noc_core_start.str(),
            target_phys_noc_core_end.str(),
            type_and_mem.second);
    } else {
        CoreCoord target_phys_noc_core = {NOC_UNICAST_ADDR_X(san->noc_addr), NOC_UNICAST_ADDR_Y(san->noc_addr)};
        auto type_and_mem = get_core_and_mem_type(device, target_phys_noc_core, noc);
        out += fmt::format(
            "{} core w/ physical coords {} {}", type_and_mem.first, target_phys_noc_core.str(), type_and_mem.second);
    }

    out += fmt::format("[addr=0x{:08x},len={}]", NOC_LOCAL_ADDR(san->noc_addr), san->len);
    return out;
}

void WatcherDeviceReader::Dump(FILE *file) {
    // If specified, override the existing file destination
    if (file != nullptr) {
        this->f = file;
    }

    // At this point, file should be valid.
    TT_ASSERT(this->f != nullptr);

    if (f != stdout && f != stderr) {
        log_info(LogLLRuntime, "Watcher checking device {}", device->id());
    }

    // Clear per-dump info
    paused_cores.clear();
    highest_stack_usage.clear();
    used_kernel_names.clear();

    // Dump worker cores
    CoreCoord grid_size = device->logical_grid_size();
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreDescriptor logical_core = {{x, y}, CoreType::WORKER};
            if (device->storage_only_cores().find(logical_core.coord) == device->storage_only_cores().end()) {
                DumpCore(logical_core, false);
            }
        }
    }

    // Dump eth cores
    for (const CoreCoord &eth_core : device->ethernet_cores()) {
        CoreDescriptor logical_core = {eth_core, CoreType::ETH};
        CoreCoord physical_core = device->ethernet_core_from_logical_core(eth_core);
        if (device->is_active_ethernet_core(eth_core)) {
            DumpCore(logical_core, true);
        } else if (device->is_inactive_ethernet_core(eth_core)) {
            DumpCore(logical_core, false);
        } else {
            continue;
        }
    }

    for (auto k_id : used_kernel_names) {
        fprintf(f, "k_id[%d]: %s\n", k_id.first, kernel_names[k_id.first].c_str());
    }

    // Print stack usage report for this device/dump
    if (!highest_stack_usage.empty()) {
        fprintf(f, "Stack usage summary:");
        for (auto &risc_id_and_stack_info : highest_stack_usage) {
            stack_usage_info_t &info = risc_id_and_stack_info.second;
            const char *riscv_name = get_riscv_name(info.core, risc_id_and_stack_info.first);
            uint16_t stack_size = get_riscv_stack_size(info.core, risc_id_and_stack_info.first);
            fprintf(
                f,
                "\n\t%s highest stack usage: %d/%d, on core %s, running kernel %s",
                riscv_name,
                info.stack_usage,
                stack_size,
                info.core.str().c_str(),
                kernel_names[info.kernel_id].c_str());
            if (info.stack_usage >= stack_size) {
                fprintf(f, " (OVERFLOW)");
                log_fatal(
                    "Watcher detected stack overflow on Device {} Core {}: {}! Kernel {} uses {}/{} of the stack.",
                    device->id(),
                    info.core.str(),
                    riscv_name,
                    kernel_names[info.kernel_id].c_str(),
                    info.stack_usage,
                    stack_size);
            } else if (stack_size - info.stack_usage <= std::min(32, stack_size / 10)) {
                fprintf(f, " (Close to overflow)");
                log_warning(
                    "Watcher detected stack usage within 10\% of max on Device {} Core {}: {}! Kernel {} uses "
                    "{}/{} of the stack.",
                    device->id(),
                    info.core.str(),
                    riscv_name,
                    kernel_names[info.kernel_id].c_str(),
                    info.stack_usage,
                    stack_size);
            }
        }
        fprintf(f, "\n");
    }

    // Handle any paused cores, wait for user input.
    if (!paused_cores.empty()) {
        string paused_cores_str = "Paused cores: ";
        for (auto &core_and_risc : paused_cores) {
            paused_cores_str += fmt::format(
                "{}:{}, ", core_and_risc.first.str(), get_riscv_name(core_and_risc.first, core_and_risc.second));
        }
        paused_cores_str += "\n";
        fprintf(f, "%s", paused_cores_str.c_str());
        log_info(LogLLRuntime, "{}Press ENTER to unpause core(s) and continue...", paused_cores_str);
        if (!tt::llrt::OptionsG.get_watcher_auto_unpause()) {
            while (std::cin.get() != '\n') {
                ;
            }
        }

        // Clear all pause flags
        for (auto &core_and_risc : paused_cores) {
            const CoreCoord &phys_core = core_and_risc.first;
            riscv_id_t risc_id = core_and_risc.second;

            uint64_t addr = GET_WATCHER_DEV_ADDR_FOR_CORE(device, phys_core, pause_status);

            // Clear only the one flag that we saved, in case another one was raised on device
            auto pause_data =
                tt::llrt::read_hex_vec_from_core(device->id(), phys_core, addr, sizeof(debug_pause_msg_t));
            auto pause_msg = reinterpret_cast<debug_pause_msg_t *>(&(pause_data[0]));
            pause_msg->flags[risc_id] = 0;
            tt::llrt::write_hex_vec_to_core(device->id(), phys_core, pause_data, addr);
        }
    }
    fflush(f);
}

void WatcherDeviceReader::DumpCore(CoreDescriptor logical_core, bool is_active_eth_core) {
    // Watcher only treats ethernet + worker cores.
    bool is_eth_core = (logical_core.type == CoreType::ETH);
    CoreCoord core = device->physical_core_from_logical_core(logical_core.coord, logical_core.type);

    // Print device id, core coords (logical)
    string core_type = is_eth_core ? "ethnet" : "worker";
    string core_str = fmt::format(
        "Device {} {} core(x={:2},y={:2}) phys(x={:2},y={:2})",
        device->id(),
        core_type,
        logical_core.coord.x,
        logical_core.coord.y,
        core.x,
        core.y);
    fprintf(f, "%s: ", core_str.c_str());

    // Ethernet cores have a different mailbox base addr
    uint64_t mailbox_addr = MEM_MAILBOX_BASE;
    if (is_eth_core) {
        if (is_active_eth_core)
            mailbox_addr = eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE;
        else
            mailbox_addr = MEM_IERISC_MAILBOX_BASE;
    }

    std::vector<uint32_t> data;
    data = tt::llrt::read_hex_vec_from_core(device->id(), core, mailbox_addr, sizeof(mailboxes_t));
    mailboxes_t *mbox_data = (mailboxes_t *)(&data[0]);

    // Validate these first since they are used in diagnostic messages below.
    ValidateKernelIDs(core, &mbox_data->launch);

    // Whether or not watcher data is available depends on a flag set on the device.
    bool enabled = (mbox_data->watcher.enable == WatcherEnabled);

    if (enabled) {
        // Dump state only gathered if device is compiled w/ watcher
        if (!tt::llrt::OptionsG.watcher_status_disabled())
            DumpWaypoints(core, mbox_data, false);
        // Ethernet cores have firmware that starts at address 0, so no need to check it for a
        // magic value.
        if (!is_eth_core)
            DumpL1Status(core, &mbox_data->launch);
        if (!tt::llrt::OptionsG.watcher_noc_sanitize_disabled()) {
            for (uint32_t noc = 0; noc < NUM_NOCS; noc++) {
                DumpNocSanitizeStatus(core, core_str, mbox_data, noc);
            }
        }
        if (!tt::llrt::OptionsG.watcher_assert_disabled())
            DumpAssertStatus(core, core_str, mbox_data);
        if (!tt::llrt::OptionsG.watcher_pause_disabled())
            DumpPauseStatus(core, core_str, mbox_data);
    }

    // Ethernet cores don't use the launch message/sync reg
    if (!is_eth_core) {
        // Dump state always available
        DumpLaunchMessage(core, mbox_data);
        if (tt::llrt::OptionsG.get_watcher_dump_all()) {
            // Reading registers while running can cause hangs, only read if
            // requested explicitly
            DumpSyncRegs(core);
        }
    } else {
        fprintf(f, "rmsg:");
        DumpRunState(core, &mbox_data->launch, mbox_data->launch.go.run);
        fprintf(f, " ");
    }

    // Eth core only reports erisc kernel id, uses the brisc field
    if (is_eth_core) {
        fprintf(f, "k_id:%d", mbox_data->launch.kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM0]);
    } else {
        fprintf(
            f,
            "k_ids:%d|%d|%d",
            mbox_data->launch.kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM0],
            mbox_data->launch.kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM1],
            mbox_data->launch.kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE]);
    }

    // Ring buffer at the end because it can print a bunch of data, same for stack usage
    if (enabled) {
        if (!tt::llrt::OptionsG.watcher_stack_usage_disabled())
            DumpStackUsage(core, mbox_data);
        if (!tt::llrt::OptionsG.watcher_ring_buffer_disabled())
            DumpRingBuffer(core, mbox_data, false);
    }

    fprintf(f, "\n");

    fflush(f);
}

void WatcherDeviceReader::DumpL1Status(CoreCoord core, const launch_msg_t *launch_msg) {
    // Read L1 address 0, looking for memory corruption
    std::vector<uint32_t> data;
    data = tt::llrt::read_hex_vec_from_core(device->id(), core, MEM_L1_BASE, sizeof(uint32_t));
    if (data[0] != llrt::generate_risc_startup_addr(false)) {
        LogRunningKernels(launch_msg);
        TT_THROW("Watcher found corruption at L1[0] on core {}: read {}", core.str(), data[0]);
    }
}

void WatcherDeviceReader::DumpNocSanitizeStatus(
    CoreCoord core, const string &core_str, const mailboxes_t *mbox_data, int noc) {
    const launch_msg_t *launch_msg = &mbox_data->launch;
    const debug_sanitize_noc_addr_msg_t *san = &mbox_data->watcher.sanitize_noc[noc];
    string error_msg;
    string error_reason = "Watcher detected NOC error and stopped device: ";

    switch (san->invalid) {
        case DebugSanitizeNocInvalidOK:
            if (san->noc_addr != DEBUG_SANITIZE_NOC_SENTINEL_OK_64 ||
                san->l1_addr != DEBUG_SANITIZE_NOC_SENTINEL_OK_32 || san->len != DEBUG_SANITIZE_NOC_SENTINEL_OK_32 ||
                san->multicast != DEBUG_SANITIZE_NOC_SENTINEL_OK_16 ||
                san->which != DEBUG_SANITIZE_NOC_SENTINEL_OK_16) {
                error_msg = fmt::format(
                    "Watcher unexpected noc debug state on core {}, reported valid got noc{}{{0x{:08x}, {} }}",
                    core.str().c_str(),
                    san->which,
                    san->noc_addr,
                    san->len);
                error_reason += "corrupted noc sanitization state - sanitization memory overwritten.";
            }
            break;
        case DebugSanitizeNocInvalidL1:
            error_msg = fmt::format(
                "{} using noc{} accesses local L1[addr=0x{:08x},len={}]",
                get_riscv_name(core, san->which),
                noc,
                san->l1_addr,
                san->len);
            error_reason += "bad NOC L1/reg address.";
            break;
        case DebugSanitizeNocInvalidUnicast:
            error_msg = get_noc_target_str(device, core, noc, san);
            error_reason += "bad NOC unicast transaction.";
            break;
        case DebugSanitizeNocInvalidMulticast:
            error_msg = get_noc_target_str(device, core, noc, san);
            error_reason += "bad NOC multicast transaction.";
            break;
        case DebugSanitizeNocInvalidAlignment:
            error_msg = get_noc_target_str(device, core, noc, san);
            error_msg += fmt::format(", misaligned with local L1[addr=0x{:08x}]", san->l1_addr);
            error_reason += "bad alignment in NOC transaction.";
            break;
        default:
            error_msg = fmt::format(
                "Watcher unexpected data corruption, noc debug state on core {}, unknown failure code: {}",
                core.str(),
                san->invalid);
            error_reason += "corrupted noc sanitization state - unknown failure code.";
    }

    // If we logged an error, print to stdout and throw.
    if (!error_msg.empty()) {
        log_warning(error_reason.c_str());
        log_warning("{}: {}", core_str, error_msg);
        DumpWaypoints(core, mbox_data, true);
        DumpRingBuffer(core, mbox_data, true);
        LogRunningKernels(launch_msg);
        // Save the error string for checking later in unit tests.
        set_watcher_exception_message(fmt::format("{}: {}", core_str, error_msg));
        TT_THROW(error_reason);
    }
}

void WatcherDeviceReader::DumpAssertStatus(CoreCoord core, const string &core_str, const mailboxes_t *mbox_data) {
    const launch_msg_t *launch_msg = &mbox_data->launch;
    const debug_assert_msg_t *assert_status = &mbox_data->watcher.assert_status;
    switch (assert_status->tripped) {
        case DebugAssertTripped: {
            // TODO: Get rid of this once #6098 is implemented.
            std::string line_num_warning =
                "Note that file name reporting is not yet implemented, and the reported line number for the assert may "
                "be from a different file.";
            string error_msg = fmt::format(
                "{}: {} tripped an assert on line {}. Current kernel: {}. {}",
                core_str,
                get_riscv_name(core, assert_status->which),
                assert_status->line_num,
                GetKernelName(core, launch_msg, assert_status->which).c_str(),
                line_num_warning.c_str());
            log_warning("Watcher stopped the device due to tripped assert, see watcher log for more details");
            log_warning(error_msg.c_str());
            DumpWaypoints(core, mbox_data, true);
            DumpRingBuffer(core, mbox_data, true);
            LogRunningKernels(launch_msg);
            set_watcher_exception_message(error_msg);
            TT_THROW("Watcher detected tripped assert and stopped device.");
            break;
        }
        case DebugAssertOK:
            if (assert_status->line_num != DEBUG_SANITIZE_NOC_SENTINEL_OK_16 ||
                assert_status->which != DEBUG_SANITIZE_NOC_SENTINEL_OK_8) {
                TT_THROW(
                    "Watcher unexpected assert state on core {}, reported OK but got risc {}, line {}.",
                    assert_status->which,
                    assert_status->line_num);
            }
            break;
        default:
            LogRunningKernels(launch_msg);
            TT_THROW(
                "Watcher data corruption, noc assert state on core {} unknown failure code: {}.\n",
                core.str(),
                assert_status->tripped);
    }
}

void WatcherDeviceReader::DumpPauseStatus(CoreCoord core, const string &core_str, const mailboxes_t *mbox_data) {
    const debug_pause_msg_t *pause_status = &mbox_data->watcher.pause_status;
    // Just record which cores are paused, printing handled at the end.
    for (int risc_id = 0; risc_id < DebugNumUniqueRiscs; risc_id++) {
        auto pause = pause_status->flags[risc_id];
        if (pause == 1) {
            paused_cores.insert({core, static_cast<riscv_id_t>(risc_id)});
        } else if (pause > 1) {
            string error_reason = fmt::format("Watcher data corruption, pause state on core {} unknown code: {}.\n", core.str(), pause);
            log_warning(error_reason.c_str());
            log_warning("{}: {}", core_str, error_reason);
            DumpWaypoints(core, mbox_data, true);
            DumpRingBuffer(core, mbox_data, true);
            LogRunningKernels(&mbox_data->launch);
            // Save the error string for checking later in unit tests.
            set_watcher_exception_message(fmt::format("{}: {}", core_str, error_reason));
            TT_THROW(error_reason);
        }
    }
}

void WatcherDeviceReader::DumpRingBuffer(CoreCoord core, const mailboxes_t *mbox_data, bool to_stdout) {
    const debug_ring_buf_msg_t *ring_buf_data = &mbox_data->watcher.debug_ring_buf;
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
                if (ring_buf_data->wrapped == 0)
                    break;  // No wrapping, so no extra data available
                else
                    curr_idx = DEBUG_RING_BUFFER_ELEMENTS - 1;  // Loop
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
            log_info(out.c_str());
        }
    } else {
        fprintf(f, "%s", out.c_str());
    }
}

void WatcherDeviceReader::DumpRunState(CoreCoord core, const launch_msg_t *launch_msg, uint32_t state) {
    char code = 'U';
    if (state == RUN_MSG_INIT)
        code = 'I';
    else if (state == RUN_MSG_GO)
        code = 'G';
    else if (state == RUN_MSG_DONE)
        code = 'D';
    if (code == 'U') {
        LogRunningKernels(launch_msg);
        TT_THROW(
            "Watcher data corruption, unexpected run state on core{}: {} (expected {} or {} or {})",
            core.str(),
            state,
            RUN_MSG_INIT,
            RUN_MSG_GO,
            RUN_MSG_DONE);
    } else {
        fprintf(f, "%c", code);
    }
}

void WatcherDeviceReader::DumpLaunchMessage(CoreCoord core, const mailboxes_t *mbox_data) {
    const launch_msg_t *launch_msg = &mbox_data->launch;
    const slave_sync_msg_t *slave_sync = &mbox_data->slave_sync;
    fprintf(f, "rmsg:");

    if (launch_msg->kernel_config.mode == DISPATCH_MODE_DEV) {
        fprintf(f, "D");
    } else if (launch_msg->kernel_config.mode == DISPATCH_MODE_HOST) {
        fprintf(f, "H");
    } else {
        LogRunningKernels(launch_msg);
        TT_THROW(
            "Watcher data corruption, unexpected launch mode on core {}: {} (expected {} or {})",
            core.str(),
            launch_msg->kernel_config.mode,
            DISPATCH_MODE_DEV,
            DISPATCH_MODE_HOST);
    }

    if (launch_msg->kernel_config.brisc_noc_id == 0 || launch_msg->kernel_config.brisc_noc_id == 1) {
        fprintf(f, "%d", launch_msg->kernel_config.brisc_noc_id);
    } else {
        LogRunningKernels(launch_msg);
        TT_THROW(
            "Watcher data corruption, unexpected brisc noc_id on core {}: {} (expected 0 or 1)",
            core.str(),
            launch_msg->kernel_config.brisc_noc_id);
    }

    DumpRunState(core, launch_msg, launch_msg->go.run);

    fprintf(f, "|");

    if (launch_msg->kernel_config.enables & ~(DISPATCH_CLASS_MASK_TENSIX_ENABLE_DM0 |
                                              DISPATCH_CLASS_MASK_TENSIX_ENABLE_DM1 |
                                              DISPATCH_CLASS_MASK_TENSIX_ENABLE_COMPUTE)) {
        LogRunningKernels(launch_msg);
        TT_THROW(
            "Watcher data corruption, unexpected kernel enable on core {}: {} (expected only low bits set)",
            core.str(),
            launch_msg->kernel_config.enables);
    }

    if (launch_msg->kernel_config.enables & DISPATCH_CLASS_MASK_TENSIX_ENABLE_DM0) {
        fprintf(f, "B");
    } else {
        fprintf(f, "b");
    }

    if (launch_msg->kernel_config.enables & DISPATCH_CLASS_MASK_TENSIX_ENABLE_DM1) {
        fprintf(f, "N");
    } else {
        fprintf(f, "n");
    }

    if (launch_msg->kernel_config.enables & DISPATCH_CLASS_MASK_TENSIX_ENABLE_COMPUTE) {
        fprintf(f, "T");
    } else {
        fprintf(f, "t");
    }

    fprintf(f, " ");

    fprintf(f, "smsg:");
    DumpRunState(core, launch_msg, slave_sync->ncrisc);
    DumpRunState(core, launch_msg, slave_sync->trisc0);
    DumpRunState(core, launch_msg, slave_sync->trisc1);
    DumpRunState(core, launch_msg, slave_sync->trisc2);

    fprintf(f, " ");
}

void WatcherDeviceReader::DumpWaypoints(CoreCoord core, const mailboxes_t *mbox_data, bool to_stdout) {
    const launch_msg_t *launch_msg = &mbox_data->launch;
    const debug_status_msg_t *debug_status = mbox_data->watcher.debug_status;
    string out;

    for (int cpu = 0; cpu < MAX_RISCV_PER_CORE; cpu++) {
        string risc_status;
        for (int byte = 0; byte < num_status_bytes_per_riscv; byte++) {
            char v = ((char *)&debug_status[cpu])[byte];
            if (v == 0)
                break;
            if (isprint(v)) {
                risc_status += v;
            } else {
                LogRunningKernels(launch_msg);
                TT_THROW(
                    "Watcher data corrupted, unexpected debug status on core {}, unprintable character {}",
                    core.str(),
                    (int)v);
            }
        }
        // Pad risc status to 4 chars for alignment
        string pad(4 - risc_status.length(), ' ');
        out += (pad + risc_status);
        if (cpu != MAX_RISCV_PER_CORE - 1)
            out += ',';
    }

    out += " ";

    // This function can either log the waypoint to the log or stdout.
    if (to_stdout) {
        out = string("Last waypoint: ") + out;
        log_info(out.c_str());
    } else {
        fprintf(f, "%s ", out.c_str());
    }
}

void WatcherDeviceReader::DumpSyncRegs(CoreCoord core) {
    // Read back all of the stream state, most of it is unused
    std::vector<uint32_t> data;
    for (uint32_t operand = 0; operand < NUM_CIRCULAR_BUFFERS; operand++) {
        // XXXX TODO(PGK) get this from device
        const uint32_t OPERAND_START_STREAM = 8;
        uint32_t base = NOC_OVERLAY_START_ADDR + (OPERAND_START_STREAM + operand) * NOC_STREAM_REG_SPACE_SIZE;

        uint32_t rcvd_addr = base + STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX * sizeof(uint32_t);
        data = tt::llrt::read_hex_vec_from_core(device->id(), core, rcvd_addr, sizeof(uint32_t));
        uint32_t rcvd = data[0];

        uint32_t ackd_addr = base + STREAM_REMOTE_DEST_BUF_START_REG_INDEX * sizeof(uint32_t);
        data = tt::llrt::read_hex_vec_from_core(device->id(), core, ackd_addr, sizeof(uint32_t));
        uint32_t ackd = data[0];

        if (rcvd != ackd) {
            fprintf(f, "cb[%d](rcv %d!=ack %d) ", operand, rcvd, ackd);
        }
    }
}

void WatcherDeviceReader::DumpStackUsage(CoreCoord core, const mailboxes_t *mbox_data) {
    const debug_stack_usage_t *stack_usage_mbox = &mbox_data->watcher.stack_usage;
    for (int risc_id = 0; risc_id < DebugNumUniqueRiscs; risc_id++) {
        uint16_t stack_usage = stack_usage_mbox->max_usage[risc_id];
        if (stack_usage != watcher::DEBUG_SANITIZE_NOC_SENTINEL_OK_16) {
            if (stack_usage > highest_stack_usage[static_cast<riscv_id_t>(risc_id)].stack_usage) {
                highest_stack_usage[static_cast<riscv_id_t>(risc_id)] = {
                    core, stack_usage, stack_usage_mbox->watcher_kernel_id[risc_id]};
            }
        }
    }
}

void WatcherDeviceReader::ValidateKernelIDs(CoreCoord core, const launch_msg_t *launch) {
    if (launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM0] >= kernel_names.size()) {
        uint16_t watcher_kernel_id = launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM0];
        TT_THROW(
            "Watcher data corruption, unexpected brisc kernel id on Device {} core {}: {} (last valid {})",
            device->id(),
            core.str(),
            watcher_kernel_id,
            kernel_names.size());
    }
    used_kernel_names[launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM0]] = true;

    if (launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM1] >= kernel_names.size()) {
        uint16_t watcher_kernel_id = launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM1];
        TT_THROW(
            "Watcher data corruption, unexpected ncrisc kernel id on Device {} core {}: {} (last valid {})",
            device->id(),
            core.str(),
            watcher_kernel_id,
            kernel_names.size());
    }
    used_kernel_names[launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM1]] = true;

    if (launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE] >= kernel_names.size()) {
        uint16_t watcher_kernel_id = launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE];
        TT_THROW(
            "Watcher data corruption, unexpected trisc kernel id on Device {} core {}: {} (last valid {})",
            device->id(),
            core.str(),
            watcher_kernel_id,
            kernel_names.size());
    }
    used_kernel_names[launch->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE]] = true;
}

void WatcherDeviceReader::LogRunningKernels(const launch_msg_t *launch_msg) {
    log_info("While running kernels:");
    log_info(" brisc : {}", kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM0]]);
    log_info(" ncrisc: {}", kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM1]]);
    log_info(" triscs: {}", kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE]]);
}

string WatcherDeviceReader::GetKernelName(CoreCoord core, const launch_msg_t *launch_msg, uint32_t type) {
    switch (type) {
        case DebugBrisc: return kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM0]];
        case DebugErisc:
        case DebugIErisc: return kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_ETH_DM0]];
        case DebugNCrisc: return kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_DM1]];
        case DebugTrisc0:
        case DebugTrisc1:
        case DebugTrisc2: return kernel_names[launch_msg->kernel_config.watcher_kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE]];
        default:
            LogRunningKernels(launch_msg);
            TT_THROW("Watcher data corrupted, unexpected riscv type on core {}: {}", core.str(), type);
    }
    return "";
}

}  // namespace tt::watcher
