// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <host_api.hpp>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "device.hpp"
#include "dispatch_core_common.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/debug/noc_logging.hpp"
#include "impl/dispatch/debug_tools.hpp"
#include "impl/dispatch/system_memory_manager.hpp"
#include <impl/debug/watcher_server.hpp>

using namespace tt;
using namespace tt::tt_metal;
using std::cout, std::endl;
using std::string;
using std::vector;

string output_dir_name = "generated/watcher/";
string logfile_name = "cq_dump.txt";

void dump_data(
    vector<ChipId>& device_ids,
    bool dump_watcher,
    bool dump_cqs,
    bool dump_cqs_raw_data,
    bool dump_noc_xfers,
    bool eth_dispatch,
    int num_hw_cqs) {
    // Don't clear L1, this way we can dump the state.
    tt_metal::MetalContext::instance().rtoptions().set_clear_l1(false);

    // Watcher should be disabled for this, so we don't (1) overwrite the kernel_names.txt and (2) do any other dumping
    // than the one we want.
    tt_metal::MetalContext::instance().rtoptions().set_watcher_enabled(false);

    std::filesystem::path parent_dir(
        tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir() + output_dir_name);
    std::filesystem::path cq_dir(parent_dir.string() + "command_queue_dump/");
    std::filesystem::create_directories(cq_dir);

    // Only look at user-specified devices
    vector<std::unique_ptr<IDevice>> devices;
    for (ChipId id : device_ids) {
        string cq_fname = cq_dir.string() + fmt::format("device_{}_completion_q.txt", id);
        std::ofstream cq_file = std::ofstream(cq_fname);
        string iq_fname = cq_dir.string() + fmt::format("device_{}_issue_q.txt", id);
        std::ofstream iq_file = std::ofstream(iq_fname);
        // Minimal setup, since we'll be attaching to a potentially hanging chip.
        IDevice* device = tt::tt_metal::CreateDeviceMinimal(
            id, num_hw_cqs, DispatchCoreConfig{eth_dispatch ? DispatchCoreType::ETH : DispatchCoreType::WORKER});
        devices.push_back(std::unique_ptr<IDevice>(device));
        if (dump_cqs) {
            cout << "Dumping Command Queues into: " << cq_dir.string() << endl;
            std::unique_ptr<SystemMemoryManager> sysmem_manager = std::make_unique<SystemMemoryManager>(id, num_hw_cqs);
            internal::dump_cqs(cq_file, iq_file, *sysmem_manager, dump_cqs_raw_data);
        }
    }

    // Watcher doesn't have kernel ids since we didn't create them here, need to read from file.
    if (dump_watcher) {
        cout << "Dumping Watcher Log into: " << MetalContext::instance().watcher_server()->log_file_name() << endl;
        MetalContext::instance().watcher_server()->isolated_dump(device_ids);
    }

    // Dump noc data if requested
    if (dump_noc_xfers) {
        DumpNocData(device_ids);
    }
}

void print_usage(const char* exec_name) {
    cout << "Usage: " << exec_name << " [OPTION]" << endl;
    cout << "\t-h, --help: Display this message." << endl;
    cout << "\t-d=LIST, --devices=LIST: Device IDs of chips to dump, LIST is comma separated list (\"0,2,3\") or "
            "\"all\"."
         << endl;
    cout << "\t-n=INT, --num-hw-cqs=INT: Number of CQs, should match the original program." << endl;
    cout << "\t-c, --dump-cqs: Dump Command Queue data." << endl;
    cout << "\t--dump-cqs-data: Dump Command Queue raw data (bytes), this can take minutes per CQ." << endl;
    cout << "\t-w, --dump-watcher: Dump watcher data, available data depends on whether watcher was enabled for "
            "original program."
         << endl;
    cout << "\t--dump-noc-transfer-data: Dump NOC transfer data. Data is only available if previous run had "
            "TT_METAL_RECORD_NOC_TRANSFER_DATA defined."
         << endl;
    cout << "\t--eth-dispatch: Assume eth dispatch, should match previous run." << endl;
}

int main(int argc, char* argv[]) {
    cout << "Running watcher dump tool..." << endl;
    // Default devices is all of them.
    vector<ChipId> device_ids;
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    device_ids.reserve(num_devices);
    for (ChipId id = 0; id < num_devices; id++) {
        device_ids.push_back(id);
    }

    // Go through user args, handle accordingly.
    bool dump_watcher = false, dump_cqs = false, dump_cqs_raw_data = false, dump_noc_xfers = false,
         eth_dispatch = false;
    int num_hw_cqs = 1;
    for (int idx = 1; idx < argc; idx++) {
        string s(argv[idx]);
        if (s == "-h" || s == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        if ((s.starts_with("-d=")) || (s.starts_with("--devices="))) {
            string list = s.substr(s.find('=') + 1);
            // "all" is acceptable, and the same as the default.
            if (list == "all") {
                continue;
            }

            // Otherwise, parse comma-separated list.
            device_ids.clear();
            std::istringstream iss(list);
            string item;
            while (std::getline(iss, item, ',')) {
                if (stoi(item) >= num_devices) {
                    cout << "Error: illegal device (" << stoi(item) << "), allowed range is [0, " << num_devices << ")"
                         << endl;
                    print_usage(argv[0]);
                    return 1;
                }
                device_ids.push_back(stoi(item));
            }
        } else if ((s.starts_with("-n=")) || (s.starts_with("--num-hw-cqs=="))) {
            string value_str = s.substr(s.find('=') + 1);
            num_hw_cqs = std::stoi(value_str);
        } else if (s == "-w" || s == "--dump-watcher") {
            dump_watcher = true;
        } else if (s == "-c" || s == "--dump-cqs") {
            cout << "CQ dumping currently disabled" << endl;
            // dump_cqs = true;
        } else if (s == "--dump-cqs-data") {
            cout << "CQ raw data dumping currently disabled" << endl;
            // dump_cqs_raw_data = true;
        } else if (s == "--dump-noc-transfer-data") {
            tt::tt_metal::MetalContext::instance().rtoptions().set_record_noc_transfers(true);
            dump_noc_xfers = true;
        } else if (s == "--eth-dispatch") {
            eth_dispatch = true;
        } else {
            cout << "Error: unrecognized command line argument: " << s << endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Call dump function with user config.
    dump_data(device_ids, dump_watcher, dump_cqs, dump_cqs_raw_data, dump_noc_xfers, eth_dispatch, num_hw_cqs);
    std::cout << "Watcher dump tool finished." << std::endl;
}
