// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include "tt_metal/host_api.hpp"
#include "impl/debug/watcher_server.hpp"

using namespace tt;
using std::vector;
using std::string;
using std::cout, std::endl;

void dump_data(vector<unsigned>& device_ids) {
    // Don't clear L1, this way we can dump the state.
    llrt::OptionsG.set_clear_l1(false);

    // Only look at user-specified devices
    for (unsigned id : device_ids) {
        // Minimal setup, since we'll be attaching to a potentially hanging chip.
        auto* device = tt::tt_metal::CreateDeviceMinimal(id);
        // Watcher attach wthout watcher init - to avoid clearing mailboxes.
        watcher_attach(device);
    }

    // Watcher doesn't have kernel ids since we didn't create them here, need to read from file.
    watcher_read_kernel_ids_from_file();
    watcher_dump();
}

void print_usage(const char* exec_name) {
    cout << "Usage: " << exec_name << " [OPTION]" << endl;
    cout << "\t-d=LIST, --devices=LIST: Device IDs of chips to dump, LIST is comma separated list (\"0,2,3\") or \"all\"." << endl;
    cout << "\t-h, --help: Display this message." << endl;
}

int main(int argc, char *argv[]) {
    cout << "Running watcher dump tool..." << endl;
    // Default devices is all of them.
    vector<unsigned> device_ids;
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    for (unsigned id = 0; id < num_devices; id++) {
        device_ids.push_back(id);
    }

    // Go through user args, handle accordingly.
    for (int idx = 1; idx < argc; idx++) {
        string s(argv[idx]);
        if (s == "-h" || s == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if ((s.rfind("-d=", 0) == 0) || (s.rfind("--devices=", 0) == 0)) {
            string list = s.substr(s.find("=")+1);
            // "all" is acceptable, and the same as the default.
            if (list == "all")
                continue;

            // Otherwise, parse comma-separated list.
            device_ids.clear();
            std::istringstream iss(list);
            string item;
            while (std::getline(iss, item, ',')) {
                if (stoi(item) >= num_devices) {
                    cout << "Error: illegal device (" << stoi(item) << "), allowed range is [0, " << num_devices << ")" << endl;
                    print_usage(argv[0]);
                    return 1;
                }
                device_ids.push_back(stoi(item));
            }
        } else {
            cout << "Error: unrecognized command line argument: " << s << endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Call dump function with user config.
    dump_data(device_ids);
    cout << "Watcher dump tool finished." << endl;
}
