// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include "builder.hpp"
#include "tt_metal/common/logger.hpp"
using namespace std;

void print_usage(const char* exec_name) {
    cout << "Usage: " << exec_name << " [OPTION]" << endl;
    cout << "\t-h, --help           : Display this message." << endl;
    cout << "\t--build-firmware     : Build Firmware." << endl;
}

int main(int argc, char* argv[]) {
    cout << "Running tt_builder " << endl;

    try {
        if (argc < 2) {
            cout << "Error: Missing argument " << endl;
            print_usage(argv[0]);
            throw(runtime_error("Missing argument "));
        } else {
            string s(argv[1]);
            if (s == "-h" || s == "--help") {
                print_usage(argv[0]);
                return 0;
            } else if (s == "--build-firmware") {
                build_firmware();
                return 0;
            } else {
                cout << "Error: unrecognized command line argument: " << s << endl;
                print_usage(argv[0]);
                throw(runtime_error("Unrecognized option"));
                // return 1;
            }
        }
    } catch (const exception& e) {
        // Capture the exception error message
        log_error(tt::LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(tt::LogTest, "System error message: {}", strerror(errno));
    }
}
