// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include <filesystem>
#include <fstream>
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/utils.hpp"
#include "tt_metal/jit_build/build.hpp"
#include "tt_metal/impl/device/device.hpp"

using namespace tt;
namespace fs = std::filesystem;
using namespace std;

void build_firmware() {
    std::filesystem::path output_dir(
        llrt::RunTimeOptions::get_instance().get_root_dir() + "build/tools/firmware_build/");
    fs::create_directories(output_dir);
    int device_id = 0;
    tt_metal::Device* device = tt_metal::CreateDevice(device_id);
    const string& gpp_tool = device->build_env().get_gpp_tool();
    JitBuildStateSet& firmware_build_states = device->firmware_build_states_;

    string log_file = output_dir.string() + "build_output.log";
    cout << "FW build state count = " << firmware_build_states.size() << endl;
    for (auto& build_state : firmware_build_states) {
        const string& target_name = build_state->get_target_name();
        const string& build_cflags = build_state->get_cflags();
        const string& build_defines = build_state->get_defines();
        const string& build_includes = build_state->get_includes();
        const string& build_lflags = build_state->get_lflags();
        const auto& build_srcs = build_state->get_srcs();
        const auto& build_objs = build_state->get_objs();
        const string& build_link_objs = build_state->get_link_objs();

        // Compiling
        string cmd;
        cmd = "cd " + output_dir.string() + " && ";
        cmd += gpp_tool;
        cmd += build_cflags;
        cmd += build_defines;
        cmd += build_includes;
        // cmd += "-c -o " + build_objs[0] + " " + build_srcs[0];
        for (size_t i = 0; i < build_srcs.size(); ++i) {
            string file_cmd = cmd + "-c -o " + build_objs[i] + " " + build_srcs[i];
            if (!tt::utils::run_command(file_cmd, log_file, false)) {
                // build_failure(target_name_, "compile", file_cmd, log_file);
                throw(runtime_error("Build failed at compile"));
            }
        }

        // Linking
        cmd = "cd " + output_dir.string() + " && ";
        cmd += gpp_tool;
        cmd += build_lflags;
        cmd += build_link_objs;

        // if (!this->is_fw_) {
        //     string weakened_elf_name =
        //         env_.out_firmware_root_ + this->target_name_ + "/" + this->target_name_ + "_weakened.elf";
        //     cmd += "-Wl,--just-symbols=" + weakened_elf_name + " ";
        // }

        cmd += "-o " + output_dir.string() + target_name + ".elf";
        if (!tt::utils::run_command(cmd, log_file, false)) {
            // build_failure(this->target_name_, "link", cmd, log_file);
            throw(runtime_error("Build failed at link"));
        }
    }

    tt_metal::CloseDevice(device);
}

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
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", strerror(errno));
    }
}
