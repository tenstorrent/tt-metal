// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include <filesystem>
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/utils.hpp"

using namespace tt;

void generate_build_inputs() {
    std::filesystem::path output_dir(tt::llrt::OptionsG.get_root_dir() + "build/tools/");
    int device_id = 0;
    tt_metal::Device* device = tt_metal::CreateDevice(device_id);
    JitBuildStateSet& firmware_build_states = device->firmware_build_states_;

    std::cout << "FW build state count = " << firmware_build_states.size() << std::endl;
    for (auto& build_state : firmware_build_states) {
        const string& target_name = build_state->get_target_name();
        const string& build_cflags = build_state->get_cflags();
        const string& build_defines = build_state->get_defines();
        const string& build_includes = build_state->get_includes();
        const string& build_lflags = build_state->get_lflags();
        string cmake_file = target_name + ".cmake";
        string fname = output_dir.string() + cmake_file;
        FILE* f;
        if ((f = fopen(fname.c_str(), "w")) == nullptr) {
            throw(std::runtime_error("Builder failed to create input file"));
        }

        fprintf(f, "# Build variable for %s\n", cmake_file.c_str());
        fprintf(f, "\nset(GPP_FLAGS_device \n %s)\n", build_cflags.c_str());
        fprintf(f, "\nset(GPP_DEFINES_device \n %s)\n", build_defines.c_str());
        fprintf(f, "\nset(GPP_INCLUDES_device \n %s)\n", build_includes.c_str());
        fprintf(f, "\nset(GPP_LINK_FLAGS_device \n %s)\n", build_lflags.c_str());
        fflush(f);
        fclose(f);
    }

    tt_metal::CloseDevice(device);
}

int main(int argc, char* argv[]) {
    std::cout << "Running tt_builder and command" << std::endl;
    std::cout << "current path is ";
    std::string cmd = "pwd";
    std::string log_file = "./build/tools/output.log";
    if (!tt::utils::run_command(cmd, log_file, false)) {
        std::cout << "Error running command" << std::endl;
    }
    std::cout << "Finished running " << cmd << " calling generate_build_inputs " << std::endl;

    try {
        generate_build_inputs();
    } catch (const std::exception& e) {
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }
}
