// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include <fstream>
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/utils.hpp"
#include "tt_metal/jit_build/build.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/tools/tt_builder/builder.hpp"

using namespace tt;
namespace fs = std::filesystem;
using namespace std;

namespace tt::tt_metal {
BuilderTool::BuilderTool() : output_dir_("/tmp/tt-metal-cache/") {}

BuilderTool::~BuilderTool() {}

void BuilderTool::set_built_path(const std::string& new_built_path) {
    output_dir_ = new_built_path;
    output_dir_ /= "";
}

void BuilderTool::build_firmware() {
    int device_id = 0;
    tt_metal::Device* device = tt_metal::CreateDevice(device_id);

    firmware_output_dir_ = output_dir_.string() + to_string(device->build_key()) + "/firmware/";
    fs::create_directories(firmware_output_dir_);

    const string& gpp_tool = device->build_env().get_gpp_tool();
    const JitBuildStateSet& firmware_build_states(device->get_firmware_build_states());

    string log_file = firmware_output_dir_.string() + "build_output.log";
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
        cmd = "cd " + firmware_output_dir_.string() + " && ";
        cmd += gpp_tool;
        cmd += build_cflags;
        cmd += build_defines;
        cmd += build_includes;
        for (size_t i = 0; i < build_srcs.size(); ++i) {
            string file_cmd = cmd + "-c -o " + build_objs[i] + " " + build_srcs[i];
            if (!tt::utils::run_command(file_cmd, log_file, false)) {
                throw(runtime_error("Build failed at compile"));
            }
        }

        // Linking
        cmd = "cd " + firmware_output_dir_.string() + " && ";
        cmd += gpp_tool;
        cmd += build_lflags;
        cmd += build_link_objs;

        cmd += "-o " + firmware_output_dir_.string() + target_name + ".elf";
        if (!tt::utils::run_command(cmd, log_file, false)) {
            throw(runtime_error("Build failed at link"));
        }
    }

    tt_metal::CloseDevice(device);
}

void BuilderTool::build_dispatch() { throw(runtime_error("Dispatch Build not implemented for builder")); }
}  // namespace tt::tt_metal
