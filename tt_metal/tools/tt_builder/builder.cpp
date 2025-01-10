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
#include "llrt/hal.hpp"
#include <magic_enum/magic_enum.hpp>

using namespace tt;
namespace fs = std::filesystem;
using namespace std;

namespace tt::tt_metal {
const uint32_t num_hw_cqs = 1;
const chip_id_t device_id = 0;

BuilderTool::BuilderTool() : output_dir_("/tmp/tt-metal-cache/") {}

BuilderTool::~BuilderTool() {}

void BuilderTool::set_built_path(const std::string& new_built_path) {
    output_dir_ = new_built_path;
    output_dir_ /= "";
}

std::vector<std::shared_ptr<JitBuildState>> BuilderTool::get_build_states(
    tt_metal::IDevice* device, int id, bool is_fw) {
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(id);
    uint32_t dispatch_message_addr = dispatch_constants::get(dispatch_core_type, num_hw_cqs)
                                         .get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_MESSAGE);
    uint32_t num_build_states = hal.get_num_risc_processors();

    std::vector<std::shared_ptr<JitBuildState>> build_states;

    build_states.resize(num_build_states);
    uint32_t programmable_core_type_count = hal.get_programmable_core_type_count();
    if (is_fw) {
        this->build_state_indices_.resize(programmable_core_type_count);
    }

    uint32_t index = 0;
    for (uint32_t programmable_core = 0; programmable_core < programmable_core_type_count; programmable_core++) {
        HalProgrammableCoreType core_type = magic_enum::enum_value<HalProgrammableCoreType>(programmable_core);
        uint32_t processor_class_count = hal.get_processor_classes_count(programmable_core);
        if (is_fw) {
            this->build_state_indices_[programmable_core].resize(processor_class_count);
        }
        for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
            auto compute_proc_class = magic_enum::enum_cast<HalProcessorClassType>(processor_class);
            bool is_compute_processor =
                compute_proc_class.has_value() and compute_proc_class.value() == HalProcessorClassType::COMPUTE;
            uint32_t processor_types_count = hal.get_processor_types_count(programmable_core, processor_class);
            if (is_fw) {
                this->build_state_indices_[programmable_core][processor_class] = {index, processor_types_count};
            }
            for (uint32_t processor_type = 0; processor_type < processor_types_count; processor_type++) {
                switch (core_type) {
                    case HalProgrammableCoreType::TENSIX: {
                        if (is_compute_processor) {
                            build_states[index] = std::make_shared<JitBuildCompute>(
                                device->build_env(),
                                JitBuiltStateConfig{
                                    .processor_id = processor_type,
                                    .is_fw = is_fw,
                                    .dispatch_message_addr = dispatch_message_addr});
                        } else {
                            // TODO: Make .processor_id = processor_type when brisc and ncrisc are considered one
                            // processor class
                            build_states[index] = std::make_shared<JitBuildDataMovement>(
                                device->build_env(),
                                JitBuiltStateConfig{
                                    .processor_id = processor_class,
                                    .is_fw = is_fw,
                                    .dispatch_message_addr = dispatch_message_addr});
                        }
                        break;
                    }
                    case HalProgrammableCoreType::ACTIVE_ETH: {
                        build_states[index] = std::make_shared<JitBuildActiveEthernet>(
                            device->build_env(),
                            JitBuiltStateConfig{
                                .processor_id = processor_class,
                                .is_fw = is_fw,
                                .dispatch_message_addr = dispatch_message_addr});
                        break;
                    }
                    case HalProgrammableCoreType::IDLE_ETH: {
                        build_states[index] = std::make_shared<JitBuildIdleEthernet>(
                            device->build_env(),
                            JitBuiltStateConfig{
                                .processor_id = processor_class,
                                .is_fw = is_fw,
                                .dispatch_message_addr = dispatch_message_addr});
                        break;
                    }
                    default:
                        TT_THROW(
                            "Unsupported programable core type {} to initialize build states",
                            magic_enum::enum_name(core_type));
                }
                index++;
            }
        }
    }

    return build_states;
}

void BuilderTool::build_firmware() {
    tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);

    firmware_output_dir_ = output_dir_.string() + to_string(device->build_key()) + "/firmware/";
    fs::create_directories(firmware_output_dir_);

    const string& gpp_tool = device->build_env().get_gpp_tool();
    const JitBuildStateSet& firmware_build_states(this->get_build_states(device, device_id, true));

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
