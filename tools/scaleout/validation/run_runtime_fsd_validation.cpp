// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <sstream>

#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <tt-metalium/distributed.hpp>
#include "tt_metal/impl/context/metal_context.hpp"
#include "tests/tt_metal/test_utils/test_common.hpp"
#include <cabling_generator/cabling_generator.hpp>
#include <factory_system_descriptor/utils.hpp>
#include "tools/scaleout/validation/utils/cluster_validation_utils.hpp"

namespace tt::scaleout_tools {

using tt::tt_metal::PhysicalSystemDescriptor;

// Input arguments for runtime FSD validation
struct InputArgs {
    std::string cabling_descriptor_path;
    std::filesystem::path output_path = "";
    bool fail_on_warning = false;
    bool help = false;
};

std::filesystem::path generate_output_dir() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    std::string dir_name = ss.str();
    const auto& rt_options = tt::tt_metal::MetalContext::instance().rtoptions();
    std::filesystem::path output_dir_path = rt_options.get_root_dir() + "runtime_fsd_validation_logs/" + dir_name;
    std::filesystem::create_directories(output_dir_path);
    return output_dir_path;
}

InputArgs parse_input_args(const std::vector<std::string>& args_vec) {
    InputArgs input_args;

    if (test_args::has_command_option(args_vec, "--cabling-descriptor-path")) {
        input_args.cabling_descriptor_path = test_args::get_command_option(args_vec, "--cabling-descriptor-path");
    } else if (!test_args::has_command_option(args_vec, "--help")) {
        TT_FATAL(false, "Cabling Descriptor Path is required. Use --cabling-descriptor-path <path>");
    }

    if (test_args::has_command_option(args_vec, "--output-path")) {
        input_args.output_path = std::filesystem::path(test_args::get_command_option(args_vec, "--output-path"));
    } else {
        input_args.output_path = generate_output_dir();
    }

    input_args.fail_on_warning = test_args::has_command_option(args_vec, "--hard-fail");
    input_args.help = test_args::has_command_option(args_vec, "--help");

    return input_args;
}

PhysicalSystemDescriptor run_physical_discovery() {
    log_output_rank0("Running Physical Discovery");
    auto& context = tt::tt_metal::MetalContext::instance();
    auto physical_system_descriptor = tt::tt_metal::PhysicalSystemDescriptor(
        context.get_distributed_context_ptr(), &context.hal(), context.rtoptions().get_mock_enabled());
    log_output_rank0("Physical Discovery Complete");

    // Log detected hosts
    const auto& hostnames = physical_system_descriptor.get_all_hostnames();
    std::stringstream ss;
    ss << "Detected " << hostnames.size() << " Host(s): ";
    for (size_t i = 0; i < hostnames.size(); ++i) {
        ss << hostnames[i];
        if (i < hostnames.size() - 1) {
            ss << ", ";
        }
    }
    log_output_rank0(ss.str());

    return physical_system_descriptor;
}

std::string dump_physical_descriptor_to_yaml(
    PhysicalSystemDescriptor& physical_system_descriptor, const std::filesystem::path& output_path) {
    std::string yaml_path = output_path / "physical_system_descriptor.yaml";
    log_output_rank0("Dumping Physical System Descriptor to YAML: " + yaml_path);
    physical_system_descriptor.dump_to_yaml(yaml_path);
    log_output_rank0("Physical System Descriptor YAML created");
    return yaml_path;
}

std::string generate_fsd_from_cabling_descriptor(
    const std::string& cabling_descriptor_path,
    const std::vector<std::string>& hostnames,
    const std::filesystem::path& output_path) {
    log_output_rank0("Generating Factory System Descriptor from Cabling Descriptor");

    // Use the new constructor that takes hostnames instead of deployment descriptor
    tt::scaleout_tools::CablingGenerator cabling_generator(cabling_descriptor_path, hostnames);

    std::string fsd_path = output_path / "generated_factory_system_descriptor.textproto";
    cabling_generator.emit_factory_system_descriptor(fsd_path);
    log_output_rank0("Factory System Descriptor generated: " + fsd_path);

    return fsd_path;
}

void validate_fsd_against_physical_descriptor(
    const std::string& fsd_path, const std::string& physical_yaml_path, bool fail_on_warning) {
    log_output_rank0("Validating Factory System Descriptor against Physical System Descriptor");

    auto missing_physical_connections =
        tt::scaleout_tools::validate_fsd_against_gsd(fsd_path, physical_yaml_path, true, fail_on_warning);

    if (missing_physical_connections.empty()) {
        log_output_rank0("✓ Validation PASSED: All expected connections are present");
    } else {
        log_output_rank0(
            "⚠ Validation found " + std::to_string(missing_physical_connections.size()) +
            " missing or mismatched connection(s)");
    }

    log_output_rank0("Factory System Descriptor Validation Complete");
}

void print_usage_info() {
    std::cout << "Runtime FSD Validation Utility" << std::endl;
    std::cout << "Discovers physical hardware, generates an FSD from a cabling descriptor, and validates them"
              << std::endl
              << std::endl;
    std::cout << "This tool:" << std::endl;
    std::cout << "  1. Runs physical discovery of the cluster" << std::endl;
    std::cout << "  2. Dumps the physical system descriptor to YAML" << std::endl;
    std::cout << "  3. Generates an FSD from the cabling descriptor using discovered hostnames" << std::endl;
    std::cout << "  4. Validates the FSD against the physical descriptor" << std::endl << std::endl;
    std::cout << "Required Arguments:" << std::endl;
    std::cout << "  --cabling-descriptor-path: Path to cabling descriptor textproto" << std::endl << std::endl;
    std::cout << "Optional Arguments:" << std::endl;
    std::cout << "  --output-path: Path to output directory (default: auto-generated timestamp directory)" << std::endl;
    std::cout << "  --hard-fail: Exit with error on any warning" << std::endl;
    std::cout << "  --help: Print this usage information" << std::endl << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  mpirun -n <num_hosts> --hostfile <hostfile> ./run_runtime_fsd_validation \\" << std::endl;
    std::cout << "    --cabling-descriptor-path path/to/cluster_descriptor.textproto \\" << std::endl;
    std::cout << "    --output-path ./validation_output" << std::endl;
}

void set_config_vars() {
    // This tool must be run with slow dispatch mode
    setenv("TT_METAL_SLOW_DISPATCH_MODE", "1", 1);
    // Set env vars required by Control Plane when running on a multi-node cluster
    setenv("TT_MESH_HOST_RANK", "0", 1);
    setenv("TT_MESH_ID", "0", 1);
}

}  // namespace tt::scaleout_tools

int main(int argc, char* argv[]) {
    using namespace tt::scaleout_tools;

    set_config_vars();

    auto input_args = parse_input_args(std::vector<std::string>(argv, argv + argc));
    if (input_args.help) {
        print_usage_info();
        return 0;
    }

    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();

    bool validation_passed = true;

    // Only rank 0 performs the validation workflow
    if (*distributed_context.rank() == 0) {
        try {
            log_output_rank0("=== Starting Runtime FSD Validation ===");
            log_output_rank0("Output Directory: " + input_args.output_path.string());

            // Step 1: Run physical discovery
            auto physical_system_descriptor = run_physical_discovery();

            // Step 2: Dump physical descriptor to YAML
            std::string physical_yaml_path =
                dump_physical_descriptor_to_yaml(physical_system_descriptor, input_args.output_path);

            // Step 3: Get hostnames from physical discovery (in order of host_id)
            std::vector<std::string> hostnames = physical_system_descriptor.get_all_hostnames();

            // Step 4: Generate FSD from cabling descriptor using discovered hostnames
            std::string fsd_path = generate_fsd_from_cabling_descriptor(
                input_args.cabling_descriptor_path, hostnames, input_args.output_path);

            // Step 5: Validate FSD against physical descriptor
            validate_fsd_against_physical_descriptor(fsd_path, physical_yaml_path, input_args.fail_on_warning);

            log_output_rank0("=== Runtime FSD Validation Complete ===");

        } catch (const std::exception& e) {
            log_output_rank0("ERROR: " + std::string(e.what()));
            validation_passed = false;
        }
    }

    distributed_context.barrier();

    if (!validation_passed) {
        if (input_args.fail_on_warning) {
            TT_THROW("Runtime FSD validation failed");
        }
        return -1;
    }

    return 0;
}
