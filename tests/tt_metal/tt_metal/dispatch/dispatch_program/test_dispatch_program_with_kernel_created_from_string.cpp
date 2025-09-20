// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include "program_with_kernel_created_from_string_fixture.hpp"

namespace tt::tt_metal {

using namespace tt;

TEST_F(ProgramWithKernelCreatedFromStringFixture, TensixDataMovementKernel) {
    const CoreRange cores({0, 0}, {1, 1});
    const std::string& kernel_src_code = R"(
    #include "debug/dprint.h"
    #include "dataflow_api.h"

    void kernel_main() {

        DPRINT_DATA0(DPRINT << "Hello, I am running a void data movement kernel on NOC 0." << ENDL());
        DPRINT_DATA1(DPRINT << "Hello, I am running a void data movement kernel on NOC 1." << ENDL());

    }
    )";

    for (const auto& mesh_device : this->devices_) {
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        Program program = CreateProgram();
        CreateKernelFromString(
            program,
            kernel_src_code,
            cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
        this->RunProgram(mesh_device, workload);
    };
}

TEST_F(ProgramWithKernelCreatedFromStringFixture, TensixComputeKernel) {
    const CoreRange cores({0, 0}, {1, 1});
    const std::string& kernel_src_code = R"(
    #include "debug/dprint.h"
    #include "compute_kernel_api.h"

    namespace NAMESPACE {

    void MAIN {

        DPRINT_MATH(DPRINT << "Hello, I am running a void compute kernel." << ENDL());

    }

    }
    )";

    for (const auto& mesh_device : this->devices_) {
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        Program program = CreateProgram();
        CreateKernelFromString(
            program,
            kernel_src_code,
            cores,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                .math_approx_mode = false,
                .compile_args = {}});
        distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
        this->RunProgram(mesh_device, workload);
    };
}

TEST_F(ProgramWithKernelCreatedFromStringFixture, ActiveEthEthernetKernel) {
    const std::string& kernel_src_code = R"(
    #include "debug/dprint.h"
    #include "dataflow_api.h"

    void kernel_main() {

        DPRINT << "Hello, I am running a void ethernet kernel." << ENDL();

    }
    )";

    for (const auto& mesh_device : this->devices_) {
        auto device = mesh_device->get_devices()[0];
        const std::unordered_set<CoreCoord>& active_ethernet_cores = device->get_active_ethernet_cores(true);
        if (active_ethernet_cores.empty()) {
            const chip_id_t device_id = device->id();
            log_info(LogTest, "Skipping this test on device {} because it has no active ethernet cores.", device_id);
            continue;
        }
        const auto erisc_count = tt::tt_metal::MetalContext::instance().hal().get_num_risc_processors(
            tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
        for (uint32_t erisc_idx = 0; erisc_idx < erisc_count; erisc_idx++) {
            log_info(tt::LogTest, "Test active ethernet DM{}", erisc_idx);
            DataMovementProcessor dm_processor = static_cast<DataMovementProcessor>(erisc_idx);
            distributed::MeshWorkload workload;
            auto zero_coord = distributed::MeshCoordinate(0, 0);
            auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
            Program program = CreateProgram();
            tt_metal::CreateKernelFromString(
                program,
                kernel_src_code,
                *active_ethernet_cores.begin(),
<<<<<<< HEAD
                tt_metal::EthernetConfig{.noc = static_cast<NOC>(erisc_idx), .processor = dm_processor});
            distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
            this->RunProgram(mesh_device, workload);
=======
                tt_metal::EthernetConfig{.noc = static_cast<tt_metal::NOC>(dm_processor), .processor = dm_processor});
            this->RunProgram(device, program);
>>>>>>> 468f504dd5 (Enforce NoC1 on RISC1 due to the lack of Dynamic NoC functionality)
        }
    };
}

}  // namespace tt::tt_metal
