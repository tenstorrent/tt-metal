// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "core_coord.hpp"
#include "detail/tt_metal.hpp"
#include "host_api.hpp"
#include "impl/device/device.hpp"
#include "impl/kernels/data_types.hpp"
#include "impl/kernels/kernel_types.hpp"
#include "impl/program/program.hpp"
#include "tt_cluster_descriptor_types.h"

#include "tests/tt_metal/tt_metal/unit_tests_common/common/common_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

class ProgramWithKernelCreatedFromStringFixture : public CommonFixture {
   protected:
    void SetUp() override {
        CommonFixture::SetUp();
        for (Device *device : this->devices_)
        {
            const chip_id_t device_id = device->id();
            this->device_ids_to_devices_[device_id] = device;
        }
    }

    void TearDown() override {
        detail::CloseDevices(this->device_ids_to_devices_);
    }

   private:
    std::map<chip_id_t, Device *> device_ids_to_devices_;
};

TEST_F(ProgramWithKernelCreatedFromStringFixture, DataMovementKernel) {
    const CoreRange cores({0, 0}, {1, 1});
    const string &kernel_src_code = R"(
    #include "debug/dprint.h"
    #include "dataflow_api.h"

    void kernel_main() {

        DPRINT_DATA0(DPRINT << "Hello, I am running a void data movement kernel on NOC 0." << ENDL());
        DPRINT_DATA1(DPRINT << "Hello, I am running a void data movement kernel on NOC 1." << ENDL());

    }
    )";

    for (Device *device : this->devices_) {
        Program program = CreateProgram();
        tt_metal::CreateKernelFromString(
            program,
            kernel_src_code,
            cores,
            tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        this->RunProgram(device, program);
    };
}

TEST_F(ProgramWithKernelCreatedFromStringFixture, ComputeKernel) {
    const CoreRange cores({0, 0}, {1, 1});
    const string &kernel_src_code = R"(
    #include "debug/dprint.h"
    #include "compute_kernel_api.h"

    namespace NAMESPACE {

    void MAIN {

        DPRINT_MATH(DPRINT << "Hello, I am running a void compute kernel." << ENDL());

    }

    }
    )";

    for (Device *device : this->devices_) {
        Program program = CreateProgram();
        tt_metal::CreateKernelFromString(
            program,
            kernel_src_code,
            cores,
            tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                .math_approx_mode = false,
                .compile_args = {}});
        this->RunProgram(device, program);
    };
}

TEST_F(ProgramWithKernelCreatedFromStringFixture, EthernetKernel) {
    const string &kernel_src_code = R"(
    #include "debug/dprint.h"
    #include "dataflow_api.h"

    void kernel_main() {

        DPRINT << "Hello, I am running a void ethernet kernel." << ENDL();

    }
    )";

    for (Device *device : this->devices_) {
        const std::unordered_set<CoreCoord> &active_ethernet_cores = device->get_active_ethernet_cores(true);
        if (active_ethernet_cores.empty()) {
            const chip_id_t device_id = device->id();
            log_info(LogTest, "Skipping this test on device {} because it has no active ethernet cores.", device_id);
            continue;
        }
        Program program = CreateProgram();
        tt_metal::CreateKernelFromString(
            program,
            kernel_src_code,
            *active_ethernet_cores.begin(),
            tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0});
        this->RunProgram(device, program);
    };
}
