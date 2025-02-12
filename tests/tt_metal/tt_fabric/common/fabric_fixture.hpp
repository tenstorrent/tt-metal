// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/rtoptions.hpp>
#include "tests/tt_metal/tt_metal/common/dispatch_fixture.hpp"
#include "tt_fabric/control_plane.hpp"
#include "tt_fabric/hw/inc/tt_fabric_interface.h"
namespace tt::tt_fabric {
namespace fabric_router_tests {

class ControlPlaneFixture : public ::testing::Test {
   protected:
       tt::ARCH arch_;
       void SetUp() override {
           auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
           if (not slow_dispatch) {
               tt::log_info(
                   tt::LogTest,
                   "Control plane test suite can only be run with slow dispatch or TT_METAL_SLOW_DISPATCH_MODE set");
               GTEST_SKIP();
           }
       }

       void TearDown() override {}
};

}  // namespace fabric_router_tests

struct DeviceFabricData {
    IDevice* device_;
    std::unique_ptr<tt::tt_metal::Program> fabric_program_;
    std::vector<CoreCoord> router_logical_cores_;
    std::vector<CoreCoord> router_virtual_cores_;
    CoreCoord gatekeeper_logical_core_;
    CoreCoord gatekeeper_virtual_core_;
    uint32_t router_mask_ = 0;
    uint32_t gatekeeper_noc_offset_ = 0;

    DeviceFabricData(IDevice* device, bool run_gk_on_idle_ethernet) : device_(device) {
        // router
        for (const auto& logical_router_core : device->get_active_ethernet_cores(true)) {
            router_logical_cores_.push_back(logical_router_core);
            router_virtual_cores_.push_back(device->ethernet_core_from_logical_core(logical_router_core));
            router_mask_ += 0x1 << logical_router_core.y;
        }
        // gatekeeper
        if (run_gk_on_idle_ethernet) {
            auto idle_eth_cores = device->get_inactive_ethernet_cores();
            if (idle_eth_cores.size() == 0) {
                throw std::runtime_error("No idle ethernet cores found on the device");
            }

            gatekeeper_logical_core_ = *idle_eth_cores.begin();
            gatekeeper_virtual_core_ = device->ethernet_core_from_logical_core(gatekeeper_logical_core_);
        } else {
            gatekeeper_logical_core_ = {0, 9};
            gatekeeper_virtual_core_ = device->worker_core_from_logical_core(gatekeeper_logical_core_);
        }
        gatekeeper_noc_offset_ = tt_metal::hal.noc_xy_encoding(gatekeeper_virtual_core_.x, gatekeeper_virtual_core_.y);
        fabric_program_ = std::make_unique<tt::tt_metal::Program>(CreateProgram());
    }
};

struct FabricData {
    FabricData(const std::vector<IDevice*>& devices, bool run_gk_on_idle_ethernet) : devices_(devices) {
        if (run_gk_on_idle_ethernet) {
            routing_table_addr_ = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
        } else {
            routing_table_addr_ = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED);
        }
        gk_interface_addr_ = routing_table_addr_ + sizeof(fabric_router_l1_config_t) * 4;
        socket_info_addr_ = gk_interface_addr_ + sizeof(gatekeeper_info_t);
        for (auto* device : devices) {
            device_fabric_data_.try_emplace(device->id(), device, run_gk_on_idle_ethernet);
        }

        // create router kernels
        {
            std::vector<uint32_t> router_compile_args = {
                (tunneler_queue_size_bytes_ >> 4),  // 0: rx_queue_size_words
                tunneler_test_results_addr_,        // 1: test_results_addr
                tunneler_test_results_size_,        // 2: test_results_size
                timeout_cycles_,                    // timeout_mcycles * 1000 * 1000 * 4, // 3: timeout_cycles
            };
            std::vector<uint32_t> zero_buf(1, 0);
            for (auto* device : devices) {
                auto& fabric_data = device_fabric_data_.at(device->id());
                uint32_t num_routers = fabric_data.router_logical_cores_.size();
                // setup run time args
                std::vector<uint32_t> runtime_args = {
                    num_routers,                         // 0: number of active fabric routers
                    fabric_data.router_mask_,            // 1: active fabric router mask
                    gk_interface_addr_,                  // 2: gk_message_addr_l
                    fabric_data.gatekeeper_noc_offset_,  // 3: gk_message_addr_h
                };
                auto& program = *fabric_data.fabric_program_;
                for (uint32_t i = 0; i < num_routers; i++) {
                    // initialize the semaphore
                    tt::llrt::write_hex_vec_to_core(
                        device->id(), fabric_data.router_virtual_cores_[i], zero_buf, FABRIC_ROUTER_SYNC_SEM);

                    auto kernel = tt_metal::CreateKernel(
                        program,
                        "tt_fabric/impl/kernels/tt_fabric_router.cpp",
                        fabric_data.router_logical_cores_[i],
                        tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0, .compile_args = router_compile_args});

                    tt_metal::SetRuntimeArgs(program, kernel, fabric_data.router_logical_cores_[i], runtime_args);
                }
            }
        }

        // create gatekeeper kernels
        {
            std::vector<uint32_t> gatekeeper_compile_args = {
                gk_interface_addr_,   // 0: gk info addr
                socket_info_addr_,    // 1:
                routing_table_addr_,  // 2:
                test_results_addr_,   // 3: test_results_addr
                test_results_size_,   // 4: test_results_size
                timeout_cycles_,      // 5: timeout_cycles
            };
            std::vector<uint32_t> zero_buf(12, 0);
            for (auto* device : devices) {
                auto& fabric_data = device_fabric_data_.at(device->id());
                uint32_t num_routers = fabric_data.router_logical_cores_.size();

                std::vector<uint32_t> runtime_args = {
                    num_routers,               // 0: number of active fabric routers
                    fabric_data.router_mask_,  // 1: active fabric router mask
                };

                // initialize the semaphore
                tt::llrt::write_hex_vec_to_core(
                    device->id(), fabric_data.gatekeeper_virtual_core_, zero_buf, gk_interface_addr_);

                KernelHandle kernel;
                auto& program = *fabric_data.fabric_program_;
                const auto& logical_core = fabric_data.gatekeeper_logical_core_;

                if (run_gk_on_idle_ethernet) {
                    kernel = tt_metal::CreateKernel(
                        program,
                        "tt_fabric/impl/kernels/tt_fabric_gatekeeper.cpp",
                        {logical_core},
                        tt_metal::EthernetConfig{
                            .eth_mode = Eth::IDLE,
                            .noc = tt_metal::NOC::NOC_0,
                            .compile_args = gatekeeper_compile_args});
                } else {
                    kernel = tt_metal::CreateKernel(
                        program,
                        "tt_fabric/impl/kernels/tt_fabric_gatekeeper.cpp",
                        {logical_core},
                        tt_metal::DataMovementConfig{
                            .processor = tt_metal::DataMovementProcessor::RISCV_0,
                            .noc = tt_metal::NOC::RISCV_0_default,
                            .compile_args = gatekeeper_compile_args});
                }

                tt_metal::SetRuntimeArgs(program, kernel, logical_core, runtime_args);
            }
        }
    }

    void launch_fabric() {
        if (fabric_active_) {
            return;
        }
        for (auto* device : devices_) {
            const auto& fabric_program = device_fabric_data_.at(device->id()).fabric_program_;
            tt_metal::detail::LaunchProgram(device, *fabric_program, false);
        }
        uint32_t sync_addr = gk_interface_addr_ + offsetof(gatekeeper_info_t, router_sync) + offsetof(sync_word_t, val);
        for (auto* device : devices_) {
            uint32_t gk_status = 0;
            uint32_t num_routers = device_fabric_data_.at(device->id()).router_logical_cores_.size();
            const auto& gatekeeper_virtual_core = device_fabric_data_.at(device->id()).gatekeeper_virtual_core_;
            while (num_routers != gk_status) {
                gk_status = tt::llrt::read_hex_vec_from_core(device->id(), gatekeeper_virtual_core, sync_addr, 4)[0];
            }
        }
        fabric_active_ = true;
    }

    void teardown_fabric() {
        if (not fabric_active_) {
            return;
        }
        std::vector<uint32_t> zero_buf(12, 0);
        // Signal gatekeeper to terminate
        // Gatekeeper will signal routers to terminate
        for (auto* device : devices_) {
            const auto& gatekeeper_virtual_core = device_fabric_data_.at(device->id()).gatekeeper_virtual_core_;
            tt::llrt::write_hex_vec_to_core(device->id(), gatekeeper_virtual_core, zero_buf, gk_interface_addr_);
        }
        for (auto* device : devices_) {
            const auto& fabric_program = device_fabric_data_.at(device->id()).fabric_program_;
            tt_metal::detail::WaitProgramDone(device, *fabric_program);
        }
        fabric_active_ = false;
    }

    const CoreCoord& get_gatekeeper_virtual_coord(chip_id_t physical_chip_id) const {
        return device_fabric_data_.at(physical_chip_id).gatekeeper_virtual_core_;
    }

    std::pair<uint32_t, uint32_t> get_gatekeeper_noc_addr(chip_id_t physical_chip_id) const {
        auto& device_fabric_data = device_fabric_data_.at(physical_chip_id);
        return {device_fabric_data.gatekeeper_noc_offset_, gk_interface_addr_};
    }

    std::vector<IDevice*> devices_;
    std::unordered_map<chip_id_t, DeviceFabricData> device_fabric_data_;
    bool fabric_active_ = false;
    uint32_t routing_table_addr_ = 0;
    uint32_t gk_interface_addr_ = 0;
    uint32_t socket_info_addr_ = 0;
    const uint32_t tunneler_queue_size_bytes_ = 0x8000;
    const uint32_t tunneler_test_results_addr_ = 0x0;
    const uint32_t tunneler_test_results_size_ = 0x0;
    const uint32_t test_results_addr_ = 0x0;
    const uint32_t test_results_size_ = 0x0;
    const uint32_t timeout_cycles_ = 0x0;
};

class FabricFixture : public DispatchFixture {
public:
protected:
    std::unique_ptr<tt::tt_fabric::ControlPlane> control_plane_;
    std::unique_ptr<FabricData> fabric_data_;
    bool run_gk_on_idle_ethernet = true;

    void SetUpFabricPrograms() {
        fabric_data_ = std::make_unique<FabricData>(devices_, run_gk_on_idle_ethernet);
        fabric_data_->launch_fabric();
    }

    void TearDownFabricPrograms() { fabric_data_->teardown_fabric(); }

    void SetUp() override {
        this->DetectDispatchMode();
        if (not this->IsSlowDispatch()) {
            tt::log_info(
                tt::LogTest, "Fabric test suite can only be run with slow dispatch or TT_METAL_SLOW_DISPATCH_MODE set");
            GTEST_SKIP();
        }
        DispatchFixture::SetUp();
        const std::filesystem::path mesh_graph_desc_path =
            std::filesystem::path(tt::llrt::RunTimeOptions::get_instance().get_root_dir()) /
            "tt_fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";
        control_plane_ = std::make_unique<tt::tt_fabric::ControlPlane>(mesh_graph_desc_path.string());
        SetUpFabricPrograms();
    }

    const FabricData& GetFabricData() { return *fabric_data_; }

    void TearDown() override {
        if (not this->IsSlowDispatch()) {
            return;
        }
        TearDownFabricPrograms();
        DispatchFixture::TearDown();
    }
};

}  // namespace tt::tt_fabric
