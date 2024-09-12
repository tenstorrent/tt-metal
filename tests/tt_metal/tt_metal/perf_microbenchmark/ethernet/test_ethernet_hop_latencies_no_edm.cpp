// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <limits>
#include <random>
#include <tuple>

#include "tt_metal/common/logger.hpp"
#include "device/tt_arch_types.h"
#include "impl/device/device.hpp"
#include "impl/kernels/data_types.hpp"
#include "impl/kernels/kernel_types.hpp"
#include "tt_backend_api_types.hpp"
#include "tt_metal/common/core_coord.h"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

#include "tt_metal/detail/persistent_kernel_cache.hpp"

using tt::tt_metal::Device;

class T3000TestDevice {
   public:
    T3000TestDevice() : device_open(false) {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            TT_THROW("This suite can only be run without TT_METAL_SLOW_DISPATCH_MODE set");
        }
        arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());

        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (arch_ == tt::ARCH::WORMHOLE_B0 and tt::tt_metal::GetNumAvailableDevices() == 8 and
            tt::tt_metal::GetNumPCIeDevices() == 4) {
            devices_ = tt::tt_metal::detail::CreateDevices({0,1,2,3,4,5,6,7});
            tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(true);

        } else {
            TT_THROW("This suite can only be run on T3000 Wormhole devices");
        }
        device_open = true;
    }
    ~T3000TestDevice() {
        if (device_open) {
            TearDown();
        }
    }

    void TearDown() {
        device_open = false;
        tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);
        for (auto [device_id, device_ptr] : devices_) {
            tt::tt_metal::CloseDevice(device_ptr);
        }
    }

    std::map<chip_id_t, Device *> devices_;
    tt::ARCH arch_;
    size_t num_devices_;

   private:
    bool device_open;
};

namespace tt {

namespace tt_metal {


std::vector<uint32_t> get_eth_receiver_rt_args(
    Device *device,
    bool is_starting_core,
    uint32_t num_samples,
    uint32_t max_concurrent_samples,
    uint32_t sample_page_size,
    CoreCoord const& eth_sender_core,
    uint32_t start_semaphore,
    uint32_t init_handshake_core_x,
    uint32_t init_handshake_core_y,
    uint32_t init_handshake_semaphore_id
    ) {
    constexpr std::size_t semaphore_size = 16;
    std::vector<uint32_t> erisc_semaphore_addresses(max_concurrent_samples, eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 16 + 16);
    std::vector<uint32_t> erisc_buffer_addresses(max_concurrent_samples, eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 16 + 16 + round_up(semaphore_size * max_concurrent_samples, 16));
    for (std::size_t i = 0; i < max_concurrent_samples; i++) {
        erisc_semaphore_addresses.at(i) += i * semaphore_size;
        erisc_buffer_addresses.at(i) += i * sample_page_size;
    }

    std::vector<uint32_t> rt_args = {
        static_cast<uint32_t>(is_starting_core ? 1 : 0),
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE,
        num_samples,
        max_concurrent_samples,
        sample_page_size,
        static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_sender_core).x),
        static_cast<uint32_t>(device->ethernet_core_from_logical_core(eth_sender_core).y),
        start_semaphore,
        init_handshake_core_x,
        init_handshake_core_y,
        init_handshake_semaphore_id};
    for (std::size_t i = 0; i < max_concurrent_samples; i++) {
        rt_args.push_back(erisc_semaphore_addresses.at(i));
        rt_args.push_back(erisc_buffer_addresses.at(i));
    }

    return rt_args;
}


std::vector<uint32_t> get_eth_sender_rt_args(
    Device *device,
    bool is_starting_core,
    uint32_t num_samples,
    uint32_t max_concurrent_samples,
    uint32_t sample_page_size,
    uint32_t receiver_x,
    uint32_t receiver_y,
    uint32_t receiver_start_semaphore_id) {
    constexpr std::size_t semaphore_size = 16;
    std::vector<uint32_t> erisc_semaphore_addresses(max_concurrent_samples, eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 16 + 16);
    std::vector<uint32_t> erisc_buffer_addresses(max_concurrent_samples, eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 16 + 16 + round_up(semaphore_size * max_concurrent_samples, 16));
    for (std::size_t i = 0; i < max_concurrent_samples; i++) {
        erisc_semaphore_addresses.at(i) += i * semaphore_size;
        erisc_buffer_addresses.at(i) += i * sample_page_size;
    }

    std::vector<uint32_t> rt_args = {
        static_cast<uint32_t>(is_starting_core ? 1 : 0),
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE,
        num_samples,
        max_concurrent_samples,
        sample_page_size,
        receiver_x,
        receiver_y,
        receiver_start_semaphore_id};
    for (std::size_t i = 0; i < max_concurrent_samples; i++) {
        rt_args.push_back(erisc_semaphore_addresses.at(i));
        rt_args.push_back(erisc_buffer_addresses.at(i));
    }

    return rt_args;
}

struct hop_eth_sockets {
    chip_id_t receiver_device_id;
    CoreCoord receiver_core;
    chip_id_t sender_device_id;
    CoreCoord sender_core;
};


void build_and_run_roundtrip_latency_test(
    std::vector<Device*> devices,
    std::vector<hop_eth_sockets> hop_eth_sockets,
    std::size_t num_samples,
    std::size_t sample_page_size,
    std::size_t max_concurrent_samples,
    std::size_t n_hops,

    std::vector<Program> &programs,
    std::vector<KernelHandle> &receiver_kernel_ids,
    std::vector<KernelHandle> &sender_kernel_ids
) {
    TT_ASSERT(hop_eth_sockets.size() == devices.size());
    TT_ASSERT(n_hops == devices.size());
    TT_ASSERT(programs.size() == 0);
    TT_ASSERT(receiver_kernel_ids.size() == 0);
    TT_ASSERT(sender_kernel_ids.size() == 0);
    programs.reserve(n_hops);
    receiver_kernel_ids.reserve(n_hops);
    sender_kernel_ids.reserve(n_hops);

    std::unordered_map<Device*,Program*> device_program_map;
    for (std::size_t i = 0; i < n_hops; i++) {
        if (device_program_map.find(devices.at(i)) == device_program_map.end()) {
            programs.emplace_back();
            device_program_map[devices.at(i)] = &programs.back();
        }
    }

    std::unordered_map<Device*, uint32_t> device_visits;

    for (std::size_t i = 0; i < n_hops; i++) {
        auto previous_hop = i == 0 ? n_hops - 1 : i - 1;
        Device *device = devices.at(i);
        auto &program = *device_program_map.at(device);
        auto const& eth_sender_core = hop_eth_sockets.at(i).sender_core;
        auto const& eth_receiver_core = hop_eth_sockets.at(previous_hop).receiver_core;

        CoreCoord init_worker_core = CoreCoord{0, device_visits[device]++};

        uint32_t worker_sem0 = CreateSemaphore(program, init_worker_core, 0, CoreType::WORKER);
        uint32_t worker_sem1 = CreateSemaphore(program, init_worker_core, 0, CoreType::WORKER);

        std::vector<uint32_t> const& receiver_eth_ct_args = {};
        std::vector<uint32_t> const& sender_eth_ct_args = {};
        bool is_starting_core = i == 0;
        uint32_t receiver_start_semaphore = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 16;//CreateSemaphore(program, eth_receiver_core, 0, CoreType::ETH);
        log_trace(tt::LogTest, "is_starting_core: {}", (is_starting_core ? 1 : 0));
        std::vector<uint32_t> const& receiver_eth_rt_args = get_eth_receiver_rt_args(
            device,
            is_starting_core,
            num_samples,
            max_concurrent_samples,
            sample_page_size,
            eth_sender_core,
            receiver_start_semaphore,
            device->physical_core_from_logical_core(init_worker_core, CoreType::WORKER).x,
            device->physical_core_from_logical_core(init_worker_core, CoreType::WORKER).y,
            worker_sem0);
        std::vector<uint32_t> const& sender_eth_rt_args = get_eth_sender_rt_args(
            device,
            is_starting_core,
            num_samples,
            max_concurrent_samples,
            sample_page_size,
            device->physical_core_from_logical_core(init_worker_core, CoreType::WORKER).x,
            device->physical_core_from_logical_core(init_worker_core, CoreType::WORKER).y,
            worker_sem1);

        std::vector<uint32_t> worker_init_rt_args = {
            worker_sem0,
            worker_sem1,
            static_cast<uint32_t>(device->physical_core_from_logical_core(eth_receiver_core, CoreType::ETH).x),
            static_cast<uint32_t>(device->physical_core_from_logical_core(eth_receiver_core, CoreType::ETH).y),
            receiver_start_semaphore
        };

        auto receiver_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_latency_ubench_eth_receiver.cpp",
            eth_receiver_core,
            tt_metal::EthernetConfig {
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = receiver_eth_ct_args});
        receiver_kernel_ids.push_back(receiver_kernel);

        auto sender_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_latency_ubench_eth_sender.cpp",
            eth_sender_core,
            tt_metal::EthernetConfig {
                .noc = tt_metal::NOC::RISCV_1_default,
                .compile_args = sender_eth_ct_args});
        sender_kernel_ids.push_back(sender_kernel);

        // This guy is only used until fast dispatch 2 is available
        // it coordinates eth core ready states so the receiver and sender only
        // communicate after they are initialized
        auto worker_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_latency_ubench_init_coordination_worker.cpp",
            init_worker_core,
            tt_metal::DataMovementConfig {
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_1_default,
                .compile_args = {}});


        log_trace(tt::LogOp, "-------------Hop: {}, Device: {}:", i, device->id());
        log_trace(tt::LogOp, "Receiver Kernel Info: Receives from {} on core[logical]: (x={},y={}), [noc]: (x={},y={}):", devices.at(previous_hop)->id(), eth_receiver_core.x, eth_receiver_core.y, device->ethernet_core_from_logical_core(eth_receiver_core).x, device->ethernet_core_from_logical_core(eth_receiver_core).y);
        log_trace(tt::LogOp, "- RT Args ({})", receiver_eth_rt_args.size());
        for (std::size_t i = 0; i < receiver_eth_rt_args.size(); i++) {
            log_trace(tt::LogOp, "  - {}: {}", i, receiver_eth_rt_args.at(i));
        }
        log_trace(tt::LogOp, "Sender Kernel Info: on core[logical]: (x={},y={}), [noc]: (x={},y={}):", eth_sender_core.x, eth_sender_core.y, device->ethernet_core_from_logical_core(eth_sender_core).x, device->ethernet_core_from_logical_core(eth_sender_core).y);
        for (std::size_t i = 0; i < sender_eth_rt_args.size(); i++) {
            log_trace(tt::LogOp, "  - {}: {}", i, sender_eth_rt_args.at(i));
        }
        log_trace(tt::LogOp, "Worker Kernel Info: on core[logical]: (x={},y={})", init_worker_core.x, init_worker_core.y);
        for (std::size_t i = 0; i < worker_init_rt_args.size(); i++) {
            log_trace(tt::LogOp, "  - {}: {}", i, worker_init_rt_args.at(i));
        }

        tt_metal::SetRuntimeArgs(program, receiver_kernel, eth_receiver_core, receiver_eth_rt_args);
        tt_metal::SetRuntimeArgs(program, sender_kernel, eth_sender_core, sender_eth_rt_args);
        log_trace(tt::LogOp, "Setting RT args for receiver. kernel_id: {}, core: (x={},y={})", receiver_kernel, eth_receiver_core.x, eth_receiver_core.y);
        log_trace(tt::LogOp, "Setting RT args for sender. kernel_id: {}, core: (x={},y={})", sender_kernel, eth_sender_core.x, eth_sender_core.y);
        tt_metal::SetRuntimeArgs(program, worker_kernel, init_worker_core, worker_init_rt_args);
        log_trace(tt::LogOp, "Setting RT args for worker. kernel_id: {}, core: (x={},y={})", worker_kernel, init_worker_core.x, init_worker_core.y);

        tt::tt_metal::detail::CompileProgram(device, program);
    }

    for (auto [device_ptr, program_ptr] : device_program_map) {
        tt_metal::EnqueueProgram(device_ptr->command_queue(), *program_ptr, false);
    }

    for (auto [device_ptr, program_ptr] : device_program_map) {
        tt_metal::Finish(device_ptr->command_queue());
    }

    for (auto [device_ptr, program_ptr] : device_program_map) {
        tt::tt_metal::detail::DumpDeviceProfileResults(device_ptr);
    }

}

}  // namespace tt_metal

}  // namespace tt


auto is_device_pcie_connected(chip_id_t device_id) {
    return device_id < 4;
}


std::vector<hop_eth_sockets> build_eth_sockets_list(std::vector<Device*> const& devices) {
    std::vector<hop_eth_sockets> sockets;
    std::unordered_map<uint64_t, std::size_t> n_edge_visits;
    for (std::size_t i = 0; i < devices.size(); i++) {
        Device *curr_device = devices.at(i);
        Device *next_device = i == devices.size() - 1 ? devices.at(0) : devices.at(i + 1);
        uint64_t edge = (static_cast<uint64_t>(curr_device->id()) << 32) | static_cast<uint64_t>(next_device->id());
        bool edge_needs_tunneling = !is_device_pcie_connected(curr_device->id()) || !is_device_pcie_connected(next_device->id());


        std::size_t conn = (edge_needs_tunneling ? 0 : 0) + n_edge_visits[edge];
        std::size_t link = 0;
        std::unordered_map<uint64_t, int> edge_link_idx;
        auto const& active_eth_cores = curr_device->get_active_ethernet_cores(true);
        auto eth_sender_core_iter = active_eth_cores.begin();
        bool found = false;
        for (; !found && eth_sender_core_iter != active_eth_cores.end(); eth_sender_core_iter++) {

            auto [device_id, receiver_core] = curr_device->get_connected_ethernet_core(*eth_sender_core_iter);
            if (device_id == next_device->id()) {
                uint64_t pair_edge = (static_cast<uint64_t>(curr_device->id()) << 32) | static_cast<uint64_t>(device_id);
                if (edge_link_idx[pair_edge] == conn) {
                    CoreCoord eth_sender_core = *eth_sender_core_iter;
                    CoreCoord eth_receiver_core = receiver_core;
                    chip_id_t receiver_device_id = device_id;
                    sockets.push_back({receiver_device_id,eth_receiver_core,curr_device->id(),eth_sender_core});
                    TT_ASSERT(receiver_device_id == next_device->id());
                    found = true;
                    break;
                }
                edge_link_idx[pair_edge]++;
            }
        }
        TT_ASSERT(eth_sender_core_iter != active_eth_cores.end());
        TT_ASSERT(found);

        n_edge_visits[edge] += 1;
    }

    return sockets;
}



int main (int argc, char** argv) {
    // num samples
    // page sizes
    // concurrent samples
    // hop counts
    // Early exit if invalid test setup
    auto arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices != 8) {
        log_trace(tt::LogTest, "Need at least 2 devices to run this test");
        return 0;
    }
    if (arch == tt::ARCH::GRAYSKULL) {
        log_trace(tt::LogTest,"Test must be run on WH");
        return 0;
    }

    // Arg setup
    assert(argc >= 4);
    std::size_t arg_idx = 1;
    std::size_t num_sample_counts = std::stoi(argv[arg_idx++]);
    log_trace(tt::LogTest, "num_sample_counts: {}", num_sample_counts);
    std::vector<std::size_t> sample_counts;
    for (std::size_t i = 0; i < num_sample_counts; i++) {
        sample_counts.push_back(std::stoi(argv[arg_idx++]));
        log_trace(tt::LogTest, "sample_counts[{}]: {}", i, sample_counts.back());
    }

    std::size_t num_page_sizes = std::stoi(argv[arg_idx++]);
    std::vector<std::size_t> page_sizes;
    log_trace(tt::LogTest, "num_page_sizes: {}", num_page_sizes);
    for (std::size_t i = 0; i < num_page_sizes; i++) {
        page_sizes.push_back(std::stoi(argv[arg_idx++]));
        log_trace(tt::LogTest, "page_sizes[{}]: {}", i, page_sizes.back());
    }

    std::size_t num_max_concurrent_samples = std::stoi(argv[arg_idx++]);
    std::vector<std::size_t> max_concurrent_samples;
    log_trace(tt::LogTest, "num_max_concurrent_samples: {}", num_max_concurrent_samples);
    for (std::size_t i = 0; i < num_max_concurrent_samples; i++) {
        max_concurrent_samples.push_back(std::stoi(argv[arg_idx++]));
        log_trace(tt::LogTest, "max_concurrent_samples[{}]: {}", i, max_concurrent_samples.back());
    }

    std::size_t num_hop_counts = std::stoi(argv[arg_idx++]);
    std::vector<uint32_t> hop_counts;
    log_trace(tt::LogTest, "num_hop_counts: {}", num_hop_counts);
    for (std::size_t i = 0; i < num_hop_counts; i++) {
        hop_counts.push_back(std::stoi(argv[arg_idx++]));
        log_trace(tt::LogTest, "hop_counts[{}]: {}", i, hop_counts.back());
    }
    TT_ASSERT(argc == arg_idx);

    // Arg Validation
    TT_ASSERT(std::all_of(sample_counts.begin(), sample_counts.end(), [](std::size_t n) { return n > 0; }));
    TT_ASSERT(std::all_of(page_sizes.begin(), page_sizes.end(), [](std::size_t n) { return n > 0; }));
    TT_ASSERT(std::all_of(page_sizes.begin(), page_sizes.end(), [](std::size_t n) { return n % 16 == 0; }));
    TT_ASSERT(std::all_of(max_concurrent_samples.begin(), max_concurrent_samples.end(), [](std::size_t n) { return n > 0; }));

    T3000TestDevice test_fixture;

    // Device setup
    std::vector<chip_id_t> device_ids = std::vector<chip_id_t>{0, 1, 2, 3, 4, 5, 6, 7};

    auto get_device_list = [](std::map<chip_id_t, Device*> &all_devices, std::size_t n_hops) {
        switch (n_hops) {
            case 2:
                return std::vector<Device*>{all_devices[0], all_devices[1]};

            case 4:
                return std::vector<Device*>{all_devices[0], all_devices[1], all_devices[2], all_devices[3]};

            case 8:
                return std::vector<Device*>{all_devices[0], all_devices[4], all_devices[5], all_devices[1], all_devices[2], all_devices[6], all_devices[7], all_devices[3]};

            case 12: // Does an extra loop through the inner ring
                return std::vector<Device*>{all_devices[0], all_devices[4], all_devices[5], all_devices[1], all_devices[2], all_devices[3], all_devices[0], all_devices[1], all_devices[2], all_devices[6], all_devices[7], all_devices[3]};

            default:
                TT_THROW("Unsupported hop_count");
                return std::vector<Device*>{};
        };
    };

    try {
        constexpr std::size_t placeholder_arg_value = 1;
        for (auto n_hops : hop_counts) {

            auto devices = get_device_list(test_fixture.devices_, n_hops);
            std::vector<hop_eth_sockets> hop_eth_sockets = build_eth_sockets_list(devices);

            for (auto max_concurrent_samples : max_concurrent_samples) {
                for (auto num_samples : sample_counts) {
                    for (auto sample_page_size : page_sizes) {
                        log_trace(tt::LogTest, "Running test with num_devices={}, num_samples={}, sample_page_size={}, max_concurrent_samples={}, n_hops={}",
                            n_hops, num_samples, sample_page_size, max_concurrent_samples, n_hops);
                        std::vector<Program> programs = {};
                        std::vector<KernelHandle> receiver_kernel_ids;
                        std::vector<KernelHandle> sender_kernel_ids;
                        tt::tt_metal::build_and_run_roundtrip_latency_test(
                            devices,
                            hop_eth_sockets,
                            num_samples,
                            sample_page_size,
                            max_concurrent_samples,
                            n_hops,

                            programs,
                            receiver_kernel_ids,
                            sender_kernel_ids
                        );
                    }
                }
            }
        }
    } catch (std::exception e) {
        test_fixture.TearDown();
        return -1;
    }

    return 0;
}
