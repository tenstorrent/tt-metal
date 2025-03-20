
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <limits>
#include <random>
#include <tuple>
#include <map>
#include <set>
#include <vector>
#include <queue>

#include "umd/device/types/arch.h"
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/device_impl.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "tt_backend_api_types.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/impl/profiler/profiler_paths.hpp"

#include <tt-metalium/persistent_kernel_cache.hpp>
#include <thread>

#include "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_ubenchmark_types.hpp"

// TODO: ARCH_NAME specific, must remove
#include "eth_l1_address_map.h"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

struct TestParams {
    BenchmarkType benchmark_type;
    uint32_t num_samples;
    uint32_t sample_page_size;
    uint32_t num_buffer_slots;
    bool test_latency;
    bool disable_trid;
    uint32_t num_iterations;
};

struct LinkStats {
    uint32_t retrain_count = 0;
    uint32_t crc_errs = 0;
    uint32_t pcs_faults = 0;
    uint64_t total_corr_cw = 0;
    uint64_t total_uncorr_cw = 0;
    uint32_t retrains_triggered_by_pcs = 0;
    uint32_t retrains_triggered_by_crcs = 0;
};

struct SenderReceiverPair {
    tt_cxy_pair sender;
    tt_cxy_pair receiver;

    const bool operator==(const SenderReceiverPair& other) const {
        bool same_s_r = sender.chip == other.sender.chip && sender.x == other.sender.x && sender.y == other.sender.y &&
                        receiver.chip == other.receiver.chip && receiver.x == other.receiver.x &&
                        receiver.y == other.receiver.y;
        bool r_is_s = sender.chip == other.receiver.chip && sender.x == other.receiver.x &&
                      sender.y == other.receiver.y && receiver.chip == other.sender.chip &&
                      receiver.x == other.sender.x && receiver.y == other.sender.y;
        return same_s_r || r_is_s;
    }

    bool operator<(const SenderReceiverPair& other) const {
        if (*this == other) {
            return false;
        }
        // this and other are diff
        // this sender before the other sender means less then
        auto sender_coord = CoreCoord(sender.x, sender.y);
        auto receiver_coord = CoreCoord(receiver.x, receiver.y);
        auto other_sender_coord = CoreCoord(other.sender.x, other.sender.y);
        auto other_receiver_coord = CoreCoord(other.receiver.x, other.receiver.y);

        // Order based on sender
        bool result = (sender.chip < other.sender.chip) ||
                      (sender.chip == other.sender.chip && sender_coord < other_sender_coord) ||
                      (sender.chip == other.sender.chip && receiver.chip < other.receiver.chip) ||
                      (sender.chip == other.sender.chip && receiver.chip == other.receiver.chip &&
                       receiver_coord < other_receiver_coord);
        return result;
    }
};

class ConnectedDevicesHelper {
public:
    ConnectedDevicesHelper() : device_open_(false) {
        this->arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        this->num_devices = tt::tt_metal::GetNumAvailableDevices();
        std::vector<chip_id_t> ids(this->num_devices, 0);
        std::iota(ids.begin(), ids.end(), 0);

        const auto& dispatch_core_config = tt::llrt::RunTimeOptions::get_instance().get_dispatch_core_config();
        tt::DevicePool::initialize(ids, 1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, dispatch_core_config);
        this->devices = tt::DevicePool::instance().get_all_active_devices();

        this->initialize_sender_receiver_pairs();
        device_open_ = true;
    }

    ~ConnectedDevicesHelper() {
        if (device_open_) {
            TearDown();
        }
    }

    void TearDown() {
        device_open_ = false;
        tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);
        for (unsigned int id = 0; id < this->devices.size(); id++) {
            tt::tt_metal::CloseDevice(this->devices.at(id));
        }
    }

    std::vector<tt_metal::IDevice*> devices;
    tt::ARCH arch;
    size_t num_devices;
    std::set<SenderReceiverPair> unique_links;

private:
    void initialize_sender_receiver_pairs() {
        // chip id -> active eth ch on chip -> (connected chip, remote active eth ch)
        auto all_eth_connections = tt::Cluster::instance().get_ethernet_connections();

        std::set<chip_id_t> sender_chips, receiver_chips;
        bool slow_dispath_mode = (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr);

        std::queue<chip_id_t> chip_q;
        chip_q.push(this->devices.at(0)->id());
        sender_chips.insert(this->devices.at(0)->id());

        // Need sender and receiver chips to be disjoint because we profile wrt. sender and don't want devices to be out
        // of sync
        while (!chip_q.empty()) {
            chip_id_t chip_id = chip_q.front();
            chip_q.pop();

            bool is_sender = sender_chips.find(chip_id) != sender_chips.end();
            bool is_receiver = receiver_chips.find(chip_id) != receiver_chips.end();

            for (chip_id_t connected_chip : tt::Cluster::instance().get_ethernet_connected_device_ids(chip_id)) {
                bool connected_chip_is_sender = sender_chips.find(connected_chip) != sender_chips.end();
                bool connected_chip_is_receiver = receiver_chips.find(connected_chip) != receiver_chips.end();
                if (!connected_chip_is_sender and !connected_chip_is_receiver) {
                    if (is_sender) {
                        receiver_chips.insert(connected_chip);
                    } else {
                        TT_FATAL(is_receiver, "Chip {} should be marked as a receiver", chip_id);
                        sender_chips.insert(connected_chip);
                    }
                    chip_q.push(connected_chip);
                } else if (is_sender && connected_chip_is_sender) {
                    TT_FATAL(false, "Chip {} and connected chip {} are both senders!", chip_id, connected_chip);
                } else if (is_receiver && connected_chip_is_receiver) {
                    TT_FATAL(false, "Chip {} and connected chip {} are both receivers!", chip_id, connected_chip);
                }
            }
        }

        std::vector<int> s_r_chip_intersection;
        std::set_intersection(
            sender_chips.begin(),
            sender_chips.end(),
            receiver_chips.begin(),
            receiver_chips.end(),
            std::back_inserter(s_r_chip_intersection));
        TT_FATAL(s_r_chip_intersection.empty(), "Expected no overlap between senders and receivers");

        for (auto sender_chip_id : sender_chips) {
            auto non_tunneling_eth_cores =
                tt::Cluster::instance().get_active_ethernet_cores(sender_chip_id, !slow_dispath_mode);
            for (auto logical_active_eth : non_tunneling_eth_cores) {
                auto sender_eth = tt_cxy_pair(sender_chip_id, logical_active_eth);
                auto receiver_eth_tuple = tt::Cluster::instance().get_connected_ethernet_core(
                    std::make_tuple(sender_chip_id, logical_active_eth));
                auto receiver_eth = tt_cxy_pair(std::get<0>(receiver_eth_tuple), std::get<1>(receiver_eth_tuple));
                this->unique_links.insert(SenderReceiverPair{.sender = sender_eth, .receiver = receiver_eth});
            }
        }

        // std::cout << "Printing unique links" << std::endl;
        // for (auto link : this->unique_links) {
        //     std::cout << "Sender " << link.sender.str() << " Receiver " << link.receiver.str() << std::endl;
        // }
    }

    bool device_open_;
};

std::vector<tt_metal::Program> build(const ConnectedDevicesHelper& device_helper, const TestParams& params) {
    std::vector<tt_metal::Program> programs(device_helper.num_devices);

    uint32_t measurement_type = (uint32_t)(params.test_latency ? MeasurementType::Latency : MeasurementType::Bandwidth);
    uint32_t benchmark_type_val = magic_enum::enum_integer(params.benchmark_type);

    // eth core ct args
    const std::vector<uint32_t>& eth_sender_ct_args = {
        benchmark_type_val,
        measurement_type,
        params.num_buffer_slots,
        0,  // worker_noc_x should make rt
        0,  // worker_noc_y should make rt
        0,  // worker_buffer_0_addr, make rt
        uint32_t(params.disable_trid)};

    const std::vector<uint32_t>& eth_receiver_ct_args = {
        benchmark_type_val,
        measurement_type,
        params.num_buffer_slots,
        0,  // worker_noc_x should make rt
        0,  // worker_noc_y should make rt
        0,  // worker_buffer_1_addr, make rt
        uint32_t(params.disable_trid)};

    // eth core rt args
    const std::vector<uint32_t>& eth_receiver_rt_args = {
        tt_metal::hal.get_dev_addr(
            tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt_metal::HalL1MemAddrType::UNRESERVED),
        static_cast<uint32_t>(params.num_samples),
        static_cast<uint32_t>(params.sample_page_size)};

    std::vector<uint32_t> eth_sender_rt_args = eth_receiver_rt_args;
    // these are overwritten below with sender/receiver link encoding
    eth_sender_rt_args.push_back(0);
    eth_sender_rt_args.push_back(0);

    for (const auto& link : device_helper.unique_links) {
        auto& sender_program = programs.at(link.sender.chip);
        auto& receiver_program = programs.at(link.receiver.chip);

        auto virtual_sender =
            device_helper.devices.at(link.sender.chip)->virtual_core_from_logical_core(link.sender, CoreType::ETH);
        auto virtual_receiver =
            device_helper.devices.at(link.receiver.chip)->virtual_core_from_logical_core(link.receiver, CoreType::ETH);

        // Pass sender/receiver tt_cxy_pair to sender kernel so we can add link info
        uint32_t sender_encoding =
            ((uint8_t)link.sender.chip << 0x10) | ((uint8_t)virtual_sender.x << 0x8) | (uint8_t)virtual_sender.y;
        uint32_t receiver_encoding =
            ((uint8_t)link.receiver.chip << 0x10) | ((uint8_t)virtual_receiver.x << 0x8) | (uint8_t)virtual_receiver.y;
        eth_sender_rt_args[eth_sender_rt_args.size() - 2] = sender_encoding;
        eth_sender_rt_args[eth_sender_rt_args.size() - 1] = receiver_encoding;

        auto sender_kernel = tt_metal::CreateKernel(
            sender_program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/"
            "ethernet_write_worker_latency_ubench_sender.cpp",
            CoreCoord(link.sender.x, link.sender.y),
            tt_metal::EthernetConfig{.noc = tt_metal::NOC::RISCV_0_default, .compile_args = eth_sender_ct_args});
        tt_metal::SetRuntimeArgs(
            sender_program, sender_kernel, CoreCoord(link.sender.x, link.sender.y), eth_sender_rt_args);

        auto receiver_kernel = tt_metal::CreateKernel(
            receiver_program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/"
            "ethernet_write_worker_latency_ubench_receiver.cpp",
            CoreCoord(link.receiver.x, link.receiver.y),
            tt_metal::EthernetConfig{.noc = tt_metal::NOC::RISCV_0_default, .compile_args = eth_receiver_ct_args});
        tt_metal::SetRuntimeArgs(
            receiver_program, receiver_kernel, CoreCoord(link.receiver.x, link.receiver.y), eth_receiver_rt_args);
    }

    for (auto device : device_helper.devices) {
        try {
            tt_metal::detail::CompileProgram(device, programs.at(device->id()));
        } catch (std::exception& e) {
            log_error(tt::LogTest, "Failed to compile program on device {}: {}", device->id(), e.what());
            throw e;
        }
    }

    return programs;
}

void validation(const std::shared_ptr<tt::tt_metal::Buffer>& worker_buffer_0) {
    std::vector<uint8_t> golden_vec(worker_buffer_0->size(), 0);
    std::vector<uint8_t> result_vec(worker_buffer_0->size(), 0);

    for (int i = 0; i < worker_buffer_0->size(); ++i) {
        golden_vec[i] = i;
    }
    tt::tt_metal::detail::ReadFromBuffer(worker_buffer_0, result_vec);

    bool pass = golden_vec == result_vec;
    TT_FATAL(pass, "validation failed");
}

tt_metal::IDevice* find_device_with_id(const ConnectedDevicesHelper& device_helper, chip_id_t chip_id) {
    for (auto device : device_helper.devices) {
        if (device->id() == chip_id) {
            return device;
        }
    }
    TT_FATAL(false, "Unexpected device id {}", chip_id);
    return nullptr;
}

void dump_eth_link_stats(
    const ConnectedDevicesHelper& device_helper,
    uint32_t iteration,
    std::map<tt_cxy_pair, std::vector<LinkStats>>& sender_stats,
    std::map<tt_cxy_pair, std::vector<LinkStats>>& receiver_stats,
    uint32_t total_iterations,
    bool write_header = false) {
    if (device_helper.arch == ARCH::BLACKHOLE) {
        return;  // link stats not populated for BH yet!
    }

    std::filesystem::path output_dir = std::filesystem::path(tt::tt_metal::get_profiler_logs_dir());
    std::filesystem::path log_path = output_dir / "eth_link_stats.csv";
    std::ofstream log_file;
    if (write_header) {
        log_file.open(log_path, std::ios_base::out);
        log_file << fmt::format(
            "Iteration,Sender Device ID,Sender X,Sender Y,S Retrain Count,S CRC Errs,S PCS Faults,S Total Corr,S Total "
            "Uncorr,S Retrain by PCS,S Retrain by CRC");
        log_file << fmt::format(
                        ",Receiver Device ID,Receiver X,Receiver Y,R Retrain Count,R CRC Errs,R PCS Faults,R Total "
                        "Corr,R Total Uncorr,R Retrain by PCS,R Retrain by CRC")
                 << std::endl;
    } else {
        log_file.open(log_path, std::ios_base::app);
    }

    uint32_t prev_iter = iteration == 0 ? iteration : iteration - 1;
    // from base FW (WH) we want to read up to test_results[51]
    std::vector<uint32_t> link_stats(56, 0);
    constexpr uint32_t link_stats_base_addr = 0x1EC0;
    for (const auto& link : device_helper.unique_links) {
        auto sender_device = find_device_with_id(device_helper, link.sender.chip);
        auto receiver_device = find_device_with_id(device_helper, link.receiver.chip);
        auto sender_virtual =
            sender_device->virtual_core_from_logical_core(CoreCoord(link.sender.x, link.sender.y), CoreType::ETH);
        auto receiver_virtual =
            receiver_device->virtual_core_from_logical_core(CoreCoord(link.receiver.x, link.receiver.y), CoreType::ETH);

        tt::Cluster::instance().read_core(
            link_stats.data(),
            link_stats.size() * sizeof(uint32_t),
            tt_cxy_pair(link.sender.chip, sender_virtual),
            link_stats_base_addr);

        auto& s_stats_per_iter = sender_stats[link.sender];
        auto& r_stats_per_iter = receiver_stats[link.receiver];
        if (s_stats_per_iter.empty()) {
            TT_ASSERT(r_stats_per_iter.empty());
            s_stats_per_iter.resize(total_iterations);
            r_stats_per_iter.resize(total_iterations);
        }

        s_stats_per_iter[iteration] = LinkStats{
            .retrain_count = write_header ? link_stats[7] : link_stats[7] - s_stats_per_iter[prev_iter].retrain_count,
            .crc_errs = write_header ? link_stats[47] : link_stats[47] - s_stats_per_iter[prev_iter].crc_errs,
            .pcs_faults = link_stats[49],
            .total_corr_cw = write_header ? (uint64_t(link_stats[52]) << 32) | link_stats[53]
                                          : ((uint64_t(link_stats[52]) << 32) | link_stats[53]) -
                                                s_stats_per_iter[prev_iter].total_corr_cw,
            .total_uncorr_cw = write_header ? (uint64_t(link_stats[54]) << 32) | link_stats[55]
                                            : ((uint64_t(link_stats[54]) << 32) | link_stats[55]) -
                                                  s_stats_per_iter[prev_iter].total_uncorr_cw,
            .retrains_triggered_by_pcs =
                write_header ? link_stats[50] : link_stats[50] - s_stats_per_iter[prev_iter].retrains_triggered_by_pcs,
            .retrains_triggered_by_crcs =
                write_header ? link_stats[51]
                             : link_stats[51] - s_stats_per_iter[prev_iter].retrains_triggered_by_crcs};

        // std::cout << "Sender retrain: " << link_stats[7]
        //           << " crc: " << link_stats[47]
        //           << " pcs: " << link_stats[49]
        //           << " total corr: " << link_stats[52] << " " << link_stats[53]
        //           << " total uncorr: " << link_stats[54] << " " << link_stats[55]
        //           << " retrain by pcs: " << link_stats[50]
        //           << " retrains by crcs: " << link_stats[51] << std::endl;

        tt::Cluster::instance().read_core(
            link_stats.data(),
            link_stats.size() * sizeof(uint32_t),
            tt_cxy_pair(link.receiver.chip, receiver_virtual),
            link_stats_base_addr);

        r_stats_per_iter[iteration] = LinkStats{
            .retrain_count = write_header ? link_stats[7] : link_stats[7] - r_stats_per_iter[prev_iter].retrain_count,
            .crc_errs = write_header ? link_stats[47] : link_stats[47] - r_stats_per_iter[prev_iter].crc_errs,
            .pcs_faults = link_stats[49],
            .total_corr_cw = write_header ? (uint64_t(link_stats[52]) << 32) | link_stats[53]
                                          : ((uint64_t(link_stats[52]) << 32) | link_stats[53]) -
                                                r_stats_per_iter[prev_iter].total_corr_cw,
            .total_uncorr_cw = write_header ? (uint64_t(link_stats[54]) << 32) | link_stats[55]
                                            : ((uint64_t(link_stats[54]) << 32) | link_stats[55]) -
                                                  r_stats_per_iter[prev_iter].total_uncorr_cw,
            .retrains_triggered_by_pcs =
                write_header ? link_stats[50] : link_stats[50] - r_stats_per_iter[prev_iter].retrains_triggered_by_pcs,
            .retrains_triggered_by_crcs =
                write_header ? link_stats[51]
                             : link_stats[51] - r_stats_per_iter[prev_iter].retrains_triggered_by_crcs};

        // std::cout << "Receiver retrain: " << link_stats[7]
        //     << " crc: " << link_stats[47]
        //     << " pcs: " << link_stats[49]
        //     << " total corr: " << link_stats[52] << " " << link_stats[53]
        //     << " total uncorr: " << link_stats[54] << " " << link_stats[55]
        //     << " retrain by pcs: " << link_stats[50]
        //     << " retrains by crcs: " << link_stats[51] << std::endl;

        if (not write_header) {
            const auto& s = s_stats_per_iter[iteration];
            const auto& r = r_stats_per_iter[iteration];

            log_file << fmt::format(
                "{},{},{},{},{},{},{},{},{},{},{}",
                iteration,
                link.sender.chip,
                sender_virtual.x,
                sender_virtual.y,
                s.retrain_count,
                s.crc_errs,
                s.pcs_faults,
                s.total_corr_cw,
                s.total_uncorr_cw,
                s.retrains_triggered_by_pcs,
                s.retrains_triggered_by_crcs);

            log_file << fmt::format(
                            ",{},{},{},{},{},{},{},{},{},{}",
                            link.receiver.chip,
                            receiver_virtual.x,
                            receiver_virtual.y,
                            r.retrain_count,
                            r.crc_errs,
                            r.pcs_faults,
                            r.total_corr_cw,
                            r.total_uncorr_cw,
                            r.retrains_triggered_by_pcs,
                            r.retrains_triggered_by_crcs)
                     << std::endl;
        }
    }

    log_file.close();
}

void run(
    const ConnectedDevicesHelper& device_helper, std::vector<tt_metal::Program>& programs, const TestParams& params) {
    // Collect stats per iteration
    std::map<tt_cxy_pair, std::vector<LinkStats>> sender_stats;
    std::map<tt_cxy_pair, std::vector<LinkStats>> receiver_stats;

    dump_eth_link_stats(device_helper, 0, sender_stats, receiver_stats, params.num_iterations, true);

    bool slow_dispath_mode = std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr;
    for (uint32_t iteration = 0; iteration < params.num_iterations; iteration++) {
        if (slow_dispath_mode) {
            std::vector<std::thread> threads;
            for (int i = 0; i < device_helper.devices.size(); ++i) {
                threads.emplace_back([&]() {
                    tt_metal::detail::LaunchProgram(
                        device_helper.devices.at(i), programs.at(device_helper.devices.at(i)->id()));
                });

                for (auto& thread : threads) {
                    thread.join();
                }
            }
        } else {
            for (auto device : device_helper.devices) {
                tt_metal::EnqueueProgram(device->command_queue(), programs.at(device->id()), false);
            }
            log_info(tt::LogTest, "Iteration {} Calling Finish", iteration);
            for (auto device : device_helper.devices) {
                tt_metal::Finish(device->command_queue());
            }
        }
        dump_eth_link_stats(device_helper, iteration, sender_stats, receiver_stats, params.num_iterations);
    }

    // Only dump profile results from sender
    for (const auto& link : device_helper.unique_links) {
        tt_metal::detail::DumpDeviceProfileResults(device_helper.devices.at(link.sender.chip));
    }

    // for (auto device : device_helper.devices) {
    //     tt_metal::detail::DumpDeviceProfileResults(device);
    // }

    // TODO: add validation to make sure we got the expected data!
    // if (benchmark_type == BenchmarkType::EthEthTensixUniDir or benchmark_type == BenchmarkType::EthEthTensixBiDir) {
    //     validation(worker_buffer_1);
    //     if (benchmark_type == BenchmarkType::EthEthTensixBiDir) {
    //         validation(worker_buffer_0);
    //     }
    // }
}

int main(int argc, char** argv) {
    std::size_t arg_idx = 1;
    uint32_t benchmark_type = (uint32_t)std::stoi(argv[arg_idx++]);

    auto benchmark_type_enum = magic_enum::enum_cast<BenchmarkType>(benchmark_type);
    TT_FATAL(
        benchmark_type_enum.has_value(),
        "Unsupported benchmark {} specified, check BenchmarkType enum for supported values",
        benchmark_type);

    std::size_t num_samples = std::stoi(argv[arg_idx++]);
    std::size_t sample_page_size = std::stoi(argv[arg_idx++]);
    std::size_t num_buffer_slots = std::stoi(argv[arg_idx++]);

    bool test_latency = std::stoi(argv[arg_idx++]);
    bool disable_trid = std::stoi(argv[arg_idx++]);
    uint32_t num_iterations = std::stoi(argv[arg_idx++]);

    TestParams params{
        .benchmark_type = benchmark_type_enum.value(),
        .num_samples = num_samples,
        .sample_page_size = sample_page_size,
        .num_buffer_slots = num_buffer_slots,
        .test_latency = test_latency,
        .disable_trid = disable_trid,
        .num_iterations = num_iterations};

    log_info(tt::LogTest, "Setting up test fixture");
    ConnectedDevicesHelper device_helper;
    log_info(tt::LogTest, "Done setting up test fixture");
    if (device_helper.num_devices < 2) {
        log_info(tt::LogTest, "Need at least 2 devices to run this test");
        return 0;
    }

    TT_FATAL(
        params.benchmark_type == BenchmarkType::EthOnlyUniDir || params.benchmark_type == BenchmarkType::EthOnlyBiDir,
        "Need to uplift for other cases");

    // worker
    auto worker_core = CoreCoord(0, 0);
    // Add more configurations here until proper argc parsing added
    bool success = false;
    success = true;
    log_info(tt::LogTest, "STARTING");
    try {
        log_info(
            tt::LogTest,
            "benchmark type: {}, measurement type: {}, num_samples: {}, sample_page_size: {}, num_buffer_slots: {}",
            magic_enum::enum_name(benchmark_type_enum.value()),
            magic_enum::enum_name(test_latency ? MeasurementType::Latency : MeasurementType::Bandwidth),
            num_samples,
            sample_page_size,
            num_buffer_slots);
        // tt_metal::ShardSpecBuffer shard_spec = tt_metal::ShardSpecBuffer(
        //     CoreRangeSet(std::set<CoreRange>({CoreRange(worker_core)})),
        //     {1, sample_page_size},
        //     tt_metal::ShardOrientation::ROW_MAJOR,
        //     {1, sample_page_size},
        //     {1, sample_page_size});
        // auto worker_buffer_0 = CreateBuffer(tt::tt_metal::ShardedBufferConfig{
        //     .device = device_0,
        //     .size = sample_page_size,
        //     .page_size = sample_page_size,
        //     .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        //     .shard_parameters = shard_spec});
        // auto worker_buffer_1 = CreateBuffer(tt::tt_metal::ShardedBufferConfig{
        //     .device = device_1,
        //     .size = sample_page_size,
        //     .page_size = sample_page_size,
        //     .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        //     .shard_parameters = shard_spec});

        auto programs = build(device_helper, params);
        run(device_helper, programs, params);

    } catch (std::exception& e) {
        log_error(tt::LogTest, "Caught exception: {}", e.what());
        device_helper.TearDown();
        return -1;
    }

    return success ? 0 : -1;
}
