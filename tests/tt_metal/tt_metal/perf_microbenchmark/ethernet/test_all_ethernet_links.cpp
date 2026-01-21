
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <tuple>
#include <map>
#include <set>
#include <vector>
#include <queue>
#include <optional>
#include <fstream>

#include <umd/device/types/arch.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-metalium/host_api.hpp>
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/impl/profiler/profiler_paths.hpp"

#include <thread>
#include "impl/context/metal_context.hpp"
#include "common/tt_backend_api_types.hpp"

#include "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_ubenchmark_types.hpp"

#include <enchantum/enchantum.hpp>
#include <llrt/tt_cluster.hpp>

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

struct TestParams {
    BenchmarkType benchmark_type;
    uint32_t num_packets;
    uint32_t packet_size;
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
    // In bidirectional mode each eth core acts as sender and receiver but we still have an assigned "base" mode
    // Measurements are always taken from the "base" sender
    tt_cxy_pair sender;
    tt_cxy_pair receiver;

    // In test modes where receiver writes results to tensix we
    std::optional<CoreCoord> sender_tensix = std::nullopt;
    std::optional<CoreCoord> receiver_tensix = std::nullopt;

    std::shared_ptr<tt_metal::distributed::MeshBuffer> sender_buffer = nullptr;
    std::shared_ptr<tt_metal::distributed::MeshBuffer> receiver_buffer = nullptr;

    bool operator==(const SenderReceiverPair& other) const {
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

tt_metal::distributed::MeshDevice* find_device_with_id(
    const std::vector<std::shared_ptr<tt_metal::distributed::MeshDevice>>& devices, ChipId chip_id) {
    for (const auto& device : devices) {
        if (device->get_devices()[0]->id() == chip_id) {
            return device.get();
        }
    }
    TT_FATAL(false, "Unexpected device id {}", chip_id);
    return nullptr;
}

// NOLINTBEGIN(cppcoreguidelines-prefer-member-initializer)
class ConnectedDevicesHelper {
private:
    std::map<ChipId, tt_metal::IDevice*> devices_map;

public:
    ConnectedDevicesHelper(const TestParams& params) {
        this->arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        this->num_devices = tt::tt_metal::GetNumAvailableDevices();
        std::vector<ChipId> ids(this->num_devices, 0);
        std::iota(ids.begin(), ids.end(), 0);

        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();

        // Use MeshDevice::create_unit_meshes instead of DevicePool
        auto device_map = tt_metal::distributed::MeshDevice::create_unit_meshes(
            ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, dispatch_core_config, {}, DEFAULT_WORKER_L1_SIZE);

        // Convert map to vector, maintaining order by chip ID
        this->devices.reserve(device_map.size());
        for (auto& [id, device] : device_map) {
            this->devices.push_back(device);
        }

        this->initialize_sender_receiver_pairs(params);
        device_open_ = true;
    }

    ~ConnectedDevicesHelper() {
        if (device_open_) {
            TearDown();
        }
    }

    void TearDown() {
        device_open_ = false;
        tt::tt_metal::MetalContext::instance().get_cluster().set_internal_routing_info_for_ethernet_cores(false);
        for (auto& device : this->devices) {
            device->close();
        }
        devices.clear();
    }

    std::vector<std::shared_ptr<tt_metal::distributed::MeshDevice>> devices;
    tt::ARCH arch;
    size_t num_devices;
    std::set<SenderReceiverPair> unique_links;

private:
    // NOLINTBEGIN(readability-make-member-function-const)
    std::pair<std::optional<CoreCoord>, std::shared_ptr<tt_metal::distributed::MeshBuffer>>
    assign_tensix_and_allocate_buffer(const TestParams& params, const tt_cxy_pair& logical_eth_core) {
        if (params.benchmark_type != BenchmarkType::EthEthTensixUniDir and
            params.benchmark_type != BenchmarkType::EthEthTensixBiDir) {
            return {std::nullopt, nullptr};
        }
        static std::unordered_map<ChipId, std::unordered_set<CoreCoord>> assigned_tensix_per_chip;
        auto& assigned_phys_tensix = assigned_tensix_per_chip[logical_eth_core.chip];
        const auto& soc_d = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(logical_eth_core.chip);

        auto physical_eth_core =
            soc_d.get_physical_ethernet_core_from_logical(CoreCoord{logical_eth_core.x, logical_eth_core.y});

        const std::vector<tt::umd::CoreCoord>& tensix_cores = soc_d.get_cores(CoreType::TENSIX, CoordSystem::NOC0);

        std::optional<CoreCoord> closest_phys_tensix = std::nullopt;
        for (auto phys_tensix : tensix_cores) {
            if (assigned_phys_tensix.contains(phys_tensix)) {
                continue;
            }
            // TODO: uplift this for BH col harvesting when that is enabled
            if (phys_tensix.x == physical_eth_core.x and phys_tensix.y > physical_eth_core.y) {
                closest_phys_tensix = phys_tensix;
                break;
            }
        }
        TT_FATAL(
            closest_phys_tensix.has_value(),
            "Could not find a Tensix core close to logical Eth core {}",
            logical_eth_core.str());
        assigned_phys_tensix.insert(closest_phys_tensix.value());

        auto logical_tensix = soc_d.translate_coord_to(
            {closest_phys_tensix.value(), CoreType::TENSIX, CoordSystem::NOC0}, CoordSystem::LOGICAL);

        tt_metal::ShardSpecBuffer shard_spec = tt_metal::ShardSpecBuffer(
            CoreRangeSet(std::set<CoreRange>({CoreRange(logical_tensix)})),
            {1, params.packet_size},
            tt_metal::ShardOrientation::ROW_MAJOR,
            {1, params.packet_size},
            {1, params.packet_size});

        tt_metal::distributed::DeviceLocalBufferConfig device_local_config{
            .page_size = params.packet_size,
            .buffer_type = tt_metal::BufferType::L1,
            .sharding_args = tt_metal::BufferShardingArgs(shard_spec, tt_metal::TensorMemoryLayout::HEIGHT_SHARDED),
            .bottom_up = false};

        tt_metal::distributed::ReplicatedBufferConfig global_buffer_config{
            .size = params.packet_size,
        };
        auto* device = find_device_with_id(this->devices, logical_eth_core.chip);
        auto buffer = tt_metal::distributed::MeshBuffer::create(global_buffer_config, device_local_config, device);

        return std::make_pair(logical_tensix, std::move(buffer));
    }
    // NOLINTEND(readability-make-member-function-const)

    void initialize_sender_receiver_pairs(const TestParams& params) {
        // chip id -> active eth ch on chip -> (connected chip, remote active eth ch)
        auto all_eth_connections = tt::tt_metal::MetalContext::instance().get_cluster().get_ethernet_connections();

        std::set<ChipId> sender_chips, receiver_chips;
        bool slow_dispath_mode = (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr);

        std::queue<ChipId> chip_q;
        chip_q.push(this->devices[0]->get_devices()[0]->id());  // Start with the first device's chip ID
        sender_chips.insert(this->devices[0]->get_devices()[0]->id());
        std::unordered_set<ChipId> visited_chips;

        // Need sender and receiver chips to be disjoint because we profile wrt. sender and don't want devices to be out
        // of sync
        while (visited_chips.size() != this->devices.size()) {
            while (!chip_q.empty()) {
                ChipId chip_id = chip_q.front();
                chip_q.pop();
                visited_chips.insert(chip_id);

                bool is_sender = sender_chips.contains(chip_id);
                bool is_receiver = receiver_chips.contains(chip_id);

                for (ChipId connected_chip :
                     tt::tt_metal::MetalContext::instance().get_cluster().get_ethernet_connected_device_ids(chip_id)) {
                    bool connected_chip_is_sender = sender_chips.contains(connected_chip);
                    bool connected_chip_is_receiver = receiver_chips.contains(connected_chip);
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

            // Handle other unconnected device clusters
            for (auto& device : this->devices) {
                if (!visited_chips.contains(device->get_devices()[0]->id())) {
                    // This device is not connected others visited above, mark it as a sender for its connected cluster
                    chip_q.push(device->get_devices()[0]->id());
                    sender_chips.insert(device->get_devices()[0]->id());
                    break;
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
                tt::tt_metal::MetalContext::instance().get_control_plane().get_active_ethernet_cores(
                    sender_chip_id, !slow_dispath_mode);
            for (auto logical_active_eth : non_tunneling_eth_cores) {
                if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(
                        sender_chip_id, logical_active_eth)) {
                    continue;
                }
                auto sender_eth = tt_cxy_pair(sender_chip_id, logical_active_eth);
                auto receiver_eth_tuple =
                    tt::tt_metal::MetalContext::instance().get_cluster().get_connected_ethernet_core(
                        std::make_tuple(sender_chip_id, logical_active_eth));
                auto receiver_eth = tt_cxy_pair(std::get<0>(receiver_eth_tuple), std::get<1>(receiver_eth_tuple));
                const auto& [sender_tensix, sender_buffer] = assign_tensix_and_allocate_buffer(params, sender_eth);
                const auto& [receiver_tensix, receiver_buffer] =
                    assign_tensix_and_allocate_buffer(params, receiver_eth);
                this->unique_links.insert(SenderReceiverPair{
                    .sender = sender_eth,
                    .receiver = receiver_eth,
                    .sender_tensix = sender_tensix,
                    .receiver_tensix = receiver_tensix,
                    .sender_buffer = sender_buffer,
                    .receiver_buffer = receiver_buffer});
            }
        }
    }

    bool device_open_ = false;
};
// NOLINTEND(cppcoreguidelines-prefer-member-initializer)

std::vector<tt_metal::Program> build(const ConnectedDevicesHelper& device_helper, const TestParams& params) {
    std::vector<tt_metal::Program> programs(device_helper.num_devices);
    // Create a mapping from chip ID to vector index
    std::map<ChipId, size_t> chip_to_index;
    for (size_t i = 0; i < device_helper.devices.size(); i++) {
        chip_to_index[device_helper.devices[i]->get_devices()[0]->id()] = i;
        log_info(tt::LogTest, "Device {} index {}", device_helper.devices[i]->get_devices()[0]->id(), i);
    }

    uint32_t measurement_type = (uint32_t)(params.test_latency ? MeasurementType::Latency : MeasurementType::Bandwidth);
    uint32_t benchmark_type_val = enchantum::to_underlying(params.benchmark_type);

    // eth core ct args
    const std::vector<uint32_t>& eth_ct_args = {
        benchmark_type_val, measurement_type, params.num_buffer_slots, uint32_t(params.disable_trid)};

    // eth core rt args
    std::vector<uint32_t> eth_receiver_rt_args = {
        tt_metal::MetalContext::instance().hal().get_dev_addr(
            tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt_metal::HalL1MemAddrType::UNRESERVED),
        static_cast<uint32_t>(params.num_packets),
        static_cast<uint32_t>(params.packet_size),
        0,  // tensix noc x updated below
        0,  // tensix noc y updated below
        0,  // buffer addr updated below
    };

    std::vector<uint32_t> eth_sender_rt_args = eth_receiver_rt_args;
    // these are overwritten below with sender/receiver link encoding
    eth_sender_rt_args.push_back(0);
    eth_sender_rt_args.push_back(0);

    for (const auto& link : device_helper.unique_links) {
        auto& sender_program = programs.at(chip_to_index[link.sender.chip]);
        auto& receiver_program = programs.at(chip_to_index[link.receiver.chip]);

        auto* sender_device = find_device_with_id(device_helper.devices, link.sender.chip);
        auto* receiver_device = find_device_with_id(device_helper.devices, link.receiver.chip);

        if (link.sender_tensix.has_value()) {
            TT_FATAL(
                link.receiver_tensix.has_value() and link.sender_buffer != nullptr and link.receiver_buffer != nullptr,
                "Did not assign tensix core to receiver or allocate buffers");

            auto virtual_sender_tensix =
                sender_device->virtual_core_from_logical_core(link.sender_tensix.value(), CoreType::WORKER);
            auto virtual_receiver_tensix =
                receiver_device->virtual_core_from_logical_core(link.receiver_tensix.value(), CoreType::WORKER);

            eth_sender_rt_args[eth_sender_rt_args.size() - 5] = virtual_sender_tensix.x;
            eth_sender_rt_args[eth_sender_rt_args.size() - 4] = virtual_sender_tensix.y;
            eth_sender_rt_args[eth_sender_rt_args.size() - 3] = link.sender_buffer->address();

            eth_receiver_rt_args[eth_receiver_rt_args.size() - 3] = virtual_receiver_tensix.x;
            eth_receiver_rt_args[eth_receiver_rt_args.size() - 2] = virtual_receiver_tensix.y;
            eth_receiver_rt_args[eth_receiver_rt_args.size() - 1] = link.receiver_buffer->address();
        }

        // Pass sender/receiver tt_cxy_pair to sender kernel so we can add link info
        uint32_t sender_encoding = ((uint8_t)link.sender.chip << 0x8) | ((uint8_t)link.sender.y);
        uint32_t receiver_encoding = ((uint8_t)link.receiver.chip << 0x8) | ((uint8_t)link.receiver.y);
        eth_sender_rt_args[eth_sender_rt_args.size() - 2] = sender_encoding;
        eth_sender_rt_args[eth_sender_rt_args.size() - 1] = receiver_encoding;

        auto sender_kernel = tt_metal::CreateKernel(
            sender_program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/"
            "ethernet_write_worker_latency_ubench_sender.cpp",
            CoreCoord(link.sender.x, link.sender.y),
            tt_metal::EthernetConfig{.noc = tt_metal::NOC::RISCV_0_default, .compile_args = eth_ct_args});
        tt_metal::SetRuntimeArgs(
            sender_program, sender_kernel, CoreCoord(link.sender.x, link.sender.y), eth_sender_rt_args);

        auto receiver_kernel = tt_metal::CreateKernel(
            receiver_program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/"
            "ethernet_write_worker_latency_ubench_receiver.cpp",
            CoreCoord(link.receiver.x, link.receiver.y),
            tt_metal::EthernetConfig{.noc = tt_metal::NOC::RISCV_0_default, .compile_args = eth_ct_args});
        tt_metal::SetRuntimeArgs(
            receiver_program, receiver_kernel, CoreCoord(link.receiver.x, link.receiver.y), eth_receiver_rt_args);
    }

    // Compile all programs
    for (size_t i = 0; i < device_helper.devices.size(); i++) {
        const auto& device = device_helper.devices[i];
        try {
            tt_metal::detail::CompileProgram(device.get(), programs[i]);
        } catch (std::exception& e) {
            log_error(tt::LogTest, "Failed to compile program on device {}: {}", i, e.what());
            throw e;
        }
    }
    return programs;
}

void validation(
    const ConnectedDevicesHelper& device_helper,
    const SenderReceiverPair& link,
    uint32_t bytes_to_read,
    bool validate_receiver,
    bool read_buffer = false) {
    static const uint32_t eth_read_addr =
        tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
            tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt_metal::HalL1MemAddrType::UNRESERVED) +
        sizeof(eth_buffer_slot_sync_t);
    std::vector<uint8_t> golden_vec(bytes_to_read);
    std::iota(std::begin(golden_vec), std::end(golden_vec), 0);
    std::vector<uint8_t> result_vec(bytes_to_read, 0);

    auto* sender_device = find_device_with_id(device_helper.devices, link.sender.chip);
    auto* receiver_device = find_device_with_id(device_helper.devices, link.receiver.chip);
    TT_FATAL(
        sender_device->get_devices()[0]->id() == link.sender.chip and
            receiver_device->get_devices()[0]->id() == link.receiver.chip,
        "Mismatch between chips");

    auto sender_virtual =
        sender_device->virtual_core_from_logical_core(CoreCoord(link.sender.x, link.sender.y), CoreType::ETH);
    auto receiver_virtual =
        receiver_device->virtual_core_from_logical_core(CoreCoord(link.receiver.x, link.receiver.y), CoreType::ETH);

    if (read_buffer) {
        auto buffer_to_validate = validate_receiver ? link.receiver_buffer : link.sender_buffer;
        // Use distributed::ReadShard to read from the mesh buffer
        // We need to get the device and coordinate for reading
        auto* device =
            find_device_with_id(device_helper.devices, validate_receiver ? link.receiver.chip : link.sender.chip);
        tt_metal::distributed::ReadShard(
            device->mesh_command_queue(),
            result_vec,
            buffer_to_validate,
            tt_metal::distributed::MeshCoordinate(0, 0),
            true  // blocking read
        );
    } else {
        auto core_to_read = validate_receiver ? tt_cxy_pair(link.receiver.chip, receiver_virtual)
                                              : tt_cxy_pair(link.sender.chip, sender_virtual);
        tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            result_vec.data(), bytes_to_read, core_to_read, eth_read_addr);
    }

    bool pass = golden_vec == result_vec;
    TT_FATAL(pass, "validation failed");
}

void dump_eth_link_stats(
    const ConnectedDevicesHelper& device_helper,
    uint32_t iteration,
    std::map<tt_cxy_pair, std::vector<LinkStats>>& sender_stats,
    std::map<tt_cxy_pair, std::vector<LinkStats>>& receiver_stats,
    uint32_t total_iterations,
    bool post_run = false,
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
            "Iteration,Sender Device ID,Sender Eth,S Retrain Count,S CRC Errs,S PCS Faults,S Total Corr,S Total "
            "Uncorr,S Retrain by PCS,S Retrain by CRC");
        log_file << fmt::format(
                        ",Receiver Device ID,Receiver Eth,R Retrain Count,R CRC Errs,R PCS Faults,R Total "
                        "Corr,R Total Uncorr,R Retrain by PCS,R Retrain by CRC")
                 << std::endl;
    } else {
        log_file.open(log_path, std::ios_base::app);
    }

    // from base FW (WH) we want to read up to test_results[55]
    std::vector<uint32_t> link_stats(56, 0);
    constexpr uint32_t link_stats_base_addr = 0x1EC0;
    for (const auto& link : device_helper.unique_links) {
        auto* sender_device = find_device_with_id(device_helper.devices, link.sender.chip);
        auto* receiver_device = find_device_with_id(device_helper.devices, link.receiver.chip);
        auto sender_virtual =
            sender_device->virtual_core_from_logical_core(CoreCoord(link.sender.x, link.sender.y), CoreType::ETH);
        auto receiver_virtual =
            receiver_device->virtual_core_from_logical_core(CoreCoord(link.receiver.x, link.receiver.y), CoreType::ETH);

        tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            link_stats.data(),
            link_stats.size() * sizeof(uint32_t),
            tt_cxy_pair(link.sender.chip, sender_virtual),
            link_stats_base_addr);

        auto& s_stats_per_iter = sender_stats[link.sender];
        auto& r_stats_per_iter = receiver_stats[link.receiver];
        if (s_stats_per_iter.empty()) {
            TT_FATAL(r_stats_per_iter.empty(), "Receiver stats should be empty when sender stats are empty");
            s_stats_per_iter.resize(total_iterations);
            r_stats_per_iter.resize(total_iterations);
        }

        s_stats_per_iter[iteration] = LinkStats{
            .retrain_count = post_run ? link_stats[7] - s_stats_per_iter[iteration].retrain_count : link_stats[7],
            .crc_errs = post_run ? link_stats[47] - s_stats_per_iter[iteration].crc_errs : link_stats[47],
            .pcs_faults = link_stats[49],
            .total_corr_cw = post_run ? ((uint64_t(link_stats[52]) << 32) | link_stats[53]) -
                                            s_stats_per_iter[iteration].total_corr_cw
                                      : (uint64_t(link_stats[52]) << 32) | link_stats[53],
            .total_uncorr_cw = post_run ? ((uint64_t(link_stats[54]) << 32) | link_stats[55]) -
                                              s_stats_per_iter[iteration].total_uncorr_cw
                                        : (uint64_t(link_stats[54]) << 32) | link_stats[55],
            .retrains_triggered_by_pcs =
                post_run ? link_stats[50] - s_stats_per_iter[iteration].retrains_triggered_by_pcs : link_stats[50],
            .retrains_triggered_by_crcs =
                post_run ? link_stats[51] - s_stats_per_iter[iteration].retrains_triggered_by_crcs : link_stats[51]};

        tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            link_stats.data(),
            link_stats.size() * sizeof(uint32_t),
            tt_cxy_pair(link.receiver.chip, receiver_virtual),
            link_stats_base_addr);

        r_stats_per_iter[iteration] = LinkStats{
            .retrain_count = post_run ? link_stats[7] - r_stats_per_iter[iteration].retrain_count : link_stats[7],
            .crc_errs = post_run ? link_stats[47] - r_stats_per_iter[iteration].crc_errs : link_stats[47],
            .pcs_faults = link_stats[49],
            .total_corr_cw = post_run ? ((uint64_t(link_stats[52]) << 32) | link_stats[53]) -
                                            r_stats_per_iter[iteration].total_corr_cw
                                      : (uint64_t(link_stats[52]) << 32) | link_stats[53],
            .total_uncorr_cw = post_run ? ((uint64_t(link_stats[54]) << 32) | link_stats[55]) -
                                              r_stats_per_iter[iteration].total_uncorr_cw
                                        : (uint64_t(link_stats[54]) << 32) | link_stats[55],
            .retrains_triggered_by_pcs =
                post_run ? link_stats[50] - r_stats_per_iter[iteration].retrains_triggered_by_pcs : link_stats[50],
            .retrains_triggered_by_crcs =
                post_run ? link_stats[51] - r_stats_per_iter[iteration].retrains_triggered_by_crcs : link_stats[51]};

        if (not write_header) {
            const auto& s = s_stats_per_iter[iteration];
            const auto& r = r_stats_per_iter[iteration];

            log_file << fmt::format(
                "{},{},{},{},{},{},{},{},{},{}",
                iteration,
                link.sender.chip,
                link.sender.y,
                s.retrain_count,
                s.crc_errs,
                s.pcs_faults,
                s.total_corr_cw,
                s.total_uncorr_cw,
                s.retrains_triggered_by_pcs,
                s.retrains_triggered_by_crcs);

            log_file << fmt::format(
                            ",{},{},{},{},{},{},{},{},{}",
                            link.receiver.chip,
                            link.receiver.y,
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

    // Create MeshWorkloads from programs for this iteration
    bool slow_dispath_mode = std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr;
    for (uint32_t iteration = 0; iteration < params.num_iterations; iteration++) {
        dump_eth_link_stats(
            device_helper, iteration, sender_stats, receiver_stats, params.num_iterations, false, iteration == 0);
        if (slow_dispath_mode) {
            std::vector<std::thread> threads;
            for (size_t i = 0; i < device_helper.devices.size(); i++) {
                const auto& device = device_helper.devices[i];
                auto& program = programs[i];
                tt_metal::distributed::MeshWorkload mesh_workload;
                mesh_workload.add_program(
                    tt_metal::distributed::MeshCoordinateRange(
                        tt_metal::distributed::MeshCoordinate(0, 0), tt_metal::distributed::MeshCoordinate(0, 0)),
                    std::move(program));
                threads.emplace_back([&]() {
                    // Use distributed::EnqueueMeshWorkload instead of LaunchProgram
                    tt_metal::distributed::EnqueueMeshWorkload(device->mesh_command_queue(), mesh_workload, false);
                });
            }
            for (auto& thread : threads) {
                thread.join();
            }
        } else {
            for (size_t i = 0; i < device_helper.devices.size(); i++) {
                const auto& device = device_helper.devices[i];
                auto& program = programs[i];
                program.set_runtime_id(0);
                tt_metal::distributed::MeshWorkload mesh_workload;
                mesh_workload.add_program(
                    tt_metal::distributed::MeshCoordinateRange(
                        tt_metal::distributed::MeshCoordinate(0, 0), tt_metal::distributed::MeshCoordinate(0, 0)),
                    std::move(program));
                tt_metal::distributed::EnqueueMeshWorkload(device->mesh_command_queue(), mesh_workload, false);
            }
            log_info(tt::LogTest, "Iteration {} Calling Finish", iteration);
            for (const auto& device : device_helper.devices) {
                tt_metal::distributed::Finish(device->mesh_command_queue());
            }
        }
        dump_eth_link_stats(device_helper, iteration, sender_stats, receiver_stats, params.num_iterations, true);
    }

    for (const auto& link : device_helper.unique_links) {
        // Only read profiler results from sender
        tt_metal::ReadMeshDeviceProfilerResults(*find_device_with_id(device_helper.devices, link.sender.chip));

        switch (params.benchmark_type) {
            case BenchmarkType::EthOnlyUniDir:
            case BenchmarkType::EthOnlyBiDir: {
                validation(device_helper, link, params.packet_size, true);
                if (params.benchmark_type == BenchmarkType::EthOnlyBiDir) {
                    validation(device_helper, link, params.packet_size, false);
                }
            } break;
            case BenchmarkType::EthEthTensixUniDir:
            case BenchmarkType::EthEthTensixBiDir: {
                TT_FATAL(
                    link.receiver_buffer != nullptr,
                    "Expected receiver eth {} to write to allocated buffer",
                    link.receiver.str());
                validation(device_helper, link, params.packet_size, true, true);
                if (params.benchmark_type == BenchmarkType::EthEthTensixBiDir) {
                    TT_FATAL(
                        link.sender_buffer != nullptr,
                        "Expected sender eth {} in bidir mode to write to allocated buffer",
                        link.sender.str());
                    validation(device_helper, link, params.packet_size, false, true);
                }
            } break;
            default: break;
        }
    }
}

int main(int /*argc*/, char** argv) {
    std::size_t arg_idx = 1;
    uint32_t benchmark_type = (uint32_t)std::stoi(argv[arg_idx++]);

    auto benchmark_type_enum = enchantum::cast<BenchmarkType>(benchmark_type);
    TT_FATAL(
        benchmark_type_enum.has_value(),
        "Unsupported benchmark {} specified, check BenchmarkType enum for supported values",
        benchmark_type);

    std::size_t num_packets = std::stoi(argv[arg_idx++]);
    std::size_t packet_size = std::stoi(argv[arg_idx++]);
    std::size_t num_buffer_slots = std::stoi(argv[arg_idx++]);

    bool test_latency = std::stoi(argv[arg_idx++]);
    bool disable_trid = std::stoi(argv[arg_idx++]);
    uint32_t num_iterations = std::stoi(argv[arg_idx++]);

    TestParams params{
        .benchmark_type = benchmark_type_enum.value(),
        .num_packets = num_packets,
        .packet_size = packet_size,
        .num_buffer_slots = num_buffer_slots,
        .test_latency = test_latency,
        .disable_trid = disable_trid,
        .num_iterations = num_iterations};

    log_info(tt::LogTest, "Setting up test fixture");
    ConnectedDevicesHelper device_helper(params);
    log_info(tt::LogTest, "Done setting up test fixture");
    if (device_helper.num_devices < 2) {
        log_info(tt::LogTest, "Need at least 2 devices to run this test");
        return 0;
    }

    // Add more configurations here until proper argc parsing added
    bool success = false;
    success = true;
    log_info(tt::LogTest, "STARTING");
    try {
        log_info(
            tt::LogTest,
            "benchmark type: {}, measurement type: {}, num_packets: {}, packet_size: {} B, num_buffer_slots: {}",
            enchantum::to_string(benchmark_type_enum.value()),
            enchantum::to_string(test_latency ? MeasurementType::Latency : MeasurementType::Bandwidth),
            num_packets,
            packet_size,
            num_buffer_slots);

        auto programs = build(device_helper, params);
        run(device_helper, programs, params);

    } catch (std::exception& e) {
        log_error(tt::LogTest, "Caught exception: {}", e.what());
        device_helper.TearDown();
        return -1;
    }

    return success ? 0 : -1;
}
