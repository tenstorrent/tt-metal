// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/fabric_telemetry.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "tt_metal/fabric/fabric_telemetry_converter.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include "tt_metal/llrt/hal.hpp"
#include "tt_metal/llrt/hal/generated/fabric_telemetry.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace {

struct Options {
    std::string output;
    std::string output_dir;
};

void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " [--output PATH | --output-dir DIR]\n"
              << "\n"
              << "Dump tt-metal fabric telemetry counters as CSV. With --output-dir, the\n"
              << "file is written under rank<N>/fabric_telemetry.csv using the MPI rank\n"
              << "environment when present.\n";
}

std::optional<int> parse_rank_env(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return std::nullopt;
    }
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

int mpi_rank() {
    for (const char* name : {"OMPI_COMM_WORLD_RANK", "PMI_RANK", "MV2_COMM_WORLD_RANK"}) {
        auto rank = parse_rank_env(name);
        if (rank.has_value()) {
            return *rank;
        }
    }
    return 0;
}

std::string hostname() {
    std::array<char, 256> buf{};
    if (gethostname(buf.data(), buf.size() - 1) != 0) {
        return "";
    }
    return std::string(buf.data());
}

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) {
        return value;
    }
    std::string out = "\"";
    for (char ch : value) {
        if (ch == '"') {
            out += "\"\"";
        } else {
            out += ch;
        }
    }
    out += '"';
    return out;
}

std::uint8_t router_state_value(tt::tt_fabric::FabricTelemetryRouterState state) {
    return static_cast<std::uint8_t>(state);
}

std::filesystem::path output_path(const Options& options) {
    if (!options.output.empty()) {
        return std::filesystem::path(options.output);
    }
    if (!options.output_dir.empty()) {
        return std::filesystem::path(options.output_dir) / ("rank" + std::to_string(mpi_rank())) /
               "fabric_telemetry.csv";
    }
    return {};
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output") {
            if (++i >= argc) {
                throw std::runtime_error("--output needs a path");
            }
            options.output = argv[i];
        } else if (arg == "--output-dir") {
            if (++i >= argc) {
                throw std::runtime_error("--output-dir needs a directory");
            }
            options.output_dir = argv[i];
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (!options.output.empty() && !options.output_dir.empty()) {
        throw std::runtime_error("use only one of --output or --output-dir");
    }
    return options;
}

void write_header(std::ostream& out) {
    out << "host,rank,device_id,eth_channel,mesh_id,neighbor_mesh_id,neighbor_device_id,direction,"
        << "fabric_config,supported_stats,has_dynamic,packets_sent,words_sent,bytes_sent,elapsed_cycles,"
        << "elapsed_active_cycles,tx_packets,tx_words,tx_bytes,tx_elapsed_cycles,tx_elapsed_active_cycles,"
        << "rx_packets,rx_words,rx_bytes,rx_elapsed_cycles,rx_elapsed_active_cycles,tx_heartbeat,rx_heartbeat,"
        << "router_state,router_state_e0,router_state_e1,tx_heartbeat_e0,tx_heartbeat_e1,rx_heartbeat_e0,"
        << "rx_heartbeat_e1,error\n";
    out.flush();
}

void write_snapshot(
    std::ostream& out,
    const std::string& host,
    int rank,
    tt::ChipId chip_id,
    std::uint32_t channel,
    const tt::tt_fabric::FabricTelemetrySnapshot& snapshot,
    const std::string& error) {
    const auto& si = snapshot.static_info;
    std::uint64_t packets = 0;
    std::uint64_t words = 0;
    std::uint64_t elapsed_cycles = 0;
    std::uint64_t elapsed_active_cycles = 0;
    std::uint64_t tx_packets = 0;
    std::uint64_t tx_words = 0;
    std::uint64_t tx_elapsed_cycles = 0;
    std::uint64_t tx_elapsed_active_cycles = 0;
    std::uint64_t rx_packets = 0;
    std::uint64_t rx_words = 0;
    std::uint64_t rx_elapsed_cycles = 0;
    std::uint64_t rx_elapsed_active_cycles = 0;
    std::uint64_t tx_heartbeat = 0;
    std::uint64_t rx_heartbeat = 0;
    std::uint64_t tx_heartbeat_e0 = 0;
    std::uint64_t tx_heartbeat_e1 = 0;
    std::uint64_t rx_heartbeat_e0 = 0;
    std::uint64_t rx_heartbeat_e1 = 0;
    std::uint8_t router_state = 0;
    std::uint8_t router_state_e0 = 0;
    std::uint8_t router_state_e1 = 0;

    if (snapshot.dynamic_info.has_value()) {
        const auto& dynamic = *snapshot.dynamic_info;
        tx_packets = dynamic.tx_bandwidth.packets_sent;
        tx_words = dynamic.tx_bandwidth.words_sent;
        tx_elapsed_cycles = dynamic.tx_bandwidth.elapsed_cycles;
        tx_elapsed_active_cycles = dynamic.tx_bandwidth.elapsed_active_cycles;
        rx_packets = dynamic.rx_bandwidth.packets_sent;
        rx_words = dynamic.rx_bandwidth.words_sent;
        rx_elapsed_cycles = dynamic.rx_bandwidth.elapsed_cycles;
        rx_elapsed_active_cycles = dynamic.rx_bandwidth.elapsed_active_cycles;
        packets = tx_packets + rx_packets;
        words = tx_words + rx_words;
        elapsed_cycles = std::max(tx_elapsed_cycles, rx_elapsed_cycles);
        elapsed_active_cycles = std::max(tx_elapsed_active_cycles, rx_elapsed_active_cycles);

        tx_heartbeat_e0 = dynamic.erisc[0].tx_heartbeat;
        tx_heartbeat_e1 = dynamic.erisc[1].tx_heartbeat;
        rx_heartbeat_e0 = dynamic.erisc[0].rx_heartbeat;
        rx_heartbeat_e1 = dynamic.erisc[1].rx_heartbeat;
        tx_heartbeat = tx_heartbeat_e0 + tx_heartbeat_e1;
        rx_heartbeat = rx_heartbeat_e0 + rx_heartbeat_e1;
        router_state_e0 = router_state_value(dynamic.erisc[0].router_state);
        router_state_e1 = router_state_value(dynamic.erisc[1].router_state);
        router_state = std::max(router_state_e0, router_state_e1);
    }

    out << csv_escape(host) << ',' << rank << ',' << chip_id << ',' << channel << ',' << si.mesh_id << ','
        << si.neighbor_mesh_id << ',' << static_cast<std::uint32_t>(si.neighbor_device_id) << ','
        << static_cast<std::uint32_t>(si.direction) << ',' << si.fabric_config << ','
        << static_cast<std::uint32_t>(si.supported_stats) << ',' << (snapshot.dynamic_info.has_value() ? 1 : 0) << ','
        << packets << ',' << words << ',' << (words * 4) << ',' << elapsed_cycles << ',' << elapsed_active_cycles << ','
        << tx_packets << ',' << tx_words << ',' << (tx_words * 4) << ',' << tx_elapsed_cycles << ','
        << tx_elapsed_active_cycles << ',' << rx_packets << ',' << rx_words << ',' << (rx_words * 4) << ','
        << rx_elapsed_cycles << ',' << rx_elapsed_active_cycles << ',' << tx_heartbeat << ',' << rx_heartbeat << ','
        << static_cast<std::uint32_t>(router_state) << ',' << static_cast<std::uint32_t>(router_state_e0) << ','
        << static_cast<std::uint32_t>(router_state_e1) << ',' << tx_heartbeat_e0 << ',' << tx_heartbeat_e1 << ','
        << rx_heartbeat_e0 << ',' << rx_heartbeat_e1 << ',' << csv_escape(error) << '\n';
}

tt::tt_fabric::FabricTelemetrySnapshot read_snapshot(
    const tt::Cluster& cluster, const tt::tt_metal::Hal& hal, tt::ChipId chip_id, tt::tt_fabric::chan_id_t channel) {
    const auto& factory = hal.get_fabric_telemetry_factory(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
    const auto telemetry_size = factory.size_of<tt::tt_fabric::fabric_telemetry::FabricTelemetry>();
    const auto telemetry_addr = hal.get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::FABRIC_TELEMETRY);

    const auto& soc_desc = cluster.get_soc_desc(chip_id);
    const auto eth_core = soc_desc.get_eth_core_for_channel(channel, tt::CoordSystem::LOGICAL);
    tt_cxy_pair virtual_eth_core(
        chip_id, cluster.get_virtual_coordinate_from_logical_coordinates(chip_id, eth_core, tt::CoreType::ETH));

    std::vector<std::byte> buffer(telemetry_size);
    cluster.read_core(buffer.data(), telemetry_size, virtual_eth_core, telemetry_addr);
    const auto view = factory.create_view<tt::tt_fabric::fabric_telemetry::FabricTelemetry>(buffer.data());
    return tt::tt_fabric::fabric_telemetry_converter::unpack_snapshot_from_hal(view);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const auto path = output_path(options);
        std::ofstream file;
        std::ostream* out = &std::cout;
        if (!path.empty()) {
            std::filesystem::create_directories(path.parent_path());
            file.open(path);
            if (!file.is_open()) {
                throw std::runtime_error("failed to open output: " + path.string());
            }
            out = &file;
        }

        write_header(*out);
        const std::string host = hostname();
        const int rank = mpi_rank();
        auto& metal_ctx = tt::tt_metal::MetalContext::instance();
        const auto& hal = metal_ctx.hal();
        auto& cluster = metal_ctx.get_cluster();

        const auto user_chip_ids = cluster.user_exposed_chip_ids();
        std::vector<tt::ChipId> chip_ids(user_chip_ids.begin(), user_chip_ids.end());
        std::sort(chip_ids.begin(), chip_ids.end());
        for (tt::ChipId chip_id : chip_ids) {
            const auto& soc_desc = cluster.get_soc_desc(chip_id);
            for (std::uint32_t channel = 0; channel < soc_desc.get_num_eth_channels(); ++channel) {
                const auto eth_core = soc_desc.get_eth_core_for_channel(channel, tt::CoordSystem::LOGICAL);
                if (!cluster.is_ethernet_link_up(chip_id, eth_core)) {
                    continue;
                }
                try {
                    const auto snapshot =
                        read_snapshot(cluster, hal, chip_id, static_cast<tt::tt_fabric::chan_id_t>(channel));
                    write_snapshot(*out, host, rank, chip_id, channel, snapshot, "");
                } catch (const std::exception& e) {
                    tt::tt_fabric::FabricTelemetrySnapshot empty{};
                    write_snapshot(*out, host, rank, chip_id, channel, empty, e.what());
                }
                out->flush();
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        usage(argv[0]);
        return 1;
    }
}
