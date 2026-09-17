// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Generate an MPI hostfile from a Factory System Descriptor (FSD), ordering hosts by their
// instance_path hierarchy. Line order == MPI rank order, so ranks are assigned deterministically
// from the FSD (rank i == the i-th host in instance_path order) instead of from an external
// --hosts list. Ordering hosts by instance_path also makes hierarchy tiers contiguous rank
// ranges, which lines up with tier-by-tier (phased) bring-up.

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cxxopts.hpp>
#include <google/protobuf/text_format.h>

#include "protobuf/factory_system_descriptor.pb.h"

namespace {

using tt::scaleout_tools::fsd::proto::FactorySystemDescriptor;

FactorySystemDescriptor load_fsd(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open factory system descriptor: " + path);
    }
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    FactorySystemDescriptor fsd;
    if (!google::protobuf::TextFormat::ParseFromString(content, &fsd)) {
        throw std::runtime_error("Failed to parse factory system descriptor: " + path);
    }
    return fsd;
}

// Host indices (host_id) ordered by instance_path (lexicographic over segments), so hosts under the
// same hierarchy subtree are contiguous. host_id ties (identical paths) fall back to host_id order.
std::vector<uint32_t> hosts_in_hierarchy_order(const FactorySystemDescriptor& fsd) {
    std::vector<uint32_t> ids(fsd.hosts_size());
    std::iota(ids.begin(), ids.end(), 0u);
    std::stable_sort(ids.begin(), ids.end(), [&](uint32_t a, uint32_t b) {
        const auto& pa = fsd.hosts()[a].instance_path();
        const auto& pb = fsd.hosts()[b].instance_path();
        return std::lexicographical_compare(pa.begin(), pa.end(), pb.begin(), pb.end());
    });
    return ids;
}

}  // namespace

int main(int argc, char* argv[]) {
    cxxopts::Options options(
        "generate_hostfile",
        "Generate an MPI hostfile from a Factory System Descriptor, ordering hosts (and therefore MPI ranks) by "
        "their instance_path hierarchy.");
    options.add_options()(
        "factory-descriptor-path", "Path to the factory system descriptor textproto", cxxopts::value<std::string>())(
        "output", "Output hostfile path", cxxopts::value<std::string>()->default_value("hostfile"))(
        "slots", "Slots per host", cxxopts::value<uint32_t>()->default_value("1"))("h,help", "Print usage");

    std::string fsd_path;
    std::string output_path;
    uint32_t slots = 1;
    try {
        auto result = options.parse(argc, argv);
        if (result.contains("help")) {
            std::cout << options.help() << std::endl;
            return 0;
        }
        if (!result.contains("factory-descriptor-path")) {
            std::cerr << "Error: --factory-descriptor-path is required" << std::endl;
            std::cerr << options.help() << std::endl;
            return 1;
        }
        fsd_path = result["factory-descriptor-path"].as<std::string>();
        output_path = result["output"].as<std::string>();
        slots = result["slots"].as<uint32_t>();
    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        return 1;
    }

    const auto fsd = load_fsd(fsd_path);
    if (fsd.hosts_size() == 0) {
        throw std::runtime_error("Factory system descriptor has no hosts: " + fsd_path);
    }
    const auto ordered = hosts_in_hierarchy_order(fsd);

    std::ofstream out(output_path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output hostfile: " + output_path);
    }
    for (uint32_t host_id : ordered) {
        out << fsd.hosts()[host_id].hostname() << " slots=" << slots << "\n";
    }
    out.close();
    if (!out) {
        throw std::runtime_error("Failed to write hostfile: " + output_path);
    }

    // Echo the rank -> host assignment for visibility.
    std::cout << "Wrote " << ordered.size() << " hosts to " << output_path
              << " (rank order = instance_path order):" << std::endl;
    for (uint32_t rank = 0; rank < ordered.size(); ++rank) {
        const auto& host = fsd.hosts()[ordered[rank]];
        std::string path;
        for (const auto& seg : host.instance_path()) {
            path += "/" + seg;
        }
        std::cout << "  rank " << rank << " -> " << host.hostname() << " [" << path << "]" << std::endl;
    }
    return 0;
}
