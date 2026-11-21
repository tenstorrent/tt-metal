// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "descriptors.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <set>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tt_stl/assert.hpp>
#include <yaml-cpp/yaml.h>

#include "internal/tt-2xx/quasar/dev_mem_map.h"

namespace tt::tt_fabric::vcluster {

DescriptorGenerationError::DescriptorGenerationError(
    std::vector<ManifestFinding> rows, std::vector<GenerationFinding> findings) :
    std::runtime_error(fmt::format(
        "descriptor generation refused: {}",
        fmt::join(
            [&] {
                std::vector<std::string> lines;
                for (const auto& row : rows) {
                    lines.push_back(fmt::format("row {}: {}", static_cast<uint32_t>(row.check), row.detail));
                }
                for (const auto& f : findings) {
                    lines.push_back(fmt::format("G{}: {}", static_cast<uint32_t>(f.refusal), f.detail));
                }
                return lines;
            }(),
            "; "))),
    rows_(std::move(rows)),
    findings_(std::move(findings)) {}

namespace {

// Chip ids carry UMD's mock Quasar board UPI (0x50) in bits 36 and up.
constexpr uint64_t kQuasarBoardUpi = 0x50;
constexpr uint32_t kBoardUpiShift = 36;
constexpr uint64_t kFirstChipUniqueId = kQuasarBoardUpi << kBoardUpiShift;
constexpr const char* kQuasar = "quasar";

// The HSIO tile pairs the mmk 26ww45.eth-pcs-output-inter bring-up wires with Ethernet (chippy bringup_mmk.py
// KER_ETH_LINKS).
const std::vector<std::pair<uint32_t, uint32_t>> kWiredEthernetTilePairs = {{0, 1}, {2, 3}};

constexpr const char* kClusterDescFile = "cluster_desc.yaml";
constexpr const char* kMeshGraphDescFile = "mesh_graph_descriptor.textproto";
constexpr const char* kEnvEmuFile = "env.emu";
constexpr const char* kEnvMockFile = "env.mock";

YAML::Node load_package(const std::filesystem::path& path) {
    try {
        return YAML::LoadFile(path.string());
    } catch (const YAML::Exception& e) {
        TT_THROW("Cannot read the package SoC descriptor {}: {}", path.string(), e.what());
    }
}

// The node at `key` of the package descriptor, which must be present.
YAML::Node package_key(const YAML::Node& package, const char* key, const std::filesystem::path& path) {
    const YAML::Node value = package[key];
    TT_FATAL(value, "The package SoC descriptor {} has no '{}'", path.string(), key);
    return value;
}

// The package descriptor's keraunos.cores entry of `cce`, or an undefined node.
YAML::Node keraunos_core_entry(const YAML::Node& cores, CceId cce) {
    for (const auto& entry : cores) {
        if (entry["hsio"].as<uint32_t>() == cce.tile && entry["eth_cce"].as<uint32_t>() == cce.cce) {
            return entry;
        }
    }
    return YAML::Node(YAML::NodeType::Undefined);
}

bool is_eth_core(const YAML::Node& eth, const std::string& core) {
    return std::any_of(eth.begin(), eth.end(), [&](const YAML::Node& e) { return e.as<std::string>() == core; });
}

const Device* device_named(const Manifest& manifest, const std::string& name) {
    const auto it =
        std::find_if(manifest.devices.begin(), manifest.devices.end(), [&](const Device& d) { return d.name == name; });
    return it == manifest.devices.end() ? nullptr : &*it;
}

// A device's Ethernet cores, its router and idle slots, in port order. Fatal when one has no port, which
// parse_manifest() refuses.
std::vector<Slot> eth_slots_by_port(const Device& device) {
    std::vector<Slot> slots;
    for (const Slot& slot : device.slots) {
        if (slot.role != SlotRole::Worker) {
            TT_FATAL(slot.port.has_value(), "Device {} has a router or idle slot without a port", device.name);
            slots.push_back(slot);
        }
    }
    std::sort(slots.begin(), slots.end(), [](const Slot& x, const Slot& y) { return *x.port < *y.port; });
    return slots;
}

bool is_programmable(const YAML::Node& view) { return view["programmable"] && view["programmable"].as<bool>(); }

// G1 to G6, every one that holds.
std::vector<GenerationFinding> refusals(
    const Manifest& manifest, const YAML::Node& eth, const YAML::Node& cores, const std::vector<YAML::Node>& views) {
    std::vector<GenerationFinding> findings;
    const auto num_devices = static_cast<uint32_t>(manifest.devices.size());

    std::set<uint32_t> chips;
    for (const Device& device : manifest.devices) {
        chips.insert(device.fabric_chip);
    }
    if (chips.size() != num_devices || *chips.rbegin() != num_devices - 1) {
        findings.push_back(
            {GenerationRefusal::NotALine,
             fmt::format("fabric_chip values {} are not 0 to {}", fmt::join(chips, ", "), num_devices - 1)});
    }
    std::map<uint32_t, std::size_t> hops;  // lower chip of each link -> number of links
    for (std::size_t i = 0; i < manifest.links.size(); i++) {
        const Link& link = manifest.links[i];
        const Device* a = device_named(manifest, link.a.device);
        const Device* b = device_named(manifest, link.b.device);
        if (a == nullptr || b == nullptr) {
            findings.push_back(
                {GenerationRefusal::NotALine, fmt::format("links[{}] names a device the manifest does not have", i)});
            continue;
        }
        const uint32_t lower = std::min(a->fabric_chip, b->fabric_chip);
        const uint32_t upper = std::max(a->fabric_chip, b->fabric_chip);
        if (upper != lower + 1) {
            findings.push_back(
                {GenerationRefusal::NotALine,
                 fmt::format("links[{}] joins chips {} and {}, which are not neighbours on a line", i, lower, upper)});
            continue;
        }
        hops[lower]++;
    }
    for (uint32_t chip = 0; chip + 1 < num_devices; chip++) {
        const std::size_t count = hops.contains(chip) ? hops.at(chip) : 0;
        if (count != 1) {
            findings.push_back(
                {GenerationRefusal::NotALine,
                 fmt::format("{} links between chips {} and {}, not 1", count, chip, chip + 1)});
        }
    }

    for (const Device& device : manifest.devices) {
        std::vector<uint32_t> ports;
        for (const Slot& slot : eth_slots_by_port(device)) {
            ports.push_back(*slot.port);
        }
        for (std::size_t port = 0; port < ports.size(); port++) {
            if (ports[port] != port) {
                findings.push_back(
                    {GenerationRefusal::RouterPortsNotDense,
                     fmt::format(
                         "device {} Ethernet ports {} are not 0 to {}",
                         device.name,
                         fmt::join(ports, ", "),
                         ports.size() - 1)});
                break;
            }
        }
    }

    if (views.size() < num_devices) {
        findings.push_back(
            {GenerationRefusal::TooFewDramViews,
             fmt::format("the package has {} DRAM views for {} devices", views.size(), num_devices)});
    }

    for (const Device& device : manifest.devices) {
        for (const Slot& slot : eth_slots_by_port(device)) {
            const YAML::Node entry = keraunos_core_entry(cores, CceId{slot.tile, slot.cce});
            if (!entry) {
                findings.push_back(
                    {GenerationRefusal::RouterNotAnEthCore,
                     fmt::format(
                         "device {} Ethernet core (tile {}, cce {}) has no keraunos.cores entry in the package",
                         device.name,
                         slot.tile,
                         slot.cce)});
            } else if (!is_eth_core(eth, entry["core"].as<std::string>())) {
                findings.push_back(
                    {GenerationRefusal::RouterNotAnEthCore,
                     fmt::format(
                         "device {} Ethernet core {} is not in the package's eth list",
                         device.name,
                         entry["core"].as<std::string>())});
            }
        }
    }

    if (manifest.capture) {
        findings.push_back(
            {GenerationRefusal::HasCapture,
             fmt::format("the manifest names a capture ({})", manifest.capture->string())});
    }
    for (std::size_t i = 0; i < views.size(); i++) {
        if (!is_programmable(views[i])) {
            findings.push_back(
                {GenerationRefusal::DramViewNotProgrammable,
                 fmt::format("the package's DRAM view {} is not programmable: true", i)});
        }
    }
    return findings;
}

void write_file(const std::filesystem::path& path, const std::string& text) {
    std::ofstream file(path, std::ios::trunc);
    TT_FATAL(file.good(), "Cannot open {} for writing", path.string());
    file << text;
    file.close();
    TT_FATAL(!file.fail(), "Cannot write {}", path.string());
}

std::string emit(const YAML::Node& node) {
    YAML::Emitter out;
    out << node;
    TT_FATAL(out.good(), "YAML emitter: {}", out.GetLastError());
    return std::string(out.c_str()) + "\n";
}

// The package descriptor with eth, keraunos.cores and dram_views replaced for `device` (fabric chip k of N).
std::string soc_desc(
    const YAML::Node& package,
    const YAML::Node& cores,
    const std::vector<YAML::Node>& views,
    const Device& device,
    uint32_t num_devices) {
    YAML::Node desc = YAML::Clone(package);
    YAML::Node eth(YAML::NodeType::Sequence);
    YAML::Node device_cores(YAML::NodeType::Sequence);
    for (const Slot& slot : eth_slots_by_port(device)) {
        const YAML::Node entry = keraunos_core_entry(cores, CceId{slot.tile, slot.cce});
        eth.push_back(entry["core"].as<std::string>());
        device_cores.push_back(YAML::Clone(entry));
    }
    eth.SetStyle(YAML::EmitterStyle::Flow);
    desc["eth"] = eth;
    desc["keraunos"]["cores"] = device_cores;
    // V views in list order, N devices: device k takes V / N consecutive views, one more for k < V % N, device 0
    // from view 0.
    const std::size_t share = views.size() / num_devices;
    const std::size_t extra = views.size() % num_devices;
    const std::size_t first = device.fabric_chip * share + std::min<std::size_t>(device.fabric_chip, extra);
    const std::size_t count = share + (device.fabric_chip < extra ? 1 : 0);
    YAML::Node dram_views(YAML::NodeType::Sequence);
    for (std::size_t i = first; i < first + count; i++) {
        dram_views.push_back(YAML::Clone(views[i]));
    }
    desc["dram_views"] = dram_views;
    return emit(desc);
}

// N Quasar chips, each its own board and MMIO chip, with one Ethernet connection for each of the manifest's links.
std::string cluster_desc(const Manifest& manifest) {
    const auto num_chips = static_cast<uint32_t>(manifest.devices.size());
    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "arch" << YAML::Value << YAML::BeginMap;
    for (uint32_t chip = 0; chip < num_chips; chip++) {
        out << YAML::Key << chip << YAML::Value << kQuasar;
    }
    out << YAML::EndMap;
    out << YAML::Key << "chips" << YAML::Value << YAML::Flow << YAML::BeginMap << YAML::EndMap;
    out << YAML::Key << "chip_unique_ids" << YAML::Value << YAML::BeginMap;
    for (uint32_t chip = 0; chip < num_chips; chip++) {
        out << YAML::Key << chip << YAML::Value << kFirstChipUniqueId + chip;
    }
    out << YAML::EndMap;
    out << YAML::Key << "ethernet_connections" << YAML::Value << YAML::BeginSeq;
    for (const Link& link : manifest.links) {
        // The end with the lower chip first, whatever the manifest's a and b.
        std::pair<uint32_t, uint32_t> ends[] = {
            {device_named(manifest, link.a.device)->fabric_chip, link.a.port},
            {device_named(manifest, link.b.device)->fabric_chip, link.b.port}};
        if (ends[1].first < ends[0].first) {
            std::swap(ends[0], ends[1]);
        }
        out << YAML::Flow << YAML::BeginSeq;
        for (const auto& [chip, chan] : ends) {
            out << YAML::Flow << YAML::BeginMap << YAML::Key << "chip" << YAML::Value << chip << YAML::Key << "chan"
                << YAML::Value << chan << YAML::EndMap;
        }
        out << YAML::EndSeq;
    }
    out << YAML::EndSeq;
    out << YAML::Key << "ethernet_connections_to_remote_devices" << YAML::Value << YAML::Flow << YAML::BeginSeq
        << YAML::EndSeq;
    out << YAML::Key << "chips_with_mmio" << YAML::Value << YAML::BeginSeq;
    for (uint32_t chip = 0; chip < num_chips; chip++) {
        out << YAML::BeginMap << YAML::Key << chip << YAML::Value << chip << YAML::EndMap;
    }
    out << YAML::EndSeq;
    out << YAML::Key << "harvesting" << YAML::Value << YAML::BeginMap;
    for (uint32_t chip = 0; chip < num_chips; chip++) {
        out << YAML::Key << chip << YAML::Value << YAML::BeginMap << YAML::Key << "noc_translation" << YAML::Value
            << false << YAML::Key << "harvest_mask" << YAML::Value << 0 << YAML::EndMap;
    }
    out << YAML::EndMap;
    out << YAML::Key << "chip_to_boardtype" << YAML::Value << YAML::BeginMap;
    for (uint32_t chip = 0; chip < num_chips; chip++) {
        out << YAML::Key << chip << YAML::Value << kQuasar;
    }
    out << YAML::EndMap;
    out << YAML::Key << "boards" << YAML::Value << YAML::BeginSeq;
    for (uint32_t chip = 0; chip < num_chips; chip++) {
        out << YAML::BeginSeq;
        out << YAML::BeginMap << YAML::Key << "board_id" << YAML::Value << kFirstChipUniqueId + chip << YAML::EndMap;
        out << YAML::BeginMap << YAML::Key << "board_type" << YAML::Value << kQuasar << YAML::EndMap;
        out << YAML::BeginMap << YAML::Key << "chips" << YAML::Value << YAML::BeginSeq << chip << YAML::EndSeq
            << YAML::EndMap;
        out << YAML::EndSeq;
    }
    out << YAML::EndSeq;
    out << YAML::Key << "asic_locations" << YAML::Value << YAML::BeginMap;
    for (uint32_t chip = 0; chip < num_chips; chip++) {
        out << YAML::Key << chip << YAML::Value << 0;
    }
    out << YAML::EndMap;
    out << YAML::EndMap;
    TT_FATAL(out.good(), "YAML emitter: {}", out.GetLastError());
    return std::string(out.c_str()) + "\n";
}

// One mesh, a line of N chips, with one channel a connection.
std::string mesh_graph_desc(uint32_t num_chips) {
    return fmt::format(
        "mesh_descriptors {{\n"
        "  name: \"M0\"\n"
        "  arch: QUASAR\n"
        "  device_topology {{ dims: [ 1, {} ] }}\n"
        "  host_topology   {{ dims: [ 1, 1 ] }}\n"
        "  channels {{ count: 1 policy: RELAXED }}\n"
        "}}\n"
        "\n"
        "top_level_instance {{ mesh {{ mesh_descriptor: \"M0\" mesh_id: 0 }} }}\n",
        num_chips);
}

PackageFacts facts_of(const YAML::Node& package, const std::filesystem::path& package_soc_desc) {
    PackageFacts facts;
    for (const auto& entry :
         package_key(package_key(package, "keraunos", package_soc_desc), "cores", package_soc_desc)) {
        facts.cores.push_back(CceId{entry["hsio"].as<uint32_t>(), entry["eth_cce"].as<uint32_t>()});
    }
    facts.harts_per_cce = MEM_IERISC_LOCAL_HARTS;
    facts.sram_bytes_per_cce = package_key(package, "eth_l1_size", package_soc_desc).as<uint64_t>();
    facts.ethernet_tile_pairs = kWiredEthernetTilePairs;
    return facts;
}

GeneratedDescriptors generate(
    const Manifest& manifest, const std::filesystem::path& package_soc_desc, const std::filesystem::path& out_dir) {
    TT_FATAL(!manifest.devices.empty(), "Manifest::devices is empty; parse_manifest() never returns one");
    const YAML::Node package = load_package(package_soc_desc);
    const YAML::Node eth = package_key(package, "eth", package_soc_desc);
    const YAML::Node cores = package_key(package_key(package, "keraunos", package_soc_desc), "cores", package_soc_desc);
    const YAML::Node dram_views = package_key(package, "dram_views", package_soc_desc);
    const std::vector<YAML::Node> views(dram_views.begin(), dram_views.end());

    std::vector<ManifestFinding> rows = check_manifest(manifest, facts_of(package, package_soc_desc));
    std::vector<GenerationFinding> findings = refusals(manifest, eth, cores, views);
    if (!rows.empty() || !findings.empty()) {
        std::stable_sort(
            findings.begin(), findings.end(), [](const auto& x, const auto& y) { return x.refusal < y.refusal; });
        throw DescriptorGenerationError(std::move(rows), std::move(findings));
    }

    // TT_METAL_EMU_SOC_DESC lists the SoC descriptors comma-separated, and each env file holds one variable a line.
    const std::filesystem::path dir = std::filesystem::absolute(out_dir);
    if (dir.string().find_first_of(",\n") != std::string::npos) {
        TT_THROW(
            "Output directory {} contains a comma or a newline, which would split TT_METAL_EMU_SOC_DESC", dir.string());
    }
    std::filesystem::create_directories(dir);
    // A rerun that fails partway leaves no env file to point at a mix of new and old descriptors.
    std::filesystem::remove(dir / kEnvEmuFile);
    std::filesystem::remove(dir / kEnvMockFile);
    const auto num_devices = static_cast<uint32_t>(manifest.devices.size());
    GeneratedDescriptors generated;
    generated.soc_descs.resize(num_devices);
    for (const Device& device : manifest.devices) {
        const std::filesystem::path path = dir / fmt::format("chip_{}_soc_desc.yaml", device.fabric_chip);
        write_file(path, soc_desc(package, cores, views, device, num_devices));
        generated.soc_descs[device.fabric_chip] = path;
    }
    generated.cluster_desc = dir / kClusterDescFile;
    write_file(generated.cluster_desc, cluster_desc(manifest));
    generated.mesh_graph_desc = dir / kMeshGraphDescFile;
    write_file(generated.mesh_graph_desc, mesh_graph_desc(num_devices));
    std::vector<std::string> soc_paths;
    for (const auto& path : generated.soc_descs) {
        soc_paths.push_back(path.string());
    }
    const auto env = [&](const char* cluster_desc_variable) {
        return fmt::format(
            "{}={}\nTT_METAL_EMU_SOC_DESC={}\nTT_MESH_GRAPH_DESC_PATH={}\nTT_METAL_EMU_PACKAGE_SOC_DESC={}\n",
            cluster_desc_variable,
            generated.cluster_desc.string(),
            fmt::join(soc_paths, ","),
            generated.mesh_graph_desc.string(),
            std::filesystem::absolute(package_soc_desc).string());
    };
    // Metal refuses TT_METAL_EMU_CLUSTER_DESC without an emulator server; Mock takes TT_METAL_MOCK_CLUSTER_DESC_PATH.
    generated.env_emu = dir / kEnvEmuFile;
    write_file(generated.env_emu, env("TT_METAL_EMU_CLUSTER_DESC"));
    generated.env_mock = dir / kEnvMockFile;
    write_file(generated.env_mock, env("TT_METAL_MOCK_CLUSTER_DESC_PATH"));
    return generated;
}

}  // namespace

// A package value that is missing or of the wrong type surfaces as a YAML::Exception from a read of an entry; it is
// rethrown naming the package file, with yaml-cpp's line and column of the value.
GeneratedDescriptors generate_descriptors(
    const Manifest& manifest, const std::filesystem::path& package_soc_desc, const std::filesystem::path& out_dir) {
    try {
        return generate(manifest, package_soc_desc, out_dir);
    } catch (const YAML::Exception& e) {
        TT_THROW("The package SoC descriptor {} has an unexpected value: {}", package_soc_desc.string(), e.what());
    }
}

std::vector<std::pair<uint32_t, uint32_t>> noc_links(const Manifest& manifest) {
    std::vector<std::pair<uint32_t, uint32_t>> links;
    for (const Link& link : manifest.links) {
        if (link.backend != LinkBackend::Noc) {
            continue;
        }
        const Device* a = device_named(manifest, link.a.device);
        const Device* b = device_named(manifest, link.b.device);
        TT_FATAL(
            a != nullptr && b != nullptr,
            "Link {}-{} names a device the manifest does not have; check_manifest() refuses it as row 7",
            link.a.device,
            link.b.device);
        links.emplace_back(std::min(a->fabric_chip, b->fabric_chip), std::max(a->fabric_chip, b->fabric_chip));
    }
    return links;
}

PackageFacts package_facts(const std::filesystem::path& package_soc_desc) {
    try {
        return facts_of(load_package(package_soc_desc), package_soc_desc);
    } catch (const YAML::Exception& e) {
        TT_THROW("The package SoC descriptor {} has an unexpected value: {}", package_soc_desc.string(), e.what());
    }
}

}  // namespace tt::tt_fabric::vcluster
