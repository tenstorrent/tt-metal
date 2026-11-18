// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The cluster, SoC and mesh graph descriptors of a virtual cluster, generated from its manifest and the package's SoC
// descriptor (SPEC-0011 § Descriptors generated from the manifest, § Link kinds). No device access.

#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tt_metal/tt_fabric/keraunos/vcluster/manifest.hpp"

namespace tt::tt_fabric::vcluster {

// Why generate_descriptors() refused a manifest; the value is the row number of SPEC-0011's G table.
enum class GenerationRefusal : uint32_t {
    NotALine = 1,             // G1: fabric_chip values are not 0..N-1, or the links are not one between i and i+1
    RouterPortsNotDense = 2,  // G2: a device's router ports are not 0..k-1
    TooFewDramViews = 3,      // G3: fewer DRAM views in the package than devices
    RouterNotAnEthCore = 4,   // G4: a router slot's core is not an eth core of the package with a keraunos.cores entry
    HasCapture = 5,           // G5: the manifest has a `capture` key
    DramViewNotProgrammable = 6,  // G6: a DRAM view of the package is not `programmable: true`
};

struct GenerationFinding {
    GenerationRefusal refusal;
    std::string detail;  // names the device, link or core; for people, not parsed
};

// generate_descriptors() refused the manifest. rows() is what check_manifest() reported, as it reported it; findings()
// lists every G refusal that holds, in G order. Either can be empty, not both. what() joins them. No file was written.
class DescriptorGenerationError : public std::runtime_error {
public:
    DescriptorGenerationError(std::vector<ManifestFinding> rows, std::vector<GenerationFinding> findings);
    const std::vector<ManifestFinding>& rows() const { return rows_; }
    const std::vector<GenerationFinding>& findings() const { return findings_; }

private:
    std::vector<ManifestFinding> rows_;
    std::vector<GenerationFinding> findings_;
};

// Absolute paths of the files generate_descriptors() wrote.
struct GeneratedDescriptors {
    std::filesystem::path cluster_desc;
    std::vector<std::filesystem::path> soc_descs;  // index = fabric_chip
    std::filesystem::path mesh_graph_desc;
    // env.emu: TT_METAL_EMU_CLUSTER_DESC, TT_METAL_EMU_SOC_DESC, TT_MESH_GRAPH_DESC_PATH,
    // TT_METAL_EMU_PACKAGE_SOC_DESC (the absolute `package_soc_desc`)
    std::filesystem::path env_emu;
    std::filesystem::path
        env_mock;  // env.mock: the same with TT_METAL_MOCK_CLUSTER_DESC_PATH for the cluster descriptor
};

// Writes, into `out_dir` (created if missing, files replaced if present), chip_<fabric_chip>_soc_desc.yaml for each
// device, cluster_desc.yaml, mesh_graph_descriptor.textproto, env.emu and env.mock, as SPEC-0011 § Descriptors
// generated from the manifest specifies. `package_soc_desc` is the whole package's SoC descriptor (UMD
// tests/soc_descs/mmk.yaml).
//
// Runs check_manifest(`manifest`, package_facts(`package_soc_desc`)) and the G1 to G6 refusals, and throws
// DescriptorGenerationError, before creating or writing anything, when any row or refusal holds; it reports every
// one. Throws std::runtime_error (TT_FATAL) when `package_soc_desc` cannot be read, lacks a key the generator reads
// (dram_views, eth, keraunos.cores, eth_l1_size) or holds a value of the wrong type (naming the file), when
// `manifest` has no devices or a router slot without a port, when the absolute `out_dir` contains a comma or a
// newline, or when a file cannot be written. Before writing, it removes `out_dir`'s env.emu and env.mock, and it
// writes them last: a write failure leaves no env file, and can leave other files of this call in `out_dir`.
GeneratedDescriptors generate_descriptors(
    const Manifest& manifest, const std::filesystem::path& package_soc_desc, const std::filesystem::path& out_dir);

// The manifest's links whose backend is noc, each as its two ends' fabric_chip, lower first, in link order.
// Precondition: `manifest` passed check_manifest(). Throws std::runtime_error (TT_FATAL) when a link end names a
// device the manifest does not have.
std::vector<std::pair<uint32_t, uint32_t>> noc_links(const Manifest& manifest);

// The package facts check_manifest() takes, from the package SoC descriptor at `package_soc_desc`: its
// keraunos.cores as (hsio, cce), MEM_IERISC_LOCAL_HARTS harts a CCE, eth_l1_size bytes of SRAM a CCE, and the
// Ethernet tile pairs (0, 1) and (2, 3) the mmk 26ww45.eth-pcs-output-inter bring-up wires (chippy bringup_mmk.py
// KER_ETH_LINKS). Throws std::runtime_error (TT_FATAL) when the file cannot be read, lacks keraunos.cores or
// eth_l1_size, or holds a value of the wrong type there (naming the file).
PackageFacts package_facts(const std::filesystem::path& package_soc_desc);

}  // namespace tt::tt_fabric::vcluster
