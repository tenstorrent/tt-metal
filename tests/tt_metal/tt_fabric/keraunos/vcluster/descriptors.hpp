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
    TooFewDramViews = 3,      // G3: fewer programmable DRAM views in the package than devices
    RouterNotAnEthCore = 4,   // G4: a router slot's core is not an eth core of the package with a keraunos.cores entry
    HasCapture = 5,           // G5: the manifest has a `capture` key
};

struct GenerationFinding {
    GenerationRefusal refusal;
    std::string detail;  // names the device, link or core; for people, not parsed
};

// generate_descriptors() refused the manifest. findings() lists every refusal that holds, in G order; what() joins
// them. No file was written.
class DescriptorGenerationError : public std::runtime_error {
public:
    explicit DescriptorGenerationError(std::vector<GenerationFinding> findings);
    const std::vector<GenerationFinding>& findings() const { return findings_; }

private:
    std::vector<GenerationFinding> findings_;
};

// Absolute paths of the files generate_descriptors() wrote.
struct GeneratedDescriptors {
    std::filesystem::path cluster_desc;
    std::vector<std::filesystem::path> soc_descs;  // index = fabric_chip
    std::filesystem::path mesh_graph_desc;
    std::filesystem::path env;
};

// Writes, into `out_dir` (created if missing, files replaced if present), chip_<fabric_chip>_soc_desc.yaml for each
// device, cluster_desc.yaml, mesh_graph_descriptor.textproto and env, as SPEC-0011 § Descriptors generated from the
// manifest specifies. `manifest` must have passed check_manifest() with package_facts(`package_soc_desc`); this
// function does not repeat those checks. `package_soc_desc` is the whole package's SoC descriptor (UMD
// tests/soc_descs/mmk.yaml).
//
// Throws DescriptorGenerationError, before writing anything, listing every G1 to G5 refusal that holds. Throws
// std::runtime_error (TT_FATAL) when `package_soc_desc` cannot be read or lacks a key the generator reads (dram_views,
// eth, keraunos.cores), or when a file cannot be written; a write failure can leave earlier files of this call in
// `out_dir`.
GeneratedDescriptors generate_descriptors(
    const Manifest& manifest, const std::filesystem::path& package_soc_desc, const std::filesystem::path& out_dir);

// The manifest's links whose backend is noc, each as its two ends' fabric_chip, lower first, in link order. Throws
// std::runtime_error (TT_FATAL) when a link end names a device the manifest does not have, which check_manifest()
// refuses as row 7.
std::vector<std::pair<uint32_t, uint32_t>> noc_links(const Manifest& manifest);

// The package facts check_manifest() takes, from the package SoC descriptor at `package_soc_desc`: its
// keraunos.cores as (hsio, cce), MEM_IERISC_LOCAL_HARTS harts a CCE, eth_l1_size bytes of SRAM a CCE, and the
// Ethernet tile pairs (0, 1) and (2, 3) the mmk 26ww45.eth-pcs-output-inter bring-up wires (chippy bringup_mmk.py
// KER_ETH_LINKS). Throws std::runtime_error (TT_FATAL) when the file cannot be read or lacks keraunos.cores or
// eth_l1_size.
PackageFacts package_facts(const std::filesystem::path& package_soc_desc);

}  // namespace tt::tt_fabric::vcluster
