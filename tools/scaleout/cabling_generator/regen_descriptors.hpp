// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <set>
#include <string>

#include <cabling_generator/cabling_generator.hpp>

namespace tt::scaleout_tools {

struct RegenDescriptorsSummary {
    // Cables (port-level connections) removed because at least one of their channels was
    // reported as unretrainable. Treated as whole-cable failures per the
    // dead-channel == dead-cable policy.
    std::set<PhysicalPortConnection> pruned_cables;
    // Number of channel-level connections in the regenerated FSD after pruning.
    size_t channels_remaining = 0;
    // Number of distinct channel endpoints read from the input YAML (informational).
    size_t input_dead_channels = 0;
    // Output paths written by the regen step.
    std::string output_fsd_path;
    std::string output_cabling_descriptor_path;
    std::string output_deployment_descriptor_path;
};

// Read a list of unretrainable channel endpoints from `unretrainable_channels_yaml_path`
// (the artifact emitted by run_cluster_validation on retrain exhaustion), construct a
// CablingGenerator from `cabling_descriptor_path` + `deployment_descriptor_path`, prune any
// cable whose channel expansion intersects the dead-channel set, and write the regenerated
// descriptor set into `output_dir`:
//   <output_dir>/factory_system_descriptor.textproto
//   <output_dir>/cabling_descriptor.textproto
//   <output_dir>/deployment_descriptor.textproto
//
// The deployment descriptor is emitted as-is (cable failures don't change hardware
// placement); it's included so the output dir is a self-contained, regen_cabling-replayable
// descriptor set. The original input descriptors are not modified.
//
// Returns a summary of what was pruned plus the emitted output paths. Throws
// std::runtime_error on I/O failures, malformed YAML, or descriptor parse errors.
RegenDescriptorsSummary regenerate_descriptors_excluding_dead_channels(
    const std::string& cabling_descriptor_path,
    const std::string& deployment_descriptor_path,
    const std::string& unretrainable_channels_yaml_path,
    const std::string& output_dir);

}  // namespace tt::scaleout_tools
