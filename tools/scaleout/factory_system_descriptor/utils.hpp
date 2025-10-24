// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace tt::scaleout_tools {

// Common utility function for validating FSD against discovered GSD
// Validates that the Factory System Descriptor (FSD) matches the Global System Descriptor (GSD)
// Parameters:
//   - fsd_filename: Path to the FSD protobuf text format file
//   - gsd_filename: Path to the GSD YAML file
//   - strict_validation: If true, checks that all connections match bidirectionally
//                        If false, only checks that GSD connections exist in FSD
//   - assert_on_connection_mismatch: If true, throws an error if there are connection mismatches
//                                    If false, missing connections are logged without an error being thrown
void validate_fsd_against_gsd(
    const std::string& fsd_filename,
    const std::string& gsd_filename,
    bool strict_validation = true,
    bool assert_on_connection_mismatch = true);

}  // namespace tt::scaleout_tools
