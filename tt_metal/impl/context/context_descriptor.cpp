// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "context_descriptor.hpp"

namespace tt::tt_metal {

MetalliumObjectDescriptor::MetalliumObjectDescriptor(const std::string& mock_cluster_desc_path) :
    mock_cluster_desc_path_(
        mock_cluster_desc_path.empty() ? std::nullopt : std::optional<std::string>(mock_cluster_desc_path)) {}
MetalliumObjectDescriptor::MetalliumObjectDescriptor(std::optional<std::string> mock_cluster_desc_path) :
    mock_cluster_desc_path_(std::move(mock_cluster_desc_path)) {}

}  // namespace tt::tt_metal
