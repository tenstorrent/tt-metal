// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/context/context_descriptor.hpp>

namespace tt::tt_metal {

MetaliumEnvDescriptor::MetaliumEnvDescriptor(const std::string& mock_cluster_desc_path) :
    mock_cluster_desc_path_(
        mock_cluster_desc_path.empty() ? std::nullopt : std::optional<std::string>(mock_cluster_desc_path)) {}
MetaliumEnvDescriptor::MetaliumEnvDescriptor(std::optional<std::string> mock_cluster_desc_path) :
    mock_cluster_desc_path_(std::move(mock_cluster_desc_path)) {}

}  // namespace tt::tt_metal
