// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include <tt-metalium/experimental/context/metal_env.hpp>

namespace tt::tt_metal {

// ClusterType now lives in metal_env.hpp; this header is kept for the free functions below.

/**
 * @brief Get the type of the current cluster
 *
 * Equivalent to MetalEnv::get_cluster_type() on the default context's MetalEnv.
 *
 * @return ClusterType The cluster type detected at runtime
 */
tt::tt_metal::ClusterType GetClusterType();

/**
 * @brief Serialize the cluster descriptor to a file and return the path
 *
 * Equivalent to MetalEnv::serialize_cluster_descriptor() on the default context's MetalEnv.
 *
 * @return std::string Path to the serialized cluster descriptor file
 */
std::string SerializeClusterDescriptor();

}  // namespace tt::tt_metal
