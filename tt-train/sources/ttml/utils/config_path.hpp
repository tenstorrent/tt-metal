// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <optional>
#include <string>

namespace ttml::utils {

/**
 * Get the tt-train root directory.
 *
 * Resolution order:
 * 1. TT_TRAIN_ROOT environment variable (if set)
 * 2. TT_METAL_HOME/tt-train (if TT_METAL_HOME is set)
 * 3. TTML_ROOT compile-time macro (set by CMake to CMAKE_SOURCE_DIR/tt-train)
 *
 * @return Path to tt-train root directory
 * @throws std::runtime_error if tt-train root cannot be determined
 */
std::filesystem::path get_tt_train_root();

/**
 * Get the configs root directory.
 *
 * @return get_tt_train_root() / "configs"
 */
std::filesystem::path get_configs_root();

/**
 * Resolve a config path. If the path is:
 * - Absolute: return as-is
 * - Relative: join with the provided configs_root (or get_configs_root() if not specified)
 *
 * @param config_path Config file name or relative path (e.g., "training_configs/mnist.yaml" or just "mnist.yaml")
 * @param configs_root Optional root directory for config resolution
 * @return Resolved absolute path to config file
 */
std::filesystem::path resolve_config_path(
    const std::string& config_path, const std::optional<std::filesystem::path>& configs_root = std::nullopt);

/**
 * Resolve a training config path.
 * Resolves relative to: get_configs_root() / "training_configs"
 *
 * @param config_name Config file name (e.g., "mnist.yaml" or "training_shakespeare_nanogpt.yaml")
 * @return Resolved absolute path to training config file
 */
std::filesystem::path resolve_training_config(const std::string& config_name);

/**
 * Resolve a model config path.
 * Resolves relative to: get_configs_root() / "model_configs"
 *
 * @param config_name Config file name (e.g., "nanogpt.yaml" or "tinyllama.yaml")
 * @return Resolved absolute path to model config file
 */
std::filesystem::path resolve_model_config(const std::string& config_name);

/**
 * Resolve a multihost config path.
 * Resolves relative to: get_configs_root() / "multihost_configs"
 *
 * @param config_name Config file name (e.g., "3worker_fabric.yaml")
 * @return Resolved absolute path to multihost config file
 */
std::filesystem::path resolve_multihost_config(const std::string& config_name);

}  // namespace ttml::utils
