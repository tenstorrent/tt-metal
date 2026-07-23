// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <ostream>
#include <tuple>

#include <tt-metalium/experimental/tensor/tensor_types.hpp>

namespace ttsl::json {
template <typename T>
struct to_json_t;
template <typename T>
struct from_json_t;
}  // namespace ttsl::json

namespace tt::tt_metal {

class MemoryConfigImpl;

class MemoryConfig final {
public:
    MemoryConfig();  // Interleaved DRAM
    explicit MemoryConfig(
        TensorMemoryLayout memory_layout,
        BufferType buffer_type = BufferType::DRAM,
        std::optional<ShardSpec> shard_spec = std::nullopt);
    explicit MemoryConfig(BufferType buffer_type, std::optional<NdShardSpec> nd_shard_spec = std::nullopt);
    ~MemoryConfig();
    MemoryConfig(const MemoryConfig& other);
    MemoryConfig& operator=(const MemoryConfig& other);
    MemoryConfig(MemoryConfig&& other) noexcept;
    MemoryConfig& operator=(MemoryConfig&& other) noexcept;

    TensorMemoryLayout memory_layout() const;
    BufferType buffer_type() const;
    const std::optional<ShardSpec>& shard_spec() const;
    const std::optional<NdShardSpec>& nd_shard_spec() const;
    bool created_with_nd_shard_spec() const;

    bool is_sharded() const;
    bool is_l1() const;
    bool is_dram() const;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "memory_layout", "buffer_type", "shard_spec", "nd_shard_spec", "created_with_nd_shard_spec");
    std::tuple<
        const TensorMemoryLayout&,
        const BufferType&,
        const std::optional<ShardSpec>&,
        const std::optional<NdShardSpec>&,
        const bool&>
    attribute_values() const;

    static MemoryConfig create_with_prepopulated_shard_specs(
        TensorMemoryLayout memory_layout,
        BufferType buffer_type,
        std::optional<ShardSpec> shard_spec,
        std::optional<NdShardSpec> nd_shard_spec,
        bool created_with_nd_shard_spec);

    friend std::ostream& operator<<(std::ostream& os, const MemoryConfig& config);

    // pre-condition: the MemoryConfig must not be in a moved-from state.
    MemoryConfigImpl& impl();
    const MemoryConfigImpl& impl() const;

private:
    MemoryConfig(
        TensorMemoryLayout memory_layout,
        BufferType buffer_type,
        std::optional<ShardSpec> shard_spec,
        std::optional<NdShardSpec> nd_shard_spec,
        bool created_with_nd_shard_spec);

    // impl_ may be nullptr if the MemoryConfig is in a moved-from state.
    // Avoid using impl_ directly; use the impl() accessor instead.
    std::unique_ptr<MemoryConfigImpl> impl_;
};

std::ostream& operator<<(std::ostream& os, const MemoryConfig& config);

bool operator==(const MemoryConfig& config_a, const MemoryConfig& config_b);
bool operator!=(const MemoryConfig& config_a, const MemoryConfig& config_b);

}  // namespace tt::tt_metal

template <>
struct ttsl::json::to_json_t<tt::tt_metal::MemoryConfig> {
    nlohmann::json operator()(const tt::tt_metal::MemoryConfig& config) const;
};

template <>
struct ttsl::json::from_json_t<tt::tt_metal::MemoryConfig> {
    tt::tt_metal::MemoryConfig operator()(const nlohmann::json& json_object) const;
};
