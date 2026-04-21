// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <tt_stl/assert.hpp>
#include <tt_stl/reflection.hpp>

#include <tt-metalium/experimental/tensor/tensor_types.hpp>
#include <tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp>

namespace tt::tt_metal {

MemoryConfig::MemoryConfig(
    TensorMemoryLayout memory_layout, BufferType buffer_type, std::optional<ShardSpec> shard_spec) :
    memory_layout_(memory_layout), buffer_type_(buffer_type), shard_spec_(std::move(shard_spec)) {}

MemoryConfig MemoryConfig::with_shard_spec(std::optional<ShardSpec> shard_spec) const {
    // with_shard_spec preserves memory_layout_ but swaps in a new (legacy 2D) shard_spec.
    // That's valid for a same-layout update (e.g. HEIGHT_SHARDED -> HEIGHT_SHARDED with a
    // different shard shape), but not for transitioning FROM ND_SHARDED: the resulting
    // MemoryConfig would have memory_layout_ == ND_SHARDED yet shard_spec_ != nullopt
    //(a legacy shard_spec_ populated), which is internally inconsistent. MemoryConfig
    // doesn't have enough context (tensor padded shape) to infer the correct downgrade
    // target (HEIGHT/WIDTH/BLOCK_SHARDED), so callers must construct a new MemoryConfig
    // directly with the intended layout.
    TT_FATAL(
        memory_layout_ != TensorMemoryLayout::ND_SHARDED,
        "MemoryConfig::with_shard_spec cannot be used to transition from ND_SHARDED; "
        "construct a MemoryConfig directly with the target memory_layout and shard_spec instead.");
    return MemoryConfig(memory_layout_, buffer_type_, std::move(shard_spec));
}

MemoryConfig::MemoryConfig(BufferType buffer_type, std::optional<NdShardSpec> nd_shard_spec) :
    memory_layout_(nd_shard_spec.has_value() ? TensorMemoryLayout::ND_SHARDED : TensorMemoryLayout::INTERLEAVED),
    buffer_type_(buffer_type),
    nd_shard_spec_(std::move(nd_shard_spec)),
    created_with_nd_shard_spec_(nd_shard_spec_.has_value()) {}

MemoryConfig::MemoryConfig(
    TensorMemoryLayout memory_layout,
    BufferType buffer_type,
    std::optional<ShardSpec> shard_spec,
    std::optional<NdShardSpec> nd_shard_spec,
    bool created_with_nd_shard_spec) :
    memory_layout_(memory_layout),
    buffer_type_(buffer_type),
    shard_spec_(std::move(shard_spec)),
    nd_shard_spec_(std::move(nd_shard_spec)),
    created_with_nd_shard_spec_(created_with_nd_shard_spec) {}

MemoryConfig MemoryConfig::create_with_prepopulated_shard_specs(
    TensorMemoryLayout memory_layout,
    BufferType buffer_type,
    std::optional<ShardSpec> shard_spec,
    std::optional<NdShardSpec> nd_shard_spec,
    bool created_with_nd_shard_spec) {
    return MemoryConfig(
        memory_layout, buffer_type, std::move(shard_spec), std::move(nd_shard_spec), created_with_nd_shard_spec);
}

bool MemoryConfig::is_sharded() const {
    switch (this->memory_layout_) {
        case TensorMemoryLayout::HEIGHT_SHARDED:
        case TensorMemoryLayout::WIDTH_SHARDED:
        case TensorMemoryLayout::BLOCK_SHARDED:
        case TensorMemoryLayout::ND_SHARDED: return true;
        default: return false;
    }
}

bool MemoryConfig::is_l1() const { return buffer_type_ == BufferType::L1 or buffer_type_ == BufferType::L1_SMALL; }

bool MemoryConfig::is_dram() const { return buffer_type_ == BufferType::DRAM; }

bool operator==(const MemoryConfig& config_a, const MemoryConfig& config_b) {
    if (config_a.buffer_type() != config_b.buffer_type() || config_a.memory_layout() != config_b.memory_layout()) {
        return false;
    }
    // Compare only the authoritative shard spec based on creation path.
    // After creating a memory_config with an nd_shard_spec or a shard_spec,
    // when that memory_config gets passed into TensorSpec, TensorSpec auto-populates
    // the other spec (nd from legacy or legacy from nd shard spec), if an equivalent spec exists.
    // so the non-authoritative field can differ between a user-constructed MemoryConfig
    // and the tensor's generated MemoryConfig without there being a semantic difference.

    // E.g., If you create a memory_config with an nd_shard_spec, when you pass that memory_config to create a tensor,
    // a TensorSpec will be constructed with that memory config. If there is an equivalent shard_spec, the TensorSpec
    // infrastructure will auto-populate the shard_spec field with the equivalent shard_spec. When you then compare
    // the two memory_configs (the one you specified and the one the created Tensor has), they will be the same
    // semantically, but the memory_config returned by the Tensor will have a shard_spec field, while the one you
    // originally specified will not. The equality operator should test for semantic equality, not structural equality.
    if (config_a.created_with_nd_shard_spec() && config_b.created_with_nd_shard_spec()) {
        return config_a.nd_shard_spec() == config_b.nd_shard_spec();
    }
    if (!config_a.created_with_nd_shard_spec() && !config_b.created_with_nd_shard_spec()) {
        return config_a.shard_spec() == config_b.shard_spec();
    }
    // Mixed creation paths: compare both fields
    return config_a.shard_spec() == config_b.shard_spec() && config_a.nd_shard_spec() == config_b.nd_shard_spec();
}

bool operator!=(const MemoryConfig& config_a, const MemoryConfig& config_b) { return not(config_a == config_b); }

std::ostream& operator<<(std::ostream& os, const MemoryConfig& config) {
    tt::stl::reflection::operator<<(os, config);
    return os;
}

}  // namespace tt::tt_metal

nlohmann::json ttsl::json::to_json_t<tt::tt_metal::MemoryConfig>::operator()(
    const tt::tt_metal::MemoryConfig& config) const {
    nlohmann::json json_object;
    json_object["memory_layout"] = config.memory_layout();
    json_object["buffer_type"] = config.buffer_type();
    json_object["created_with_nd_shard_spec"] = config.created_with_nd_shard_spec();
    if (config.created_with_nd_shard_spec()) {
        if (config.nd_shard_spec().has_value()) {
            json_object["nd_shard_spec"] = ttsl::json::to_json(config.nd_shard_spec().value());
        }
    } else {
        if (config.shard_spec().has_value()) {
            json_object["shard_spec"] = ttsl::json::to_json(config.shard_spec().value());
        }
    }
    return json_object;
}

tt::tt_metal::MemoryConfig ttsl::json::from_json_t<tt::tt_metal::MemoryConfig>::operator()(
    const nlohmann::json& json_object) const {
    auto memory_layout = json_object["memory_layout"].get<tt::tt_metal::TensorMemoryLayout>();
    auto buffer_type = json_object["buffer_type"].get<tt::tt_metal::BufferType>();
    auto created_with_nd_shard_spec = json_object["created_with_nd_shard_spec"].get<bool>();
    if (created_with_nd_shard_spec) {
        auto nd_shard_spec = ttsl::json::from_json<tt::tt_metal::NdShardSpec>(json_object["nd_shard_spec"]);
        return tt::tt_metal::MemoryConfig(buffer_type, std::move(nd_shard_spec));
    }
    std::optional<tt::tt_metal::ShardSpec> shard_spec;
    if (json_object.contains("shard_spec")) {
        shard_spec = ttsl::json::from_json<tt::tt_metal::ShardSpec>(json_object["shard_spec"]);
    }
    return tt::tt_metal::MemoryConfig(memory_layout, buffer_type, std::move(shard_spec));
}
