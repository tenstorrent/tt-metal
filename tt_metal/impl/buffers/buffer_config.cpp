// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <buffer_config.hpp>

namespace tt::tt_metal {

std::ostream& operator<<(std::ostream& os, const ShardSpec& spec) {
    tt::stl::reflection::operator<<(os, spec);
    return os;
}

bool is_sharded(const TensorMemoryLayout& layout) {
    return (
        layout == TensorMemoryLayout::HEIGHT_SHARDED || layout == TensorMemoryLayout::WIDTH_SHARDED ||
        layout == TensorMemoryLayout::BLOCK_SHARDED);
}

bool ShardSpec::operator==(const ShardSpec&) const = default;
bool ShardSpec::operator!=(const ShardSpec&) const = default;

std::array<uint32_t, 2> ShardSpecBuffer::shape_in_pages() const {
    auto height_in_pages = page_shape[0] == 0 ? 0 : tensor_shard_spec.shape[0] / page_shape[0];
    auto width_in_pages = page_shape[1] == 0 ? 0 : tensor_shard_spec.shape[1] / page_shape[1];
    return {height_in_pages, width_in_pages};
}

DeviceAddr ShardSpecBuffer::num_pages() const {
    auto shape_in_pages_ = this->shape_in_pages();
    return shape_in_pages_[0] * shape_in_pages_[1];
}
}  // namespace tt::tt_metal

namespace tt::stl::json {
tt_metal::ShardSpec from_json_t<tt_metal::ShardSpec>::operator()(const nlohmann::json& json_object) const {
    const auto& shard_mode = from_json<tt_metal::ShardMode>(json_object.at("mode"));
    const auto& physical_shard_shape =
        from_json<std::optional<std::array<uint32_t, 2>>>(json_object.at("physical_shard_shape"));
    if (physical_shard_shape.has_value()) {
        TT_FATAL(
            shard_mode == tt::tt_metal::ShardMode::LOGICAL,
            "Physical shard shape can only be provided in logical sharding mode!");
        return tt_metal::ShardSpec{
            from_json<CoreRangeSet>(json_object.at("grid")),
            from_json<std::array<uint32_t, 2>>(json_object.at("shape")),
            physical_shard_shape.value(),
            from_json<tt_metal::ShardOrientation>(json_object.at("orientation"))};
    }
    return tt_metal::ShardSpec{
        from_json<CoreRangeSet>(json_object.at("grid")),
        from_json<std::array<uint32_t, 2>>(json_object.at("shape")),
        from_json<tt_metal::ShardOrientation>(json_object.at("orientation")),
        shard_mode};
}
}  // namespace tt::stl::json
