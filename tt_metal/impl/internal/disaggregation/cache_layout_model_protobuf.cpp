// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/internal/disaggregation/cache_layout_model_protobuf.hpp"

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <enchantum/enchantum.hpp>

#include "protobuf/cache_layout_model.pb.h"

namespace tt::tt_metal::internal::disaggregation {

namespace {

namespace pb = ::tt::disaggregation::proto;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshMapperConfig;
using tt::tt_metal::distributed::MeshShape;

constexpr uint32_t kSchemaVersion = 1;

template <typename Enum>
std::string name_of(Enum value) {
    return std::string(enchantum::to_string(value));
}

template <typename Enum>
Enum parse(const std::string& name, std::string_view what) {
    const auto value = enchantum::cast<Enum>(name);
    if (!value.has_value()) {
        throw std::runtime_error("CacheLayoutModel: unknown " + std::string(what) + " '" + name + "'");
    }
    return *value;
}

template <typename Repeated>
std::vector<uint32_t> to_vector(const Repeated& values) {
    return std::vector<uint32_t>(values.begin(), values.end());
}

void to_proto(const TensorSpec& tensor, pb::CacheTensorSpec& out) {
    if (tensor.tile() != Tile()) {
        throw std::runtime_error("CacheLayoutModel: only the default 32x32 tile is serialized");
    }
    const auto& shape = tensor.logical_shape();
    for (size_t i = 0; i < shape.rank(); ++i) {
        out.add_logical_shape(static_cast<uint32_t>(shape[i]));
    }
    out.set_dtype(name_of(tensor.data_type()));
    out.set_layout(name_of(tensor.layout()));
    out.set_buffer_type(name_of(tensor.memory_config().buffer_type()));
    const auto& nd = tensor.memory_config().nd_shard_spec();
    if (!nd.has_value()) {
        throw std::runtime_error("CacheLayoutModel: a cache tensor must be ND-sharded");
    }
    auto& shard = *out.mutable_nd_shard_spec();
    for (size_t i = 0; i < nd->shard_shape.rank(); ++i) {
        shard.add_shard_shape(static_cast<uint32_t>(nd->shard_shape[i]));
    }
    for (const CoreRange& range : nd->grid.ranges()) {
        auto& r = *shard.add_grid();
        r.set_start_x(range.start_coord.x);
        r.set_start_y(range.start_coord.y);
        r.set_end_x(range.end_coord.x);
        r.set_end_y(range.end_coord.y);
    }
    shard.set_orientation(name_of(nd->orientation));
    shard.set_shard_distribution_strategy(name_of(nd->shard_distribution_strategy));
}

TensorSpec from_proto(const pb::CacheTensorSpec& in) {
    std::vector<CoreRange> ranges;
    for (const auto& r : in.nd_shard_spec().grid()) {
        ranges.emplace_back(CoreCoord(r.start_x(), r.start_y()), CoreCoord(r.end_x(), r.end_y()));
    }
    NdShardSpec nd{
        Shape(to_vector(in.nd_shard_spec().shard_shape())),
        CoreRangeSet(ranges),
        parse<ShardOrientation>(in.nd_shard_spec().orientation(), "shard orientation"),
        parse<ShardDistributionStrategy>(
            in.nd_shard_spec().shard_distribution_strategy(), "shard distribution strategy")};
    MemoryConfig memory_config(parse<BufferType>(in.buffer_type(), "buffer type"), nd);
    return TensorSpec(
        Shape(to_vector(in.logical_shape())),
        TensorLayout(
            parse<DataType>(in.dtype(), "dtype"), PageConfig(parse<Layout>(in.layout(), "layout")), memory_config));
}

void to_proto(const TemporalPolicy& temporal, pb::CacheTemporalPolicy& out) {
    std::visit(
        [&out](const auto& policy) {
            using T = std::decay_t<decltype(policy)>;
            if constexpr (std::is_same_v<T, temporal::Dense>) {
                out.mutable_dense();
            } else if constexpr (std::is_same_v<T, temporal::Window>) {
                out.mutable_window()->set_width_tokens(policy.width.get());
            } else if constexpr (std::is_same_v<T, temporal::BlockLocal>) {
                out.mutable_block_local()->set_span_tokens(policy.span.get());
            } else if constexpr (std::is_same_v<T, temporal::Rolling>) {
                out.mutable_rolling();
            } else if constexpr (std::is_same_v<T, temporal::Static>) {
                out.mutable_static_();
            } else {
                out.mutable_no_sequence();
            }
        },
        temporal);
}

TemporalPolicy from_proto(const pb::CacheTemporalPolicy& in) {
    switch (in.policy_case()) {
        case pb::CacheTemporalPolicy::kDense: return temporal::Dense{};
        case pb::CacheTemporalPolicy::kWindow: return temporal::Window{WindowTokens{in.window().width_tokens()}};
        case pb::CacheTemporalPolicy::kBlockLocal:
            return temporal::BlockLocal{BlockLocalSpan{in.block_local().span_tokens()}};
        case pb::CacheTemporalPolicy::kRolling: return temporal::Rolling{};
        case pb::CacheTemporalPolicy::kStatic: return temporal::Static{};
        case pb::CacheTemporalPolicy::kNoSequence: return temporal::None{};
        default: throw std::runtime_error("CacheLayoutModel: a cache has no temporal policy");
    }
}

void to_proto(const TensorTopology& topology, pb::CacheTensorTopology& out) {
    const MeshShape& shape = topology.distribution_shape();
    for (size_t i = 0; i < shape.dims(); ++i) {
        out.add_distribution_shape(shape[static_cast<int32_t>(i)]);
    }
    for (const auto& placement : topology.placements()) {
        auto& p = *out.add_placements();
        if (const auto* shard = std::get_if<MeshMapperConfig::Shard>(&placement)) {
            p.mutable_shard()->set_dim(shard->dim);
        } else {
            p.mutable_replicate();
        }
    }
    for (const MeshCoordinate& coord : topology.mesh_coords()) {
        auto& c = *out.add_mesh_coords();
        for (uint32_t value : coord.coords()) {
            c.add_coords(value);
        }
    }
}

TensorTopology from_proto(const pb::CacheTensorTopology& in) {
    ttsl::SmallVector<MeshMapperConfig::Placement> placements;
    for (const auto& p : in.placements()) {
        if (p.has_shard()) {
            placements.push_back(MeshMapperConfig::Shard{.dim = p.shard().dim()});
        } else if (p.has_replicate()) {
            placements.push_back(MeshMapperConfig::Replicate{});
        } else {
            throw std::runtime_error("CacheLayoutModel: a placement is neither Shard nor Replicate");
        }
    }
    std::vector<MeshCoordinate> coords;
    for (const auto& c : in.mesh_coords()) {
        const std::vector<uint32_t> values = to_vector(c.coords());
        coords.emplace_back(ttsl::Span<const uint32_t>(values));
    }
    const std::vector<uint32_t> shape = to_vector(in.distribution_shape());
    return TensorTopology(MeshShape(ttsl::Span<const uint32_t>(shape)), placements, coords);
}

void to_proto(const GenerationPolicy& policy, pb::CacheGenerationPolicy& out) {
    out.set_bank_scheme(static_cast<pb::CacheBankScheme>(policy.bank_scheme));
    out.set_bank_order(static_cast<pb::CacheBankOrder>(policy.bank_order));
    out.set_k_chunk_size(policy.k_chunk_size.get());
    if (policy.device_chunk_size.has_value()) {
        out.set_device_chunk_size(policy.device_chunk_size->get());
    }
    out.set_banks_per_head(policy.banks_per_head.get());
    out.set_num_blocks(policy.num_blocks);
}

GenerationPolicy from_proto(const pb::CacheGenerationPolicy& in) {
    if (!pb::CacheBankScheme_IsValid(in.bank_scheme()) || !pb::CacheBankOrder_IsValid(in.bank_order())) {
        throw std::runtime_error("CacheLayoutModel: unknown bank scheme or bank order");
    }
    GenerationPolicy policy;
    policy.bank_scheme = static_cast<BankScheme>(in.bank_scheme());
    policy.bank_order = static_cast<BankOrder>(in.bank_order());
    policy.k_chunk_size = KChunkSize{in.k_chunk_size()};
    if (in.has_device_chunk_size()) {
        policy.device_chunk_size = DeviceChunkSize{in.device_chunk_size()};
    }
    policy.banks_per_head = BanksPerHead{in.banks_per_head()};
    policy.num_blocks = in.num_blocks();
    return policy;
}

pb::CacheLayoutModel to_proto_message(const CacheLayoutModel& model) {
    pb::CacheLayoutModel out;
    out.set_schema_version(kSchemaVersion);
    for (const CacheConfig& cache : model.caches) {
        auto& c = *out.add_caches();
        c.set_name(cache.name);
        to_proto(cache.spec.tensor, *c.mutable_spec()->mutable_tensor());
        to_proto(cache.spec.temporal, *c.mutable_spec()->mutable_temporal());
        c.mutable_spec()->set_arch(name_of(cache.spec.arch));
        to_proto(cache.topology, *c.mutable_topology());
        to_proto(cache.policy, *c.mutable_policy());
        c.set_base_addr(cache.base_addr);
        for (uint32_t layer : cache.layers) {
            c.add_layers(layer);
        }
        c.set_first_layer(cache.first_layer);
        c.set_num_layers(cache.num_layers);
        if (cache.mesh_id.has_value()) {
            c.set_mesh_id(**cache.mesh_id);
        }
        if (cache.head.has_value()) {
            c.set_head(*cache.head);
        }
    }
    out.set_mesh_id(*model.mesh_id);
    for (size_t i = 0; i < model.mesh_shape.dims(); ++i) {
        out.add_mesh_shape(model.mesh_shape[static_cast<int32_t>(i)]);
    }
    auto& g = *out.mutable_geometry();
    g.set_num_layers(model.geometry.num_layers);
    g.set_num_slots(model.geometry.num_slots);
    g.set_max_seq_len(model.geometry.max_seq_len);
    g.set_position_step(model.geometry.position_step);
    return out;
}

CacheLayoutModel from_proto_message(const pb::CacheLayoutModel& in) {
    if (in.schema_version() != kSchemaVersion) {
        throw std::runtime_error(
            "CacheLayoutModel: schema_version " + std::to_string(in.schema_version()) + " is not " +
            std::to_string(kSchemaVersion));
    }
    CacheLayoutModel model;
    for (const auto& c : in.caches()) {
        model.caches.push_back(CacheConfig{
            .spec =
                KvLayoutSpec{
                    .tensor = from_proto(c.spec().tensor()),
                    .temporal = from_proto(c.spec().temporal()),
                    .arch = parse<tt::ARCH>(c.spec().arch(), "arch")},
            .topology = from_proto(c.topology()),
            .policy = from_proto(c.policy()),
            .base_addr = c.base_addr(),
            .name = c.name(),
            .layers = to_vector(c.layers()),
            .first_layer = c.first_layer(),
            .num_layers = c.num_layers(),
            .mesh_id = c.has_mesh_id() ? std::optional<tt::tt_fabric::MeshId>(tt::tt_fabric::MeshId{c.mesh_id()})
                                       : std::nullopt,
            .head = c.has_head() ? std::optional<uint32_t>(c.head()) : std::nullopt});
    }
    model.mesh_id = tt::tt_fabric::MeshId{in.mesh_id()};
    const std::vector<uint32_t> mesh_shape = to_vector(in.mesh_shape());
    model.mesh_shape = MeshShape(ttsl::Span<const uint32_t>(mesh_shape));
    model.geometry = MapGeometry{
        .num_layers = in.geometry().num_layers(),
        .num_slots = in.geometry().num_slots(),
        .max_seq_len = in.geometry().max_seq_len(),
        .position_step = in.geometry().position_step()};
    return model;
}

}  // namespace

std::string export_to_protobuf(const CacheLayoutModel& model) {
    std::string out;
    if (!to_proto_message(model).SerializeToString(&out)) {
        throw std::runtime_error("Failed to serialize CacheLayoutModel to protobuf");
    }
    return out;
}

void export_to_protobuf_file(const CacheLayoutModel& model, const std::string& path) {
    const std::string data = export_to_protobuf(model);
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

CacheLayoutModel import_cache_layout_model_from_protobuf(const std::string& data) {
    pb::CacheLayoutModel message;
    if (!message.ParseFromString(data)) {
        throw std::runtime_error("Failed to parse protobuf data as CacheLayoutModel");
    }
    return from_proto_message(message);
}

CacheLayoutModel import_cache_layout_model_from_protobuf_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }
    const std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return import_cache_layout_model_from_protobuf(data);
}

}  // namespace tt::tt_metal::internal::disaggregation
