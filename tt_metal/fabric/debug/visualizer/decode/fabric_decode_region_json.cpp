// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_region_json.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>

#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_type_registry.hpp"

namespace tt::tt_fabric::decode {

namespace {

// How the manifest says to shape one region's reads.
struct RegionShape {
    // One element is written as `value`, several as `elements`.
    bool one_element = true;
    // A stream register's value width, for upper_bits. Null for an L1 region.
    std::optional<uint32_t> value_bits;
    // Whether the region has rounds, so `rounds` is written, even when no round read it.
    bool multi_sampled = false;
};

// One read's elements, interpreted: `value` or `elements`, or, for a type shown only as a whole region, its keys.
template <typename T>
json make_read_values_json(const RegionRead<T>& read, const RegionShape& shape, const InterpretContext& context) {
    if constexpr (requires { Interpreter<T>::make_region_json(std::span<const T>{}, context); }) {
        return Interpreter<T>::make_region_json(std::span<const T>(read.elements), context);
    } else {
        const auto interpret_element = [&](uint32_t index) {
            InterpretContext element_context = context;
            element_context.element_index = index;
            return Interpreter<T>::make_json(read.elements[index], element_context);
        };
        json out;
        if (shape.one_element) {
            out["value"] = interpret_element(0);
            return out;
        }
        json elements = json::array();
        for (uint32_t index = 0; index < read.elements.size(); ++index) {
            elements.push_back(interpret_element(index));
        }
        out["elements"] = std::move(elements);
        return out;
    }
}

// Adds `upper_bits` to `out` when the read is a stream register's and its word has bits set above value_bits.
template <typename T>
void add_upper_bits(json& out, const RegionRead<T>& read, const RegionShape& shape) {
    if (!read.word.has_value() || !shape.value_bits.has_value() || *shape.value_bits >= 32) {
        return;
    }
    const uint32_t upper_bits = *read.word & ~((uint32_t{1} << *shape.value_bits) - 1);
    if (upper_bits != 0) {
        out["upper_bits"] = upper_bits;
    }
}

// The region's reads, in its shape: the full read's `at` and values, then its rounds.
template <typename T>
json make_reads_json(const RegionReads<T>& reads, const RegionShape& shape, const InterpretContext& context) {
    json out = json::object();
    const auto full_read =
        std::ranges::find_if(reads, [](const RegionRead<T>& read) { return !read.round.has_value(); });
    if (full_read != reads.end()) {
        out["at"] = full_read->bin_offset;
        out.update(make_read_values_json(*full_read, shape, context));
        add_upper_bits(out, *full_read, shape);
    }
    if (shape.multi_sampled) {
        json rounds = json::array();
        for (const RegionRead<T>& read : reads) {
            if (!read.round.has_value()) {
                continue;
            }
            json round;
            round["round"] = *read.round;
            round["t_ns"] = read.t_ns;
            round.update(make_read_values_json(read, shape, context));
            add_upper_bits(round, read, shape);
            rounds.push_back(std::move(round));
        }
        out["rounds"] = std::move(rounds);
    }
    return out;
}

// The region at context.path, whatever its kind: unread, inactive, or its reads as its registered type `type_name`.
json make_region_decoded_json(
    const RouterValues& values, std::string_view type_name, const RegionShape& shape, const InterpretContext& context) {
    if (const Unread* unread = values.find_unread(context.path)) {
        json out;
        out["unread"] = manifest::lower_enum_name(unread->code);
        return out;
    }
    if (const InactiveRegion* inactive = values.find_inactive(context.path)) {
        json out;
        out["active"] = false;
        out["at"] = inactive->bin_offset;
        out.update(summarize_bytes(inactive->bytes));
        return out;
    }
    std::optional<json> out;
    const bool registered = manifest::visit_registered_type(type_name, [&]<typename T>(std::type_identity<T>) {
        if (const RegionReads<T>* reads = values.find_reads<T>(context.path)) {
            out = make_reads_json(*reads, shape, context);
        }
    });
    TT_FATAL(registered, "Fabric decode: {} names type {}, which this build doesn't register", context.path, type_name);
    TT_FATAL(
        out.has_value(),
        "Fabric decode: {} has no reads, inactive bytes or unread reason, so typed read didn't walk it",
        context.path);
    return std::move(*out);
}

}  // namespace

json make_decoded_region_json(
    const RouterValues& values, const manifest::content::L1& region, const InterpretContext& context) {
    return make_region_decoded_json(
        values,
        region.type,
        RegionShape{.one_element = region.num_elements == 1, .multi_sampled = region.multi_sampled.has_value()},
        context);
}

json make_decoded_region_json(
    const RouterValues& values, const manifest::content::StreamReg& stream_register, const InterpretContext& context) {
    return make_region_decoded_json(
        values,
        stream_register.type,
        RegionShape{
            .one_element = true,
            .value_bits = stream_register.value_bits,
            .multi_sampled = stream_register.multi_sampled.has_value()},
        context);
}

}  // namespace tt::tt_fabric::decode
