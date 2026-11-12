// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_region_json.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>
#include <variant>

#include <tt_stl/assert.hpp>
#include <tt_stl/overloaded.hpp>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_type_registry.hpp"

namespace tt::tt_fabric::decode {

namespace {

// A stream register, as its region's reads are shaped by it.
struct StreamRegisterShape {
    const manifest::content::StreamReg* stream_register = nullptr;
    const StreamRegisterFacts* register_facts = nullptr;
};

// How the manifest says to shape one region's reads.
struct RegionShape {
    // One element is written as `value`, several as `elements`.
    bool one_element = true;
    // Null for an L1 region.
    std::optional<StreamRegisterShape> stream_register;
    // Whether the region has rounds, so `rounds` is written, even when no round read it.
    bool multi_sampled = false;
};

// A stream register's read: `raw`, its whole word, then its value under the read mask by its role. A count or the
// ERISC sync is shown by what it means; any other value, such as the context switch's state, as its type's
// interpreter shows it.
template <typename T>
json make_stream_register_read_json(
    const RegionRead<T>& read, const StreamRegisterShape& shape, const InterpretContext& context) {
    json out;
    out["raw"] = read.word.has_value() ? json(*read.word) : json(nullptr);
    if constexpr (std::is_same_v<T, uint32_t>) {
        const uint32_t value = read.elements.at(0);
        if (const auto* stream_role = std::get_if<StreamRole>(&shape.stream_register->role)) {
            out.update(make_stream_count_json(value, *stream_role, *shape.register_facts));
            return out;
        }
        if (shape.stream_register->role ==
            manifest::StreamRegisterRole(manifest::ScratchRole::MULTI_RISC_TEARDOWN_SYNC)) {
            out.update(make_erisc_sync_json(value));
            return out;
        }
    }
    out["value"] = Interpreter<T>::make_json(read.elements.at(0), context);
    return out;
}

// One read's elements, interpreted: `value` or `elements`, or, for a type shown only as a whole region, its keys. A
// stream register's as make_stream_register_read_json shows it.
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
        if (shape.stream_register.has_value()) {
            return make_stream_register_read_json(read, *shape.stream_register, context);
        }
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

// The region's reads, in its shape: the full read's `at` and values, then its rounds.
template <typename T>
json make_reads_json(const RegionReads<T>& reads, const RegionShape& shape, const InterpretContext& context) {
    json out = json::object();
    const auto full_read =
        std::ranges::find_if(reads, [](const RegionRead<T>& read) { return !read.round.has_value(); });
    if (full_read != reads.end()) {
        out["at"] = full_read->bin_offset;
        out.update(make_read_values_json(*full_read, shape, context));
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
        std::visit(
            ttsl::overloaded{
                [&](const InactiveL1& l1) {
                    out["at"] = l1.bin_offset;
                    out.update(summarize_bytes(l1.bytes));
                },
                [&](const InactiveRegister& stream_register) {
                    out["at"] = stream_register.bin_offset;
                    out["raw"] = stream_register.word;
                },
            },
            *inactive);
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
    const RouterValues& values,
    const manifest::content::StreamReg& stream_register,
    const StreamRegisterFacts& register_facts,
    const InterpretContext& context) {
    return make_region_decoded_json(
        values,
        stream_register.type,
        RegionShape{
            .one_element = true,
            .stream_register =
                StreamRegisterShape{.stream_register = &stream_register, .register_facts = &register_facts},
            .multi_sampled = stream_register.multi_sampled.has_value()},
        context);
}

}  // namespace tt::tt_fabric::decode
