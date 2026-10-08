// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_struct_layouts.hpp"

#include <algorithm>
#include <optional>
#include <type_traits>
#include <utility>

#include <enchantum/enchantum.hpp>
#include <tt_stl/assert.hpp>

#include "tt_metal/llrt/hal.hpp"

namespace tt::tt_fabric::layout {

namespace {

using tt::tt_metal::dev_msgs::Factory;

// Layout of a scalar member in a HAL-generated struct. The name from the generated Field enum, the offset from the
// factory, and the type from the generated accessor.
template <typename Struct, typename Struct::Field F>
Member hal_scalar_member(const Factory& factory) {
    using Traits = typename Struct::template FieldTraits<true, F>;
    static_assert(std::is_reference_v<typename Traits::type>, "only scalar fields are supported");
    using Scalar = std::remove_cv_t<typename Traits::element_type>;
    return Member{enchantum::to_string(F), static_cast<uint32_t>(factory.offset_of<Struct>(F)), type_of<Scalar>()};
}

template <typename Struct>
void add_hal_struct_type(const Factory& factory, std::vector<StructType>& types);

template <typename Struct>
Type hal_struct_of(const Factory& factory, uint32_t count, std::vector<StructType>& types) {
    add_hal_struct_type<Struct>(factory, types);
    const auto size = static_cast<uint32_t>(factory.size_of<Struct>());
    return {element::Struct{enchantum::type_name<Struct>}, size * std::max(count, 1u), count};
}

// Layout of any member in a HAL-generated struct, adding the type of a struct it holds to `types`. Null for an
// array of length 0, which takes no bytes.
template <typename Struct, typename Struct::Field F>
std::optional<Member> hal_member(
    const Factory& factory, const typename Struct::ConstView& view, std::vector<StructType>& types) {
    using Traits = typename Struct::template FieldTraits<true, F>;
    const std::string_view name = enchantum::to_string(F);
    const auto offset = static_cast<uint32_t>(factory.offset_of<Struct>(F));
    if constexpr (requires { typename Traits::struct_type; }) {
        return Member{name, offset, hal_struct_of<typename Traits::struct_type>(factory, 0, types)};
    } else if constexpr (std::is_reference_v<typename Traits::type>) {
        return hal_scalar_member<Struct, F>(factory);
    } else {
        // An array, whose length the HAL gives only at run time, as the length of the view's span.
        using Element = std::remove_cv_t<typename Traits::element_type>;
        const auto count = static_cast<uint32_t>(view.template get<F>().size());
        if (count == 0) {
            return std::nullopt;
        }
        if constexpr (std::is_integral_v<Element> || std::is_enum_v<Element>) {
            return Member{name, offset, array_of<Element>(count)};
        } else {
            return Member{name, offset, hal_struct_of<Element>(factory, count, types)};
        }
    }
}

// The fields in order, with each gap between them as padding. The gaps are the members the HAL's generator skips
// (CODEGEN:skip).
std::vector<Member> with_padding(std::string_view struct_name, uint32_t size, const std::vector<Member>& fields) {
    const auto pad = [](uint32_t offset, uint32_t bytes) { return Member{"pad", offset, {element::Pad{}, bytes, 0}}; };
    std::vector<Member> members;
    uint32_t end = 0;
    for (const auto& field : fields) {
        TT_FATAL(
            field.offset >= end,
            "Fabric manifest: {}'s {} overlaps the member before it, so {} can't be described member by member",
            struct_name,
            field.name,
            struct_name);
        if (field.offset > end) {
            members.push_back(pad(end, field.offset - end));
        }
        members.push_back(field);
        end = field.offset + field.type.size;
    }
    TT_FATAL(end <= size, "Fabric manifest: {}'s members end past its {} bytes", struct_name, size);
    if (end < size) {
        members.push_back(pad(end, size - end));
    }
    return members;
}

// Adds Struct's type to `types`, and the type of every struct it holds, once each.
template <typename Struct>
void add_hal_struct_type(const Factory& factory, std::vector<StructType>& types) {
    const std::string_view name = enchantum::type_name<Struct>;
    if (std::ranges::any_of(types, [&](const StructType& type) { return type.name == name; })) {
        return;
    }
    const auto size = static_cast<uint32_t>(factory.size_of<Struct>());
    const size_t index = types.size();
    types.push_back({name, size, {}});

    const auto buffer = factory.create<Struct>();
    const typename Struct::ConstView view = buffer.view();
    std::vector<Member> fields;
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        const auto add = [&](std::optional<Member> member) {
            if (member.has_value()) {
                fields.push_back(*member);
            }
        };
        (add(hal_member<Struct, static_cast<typename Struct::Field>(Is)>(factory, view, types)), ...);
    }(std::make_index_sequence<Struct::fields_count>());
    types[index].members = with_padding(name, size, fields);
}

}  // namespace

std::vector<Member> go_msg_layout(const tt::tt_metal::Hal& hal) {
    using tt::tt_metal::dev_msgs::go_msg_t;
    using Field = go_msg_t::Field;
    const auto& factory = hal.get_dev_msgs_factory(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
    // go_msg_t is a union of `all` and four bytes. We list the four bytes individually.
    return {
        hal_scalar_member<go_msg_t, Field::dispatch_message_offset>(factory),
        hal_scalar_member<go_msg_t, Field::master_x>(factory),
        hal_scalar_member<go_msg_t, Field::master_y>(factory),
        hal_scalar_member<go_msg_t, Field::signal>(factory),
    };
}

std::vector<StructType> hal_struct_types(const tt::tt_metal::Hal& hal) {
    namespace dev_msgs = tt::tt_metal::dev_msgs;
    const auto& factory = hal.get_dev_msgs_factory(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
    std::vector<StructType> types{
        {go_msg_name, static_cast<uint32_t>(factory.size_of<dev_msgs::go_msg_t>()), go_msg_layout(hal)}};
    add_hal_struct_type<dev_msgs::launch_msg_t>(factory, types);
    return types;
}

}  // namespace tt::tt_fabric::layout
