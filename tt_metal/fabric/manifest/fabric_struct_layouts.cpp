// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "tt_metal/fabric/manifest/fabric_struct_layouts.hpp"

#include <type_traits>

#include <enchantum/enchantum.hpp>

#include "tt_metal/llrt/hal.hpp"

namespace tt::tt_fabric::layout {

namespace {

// Layout of a scalar member in a HAL-generated struct. The name from the generated Field enum, the offset from the
// factory, and the type from the generated accessor.
template <typename Struct, typename Struct::Field F>
Member hal_scalar_member(const tt::tt_metal::dev_msgs::Factory& factory) {
    using Traits = typename Struct::template FieldTraits<true, F>;
    static_assert(std::is_reference_v<typename Traits::type>, "only scalar fields are supported");
    using Scalar = std::remove_cv_t<typename Traits::element_type>;
    return Member{enchantum::to_string(F), static_cast<uint32_t>(factory.offset_of<Struct>(F)), type_of<Scalar>()};
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

}  // namespace tt::tt_fabric::layout
