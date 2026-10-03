// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <tuple>

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "tt_metal/fabric/manifest/struct_layout.hpp"

namespace tt::tt_fabric {

template <>
struct StructLayout<SenderChannelProducerCursor> {
    static constexpr std::array fields = {
        LAYOUT_FIELD(SenderChannelProducerCursor, write_counter),
        LAYOUT_FIELD(SenderChannelProducerCursor, write_index),
        LAYOUT_PAD(SenderChannelProducerCursor, align_pad_0),
        LAYOUT_PAD(SenderChannelProducerCursor, align_pad_1),
    };
};
static_assert(validate_struct_fields<SenderChannelProducerCursor>(StructLayout<SenderChannelProducerCursor>::fields));

// Every struct described at compile time. The manifest writes one type entry per element, which describes the
// struct's fields.
using DescribedStructs = std::tuple<SenderChannelProducerCursor>;

}  // namespace tt::tt_fabric
