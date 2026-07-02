// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <tt_stl/span.hpp>
#include <sub_device_types.hpp>

#include "flatbuffer/program_types_to_flatbuffer.hpp"
#include "flatbuffer/program_types_from_flatbuffer.hpp"

namespace tt::tt_metal {
namespace {

// Regression test for the from_flatbuffer(SubDeviceId vector) loop bug (PR #48181).
//
// The original loop iterated over the destination vector's size:
//     sub_device_ids.reserve(fb_sub_device_ids->size());
//     for (size_t i = 0; i < sub_device_ids.size(); ++i) { ... }   // size() is still 0!
// reserve() only changes capacity, not size(), so the loop body never executed and
// deserialization always returned an empty vector regardless of the input. The fix
// iterates over fb_sub_device_ids->size() instead.
//
// Before the fix this test fails at the ASSERT_EQ below (0 != 3).
TEST(ProgramTypesFromFlatbuffer, SubDeviceIdVectorRoundtrip) {
    const std::vector<SubDeviceId> original = {SubDeviceId{0}, SubDeviceId{2}, SubDeviceId{5}};

    flatbuffers::FlatBufferBuilder builder;
    const auto fb_offset = to_flatbuffer(builder, tt::stl::Span<const SubDeviceId>(original.data(), original.size()));
    builder.Finish(fb_offset);

    const auto* fb_sub_device_ids = flatbuffers::GetRoot<flatbuffers::Vector<uint8_t>>(builder.GetBufferPointer());
    const std::vector<SubDeviceId> roundtripped = from_flatbuffer(fb_sub_device_ids);

    ASSERT_EQ(roundtripped.size(), original.size());
    EXPECT_EQ(roundtripped, original);
}

// A null input is the documented "no sub-device ids" case and must yield an empty vector.
TEST(ProgramTypesFromFlatbuffer, SubDeviceIdVectorNullIsEmpty) {
    const std::vector<SubDeviceId> roundtripped =
        from_flatbuffer(static_cast<const flatbuffers::Vector<uint8_t>*>(nullptr));
    EXPECT_TRUE(roundtripped.empty());
}

// An empty (but non-null) flatbuffer vector must also round-trip to an empty vector.
TEST(ProgramTypesFromFlatbuffer, SubDeviceIdVectorEmptyRoundtrip) {
    const std::vector<SubDeviceId> original = {};

    flatbuffers::FlatBufferBuilder builder;
    const auto fb_offset = to_flatbuffer(builder, tt::stl::Span<const SubDeviceId>(original.data(), original.size()));
    builder.Finish(fb_offset);

    const auto* fb_sub_device_ids = flatbuffers::GetRoot<flatbuffers::Vector<uint8_t>>(builder.GetBufferPointer());
    const std::vector<SubDeviceId> roundtripped = from_flatbuffer(fb_sub_device_ids);

    EXPECT_TRUE(roundtripped.empty());
}

}  // namespace
}  // namespace tt::tt_metal
