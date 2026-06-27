// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <core_coord.hpp>

#include "flatbuffer/program_types_to_flatbuffer.hpp"
#include "flatbuffer/program_types_from_flatbuffer.hpp"

namespace tt::tt_metal {
namespace {

// Regression test for from_flatbuffer(std::vector<CoreCoord>): the deserializer
// size-constructed the output vector (N default-initialized {0,0} entries) and then
// emplace_back'd the real values on top, returning a vector of size 2N with junk at the
// front. Before the fix this test fails (size is 2N, not N).
TEST(ProgramTypesFromFlatbuffer, CoreCoordVectorRoundtrip) {
    const std::vector<CoreCoord> original = {CoreCoord{1, 2}, CoreCoord{3, 4}, CoreCoord{5, 6}};

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(to_flatbuffer(builder, original));

    const auto* fb = flatbuffers::GetRoot<flatbuffers::Vector<flatbuffers::Offset<flatbuffer::CoreCoord>>>(
        builder.GetBufferPointer());
    const std::vector<CoreCoord> roundtripped = from_flatbuffer(fb);

    ASSERT_EQ(roundtripped.size(), original.size());
    for (size_t i = 0; i < original.size(); ++i) {
        EXPECT_EQ(roundtripped[i].x, original[i].x);
        EXPECT_EQ(roundtripped[i].y, original[i].y);
    }
}

TEST(ProgramTypesFromFlatbuffer, CoreCoordVectorEmptyRoundtrip) {
    const std::vector<CoreCoord> original = {};

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(to_flatbuffer(builder, original));

    const auto* fb = flatbuffers::GetRoot<flatbuffers::Vector<flatbuffers::Offset<flatbuffer::CoreCoord>>>(
        builder.GetBufferPointer());
    EXPECT_TRUE(from_flatbuffer(fb).empty());
}

// Regression test for from_flatbuffer(std::vector<std::vector<uint32_t>>): same bug class as
// above - the output was size-constructed (N empty sub-vectors) and then push_back'd onto,
// returning size 2N with empty sub-vectors at the front. Before the fix this test fails.
TEST(ProgramTypesFromFlatbuffer, UInt32VecOfVecRoundtrip) {
    const std::vector<std::vector<uint32_t>> original = {{1, 2, 3}, {4, 5}, {6}};

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(to_flatbuffer(builder, original));

    const auto* fb = flatbuffers::GetRoot<flatbuffers::Vector<flatbuffers::Offset<flatbuffer::UInt32Vector>>>(
        builder.GetBufferPointer());
    const std::vector<std::vector<uint32_t>> roundtripped = from_flatbuffer(fb);

    ASSERT_EQ(roundtripped.size(), original.size());
    EXPECT_EQ(roundtripped, original);
}

TEST(ProgramTypesFromFlatbuffer, UInt32VecOfVecEmptyRoundtrip) {
    const std::vector<std::vector<uint32_t>> original = {};

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(to_flatbuffer(builder, original));

    const auto* fb = flatbuffers::GetRoot<flatbuffers::Vector<flatbuffers::Offset<flatbuffer::UInt32Vector>>>(
        builder.GetBufferPointer());
    EXPECT_TRUE(from_flatbuffer(fb).empty());
}

}  // namespace
}  // namespace tt::tt_metal
