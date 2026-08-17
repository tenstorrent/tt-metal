// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>
#include <span>
#include <vector>

#include "tt_metal/llrt/tt_elffile.hpp"
#include "tt_metal/llrt/tt_memory.h"

namespace ll_api {
namespace {

using word_t = memory::word_t;

// Collect (addr, len) for each span, in the order the packing logic emitted them.
std::vector<std::pair<std::uint64_t, std::uint32_t>> span_order(const memory& m) {
    std::vector<std::pair<std::uint64_t, std::uint32_t>> spans;
    m.process_spans([&](std::vector<std::uint32_t>::const_iterator, std::uint64_t addr, std::uint32_t len) {
        spans.emplace_back(addr, len);
    });
    return spans;
}

// DISCRETE loading must emit spans in ascending address order even when the address sort is a
// non-identity permutation. An ncrisc-style binary places its data segment at a LOWER address than
// text, which produces exactly such a permutation, so it exercises that ordering.
TEST(MemorySegmentOrdering, DiscreteEmitsSpansInAscendingAddressOrder) {
    // segments[0] is text (the loader treats the first segment as text), placed high in memory;
    // the data segment is placed lower -- the ncrisc layout that triggers a non-identity sort.
    const std::vector<word_t> text_contents = {0x1111'1111, 0x2222'2222};  // addr 0x2000, 2 words
    const std::vector<word_t> data_contents = {0x3333'3333};               // addr 0x1000, 1 word

    std::vector<ElfFile::Segment> segments;
    segments.emplace_back(std::span<const word_t>(text_contents), /*addr=*/0x2000, /*lma=*/0x2000, /*membytes=*/8);
    segments.emplace_back(std::span<const word_t>(data_contents), /*addr=*/0x1000, /*lma=*/0x1000, /*membytes=*/4);

    const memory m = memory::from_segments(segments, memory::Loading::DISCRETE);

    // Ascending address order: data (0x1000) first, then text (0x2000).
    // Source/segment order would place text (0x2000) first, so this verifies the address sort is applied.
    const std::vector<std::pair<std::uint64_t, std::uint32_t>> expected = {{0x1000, 1}, {0x2000, 2}};
    EXPECT_EQ(span_order(m), expected);

    // data_ is packed in span order too, so a wrong span order corrupts the packed image.
    const std::vector<word_t> expected_data = {0x3333'3333, 0x1111'1111, 0x2222'2222};
    EXPECT_EQ(m.data(), expected_data);

    // text_addr_/text_size_ always come from segments[0] regardless of ordering.
    EXPECT_EQ(m.get_text_addr(), 0x2000u);
    EXPECT_EQ(m.get_text_size(), 8u);
}

// The identity-permutation paths (non-DISCRETE) coalesce contiguous segments into a single span.
// This guards that non-DISCRETE behaviour: contiguous segments must still collapse to one span.
TEST(MemorySegmentOrdering, ContiguousCoalescesIntoSingleSpan) {
    const std::vector<word_t> text_contents = {0x1111'1111, 0x2222'2222};  // lma 0x1000, 2 words
    const std::vector<word_t> data_contents = {0x3333'3333};               // lma 0x1008, 1 word

    std::vector<ElfFile::Segment> segments;
    segments.emplace_back(std::span<const word_t>(text_contents), /*addr=*/0x1000, /*lma=*/0x1000, /*membytes=*/8);
    segments.emplace_back(std::span<const word_t>(data_contents), /*addr=*/0x1008, /*lma=*/0x1008, /*membytes=*/4);

    const memory m = memory::from_segments(segments, memory::Loading::CONTIGUOUS);

    const std::vector<std::pair<std::uint64_t, std::uint32_t>> expected = {{0x1000, 3}};
    EXPECT_EQ(span_order(m), expected);

    const std::vector<word_t> expected_data = {0x1111'1111, 0x2222'2222, 0x3333'3333};
    EXPECT_EQ(m.data(), expected_data);
}

// Regression test for a heap buffer underflow in pack_from_segments(). Under DISCRETE loading a
// span is only started for segments that have file contents, so if the *first* segment in address
// order has none (a .bss-only PT_LOAD has p_filesz == 0, so zero content words) link_spans_ is
// still empty when the accumulate step runs, and link_spans_.back() writes 8 bytes before the
// start of the buffer. Such a segment carries no data, so it must simply contribute no span.
TEST(MemorySegmentOrdering, DiscreteHandlesEmptyFirstSegmentInAddressOrder) {
    // segments[0] is text (the loader requires the first segment to be text), placed high in
    // memory; a .bss-only segment sits lower, so the address sort puts the empty one first.
    const std::vector<word_t> text_contents = {0x1111'1111, 0x2222'2222};  // addr 0x2000, 2 words

    std::vector<ElfFile::Segment> segments;
    segments.emplace_back(std::span<const word_t>(text_contents), /*addr=*/0x2000, /*lma=*/0x2000, /*membytes=*/8);
    // No file contents, but a non-zero memory image -- exactly a .bss-only PT_LOAD.
    segments.emplace_back(std::span<const word_t>(), /*addr=*/0x1000, /*lma=*/0x1000, /*membytes=*/0x40);

    const memory m = memory::from_segments(segments, memory::Loading::DISCRETE);

    // The contentless segment contributes no span and no data; only text is emitted.
    const std::vector<std::pair<std::uint64_t, std::uint32_t>> expected = {{0x2000, 2}};
    EXPECT_EQ(span_order(m), expected);
    EXPECT_EQ(m.num_spans(), 1u);
    EXPECT_EQ(m.data(), text_contents);
    EXPECT_EQ(m.get_text_addr(), 0x2000u);
    EXPECT_EQ(m.get_text_size(), 8u);
}

// A contentless segment that does not sort first must also be dropped -- this already worked, and
// guards that the underflow fix did not change the behaviour of the previously-reachable case.
TEST(MemorySegmentOrdering, DiscreteDropsEmptyTrailingSegment) {
    const std::vector<word_t> text_contents = {0x1111'1111, 0x2222'2222};  // addr 0x1000, 2 words

    std::vector<ElfFile::Segment> segments;
    segments.emplace_back(std::span<const word_t>(text_contents), /*addr=*/0x1000, /*lma=*/0x1000, /*membytes=*/8);
    segments.emplace_back(std::span<const word_t>(), /*addr=*/0x2000, /*lma=*/0x2000, /*membytes=*/0x40);

    const memory m = memory::from_segments(segments, memory::Loading::DISCRETE);

    const std::vector<std::pair<std::uint64_t, std::uint32_t>> expected = {{0x1000, 2}};
    EXPECT_EQ(span_order(m), expected);
    EXPECT_EQ(m.data(), text_contents);
}

}  // namespace
}  // namespace ll_api
