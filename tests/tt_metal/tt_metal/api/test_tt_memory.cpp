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

// Friend of ll_api::memory (declared in tt_memory.h). Lets the tests inject synthetic ELF
// segments and drive the address-ordering logic without an on-disk ELF file.
class MemorySegmentOrderingTest : public ::testing::Test {
protected:
    using word_t = memory::word_t;

    // Build a memory from synthetic segments, running the same packing logic the
    // (path, loading) constructor uses.
    static memory build(const std::vector<ElfFile::Segment>& segments, memory::Loading loading) {
        memory m;
        m.loading_ = loading;
        m.pack_from_segments("<test>", segments);
        return m;
    }

    // Collect (addr, len) for each span, in the order pack_from_segments emitted them.
    static std::vector<std::pair<std::uint64_t, std::uint32_t>> span_order(const memory& m) {
        std::vector<std::pair<std::uint64_t, std::uint32_t>> spans;
        m.process_spans([&](std::vector<std::uint32_t>::const_iterator, std::uint64_t addr, std::uint32_t len) {
            spans.emplace_back(addr, len);
        });
        return spans;
    }
};

// Regression test for the DISCRETE double-indexing bug (issue #48262). An ncrisc-style binary
// places its data segment at a LOWER address than text, so the address sort produces a
// non-identity permutation. The spans must come out in ascending address order.
TEST_F(MemorySegmentOrderingTest, DiscreteEmitsSpansInAscendingAddressOrder) {
    // segments[0] is text (the loader treats the first segment as text), placed high in memory;
    // the data segment is placed lower -- the ncrisc layout that triggers a non-identity sort.
    const std::vector<word_t> text_contents = {0x1111'1111, 0x2222'2222};  // addr 0x2000, 2 words
    const std::vector<word_t> data_contents = {0x3333'3333};               // addr 0x1000, 1 word

    std::vector<ElfFile::Segment> segments;
    segments.emplace_back(std::span<const word_t>(text_contents), /*addr=*/0x2000, /*lma=*/0x2000, /*membytes=*/8);
    segments.emplace_back(std::span<const word_t>(data_contents), /*addr=*/0x1000, /*lma=*/0x1000, /*membytes=*/4);

    const memory m = build(segments, memory::Loading::DISCRETE);

    // Ascending address order: data (0x1000) first, then text (0x2000).
    // With the bug the order was reversed (text 0x2000 first), so this is the fail-without-fix check.
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
// This guards the refactor: the fix must not change non-DISCRETE behaviour.
TEST_F(MemorySegmentOrderingTest, ContiguousCoalescesIntoSingleSpan) {
    const std::vector<word_t> text_contents = {0x1111'1111, 0x2222'2222};  // lma 0x1000, 2 words
    const std::vector<word_t> data_contents = {0x3333'3333};               // lma 0x1008, 1 word

    std::vector<ElfFile::Segment> segments;
    segments.emplace_back(std::span<const word_t>(text_contents), /*addr=*/0x1000, /*lma=*/0x1000, /*membytes=*/8);
    segments.emplace_back(std::span<const word_t>(data_contents), /*addr=*/0x1008, /*lma=*/0x1008, /*membytes=*/4);

    const memory m = build(segments, memory::Loading::CONTIGUOUS);

    const std::vector<std::pair<std::uint64_t, std::uint32_t>> expected = {{0x1000, 3}};
    EXPECT_EQ(span_order(m), expected);

    const std::vector<word_t> expected_data = {0x1111'1111, 0x2222'2222, 0x3333'3333};
    EXPECT_EQ(m.data(), expected_data);
}

}  // namespace ll_api
