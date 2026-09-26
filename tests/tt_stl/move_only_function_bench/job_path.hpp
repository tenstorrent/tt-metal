// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "candidates.hpp"

#include <cstddef>
#include <cstdint>

// The ring-buffer job path, declared here and defined in job_path.cpp.
//
// The split mirrors tt::tt_metal::ThreadPool and is what makes the measurement honest. The job is
// erased at the call site, in the benchmark's TU, exactly as an enqueue site does it. The slot
// write, the move back out and the invocation happen over in job_path.cpp, which never sees the
// callable's concrete type -- only Fn. That is the position ThreadPool::push and its worker thread
// are in, and it is why the capture type does not appear in this signature.
//
// Getting this wrong is easy and quiet. An earlier version ran the whole loop, lambda construction
// included, inside the other TU; gcc still saw the concrete type there, devirtualised, and reported
// std::function about 2.6x faster than the candidates while clang had all three within 4%.

namespace bench {

// Captures sized around the inline buffer. Shared so both translation units agree on the types.
struct SmallCapture {
    std::uint64_t a = 1;
    std::uint64_t b = 2;
    std::uint64_t value() const { return a; }
};

// Three 64-bit words. On LP64 that is 24 bytes: inline under libc++ (3-pointer buffer), heap under
// libstdc++ (2-pointer). Probes the band where the two standard libraries disagree.
struct BoundaryCapture {
    std::uint64_t a = 1;
    std::uint64_t b = 2;
    std::uint64_t c = 3;
    std::uint64_t value() const { return a; }
};

// Same 16 bytes as SmallCapture, but a user-defined move constructor makes it non-trivially
// relocatable. libstdc++'s std::function only stores a capture inline when it can relocate it with
// a byte copy, so this one goes to the heap for std::function while both candidates keep it inline.
// Without this case the comparison silently favours std::function: every other capture here is a
// POD, which is exactly its best case, and that alone accounts for it measuring ~2x faster on gcc.
struct NonTrivialCapture {
    std::uint64_t a = 1;
    std::uint64_t b = 2;
    NonTrivialCapture() = default;
    NonTrivialCapture(const NonTrivialCapture& o) noexcept : a(o.a), b(o.b) {}
    NonTrivialCapture(NonTrivialCapture&& o) noexcept : a(o.a), b(o.b) {}
    NonTrivialCapture& operator=(const NonTrivialCapture&) = default;
    NonTrivialCapture& operator=(NonTrivialCapture&&) = default;
    std::uint64_t value() const { return a; }
};

struct LargeCapture {
    std::uint64_t data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::uint64_t value() const { return data[0]; }
};

// One job: move-assign it into the slot (push), move it back out (pop), invoke.
template <typename Fn>
void push_pop_invoke(Fn* slot, Fn&& job);

}  // namespace bench
