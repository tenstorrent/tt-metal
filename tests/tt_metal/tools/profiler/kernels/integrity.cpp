// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The integrity sweep's kernel, on every RISC: each catalog struct of group GROUP, sent every way it can be, its
// fields filled from its call's seed.
#include <cstdint>
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#endif
#include "api/debug/device_publish.h"
#include "../records_types.hpp"

// The processor index each record's seed holds: BRISC, NCRISC, then the unpack, math and pack TRISCs.
#if defined(COMPILE_FOR_BRISC)
constexpr uint32_t kProcessor = 0;
#elif defined(COMPILE_FOR_NCRISC)
constexpr uint32_t kProcessor = 1;
#else
constexpr uint32_t kProcessor = 2 + COMPILE_FOR_TRISC;
#endif

namespace semantics_test {
using tt::tt_metal::experimental::streaming_profiler::kNoInline;

// A struct alone, through the out-of-line writer, after another argument, and two of it as a list.
template <typename S>
FORCE_INLINE void send(const S& s, const S& t, uint64_t seed) {
    DevicePublish<Send{"alone"_s}>(s);
    DevicePublish<Send{"no inline"_s}, kNoInline>(s);
    DevicePublish<Send{"pair"_s}>(uint8_t(seed), s);
    DevicePublish<Send{"list"_s}>(s, t);
}
template <typename S>
FORCE_INLINE void send_filled(uint32_t index, uint32_t processor, uint32_t i) {
    const uint64_t seed = seed_of(index, processor, i);
    send(filled<S>(seed), filled<S>(mix(seed)), seed);
}

// Too large for the stack, so filled in place, word k as filled() fills field k.
Run runs[2];
Largest largest;
StatRows stat_rows[2]{};
template <typename S>
void fill_words(S& s, uint64_t seed) {
    for (uint32_t k = 0; k < sizeof(s.words) / sizeof(s.words[0]); k++) {
        s.words[k] = uint32_t(mix(seed + k));
    }
}

uint64_t ticks_of(Timestamp t) { return __builtin_bit_cast(uint64_t, t); }
Timestamp at_ticks(uint64_t ticks) { return __builtin_bit_cast(Timestamp, ticks); }
Stamps stamps(uint64_t seed, uint32_t i) {
    const uint64_t now = ticks_of(Timestamp::now());
    return {at_ticks(now), uint8_t(seed), at_ticks(now - stamp_delta(i, 1)), at_ticks(now - stamp_delta(i, 2))};
}
StampedRun stamped_run(uint64_t seed, uint32_t i) {
    StampedRun run{at_ticks(ticks_of(Timestamp::now()) - stamp_delta(i, 1)), {}};
    for (uint32_t k = 0; k < 12; k++) {
        run.words[k] = uint32_t(mix(seed + 1 + k));
    }
    return run;
}
void fill_stat_rows(StatRows& rows, uint64_t seed, uint32_t i) {
    rows.at = at_ticks(ticks_of(Timestamp::now()) - stamp_delta(i, 1));
    for (uint32_t r = 0; r < 20; r++) {
        rows.rows[r] = {
            uint32_t(mix(seed + 1 + 3 * r)), uint32_t(mix(seed + 2 + 3 * r)), uint16_t(mix(seed + 3 + 3 * r))};
    }
}

// A program holds at most 70 KiB of kernels, so the catalog goes in groups; COMPUTE_GROUP is the set the compute
// kernels send.
void publish_group(uint32_t i) {
#if GROUP == 0
    send_filled<AllScalars>(0, kProcessor, i);
    send_filled<Descending>(1, kProcessor, i);
    send_filled<Ascending>(2, kProcessor, i);
    send_filled<Bools>(3, kProcessor, i);
#elif GROUP == 1
    send_filled<WordBits>(4, kProcessor, i);
    send_filled<CrossingBits>(5, kProcessor, i);
    send_filled<SignedBits>(6, kProcessor, i);
    send_filled<Packed>(7, kProcessor, i);
#elif GROUP == 2
    send_filled<AlignedMember>(8, kProcessor, i);
    send_filled<Nested>(9, kProcessor, i);
    send_filled<Derived>(10, kProcessor, i);
    send_filled<Tagged>(11, kProcessor, i);
#elif GROUP == 3
    send_filled<Pairs>(12, kProcessor, i);
    send_filled<WithEnum>(13, kProcessor, i);
    send_filled<Floats>(14, kProcessor, i);
    send_filled<LateField>(19, kProcessor, i);
#elif GROUP == 4
    const uint64_t run_seed = seed_of(15, kProcessor, i);
    fill_words(runs[0], run_seed);
    fill_words(runs[1], mix(run_seed));
    send(runs[0], runs[1], run_seed);
    // A record holds one Largest at most.
    fill_words(largest, seed_of(16, kProcessor, i));
    DevicePublish<Send{"alone"_s}>(largest);
    DevicePublish<Send{"no inline"_s}, kNoInline>(largest);
#elif GROUP == 6
    const uint64_t rows_seed = seed_of(20, kProcessor, i);
    fill_stat_rows(stat_rows[0], rows_seed, i);
    fill_stat_rows(stat_rows[1], mix(rows_seed), i);
    send(stat_rows[0], stat_rows[1], rows_seed);
#elif GROUP == 5 || GROUP == COMPUTE_GROUP
#if GROUP == COMPUTE_GROUP
    send_filled<AllScalars>(0, kProcessor, i);
    send_filled<SignedBits>(6, kProcessor, i);
#endif
    const uint64_t stamps_seed = seed_of(17, kProcessor, i);
    send(stamps(stamps_seed, i), stamps(mix(stamps_seed), i), stamps_seed);
#if GROUP == 5
    const uint64_t stamped_seed = seed_of(18, kProcessor, i);
    send(stamped_run(stamped_seed, i), stamped_run(mix(stamped_seed), i), stamped_seed);
#endif
#endif
}

}  // namespace semantics_test

void kernel_main() {
    for (uint32_t i = 0; i < N_ITERS; i++) {
        semantics_test::publish_group(i);
    }
}
