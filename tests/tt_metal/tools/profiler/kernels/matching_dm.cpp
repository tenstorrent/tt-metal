// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// One call per argument shape the matching table in test_streaming_profiler_semantics.cpp covers, each named by its
// metadata, N_ITERS times.
#include <array>
#include <cstdint>
#include "api/debug/device_publish.h"
#include "../records_types.hpp"

using namespace semantics_test;

void kernel_main() {
    for (uint32_t i = 0; i < N_ITERS; i++) {
        DevicePublish<Call{"none"_s}>();
        DevicePublish<Call{"u32"_s}>(i);
        DevicePublish<Call{"i32"_s}>(-int32_t(i));
        DevicePublish<Call{"u16"_s}>(uint16_t(i));
        DevicePublish<Call{"u64"_s}>(uint64_t(i));
        DevicePublish<Call{"f32"_s}>(float(i));
        DevicePublish<Call{"f64"_s}>(double(i));
        DevicePublish<Call{"bool"_s}>(i % 2 == 0);
        DevicePublish<Call{"u8"_s}>(uint8_t(i));
        DevicePublish<Call{"i8"_s}>(int8_t(i));
        DevicePublish<Call{"char"_s}>(char('a' + i));
        DevicePublish<Call{"Color"_s}>(Color::Blue);
        DevicePublish<Call{"u32 u32"_s}>(i, i);
        DevicePublish<Call{"u32 f32"_s}>(i, float(i));
        DevicePublish<Call{"f32 u32"_s}>(float(i), i);
        DevicePublish<Call{"u32 u32 u32"_s}>(i, i, i);
        const uint32_t three[3] = {i, i, i};
        DevicePublish<Call{"u32[3]"_s}>(three);
        DevicePublish<Call{"u32[4]"_s}>(std::array<uint32_t, 4>{i, i, i, i});
        DevicePublish<Call{"f32[3]"_s}>(std::array<float, 3>{});
        DevicePublish<Call{"PairA"_s}>(PairA{i, i});
        DevicePublish<Call{"PairB"_s}>(PairB{i, i});
        DevicePublish<Call{"PairA u32"_s}>(PairA{i, i}, i);
        DevicePublish<Call{"PairA u32[2]"_s}>(PairA{i, i}, std::array<uint32_t, 2>{i, i});
        DevicePublish<Call{"PairA[2]"_s}>(std::array<PairA, 2>{});
        DevicePublish<Call{"Timestamp"_s}>(Timestamp::now());
        DevicePublish<Other{"u32"_s}>(i);
        DevicePublish<>(i);
    }
}
