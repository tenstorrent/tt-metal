// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// One DevicePublish call that breaks one rule for each RECORD_ERROR value. The records test builds each one and checks
// that the build, or the load, fails with that rule's message.
#include <array>
#include <cstdint>
#include "api/debug/device_publish.h"
#include "../records_types.hpp"

using namespace records_test;

struct WithConstructor {
    WithConstructor() = default;
    WithConstructor(uint32_t v) : v(v) {}
    uint32_t v = 0;
};
struct HoldsWithConstructor {
    WithConstructor w;
};
template <int N>
struct Templated {
    uint32_t v;
};
struct ConstMember {
    const uint32_t v;
};
enum Unfixed { kUnfixed };
struct UnfixedField {
    Unfixed v;
};
struct PointerField {
    uint32_t* p;
};
struct CopiedByHand {
    CopiedByHand() = default;
    CopiedByHand(const CopiedByHand&) {}
};
struct HoldsCopiedByHand {
    CopiedByHand c;
};
union Word {
    uint32_t raw;
    float f;
};
struct HoldsWord {
    Word w;
};
// The two data-movement RISCs build this kernel with different structs under one name.
#if defined(COMPILE_FOR_BRISC)
struct Clash {
    uint32_t v;
};
#else
struct Clash {
    uint16_t v;
};
#endif

void kernel_main() {
#if RECORD_ERROR == 1
    DevicePublish<Name{"error"_s}>("text");
#elif RECORD_ERROR == 2
    DevicePublish<Name{"text"}>();
#elif RECORD_ERROR == 3
    DevicePublish<Name{"error"_s}>(Name{"text"_s});
#elif RECORD_ERROR == 4
    DevicePublish<Name{"error"_s}>(WithConstructor{1});
#elif RECORD_ERROR == 5
    DevicePublish<Name{"error"_s}>(Templated<1>{1});
#elif RECORD_ERROR == 6
    struct Local {
        uint32_t v;
    };
    DevicePublish<Name{"error"_s}>(Local{1});
#elif RECORD_ERROR == 7
    DevicePublish<Name{"error"_s}>(ConstMember{1});
#elif RECORD_ERROR == 8
    DevicePublish<Name{"error"_s}>(UnfixedField{kUnfixed});
#elif RECORD_ERROR == 9
    DevicePublish<Name{"error"_s}>(PointerField{nullptr});
#elif RECORD_ERROR == 10
    DevicePublish<Name{"error"_s}>(std::array<std::array<uint32_t, 2>, 2>{});
#elif RECORD_ERROR == 11
    DevicePublish<Name{"error"_s}>(std::array<uint32_t, 0>{});
#elif RECORD_ERROR == 12
    enum class LocalEnum : uint8_t { kA };
    DevicePublish<Name{"error"_s}>(LocalEnum::kA);
#elif RECORD_ERROR == 13
    DevicePublish<Name{"error"_s}>(std::array<uint32_t, 502>{});  // one word over the limit
#elif RECORD_ERROR == 14
    DevicePublish<Name{"error"_s}, 5>();
#elif RECORD_ERROR == 15
    DevicePublish<Name{"error"_s}>(HoldsCopiedByHand{});
#elif RECORD_ERROR == 16
    uint32_t v = 0;
    DevicePublish<Name{"error"_s}>(&v, 1u);
#elif RECORD_ERROR == 17
    DevicePublish<Name{"error"_s}>(Word{1});
#elif RECORD_ERROR == 18
    DevicePublish<Name{"error"_s}>(HoldsWord{});
#elif RECORD_ERROR == 19
    DevicePublish<Name{"error"_s}>(HoldsWithConstructor{});
#elif RECORD_ERROR == 20
    DevicePublish<Name{"error"_s}>(Clash{1});
#elif RECORD_ERROR == 21
    DevicePublish<Name{"error"_s}>(LongField{1});
#elif RECORD_ERROR == 22
    DevicePublish<Name{"error"_s}>("text"_s);
#elif RECORD_ERROR == 23
    DevicePublish<tt::tt_metal::experimental::streaming_profiler::kNoInline, Name{"error"_s}>();
#elif RECORD_ERROR == 24
    DevicePublish<5u>();
#elif RECORD_ERROR == 25
    DevicePublish<
        Name{"error"_s},
        tt::tt_metal::experimental::device_observability::kPublishBatched,
        tt::tt_metal::experimental::device_observability::kSendToExternalProcess>(1u);
#endif
}
