// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Host-only wire-format regression: large NoC requests must not become8KB.
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include "tools/profiler/event_metadata.hpp"

using Metadata = KernelProfilerNocEventMetadata;

int main() {
    static_assert(sizeof(Metadata) == 8);
    for (bool posted : {false, true}) {
        for (uint32_t bytes : std::array<uint32_t, 13>{
                 0, 1, 32, 576, 1088, 8160, 8192, 16128, 17408, 69632, 1048544, 1048576,
                 std::numeric_limits<uint32_t>::max()}) {
            Metadata source;
            auto& event = source.data.local_event;
            event.noc_xfer_type = Metadata::NocEventType::READ;
            event.dst_x = 8;
            event.dst_y = 3;
            event.setAttributes(bytes, posted);
            Metadata decoded(source.asU64());
            const auto result = std::get<Metadata::LocalNocEvent>(decoded.getContents());
            const uint32_t expected = std::min<uint64_t>((uint64_t(bytes) + 31) / 32, 0x7fff) * 32;
            if (result.getNumBytes() != expected || result.posted != posted || result.dst_x != 8 || result.dst_y != 3) {
                std::fprintf(stderr, "NoC metadata: requested=%u expected=%u decoded=%u posted=%d\n",
                             bytes, expected, result.getNumBytes(), posted);
                return 1;
            }
        }
    }
    // Previously emitted events have zero in the seven reserved high bits.
    // The low payload byte and posted flag retain their original wire positions.
    for (uint32_t chunks : {0u, 1u, 255u}) {
        for (uint64_t posted : {0ull, 1ull}) {
            const uint64_t old_wire = uint64_t(Metadata::NocEventType::READ) | (uint64_t(chunks) << 48) | (posted << 56);
            Metadata decoded(old_wire);
            const auto result = std::get<Metadata::LocalNocEvent>(decoded.getContents());
            if (result.getNumBytes() != chunks * 32 || result.posted != posted) {
                std::fprintf(stderr, "Legacy NoC metadata decode changed\n");
                return 1;
            }
        }
    }
    std::puts("NoC metadata payload boundaries and legacy wire compatibility passed (host only)");
}
