// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-only: the consumer's FIFO-side contract -- widening the device's 32-bit progress word, deciding which frames it
// cannot have reached, and walking frames out of a ring a "device" keeps overwriting.

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <thread>
#include <vector>

#include "impl/streaming_profiler/streaming_profiler_service.hpp"

using namespace tt::tt_metal::streaming_profiler;
namespace kp = kernel_profiler;

constexpr uint64_t kPage = kp::SPSC_SPAN_PAGE_WORDS * 4;
constexpr uint64_t kW = kp::SPSC_NOTIFY_CAP_BYTES;
constexpr uint64_t kFifo = uint64_t{1} << 20;

void check(bool v, const char* why) {
    if (!v) {
        std::fprintf(stderr, "FAIL: %s\n", why);
        std::exit(1);
    }
}

// A FIFO whose frame k occupies stream bytes [k*fp*kPage, (k+1)*fp*kPage): a valid header, then the word k throughout.
struct Ring {
    uint64_t fp;
    std::vector<std::byte> mem;
    std::vector<std::atomic<uint64_t>> marks;
    uint64_t mark_block = UINT64_MAX;
    Ring(uint64_t pages, uint64_t frame_pages) : fp(frame_pages), mem(pages * kPage), marks(pages * kPage / kMarkBytes) {
        for (auto& m : marks) {
            m.store(UINT64_MAX);
        }
    }
    uint64_t bytes() const { return fp * kPage; }
    std::span<const std::atomic<uint64_t>> mark_span() const { return {marks.data(), marks.size()}; }
    // The writer doubles as the ingest: it records the first frame boundary of each block as it goes.
    void write(uint64_t k) {
        const uint64_t abs = k * bytes();
        if (abs / kMarkBytes != mark_block) {
            mark_block = abs / kMarkBytes;
            marks[mark_block % marks.size()].store(abs, std::memory_order_release);
        }
        const uint64_t at = abs % mem.size();
        const uint32_t v = static_cast<uint32_t>(k);
        const uint32_t hdr[2] = {kp::spsc_span_w0(), static_cast<uint32_t>(fp * kp::SPSC_SPAN_PAGE_WORDS - kp::SPSC_SPAN_PREFIX_WORDS)};
        for (uint64_t i = 0; i < bytes(); i += 4) {
            const uint32_t w = i < 8 ? hdr[i / 4] : v;
            std::memcpy(mem.data() + (at + i) % mem.size(), &w, 4);
        }
    }
};

// The frame at `frame` (already copied out) is frame k: right length, header, and the word k after it.
bool holds(const std::byte* frame, uint32_t fw, const Ring& r, uint64_t k) {
    if (fw * 4 != r.bytes()) {
        return false;
    }
    for (size_t i = 8; i < r.bytes(); i += 4) {
        uint32_t w;
        std::memcpy(&w, frame + i, 4);
        if (w != static_cast<uint32_t>(k)) {
            return false;
        }
    }
    return true;
}

int main() {
    check(widen_head(0, 100) == 100, "widen: plain");
    check(widen_head((uint64_t{1} << 32) + 50, 60) == (uint64_t{1} << 32) + 60, "widen: high half kept");
    const uint64_t near = (uint64_t{1} << 32) - 100;
    check(widen_head(near, 50) == (uint64_t{1} << 32) + 50, "widen: across the wrap");
    check(widen_head(near, static_cast<uint32_t>(near)) == near, "widen: no advance");

    check(frame_intact(0, kFifo, kFifo - kW), "intact: at the cap");
    check(!frame_intact(0, kFifo, kFifo - kW + 1), "intact: one byte past the cap");
    check(frame_intact(3 * kFifo, kFifo, 4 * kFifo - kW), "intact: later lap, at the cap");
    check(!frame_intact(3 * kFifo, kFifo, 4 * kFifo - kW + 1), "intact: later lap, past the cap");

    std::vector<std::byte> out(64 * 8 * kPage);
    uint32_t fw[64];
    {
        // Frame 349525 straddles the end of a 16384-page ring (349525 * 3 = 16383 mod 16384).
        Ring r(kFifo / kPage, 3);
        const uint64_t k = 349525;
        for (uint64_t i = k - 2; i <= k + 1; i++) {
            r.write(i);
        }
        const uint64_t head = (k + 2) * r.bytes();
        const Walked w = walk_frames(r.mem, (k - 2) * r.bytes(), (k + 2) * r.bytes(), r.mark_span(), out, fw, [&] { return head; });
        check(w.frames == 4 && w.dropped == 0 && w.cursor == (k + 2) * r.bytes(), "wrap: four frames, none dropped");
        for (uint64_t i = 0; i < 4; i++) {
            check(holds(out.data() + i * r.bytes(), fw[i], r, k - 2 + i), "wrap: contiguous copy");
        }
    }
    {
        // A cursor whose frame the device has reached resumes at the ingest's boundary and counts what it skipped.
        Ring r(kFifo / kPage, 4);
        for (uint64_t k = 0; k < 8; k++) {
            r.write(k);
        }
        const uint64_t head = kFifo - kW + 3 * r.bytes();
        const Walked w = walk_frames(r.mem, 0, 8 * r.bytes(), r.mark_span(), out, fw, [&] { return head; });
        check(w.frames == 0 && w.dropped == 8 * r.bytes() && w.cursor == 8 * r.bytes(), "lapped: jump to the boundary");
        const Walked w2 = walk_frames(r.mem, 3 * r.bytes(), 8 * r.bytes(), r.mark_span(), out, fw, [&] { return head; });
        check(w2.frames == 5 && w2.dropped == 0, "lapped: the first intact frame walks");
    }
    {
        // With boundaries recorded, a lapped cursor resumes at the first boundary an eighth of the ring past the
        // horizon (frames 0-2 overwritten -> horizon 768 B; + 128 KB -> block 2 starts below that, block 3's boundary is
        // frame 768) and drops only up to there.
        Ring r(kFifo / kPage, 4);
        for (uint64_t k = 0; k < 1024; k++) {
            r.write(k);
        }
        const uint64_t head = kFifo - kW + 3 * r.bytes();
        const Walked w = walk_frames(r.mem, 0, 1024 * r.bytes(), r.mark_span(), out, fw, [&] { return head; });
        check(w.dropped == 768 * r.bytes() && w.frames == 64 && w.cursor == 832 * r.bytes(), "resume: at the boundary past the horizon");
        for (uint64_t i = 0; i < 64; i++) {
            check(holds(out.data() + i * r.bytes(), fw[i], r, 768 + i), "resume: the frames after it are whole");
        }
        const Walked w2 = walk_frames(r.mem, w.cursor, 1024 * r.bytes(), r.mark_span(), out, fw, [&] { return head; });
        check(w2.dropped == 0 && w2.frames == 64 && w2.cursor == 896 * r.bytes(), "resume: the walk continues");
    }
    {
        // A device overwriting the ring forever, announcing each frame after its bytes land, and a consumer walking
        // from just behind the lap point: no frame it is handed may be torn, and a lost position only moves forward. `observed` lags like the ingest thread's arrival count does.
        Ring r(kFifo / kPage, 4);
        std::atomic<uint32_t> word{0};
        std::atomic<uint64_t> observed{0};
        std::atomic<uint64_t> walked{0};
        std::atomic<bool> stop{false};
        std::thread device([&] {
            for (uint64_t k = 0; !stop.load(std::memory_order_relaxed); k++) {
                r.write(k);
                word.store(static_cast<uint32_t>((k + 1) * r.bytes()), std::memory_order_release);
                walked.store((k + 1) * r.bytes(), std::memory_order_release);
                if ((k & 63) == 0) {
                    observed.store((k + 1) * r.bytes(), std::memory_order_release);
                }
            }
        });
        const auto live_head = [&] { return widen_head(observed.load(std::memory_order_acquire), word.load(std::memory_order_acquire)); };
        while (walked.load(std::memory_order_acquire) < r.mem.size()) {
        }
        uint64_t cursor = 0, walks = 0, dropped_passes = 0, frames = 0;
        while (walks < 20000) {
            const uint64_t end = walked.load(std::memory_order_acquire);
            // Stay a quarter ring outside the danger zone, so most passes walk and the device, writing a frame every
            // few hundred nanoseconds, laps the consumer now and then during a copy.
            const uint64_t margin = r.mem.size() / 4;
            if (cursor + r.mem.size() < end + kW + margin) {
                cursor = end + kW - r.mem.size() + margin;
                cursor -= cursor % r.bytes();
            }
            const Walked w = walk_frames(r.mem, cursor, end, r.mark_span(), out, fw, live_head);
            walks++;
            if (w.dropped != 0) {
                check(w.cursor > cursor && w.cursor <= end, "concurrent: a lost position moves forward, never past the boundary");
                dropped_passes++;
            }
            const uint64_t first = (w.cursor - w.bytes) / r.bytes();
            for (uint32_t i = 0; i < w.frames; i++) {
                check(holds(out.data() + i * r.bytes(), fw[i], r, first + i), "concurrent: an intact frame is whole");
                frames++;
            }
            cursor = w.cursor;
        }
        stop.store(true);
        device.join();
        check(frames > 1000, "concurrent: the walk made progress");
        std::printf("concurrent: %llu walks, %llu frames, %llu passes lost their position\n",
                    static_cast<unsigned long long>(walks), static_cast<unsigned long long>(frames), static_cast<unsigned long long>(dropped_passes));
    }
    std::puts("PASS");
    return 0;
}
