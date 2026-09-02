// Standalone check of spsc_atomic16_avx512 + SpscNtCarry against a scalar reference.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "spsc_marker_decode.hpp"

using namespace tt::tt_metal::profiler;

int main() {
    std::mt19937_64 rng(0xC0FFEE);
    constexpr uint32_t kSlots = 1u << 12;  // power-of-two slot count, 24 B slots
    const uint64_t cap_bytes = kSlots * 24ull;
    std::vector<uint8_t> ring(cap_bytes + 64, 0);
    uint8_t* rb = ring.data();
    // align to 64
    while (reinterpret_cast<uintptr_t>(rb) & 63) rb++;

    for (int trial = 0; trial < 20000; trial++) {
        SpscNtCarry sw{};
        sw.ring = rb;
        sw.cap_bytes = cap_bytes;
        sw.line_off = 0;
        std::memset(rb, 0xAB, cap_bytes);

        std::vector<uint64_t> expect;  // qwords
        // random mix of put3 records and atomic blocks
        uint32_t th = rng() & 0x7FFFFFF, prog = (uint32_t)rng(), lane = rng() & 0x3FF, dev = rng() & 7;
        const uint64_t meta = ((uint64_t)((lane << 16) | (dev << 26))) << 32;
        int ops = 1 + (int)(rng() % 8);
        for (int op = 0; op < ops; op++) {
            if (rng() % 3 == 0) {  // scalar-ish record via put3
                uint64_t a = rng(), b = rng(), c = rng();
                sw.put3(a, b, c);
                expect.push_back(a);
                expect.push_back(b);
                expect.push_back(c);
                continue;
            }
            // atomic block: build a wire run of `m` atomic records, possibly followed by a non-atomic word
            uint32_t m = 1 + (uint32_t)(rng() % 16);
            uint32_t extra_nonatomic = rng() % 2;
            std::vector<uint32_t> wire;
            for (uint32_t r = 0; r < m; r++) {
                uint32_t id = (uint32_t)rng() & 0x07FFFFFF;
                wire.push_back((PP_ZONE_ATOMIC << PP_TYPE_SHIFT) | id);
                wire.push_back((uint32_t)rng());
                wire.push_back((uint32_t)rng());
            }
            if (extra_nonatomic) {
                wire.push_back((PP_EVENT << PP_TYPE_SHIFT) | 123);  // type 12: screen must stop here
                wire.push_back((uint32_t)rng());
            }
            // slack so full loads are legal when we claim avail>=48
            size_t live = wire.size();
            while (wire.size() < live + 48) wire.push_back((uint32_t)rng());
            // production contract: max_recs = (run-i)/3 -- the screen always meets a genuine non-atomic
            // packet boundary (or the cap) before any aliasing slack word
            uint32_t max_recs = extra_nonatomic ? m + (uint32_t)(rng() % 5) : m;
            uint32_t avail;
            if (rng() % 2) {
                avail = 48;  // frame-interior claim
            } else {
                avail = (uint32_t)live;  // exact live-window claim
                if (avail > 48) avail = 48;
            }
            const uint32_t nexp = std::min(std::min(m, max_recs), 16u);
            auto res = spsc_atomic16_avx512(wire.data(), avail, max_recs, th, prog, lane, dev, sw);
            if (res.n != nexp) {
                printf("trial %d op %d: n=%u expected %u (m=%u max=%u avail=%u)\n", trial, op, res.n, nexp, m,
                       max_recs, avail);
                return 1;
            }
            if (nexp == 0) continue;
            const uint64_t th_hi = (uint64_t)th << 32;
            uint64_t ts_first = th_hi | wire[1];
            uint64_t ts_last = th_hi | wire[3 * nexp - 2];
            if (res.ts_first != ts_first || res.ts_last != ts_last) {
                printf("trial %d op %d: ts endpoints wrong\n", trial, op);
                return 1;
            }
            for (uint32_t r = 0; r < nexp; r++) {
                uint64_t q0 = th_hi | wire[3 * r + 1];
                uint64_t q1 = meta | (wire[3 * r] & 0x07FFFFFF);
                uint64_t q2 = ((uint64_t)wire[3 * r + 2] << 32) | prog;
                expect.push_back(q0);
                expect.push_back(q1);
                expect.push_back(q2);
            }
        }
        sw.flush_tail();
        // verify ring contents
        for (size_t q = 0; q < expect.size(); q++) {
            uint64_t got;
            std::memcpy(&got, rb + (q * 8) % cap_bytes, 8);
            if (got != expect[q]) {
                printf("trial %d: qword %zu mismatch: got %016lx want %016lx\n", trial, q, got, expect[q]);
                return 1;
            }
        }
    }
    // wrap test: fill past cap_bytes
    {
        SpscNtCarry sw{};
        sw.ring = rb;
        sw.cap_bytes = cap_bytes;
        sw.line_off = 0;
        uint64_t next = 1;
        std::vector<uint64_t> all;
        for (uint64_t total = 0; total < 3 * kSlots + 7; total++) {
            sw.put3(next, next + 1, next + 2);
            all.push_back(next);
            all.push_back(next + 1);
            all.push_back(next + 2);
            next += 3;
        }
        sw.flush_tail();
        // last cap_bytes worth of qwords must be present at their wrapped positions
        size_t nq = all.size();
        size_t ring_q = cap_bytes / 8;
        for (size_t q = nq - ring_q; q < nq; q++) {
            uint64_t got;
            std::memcpy(&got, rb + (q * 8) % cap_bytes, 8);
            if (got != all[q]) {
                // the trailing partial line is restreamed by flush_tail with only cq lanes -- qwords in the
                // same line AFTER the flush point but before an older overwrite are stale; only qwords the
                // carry ever held must match. Everything up to the final commit point must match though:
                printf("wrap: qword %zu mismatch got %016lx want %016lx\n", q, got, all[q]);
                return 1;
            }
        }
    }
    printf("OK\n");
    return 0;
}
