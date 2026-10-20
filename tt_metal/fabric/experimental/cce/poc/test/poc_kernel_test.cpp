// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Runs the POC kernel's two threads on std::thread against the fakes and
// checks the handoff, the result words and the bytes in SRAM. Returns nonzero
// on any failure.

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <barrier>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "api/dataflow/dataflow_api.h"
#include "api/kernel_thread_globals.h"
#include "dev_mem_map.h"
#include "internal/ethernet/tunneling.h"

void kernel_main();

// ---------------------------------------------------------------- fakes

// Host-side model of one CCE: SRAM sits at the uncached alias
// (addr + MEM_L1_UNCACHED_BASE) and the CPU's cached view is a separate copy at
// the CCE address itself, so data moves between them only through the flush
// and invalidate fakes.

namespace {

constexpr uint32_t kNumThreads = 2;
// The CCE-local window the test maps: the buffer regions and launch block.
constexpr uintptr_t kWindowBase = 0x3D0000;
constexpr size_t kWindowSize = 0x30000;

thread_local uint32_t t_thread_id;
std::barrier<> g_barrier(kNumThreads);
std::vector<uint32_t> g_rt_args;
std::vector<uint32_t> g_common_args;

uint8_t* cached(uintptr_t addr) { return reinterpret_cast<uint8_t*>(addr); }
uint8_t* sram(uintptr_t addr) { return reinterpret_cast<uint8_t*>(addr + MEM_L1_UNCACHED_BASE); }

void check_range(uintptr_t addr, size_t size) {
    if (addr < kWindowBase || addr + size > kWindowBase + kWindowSize) {
        std::fprintf(stderr, "access [%#lx, +%#zx) outside the mapped window\n", addr, size);
        std::abort();
    }
}

struct WordRange {
    uintptr_t start;
    size_t bytes;
};

// Widen to whole 64-bit words, as libs/cce_cache does.
WordRange word_range(uintptr_t addr, size_t size) {
    WordRange r{addr & ~uintptr_t{7}, 0};
    r.bytes = (addr + size - r.start + 7) & ~size_t{7};
    check_range(r.start, r.bytes);
    return r;
}

void sram_copy(uintptr_t dst, uintptr_t src, size_t size) {
    check_range(src, size);
    check_range(dst, size);
    std::memmove(sram(dst), sram(src), size);
}

}  // namespace

uint32_t get_num_threads() { return kNumThreads; }
uint32_t get_my_thread_id() { return t_thread_id; }
void sync_threads() { g_barrier.arrive_and_wait(); }

uint32_t fake_arg(int arg_idx) { return g_rt_args.at(arg_idx); }
uint32_t fake_common_arg(int arg_idx) { return g_common_args.at(arg_idx); }

void flush_l2_cache_range(uintptr_t start_addr, size_t size) {
    WordRange r = word_range(start_addr, size);
    std::memcpy(sram(r.start), cached(r.start), r.bytes);
}

void invalidate_l2_cache_range(uintptr_t start_addr, size_t size) {
    WordRange r = word_range(start_addr, size);
    std::memcpy(cached(r.start), sram(r.start), r.bytes);
}

void noc_async_write(uint32_t src_local_l1_addr, uint64_t dst_noc_addr, uint32_t size, uint8_t, uint32_t) {
    sram_copy(dst_noc_addr, src_local_l1_addr, size);
}

void internal_::eth_send_packet_fake(uint32_t src_word_addr, uint32_t dest_word_addr, uint32_t num_words) {
    sram_copy(uintptr_t{dest_word_addr} << 4, uintptr_t{src_word_addr} << 4, size_t{num_words} << 4);
}

// ---------------------------------------------------------------- test

namespace {

constexpr uint32_t kPatternWords = POC_PATTERN_WORDS;
constexpr uint32_t kSrc = 0x3E0000;
constexpr uint32_t kNocDst = 0x3E2000;
constexpr uint32_t kEthDst = 0x3E8000;
constexpr uint32_t kResults = 0x3F80C0;
constexpr uint32_t kSeed = 0x12345678;

// The kernel's documented pattern.
uint32_t pattern_word(uint32_t i) { return kSeed + i * 0x01000193u; }

bool map_window() {
    int fd = memfd_create("cce_sram", 0);
    if (fd < 0 || ftruncate(fd, kWindowSize) != 0) {
        std::perror("memfd");
        return false;
    }
    void* sram_map =
        mmap(sram(kWindowBase), kWindowSize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED_NOREPLACE, fd, 0);
    void* view_map = mmap(
        cached(kWindowBase),
        kWindowSize,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE,
        -1,
        0);
    close(fd);
    if (sram_map != sram(kWindowBase) || view_map != cached(kWindowBase)) {
        std::perror("mmap at the CCE addresses");
        return false;
    }
    return true;
}

// Stale data in both views, so only real moves can make a destination match.
void poison(uint32_t addr) {
    for (uint32_t i = 0; i < kPatternWords; i++) {
        uint32_t v = 0xDEAD0000u ^ i;
        std::memcpy(cached(addr + 4 * i), &v, 4);
        std::memcpy(sram(addr + 4 * i), &v, 4);
    }
}

uint32_t sram_word(uint32_t addr) {
    uint32_t v;
    std::memcpy(&v, sram(addr), 4);
    return v;
}

int check_pattern(const char* what, uint32_t addr) {
    int bad = 0;
    for (uint32_t i = 0; i < kPatternWords; i++) {
        bad += sram_word(addr + 4 * i) != pattern_word(i);
    }
    if (bad) {
        std::fprintf(stderr, "FAIL: %s: %d of %u words wrong in SRAM\n", what, bad, kPatternWords);
    }
    return bad;
}

}  // namespace

int main() {
    if (!map_window()) {
        return 2;
    }
    poison(kSrc);
    poison(kNocDst);
    poison(kEthDst);
    const uint32_t results_poison = 0xFFFFFFFF;
    std::memcpy(sram(kResults), &results_poison, 4);
    std::memcpy(sram(kResults + 4), &results_poison, 4);

    g_rt_args = {kSrc, kNocDst, kEthDst, kResults};
    g_common_args = {kSeed};

    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < kNumThreads; t++) {
        threads.emplace_back([t] {
            t_thread_id = t;
            kernel_main();
        });
    }
    for (auto& t : threads) {
        t.join();
    }

    int failures = check_pattern("source", kSrc) + check_pattern("NOC destination", kNocDst) +
                   check_pattern("Ethernet destination", kEthDst);
    uint32_t uncached_bad = sram_word(kResults);
    uint32_t cached_bad = sram_word(kResults + 4);
    if (uncached_bad != 0 || cached_bad != 0) {
        std::fprintf(stderr, "FAIL: result words: uncached %u, cached %u (want 0, 0)\n", uncached_bad, cached_bad);
        failures++;
    }
    std::printf("%s\n", failures ? "poc_kernel_test: FAIL" : "poc_kernel_test: PASS");
    return failures ? 1 : 0;
}
