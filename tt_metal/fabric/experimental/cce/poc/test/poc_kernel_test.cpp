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
#include "cce_fake.h"
#include "dev_mem_map.h"
#include "internal/ethernet/tunneling.h"

void kernel_main();

// ---------------------------------------------------------------- fakes

namespace {

constexpr uint32_t kNumThreads = 2;

thread_local uint32_t t_thread_id;
std::barrier<> g_barrier(kNumThreads);
std::vector<uint32_t> g_rt_args;
std::vector<uint32_t> g_common_args;

uint8_t* cached(uintptr_t addr) { return reinterpret_cast<uint8_t*>(addr); }

void check_range(uintptr_t addr, size_t size) {
    if (addr < cce_fake::kWindowBase || addr + size > cce_fake::kWindowBase + cce_fake::kWindowSize) {
        std::fprintf(stderr, "access [%#lx, +%#zx) outside the mapped window\n", addr, size);
        std::abort();
    }
}

// Widen to whole 64-bit words, as libs/cce_cache does.
void word_range(uintptr_t addr, size_t size, uintptr_t& start, size_t& bytes) {
    start = addr & ~uintptr_t{7};
    bytes = (addr + size - start + 7) & ~size_t{7};
    check_range(start, bytes);
}

}  // namespace

namespace cce_fake {

void set_thread_id(uint32_t id) { t_thread_id = id; }

void set_args(const uint32_t* rt_args, size_t num_rt_args, const uint32_t* common_args, size_t num_common_args) {
    g_rt_args.assign(rt_args, rt_args + num_rt_args);
    g_common_args.assign(common_args, common_args + num_common_args);
}

uint8_t* sram(uintptr_t addr) { return reinterpret_cast<uint8_t*>(addr + MEM_L1_UNCACHED_BASE); }

}  // namespace cce_fake

uint32_t get_num_threads() { return kNumThreads; }
uint32_t get_my_thread_id() { return t_thread_id; }
void sync_threads() { g_barrier.arrive_and_wait(); }

uint32_t fake_arg(int arg_idx) { return g_rt_args.at(arg_idx); }
uint32_t fake_common_arg(int arg_idx) { return g_common_args.at(arg_idx); }

void flush_l2_cache_range(uintptr_t start_addr, size_t size) {
    uintptr_t start;
    size_t bytes;
    word_range(start_addr, size, start, bytes);
    std::memcpy(cce_fake::sram(start), cached(start), bytes);
}

void invalidate_l2_cache_range(uintptr_t start_addr, size_t size) {
    uintptr_t start;
    size_t bytes;
    word_range(start_addr, size, start, bytes);
    std::memcpy(cached(start), cce_fake::sram(start), bytes);
}

void noc_async_write(uint32_t src_local_l1_addr, uint64_t dst_noc_addr, uint32_t size, uint8_t, uint32_t) {
    check_range(src_local_l1_addr, size);
    check_range(dst_noc_addr, size);
    std::memmove(cce_fake::sram(dst_noc_addr), cce_fake::sram(src_local_l1_addr), size);
}

void internal_::eth_send_packet_fake(uint32_t src_word_addr, uint32_t dest_word_addr, uint32_t num_words) {
    uintptr_t src = uintptr_t{src_word_addr} << 4;
    uintptr_t dst = uintptr_t{dest_word_addr} << 4;
    size_t size = size_t{num_words} << 4;
    check_range(src, size);
    check_range(dst, size);
    std::memmove(cce_fake::sram(dst), cce_fake::sram(src), size);
}

// ---------------------------------------------------------------- test

namespace {

constexpr uint32_t kPatternWords = 64;  // named_ct_arg_map_generated.h
constexpr uint32_t kSrc = 0x3E0000;
constexpr uint32_t kNocDst = 0x3E2000;
constexpr uint32_t kEthDst = 0x3E8000;
constexpr uint32_t kResults = 0x3F80C0;
constexpr uint32_t kSeed = 0x12345678;

// The kernel's documented pattern.
uint32_t pattern_word(uint32_t i) { return kSeed + i * 0x01000193u; }

bool map_window() {
    int fd = memfd_create("cce_sram", 0);
    if (fd < 0 || ftruncate(fd, cce_fake::kWindowSize) != 0) {
        std::perror("memfd");
        return false;
    }
    void* sram = mmap(
        cce_fake::sram(cce_fake::kWindowBase),
        cce_fake::kWindowSize,
        PROT_READ | PROT_WRITE,
        MAP_SHARED | MAP_FIXED_NOREPLACE,
        fd,
        0);
    void* view = mmap(
        cached(cce_fake::kWindowBase),
        cce_fake::kWindowSize,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE,
        -1,
        0);
    close(fd);
    if (sram != cce_fake::sram(cce_fake::kWindowBase) || view != cached(cce_fake::kWindowBase)) {
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
        std::memcpy(cce_fake::sram(addr + 4 * i), &v, 4);
    }
}

uint32_t sram_word(uint32_t addr) {
    uint32_t v;
    std::memcpy(&v, cce_fake::sram(addr), 4);
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
    std::memcpy(cce_fake::sram(kResults), &results_poison, 4);
    std::memcpy(cce_fake::sram(kResults + 4), &results_poison, 4);

    const uint32_t rt_args[] = {kSrc, kNocDst, kEthDst, kResults};
    const uint32_t common_args[] = {kSeed};
    cce_fake::set_args(rt_args, std::size(rt_args), common_args, std::size(common_args));

    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < kNumThreads; t++) {
        threads.emplace_back([t] {
            cce_fake::set_thread_id(t);
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
