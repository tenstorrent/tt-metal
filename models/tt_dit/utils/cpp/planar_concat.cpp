// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Outer scatter loop + persistent std::thread pool for the C++ planar
// concat path.  Inner SIMD work lives in transpose_avx2.cpp.

#include "planar_concat.hpp"
#include "transpose_avx2.hpp"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <emmintrin.h>  // SSE2 (_mm_stream_si128, _mm_sfence)
#include <immintrin.h>  // AVX2 (_mm256_stream_si256)

namespace tt_dit_planar {

// ---------------------------------------------------------------------------
// Static thread pool — created lazily on first use, kept alive for the rest
// of the process lifetime.  Matches the warm-pool semantics of the Python
// `_get_default_reassemble_pool()` it replaces.
// ---------------------------------------------------------------------------

namespace {

class ThreadPool {
public:
    explicit ThreadPool(int n_threads) : n_threads_(n_threads), stop_(false) {
        workers_.reserve(n_threads);
        for (int i = 0; i < n_threads; ++i) {
            workers_.emplace_back([this] { run_worker(); });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) {
            w.join();
        }
    }

    // Submit `n_tasks` tasks; each task is `fn(task_idx)`.  Blocks until all
    // complete.  Re-entrant calls are supported but rare (the planar path
    // doesn't recurse).
    template <typename Fn>
    void run(int n_tasks, Fn fn) {
        if (n_tasks <= 0) {
            return;
        }
        std::atomic<int> remaining(n_tasks);
        std::atomic<int> next_idx(0);
        std::mutex done_mu;
        std::condition_variable done_cv;

        {
            std::unique_lock<std::mutex> lock(mu_);
            for (int i = 0; i < n_tasks; ++i) {
                tasks_.emplace([&, this] {
                    int idx;
                    while ((idx = next_idx.fetch_add(1, std::memory_order_relaxed)) < n_tasks) {
                        fn(idx);
                    }
                    if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        std::lock_guard<std::mutex> lk(done_mu);
                        done_cv.notify_one();
                    }
                });
            }
        }
        cv_.notify_all();

        std::unique_lock<std::mutex> lk(done_mu);
        done_cv.wait(lk, [&] { return remaining.load() == 0; });
    }

    int n_threads() const { return n_threads_; }

private:
    void run_worker() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) {
                    return;
                }
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    int n_threads_;
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_;
};

// Process-wide singleton.  Configured via set_thread_pool_size() before
// first use; otherwise defaults to min(8, hardware_concurrency()).
static int g_requested_threads = 0;
static std::once_flag g_init_flag;
static ThreadPool* g_pool = nullptr;

ThreadPool& get_pool() {
    std::call_once(g_init_flag, [] {
        int n = g_requested_threads > 0 ? g_requested_threads : static_cast<int>(std::thread::hardware_concurrency());
        if (n <= 0) {
            n = 1;
        }
        if (n > 8) {
            n = 8;
        }
        g_pool = new ThreadPool(n);
    });
    return *g_pool;
}

}  // namespace

void set_thread_pool_size(int n_threads) {
    if (n_threads > 0) {
        g_requested_threads = n_threads;
    }
}

// ---------------------------------------------------------------------------
// Inner per-shard scatter kernels.
//
// Both kernels write a (T, h_per, w_per) destination region for one shard.
// The dest pointer (`out_plane_base`) is the byte address of plane row 0,
// pixel (r * h_per, c * w_per) — i.e. the upper-left of this shard's
// rectangle in the plane.  The dest stride between successive `t` rows is
// `row_stride` bytes; between successive `h_g` rows of one frame it's
// `plane_W` bytes; W stride is 1 byte.
// ---------------------------------------------------------------------------

namespace {

// Non-temporal copy of `n` bytes: writes via SSE/AVX streaming stores so the
// destination cache lines aren't pulled into L1/L2 (skipping read-for-
// ownership traffic).  Caller must `_mm_sfence()` before any later read of
// the destination region.
//
// Path selection by alignment:
//   - dst & 31 == 0 and n % 32 == 0  → AVX2 32-byte streams.
//   - dst & 15 == 0 and n % 16 == 0  → SSE2 16-byte streams.
//   - otherwise                       → libc memcpy (fallback).
//
// Loads are unaligned (`loadu`) — sources come from arbitrary torch tensor
// allocations whose alignment we don't control.  All our actual call sites
// hit the AVX2 path (Y plane, w_per=160, dst always 32-aligned) or the SSE2
// path (UV plane, w_per=80, dst at least 16-aligned for any c).
static inline void stream_copy_n(uint8_t* __restrict dst, const uint8_t* __restrict src, size_t n) {
    const uintptr_t addr = reinterpret_cast<uintptr_t>(dst);
    if ((addr & 31u) == 0 && (n & 31u) == 0) {
        for (size_t i = 0; i < n; i += 32) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
            _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + i), v);
        }
        return;
    }
    if ((addr & 15u) == 0 && (n & 15u) == 0) {
        for (size_t i = 0; i < n; i += 16) {
            __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
            _mm_stream_si128(reinterpret_cast<__m128i*>(dst + i), v);
        }
        return;
    }
    std::memcpy(dst, src, n);
}

// CTHW: src layout is (T, h_per, w_per), W innermost (stride 1).  For each
// (t, h), we have w_per contiguous source bytes that go to w_per
// contiguous dest bytes — a straight copy.  We use non-temporal stores so the
// 112 MB output buffer doesn't displace useful data from L2/L3 and so we
// don't pay the read-for-ownership tax on every cache line written.
void scatter_one_cthw(const uint8_t* src, uint8_t* dst, int T, int h_per, int w_per, int plane_W, int row_stride) {
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < h_per; ++h) {
            stream_copy_n(
                dst + t * static_cast<std::ptrdiff_t>(row_stride) + h * static_cast<std::ptrdiff_t>(plane_W),
                src + (t * h_per + h) * static_cast<std::ptrdiff_t>(w_per),
                static_cast<size_t>(w_per));
        }
    }
    // Drain WC buffers so subsequent readers (other threads, the Python
    // caller) see the data.  Cheap when no streaming stores were issued
    // (i.e. fallback memcpy path).
    _mm_sfence();
}

// CHWT: src layout is (h_per, w_per, T), T innermost (stride 1).  Dest has
// W innermost (stride 1) and T at stride row_stride.  Per-h slice is a
// (w_per, T) → (T, w_per) byte transpose, tiled in 32×32 blocks.
//
// w_per for our case is in {160, 80}.  T is 81 (2 full 32-T tiles + a
// 17-T tail).  We handle both:
//   - w_chunks of 32 (Y: 5 chunks; UV: 2 full chunks + 1 tail of 16).
//   - t_chunks of 32 (full: 2; tail: 1 of 17).
void scatter_one_chwt(const uint8_t* src, uint8_t* dst, int T, int h_per, int w_per, int plane_W, int row_stride) {
    const std::ptrdiff_t src_w_stride = static_cast<std::ptrdiff_t>(T);  // bytes between adjacent w in source

    const int n_w_full_tiles = w_per / 32;
    const int w_tail = w_per - n_w_full_tiles * 32;  // 0 or 16 in our shapes

    const int n_t_full_tiles = T / 32;
    const int t_tail = T - n_t_full_tiles * 32;

    for (int h = 0; h < h_per; ++h) {
        const uint8_t* src_h = src + static_cast<std::ptrdiff_t>(h) * w_per * T;
        uint8_t* dst_h = dst + static_cast<std::ptrdiff_t>(h) * plane_W;

        // Full 32×32 (W × T) tiles.
        for (int w_tile = 0; w_tile < n_w_full_tiles; ++w_tile) {
            const uint8_t* src_w = src_h + static_cast<std::ptrdiff_t>(w_tile) * 32 * T;
            uint8_t* dst_w = dst_h + static_cast<std::ptrdiff_t>(w_tile) * 32;

            for (int t_tile = 0; t_tile < n_t_full_tiles; ++t_tile) {
                const uint8_t* src_tile = src_w + t_tile * 32;
                uint8_t* dst_tile = dst_w + static_cast<std::ptrdiff_t>(t_tile) * 32 * row_stride;
                transpose_32x32_u8(src_tile, src_w_stride, dst_tile, row_stride);
            }
            if (t_tail > 0) {
                const uint8_t* src_tile = src_w + n_t_full_tiles * 32;
                uint8_t* dst_tile = dst_w + static_cast<std::ptrdiff_t>(n_t_full_tiles) * 32 * row_stride;
                transpose_32xN_u8(src_tile, src_w_stride, dst_tile, row_stride, t_tail);
            }
        }

        // W-tail (when w_per % 32 != 0).
        //
        // Fast path for w_tail == 16 (UV plane, w_per_uv = 80 = 2*32 + 16):
        // 16 W-source rows × 32 T-source cols → 32 T-dest rows × 16 W-dest
        // cols, done with two back-to-back 16×16 SSE transposes.  No bounce
        // buffer, no zero-padding pass.
        //
        // General path (w_tail in [1, 31] \ {16}): zero-pad source into a
        // 32×32 stack temp, transpose, copy only the valid w_tail cols from
        // each output row.  Slower but covers every shape we might see.
        // (Currently only triggered if w_tail != 16, which doesn't happen for
        // 720p Y/UV; kept for safety.)
        if (w_tail > 0) {
            const uint8_t* src_w = src_h + static_cast<std::ptrdiff_t>(n_w_full_tiles) * 32 * T;
            uint8_t* dst_w = dst_h + static_cast<std::ptrdiff_t>(n_w_full_tiles) * 32;

            if (w_tail == 16) {
                for (int t_tile = 0; t_tile < n_t_full_tiles; ++t_tile) {
                    transpose_32x16_u8(
                        src_w + t_tile * 32,
                        src_w_stride,
                        dst_w + static_cast<std::ptrdiff_t>(t_tile) * 32 * row_stride,
                        row_stride);
                }
                if (t_tail > 0) {
                    // T-tail × 16 W-cols: small enough to keep the bounce
                    // buffer; reduces case explosion in this code path.
                    alignas(32) uint8_t tmp_src[32 * 32];
                    alignas(32) uint8_t tmp_dst[32 * 32];
                    std::memset(tmp_src, 0, sizeof(tmp_src));
                    for (int i = 0; i < 16; ++i) {
                        std::memcpy(tmp_src + i * 32, src_w + i * src_w_stride + n_t_full_tiles * 32, t_tail);
                    }
                    transpose_32x32_u8(tmp_src, 32, tmp_dst, 32);
                    uint8_t* dst_tile = dst_w + static_cast<std::ptrdiff_t>(n_t_full_tiles) * 32 * row_stride;
                    for (int i = 0; i < t_tail; ++i) {
                        std::memcpy(dst_tile + i * row_stride, tmp_dst + i * 32, 16);
                    }
                }
            } else {
                alignas(32) uint8_t tmp_src[32 * 32];
                alignas(32) uint8_t tmp_dst[32 * 32];

                for (int t_tile = 0; t_tile < n_t_full_tiles; ++t_tile) {
                    std::memset(tmp_src, 0, sizeof(tmp_src));
                    for (int i = 0; i < w_tail; ++i) {
                        std::memcpy(tmp_src + i * 32, src_w + i * src_w_stride + t_tile * 32, 32);
                    }
                    transpose_32x32_u8(tmp_src, 32, tmp_dst, 32);
                    uint8_t* dst_tile = dst_w + static_cast<std::ptrdiff_t>(t_tile) * 32 * row_stride;
                    for (int i = 0; i < 32; ++i) {
                        std::memcpy(dst_tile + i * row_stride, tmp_dst + i * 32, w_tail);
                    }
                }
                if (t_tail > 0) {
                    std::memset(tmp_src, 0, sizeof(tmp_src));
                    for (int i = 0; i < w_tail; ++i) {
                        std::memcpy(tmp_src + i * 32, src_w + i * src_w_stride + n_t_full_tiles * 32, t_tail);
                    }
                    transpose_32x32_u8(tmp_src, 32, tmp_dst, 32);
                    uint8_t* dst_tile = dst_w + static_cast<std::ptrdiff_t>(n_t_full_tiles) * 32 * row_stride;
                    for (int i = 0; i < t_tail; ++i) {
                        std::memcpy(dst_tile + i * row_stride, tmp_dst + i * 32, w_tail);
                    }
                }
            }
        }
    }
}

}  // namespace

void scatter_component(
    const std::vector<ShardView>& shards,
    DimOrder dim_order,
    uint8_t* out,
    int T,
    int plane_offset,
    int plane_W,
    int row_stride,
    int h_per,
    int w_per) {
    auto& pool = get_pool();
    const int n = static_cast<int>(shards.size());

    pool.run(n, [&](int idx) {
        const ShardView& sv = shards[idx];
        uint8_t* dst_base = out + static_cast<std::ptrdiff_t>(plane_offset) +
                            static_cast<std::ptrdiff_t>(sv.r) * h_per * plane_W +
                            static_cast<std::ptrdiff_t>(sv.c) * w_per;

        if (dim_order == DimOrder::CTHW) {
            scatter_one_cthw(sv.data, dst_base, T, h_per, w_per, plane_W, row_stride);
        } else {
            scatter_one_chwt(sv.data, dst_base, T, h_per, w_per, plane_W, row_stride);
        }
    });
}

void planar_concat(
    const std::vector<ShardView>& y_shards,
    int y_h_per,
    int y_w_per,
    const std::vector<ShardView>& cb_shards,
    int uv_h_per,
    int uv_w_per,
    const std::vector<ShardView>& cr_shards,
    DimOrder dim_order,
    int T,
    int H,
    int W,
    uint8_t* out) {
    const int Hu = H / 2;
    const int Wu = W / 2;
    const int hw = H * W;
    const int uv = Hu * Wu;
    const int row_stride = hw + 2 * uv;

    // Build a flat task list across all 3 components so the thread pool can
    // load-balance Y (4× the bytes of UV) against UV.  Each entry is a
    // resolved per-shard task.
    struct Task {
        const ShardView* shard;
        int plane_offset;
        int plane_W;
        int h_per;
        int w_per;
    };

    std::vector<Task> tasks;
    tasks.reserve(y_shards.size() + cb_shards.size() + cr_shards.size());

    auto add_component = [&](const std::vector<ShardView>& s, int plane_off, int plane_w_arg, int h_p, int w_p) {
        for (const auto& sv : s) {
            tasks.push_back({&sv, plane_off, plane_w_arg, h_p, w_p});
        }
    };
    add_component(y_shards, 0, W, y_h_per, y_w_per);
    add_component(cb_shards, hw, Wu, uv_h_per, uv_w_per);
    add_component(cr_shards, hw + uv, Wu, uv_h_per, uv_w_per);

    auto& pool = get_pool();
    pool.run(static_cast<int>(tasks.size()), [&](int idx) {
        const Task& tk = tasks[idx];
        const ShardView& sv = *tk.shard;
        uint8_t* dst_base = out + static_cast<std::ptrdiff_t>(tk.plane_offset) +
                            static_cast<std::ptrdiff_t>(sv.r) * tk.h_per * tk.plane_W +
                            static_cast<std::ptrdiff_t>(sv.c) * tk.w_per;

        if (dim_order == DimOrder::CTHW) {
            scatter_one_cthw(sv.data, dst_base, T, tk.h_per, tk.w_per, tk.plane_W, row_stride);
        } else {
            scatter_one_chwt(sv.data, dst_base, T, tk.h_per, tk.w_per, tk.plane_W, row_stride);
        }
    });
}

}  // namespace tt_dit_planar
