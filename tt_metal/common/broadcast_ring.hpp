// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>

#include <tt_stl/assert.hpp>
#include <tt_stl/tt_pause.hpp>

#if defined(__x86_64__)
#include <emmintrin.h>
#endif
#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace tt::tt_metal {

namespace broadcast_ring_detail {

// Concurrent slot copy. BOTH the writer and every reader MUST go through these helpers (never a
// plain C++ load/store or std::memcpy on live slots): concurrent non-atomic access is a data race /
// UB in the C++ abstract machine even when claim-drop makes torn values harmless.
//
// x86: MOV / MOVDQU / MOVNTDQ in asm with a "memory" clobber -- outside the abstract machine; we rely
// on x86 TSO (+ sfence after NT) for visibility, and on claim-recheck to drop any torn snapshot.
// Elsewhere: std::atomic_ref<uint64_t> relaxed on 8 B units (ISO-defined; values may still tear
// across units, which claim-drop handles the same way).

inline void concurrent_copy(void* dst, const void* src, size_t bytes) {
#if defined(__x86_64__)
    auto* d = static_cast<unsigned char*>(dst);
    const auto* s = static_cast<const unsigned char*>(src);
    // Prefer 16 B moves when dst is 16-aligned (ring slots are); src may be unaligned (batch scratch).
    if (bytes >= 16 && (reinterpret_cast<uintptr_t>(d) & 15u) == 0) {
        while (bytes >= 16) {
            asm volatile(
                "movdqu (%[src]), %%xmm0\n\t"
                "movdqa %%xmm0, (%[dst])\n\t"
                :
                : [dst] "r"(d), [src] "r"(s)
                : "xmm0", "memory");
            d += 16;
            s += 16;
            bytes -= 16;
        }
    }
    while (bytes >= 8) {
        asm volatile(
            "movq (%[src]), %%rax\n\t"
            "movq %%rax, (%[dst])\n\t"
            :
            : [dst] "r"(d), [src] "r"(s)
            : "rax", "memory");
        d += 8;
        s += 8;
        bytes -= 8;
    }
    while (bytes != 0) {
        asm volatile(
            "movb (%[src]), %%al\n\t"
            "movb %%al, (%[dst])\n\t"
            :
            : [dst] "r"(d), [src] "r"(s)
            : "al", "memory");
        d += 1;
        s += 1;
        bytes -= 1;
    }
#else
    // ISO path: relaxed atomic_ref on both ends so either side may be the shared slot.
    auto* d = static_cast<unsigned char*>(dst);
    const auto* s = static_cast<const unsigned char*>(src);
    while (bytes >= sizeof(uint64_t) && (reinterpret_cast<uintptr_t>(d) % alignof(uint64_t)) == 0 &&
           (reinterpret_cast<uintptr_t>(s) % alignof(uint64_t)) == 0) {
        uint64_t w = std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t*>(const_cast<unsigned char*>(s)))
                         .load(std::memory_order_relaxed);
        std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t*>(d)).store(w, std::memory_order_relaxed);
        d += sizeof(uint64_t);
        s += sizeof(uint64_t);
        bytes -= sizeof(uint64_t);
    }
    while (bytes != 0) {
        const unsigned char b =
            std::atomic_ref<unsigned char>(*const_cast<unsigned char*>(s)).load(std::memory_order_relaxed);
        std::atomic_ref<unsigned char>(*d).store(b, std::memory_order_relaxed);
        d += 1;
        s += 1;
        bytes -= 1;
    }
#endif
}

// Non-temporal write into ring slots. Same concurrent-access contract as concurrent_copy: asm only,
// reader still loads via concurrent_copy. WC stores need stream_fence() before publishing head.
// Always AVX-32 NT (vmovdqu/vmovntdq) then 16 B SSE NT for large copies -- no env knobs.
inline void concurrent_stream_copy(void* dst, const void* src, size_t bytes) {
#if defined(__x86_64__)
    auto* d = static_cast<unsigned char*>(dst);
    const auto* s = static_cast<const unsigned char*>(src);
    // Align dst to 32 B for AVX NT stores.
    const size_t misalign = reinterpret_cast<uintptr_t>(d) & 31u;
    if (misalign != 0) {
        const size_t head = std::min(32u - static_cast<unsigned>(misalign), static_cast<unsigned>(bytes));
        concurrent_copy(d, s, head);
        d += head;
        s += head;
        bytes -= head;
    }
    while (bytes >= 32) {
        asm volatile(
            "vmovdqu (%[src]), %%ymm0\n\t"
            "vmovntdq %%ymm0, (%[dst])\n\t"
            :
            : [dst] "r"(d), [src] "r"(s)
            : "ymm0", "memory");
        d += 32;
        s += 32;
        bytes -= 32;
    }
    asm volatile("vzeroupper" ::: "ymm0", "memory");
    while (bytes >= 16) {
        asm volatile(
            "movdqu (%[src]), %%xmm0\n\t"
            "movntdq %%xmm0, (%[dst])\n\t"
            :
            : [dst] "r"(d), [src] "r"(s)
            : "xmm0", "memory");
        d += 16;
        s += 16;
        bytes -= 16;
    }
    if (bytes != 0) {
        concurrent_copy(d, s, bytes);
    }
#else
    concurrent_copy(dst, src, bytes);
#endif
}

inline void stream_fence() {
#if defined(__x86_64__)
    _mm_sfence();
#endif
}

}  // namespace broadcast_ring_detail

/**
 * @brief Single-producer, multi-consumer broadcast ring buffer.
 *
 * One writer publishes a stream of items; each reader observes that stream independently and in
 * order, starting from the point at which the reader is created. The writer never blocks on a reader: a
 * reader that cannot keep up loses its oldest unread items (tracked by Reader::dropped()).
 *
 * If the compile-time constant `is_always_lock_free` is true, then the ring is guaranteed to be lock-free.
 * In particular, all publish and read operations are wait-free in this case.
 *
 * Obtain the writer from writer() (driven by a single thread) and each reader from make_reader().
 * Readers are single-threaded, and all readers must be destroyed before the ring.
 *
 * @tparam T Element type.
 */
template <typename T>
class BroadcastRing {
    static constexpr bool kTriviallyCopyable = std::is_trivially_copyable_v<T>;
#if defined(__cpp_lib_atomic_lock_free_type_aliases)
    using WakeTokenAtomic = std::atomic_unsigned_lock_free;
#else
    using WakeTokenAtomic = std::atomic<uint32_t>;
#endif
    static constexpr size_t kFalseSharingSize = 128;
    // Below this, a batch fits in cache and readers likely consume it soon -- a cached copy wins.
    // 4 KiB: profiler publishes multi-KB slot batches; NT sooner keeps L3 for decode/copy.
    static constexpr size_t kStreamCopyMinBytes = 4096;
    struct SlotsView;
    struct SharedState;

public:
    /**
     * @brief True when the ring is guaranteed to be lock-free: requires T to be trivially copyable (otherwise
     * each slot is guarded by a mutex) and the platform's 64-bit and wake-token atomics to be lock-free.
     */
    static constexpr bool is_always_lock_free =
        kTriviallyCopyable && std::atomic<uint64_t>::is_always_lock_free && WakeTokenAtomic::is_always_lock_free;

    /**
     * @brief Constructs a broadcast ring with at least @p capacity slots.
     * @param capacity Requested slot count; gets rounded up to the next power of two.
     */
    explicit BroadcastRing(size_t capacity) :
        capacity_(capacity ? std::bit_ceil(capacity) : 1),
        slots_(std::make_unique<Slot[]>(capacity_)),
        writer_(&shared_state_, view()) {
        warm_pages(/*lock=*/true);
    }

    /**
     * @brief Prefault (and optionally mlock) slot storage so the first publish isn't eating soft faults
     * on the timed path. Safe to call again right before a capture -- re-touches every page.
     *
     * Touches the same Slot layout used by publish_batch (Slot == T-sized storage for trivially-copyable
     * T), plus asks for THP and pins so a long device bring-up can't reclaim the working set before the
     * first D2H.
     */
    void warm_pages(bool lock = true) {
        if constexpr (!kTriviallyCopyable) {
            return;
        }
        void* const base = static_cast<void*>(slots_.get());
        const size_t bytes = capacity_ * sizeof(Slot);
#if defined(__linux__)
        if (bytes >= (size_t{8} << 20)) {
            const uintptr_t addr = reinterpret_cast<uintptr_t>(base);
            const uintptr_t page = 4096;
            const uintptr_t lo = (addr + page - 1) & ~(page - 1);
            const uintptr_t hi = (addr + bytes) & ~(page - 1);
            if (hi > lo) {
                // Fault-time is when THP is granted -- ask before the touch.
                madvise(reinterpret_cast<void*>(lo), hi - lo, MADV_HUGEPAGE);
                madvise(reinterpret_cast<void*>(lo), hi - lo, MADV_WILLNEED);
            }
        }
#endif
        // Full memset: hard-faults every 4K (or THP) into the calling thread's address space.
        std::memset(base, 0, bytes);
#if defined(__linux__)
        if (lock && bytes >= (size_t{1} << 20)) {
            // Best-effort pin. Failure is fine (RLIMIT_MEMLOCK); pages are still resident from memset.
            (void)mlock(base, bytes);
        }
#endif
        // Second pass: stride by page with a volatile byte store. Catches holes and warms the TLB for
        // the emit path's access pattern. (Avoid volatile T= -- T may not have a volatile assignment.)
        auto* const bytes_p = static_cast<unsigned char*>(base);
        constexpr size_t kPage = 4096;
        for (size_t off = 0; off < bytes; off += kPage) {
            volatile unsigned char* p = bytes_p + off;
            *p = static_cast<unsigned char>(*p + 1);
            *p = static_cast<unsigned char>(*p - 1);
        }
        if (bytes != 0) {
            volatile unsigned char* last = bytes_p + (bytes - 1);
            *last = static_cast<unsigned char>(*last + 1);
            *last = static_cast<unsigned char>(*last - 1);
        }
    }

    ~BroadcastRing() {
        TT_FATAL(
            active_readers_.load(std::memory_order_relaxed) == 0,
            "BroadcastRing readers must be destroyed before the ring");
    }

    [[nodiscard]] size_t capacity() const noexcept { return capacity_; }

    class alignas(kFalseSharingSize) Writer {
    public:
        /** @brief Publishes a single item (does not wake readers; see wake_readers()). */
        void publish(const T& item) noexcept { publish_batch({&item, 1}); }

        /**
         * @brief Publishes a batch of items (does not wake readers; see wake_readers()).
         *
         * If @p items is larger than capacity(), only its last capacity() items are retained.
         * Always sfences after NT stores before publishing head.
         */
        void publish_batch(std::span<const T> items) noexcept {
            static_assert(kStoreNoexcept, "T must be nothrow-copyable; use publish_batch_move otherwise");
            publish_impl(items);
        }

        /**
         * @brief Publishes a batch of items with std::move (does not wake readers; see wake_readers()).
         *
         * If @p items is larger than capacity(), only its last capacity() items are retained.
         */
        void publish_batch_move(std::span<T> items) noexcept
            requires std::is_move_constructible_v<T>
        {
            static_assert(kMoveStoreNoexcept, "T must be nothrow-movable");
            publish_impl(items);
        }

        /**
         * @brief Wakes readers blocked in Reader::wait().
         *
         * publish()/publish_batch() do not wake on their own; this must be called explicitly.
         */
        void wake_readers() noexcept {
            shared_state_->wake_token.fetch_add(1, std::memory_order_release);
            shared_state_->wake_token.notify_all();
        }

        Writer(const Writer&) = delete;
        Writer& operator=(const Writer&) = delete;
        Writer(Writer&&) = delete;
        Writer& operator=(Writer&&) = delete;

    private:
        friend class BroadcastRing;
        Writer(SharedState* shared_state, SlotsView view) noexcept :
            shared_state_(shared_state), view_(view), head_cache_(shared_state->head.load(std::memory_order_relaxed)) {}

        template <typename U>
        void publish_impl(std::span<U> items) {
            const size_t n = items.size();
            const uint64_t head = head_cache_;
            const SlotsView view = view_;
            SharedState* const shared_state = shared_state_;
            const size_t skip = n > view.capacity ? n - view.capacity : 0;

            // bump claim before the slot stores, so a reader we lap mid-copy sees the
            // raised claim in its recheck and drops the potentially overwritten items
            shared_state->claim.store(head + n, std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_release);
            bool streamed = false;
            if constexpr (kTriviallyCopyable && std::is_const_v<U>) {
                // Contiguous memcpy into ring slots (1 or 2 segments on wrap). Separating a compact
                // producer scratch fill from this burst is what keeps host decode from thrashing
                // against a multi-hundred-MB ring on every marker.
                static_assert(sizeof(Slot) == sizeof(T), "trivially-copyable Slot must be raw T storage");
                const T* src = items.data() + skip;
                size_t remain = n - skip;
                uint64_t pos = head + skip;
                while (remain != 0) {
                    const size_t idx = pos & (view.capacity - 1);
                    const size_t take = std::min(remain, view.capacity - idx);
                    const size_t bytes = take * sizeof(T);
                    if (bytes >= kStreamCopyMinBytes) {
                        broadcast_ring_detail::concurrent_stream_copy(
                            static_cast<void*>(&view.slots[idx]), static_cast<const void*>(src), bytes);
                        streamed = true;
                    } else {
                        broadcast_ring_detail::concurrent_copy(
                            static_cast<void*>(&view.slots[idx]), static_cast<const void*>(src), bytes);
                    }
                    src += take;
                    pos += take;
                    remain -= take;
                }
            } else {
                for (size_t k = skip; k < n; k++) {
                    if constexpr (std::is_const_v<U>) {
                        view.slot_at(head + k).store(items[k]);
                    } else {
                        view.slot_at(head + k).store(std::move(items[k]));
                    }
                }
            }
            // NT (WC) stores need sfence before readers may observe head.
            if (streamed) {
                broadcast_ring_detail::stream_fence();
            }
            head_cache_ = head + n;
            shared_state->head.store(head_cache_, std::memory_order_release);
        }

        SharedState* shared_state_;
        SlotsView view_;
        uint64_t head_cache_;
    };

    [[nodiscard]] Writer& writer() noexcept { return writer_; }

    class alignas(kFalseSharingSize) Reader {
    public:
        /**
         * @brief Reads the next available items, oldest first; non-blocking.
         *
         * Consume only the returned span. It is a view into @p out, but may be shorter and offset
         * within it; the rest of @p out holds unspecified data.
         *
         * A reader that has fallen too far behind drops its oldest unread items during this call;
         * this is tracked by dropped().
         *
         * @param out Scratch buffer the items are read into; its size bounds how many are read.
         * @return The items read, oldest first (a sub-span of @p out); empty in two cases: the reader is
         *         caught up, or it fell behind and the writer overwrote the items this call would have
         *         returned.
         */
        [[nodiscard]] std::span<T> read_batch(std::span<T> out) noexcept(kLoadNoexcept) {
            if (out.empty()) {
                return out;
            }
            const SharedState* const shared_state = shared_state_;
            const uint64_t head = shared_state->head.load(std::memory_order_acquire);
            if (cursor_ >= head) {
                return out.first(0);
            }

            const uint64_t claim_before = shared_state->claim.load(std::memory_order_relaxed);
            const SlotsView view = view_;

            const uint64_t max_lag = view.capacity - std::min<uint64_t>(writer_advance_estimate_, view.capacity >> 1);
            if (uint64_t lag = claim_before - cursor_; lag > max_lag) {
                const uint64_t drop = lag - max_lag;
                dropped_ += drop;
                cursor_ += drop;
            }

            const uint64_t start = cursor_;
            const size_t n = start < head ? std::min<uint64_t>(out.size(), head - start) : 0;
            for (size_t k = 0; k < n; k++) {
                view.slot_at(start + k).load(out[k]);
            }
            // we order the slot loads before we reload claim, so slots the writer lapped mid-copy show up as
            // claim - start > capacity below and are dropped
            std::atomic_thread_fence(std::memory_order_acquire);

            const uint64_t claim = shared_state->claim.load(std::memory_order_relaxed);
            observe_advance(claim - claim_before);
            if (claim - start > view.capacity) {
                const uint64_t oldest = claim - view.capacity;
                const uint64_t lost = oldest - start;
                dropped_ += lost;
                if (lost >= n) {
                    cursor_ = oldest;
                    return out.first(0);
                }
                cursor_ = start + n;
                return out.subspan(lost, n - lost);
            }

            cursor_ = start + n;
            return out.first(n);
        }

        /** @brief Reads one item into @p out; returns false when none is available (caught up or dropped). */
        [[nodiscard]] bool read(T& out) noexcept(kLoadNoexcept) { return !read_batch({&out, 1}).empty(); }

        using WakeToken = typename WakeTokenAtomic::value_type;

        /** @brief Snapshots the wake state to pass to wait(); take it before testing any wait condition. */
        [[nodiscard]] WakeToken wait_token() const noexcept {
            return shared_state_->wake_token.load(std::memory_order_acquire);
        }

        /**
         * @brief If no items are available to this reader, blocks until the next wake_readers() after @p since was
         * read.
         *
         * Required instead of the parameterless wait() when the reader also waits on its own condition signalled
         * through wake_readers() (e.g. a stop flag): take @p since before testing that condition, so a
         * wake_readers() between the test and the wait is not lost.
         */
        void wait(WakeToken since) const noexcept {
            if (cursor_ < shared_state_->head.load(std::memory_order_acquire)) {
                return;
            }
            for (uint32_t spin = 0; spin < kWaitSpinIterations; ++spin) {
                if (shared_state_->wake_token.load(std::memory_order_acquire) != since) {
                    return;
                }
                ttsl::pause();
            }
            shared_state_->wake_token.wait(since, std::memory_order_acquire);
        }

        /** @brief If no items are available to this reader, blocks until wake_readers() is called. */
        void wait() const noexcept { wait(wait_token()); }

        /** @brief Number of items this reader skipped after lagging too far behind; updated only during read_batch().
         */
        [[nodiscard]] uint64_t dropped() const noexcept { return dropped_; }

        Reader(const Reader&) = delete;
        Reader& operator=(const Reader&) = delete;
        Reader(Reader&& other) noexcept :
            shared_state_(std::exchange(other.shared_state_, nullptr)),
            view_(other.view_),
            cursor_(other.cursor_),
            dropped_(other.dropped_),
            writer_advance_estimate_(other.writer_advance_estimate_),
            active_readers_(std::exchange(other.active_readers_, nullptr)) {}
        Reader& operator=(Reader&& other) noexcept {
            if (this != &other) {
                release();
                shared_state_ = std::exchange(other.shared_state_, nullptr);
                view_ = other.view_;
                cursor_ = other.cursor_;
                dropped_ = other.dropped_;
                writer_advance_estimate_ = other.writer_advance_estimate_;
                active_readers_ = std::exchange(other.active_readers_, nullptr);
            }
            return *this;
        }
        ~Reader() { release(); }

    private:
        friend class BroadcastRing;
        Reader(
            const SharedState* shared_state,
            SlotsView view,
            uint64_t start,
            std::atomic<uint32_t>* active_readers) noexcept :
            shared_state_(shared_state), view_(view), cursor_(start), active_readers_(active_readers) {
            active_readers_->fetch_add(1, std::memory_order_relaxed);
        }

        void release() noexcept {
            if (active_readers_ != nullptr) {
                active_readers_->fetch_sub(1, std::memory_order_relaxed);
                active_readers_ = nullptr;
            }
        }

        // proactive dropping to avoid copying records that are likely to be discarded by the claim re-check
        static constexpr unsigned kAdvanceDecayShift = 4;     // estimate decays by ~1/16 per read
        static constexpr unsigned kAdvanceHeadroomShift = 2;  // 25% headroom on the tracked advance
        void observe_advance(uint64_t advance) noexcept {
            const uint64_t decayed = writer_advance_estimate_ - (writer_advance_estimate_ >> kAdvanceDecayShift);
            const uint64_t target = advance + (advance >> kAdvanceHeadroomShift);
            writer_advance_estimate_ = std::max(target, decayed);
        }

        const SharedState* shared_state_;
        SlotsView view_;
        uint64_t cursor_;
        uint64_t dropped_ = 0;
        // decaying max of how far the writer advances while we copy a batch, plus headroom;
        // decays so it tracks recent advances rather than pinning to the highest ever seen
        uint64_t writer_advance_estimate_ = 0;
        std::atomic<uint32_t>* active_readers_;
    };

    /** @brief Creates a reader at the current end of the stream; it sees only items published after this call. */
    [[nodiscard]] Reader make_reader() const noexcept {
        return Reader(&shared_state_, view(), shared_state_.head.load(std::memory_order_acquire), &active_readers_);
    }

private:
    // avoids futex sleeps during short publish gaps
    static constexpr uint32_t kWaitSpinIterations = 2048;

    static constexpr bool kStoreNoexcept =
        kTriviallyCopyable || (std::is_nothrow_copy_constructible_v<T> && std::is_nothrow_copy_assignable_v<T>);
    static constexpr bool kMoveStoreNoexcept =
        kTriviallyCopyable || (std::is_nothrow_move_constructible_v<T> && std::is_nothrow_move_assignable_v<T>);
    static constexpr bool kLoadNoexcept = kTriviallyCopyable || std::is_nothrow_copy_assignable_v<T>;

    // Trivially-copyable slot storage. Payload is NOT a C++ atomic object: concurrent writer/reader
    // access goes through broadcast_ring_detail::concurrent_copy (x86 asm MOVs, or atomic_ref
    // elsewhere) so the abstract machine never sees a plain data race. claim-recheck still drops
    // torn multi-unit snapshots.
    //
    // Visibility of a fully published slot: Writer::publish does a release fence then head.store(release);
    // a Reader acquire-loads that head before concurrent_copy'ing the slot.
    struct AtomicSlot {
        alignas(T) std::byte storage[sizeof(T)];

        void store(const T& v) noexcept { broadcast_ring_detail::concurrent_copy(storage, &v, sizeof(T)); }
        void store(T&& v) noexcept { store(static_cast<const T&>(v)); }

        void load(T& out) const noexcept { broadcast_ring_detail::concurrent_copy(&out, storage, sizeof(T)); }
    };

    struct LockedSlot {
        mutable std::mutex mutex;
        std::optional<T> value;

        void store(const T& v) noexcept {
            std::lock_guard lock(mutex);
            value = v;
        }
        void store(T&& v) noexcept {
            std::lock_guard lock(mutex);
            value = std::move(v);
        }
        void load(T& out) const noexcept(kLoadNoexcept) {
            std::lock_guard lock(mutex);
            out = *value;
        }
    };

    using Slot = std::conditional_t<kTriviallyCopyable, AtomicSlot, LockedSlot>;

    struct SlotsView {
        Slot* slots;
        size_t capacity;
        Slot& slot_at(uint64_t position) const noexcept { return slots[position & (capacity - 1)]; }
    };

    // head/claim are accessed together so they share a cache line; wake_token is on its own line so a
    // reader spin-waiting on it in wait() can't steal the head/claim line from the writer
    struct SharedState {
        alignas(kFalseSharingSize) std::atomic<uint64_t> head{0};  // count of fully written items a reader may consume
        std::atomic<uint64_t> claim{0};  // count the writer has started writing; always >= `head`
        alignas(kFalseSharingSize) WakeTokenAtomic wake_token{0};
    };

    SlotsView view() const noexcept { return {slots_.get(), capacity_}; }

    const size_t capacity_;
    const std::unique_ptr<Slot[]> slots_;
    SharedState shared_state_;
    mutable std::atomic<uint32_t> active_readers_{0};
    Writer writer_;
};

}  // namespace tt::tt_metal
