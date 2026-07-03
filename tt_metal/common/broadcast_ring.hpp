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

namespace tt::tt_metal {

/**
 * @brief Single-producer, multi-consumer broadcast ring buffer.
 *
 * One writer publishes a stream of items; each reader observes that stream independently and in
 * order, starting from the point at which the reader is created. The writer never blocks on a reader: a
 * reader that cannot keep up loses its oldest unread items (tracked by Reader::dropped()).
 *
 * If the compile-time constant `is_always_lock_free` is true, then the ring is guaranteed to be lock-free.
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
        writer_(&shared_state_, view()) {}

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

            shared_state->claim.store(head + n, std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_release);
            for (size_t k = skip; k < n; k++) {
                if constexpr (std::is_const_v<U>) {
                    view.slot_at(head + k).store(items[k]);
                } else {
                    view.slot_at(head + k).store(std::move(items[k]));
                }
            }
            shared_state->head.store(head + n, std::memory_order_release);
            head_cache_ = head + n;
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
            // the furthest the cursor may fall behind claim; beyond it the oldest records would (likely) be
            // overwritten before we finish copying them, so we drop them now and spend the copies on records
            // we can actually keep
            const auto max_lag = [this](uint64_t want, uint64_t cap) {
                const uint64_t margin = std::min<uint64_t>(std::max<uint64_t>(skip_margin_, want), cap >> 1);
                return std::max<uint64_t>(want, cap - margin);
            };
            // running max of how far the writer advances while we copy a batch, plus 25% headroom; decays
            // slowly so it tracks recent advances rather than pinning to the highest ever seen
            const auto update_skip_margin = [this](uint64_t advance) {
                const uint64_t decayed = skip_margin_ - (skip_margin_ >> 6);
                const uint64_t target = advance + (advance >> 2);
                skip_margin_ = target > decayed ? target : decayed;
            };

            const SharedState* const shared_state = shared_state_;
            const uint64_t head = shared_state->head.load(std::memory_order_acquire);
            if (cursor_ >= head) {
                return out.first(0);
            }

            const uint64_t claim_before = shared_state->claim.load(std::memory_order_relaxed);
            const SlotsView view = view_;
            const uint64_t want = std::min<uint64_t>(out.size(), view.capacity);
            const uint64_t lag_limit = max_lag(want, view.capacity);
            if (claim_before - cursor_ > lag_limit) {
                dropped_ += (claim_before - cursor_) - lag_limit;
                cursor_ = claim_before - lag_limit;
            }

            const uint64_t start = cursor_;
            const size_t n = start < head ? std::min<uint64_t>(out.size(), head - start) : 0;
            for (size_t k = 0; k < n; k++) {
                view.slot_at(start + k).load(out[k]);
            }
            std::atomic_thread_fence(std::memory_order_acquire);

            const uint64_t claim = shared_state->claim.load(std::memory_order_relaxed);
            update_skip_margin(claim - claim_before);
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
            skip_margin_(other.skip_margin_),
            active_readers_(std::exchange(other.active_readers_, nullptr)) {}
        Reader& operator=(Reader&& other) noexcept {
            if (this != &other) {
                release();
                shared_state_ = std::exchange(other.shared_state_, nullptr);
                view_ = other.view_;
                cursor_ = other.cursor_;
                dropped_ = other.dropped_;
                skip_margin_ = other.skip_margin_;
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

        const SharedState* shared_state_;
        SlotsView view_;
        uint64_t cursor_;
        uint64_t dropped_ = 0;
        uint64_t skip_margin_ = 0;
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

    struct AtomicSlot {
        static constexpr size_t kWordCount = (sizeof(T) + sizeof(uint64_t) - 1) / sizeof(uint64_t);

        std::array<std::atomic<uint64_t>, kWordCount> words;

        void store(const T& v) noexcept {
            const std::byte* src = reinterpret_cast<const std::byte*>(&v);
#pragma GCC unroll 8
            for (size_t k = 0; k < kWordCount; k++) {
                uint64_t w = 0;
                std::memcpy(&w, src + k * sizeof(uint64_t), word_bytes(k));
                words[k].store(w, std::memory_order_relaxed);
            }
        }
        void store(T&& v) noexcept { store(static_cast<const T&>(v)); }

        void load(T& out) const noexcept {
            std::byte* dst = reinterpret_cast<std::byte*>(&out);
#pragma GCC unroll 8
            for (size_t k = 0; k < kWordCount; k++) {
                const uint64_t w = words[k].load(std::memory_order_relaxed);
                std::memcpy(dst + k * sizeof(uint64_t), &w, word_bytes(k));
            }
        }

        static constexpr size_t word_bytes(size_t k) noexcept {
            return std::min(sizeof(uint64_t), sizeof(T) - k * sizeof(uint64_t));
        }
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
