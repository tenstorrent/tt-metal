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
        storage_(allocate_slots(capacity_, /*construct_slots=*/true)),
        writer_(&shared_state_, view()) {}

    /**
     * @brief Constructs the ring but leaves its slots unconstructed, for a caller that will call
     *        construct_slots() from the thread that is going to write them.
     *
     * First touch decides a page's NUMA node for the life of the mapping, so constructing the slots here
     * would bind a multi-GB ring to whichever thread built it rather than the one streaming into it.
     */
    struct DeferSlotInit {};
    BroadcastRing(size_t capacity, DeferSlotInit) :
        capacity_(capacity ? std::bit_ceil(capacity) : 1),
        storage_(allocate_slots(capacity_, /*construct_slots=*/false)),
        writer_(&shared_state_, view()) {}

    /**
     * @brief The anonymous mapping backing the slots, or {nullptr, 0} when the slots are heap-backed.
     *
     * Exposed so a caller can set a NUMA policy on the pages before construct_slots() faults them, making
     * placement independent of which thread touches them first.
     */
    std::pair<void*, size_t> raw_mapping() const noexcept { return {storage_.map_base, storage_.map_bytes}; }

    /** @brief Constructs the deferred slots. Call once, before any reader or writer runs. */
    void construct_slots() noexcept {
        for (size_t i = 0; i < capacity_; i++) {
            new (storage_.slots + i) Slot();
        }
    }

    ~BroadcastRing() {
        TT_FATAL(
            active_readers_.load(std::memory_order_relaxed) == 0,
            "BroadcastRing readers must be destroyed before the ring");
#if defined(__linux__)
        if (storage_.map_base != nullptr) {
            for (size_t i = 0; i < capacity_; i++) {
                storage_.slots[i].~Slot();
            }
            ::munmap(storage_.map_base, storage_.map_bytes);
        }
#endif
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
         * @brief Current stream position: the count of items ever published. The start of a
         *        direct-emit region (see emit_reserve/emit_store/emit_commit).
         */
        [[nodiscard]] uint64_t position() const noexcept { return head_cache_; }

        /**
         * @brief Direct emit, step 1: raise the claim to @p upto before storing items into
         *        [position(), upto). May be called repeatedly with a growing bound while a region
         *        is open (e.g. a worst-case bump per input chunk); @p upto must never decrease
         *        within the region. Readers treat claimed-but-uncommitted slots as potentially
         *        overwritten, so keep the over-claim small relative to capacity.
         */
        void emit_reserve(uint64_t upto) noexcept {
            shared_state_->claim.store(upto, std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_release);
        }

        /** @brief Direct emit, step 2: store one item at @p pos, which must be below the reserved bound. */
        void emit_store(uint64_t pos, const T& item) noexcept {
#if defined(__x86_64__)
            // Non-temporal stores: the ring is written far beyond cache capacity and the writer never
            // re-reads it, so a normal store's read-for-ownership is pure waste, and emit_commit's sfence
            // orders these before the head release. They bypass the slot's atomic words, which leaves
            // tearing bounded by the claim-recheck readers already tolerate. Every writer path must stay
            // non-temporal: mixing cached and NT stores into the same lines forces a WC-buffer flush plus an
            // RFO per collision, worth ~4x on the streaming profiler decode.
            if constexpr (kTriviallyCopyable && sizeof(T) == 16 && sizeof(Slot) == 16) {
                _mm_stream_si128(
                    reinterpret_cast<__m128i*>(&view_.slot_at(pos)),
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(&item)));
                return;
            }
            // 8-byte-multiple slots (the 24 B streaming profiler record): movnti per quadword, on the
            // 8-byte alignment AtomicSlot's layout already gives.
            if constexpr (kTriviallyCopyable && sizeof(T) % 8 == 0 && sizeof(Slot) == sizeof(T)) {
                auto* q = reinterpret_cast<long long*>(&view_.slot_at(pos));
                const auto* src = reinterpret_cast<const long long*>(&item);
#pragma GCC unroll 8
                for (size_t k = 0; k < sizeof(T) / 8; k++) {
                    _mm_stream_si64(q + k, src[k]);
                }
                return;
            }
#endif
            view_.slot_at(pos).store(item);
        }

        /**
         * @brief Address of the slot at @p pos, for direct-emit bulk stores built outside the ring
         *        (same reserve/commit contract and the same non-temporal caveats as emit_store).
         */
        [[nodiscard]] void* emit_slot_ptr(uint64_t pos) noexcept { return &view_.slot_at(pos); }

        /**
         * @brief Direct emit, step 3: publish items [position(), pos) and settle the claim to the
         *        committed position. Does not wake readers; see wake_readers().
         */
        void emit_commit(uint64_t pos) noexcept {
#if defined(__x86_64__)
            _mm_sfence();
#endif
            shared_state_->head.store(pos, std::memory_order_release);
            shared_state_->claim.store(pos, std::memory_order_relaxed);
            head_cache_ = pos;
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

    // 16 B slots are 16-aligned so Writer::emit_store's non-temporal store path is usable on them.
    static constexpr size_t kSlotAlign = (kTriviallyCopyable && sizeof(T) == 16) ? 16 : alignof(std::atomic<uint64_t>);

    struct alignas(kSlotAlign) AtomicSlot {
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

    struct SlotStorage {
        Slot* slots = nullptr;
        std::unique_ptr<Slot[]> owned;
        void* map_base = nullptr;
        size_t map_bytes = 0;
    };

    // mmap-backed at every size: direct emitters stream 64 B non-temporal stores at emit_slot_ptr
    // addresses, which fault unless the base is cache-line aligned, and the new[] fallback only guarantees
    // 16 B. Large slot arrays are also walked far beyond TLB reach, so they ask for 2 MiB pages explicitly
    // (THP is madvise-opt-in on typical deployments), over-mapped by one huge page because the huge-page
    // fault path requires an aligned start.
    static SlotStorage allocate_slots(size_t n, bool construct_slots) {
        SlotStorage storage;
#if defined(__linux__)
        static constexpr size_t kHugePageSize = size_t{2} << 20;
        static constexpr size_t kHugePageMinBytes = size_t{64} << 20;
        const size_t bytes = n * sizeof(Slot);
        const bool huge = bytes >= kHugePageMinBytes;
        const size_t map_bytes = huge ? bytes + kHugePageSize : bytes;
        void* base = ::mmap(nullptr, map_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base != MAP_FAILED) {
            storage.map_base = base;
            storage.map_bytes = map_bytes;
            uintptr_t aligned = reinterpret_cast<uintptr_t>(base);
            if (huge) {
                aligned = (aligned + kHugePageSize - 1) & ~(kHugePageSize - 1);
                ::madvise(reinterpret_cast<void*>(aligned), bytes, MADV_HUGEPAGE);
            }
            storage.slots = reinterpret_cast<Slot*>(aligned);
            if (construct_slots) {
                for (size_t i = 0; i < n; i++) {
                    new (storage.slots + i) Slot();
                }
            }
            return storage;
        }
#endif
        storage.owned = std::make_unique<Slot[]>(n);
        storage.slots = storage.owned.get();
        return storage;
    }

    // head/claim are accessed together so they share a cache line; wake_token is on its own line so a
    // reader spin-waiting on it in wait() can't steal the head/claim line from the writer
    struct SharedState {
        alignas(kFalseSharingSize) std::atomic<uint64_t> head{0};  // count of fully written items a reader may consume
        std::atomic<uint64_t> claim{0};  // count the writer has started writing; always >= `head`
        alignas(kFalseSharingSize) WakeTokenAtomic wake_token{0};
    };

    SlotsView view() const noexcept { return {storage_.slots, capacity_}; }

    const size_t capacity_;
    SlotStorage storage_;
    SharedState shared_state_;
    mutable std::atomic<uint32_t> active_readers_{0};
    Writer writer_;
};

}  // namespace tt::tt_metal
