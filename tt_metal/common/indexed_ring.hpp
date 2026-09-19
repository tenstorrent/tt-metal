// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>

namespace tt::tt_metal {

/**
 * @brief Single-writer, multi-reader ring of the newest items, read by their append index without locks.
 *
 * One writer appends items, which get consecutive indices; any thread reads any retained item by index. Items live
 * in fixed chunks reached through a table, so an append never moves an item. A full ring retires its oldest chunk
 * and reuses it for the next items; the retained items are [first(), count()), and both only grow.
 *
 * A read is lock-free and never blocks the writer: it copies the item and then checks the chunk's sequence
 * counter, which the writer bumps when it reuses the chunk, so a copy that overlapped the reuse fails instead of
 * returning a torn or replaced item. A failed read means the index is no longer retained (or is not yet); the
 * caller re-reads first() and count() and decides what to do with an index outside them.
 *
 * The writer's operations (push, clear) are driven by one thread. T must be trivially copyable; it is stored as
 * relaxed atomic words, so a read racing a reuse is defined behaviour and caught by the sequence check.
 *
 * @tparam T Element type.
 */
template <typename T>
class IndexedRing {
    static_assert(std::is_trivially_copyable_v<T>, "IndexedRing items are copied as atomic words");

public:
    static constexpr uint64_t kChunkItems = 4096;

    /**
     * @brief Constructs a log retaining at most @p capacity items.
     * @param capacity Rounded down to whole chunks, at least one chunk. The chunk table is allocated at the first
     *        push and chunks as items reach them, so an unused ring costs nothing but this object.
     */
    explicit IndexedRing(uint64_t capacity) : chunks_(std::max<uint64_t>(1, capacity / kChunkItems)) {}

    ~IndexedRing() {
        if (std::atomic<Chunk*>* t = table_.load(std::memory_order_relaxed); t != nullptr) {
            for (uint64_t k = 0; k < chunks_; k++) {
                delete t[k].load(std::memory_order_relaxed);
            }
            delete[] t;
        }
    }
    IndexedRing(const IndexedRing&) = delete;
    IndexedRing& operator=(const IndexedRing&) = delete;

    /** @brief The most items retained at once: whole chunks. */
    [[nodiscard]] uint64_t capacity() const noexcept { return chunks_ * kChunkItems; }

    /** @brief Index of the oldest retained item. */
    [[nodiscard]] uint64_t first() const noexcept { return first_.load(std::memory_order_acquire); }

    /** @brief Index past the newest item; the next push gets this index. */
    [[nodiscard]] uint64_t count() const noexcept { return count_.load(std::memory_order_acquire); }

    /**
     * @brief Appends an item at index count(). When the log is full, the oldest chunk is retired first: first()
     * moves up by one chunk and that chunk takes the new items.
     */
    void push(const T& item) noexcept {
        const uint64_t n = count_.load(std::memory_order_relaxed);
        const uint64_t f = first_.load(std::memory_order_relaxed);
        std::atomic<Chunk*>* table = table_.load(std::memory_order_relaxed);
        if (table == nullptr) {
            table = new std::atomic<Chunk*>[chunks_]();
            table_.store(table, std::memory_order_release);
        }
        Chunk* c = table[slot_of(n)].load(std::memory_order_relaxed);
        if (n - f == capacity()) {
            // n is at a chunk boundary (first and the capacity are whole chunks): this chunk holds the oldest
            // items. Readers that copied one of them across the bump see the changed sequence and fail.
            first_.store(f + kChunkItems, std::memory_order_release);
            c->seq.fetch_add(1, std::memory_order_release);
            std::atomic_thread_fence(std::memory_order_release);
        } else if (c == nullptr) {
            c = new Chunk;
            table[slot_of(n)].store(c, std::memory_order_release);
        }
        c->slots[n % kChunkItems].store(item);
        count_.store(n + 1, std::memory_order_release);
    }

    /**
     * @brief Retires every item; indices keep growing from the next chunk boundary, so no retired index is ever
     * reused as a retained one.
     */
    void clear() noexcept {
        const uint64_t n = count_.load(std::memory_order_relaxed);
        const uint64_t aligned = (n + kChunkItems - 1) / kChunkItems * kChunkItems;
        first_.store(aligned, std::memory_order_release);
        count_.store(aligned, std::memory_order_release);
        // The next pushes overwrite retired items at once, with no chunk boundary to bump at: fail every read in
        // flight instead.
        if (std::atomic<Chunk*>* table = table_.load(std::memory_order_relaxed); table != nullptr) {
            for (uint64_t k = 0; k < chunks_; k++) {
                if (Chunk* c = table[k].load(std::memory_order_relaxed); c != nullptr) {
                    c->seq.fetch_add(1, std::memory_order_release);
                }
            }
        }
        std::atomic_thread_fence(std::memory_order_release);
    }

    /**
     * @brief Copies item @p i into @p out.
     * @return False when @p i is not retained: before first(), at or past count(), or in a chunk the writer reused
     *         while it was being copied.
     */
    [[nodiscard]] bool read(uint64_t i, T& out) const noexcept {
        const std::atomic<Chunk*>* table = table_.load(std::memory_order_acquire);
        if (table == nullptr) {
            return false;
        }
        const Chunk* c = table[slot_of(i)].load(std::memory_order_acquire);
        if (c == nullptr) {
            return false;
        }
        // The sequence first: a reuse stores first() before bumping it, so a reader that sees the new sequence
        // also sees the new first() and rejects the index below.
        const uint64_t seq = c->seq.load(std::memory_order_acquire);
        if (i < first_.load(std::memory_order_relaxed) || i >= count_.load(std::memory_order_acquire)) {
            return false;
        }
        c->slots[i % kChunkItems].load(out);
        std::atomic_thread_fence(std::memory_order_acquire);
        return c->seq.load(std::memory_order_relaxed) == seq;
    }

private:
    struct Slot {
        static constexpr size_t kWordCount = (sizeof(T) + sizeof(uint64_t) - 1) / sizeof(uint64_t);
        std::array<std::atomic<uint64_t>, kWordCount> words;

        void store(const T& v) noexcept {
            const std::byte* src = reinterpret_cast<const std::byte*>(&v);
            for (size_t k = 0; k < kWordCount; k++) {
                uint64_t w = 0;
                std::memcpy(&w, src + k * sizeof(uint64_t), word_bytes(k));
                words[k].store(w, std::memory_order_relaxed);
            }
        }
        void load(T& out) const noexcept {
            std::byte* dst = reinterpret_cast<std::byte*>(&out);
            for (size_t k = 0; k < kWordCount; k++) {
                const uint64_t w = words[k].load(std::memory_order_relaxed);
                std::memcpy(dst + k * sizeof(uint64_t), &w, word_bytes(k));
            }
        }
        static constexpr size_t word_bytes(size_t k) noexcept {
            return std::min(sizeof(uint64_t), sizeof(T) - k * sizeof(uint64_t));
        }
    };
    struct Chunk {
        alignas(64) std::atomic<uint64_t> seq{0};
        std::array<Slot, kChunkItems> slots;
    };

    uint64_t slot_of(uint64_t i) const noexcept { return (i / kChunkItems) % chunks_; }

    const uint64_t chunks_;
    std::atomic<std::atomic<Chunk*>*> table_{nullptr};
    alignas(64) std::atomic<uint64_t> first_{0};
    alignas(64) std::atomic<uint64_t> count_{0};
};

}  // namespace tt::tt_metal
