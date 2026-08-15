// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>

namespace tt::tt_metal {

/**
 * @brief Fixed-capacity single-threaded FIFO ring buffer.
 *
 * Elements are appended at the back and consumed from the front; when the buffer is full,
 * push_back()/emplace_back() overwrite the oldest element. Indexing via operator[] is oldest-first.
 *
 * Not thread-safe: all access must come from one thread (or be externally synchronized).
 *
 * @tparam T Element type; must be default-constructible, assignable, and nothrow-destructible.
 */
template <typename T>
class RingBuffer {
    static_assert(std::is_default_constructible_v<T> && std::is_nothrow_destructible_v<T>);

public:
    /**
     * @brief Constructs an empty ring buffer with at least @p capacity slots.
     * @param capacity Requested slot count; gets rounded up to the next power of two.
     */
    explicit RingBuffer(size_t capacity) : slots_(std::bit_ceil(capacity)) {}

    /** @brief Appends @p item; overwrites the oldest element when full. */
    template <typename U = T>
        requires std::is_convertible_v<U&&, T> && std::is_assignable_v<T&, U&&>
    void push_back(U&& item) noexcept(std::is_nothrow_assignable_v<T&, U&&>) {
        if (full()) {
            ++head_;
        }
        slot_at(tail_) = std::forward<U>(item);
        ++tail_;
    }

    /**
     * @brief Appends a T constructed from @p args; overwrites the oldest element when full.
     * @return Reference to the appended element.
     */
    template <typename... Args>
    T& emplace_back(Args&&... args) noexcept(
        std::is_nothrow_constructible_v<T, Args...> && std::is_nothrow_move_assignable_v<T>) {
        push_back(T(std::forward<Args>(args)...));
        return back();
    }

    /**
     * @brief Removes and returns the oldest element.
     * @pre !empty()
     */
    T pop_front() noexcept(std::is_nothrow_move_constructible_v<T>) {
        TT_ASSERT(!empty());
        T item = std::move(slot_at(head_));
        ++head_;
        return item;
    }

    /**
     * @brief Oldest element.
     * @pre !empty()
     */
    [[nodiscard]] T& front() noexcept {
        TT_ASSERT(!empty());
        return slot_at(head_);
    }
    [[nodiscard]] const T& front() const noexcept {
        TT_ASSERT(!empty());
        return slot_at(head_);
    }

    /**
     * @brief Newest element.
     * @pre !empty()
     */
    [[nodiscard]] T& back() noexcept {
        TT_ASSERT(!empty());
        return slot_at(tail_ - 1);
    }
    [[nodiscard]] const T& back() const noexcept {
        TT_ASSERT(!empty());
        return slot_at(tail_ - 1);
    }

    /**
     * @brief Element @p index positions after the oldest.
     * @pre index < size()
     */
    [[nodiscard]] T& operator[](size_t index) noexcept {
        TT_ASSERT(index < size());
        return slot_at(head_ + index);
    }
    [[nodiscard]] const T& operator[](size_t index) const noexcept {
        TT_ASSERT(index < size());
        return slot_at(head_ + index);
    }

    /** @brief Empties the buffer. */
    void clear() {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (T& slot : slots_) {
                slot = T{};
            }
        }
        head_ = tail_ = 0;
    }

    [[nodiscard]] size_t size() const noexcept { return tail_ - head_; }
    [[nodiscard]] size_t capacity() const noexcept { return slots_.size(); }
    [[nodiscard]] bool empty() const noexcept { return head_ == tail_; }
    [[nodiscard]] bool full() const noexcept { return size() == capacity(); }

private:
    T& slot_at(uint64_t position) noexcept { return slots_[position & (slots_.size() - 1)]; }
    const T& slot_at(uint64_t position) const noexcept { return slots_[position & (slots_.size() - 1)]; }

    std::vector<T> slots_;
    uint64_t head_ = 0;
    uint64_t tail_ = 0;
};

}  // namespace tt::tt_metal
