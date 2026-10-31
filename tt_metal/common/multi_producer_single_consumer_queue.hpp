// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <utility>
#include <tt_stl/assert.hpp>
#include <tt_stl/tt_pause.hpp>

template <typename T>
class MultiProducerSingleConsumerQueue {
private:
    struct Node {
        T data{};
        Node* next = nullptr;
    };

    std::atomic<Node*> head;
    std::atomic<Node*> tail;

    std::mutex queue_mutex;

    Node* pop_head() {
        Node* oldHead = head.load();
        if (oldHead == tail.load()) {
            return nullptr;  // Queue is empty
        }
        head.store(oldHead->next);
        return oldHead;
    }
    // Statically allocated ring buffer containing
    // node objects, which contain the data
    // and another node object to traverse ring buffer.
    const static uint32_t ring_buffer_size = 8192;
    Node ring_buffer[ring_buffer_size];

public:
    // Optional - Set these if the worker and parent thread state needs to be tracked
    std::atomic<std::thread::id> worker_thread_id;
    std::atomic<std::thread::id> parent_thread_id;

    MultiProducerSingleConsumerQueue() {
        // Initialize ring buffer for traversal. Each node points to the subsequent node, except for the last one, which
        // points to the head.
        for (int node_idx = 0; node_idx < ring_buffer_size; node_idx++) {
            (node_idx < ring_buffer_size - 1) ? ring_buffer[node_idx].next = (&ring_buffer[node_idx + 1])
                                              : ring_buffer[node_idx].next = &(ring_buffer[0]);
        }
        // Initialize head and tail ptrs to start of ring buffer.
        this->head = ring_buffer;
        this->tail = ring_buffer;
    }

    MultiProducerSingleConsumerQueue(MultiProducerSingleConsumerQueue&& other) = delete;
    MultiProducerSingleConsumerQueue& operator=(MultiProducerSingleConsumerQueue&& other) = delete;
    MultiProducerSingleConsumerQueue(const MultiProducerSingleConsumerQueue& other) = delete;
    MultiProducerSingleConsumerQueue& operator=(const MultiProducerSingleConsumerQueue& other) = delete;

    // Stored in place, so a push does not allocate.
    void push(T value) {
        // Stall condition: this push will update the tail (wptr)
        // to match the location of head (rptr). The current push can
        // thus overwrite data that's being read. Stall until head
        // has progressed (data has been read).
        std::unique_lock<std::mutex> lock(this->queue_mutex);
        ttsl::nice_spin_until([this] { return tail.load()->next != head.load(); });
        tail.load()->data = std::move(value);
        tail.store(tail.load()->next);
    }

    T pop() {
        // Precondition: this non-blocking pop must only be called when the
        // single consumer knows an entry is available.
        Node* oldHead = pop_head();
        TT_FATAL(oldHead != nullptr, "Cannot pop from an empty MultiProducerSingleConsumerQueue.");
        return std::exchange(oldHead->data, T{});
    }

    void clear() {
        while (!empty()) {
            (void)pop();
        }
    }

    bool empty() const { return head.load() == tail.load(); }

    class Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

    private:
        Node* current;

    public:
        // Constructor initializes the iterator with a pointer to a Node
        Iterator(Node* start) : current(start) {}

        // Prefix increment operator overloading
        Iterator& operator++() {
            if (current != nullptr) {
                current = current->next;
            }
            return *this;
        }

        // Inequality operator overloading
        bool operator!=(const Iterator& other) const { return current != other.current; }

        // Dereference operator overloading
        const T& operator*() const { return current->data; }
    };

    Iterator begin() { return Iterator(head.load()); }
    Iterator end() { return Iterator(tail.load()); }
};
