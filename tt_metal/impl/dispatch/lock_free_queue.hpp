// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include "tt_metal/common/assert.hpp"

/*
    Supports single writer, single reader
*/
template<typename T>
class LockFreeQueue {
    private:
        struct Node {
            std::shared_ptr<T> data = nullptr;
            Node* next = nullptr;
        };

        std::atomic<Node*> head;
        std::atomic<Node*> tail;

        std::mutex queue_mutex;
        std::function<void()> lock_func = [](){};
        std::function<void()> unlock_func = [](){};

        inline Node* pop_head() {
            Node* oldHead = head.load();
            if (oldHead == tail.load()) {
                return nullptr; // Queue is empty
            }
            head.store(oldHead->next);
            return oldHead;
        }
        // Statically allocated ring buffer containing
        // node objects, which contain handles to data
        // and another node object to traverse ring buffer.
        const static uint32_t ring_buffer_size = 8192;
        Node ring_buffer[ring_buffer_size];

    public:
        // Optional - Set these if the worker and parent thread state needs to be tracked
        std::atomic<std::thread::id> worker_thread_id;
        std::atomic<std::thread::id> parent_thread_id;
        LockFreeQueue()
        {
            // Initialize ring buffer for traversal. Each node points to the subsequent node, except for the last one, which points to the head.
            for (int node_idx = 0; node_idx < ring_buffer_size; node_idx++) {
                (node_idx < ring_buffer_size - 1) ? ring_buffer[node_idx].next = (&ring_buffer[node_idx + 1]) : ring_buffer[node_idx].next = &(ring_buffer[0]);
            }
            // Initialize head and tail ptrs to start of ring buffer.
            this->head = ring_buffer;
            this->tail = ring_buffer;
        }

        LockFreeQueue(LockFreeQueue&& other) {
            Node ring_buffer = other.ring_buffer;
            head.store(other.head.load());
            tail.store(other.tail.load());
            worker_thread_id.store(other.worker_thread_id.load());
            parent_thread_id.store(other.parent_thread_id.load());
        }

        inline void push(const T& value) {
            // Legacy Push API allowing copy by value
            // for object T.

            // Stall condition: this push will update the tail (wptr)
            // to match the location of head (rptr). The current push can
            // thus overwrite data that's being read. Stall until head
            // has progressed (data has been read).
            lock_func();
            while(tail.load()->next == head.load()) {};
            tail.load()->data = std::make_shared<T>(value);
            tail.store(tail.load()->next);
            unlock_func();
        }

        inline void push(std::shared_ptr<T> value) {
            // Latest Push API, passing ptrs around.
            // Usually faster, since no data-copies.

            // Stall condition: this push will update the tail (wptr)
            // to match the location of head (rptr). The current push can
            // thus overwrite data that's being read. Stall until head
            // has progressed (data has been read).
            lock_func();
            while(tail.load()->next == head.load()) {};
            tail.load()->data = value;
            tail.store(tail.load()->next);
            unlock_func();
        }

        inline std::shared_ptr<T> pop() {
            Node* oldHead = pop_head();
            std::shared_ptr<T> result(oldHead->data);
            // Does not actually delete oldHead->data.
            // Just mark is to null to mark prev node as empty.
            (oldHead->data).reset();
            return result;
        }

        void clear() {
            while (!empty()) {
                void(pop());
            }
        }

        bool empty() const {
            return head.load() == tail.load();
        }
        void set_lock_free() {
            lock_func = [](){};
            unlock_func = [](){};
        }

        void set_lock_based() {
            lock_func = [this](){this->queue_mutex.lock();};
            unlock_func = [this](){this->queue_mutex.unlock();};
        }

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
            const T& operator*() const { return *(current->data); }
        };

        Iterator begin() { return Iterator(head.load()); }
        Iterator end() { return Iterator(tail.load()); }
};
