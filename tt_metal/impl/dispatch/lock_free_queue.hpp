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

        Node* pop_head() {
            Node* oldHead = head.load();
            if (oldHead == tail.load()) {
                return nullptr; // Queue is empty
            }
            head.store(oldHead->next);
            return oldHead;
        }

    public:
        // Optional - Set these if the worker and parent thread state needs to be tracked
        std::atomic<uint64_t> worker_thread_id = 0;
        std::atomic<uint64_t> parent_thread_id = 0;
        LockFreeQueue() : head(new Node), tail(head.load()) {}
        LockFreeQueue(LockFreeQueue&& other) {
            head.store(other.head.load());
            tail.store(other.tail.load());
            worker_thread_id.store(other.worker_thread_id.load());
            parent_thread_id.store(other.parent_thread_id.load());
        }
        void push(const T& value) {
            std::shared_ptr<T> newData(std::make_shared<T>(value));
            Node* newNode = new Node;
            tail.load()->data = newData;
            tail.load()->next = newNode;
            tail.store(newNode);
        }

        std::shared_ptr<T> pop() {
            Node* oldHead = pop_head();
            if (!oldHead) {
                TT_THROW("Queue is empty");
            }
            std::shared_ptr<T> result(oldHead->data);
            delete oldHead;
            return result;
        }

        bool empty() const {
            return head.load() == tail.load();
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
