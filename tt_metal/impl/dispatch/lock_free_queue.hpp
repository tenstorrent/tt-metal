// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <memory>

template<typename T>
class LockFreeQueue {
    private:
        struct Node {
            std::shared_ptr<T> data;
            Node* next;
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
        LockFreeQueue() : head(new Node), tail(head.load()) {}

        void push(const T& value) {
            std::shared_ptr<T> newData(std::make_shared<T>(value));
            Node* newNode = new Node;
            newNode->data = newData;
            newNode->next = nullptr;
            Node* oldTail = tail.exchange(newNode);
            oldTail->next = newNode;
        }

        std::shared_ptr<T> pop() {
            Node* oldHead = pop_head();
            if (!oldHead) {
                throw std::runtime_error("Queue is empty");
            }
            std::shared_ptr<T> result(oldHead->data);
            delete oldHead;
            return result;
        }

        bool empty() const {
            return head.load() == tail.load();
        }
};
