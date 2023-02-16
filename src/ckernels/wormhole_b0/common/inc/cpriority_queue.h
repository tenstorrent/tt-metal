#pragma once

#include <cstdint>
#include <utility>
#include "fw_debug.h"

// Provides a priority queue where lowest priority value has highest priority (e.g. priority 0 is higher priority than 5)
// Can be reversed with REVERSE_PRIORITY (e.g. priority 5 will be higher priority than 0)
// Also provides a version that allocates memory for you (such as on the stack), see below

template <bool REVERSE_PRIORITY=false>
class FixedSizePriorityQueue 
{
    protected:
    std::pair<uint32_t, uint32_t> *heap;
    uint32_t num_elem;
    uint32_t max_size;

    public:
    FixedSizePriorityQueue(uint32_t addr_, uint32_t max_size_) 
    {
        heap = (std::pair<uint32_t, uint32_t> *) addr_;
        num_elem = 0;
        max_size = max_size_;
    }

    void push(uint32_t value, uint32_t priority)
    {
        push(std::make_pair(value, priority));
    }

    void push(std::pair<uint32_t, uint32_t> value)
    {
        FWASSERT("You are trying to push a full priority queue.", !is_full());

        heap[num_elem] = value;

        num_elem++;
        bubble_up(num_elem - 1);
    }

    std::pair<uint32_t, uint32_t> pop()
    {
        FWASSERT("You are trying to pop an empty priority queue.", !is_empty());

        // Swap first with last
        auto first_elem = heap[0];
        heap[0] = heap[num_elem - 1];
        heap[num_elem - 1] = first_elem;

        num_elem--;
        bubble_down(0);

        return first_elem;
    }

    __attribute__((always_inline)) 
    inline const std::pair<uint32_t, uint32_t>& top() const
    {
        FWASSERT("You are trying to view an empty priority queue.", !is_empty());

        return heap[0];
    }

    __attribute__((always_inline)) 
    inline const uint32_t size() const
    {
        return num_elem;
    }

    __attribute__((always_inline)) 
    inline const bool is_empty() const
    {
        return size() == 0;
    }

    __attribute__((always_inline)) 
    inline const bool is_full() const
    {
        return size() == max_size;
    }

    protected:

    void bubble_up(uint32_t idx)
    {
        if (idx == 0)
            return;

        uint32_t parent = ((idx + 1) >> 1) - 1;

        if ((REVERSE_PRIORITY && (heap[parent].second < heap[idx].second)) ||
            (!REVERSE_PRIORITY && (heap[parent].second > heap[idx].second))) {
            // swap
            auto tmp = heap[idx];
            heap[idx] = heap[parent];
            heap[parent] = tmp;

            bubble_up(parent);
        }
    }

    void bubble_down(uint32_t idx)
    {
        uint32_t left = ((idx + 1) << 1) - 1;
        uint32_t right = ((idx + 1) << 1);
        uint32_t higher_priority = idx;

        if (left < size()) {
            if ((REVERSE_PRIORITY && (heap[left].second > heap[higher_priority].second)) ||
                (!REVERSE_PRIORITY && (heap[left].second < heap[higher_priority].second))) {
                higher_priority = left;
            }
        }

        if (right < size()) {
            if ((REVERSE_PRIORITY && (heap[right].second > heap[higher_priority].second)) ||
                (!REVERSE_PRIORITY && (heap[right].second < heap[higher_priority].second))) {
                higher_priority = right;
            }
        }

        if (higher_priority != idx) {
            // swap
            auto tmp = heap[idx];
            heap[idx] = heap[higher_priority];
            heap[higher_priority] = tmp;

            bubble_down(higher_priority);
        }
    }

};

