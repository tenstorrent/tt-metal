// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

// Target-specific ATen CPU selection compatibility: 512 logits, k=10.
// Preserve the comparison/partition ordering of the installed libstdc++
// introselect used by ATen TopKImpl.h. Equal values have no index tiebreak.
// Integer IEEE754 comparisons avoid soft-float arithmetic on the dataflow core.
struct RouterItem {
    uint32_t bits;
    uint32_t index;
};

inline bool router_greater(RouterItem a, RouterItem b) {
    const uint32_t aa = a.bits & 0x7fffffffu, bb = b.bits & 0x7fffffffu;
    const bool an = aa > 0x7f800000u, bn = bb > 0x7f800000u;
    if (an || bn) {
        return an && !bn;
    }
    if (aa == 0 && bb == 0) {
        return false;
    }
    if ((a.bits ^ b.bits) & 0x80000000u) {
        return !(a.bits & 0x80000000u);
    }
    return a.bits & 0x80000000u ? a.bits < b.bits : a.bits > b.bits;
}

inline void router_swap(RouterItem* q, int a, int b) {
    const RouterItem value = q[a];
    q[a] = q[b];
    q[b] = value;
}

inline void router_insertion(RouterItem* q, int first, int end) {
    for (int i = first + 1; i < end; ++i) {
        const RouterItem value = q[i];
        int slot = i;
        while (slot > first && router_greater(value, q[slot - 1])) {
            q[slot] = q[slot - 1];
            --slot;
        }
        q[slot] = value;
    }
}

inline void router_heap_adjust(RouterItem* q, int first, int slot, int count, RouterItem value) {
    const int top = slot;
    int child = slot;
    while (child < (count - 1) / 2) {
        child = 2 * (child + 1);
        if (router_greater(q[first + child], q[first + child - 1])) {
            --child;
        }
        q[first + slot] = q[first + child];
        slot = child;
    }
    if (!(count & 1) && child == (count - 2) / 2) {
        child = 2 * (child + 1);
        q[first + slot] = q[first + child - 1];
        slot = child - 1;
    }
    int parent = (slot - 1) / 2;
    while (slot > top && router_greater(q[first + parent], value)) {
        q[first + slot] = q[first + parent];
        slot = parent;
        parent = (slot - 1) / 2;
    }
    q[first + slot] = value;
}

inline void router_heap_select(RouterItem* q, int first, int end) {
    const int count = 10 - first;
    if (count > 1) {
        for (int parent = (count - 2) / 2; parent >= 0; --parent) {
            router_heap_adjust(q, first, parent, count, q[first + parent]);
        }
    }
    for (int i = 10; i < end; ++i) {
        if (router_greater(q[i], q[first])) {
            const RouterItem value = q[i];
            q[i] = q[first];
            router_heap_adjust(q, first, 0, count, value);
        }
    }
    router_swap(q, first, 9);
}

inline void router_top10(RouterItem* q, int depth = 18) {
    int first = 0, end = 512;
    while (end - first > 3) {
        if (depth == 0) {
            router_heap_select(q, first, end);
            router_insertion(q, 0, 9);
            return;
        }
        --depth;
        const int a = first + 1, b = first + (end - first) / 2, c = end - 1;
        int median;
        if (router_greater(q[a], q[b])) {
            median = router_greater(q[b], q[c]) ? b : (router_greater(q[a], q[c]) ? c : a);
        } else {
            median = router_greater(q[a], q[c]) ? a : (router_greater(q[b], q[c]) ? c : b);
        }
        router_swap(q, first, median);
        int left = first + 1, right = end;
        while (true) {
            while (router_greater(q[left], q[first])) {
                ++left;
            }
            --right;
            while (router_greater(q[first], q[right])) {
                --right;
            }
            if (left >= right) {
                break;
            }
            router_swap(q, left, right);
            ++left;
        }
        if (left <= 9) {
            first = left;
        } else {
            end = left;
        }
    }
    router_insertion(q, first, end);
    // ATen sorts [begin, begin+k-1), excluding its nth element at index9.
    router_insertion(q, 0, 9);
}
