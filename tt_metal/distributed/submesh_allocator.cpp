// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "submesh_allocator.hpp"

#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace tt::tt_metal::distributed {

namespace {

struct AddressRange {
    unsigned long long start;
    unsigned long long size;
    unsigned long long end() const { return start + size; }
};

class LinearAllocatorView {
public:
    LinearAllocatorView(unsigned long long capacity_bytes, unsigned long long alignment_bytes) :
        capacity_(capacity_bytes), alignment_(alignment_bytes) {
        free_.insert({0, capacity_bytes});
    }

    bool allocate_at(unsigned long long start, unsigned long long size) {
        start = align_down(start);
        size = align_up(size);
        for (auto it = free_.begin(); it != free_.end(); ++it) {
            const auto range_start = it->first;
            const auto range_size = it->second;
            const unsigned long long range_end = range_start + range_size;
            const unsigned long long end = start + size;
            if (start >= range_start && end <= range_end) {
                free_.erase(it);
                if (range_start < start) {
                    free_.insert({range_start, start - range_start});
                }
                if (end < range_end) {
                    free_.insert({end, range_end - end});
                }
                used_.insert({start, size});
                return true;
            }
        }
        return false;
    }

    std::optional<unsigned long long> first_fit(unsigned long long size) const {
        size = align_up(size);
        for (const auto& kv : free_) {
            const auto start = kv.first;
            const auto free_size = kv.second;
            const unsigned long long aligned_start = align_up(start);
            const unsigned long long padding = aligned_start - start;
            if (free_size >= padding + size) {
                return aligned_start;
            }
        }
        return std::nullopt;
    }

    std::optional<unsigned long long> first_fit_excluding(
        const std::vector<AddressRange>& ex, unsigned long long size) const {
        size = align_up(size);
        for (const auto& kv : free_) {
            const unsigned long long range_start = kv.first;
            const unsigned long long range_size = kv.second;
            unsigned long long cursor = align_up(range_start);
            const unsigned long long range_end = range_start + range_size;
            while (cursor + size <= range_end) {
                bool blocked = false;
                unsigned long long next_cursor = cursor;
                for (const auto& e : ex) {
                    const bool intersects = !((cursor + size) <= e.start || cursor >= e.end());
                    if (intersects) {
                        blocked = true;
                        const unsigned long long aligned = align_up(e.end());
                        if (next_cursor < aligned) {
                            next_cursor = aligned;
                        }
                    }
                }
                if (!blocked) {
                    return cursor;
                }
                if (next_cursor <= cursor) {
                    next_cursor = cursor + alignment_;
                }
                cursor = next_cursor;
            }
        }
        return std::nullopt;
    }

    void free(unsigned long long start) {
        auto it = used_.find(start);
        if (it == used_.end()) {
            return;
        }
        const unsigned long long size = it->second;
        used_.erase(it);
        coalesce_insert_free({start, size});
    }

    bool is_free(unsigned long long start, unsigned long long size) const {
        size = align_up(size);
        for (const auto& kv : free_) {
            const auto range_start = kv.first;
            const auto range_size = kv.second;
            if (start >= range_start && start + size <= range_start + range_size) {
                return true;
            }
        }
        return false;
    }

    std::vector<AddressRange> used_ranges() const {
        std::vector<AddressRange> v;
        v.reserve(used_.size());
        for (const auto& kv : used_) {
            v.push_back({kv.first, kv.second});
        }
        return v;
    }

    unsigned long long alignment() const { return alignment_; }

private:
    unsigned long long align_up(unsigned long long v) const { return (v + alignment_ - 1) / alignment_ * alignment_; }
    unsigned long long align_down(unsigned long long v) const { return v / alignment_ * alignment_; }

    void coalesce_insert_free(AddressRange r) {
        unsigned long long start = r.start;
        unsigned long long end = r.end();
        for (auto it = free_.begin(); it != free_.end();) {
            const unsigned long long s = it->first;
            const unsigned long long e = s + it->second;
            if (!(end < s || start > e)) {
                start = (start < s) ? start : s;
                end = (end > e) ? end : e;
                it = free_.erase(it);
            } else {
                ++it;
            }
        }
        free_.insert({start, end - start});
    }

    unsigned long long capacity_;
    unsigned long long alignment_;
    std::map<unsigned long long, unsigned long long> free_;
    std::map<unsigned long long, unsigned long long> used_;
};

static bool intersects_any(unsigned long long start, unsigned long long size, const std::vector<AddressRange>& rs) {
    const unsigned long long end = start + size;
    for (const auto& r : rs) {
        if (!(end <= r.start || start >= r.end())) {
            return true;
        }
    }
    return false;
}

}  // namespace

struct SubmeshAllocator::Impl {
    explicit Impl(unsigned int n, unsigned long long cap, unsigned long long align) :
        pools(n, LinearAllocatorView(cap, align)), alignment(align), parents(n), children(n) {}

    bool allocate_internal(PoolId pool, unsigned long long size, AllocationHandle& out) {
        size = align_up(size);
        const auto excl = compute_exclusions_for_pool(pool);
        auto cand = pools[pool].first_fit_excluding(excl, size);
        while (cand.has_value()) {
            const auto addr = *cand;
            if (ancestors_are_free(pool, addr, size) && descendants_are_free(pool, addr, size)) {
                if (pools[pool].allocate_at(addr, size)) {
                    out = AllocationHandle{pool, addr, size};
                    return true;
                }
            }
            cand = first_fit_after(pool, addr, size, excl);
        }
        return false;
    }

    bool allocate_at_internal(PoolId pool, unsigned long long start, unsigned long long size, AllocationHandle& out) {
        start = align_up(start);
        size = align_up(size);
        const auto excl = compute_exclusions_for_pool(pool);
        if (intersects_any(start, size, excl)) {
            return false;
        }
        if (!ancestors_are_free(pool, start, size)) {
            return false;
        }
        if (!descendants_are_free(pool, start, size)) {
            return false;
        }
        if (!pools[pool].allocate_at(start, size)) {
            return false;
        }
        out = AllocationHandle{pool, start, size};
        return true;
    }

    void free_internal(const AllocationHandle& h) { pools[h.pool].free(h.start); }

    std::optional<unsigned long long> first_fit_after(
        PoolId pool,
        unsigned long long prev_addr,
        unsigned long long size,
        const std::vector<AddressRange>& excl) const {
        unsigned long long next_cursor = prev_addr + pools[pool].alignment();
        std::vector<AddressRange> ex = excl;
        ex.push_back({0, next_cursor});
        return pools[pool].first_fit_excluding(ex, size);
    }

    std::vector<AddressRange> compute_exclusions_for_pool(PoolId pool) const {
        // Pool cannot use addresses occupied by any of its ancestors. If multiple ancestors share a start,
        // use the max size at that start.
        std::map<unsigned long long, unsigned long long> max_by_start;
        std::set<PoolId> seen;
        std::vector<PoolId> stack;
        stack.push_back(pool);
        // DFS up to ancestors
        while (!stack.empty()) {
            PoolId cur = stack.back();
            stack.pop_back();
            for (auto p : parents[cur]) {
                if (seen.insert(p).second) {
                    for (const auto& r : pools[p].used_ranges()) {
                        auto& sz = max_by_start[r.start];
                        sz = (sz > r.size) ? sz : r.size;
                    }
                    stack.push_back(p);
                }
            }
        }
        std::vector<AddressRange> ex;
        ex.reserve(max_by_start.size());
        for (const auto& kv : max_by_start) {
            ex.push_back({kv.first, kv.second});
        }
        return ex;
    }

    bool ancestors_are_free(PoolId pool, unsigned long long start, unsigned long long size) const {
        std::set<PoolId> visited;
        std::vector<PoolId> stack;
        stack.push_back(pool);
        while (!stack.empty()) {
            PoolId cur = stack.back();
            stack.pop_back();
            for (auto p : parents[cur]) {
                if (!visited.insert(p).second) {
                    continue;
                }
                if (!pools[p].is_free(start, size)) {
                    return false;
                }
                stack.push_back(p);
            }
        }
        return true;
    }

    bool descendants_are_free(PoolId pool, unsigned long long start, unsigned long long size) const {
        std::set<PoolId> visited;
        std::vector<PoolId> queue;
        queue.push_back(pool);
        while (!queue.empty()) {
            PoolId cur = queue.back();
            queue.pop_back();
            for (auto c : children[cur]) {
                if (!visited.insert(c).second) {
                    continue;
                }
                if (!pools[c].is_free(start, size)) {
                    return false;
                }
                queue.push_back(c);
            }
        }
        return true;
    }

    static unsigned long long align_up(unsigned long long v, unsigned long long a) { return (v + a - 1) / a * a; }
    unsigned long long align_up(unsigned long long v) const { return align_up(v, alignment); }

    std::vector<LinearAllocatorView> pools;
    unsigned long long alignment;
    std::vector<std::vector<PoolId>> parents;   // parents[child] = list of parents
    std::vector<std::vector<PoolId>> children;  // children[parent] = list of children
    std::mutex m;
};

SubmeshAllocator::SubmeshAllocator(
    unsigned int num_pools, unsigned long long capacity_bytes, unsigned long long alignment_bytes) :
    impl_(new Impl(num_pools, capacity_bytes, alignment_bytes)) {}

SubmeshAllocator::~SubmeshAllocator() { delete impl_; }

SubmeshAllocator::SubmeshAllocator(SubmeshAllocator&& other) noexcept : impl_(other.impl_) { other.impl_ = nullptr; }

SubmeshAllocator& SubmeshAllocator::operator=(SubmeshAllocator&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

void SubmeshAllocator::add_dependency(PoolId parent, PoolId child) {
    std::lock_guard<std::mutex> g(impl_->m);
    if (parent >= impl_->pools.size() || child >= impl_->pools.size()) {
        return;
    }
    impl_->children[parent].push_back(child);
    impl_->parents[child].push_back(parent);
}

bool SubmeshAllocator::allocate(PoolId pool, unsigned long long size, AllocationHandle& out) {
    std::lock_guard<std::mutex> g(impl_->m);
    if (pool >= impl_->pools.size()) {
        return false;
    }
    return impl_->allocate_internal(pool, size, out);
}

bool SubmeshAllocator::allocate_at(
    PoolId pool, unsigned long long start, unsigned long long size, AllocationHandle& out) {
    std::lock_guard<std::mutex> g(impl_->m);
    if (pool >= impl_->pools.size()) {
        return false;
    }
    return impl_->allocate_at_internal(pool, start, size, out);
}

void SubmeshAllocator::free(const AllocationHandle& handle) {
    std::lock_guard<std::mutex> g(impl_->m);
    if (handle.pool >= impl_->pools.size()) {
        return;
    }
    impl_->free_internal(handle);
}

}  // namespace tt::tt_metal::distributed
