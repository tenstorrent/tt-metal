#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <list>
#include <cstdint>
#include <optional>
#include <functional>
#include <string>
#include <limits>
#include <map>
#include <assert.h>

// TODOs (some sprinkled throughout below):
// Score reuse_window of eg 4, 3, 2, 1, pick best
// Re-enter pre-allocate mode w/ large free node at bottom
// Re-enter pre-allocate mode after sync

// Number of programs executed before eviction + 1
// When this is 1, programs can execute back to back
constexpr std::int32_t max_reuse_window = 2;

class Program {
private:
    std::uint32_t size_;
    std::uint32_t cost_;

public:
    Program(std::uint32_t size, std::uint32_t cost);
    Program() = default;

    std::uint32_t get_size() const { return this->size_; }
    std::uint32_t get_cost() const { return this->cost_; }
    static std::string get_name(std::uint32_t pgm_id);
};

Program::Program(std::uint32_t size, std::uint32_t cost) : size_(size), cost_(cost) {}

std::string Program::get_name(std::uint32_t pgm_id) {
    std::string name;
    if (pgm_id == 0xffffffff) {
        name = std::string("--");
    } else {
        name = std::string() + static_cast<char>('A' + pgm_id / 26) + static_cast<char>('A' + pgm_id % 26);
    }
    return name;
}

class TraceNode {
private:
    static constexpr std::int32_t next_use_unused = 0x7fffffff;
    static constexpr std::uint32_t unallocated_addr = 0xffffffff;

    const std::uint32_t pgm_id_;  // some handle to pgm
    std::int32_t remaining_;      // number of uses of pgm after this one
    std::int32_t next_idx_;       // index of next use of this pgm
    float weight_;                // normalized pgm_cost * uses remaining

    std::uint32_t addr_;         // allocation address
    bool does_dispatch_;         // True when data is not resident
    std::int32_t stall_idx_;     // TraceNode idx this node must stall on (memory reuse)

public:
    TraceNode(uint32_t program_id);

    std::uint32_t get_pgm_id() const { return this->pgm_id_; }
    std::int32_t get_next_idx() const { return this->next_idx_; }
    std::int32_t get_remaining() const { return this->remaining_; }
    std::uint32_t get_addr() const { return this->addr_; }
    bool does_dispatch() const { return this->does_dispatch_; }
    std::int32_t get_stall_idx() const { return this->stall_idx_; }
    bool does_stall() const { return this->stall_idx_ != -1; }
    float get_weight() const { return this->weight_; }
    bool is_allocated() const { return this->addr_ != unallocated_addr; }
    static bool is_index_valid(std::int32_t idx) { return idx != next_use_unused; }

    void set_remaining(std::int32_t remaining) { this->remaining_ = remaining; }
    void set_next_idx(std::uint32_t idx) { this->next_idx_ = idx; }
    void calculate_weight(float weight) { this->weight_ = weight * this->remaining_; }
    void set_addr(std::uint32_t addr) { this->addr_ = addr; }
    void set_does_dispatch() { this->does_dispatch_ = true; }
    void set_stall_idx(std::int32_t idx) { this->stall_idx_ = idx; }

    void print() const;
};

TraceNode::TraceNode(std::uint32_t pgm_id) :
    pgm_id_(pgm_id),
    remaining_(0),
    next_idx_(next_use_unused),
    weight_(0.0f),
    addr_(unallocated_addr),
    does_dispatch_(false),
    stall_idx_(-1) {}

void TraceNode::print() const {
    fprintf(
        stderr,
        "%s R%d N%2d W%1.2f",
        Program::get_name(this->pgm_id_).c_str(),
        this->remaining_,
        (this->next_idx_ == next_use_unused) ? -1 : this->next_idx_,
        this->weight_);
}

// Temporary meta data per program data used for allocating programs in traces
class TraceProgramData {
private:
    enum AllocationClass {
        NOT_ALLOCED,
        ALLOCED,
    };

    std::int32_t count_;     // number of times this pgm was used in the trace
    std::int32_t used_idx_;  // trace node index of previous use of this pgm, valid during consrtruction
    std::int32_t last_used_idx_;  // idx of last use of this pgm
    AllocationClass alloced_;

public:
    TraceProgramData();

    void use(std::int32_t idx) {
        if (this->count_ == 0) {
            this->last_used_idx_ = idx;
        }
        this->used_idx_ = idx;
        this->count_++;
    }

    void inc_count() { this->count_++; }
    void alloc() { this->alloced_ = ALLOCED; }
    void dealloc() { this->alloced_ = NOT_ALLOCED; }

    std::int32_t get_count() const { return this->count_; }
    std::int32_t get_used() const { return this->used_idx_; }
    std::int32_t get_last_used() const { return this->last_used_idx_; }
    bool is_not_alloced() const { return this->alloced_ == NOT_ALLOCED; }
};

TraceProgramData::TraceProgramData() : count_(0), used_idx_(0), last_used_idx_(0), alloced_(NOT_ALLOCED) {}

class AllocNode {
private:
    static constexpr std::int32_t never_used = -max_reuse_window;

    std::uint32_t pgm_id_;
    std::uint32_t addr_;       // base address of this allocation unit
    std::uint32_t size_;       // size of this allocation unit
    bool is_free_;             // true if this is just tracking unallocated space
    std::int32_t first_use_;   // trace id of when this was first used
    std::int32_t prev_use_;    // trace id of when this was last used

public:
    AllocNode(std::uint32_t pgm_id, std::uint32_t addr, std::uint32_t size, std::int32_t used_idx);
    AllocNode(std::uint32_t addr, std::uint32_t size);

    std::uint32_t get_pgm_id() const { return this->pgm_id_; }
    std::uint32_t get_addr() const { return this->addr_; }
    std::uint32_t get_size() const { return this->size_; }
    std::int32_t get_first_use() const { return this->first_use_; }
    std::int32_t get_prev_use() const { return this->prev_use_; }
    bool is_free() const { return this->is_free_; }
    template <typename Container, typename Method>
    float get_weight(const Container& indexed_items, Method method) const;

    void set_prev_use(std::int32_t trace_idx) { this->prev_use_ = trace_idx; }
    void set_addr(std::uint32_t addr) { this->addr_ = addr; }
    void set_size(std::uint32_t size) { this->size_ = size; }
};

AllocNode::AllocNode(std::uint32_t pgm_id, std::uint32_t addr, std::uint32_t size, std::int32_t used_idx) :
    pgm_id_(pgm_id), addr_(addr), size_(size), is_free_(false), first_use_(used_idx), prev_use_(used_idx) {}

AllocNode::AllocNode(std::uint32_t addr, std::uint32_t size) :
    pgm_id_(-1), addr_(addr), size_(size), is_free_(true), first_use_(never_used), prev_use_(never_used) {}

// AllocNodes for free blocks don't have trace nodes
// The weight is in the trace node when alloced, so indirect through the trace
// to get there
template <typename Container, typename Method>
float AllocNode::get_weight(const Container& indexed_items, Method method) const {
    if (is_free_) {
        return 0.0f;
    }

    return (indexed_items[this->prev_use_].*method)();
}

class WorkerBufferManager {
private:
    std::uint32_t buffer_size_;  // total size of the managed buffer
    std::vector<TraceProgramData> program_data_;

    std::list<AllocNode> allocator_;
    std::list<std::list<AllocNode>::iterator> lru_;
    std::vector<std::optional<std::list<std::list<AllocNode>::iterator>::iterator>> alloced_pgms_;

    // Init
    std::vector<TraceProgramData> build_use_data(std::vector<TraceNode>& trace, const std::vector<Program>& programs);

    // Allocate
    bool handle_trivial_cases(std::vector<TraceNode>& trace, const std::vector<Program>& programs);
    void alloc(std::int32_t reuse_window, std::vector<TraceNode>& trace, const std::vector<Program>& programs);

    // Helpers
    std::list<std::list<AllocNode>::iterator>::iterator find_eviction_candidates(
        bool only_stale,
        std::int32_t window,
        std::uint32_t size_needed,
        std::int32_t node_idx,
        const std::vector<TraceNode>& trace);
    std::list<AllocNode>::iterator evict(
        std::uint32_t& freed_size,
        std::int32_t trace_idx,
        std::uint32_t size_needed,
        std::list<std::list<AllocNode>::iterator>::iterator evict_it,
        std::vector<TraceNode>& trace);
    void allocate_in_hole(
        std::int32_t trace_idx,
        std::uint32_t freed_size,
        std::uint32_t size_needed,
        std::list<AllocNode>::iterator alloc_it,
        std::vector<TraceNode>& trace);
    void sort_preallocations(std::list<AllocNode>::iterator uncommitted_it, const std::vector<TraceNode>& trace);
    bool try_to_reenter_preallocation_mode(
        std::int32_t reuse_window,
        std::uint32_t& pre_alloc_addr_top,
        std::uint32_t& pre_alloc_addr,
        std::list<AllocNode>::iterator& uncommitted_it,
        std::int32_t trace_idx,
        std::vector<TraceNode>& trace);

    void commit_preallocations(
        std::uint32_t addr_top, std::list<AllocNode>::iterator commit_start, std::vector<TraceNode>& trace);

public:
    WorkerBufferManager(std::uint32_t buffer_size);

    void process_trace(std::vector<TraceNode>& trace, const std::vector<Program>& programs);

    void verify(const std::vector<TraceNode>& trace, const std::vector<Program>& programs);
    void dump(const std::vector<TraceNode>& trace, const std::vector<Program>& programs);
    void dump_stats(const std::vector<TraceNode>& trace, const std::vector<Program>& programs);
    void dump_allocator(std::optional<std::reference_wrapper<const std::vector<TraceNode>>> opt_trace = std::nullopt);
};

WorkerBufferManager::WorkerBufferManager(std::uint32_t buffer_size) : buffer_size_(buffer_size) {}

void WorkerBufferManager::dump_allocator(
    std::optional<std::reference_wrapper<const std::vector<TraceNode>>> opt_trace) {
    fprintf(stderr, "Allocator:\n");
    for (const auto& node : this->allocator_) {
        std::uint32_t pgm_id = node.get_pgm_id();
        if (opt_trace.has_value()) {
            float weight = node.is_free() ? 0.0f : opt_trace->get()[node.get_prev_use()].get_weight();
            fprintf(
                stderr,
                "  %s: %d %d %f\n",
                Program::get_name(pgm_id).c_str(),
                node.get_addr(),
                node.get_size(),
                weight);
        } else {
            fprintf(stderr, "  %s: %d %d\n", Program::get_name(pgm_id).c_str(), node.get_addr(), node.get_size());
        }
    }

    fprintf(stderr, "LRU:\n");
    for (const auto& node : this->lru_) {
        std::uint32_t pgm_id = node->get_pgm_id();
        fprintf(stderr, "  %s: last use %d\n", Program::get_name(pgm_id).c_str(), node->get_prev_use());
    }

    // Check consistency
    fprintf(stderr, "Check allocator consistency: #entries(%ld)\n", this->allocator_.size());
    if (this->allocator_.begin() != this->allocator_.end()) {
        std::uint32_t addr = this->allocator_.begin()->get_addr() + this->allocator_.begin()->get_size();
        for (auto it = this->allocator_.begin(); it != this->allocator_.end(); ++it) {
            if (addr != it->get_addr() + it->get_size()) {
                fprintf(stderr, "Failed: %d %d %d\n", addr, it->get_addr(), it->get_size());
                exit(0);
            }

            addr -= it->get_size();
        }
    }
}

class VerifyAllocNode : public AllocNode {
private:
    std::int32_t used_idx_;  // Trace idx when this was used

public:
    VerifyAllocNode(std::uint32_t pgm_id, std::uint32_t addr, std::uint32_t size, std::int32_t used);

    std::int32_t get_used_idx() const { return this->used_idx_; }

    void set_used_idx(std::int32_t used_idx) { this->used_idx_ = used_idx; }
};

VerifyAllocNode::VerifyAllocNode(std::uint32_t pgm_id, std::uint32_t addr, std::uint32_t size, std::int32_t used) :
    AllocNode(pgm_id, addr, size, used), used_idx_(used) {}

// Populate trace nodes w/ use data
std::vector<TraceProgramData> WorkerBufferManager::build_use_data(
    std::vector<TraceNode>& trace, const std::vector<Program>& programs) {
    std::vector<TraceProgramData> program_data;

    std::uint32_t max_cost = 0;

    // Traverse trace back to front and gather stats
    for (std::int32_t trace_idx = trace.size() - 1; trace_idx >= 0; trace_idx--) {
        std::uint32_t pgm_id = trace[trace_idx].get_pgm_id();

        std::uint32_t size_needed = programs[pgm_id].get_size();
        if (size_needed > this->buffer_size_) {
            fprintf(stderr, "Program %d's size %d exceeds buffer size %d\n", pgm_id, size_needed, this->buffer_size_);
            throw("failed");
        }

        if (pgm_id >= program_data.size()) {
            program_data.resize(pgm_id + 1);
        }

        if (program_data[pgm_id].get_count() != 0) {
            trace[trace_idx].set_next_idx(program_data[pgm_id].get_used());
        }
        if (programs[pgm_id].get_cost() > max_cost) {
            max_cost = programs[pgm_id].get_cost();
        }

        trace[trace_idx].set_remaining(program_data[pgm_id].get_count());
        program_data[pgm_id].use(trace_idx);
    }

    // Traverse trace and set weight based on pgm cost and max cost
    for (std::int32_t trace_idx = 0; trace_idx < (std::int32_t)trace.size(); trace_idx++) {
        std::uint32_t pgm_id = trace[trace_idx].get_pgm_id();
        uint32_t cost = programs[pgm_id].get_cost();
        float normalized_cost = (float)cost / float(max_cost);
        trace[trace_idx].calculate_weight(normalized_cost);
    }

    return program_data;
}

bool WorkerBufferManager::handle_trivial_cases(std::vector<TraceNode>& trace, const std::vector<Program>& programs) {
    std::uint32_t total_alloced = 0;
    for (std::int32_t trace_idx = 0; trace_idx < (std::int32_t)trace.size(); trace_idx++) {
        std::uint32_t pgm_id = trace[trace_idx].get_pgm_id();
        std::uint32_t size_needed = programs[pgm_id].get_size();

        assert(size_needed <= this->buffer_size_);

        total_alloced += size_needed;
    }

    if (total_alloced <= this->buffer_size_) {
        fprintf(stderr, "Trivial fit, allocating statically\n");

        std::uint32_t alloc_addr = 0;
        for (std::int32_t trace_idx = 0; trace_idx < (std::int32_t)trace.size(); trace_idx++) {
            if (not trace[trace_idx].is_allocated()) {
                std::uint32_t pgm_id = trace[trace_idx].get_pgm_id();
                std::int32_t child_idx = trace_idx;
                while (TraceNode::is_index_valid(child_idx)) {
                    trace[child_idx].set_addr(alloc_addr);
                    child_idx = trace[child_idx].get_next_idx();
                }

                alloc_addr += programs[pgm_id].get_size();
                this->program_data_[pgm_id].alloc();
                trace[trace_idx].set_does_dispatch();
            }
        }

        return true;
    }

    return false;
}

void WorkerBufferManager::commit_preallocations(
    std::uint32_t addr_top, std::list<AllocNode>::iterator commit_start, std::vector<TraceNode>& trace) {
    std::uint32_t addr = addr_top;

    std::list<AllocNode>::iterator alloc_it =
        (commit_start == this->allocator_.end()) ? this->allocator_.begin() : std::next(commit_start);
    while (alloc_it != this->allocator_.end()) {
        if (not alloc_it->is_free()) {  // hmm, can this happen?
            std::int32_t trace_idx = alloc_it->get_first_use();
            std::uint32_t pgm_id = alloc_it->get_pgm_id();
            std::uint32_t size = alloc_it->get_size();
            addr -= size;
            alloc_it->set_addr(addr);
            this->program_data_[pgm_id].alloc();
            trace[trace_idx].set_does_dispatch();

            // Commit allocations up to how far we've progressed through this trace
            while (trace_idx <= alloc_it->get_prev_use()) {
                trace[trace_idx].set_addr(alloc_it->get_addr());
                trace_idx = trace[trace_idx].get_next_idx();
            }
        }

        ++alloc_it;
    }
}

// Selecting what to evict is based on one of two heuristics:
//  - evict node(s) with lowest weight
//  - evict nodes(s) used furthest in the future
// Lowest weight seems to make sense as we want to keep expensive/highly re-used items in the buffer
// However, if you consider thrashing, maybe we're better off kicking out something that isn't used for a while
// w/ 8 randomized test cases, furthest out wins by a little (2-5%)
// w/ data gathered from llama and trying 4 buffer sizes, lowest weight wins by a lot (~20%)
// Should experiment further and maybe choose dynamically.  Or maybe there is a better heurstic
#define EVICT_ON_DISTANCE 0

// The LRU side of the LRU list contains the free nodes, free nodes are at the
// the old end, so we start w/ those
// Walk from LRU to MRU:
//   aggregate nodes from low memory to high memory to meet the size requirement
//   fail a required node doesn't has been used within the reuse window
//   score sets of nodes
//   select the set w/ the lowest score
// Returns an iterator into LRU of the first (lowest address) node to use, points to end if not
// If this call fails, call it again w/ a smaller reuse window
// If only_stale is set, only considers allocations w/ a weight of 0.0f (no more uses)
std::list<std::list<AllocNode>::iterator>::iterator WorkerBufferManager::find_eviction_candidates(
    bool only_stale,
    std::int32_t window,
    std::uint32_t size_needed,
    std::int32_t trace_idx,
    const std::vector<TraceNode>& trace) {
    std::list<std::list<AllocNode>::iterator>::iterator match = this->lru_.end();

#if EVICT_ON_DISTANCE
    std::int32_t best_dist = -1;
#else
    float best_score = std::numeric_limits<float>::infinity();
#endif

    fprintf(stderr, "Trying to find eviction candidate for size %d\n", size_needed);
    for (auto lru_it = this->lru_.begin(); lru_it != this->lru_.end(); ++lru_it) {
        std::uint32_t free_size = 0;
        bool found_one = false;

#if EVICT_ON_DISTANCE
        std::int32_t dist = std::numeric_limits<std::int32_t>::max();
#else
        float score = 0.0f;
#endif

        // Iterates over allocator from high to low memory
        for (auto alloc_it = *lru_it; alloc_it != this->allocator_.end(); ++alloc_it) {
            if (alloc_it->get_prev_use() + window > trace_idx) {
                break;  // failed
            }

            float weight = alloc_it->get_weight(trace, &TraceNode::get_weight);
            if (not only_stale or weight == 0.0f) {
                free_size += alloc_it->get_size();
#if EVICT_ON_DISTANCE
                std::int32_t this_dist = alloc_it->is_free() ? 10000 : trace[alloc_it->get_prev_use()].get_next_idx();
                dist = std::min(dist, this_dist);
#else
                score += weight;
#endif
                if (size_needed <= free_size) {
                    found_one = true;
                    break;
                }
            }
        }

#if EVICT_ON_DISTANCE
        if (found_one && dist > best_dist)
#else
        if (found_one && score < best_score)
#endif
        {
#if EVICT_ON_DISTANCE
            best_dist = dist;
#else
            best_score = score;
#endif
            match = lru_it;
        }
    }

    if (match != this->lru_.end()) {
        fprintf(stderr, "Found eviction candidate(s) at addr %d\n", (*match)->get_addr());
    }

    return match;
}

// Called after finding eviction candidates, so guaranteed to succeed
// evict_it points to the first (highest address) node to evict
// Removes alloc nodes up to the requested size
// Returns an iterator to below the created hole in the allocator
std::list<AllocNode>::iterator WorkerBufferManager::evict(
    std::uint32_t& freed_size,
    std::int32_t trace_idx,
    std::uint32_t size_needed,
    std::list<std::list<AllocNode>::iterator>::iterator evict_it,
    std::vector<TraceNode>& trace) {
    freed_size = 0;
    auto alloc_it = *evict_it;

    fprintf(stderr, "Evicting\n");
    while (freed_size < size_needed) {
        freed_size += alloc_it->get_size();
        // TODO: could use back-iterators from allocator to LRU to save this search
        for (auto lru_it = this->lru_.begin(); lru_it != this->lru_.end(); ++lru_it) {
            if (*lru_it == alloc_it) {
                std::uint32_t pgm_id = alloc_it->get_pgm_id();

                if (not alloc_it->is_free()) {
                    this->program_data_[pgm_id].dealloc();
                    this->alloced_pgms_[pgm_id].reset();
                }

                // May straddle the line so need to check both heap/rb for stalls
                if (alloc_it->get_prev_use() > trace[trace_idx].get_stall_idx()) {
                    trace[trace_idx].set_stall_idx(alloc_it->get_prev_use());
                }

                this->lru_.erase(lru_it);
                alloc_it = this->allocator_.erase(alloc_it);
                break;
            }
        }
    }

    return alloc_it;
}

void WorkerBufferManager::allocate_in_hole(
    std::int32_t trace_idx,
    std::uint32_t freed_size,
    std::uint32_t size_needed,
    std::list<AllocNode>::iterator alloc_it,
    std::vector<TraceNode>& trace) {
    std::uint32_t pgm_id = trace[trace_idx].get_pgm_id();

    assert(this->allocator_.size() != 0);
    std::uint32_t alloc_addr;

    // alloc_it points to the entry below a hole of freed_size
    if (alloc_it == this->allocator_.end()) {
        // Grow down from previous allocation
        auto prev_it = std::prev(alloc_it);
        alloc_addr = prev_it->get_addr() - size_needed;
        this->allocator_.push_back({pgm_id, alloc_addr, size_needed, trace_idx});
        this->lru_.push_back(--this->allocator_.end());

        // Add an empty free node at the bottom of the heap
        // This simplifies book keeping by not having to special case the gap
        if (alloc_addr != 0) {
            this->allocator_.push_back({0, alloc_addr});
            this->lru_.push_front(--this->allocator_.end());
        }
    } else {
        // Is there a hole?  Which side does it go on?  Merge if adjacent
        std::uint32_t base_addr = alloc_it->get_addr() + alloc_it->get_size();

        if (freed_size == size_needed) {
            // Perfect fit
            alloc_addr = base_addr;
            auto new_it = this->allocator_.insert(alloc_it, {pgm_id, alloc_addr, size_needed, trace_idx});
            this->lru_.push_back(new_it);
        } else {
            bool hole_top = false;
            assert(alloc_it != this->allocator_.end());
            if (alloc_it != this->allocator_.begin()) {
                auto prev_it = std::prev(alloc_it);
                if (prev_it->get_weight(trace, &TraceNode::get_weight) <
                    alloc_it->get_weight(trace, &TraceNode::get_weight)) {
                    hole_top = true;
                } else {
                    hole_top = false;
                }
            }

            std::uint32_t hole_size = freed_size - size_needed;
            if (hole_top) {
                alloc_addr = base_addr;
                auto above_it = std::prev(alloc_it);
                if (above_it->is_free()) {
                    // This should be rare.  When freeing, we walk LRU from oldest to newest and
                    // put free blocks at the oldest side, so they should be hit first
                    above_it->set_size(above_it->get_size() + hole_size);
                    above_it->set_addr(base_addr + size_needed);
                } else {
                    auto new_it = this->allocator_.insert(alloc_it, {base_addr + size_needed, hole_size});
                    this->lru_.push_front(new_it);
                }
                auto new_it = this->allocator_.insert(alloc_it, {pgm_id, alloc_addr, size_needed, trace_idx});
                this->lru_.push_back(new_it);
            } else {
                alloc_addr = base_addr + hole_size;
                auto new_it = this->allocator_.insert(alloc_it, {pgm_id, alloc_addr, size_needed, trace_idx});
                this->lru_.push_back(new_it);
                if (alloc_it->is_free()) {
                    alloc_it->set_size(alloc_it->get_size() + hole_size);
                } else {
                    new_it = this->allocator_.insert(alloc_it, {base_addr, hole_size});
                    this->lru_.push_front(new_it);
                }
            }
        }
    }

    trace[trace_idx].set_does_dispatch();
    trace[trace_idx].set_addr(alloc_addr);
    this->program_data_[pgm_id].alloc();
    this->alloced_pgms_[pgm_id] = std::prev(this->lru_.end());
}

void WorkerBufferManager::sort_preallocations(
    std::list<AllocNode>::iterator uncommitted_it, const std::vector<TraceNode>& trace) {
    // Sort just the sub-range that we pre-allocated since last commit
    std::list<AllocNode> tmp;
    if (uncommitted_it == this->allocator_.end()) {
        tmp.splice(tmp.end(), this->allocator_, this->allocator_.begin(), this->allocator_.end());
    } else {
        tmp.splice(tmp.end(), this->allocator_, std::next(uncommitted_it), this->allocator_.end());
    }

    tmp.sort([&](const AllocNode& first, const AllocNode& second) {
        float first_weight = first.get_weight(trace, &TraceNode::get_weight);
        float second_weight = second.get_weight(trace, &TraceNode::get_weight);

        if (first_weight == second_weight) {
            // Probably weights are 0, sort so newest is at the bottom of heap
            // Allocate from the tie point down to minimize syncing
            return first.get_prev_use() > second.get_prev_use();
        } else {
            return first_weight > second_weight;
        }
    });

    this->allocator_.splice(this->allocator_.end(), tmp);
}

// Walk the allocations backwards from the bottom, look for the last
// entry which doesn't have a next use. Set addr_top and retry
// Note: these are sorted above from newest use down to oldest use;
// this is simple but not optimal.  Sorting the other way and tracking if
// we need a sync could give a better solution (use a slot up high, then
// sync after that which defers the sync)
bool WorkerBufferManager::try_to_reenter_preallocation_mode(
    std::int32_t reuse_window,
    std::uint32_t& pre_alloc_addr_top,
    std::uint32_t& pre_alloc_addr,
    std::list<AllocNode>::iterator& uncommitted_it,
    std::int32_t trace_idx,
    std::vector<TraceNode>& trace) {
    bool eviction_mode = true;  // tentative
    bool done = false;
    std::int32_t stall_idx = -1;

    while (!done && !this->allocator_.empty()) {
        auto& node = this->allocator_.back();
        std::int32_t last_use = node.get_prev_use();
        pre_alloc_addr = pre_alloc_addr_top = node.get_addr();

        if (last_use + reuse_window >= trace_idx || trace[last_use].get_remaining() != 0) {
            done = true;
        } else {
            this->program_data_[node.get_pgm_id()].dealloc();
            this->alloced_pgms_[node.get_pgm_id()].reset();
            stall_idx = node.get_prev_use();
            // Ugh, search through LRU.  Should be near the front so not too deep a search
            // TODO: could use back-iterators from allocator to LRU to save this search
            auto lru_it = this->lru_.begin();
            auto last = std::prev(this->allocator_.end());
            while (*lru_it != last) {
                ++lru_it;
            }
            this->lru_.erase(lru_it);
            this->allocator_.pop_back();
            eviction_mode = false;
        }
    }

    if (eviction_mode) {
        fprintf(stderr, "Transitioning to eviction mode with %ld allocations\n", this->allocator_.size());

        // This simplifies book keeping by not having to special case the gap
        std::uint32_t addr = this->allocator_.back().get_addr();
        if (addr > 0) {
            this->allocator_.push_back({0, addr});
            this->lru_.push_front(--this->allocator_.end());
        }
    } else {
        uncommitted_it = std::prev(this->allocator_.end());
        fprintf(
            stderr,
            "Staying in pre-allocation mode at %d: %ld existing allocations ending at addr %d\n",
            trace_idx,
            this->allocator_.size(),
            pre_alloc_addr_top);
    }

    trace[trace_idx].set_stall_idx(stall_idx);

    return eviction_mode;
}

// General approach:
// 1) Track total allocated size up until the heap overflows
// 2) At this point, allocate based on last seen priority for the pgm putting
// high priority at the top and low priority at the bottom
// 3) Free up what can be freed up and restart to first step
// 4) At some point this may exhaust itself and we have to enter eviction mode
// 5) Evict based on LRU and hole size from here on out
void WorkerBufferManager::alloc(
    std::int32_t reuse_window, std::vector<TraceNode>& trace, const std::vector<Program>& programs) {
    this->alloced_pgms_.resize(programs.size());
    std::int32_t trace_idx = 0;

    fprintf(stderr, "+++++Starting heap allocation up to %d\n", this->buffer_size_);

    bool eviction_mode = false;
    std::uint32_t pre_alloc_addr_top = this->buffer_size_;
    std::uint32_t pre_alloc_addr = this->buffer_size_;
    std::list<AllocNode>::iterator uncommitted_it = this->allocator_.end();
    while (trace_idx < (std::int32_t)trace.size()) {
        std::uint32_t pgm_id = trace[trace_idx].get_pgm_id();

        if (this->alloced_pgms_[pgm_id].has_value()) {
            fprintf(stderr, "Reusing: %d (%s)\n", trace_idx, Program::get_name(pgm_id).c_str());
            // splice moves to end of lru
            trace[trace_idx].set_addr((*this->alloced_pgms_[pgm_id].value())->get_addr());
            this->lru_.splice(this->lru_.end(), this->lru_, this->alloced_pgms_[pgm_id].value());
            this->alloced_pgms_[pgm_id] = std::prev(this->lru_.end());
            this->lru_.back()->set_prev_use(trace_idx);
        } else {
            std::uint32_t size_needed = programs[pgm_id].get_size();
            if (eviction_mode) {
                fprintf(stderr, "Eviction mode: %d\n", trace_idx);

                std::list<std::list<AllocNode>::iterator>::iterator evict_it;
                evict_it = find_eviction_candidates(true, reuse_window, size_needed, trace_idx, trace);
                if (evict_it == this->lru_.end()) {
                    for (std::int32_t window = reuse_window; window != 0; window--) {
                        fprintf(stderr, "Trying eviction window: %d\n", window);
                        evict_it = find_eviction_candidates(false, window, size_needed, trace_idx, trace);
                        if (evict_it != this->lru_.end()) {
                            break;
                        }
                    }
                }
                if (evict_it != this->lru_.end()) {
                    std::uint32_t freed_size = 0;
                    auto alloc_it = evict(freed_size, trace_idx, size_needed, evict_it, trace);
                    // TODO: can re-enter pre-alloc w/ a "large" free block near end
                    if (this->allocator_.size() == 0) {
                        pre_alloc_addr_top = this->buffer_size_;
                        pre_alloc_addr = this->buffer_size_;
                        uncommitted_it = this->allocator_.end();
                        eviction_mode = false;
                        continue;
                    }
                    allocate_in_hole(trace_idx, freed_size, size_needed, alloc_it, trace);
                } else {
                    fprintf(stderr, "Failed allocate %d bytes at trace_idx %d\n", size_needed, trace_idx);
                    dump_allocator();
                    throw("failed");
                }
            } else {
                fprintf(stderr, "Pre-allocation mode: %d\n", trace_idx);

                assert(this->program_data_[pgm_id].is_not_alloced());

                if (pre_alloc_addr < size_needed) {
                    fprintf(
                        stderr,
                        "Committing prior allocations on trace node %d start at addr %d\n",
                        trace_idx,
                        pre_alloc_addr_top);

                    // Allocate based on priority, then either evict from bottom or transition to eviction mode
                    sort_preallocations(uncommitted_it, trace);
                    commit_preallocations(pre_alloc_addr_top, uncommitted_it, trace);
                    eviction_mode = try_to_reenter_preallocation_mode(
                        reuse_window, pre_alloc_addr_top, pre_alloc_addr, uncommitted_it, trace_idx, trace);
                    continue;  // Reprocess this trace_idx
                } else {
                    pre_alloc_addr -= size_needed;
                    fprintf(
                        stderr,
                        "Pre-allocating node %d %s %d bytes at %d\n",
                        trace_idx,
                        Program::get_name(pgm_id).c_str(),
                        size_needed,
                        pre_alloc_addr);
                    this->allocator_.push_back({pgm_id, pre_alloc_addr, size_needed, trace_idx});
                    this->lru_.push_back(--this->allocator_.end());
                    this->alloced_pgms_[pgm_id] = std::prev(this->lru_.end());
                }
            }
        }

        trace_idx++;
    }

    if (not eviction_mode) {
        // Allocated everything w/o eviction, just finalize the addresses
        commit_preallocations(pre_alloc_addr_top, uncommitted_it, trace);
    }
}

void WorkerBufferManager::process_trace(std::vector<TraceNode>& trace, const std::vector<Program>& programs) {
    this->program_data_.clear();
    this->allocator_.clear();
    this->lru_.clear();
    this->alloced_pgms_.clear();

    this->program_data_ = build_use_data(trace, programs);

    if (!handle_trivial_cases(trace, programs)) {
        alloc(max_reuse_window, trace, programs);
    }
}

void WorkerBufferManager::verify(const std::vector<TraceNode>& trace, const std::vector<Program>& programs) {
    std::list<VerifyAllocNode> allocator;

    std::int32_t last_sync_idx = 0;
    bool passed = true;

    for (std::int32_t trace_idx = 0; trace_idx < (std::int32_t)trace.size(); trace_idx++) {
        const auto& trace_node = trace[trace_idx];
        std::uint32_t pgm_id = trace_node.get_pgm_id();
        std::uint32_t addr = trace_node.get_addr();
        std::uint32_t size = programs[pgm_id].get_size();

        if (not trace_node.is_allocated()) {
            fprintf(stderr, "Using unallocated program %s at node %d\n", Program::get_name(pgm_id).c_str(), trace_idx);
            passed = false;
        }

        if (addr + size > this->buffer_size_) {
            fprintf(
                stderr,
                "Allocation address out of range for program %s at node %d\n",
                Program::get_name(pgm_id).c_str(),
                trace_idx);
            passed = false;
        }

        if (trace[trace_idx].does_stall()) {
            if (last_sync_idx < trace[trace_idx].get_stall_idx()) {
                last_sync_idx = trace[trace_idx].get_stall_idx();
            }
            if (trace[trace_idx].get_stall_idx() >= trace_idx) {
                fprintf(stderr, "Node %d stalling on future node %d\n", trace_idx, trace[trace_idx].get_stall_idx());
                passed = false;
            }
        }

        if (trace_node.does_dispatch()) {
            std::uint32_t end = addr + size;

            for (auto alloc_it = allocator.begin(); alloc_it != allocator.end();) {
                std::uint32_t o_start = alloc_it->get_addr();
                std::uint32_t o_end = o_start + alloc_it->get_size();

                if (addr < o_end && o_start < end) {
                    if (alloc_it->get_used_idx() > last_sync_idx) {
                        fprintf(
                            stderr,
                            "New allocation at node %d (%d, %d) overlaps existing allocation (%d, %d) - missing "
                            "stall\n",
                            trace_idx,
                            addr,
                            end,
                            o_start,
                            o_end);
                        passed = false;
                    } else {
                        // Deallocate the old.  If we re-use it, that's an error
                        alloc_it = allocator.erase(alloc_it);
                        continue;
                    }
                }

                ++alloc_it;
            }

            allocator.push_back({pgm_id, addr, size, trace_idx});
        } else {
            bool found = false;
            for (auto alloc_it = allocator.begin(); alloc_it != allocator.end(); ++alloc_it) {
                if (addr == alloc_it->get_addr() && size == alloc_it->get_size()) {
                    alloc_it->set_used_idx(trace_idx);
                    found = true;
                }
            }
            if (!found) {
                fprintf(
                    stderr,
                    "Failed to find allocation for program %s at node %d\n",
                    Program::get_name(pgm_id).c_str(),
                    trace_idx);
                passed = false;
            }
        }
    }

    if (not passed) {
        throw("Verification: FAILED\n");
    }

    fprintf(stderr, "Verification: PASSED\n");
}

void WorkerBufferManager::dump(const std::vector<TraceNode>& trace, const std::vector<Program>& programs) {
    // Debug state dump
    fprintf(stderr, "Programs: size cost\n");
    for (std::uint32_t pgm_id = 0; pgm_id < programs.size(); pgm_id++) {
        fprintf(
            stderr,
            "%s: %5d %4d\n",
            Program::get_name(pgm_id).c_str(),
            programs[pgm_id].get_size(),
            programs[pgm_id].get_cost());
    }

    fprintf(stderr, "Trace:\n");
    for (std::int32_t trace_idx = 0; trace_idx < (std::int32_t)trace.size(); trace_idx++) {
        fprintf(stderr, "%2d: ", trace_idx);
        trace[trace_idx].print();

        std::uint32_t pgm_id = trace[trace_idx].get_pgm_id();
        std::uint32_t addr = trace[trace_idx].get_addr();
        fprintf(stderr, " %c", (this->program_data_[pgm_id].get_count() == 1) ? 'S' : 'H');
        fprintf(
            stderr,
            "[%5d-%5d) %c S:%2d",
            addr,
            addr + programs[pgm_id].get_size(),
            trace[trace_idx].does_dispatch() ? 'D' : 'x',
            trace[trace_idx].get_stall_idx());
        fprintf(stderr, "\n");
    }
}

void WorkerBufferManager::dump_stats(const std::vector<TraceNode>& trace, const std::vector<Program>& programs) {
    std::int32_t min_stall = trace.size();
    std::int32_t max_stall = 0;
    std::uint32_t n_stall = 0;
    std::int32_t total_stall = 0;
    std::uint32_t n_single_cycle = 0;

    std::vector<bool> dispatched(programs.size(), false);
    std::uint32_t total_xfer = 0;
    std::uint32_t worst_xfer = 0;
    std::uint32_t ideal_xfer = 0;

    for (std::int32_t trace_idx = 0; trace_idx < (std::int32_t)trace.size(); trace_idx++) {
        if (trace[trace_idx].does_stall()) {
            std::int32_t stall_idx = trace_idx - trace[trace_idx].get_stall_idx();

            if (stall_idx == 1) {
                n_single_cycle++;
                fprintf(stderr, "Single cycle stall at %d\n", trace_idx);
            }
            if (stall_idx < min_stall) {
                min_stall = stall_idx;
            }
            if (stall_idx > max_stall) {
                max_stall = stall_idx;
            }

            n_stall++;
            total_stall += stall_idx;
        }

        std::uint32_t pgm_id = trace[trace_idx].get_pgm_id();
        std::uint32_t pgm_size = programs[pgm_id].get_size();
        total_xfer += trace[trace_idx].does_dispatch() ? pgm_size : 0;
        worst_xfer += pgm_size;
        ideal_xfer += dispatched[pgm_id] ? 0 : pgm_size;
        dispatched[pgm_id] = true;
    }

    fprintf(
        stderr,
        "Stalls: min_#nodes(%d) max_#nodes(%d) avg_#nodes(%2.1f) total(%d) one_node_count(%d)\n",
        min_stall,
        max_stall,
        (float)total_stall / (float)n_stall,
        n_stall,
        n_single_cycle);
    fprintf(
        stderr,
        "Dispatch: %%ideal(%2.1f) %%eager(%2.1f)\n",
        (float)total_xfer / (float)ideal_xfer * 100.0f,
        (float)total_xfer / (float)worst_xfer * 100.0f);
}


std::vector<Program> setup_programs() {
    std::vector<Program> programs;
    programs.resize(155);
    programs[2] = {4624, 4484};
programs[4] = {1472, 1356};
programs[8] = {6864, 18424};
programs[6] = {6864, 18420};
programs[10] = {13488, 13024};
programs[12] = {8688, 39088};
programs[14] = {2480, 2368};
programs[16] = {11728, 11196};
programs[18] = {11824, 11296};
programs[20] = {4224, 4116};
programs[22] = {1648, 1568};
programs[24] = {4688, 4580};
programs[26] = {39216, 38564};
programs[28] = {1520, 1448};
programs[30] = {10592, 47548};
programs[34] = {9296, 19228};
programs[32] = {9296, 19228};
programs[36] = {4960, 4812};
programs[38] = {10544, 47232};
programs[40] = {6880, 6724};
programs[42] = {10528, 47096};
programs[46] = {9264, 19192};
programs[44] = {9264, 19192};
programs[48] = {4928, 4784};
programs[50] = {1472, 1356};
programs[54] = {6864, 18424};
programs[52] = {6864, 18420};
programs[56] = {13488, 13024};
programs[58] = {1232, 1136};
programs[62] = {11024, 10188};
programs[60] = {11024, 10188};
programs[64] = {1216, 2272};
programs[66] = {1984, 1904};
programs[68] = {5568, 15496};
programs[70] = {1824, 1764};
programs[72] = {1504, 1368};
programs[74] = {1648, 3144};
programs[76] = {1472, 1356};
programs[78] = {5456, 5124};
programs[80] = {5328, 5200};
programs[82] = {2016, 1948};
programs[84] = {1232, 1136};
programs[86] = {4608, 4472};
programs[88] = {1232, 1136};
programs[92] = {8784, 20748};
programs[90] = {8784, 20748};
programs[94] = {14848, 37996};
programs[98] = {10048, 9504};
programs[96] = {10048, 9504};
programs[100] = {1200, 1116};
programs[102] = {2096, 1768};
programs[104] = {10656, 10208};
programs[106] = {10656, 10208};
programs[108] = {11136, 10580};
programs[110] = {51776, 50064};
programs[112] = {1232, 1136};
programs[114] = {1792, 1400};
programs[118] = {9888, 9496};
programs[116] = {9888, 9496};
programs[122] = {10528, 20484};
programs[120] = {10528, 20484};
programs[124] = {4304, 4152};
programs[128] = {10848, 10308};
programs[126] = {10848, 10308};
programs[130] = {6240, 6084};
programs[134] = {11120, 10604};
programs[132] = {11120, 10604};
programs[138] = {10512, 20472};
programs[136] = {10512, 20472};
programs[140] = {2336, 1884};
programs[144] = {8832, 20772};
programs[142] = {8832, 20772};
programs[146] = {14032, 35500};
programs[150] = {6800, 18368};
programs[148] = {6800, 18364};
programs[152] = {5504, 15472};
programs[154] = {1824, 1764};

return programs;
}


std::vector<TraceNode> setup_trace() {
    std::vector<TraceNode> trace_nodes;
trace_nodes.push_back(78);
trace_nodes.push_back(80);
trace_nodes.push_back(78);
trace_nodes.push_back(80);
trace_nodes.push_back(82);
trace_nodes.push_back(82);
trace_nodes.push_back(84);
trace_nodes.push_back(84);
trace_nodes.push_back(86);
trace_nodes.push_back(88);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(98);
trace_nodes.push_back(96);
trace_nodes.push_back(100);
trace_nodes.push_back(102);
trace_nodes.push_back(104);
trace_nodes.push_back(106);
trace_nodes.push_back(108);
trace_nodes.push_back(108);
trace_nodes.push_back(110);
trace_nodes.push_back(112);
trace_nodes.push_back(114);
trace_nodes.push_back(118);
trace_nodes.push_back(116);
trace_nodes.push_back(122);
trace_nodes.push_back(120);
trace_nodes.push_back(124);
trace_nodes.push_back(92);
trace_nodes.push_back(90);
trace_nodes.push_back(94);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(128);
trace_nodes.push_back(126);
trace_nodes.push_back(130);
trace_nodes.push_back(134);
trace_nodes.push_back(132);
trace_nodes.push_back(138);
trace_nodes.push_back(136);
trace_nodes.push_back(140);
trace_nodes.push_back(124);
trace_nodes.push_back(144);
trace_nodes.push_back(142);
trace_nodes.push_back(146);
trace_nodes.push_back(62);
trace_nodes.push_back(60);
trace_nodes.push_back(64);
trace_nodes.push_back(62);
trace_nodes.push_back(60);
trace_nodes.push_back(64);
trace_nodes.push_back(66);
trace_nodes.push_back(150);
trace_nodes.push_back(148);
trace_nodes.push_back(152);
trace_nodes.push_back(154);
return trace_nodes;
}



int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: cache <npgms> <ntracenodes> <bufsize>\n");
        exit(-1);
    }

    int npgms = atoi(argv[1]);
    int ntracenodes = atoi(argv[2]);
    int buf_size = atoi(argv[3]);

    fprintf(stderr, "Proggrams:\t%d\nNodes:\t\t%d\nBuf:\t\t%d\n", npgms, ntracenodes, buf_size);

#if 1
    std::vector<Program> programs = setup_programs();
    std::vector<TraceNode> trace = setup_trace();
#else
    std::vector<Program> programs;
    std::vector<TraceNode> trace;
    // Random is a terrible model for all of this
    // Size and cost are definitely correlated
    // Traces likely have coherence or at least some pgms that are reused while others aren't
    for (int i = 0; i < npgms; i++) {
        std::uint32_t size = rand() % 1024 * 15 + 1024;
        std::uint32_t cost = rand() % 1000;
        programs.push_back({size, cost});
    }

    for (int i = 0; i < ntracenodes; i++) {
        trace.push_back(rand() % npgms);
    }
#endif

    WorkerBufferManager wbm(buf_size);
    wbm.process_trace(trace, programs);
    wbm.dump(trace, programs);
    wbm.verify(trace, programs);
    wbm.dump_stats(trace, programs);
}
