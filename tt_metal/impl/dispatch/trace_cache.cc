#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <list>
#include <cstdint>
#include <optional>
#include <functional>
#include <string>
#include <assert.h>

// TODOs (some sprinkled throughout below):
// Don't use global rb size (shrink after peak(s))
// Allow allocation prior to peak rb size (look ahead window of N)
// Use rb space w/ full sync. Reset RB after? (expensive) (or use allocation fence)
// Score reuse_window of eg 4, 3, 2, 1, pick best
// On failure, fall back to ring buffer.  Maybe fallback on too many stalls
// Re-enter pre-allocate mode w/ large free node at bottom
// Re-enter pre-allocate mode after sync
// Merge free blocks?

// Number of programs executed before eviction
// For now this is a const.  Could be a min combined w/ an allocationsize
// heuritic for the max
constexpr std::int32_t reuse_window = 2;

class Program {
private:
    std::uint32_t size_;
    std::uint32_t cost_;

public:
    Program(std::uint32_t size, std::uint32_t cost);

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
    std::uint32_t rb_size_;      // size of the rb at this point
    std::int32_t rb_reset_idx_;  // TraceNode to stall on to reset rb (or -1)
    bool does_dispatch_;         // True when data is not resident
    std::int32_t stall_idx_;     // TraceNode idx this node must stall on (memory reuse)

public:
    TraceNode(uint32_t program_id);

    std::uint32_t get_pgm_id() const { return this->pgm_id_; }
    std::int32_t get_next_idx() const { return this->next_idx_; }
    std::int32_t get_remaining() const { return this->remaining_; }
    std::uint32_t get_addr() const { return this->addr_; }
    std::uint32_t get_rb_size() const { return this->rb_size_; }
    std::int32_t get_rb_reset_idx() const { return this->rb_reset_idx_; }
    bool does_dispatch() const { return this->does_dispatch_; }
    std::int32_t get_stall_idx() const { return this->stall_idx_; }
    bool does_stall() const { return this->stall_idx_ != -1; }
    float get_weight() const { return this->weight_; }
    bool is_allocated() const { return this->addr_ != unallocated_addr; }

    void set_remaining(std::int32_t remaining) { this->remaining_ = remaining; }
    void set_next_idx(std::uint32_t idx) { this->next_idx_ = idx; }
    void calculate_weight(float weight) { this->weight_ = weight * this->remaining_; }
    void set_addr(std::uint32_t addr) { this->addr_ = addr; }
    void set_rb_size(std::uint32_t size) { this->rb_size_ = size; }
    void set_rb_reset_idx(std::int32_t idx) { this->rb_reset_idx_ = idx; }
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
    rb_size_(0),
    rb_reset_idx_(-1),
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
        HEAP,
        RING_BUFFER,
    };

    std::int32_t count_;     // number of times this pgm was used in the trace
    std::int32_t used_idx_;  // trace node index of previous use of this pgm
    AllocationClass alloced_;

public:
    TraceProgramData();

    void use(std::int32_t idx) {
        this->used_idx_ = idx;
        this->count_++;
    }

    void inc_count() { this->count_++; }
    void alloc_heap() { this->alloced_ = HEAP; }
    void alloc_rb() { this->alloced_ = RING_BUFFER; }
    void dealloc() { this->alloced_ = NOT_ALLOCED; }

    std::int32_t get_count() const { return this->count_; }
    std::int32_t get_used() const { return this->used_idx_; }
    bool is_not_alloced() const { return this->alloced_ == NOT_ALLOCED; }
    bool is_heap() const { return this->alloced_ == HEAP; }
    bool is_rb() const { return this->alloced_ == RING_BUFFER; }
};

TraceProgramData::TraceProgramData() : count_(0), used_idx_(0), alloced_(NOT_ALLOCED) {}

// TODO
class TraceFence {
private:
    std::int32_t trace_idx1;
    std::int32_t trace_idx2;

public:
};

class AllocNode {
private:
    static constexpr std::int32_t never_used = -reuse_window;

    std::uint32_t pgm_id_;
    std::uint32_t addr_;       // base address of this allocation unit
    std::uint32_t size_;       // size of this allocation unit
    bool is_free_;             // true if this is just tracking unallocated space
    std::int32_t first_use_;   // trace id of when this was first used
    std::int32_t last_use_;    // trace id of when this was last used

public:
    AllocNode(std::uint32_t pgm_id, std::uint32_t addr, std::uint32_t size, std::int32_t used_idx);
    AllocNode(std::uint32_t addr, std::uint32_t size);

    std::uint32_t get_pgm_id() const { return this->pgm_id_; }
    std::uint32_t get_addr() const { return this->addr_; }
    std::uint32_t get_size() const { return this->size_; }
    std::int32_t get_first_use() const { return this->first_use_; }
    std::int32_t get_last_use() const { return this->last_use_; }
    bool is_free() const { return this->is_free_; }
    template <typename Container, typename Method>
    float get_weight(const Container& indexed_items, Method method) const;

    void set_last_use(std::int32_t trace_idx) { this->last_use_ = trace_idx; }
    void set_addr(std::uint32_t addr) { this->addr_ = addr; }
    void set_size(std::uint32_t size) { this->size_ = size; }
};

AllocNode::AllocNode(std::uint32_t pgm_id, std::uint32_t addr, std::uint32_t size, std::int32_t used_idx) :
    pgm_id_(pgm_id), addr_(addr), size_(size), is_free_(false), first_use_(used_idx), last_use_(used_idx) {}

AllocNode::AllocNode(std::uint32_t addr, std::uint32_t size) :
    pgm_id_(-1), addr_(addr), size_(size), is_free_(true), first_use_(never_used), last_use_(never_used) {}

// AllocNodes for free blocks don't have trace nodes
// The weight is in the trace node when alloced, so indirect through the trace
// to get there
template <typename Container, typename Method>
float AllocNode::get_weight(const Container& indexed_items, Method method) const {
    if (is_free_) {
        return 0.0f;
    }

    return (indexed_items[this->last_use_].*method)();
}

void dump_allocator(const std::list<AllocNode> allocator, const std::list<std::list<AllocNode>::iterator> lru) {
    fprintf(stderr, "Allocator:\n");
    for (const auto& node : allocator) {
        std::uint32_t pgm_id = node.get_pgm_id();
        fprintf(stderr, "  %s: %d %d\n", Program::get_name(pgm_id).c_str(), node.get_addr(), node.get_size());
    }

    fprintf(stderr, "LRU:\n");
    for (const auto& node : lru) {
        std::uint32_t pgm_id = node->get_pgm_id();
        fprintf(stderr, "  %s: last use %d\n", Program::get_name(pgm_id).c_str(), node->get_last_use());
    }
}

struct RingBufferState {
    std::vector<AllocNode> rb_;
    std::int32_t evict_idx_;    // idx into rb of successor rb's next eviction
    std::uint32_t evict_addr_;  // addr in rb of succsessor rb's next allocation

    RingBufferState() : evict_idx_(0), evict_addr_(0) {}
};

class WorkerBufferManager {
private:
    std::uint32_t buffer_size_;
    std::vector<TraceProgramData> program_data_;
    RingBufferState rb_;
    std::uint32_t rb_max_size_;

    std::list<AllocNode> allocator_;
    std::list<std::list<AllocNode>::iterator> lru_;
    std::vector<std::optional<std::list<std::list<AllocNode>::iterator>::iterator>> alloced_pgms_;

    // Init
    std::vector<TraceProgramData> build_use_data(std::vector<TraceNode>& trace, const std::vector<Program>& programs);

    // Allocate
    bool handle_trivial_cases(std::vector<TraceNode>& trace, const std::vector<Program>& programs);
    RingBufferState alloc_ring_buffer(std::vector<TraceNode>& trace, const std::vector<Program>& programs);
    void post_process_ring_buffer(
        const RingBufferState& rb, std::vector<TraceNode>& trace, const std::vector<Program>& programs);
    void alloc_heap(std::vector<TraceNode>& trace, const std::vector<Program>& programs);

    // Helpers
    bool check_rb_evictable_at_idx(
        std::int32_t& rb_idx,
        std::uint32_t alloc_addr,
        const std::vector<AllocNode>& rb,
        std::int32_t size_left,
        std::int32_t trace_idx);
    std::list<std::list<AllocNode>::iterator>::iterator find_eviction_candidates(
        std::int32_t window, std::uint32_t size_needed, std::int32_t node_idx, const std::vector<TraceNode>& trace);
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
};

WorkerBufferManager::WorkerBufferManager(std::uint32_t buffer_size) : buffer_size_(buffer_size) {}

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

        if (pgm_id >= program_data.size()) {
            program_data.resize(pgm_id + 1);
        }

        if (program_data[pgm_id].get_count() != 0) {
            trace[trace_idx].set_next_idx(program_data[pgm_id].get_used());
            if (programs[pgm_id].get_cost() > max_cost) {
                max_cost = programs[pgm_id].get_cost();
            }
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

        // TODO: move this to build_use_data?  in production code, probably just assert (checked elsewhere?)
        if (size_needed > this->buffer_size_) {
            fprintf(stderr, "Program %d's size %d exceeds buffer size %d\n", pgm_id, size_needed, this->buffer_size_);
            throw("");
        }

        total_alloced += size_needed;
    }

    if (total_alloced <= this->buffer_size_) {
        fprintf(stderr, "Trivial fit, allocating statically\n");

        std::uint32_t alloc_addr = 0;
        for (std::int32_t trace_idx = 0; trace_idx < (std::int32_t)trace.size(); trace_idx++) {
            if (not trace[trace_idx].is_allocated()) {
                std::uint32_t pgm_id = trace[trace_idx].get_pgm_id();
                std::int32_t child_idx = trace_idx;
                // TODO: move sentinal values behind the interface
                while (child_idx != 0x7fffffff) {
                    trace[child_idx].set_addr(alloc_addr);
                    child_idx = trace[child_idx].get_next_idx();
                }

                alloc_addr += programs[pgm_id].get_size();
                this->program_data_[pgm_id].alloc_heap();
                trace[trace_idx].set_does_dispatch();
            }
        }

        return true;
    }

    return false;
}

// Tries to allocate size_left bytes in rb by evicting at alloc_addr in slot
// rb_idx (and beyond) if those slots are beyond the reuse window of trace_idx
//
// Returns true on success w/ rb_idx updated (alloc_addr needs to be udpated
// by caller)
bool WorkerBufferManager::check_rb_evictable_at_idx(
    std::int32_t& rb_idx,
    std::uint32_t alloc_addr,
    const std::vector<AllocNode>& rb,
    std::int32_t size_left,
    std::int32_t trace_idx) {
    bool big_enough = false;
    std::int32_t idx = rb_idx;

    while (size_left > 0) {
        // Don't re-use too soon
        if ((std::int32_t)rb.size() > idx && rb[idx].get_last_use() + reuse_window < trace_idx) {
            alloc_addr = rb[idx].get_addr();
            std::uint32_t size_avail = rb[idx].get_size() - (alloc_addr - rb[idx].get_addr());
            size_left -= size_avail;
            if (size_left <= 0) {
                rb_idx = idx;
                big_enough = true;
                break;
            }
        } else {
            break;
        }

        idx++;
    }

    if (idx == (std::int32_t)rb.size()) {
        // If here, we actuallly succeeded but need to extend the RB beyond the prior iteration
        big_enough = true;
    }

    return big_enough;
}

void WorkerBufferManager::post_process_ring_buffer(
    const RingBufferState& rb, std::vector<TraceNode>& trace, const std::vector<Program>& programs) {
    // Ugh.  Simulate the RB to figure out the stalls
    // START HERE
    std::uint32_t max_size = 0;
    std::uint32_t last_rb_use = 0xffffffff;
    for (std::int32_t trace_idx = 0; trace_idx < (std::int32_t)trace.size(); trace_idx++) {
        if (trace[trace_idx].is_allocated()) {
            std::uint32_t pgm_id = trace[trace_idx].get_pgm_id();

            if (trace[trace_idx].get_addr() + programs[pgm_id].get_size() > max_size) {
                max_size = trace[trace_idx].get_addr() + programs[pgm_id].get_size();
            }

            if (trace[trace_idx].get_addr() == 0) {
                // Wrapped
                if (last_rb_use != 0xffffffff) {
                    trace[trace_idx].set_stall_idx(last_rb_use);
                }
            }

            last_rb_use = trace_idx;
        }
    }
    this->rb_max_size_ = max_size;
}

// I believe this solution is optimal and O(n) given these simplifications:
//  - fixed lookback (reuse window)
//  - no gaps in the allocations
// (those additions would yield a better solution with higher complexity)
//
// It works by looking to the base of the ring buffer and if that allocation
// is old enough, replacing it. Otherwise, allocate by freeing if possible.
// If freeing the next node doesn't satisify the lookback, then move the base
// of the ring buffer to the top and try again from there. Since the lookback
// is fixed, this isn't restarting back as a f(n) but instead as f(c), hence
// the overall O(n)
//
// A "better" solution could be found at O(n^2) if gaps are allowed. In this
// case, rather than moving the base to the top, perhaps the previous
// iteration could have left a gap such that the new allocation wouldn't need
// to evict.  Doing this could ripple all the way back, so, actually this may
// be worse than O(n^2)
//
// Implementation below keeps each iteration through the ring buffer in
// a vector of RingBufferState for simplicity of rolling back.  If we roll
// back (buffer n-1) we need to look at the previous buffer for evictions
// (n-2), though since we were able to promote previously, no lookback checks
// will fail, however, bookkeeping that information is messy relative to just
// looking back and checking the state
RingBufferState WorkerBufferManager::alloc_ring_buffer(
    std::vector<TraceNode>& trace, const std::vector<Program>& programs) {
    // TODO: handle large allocations that won't fit all at once within lookback

    // Track prior ring buffers so we can roll back
    // Can only go back 2, but simpler to track them all
    std::vector<RingBufferState> rbs;
    rbs.push_back({});

    fprintf(stderr, "+++++Starting ring buffer allocation\n");

    for (std::int32_t trace_idx = 0; trace_idx < (std::int32_t)trace.size(); trace_idx++) {
        fprintf(stderr, "Processing idx %d\n", trace_idx);
        RingBufferState& rb = rbs.back();

        std::uint32_t pgm_id = trace[trace_idx].get_pgm_id();
        std::uint32_t size_needed = programs[pgm_id].get_size();

        // Only count==1 goes in RB
        if (this->program_data_[pgm_id].get_count() == 1) {
            std::uint32_t alloc_addr;

            // Cases:
            // 1: allocate at the front (wraps the RB) if possible
            // 2a: allocate in the next slot by evicting from previous iteration
            // 2b: if slot isn't beyond the lookback window, roll back
            // 3: we're at the end, grow the RB

            std::int32_t rb_idx = 0;
            if ((std::int32_t)rbs.size() == 1 && rb.rb_.size() == 0) {
                // Simplifies code below to insert the first element here
                fprintf(stderr, "Inserting first element\n");
                std::uint32_t pgm_id = trace[trace_idx].get_pgm_id();
                std::uint32_t size_needed = programs[pgm_id].get_size();
                rb.rb_.push_back({pgm_id, 0, size_needed, trace_idx});
                alloc_addr = 0;
            } else if (check_rb_evictable_at_idx(rb_idx, 0, rb.rb_, size_needed, trace_idx)) {
                // Case 1: moved to front, wrapped the current ring buffer
                fprintf(stderr, "Moving to front\n");
                rb.evict_idx_ = rb_idx;
                rb.evict_addr_ = size_needed;
                rbs.push_back({});
                alloc_addr = 0;
                rbs.back().rb_.push_back({pgm_id, alloc_addr, size_needed, trace_idx});
            } else {
                if (rbs.size() > 1) {
                    RingBufferState& prev_rb = rbs[rbs.size() - 2];
                    std::int32_t evict_idx = prev_rb.evict_idx_;
                    if (check_rb_evictable_at_idx(
                            evict_idx, prev_rb.evict_addr_, prev_rb.rb_, size_needed, trace_idx)) {
                        // Case 2a, evict
                        fprintf(stderr, "Evicting from prior round\n");
                        prev_rb.evict_idx_ = evict_idx;
                        alloc_addr = prev_rb.evict_addr_;
                        prev_rb.evict_addr_ += size_needed;
                        rb.rb_.push_back({pgm_id, alloc_addr, size_needed, trace_idx});
                    } else {
                        // Case 2b, roll back
                        fprintf(stderr, "Rolling back\n");
                        AllocNode old_entry = rb.rb_[0];
                        rbs.pop_back();
                        alloc_addr = rb.rb_.back().get_addr() + rb.rb_.back().get_size();
                        trace_idx = old_entry.get_last_use();
                        AllocNode new_entry(old_entry.get_pgm_id(), alloc_addr, old_entry.get_size(), trace_idx);
                        rb.rb_.push_back(new_entry);
                    }
                } else {
                    // Case 3, grow the rb
                    fprintf(stderr, "Growing the rb\n");
                    alloc_addr =
                        (rbs[0].rb_.size() == 0) ? 0 : rbs[0].rb_.back().get_addr() + rbs[0].rb_.back().get_size();
                    rbs[0].rb_.push_back({pgm_id, alloc_addr, size_needed, trace_idx});
                }
            }

            this->program_data_[pgm_id].alloc_rb();
            trace[trace_idx].set_addr(alloc_addr);
            trace[trace_idx].set_does_dispatch();
        }

        if (rbs.back().rb_.size() != 0) {
            trace[trace_idx].set_rb_size(rbs.back().rb_.back().get_addr() + rbs.back().rb_.back().get_size());
            if (!this->program_data_[pgm_id].is_rb()) {
                trace[trace_idx].set_rb_reset_idx(rbs.back().rb_.back().get_last_use());
            }
        }
    }

    post_process_ring_buffer(rbs.back(), trace, programs);

    return rbs.back();
}

void WorkerBufferManager::commit_preallocations(
    std::uint32_t addr_top, std::list<AllocNode>::iterator commit_start, std::vector<TraceNode>& trace) {
    std::uint32_t addr = addr_top;

    std::list<AllocNode>::iterator alloc_it =
        (commit_start == this->allocator_.end()) ? this->allocator_.begin() : commit_start;
    while (alloc_it != this->allocator_.end()) {
        if (not alloc_it->is_free()) {  // hmm, can this happen?
            std::int32_t trace_idx = alloc_it->get_first_use();
            std::uint32_t pgm_id = alloc_it->get_pgm_id();
            std::uint32_t size = alloc_it->get_size();
            addr -= size;
            alloc_it->set_addr(addr);
            this->program_data_[pgm_id].alloc_heap();
            trace[trace_idx].set_does_dispatch();
            fprintf(stderr, "addr: %d %d\n", addr, size);

            // Commit allocations up to how far we've progressed through this trace
            while (trace_idx <= alloc_it->get_last_use()) {
                trace[trace_idx].set_addr(alloc_it->get_addr());
                trace_idx = trace[trace_idx].get_next_idx();
            }
        }

        alloc_it++;
    }
}

// The LRU side of the LRU list contains the free nodes, free nodes are at the
// the old end, so we start w/ those
// Walk from LRU to MRU:
//   aggregate nodes from low memory to high memory to meet the size requirement
//   fail a required node doesn't has been used within the reuse threshold
//   score sets of nodes
//   select the set w/ the lowest score
// Returns an iterator into LRU of the first (lowest address) node to use, points to end if not
// If this call fails, call it again w/ a lower reuse threshold
std::list<std::list<AllocNode>::iterator>::iterator WorkerBufferManager::find_eviction_candidates(
    std::int32_t window, std::uint32_t size_needed, std::int32_t trace_idx, const std::vector<TraceNode>& trace) {
    constexpr float best_score = std::numeric_limits<float>::infinity();
    std::list<std::list<AllocNode>::iterator>::iterator match = this->lru_.end();

    fprintf(stderr, "Trying to find eviction candidate for size %d\n", size_needed);
    for (auto lru_it = this->lru_.begin(); lru_it != this->lru_.end(); lru_it++) {
        std::uint32_t free_size = 0;
        float score = 0.0f;
        bool found_one = false;

        // Iterates over allocator from high to low memory
        for (auto alloc_it = *lru_it; alloc_it != this->allocator_.end(); alloc_it++) {
            if (alloc_it->get_last_use() + window > trace_idx) {
                break;  // failed
            }

            free_size += alloc_it->get_size();
            score += alloc_it->get_weight(trace, &TraceNode::get_weight);
            if (size_needed <= free_size) {
                found_one = true;
                break;
            }
        }

        if (found_one && score < best_score) {
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

    std::int32_t stall_idx = -1;
    while (freed_size < size_needed) {
        freed_size += alloc_it->get_size();
        // TODO: could use back-iterators from allocator to LRU to save this search
        for (auto lru_it = this->lru_.begin(); lru_it != this->lru_.end(); lru_it++) {
            if (*lru_it == alloc_it) {
                // TODO: clean this up (sentinel)
                if (alloc_it->get_pgm_id() != 0xffffffff) {
                    this->program_data_[alloc_it->get_pgm_id()].dealloc();
                    this->alloced_pgms_[alloc_it->get_pgm_id()].reset();
                }
                if (alloc_it->get_last_use() > stall_idx) {
                    stall_idx = alloc_it->get_last_use();
                }
                this->lru_.erase(lru_it);
                alloc_it = this->allocator_.erase(alloc_it);
                break;
            }
        }
    }

    trace[trace_idx].set_stall_idx(stall_idx);

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

        // Add an empty free node at the end down to the top of the ring buffer
        // This simplifies book keeping by not having to special case the gap
        if (alloc_addr > this->rb_max_size_) {
            this->allocator_.push_back({this->rb_max_size_, alloc_addr - this->rb_max_size_});
            this->lru_.push_front(--this->allocator_.end());
        }
    } else {
        // Is there a hole?  Which side does it go on?  Merge if adjacent
        std::uint32_t base_addr = alloc_it->get_addr() + alloc_it->get_size();
        bool hole_top = false;
        if (alloc_it != this->allocator_.begin()) {
            auto prev_it = std::prev(alloc_it);
            if (prev_it->get_weight(trace, &TraceNode::get_weight) <
                alloc_it->get_weight(trace, &TraceNode::get_weight)) {
                hole_top = true;
            } else {
                hole_top = false;
            }
        }
        if (hole_top) {
            alloc_addr = base_addr;
            auto new_it = this->allocator_.insert(alloc_it, {base_addr + size_needed, freed_size - size_needed});
            this->lru_.push_front(new_it);
            new_it = this->allocator_.insert(alloc_it, {pgm_id, alloc_addr, size_needed, trace_idx});
            this->lru_.push_back(new_it);
        } else {
            alloc_addr = base_addr + freed_size - size_needed;
            auto new_it = this->allocator_.insert(alloc_it, {pgm_id, alloc_addr, size_needed, trace_idx});
            this->lru_.push_back(new_it);
            new_it = this->allocator_.insert(alloc_it, {base_addr, freed_size - size_needed});
            this->lru_.push_front(new_it);
        }
    }

    // TODO: aggegate all this bookkeeping in a helper?
    trace[trace_idx].set_does_dispatch();
    trace[trace_idx].set_addr(alloc_addr);
    this->program_data_[pgm_id].alloc_heap();
    this->alloced_pgms_[pgm_id] = std::prev(this->lru_.end());
}

void WorkerBufferManager::sort_preallocations(
    std::list<AllocNode>::iterator uncommitted_it, const std::vector<TraceNode>& trace) {
    // Sort just the sub-range that we pre-allocated since last commit
    std::list<AllocNode> tmp;
    tmp.splice(tmp.end(), this->allocator_, uncommitted_it, this->allocator_.end());
    tmp.sort([&](const AllocNode& first, const AllocNode& second) {
        float first_weight = first.get_weight(trace, &TraceNode::get_weight);
        float second_weight = second.get_weight(trace, &TraceNode::get_weight);

        if (first_weight == second_weight) {
            // Probably weights are 0, sort so newest is at the bottom of heap
            // Allocate from the tie point down to minimize syncing
            return first.get_last_use() > second.get_last_use();
        } else {
            return first_weight > second_weight;
        }
    });

    this->allocator_.splice(this->allocator_.end(), tmp);
}

// Walk the allocation backwards from the bottom, look for the last
// entry which doesn't have a next use. Set addr_top and retry
// Note: these are sorted above from newest use down to oldest use;
// this is simple but not optimal.  Sorting the other way and tracking if
// we need a sync could give a better solution (use a slot up high, then
// sync after that which defers the sync)
bool WorkerBufferManager::try_to_reenter_preallocation_mode(
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
        std::int32_t last_use = node.get_last_use();
        pre_alloc_addr = pre_alloc_addr_top = node.get_addr();
        if (last_use + reuse_window >= trace_idx || trace[last_use].get_remaining() != 0) {
            done = true;
        } else {
            this->program_data_[node.get_pgm_id()].dealloc();
            this->alloced_pgms_[node.get_pgm_id()].reset();
            stall_idx = node.get_last_use();
            // Ugh, search through LRU.  Should be near the front so not too deep a search
            auto lru_it = this->lru_.begin();
            auto lru_end = std::prev(this->allocator_.end());
            while (1) {
                if (*lru_it == lru_end) {
                    this->lru_.erase(lru_it);
                    break;
                }
                lru_it++;
            }
            this->allocator_.pop_back();
            eviction_mode = false;
        }
    }

    if (eviction_mode) {
        fprintf(stderr, "Transitioning to eviction mode with %ld allocations\n", this->allocator_.size());

        // Add an empty free node at the end down to the top of the ring buffer
        // This simplifies book keeping by not having to special case the gap
        std::uint32_t addr = this->allocator_.back().get_addr();
        if (addr > this->rb_max_size_) {
            this->allocator_.push_back({this->rb_max_size_, addr - this->rb_max_size_});
            this->lru_.push_front(--this->allocator_.end());
        }
    } else {
        uncommitted_it = std::prev(this->allocator_.end());
        fprintf(
            stderr, "Staying in pre-allocation mode: %ld at addr %d\n", this->allocator_.size(), pre_alloc_addr_top);
    }

    trace[trace_idx].set_stall_idx(stall_idx);

    return eviction_mode;
}

// General approach:
// 1) Track total allocated size up until the heap collides w/ the ring buffer
// 2) At this point, allocate based on last seen priority for the pgm putting
// high priority at the top and low priority at the bottom
// 3) Free up what can be freed up and restart to first step
// 4) At some point this may exhaust itself and we have to enter eviction mode
// 5) Evict based on LRU and hole size from here on out
void WorkerBufferManager::alloc_heap(std::vector<TraceNode>& trace, const std::vector<Program>& programs) {
    this->alloced_pgms_.resize(programs.size());
    std::int32_t trace_idx = 0;

    fprintf(stderr, "+++++Starting heap allocation from %d to %d\n", this->rb_max_size_, this->buffer_size_);

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
            this->lru_.back()->set_last_use(trace_idx);
        } else if (this->program_data_[pgm_id].get_count() != 1) {
            // TODO: consider if get_count()==0 regenerating the ring buffer from here on
            std::uint32_t size_needed = programs[pgm_id].get_size();

            if (eviction_mode) {
                fprintf(stderr, "Eviction mode: %d\n", trace_idx);

                std::list<std::list<AllocNode>::iterator>::iterator evict_it;
                for (std::int32_t window = reuse_window; window != 0; window--) {
                    fprintf(stderr, "Trying eviction window: %d\n", window);
                    evict_it = find_eviction_candidates(window, size_needed, trace_idx, trace);
                    if (evict_it != this->lru_.end()) {
                        break;
                    }
                }
                if (evict_it == this->lru_.end()) {
                    fprintf(stderr, "Failed allocate %d bytes at trace_idx %d\n", size_needed, trace_idx);
                    // TODO: need to be able to stall on the RB and vice versa to handle all cases
                    throw("failed");
                } else {
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
                }
            } else {
                fprintf(stderr, "Pre-allocation mode: %d\n", trace_idx);

                assert(this->program_data_[pgm_id].is_not_alloced());

                // TODO: replace static rb size w/ dynamic rb size

                if ((int)pre_alloc_addr - (int)size_needed < (int)this->rb_max_size_) {
                    fprintf(
                        stderr,
                        "Committing prior allocations on trace node %d start at addr %d\n",
                        trace_idx,
                        pre_alloc_addr_top);

                    // Allocate based on priority, then either evict from bottom or transition to eviction mode
                    sort_preallocations(uncommitted_it, trace);
                    commit_preallocations(pre_alloc_addr_top, uncommitted_it, trace);
                    eviction_mode = try_to_reenter_preallocation_mode(
                        pre_alloc_addr_top, pre_alloc_addr, uncommitted_it, trace_idx, trace);

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
        fprintf(stderr, "at the end\n");
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
        // Eventually may want to interleave these two
        this->rb_ = alloc_ring_buffer(trace, programs);
        alloc_heap(trace, programs);
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

        if (trace[trace_idx].does_stall() && last_sync_idx < trace[trace_idx].get_stall_idx()) {
            last_sync_idx = trace[trace_idx].get_stall_idx();
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
                            "New allocation of node %d (%d, %d) overlaps existing allocation (%d, %d) - missing "
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

                alloc_it++;
            }

            allocator.push_back({pgm_id, addr, size, trace_idx});
        } else {
            bool found = false;
            for (auto alloc_it = allocator.begin(); alloc_it != allocator.end(); alloc_it++) {
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
        fprintf(stderr, " %c", (this->program_data_[pgm_id].get_count() == 1) ? 'R' : 'H');
        fprintf(
            stderr,
            "[%5d-%5d) %c S:%2d %5d %2d",
            addr,
            addr + programs[pgm_id].get_size(),
            trace[trace_idx].does_dispatch() ? 'D' : 'x',
            trace[trace_idx].get_stall_idx(),
            trace[trace_idx].get_rb_size(),
            trace[trace_idx].get_rb_reset_idx());
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

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: cache <npgms> <ntracenodes> <bufsize>\n");
        exit(-1);
    }

    int npgms = atoi(argv[1]);
    int ntracenodes = atoi(argv[2]);
    int buf_size = atoi(argv[3]);

    fprintf(stderr, "Proggrams:\t%d\nNodes:\t\t%d\nBuf:\t\t%d\n", npgms, ntracenodes, buf_size);

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

    WorkerBufferManager wbm(buf_size);
    wbm.process_trace(trace, programs);
    wbm.dump(trace, programs);
    wbm.verify(trace, programs);
    wbm.dump_stats(trace, programs);
}
