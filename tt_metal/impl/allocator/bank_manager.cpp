// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "bank_manager.hpp"

#include <enchantum/enchantum.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/allocator_state.hpp>
#include <limits>
#include <string_view>
#include <utility>
#include <algorithm>

#include "allocator/algorithms/allocator_algorithm.hpp"
#include <tt_stl/assert.hpp>
#include "buffer_types.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-logger/tt-logger.hpp>
#include "tt_metal/impl/allocator/algorithms/free_list_opt.hpp"

namespace tt::tt_metal {

BankManager::AllocatorDependencies::AllocatorDependencies() = default;

BankManager::AllocatorDependencies::AllocatorDependencies(
    const std::unordered_map<AllocatorID, ttsl::SmallVector<AllocatorID>>& dependencies_map) {
    // Determine total number of allocators as max allocator id seen anywhere (keys or values)
    if (dependencies_map.empty()) {
        dependencies = AdjacencyList{{}};
        return;
    }

    uint32_t max_id = 0;
    for (const auto& [allocator_id, dependencies] : dependencies_map) {
        max_id = std::max(max_id, allocator_id.get());
        for (const auto dep_allocator_id : dependencies) {
            max_id = std::max(max_id, dep_allocator_id.get());
        }
    }
    const uint32_t num_allocators = static_cast<uint32_t>(max_id) + 1;  // +1 because allocator ids are zero-indexed

    // Use set form to ensure uniqueness while building undirected adjacency list
    ttsl::SmallVector<std::unordered_set<uint32_t>> undirected_sets(num_allocators);

    // Build undirected adjacency: for each edge (u -> v), add v to u and u to v
    for (const auto& [allocator_id, dependencies] : dependencies_map) {
        // Ensure uniqueness of dependency values per allocator (catch duplicates in provided list)
        std::unordered_set<uint32_t> seen_values;
        for (const auto dep_allocator_id : dependencies) {
            const bool inserted = seen_values.insert(dep_allocator_id.get()).second;
            TT_FATAL(
                inserted,
                "Duplicate dependency for allocator {}: {} appears more than once!",
                allocator_id.get(),
                dep_allocator_id.get());
        }

        for (const auto dep_allocator_id : dependencies) {
            const uint32_t u = allocator_id.get();
            const uint32_t v = dep_allocator_id.get();
            undirected_sets[u].insert(v);
            undirected_sets[v].insert(u);
        }
    }

    // Build dependencies list: for each allocator s, which allocators s depends on / depends on s
    // Note: Need to sort dependencies because dependencies_map is unordered
    dependencies.clear();
    dependencies.resize(num_allocators);
    for (uint32_t i = 0; i < num_allocators; ++i) {
        auto& dst = dependencies[i];
        dst.assign(undirected_sets[i].begin(), undirected_sets[i].end());
        std::sort(dst.begin(), dst.end());
    }
}

uint32_t BankManager::AllocatorDependencies::num_allocators() const {
    return static_cast<uint32_t>(dependencies.size());
}

ttsl::SmallVector<BankManager::AllocatorDependencies::AllocatorID> BankManager::AllocatorDependencies::allocator_ids()
    const {
    ttsl::SmallVector<AllocatorID> allocators;
    allocators.reserve(num_allocators());
    for (size_t i = 0; i < num_allocators(); i++) {
        allocators.push_back(AllocatorID{static_cast<uint32_t>(i)});
    }
    return allocators;
}

bool BankManager::AllocatorDependencies::operator==(const BankManager::AllocatorDependencies& other) const noexcept {
    if (dependencies.size() != other.dependencies.size()) {
        return false;
    }
    // Direct comparison assuming dependencies are already sorted
    for (size_t i = 0; i < dependencies.size(); ++i) {
        if (dependencies[i] != other.dependencies[i]) {
            return false;
        }
    }
    return true;
}

void BankManager::init_allocators(DeviceAddr size_bytes, uint32_t alignment_bytes, DeviceAddr offset) {
    const uint32_t n = allocator_dependencies_.num_allocators();
    allocators_.resize(n);
    allocated_buffers_.resize(n);
    allocated_ranges_cache_.resize(n);

    for (uint32_t allocator_id = 0; allocator_id < n; ++allocator_id) {
        allocators_[allocator_id] = std::make_unique<allocator::FreeListOpt>(
            size_bytes, offset, alignment_bytes, alignment_bytes, allocator::FreeListOpt::SearchPolicy::FIRST);
    }
}

void validate_num_banks(uint32_t num_banks, const BufferType& buffer_type, bool disable_interleaved) {
    bool doesnt_support_interleaved = buffer_type == BufferType::L1_SMALL or disable_interleaved;
    bool is_pow2_num_banks = num_banks && (!(num_banks & (num_banks - 1)));
    // Dataflow API does not have a working implementation of generic modulo to determine bank_id for interleaved
    // address gen For non pow2 num banks, special cases need to be added to avoid falling back to generic
    // implementation. See https://github.com/tenstorrent/tt-metal/issues/3321
    std::unordered_set<uint32_t> acceptable_num_non_pow2_mem_banks = {
        7, 12, 20, 48, 56, 63, 70, 72, 80, 94, 108, 110, 117, 120, 124, 126, 130, 140};
    bool custom_mod_bank_id_calculation_exists = acceptable_num_non_pow2_mem_banks.contains(num_banks);
    bool valid_num_banks = (is_pow2_num_banks or custom_mod_bank_id_calculation_exists or doesnt_support_interleaved);
    if (not valid_num_banks) {
        TT_THROW(
            "Invalid number of memory banks {} for {}. Num banks must be power of 2 or have a dedicated modulo "
            "implementation",
            num_banks,
            enchantum::to_string(buffer_type));
    }
}

BankManager::BankManager(
    const BufferType& buffer_type,
    const std::vector<int64_t>& bank_offsets,
    DeviceAddr size_bytes,
    uint32_t alignment_bytes,
    DeviceAddr alloc_offset,
    bool disable_interleaved,
    const AllocatorDependencies& dependencies) :
    buffer_type_(buffer_type),

    alignment_bytes_(alignment_bytes),
    allocator_dependencies_(dependencies) {
    unsigned int bank_id = 0;
    for (const auto bank_offset : bank_offsets) {
        bank_id_to_bank_offset_.insert({bank_id, bank_offset});
        bank_id++;
    }

    validate_num_banks(bank_id_to_bank_offset_.size(), buffer_type_, disable_interleaved);

    // Initialize all allocators; sets up allocator-dependent members
    this->init_allocators(size_bytes, MetalContext::instance().hal().get_alignment(HalMemType::DRAM), alloc_offset);
}

BankManager::BankManager(
    const BufferType& buffer_type,
    const std::unordered_map<uint32_t, int64_t>& bank_id_to_bank_offset,
    DeviceAddr size_bytes,
    DeviceAddr interleaved_address_limit,
    uint32_t alignment_bytes,
    DeviceAddr alloc_offset,
    bool disable_interleaved,
    const AllocatorDependencies& dependencies) :
    buffer_type_(buffer_type),
    bank_id_to_bank_offset_(bank_id_to_bank_offset),
    interleaved_address_limit_(interleaved_address_limit),
    alignment_bytes_(alignment_bytes),
    allocator_dependencies_(dependencies) {
    validate_num_banks(bank_id_to_bank_offset_.size(), buffer_type_, disable_interleaved);

    // Initialize all allocators; sets up allocator-dependent members
    this->init_allocators(size_bytes, MetalContext::instance().hal().get_alignment(HalMemType::DRAM), alloc_offset);
}

uint32_t BankManager::num_banks() const { return bank_id_to_bank_offset_.size(); }

DeviceAddr BankManager::bank_size() const {
    const auto* alloc0 = this->get_allocator_from_id(AllocatorDependencies::AllocatorID{0});
    TT_FATAL(alloc0, "Allocator not initialized!");
    return alloc0->max_size_bytes();
}

int64_t BankManager::bank_offset(uint32_t bank_id) const {
    this->validate_bank_id(bank_id);
    return bank_id_to_bank_offset_.at(bank_id);
}

void BankManager::validate_bank_id(uint32_t bank_id) const {
    TT_FATAL(
        bank_id_to_bank_offset_.contains(bank_id),
        "Expected bank {} to be tracked!",
        bank_id,
        bank_id_to_bank_offset_.size());
}

allocator::Algorithm* BankManager::get_allocator_from_id(BankManager::AllocatorDependencies::AllocatorID allocator_id) {
    TT_FATAL(
        allocator_id.get() < allocator_dependencies_.num_allocators(),
        "Invalid allocator ID {} (num_allocators={})",
        allocator_id.get(),
        allocator_dependencies_.num_allocators());
    return allocators_[allocator_id.get()] ? allocators_[allocator_id.get()].get() : nullptr;
}

const allocator::Algorithm* BankManager::get_allocator_from_id(
    BankManager::AllocatorDependencies::AllocatorID allocator_id) const {
    return const_cast<BankManager*>(this)->get_allocator_from_id(allocator_id);
}

void BankManager::invalidate_allocated_ranges_cache_for_dependent_allocators(
    BankManager::AllocatorDependencies::AllocatorID allocator_id) {
    TT_FATAL(
        allocator_id.get() < allocator_dependencies_.num_allocators(),
        "Invalid allocator ID {} (num_allocators={})",
        allocator_id.get(),
        allocator_dependencies_.num_allocators());
    const auto& dependent_allocators = allocator_dependencies_.dependencies[allocator_id.get()];
    for (const auto dep_allocator_id : dependent_allocators) {
        allocated_ranges_cache_[dep_allocator_id.get()].reset();
    }
}

const std::vector<std::pair<DeviceAddr, DeviceAddr>>& BankManager::compute_merged_allocated_ranges(
    BankManager::AllocatorDependencies::AllocatorID allocator_id) {
    TT_FATAL(
        allocator_id.get() < allocator_dependencies_.num_allocators(),
        "Invalid allocator ID {} (num_allocators={})",
        allocator_id.get(),
        allocator_dependencies_.num_allocators());

    // Return cached value if available
    if (allocated_ranges_cache_[allocator_id.get()].has_value()) {
        return allocated_ranges_cache_[allocator_id.get()].value();
    }

    // Collect allocated address ranges per dependent allocator (single pass)
    const auto& dependent_allocators = allocator_dependencies_.dependencies[allocator_id.get()];
    std::vector<std::pair<DeviceAddr, DeviceAddr>> all_allocated_ranges;
    for (const auto dep_allocator_id : dependent_allocators) {
        auto* dep_alloc = this->get_allocator_from_id(dep_allocator_id);
        TT_FATAL(dep_alloc, "Allocator not initialized!");
        const auto allocated_addresses = dep_alloc->allocated_addresses();
        all_allocated_ranges.reserve(all_allocated_ranges.size() + allocated_addresses.size());
        for (const auto& [addr, end_addr] : allocated_addresses) {
            if (end_addr > addr) {
                all_allocated_ranges.emplace_back(addr, end_addr);
            }
        }
    }

    // Sort allocated ranges across all dependent allocators by start address
    std::sort(all_allocated_ranges.begin(), all_allocated_ranges.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // Coalesce overlaps across all dependent allocators
    std::vector<std::pair<DeviceAddr, DeviceAddr>> coalesced_ranges;
    coalesced_ranges.reserve(all_allocated_ranges.size());
    for (const auto& r : all_allocated_ranges) {
        if (coalesced_ranges.empty() || r.first > coalesced_ranges.back().second) {
            coalesced_ranges.push_back(r);
        } else {
            coalesced_ranges.back().second = std::max(coalesced_ranges.back().second, r.second);
        }
    }

    allocated_ranges_cache_[allocator_id.get()] = std::move(coalesced_ranges);
    return allocated_ranges_cache_[allocator_id.get()].value();
}

std::vector<std::pair<DeviceAddr, DeviceAddr>> BankManager::compute_available_addresses(
    BankManager::AllocatorDependencies::AllocatorID allocator_id, DeviceAddr size_per_bank, DeviceAddr address_limit) {
    auto* alloc = this->get_allocator_from_id(allocator_id);
    TT_FATAL(alloc, "Allocator not initialized!");

    // Helper for clamping ranges in-place according to address_limit
    // This is needed because the allocator's available_addresses method does not clamp to address_limit
    auto clamp_ranges = [address_limit](std::vector<std::pair<DeviceAddr, DeviceAddr>>& ranges) {
        if (address_limit == 0) {
            return;
        }

        // First pass: clamp ranges in-place
        for (auto& r : ranges) {
            r.first = std::max(r.first, address_limit);
        }

        // Second pass: remove empty ranges (where r.second <= r.first)
        ranges.erase(
            std::remove_if(ranges.begin(), ranges.end(), [](const auto& r) { return r.second <= r.first; }),
            ranges.end());
    };

    // Current allocator's available ranges (clamped if needed)
    std::vector<std::pair<DeviceAddr, DeviceAddr>> available_ranges = alloc->available_addresses(size_per_bank);
    clamp_ranges(available_ranges);
    std::sort(available_ranges.begin(), available_ranges.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // Allocated ranges from dependent allocators; ranges are merged
    const auto& allocated_ranges_in_dependent_allocators = this->compute_merged_allocated_ranges(allocator_id);

    // Helper for subtracting allocated ranges from available ranges
    // Ranges consist of half-open intervals throughout: [start, end)
    // Assumptions:
    // - Each of 'available_ranges' and 'allocated_ranges' is individually sorted by start and internally
    // non-overlapping
    // Strategy:
    // - Two-pointer sweep: advance a single shared pointer 'j' through allocated ranges across all available ranges.
    //   Complexity O(|available_ranges| + |allocated_ranges|).
    // Example:
    //   available_ranges = [(10, 30)], allocated_ranges = [(12, 15), (18, 25)]
    //   result           = [(10, 12), (15, 18), (25, 30)]
    auto subtract_ranges = [](const std::vector<std::pair<DeviceAddr, DeviceAddr>>& available_ranges,
                              const std::vector<std::pair<DeviceAddr, DeviceAddr>>& allocated_ranges) {
        std::vector<std::pair<DeviceAddr, DeviceAddr>> out;
        out.reserve(available_ranges.size());

        // Monotonic pointer to scan all allocated ranges; we don't reset the pointer after each available range
        // This assumes available and allocated ranges are sorted and non-overlapping
        size_t j = 0;
        for (const auto& available : available_ranges) {
            DeviceAddr available_start = available.first;
            DeviceAddr available_end = available.second;
            // 1) Skip allocated ranges that are completely to the left of available range
            //    available:            [s --- e)
            //    allocated: [s --- e)
            while (j < allocated_ranges.size() && allocated_ranges[j].second <= available_start) {
                j++;
            }

            DeviceAddr cur_available_start = available_start;
            // 2) Scan allocated ranges that might overlap current available range
            //    Loop until we hit allocated ranges that are to the right of available range
            //    ie. stop until we hit ranges like this:
            //    available: [s --- e)
            //    allocated:            [s --- e)
            while (j < allocated_ranges.size() && allocated_ranges[j].first < available_end) {
                DeviceAddr allocated_start = allocated_ranges[j].first;
                DeviceAddr allocated_end = allocated_ranges[j].second;
                // Case A: Gap exists before the allocated range
                //   available: [cur --- e)
                //   allocated:   [s ---
                //   -> Keep free gap: [cur, allocated_s)
                if (allocated_start > cur_available_start) {
                    out.emplace_back(cur_available_start, allocated_start);
                }
                // Case B: The allocated range overlaps with rest of available range
                //   available: [cur ----- e)
                //   allocated:     [s ----- e)
                //   -> No remaining free space in [cur, e). Update cur_available_start to available_end and break
                if (allocated_end >= available_end) {
                    cur_available_start = available_end;
                    break;
                }
                // Case C: The allocated range ends before the end of available range
                //   available: [cur ----- e)
                //   allocated:   [s --- e)
                //   -> Update cur_available_start to the end of allocated range
                //      We continue scanning the next allocated range and emit any remaining free space at the tail
                cur_available_start = allocated_end;
                j++;
            }
            // 3) Emit remaining tail (if any) after scanning all allocated ranges
            if (cur_available_start < available_end) {
                out.emplace_back(cur_available_start, available_end);
            }
        }
        return out;
    };

    std::vector<std::pair<DeviceAddr, DeviceAddr>> updated_available_ranges =
        subtract_ranges(available_ranges, allocated_ranges_in_dependent_allocators);

    return updated_available_ranges;
}

uint64_t BankManager::allocate_buffer(
    DeviceAddr size,
    DeviceAddr page_size,
    bool bottom_up,
    const CoreRangeSet& compute_grid,
    std::optional<uint32_t> num_shards,
    BankManager::AllocatorDependencies::AllocatorID allocator_id) {
    auto* alloc = this->get_allocator_from_id(allocator_id);
    TT_FATAL(alloc, "Allocator not initialized!");

    uint32_t num_banks = this->num_banks();
    bool is_sharded = false;
    if (num_shards.has_value()) {
        auto num_compute_banks = compute_grid.num_cores();
        is_sharded = true;
        TT_FATAL(
            num_shards.value() <= num_compute_banks,
            "Expected number of shards {} to be less than or equal to total number of L1 banks {} in compute cores",
            num_shards.value(),
            num_compute_banks);
        num_banks = num_shards.value();
    }
    DeviceAddr size_per_bank = tt::tt_metal::detail::calculate_bank_size_spread(size, page_size, num_banks, alignment_bytes_);
    DeviceAddr address_limit = 0;
    if (!is_sharded and buffer_type_ == BufferType::L1) {
        address_limit = interleaved_address_limit_;
        TT_FATAL(address_limit > 0, "Address limit {} needs to be larger than zero.", address_limit);
    }

    // If there are no dependent allocators, fall back to allocator's single allocator strategy
    const auto& dependent_allocators = allocator_dependencies_.dependencies[allocator_id.get()];

    // If using single allocator strategy, Algorithm::allocate handles address limit, which means it can only be used
    // with top-down allocation. Otherwise, address limit is used to clamp the available ranges regardless of bottom_up
    // vs. top-down allocation
    if (dependent_allocators.empty()) {
        auto address = alloc->allocate(size_per_bank, bottom_up, address_limit);
        if (!address.has_value()) {
            auto mem_stats = alloc->get_statistics();
            TT_FATAL(
                false,
                "Out of Memory: Not enough space to allocate {} B {} buffer across {} banks, where each bank needs to "
                "store {} B, but bank size is {} B (allocated: {} B, free: {} B, largest free block: {} B)",
                size,
                enchantum::to_string(buffer_type_),
                num_banks,
                size_per_bank,
                bank_size(),
                mem_stats.total_allocated_bytes,
                mem_stats.total_free_bytes,
                mem_stats.largest_free_block_bytes);
        }
        allocated_buffers_[allocator_id.get()].insert(address.value());
        // No neighbors, nothing to invalidate
        return address.value();
    }

    // Get available address ranges after subtracting dependencies
    // The pair represents (start, end) of the available address range(s)
    std::vector<std::pair<DeviceAddr, DeviceAddr>> available_ranges =
        this->compute_available_addresses(allocator_id, size_per_bank, address_limit);

    // Choose an address from the allowed ranges respecting alignment and direction
    // Addresses should already be aligned to alignment_bytes_
    std::optional<DeviceAddr> chosen;
    if (bottom_up) {
        for (const auto& r : available_ranges) {
            DeviceAddr s = r.first;
            if (s + size_per_bank <= r.second) {
                chosen = s;
                break;
            }
        }
    } else {
        for (ssize_t i = static_cast<ssize_t>(available_ranges.size()) - 1; i >= 0; --i) {
            const auto& r = available_ranges[static_cast<size_t>(i)];
            DeviceAddr s = r.second - size_per_bank;
            if (s >= r.first) {
                chosen = s;
                break;
            }
        }
    }

    if (!chosen.has_value()) {
        auto mem_stats = alloc->get_statistics();
        TT_FATAL(
            false,
            "Out of Memory: Not enough space after considering dependencies to allocate {} B {} across {} banks ({} B "
            "per bank), bank size is {} B (allocated: {} B, free: {} B, largest free block: {} B)",
            size,
            enchantum::to_string(buffer_type_),
            num_banks,
            size_per_bank,
            bank_size(),
            mem_stats.total_allocated_bytes,
            mem_stats.total_free_bytes,
            mem_stats.largest_free_block_bytes);
    }
    TT_FATAL(
        chosen.value() % alignment_bytes_ == 0,
        "Chosen address {} is not aligned to {} B",
        chosen.value(),
        alignment_bytes_);

    auto address = alloc->allocate_at_address(chosen.value(), size_per_bank);
    TT_FATAL(address.has_value(), "Allocator failed to place at chosen address {}", chosen.value());
    allocated_buffers_[allocator_id.get()].insert(address.value());
    // Allocation in this allocator invalidates caches in allocators that depend on this allocator
    this->invalidate_allocated_ranges_cache_for_dependent_allocators(allocator_id);
    return address.value();
}

void BankManager::deallocate_buffer(DeviceAddr address, BankManager::AllocatorDependencies::AllocatorID allocator_id) {
    auto* alloc = this->get_allocator_from_id(allocator_id);
    TT_FATAL(alloc, "Allocator not initialized!");
    alloc->deallocate(address);
    allocated_buffers_[allocator_id.get()].erase(address);
    // Deallocation in this allocator invalidates caches in allocators that depend on this allocator
    this->invalidate_allocated_ranges_cache_for_dependent_allocators(allocator_id);
}

void BankManager::deallocate_all() {
    for (const auto allocator_id : allocator_dependencies_.allocator_ids()) {
        auto* alloc = this->get_allocator_from_id(allocator_id);
        TT_FATAL(alloc, "Allocator not initialized!");
        for (DeviceAddr addr : allocated_buffers_[allocator_id.get()]) {
            alloc->deallocate(addr);
        }
        allocated_buffers_[allocator_id.get()].clear();
        allocated_ranges_cache_[allocator_id.get()].reset();
    }
}

void BankManager::clear() {
    for (const auto allocator_id : allocator_dependencies_.allocator_ids()) {
        if (allocators_[allocator_id.get()]) {
            allocators_[allocator_id.get()]->clear();
            allocated_buffers_[allocator_id.get()].clear();
        }
        allocated_ranges_cache_[allocator_id.get()].reset();
    }
}

BankManager& BankManager::operator=(BankManager&& that) noexcept {
    buffer_type_ = that.buffer_type_;
    allocated_buffers_ = std::move(that.allocated_buffers_);
    bank_id_to_bank_offset_ = std::move(that.bank_id_to_bank_offset_);
    allocators_ = std::move(that.allocators_);
    interleaved_address_limit_ = that.interleaved_address_limit_;
    alignment_bytes_ = that.alignment_bytes_;
    allocator_dependencies_ = std::move(that.allocator_dependencies_);
    allocated_ranges_cache_ = std::move(that.allocated_ranges_cache_);
    return *this;
}

std::optional<DeviceAddr> BankManager::lowest_occupied_address(
    uint32_t bank_id, BankManager::AllocatorDependencies::AllocatorID allocator_id) const {
    const auto* alloc = this->get_allocator_from_id(allocator_id);
    if (alloc == nullptr) {
        return std::nullopt;
    }
    auto lowest_address = alloc->lowest_occupied_address();
    if (not lowest_address.has_value()) {
        return lowest_address;
    }
    DeviceAddr adjusted_abs_addr = lowest_address.value() + this->bank_offset(bank_id);
    return adjusted_abs_addr;
}

Statistics BankManager::get_statistics(BankManager::AllocatorDependencies::AllocatorID allocator_id) const {
    const auto* alloc = this->get_allocator_from_id(allocator_id);
    return alloc ? alloc->get_statistics() : Statistics();
}

void BankManager::dump_blocks(std::ostream& out, BankManager::AllocatorDependencies::AllocatorID allocator_id) const {
    const auto* alloc = this->get_allocator_from_id(allocator_id);
    if (alloc) {
        alloc->dump_blocks(out);
    }
}

MemoryBlockTable BankManager::get_memory_block_table(
    BankManager::AllocatorDependencies::AllocatorID allocator_id) const {
    const auto* alloc = this->get_allocator_from_id(allocator_id);
    if (alloc) {
        return alloc->get_memory_block_table();
    }

    log_warning(tt::LogAlways, "allocator is not initialized, cannot get block table for memory");
    return {};
}

void BankManager::shrink_size(
    DeviceAddr shrink_size, bool bottom_up, BankManager::AllocatorDependencies::AllocatorID allocator_id) {
    TT_FATAL(allocator_dependencies_.num_allocators() == 1, "Expected single allocator!");
    auto* alloc = this->get_allocator_from_id(allocator_id);
    if (alloc) {
        alloc->shrink_size(shrink_size, bottom_up);
    }
}

void BankManager::reset_size(BankManager::AllocatorDependencies::AllocatorID allocator_id) {
    TT_FATAL(allocator_dependencies_.num_allocators() == 1, "Expected single allocator!");
    auto* alloc = this->get_allocator_from_id(allocator_id);
    if (alloc) {
        alloc->reset_size();
    }
}

// ============================================================================
// Allocator State Methods
// ============================================================================

AllocatorState::BufferTypeState BankManager::extract_state(
    BankManager::AllocatorDependencies::AllocatorID allocator_id) const {
    AllocatorState::BufferTypeState state;

    // Copy metadata
    state.buffer_type = buffer_type_;
    state.num_banks = num_banks();
    state.bank_size = bank_size();
    state.interleaved_address_limit = interleaved_address_limit_;
    state.alignment_bytes = alignment_bytes_;
    state.bank_id_to_bank_offset = bank_id_to_bank_offset_;

    // Extract allocated regions from the allocator
    const auto* alloc = get_allocator_from_id(allocator_id);
    if (alloc) {
        auto allocated_addresses = alloc->allocated_addresses();
        state.allocated_regions.reserve(allocated_addresses.size());

        for (const auto& [start, end] : allocated_addresses) {
            if (end > start) {
                state.allocated_regions.emplace_back(start, end);
            }
        }
    }

    // Normalize to sort and coalesce
    state.normalize();

    return state;
}

AllocatorState::BufferTypeState BankManager::extract_merged_state() const {
    AllocatorState::BufferTypeState merged_state;
    bool first = true;

    // Merge states from all allocators
    for (const auto& allocator_id : allocator_dependencies_.allocator_ids()) {
        auto state = extract_state(allocator_id);

        if (first) {
            merged_state = std::move(state);
            first = false;
        } else {
            merged_state.merge(state);
        }
    }

    return merged_state;
}

void BankManager::apply_state(
    const AllocatorState::BufferTypeState& state, BankManager::AllocatorDependencies::AllocatorID target_allocator_id) {
    // Validate compatibility
    TT_FATAL(
        can_apply_state(state),
        "Cannot apply state: incompatible configuration (buffer_type: expected {}, got {} | num_banks: expected {}, "
        "got {} | bank_size: expected {}, got {})",
        enchantum::to_string(buffer_type_),
        enchantum::to_string(state.buffer_type),
        num_banks(),
        state.num_banks,
        bank_size(),
        state.bank_size);

    auto* alloc = get_allocator_from_id(target_allocator_id);
    TT_FATAL(alloc, "Allocator not initialized for ID {}", target_allocator_id.get());

    // Apply each allocated region
    for (const auto& [start_addr, end_addr] : state.allocated_regions) {
        DeviceAddr size = end_addr - start_addr;

        // Use allocate_at_address to mark this region as allocated
        auto result = alloc->allocate_at_address(start_addr, size);
        TT_FATAL(
            result.has_value(),
            "Failed to apply state: cannot allocate region [{}, {}) of size {} B at address {} in {} buffer type. "
            "Region may already be occupied or invalid.",
            start_addr,
            end_addr,
            size,
            start_addr,
            enchantum::to_string(buffer_type_));

        // Track the allocation
        allocated_buffers_[target_allocator_id.get()].insert(start_addr);
    }

    // Invalidate caches for dependent allocators
    invalidate_allocated_ranges_cache_for_dependent_allocators(target_allocator_id);
}

void BankManager::override_state(
    const AllocatorState::BufferTypeState& state, BankManager::AllocatorDependencies::AllocatorID target_allocator_id) {
    // Validate that state can be applied
    TT_FATAL(can_apply_state(state), "Cannot apply state: incompatible configuration");

    auto* alloc = get_allocator_from_id(target_allocator_id);
    TT_FATAL(alloc, "Allocator not initialized for ID {}", target_allocator_id.get());

    // Clear current allocations
    for (DeviceAddr addr : allocated_buffers_[target_allocator_id.get()]) {
        alloc->deallocate(addr);
    }
    allocated_buffers_[target_allocator_id.get()].clear();
    allocated_ranges_cache_[target_allocator_id.get()].reset();

    // Apply state
    apply_state(state, target_allocator_id);
}

bool BankManager::can_apply_state(const AllocatorState::BufferTypeState& state) const {
    return buffer_type_ == state.buffer_type &&                              //
           interleaved_address_limit_ == state.interleaved_address_limit &&  //
           num_banks() == state.num_banks &&                                 //
           bank_size() == state.bank_size &&                                 //
           alignment_bytes_ == state.alignment_bytes;
}

}  // namespace tt::tt_metal
