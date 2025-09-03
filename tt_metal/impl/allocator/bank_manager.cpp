// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bank_manager.hpp"

#include <enchantum/enchantum.hpp>
#include <util.hpp>
#include <limits>
#include <string_view>
#include <utility>
#include <algorithm>

#include "allocator/algorithms/allocator_algorithm.hpp"
#include "allocator_types.hpp"
#include "assert.hpp"
#include "buffer_types.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-logger/tt-logger.hpp>
#include "tt_metal/impl/allocator/algorithms/free_list_opt.hpp"

namespace tt {

namespace tt_metal {

// --- StateDependencies impl ---
BankManager::StateDependencies::StateDependencies() {
    // Default: single state (0) with no dependencies
    dependencies = AdjacencyList{{}};
}

BankManager::StateDependencies::StateDependencies(
    const std::unordered_map<StateId, ttsl::SmallVector<StateId>>& dependencies_map) {
    // Determine total number of states as max state id seen anywhere (keys or values)
    if (dependencies_map.empty()) {
        dependencies = AdjacencyList{{}};
        return;
    }

    uint32_t max_id = 0;
    for (const auto& kv : dependencies_map) {
        max_id = std::max(max_id, kv.first.value);
        for (const auto dep_state : kv.second) {
            max_id = std::max(max_id, dep_state.value);
        }
    }
    const uint32_t num_states = static_cast<uint32_t>(max_id) + 1;  // +1 because state ids are zero-indexed

    // Use set form to ensure uniqueness while building undirected adjacency list
    ttsl::SmallVector<std::unordered_set<uint32_t>> undirected_sets(num_states);

    // Build undirected adjacency: for each edge (u -> v), add v to u and u to v
    for (const auto& kv : dependencies_map) {
        // Ensure uniqueness of dependency values per state input (catch duplicates in provided list)
        std::unordered_set<uint32_t> seen_values;
        for (const auto dep_state : kv.second) {
            const bool inserted = seen_values.insert(dep_state.value).second;
            TT_FATAL(
                inserted,
                "Duplicate dependency for state {}: {} appears more than once!",
                kv.first.value,
                dep_state.value);
        }

        for (const auto dep_state : kv.second) {
            const uint32_t u = kv.first.value;
            const uint32_t v = dep_state.value;
            undirected_sets[u].insert(v);
            undirected_sets[v].insert(u);
        }
    }

    // Build dependencies list: for each state s, which states s depends on / depends on s
    // Note: Dependencies are not sorted because dependencies_map is unordered
    dependencies.clear();
    dependencies.resize(num_states);
    for (uint32_t i = 0; i < num_states; ++i) {
        auto& dst = dependencies[i];
        dst.clear();
        dst.reserve(undirected_sets[i].size());
        for (uint32_t neighbor : undirected_sets[i]) {
            dst.push_back(StateId{neighbor});
        }
    }
}

uint32_t BankManager::StateDependencies::num_states() const { return static_cast<uint32_t>(dependencies.size()); }

ttsl::SmallVector<BankManager::StateDependencies::StateId> BankManager::StateDependencies::states() const {
    ttsl::SmallVector<StateId> states;
    states.reserve(num_states());
    for (size_t i = 0; i < num_states(); i++) {
        states.push_back(StateId{static_cast<uint32_t>(i)});
    }
    return states;
}

bool BankManager::StateDependencies::operator==(const BankManager::StateDependencies& other) const noexcept {
    if (dependencies.size() != other.dependencies.size()) {
        return false;
    }
    // Compare as sorted adjacency lists per row, without mutating originals
    for (size_t i = 0; i < dependencies.size(); ++i) {
        auto a = dependencies[i];
        auto b = other.dependencies[i];
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        if (a != b) {
            return false;
        }
    }
    return true;
}

void BankManager::init_allocators_across_states(DeviceAddr size_bytes, uint32_t alignment_bytes, DeviceAddr offset) {
    const uint32_t n = state_dependencies_.num_states();
    allocators_.resize(n);
    allocated_buffers_.resize(n);

    // Reverse edges now built inside StateDependencies
    for (uint32_t state = 0; state < n; ++state) {
        allocators_[state] = std::make_unique<allocator::FreeListOpt>(
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
        7, 12, 20, 48, 56, 63, 70, 72, 80, 94, 110, 120, 124, 130, 140};
    bool custom_mod_bank_id_calculation_exists = acceptable_num_non_pow2_mem_banks.count(num_banks) > 0;
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
    const StateDependencies& dependencies) :
    buffer_type_(buffer_type), alignment_bytes_(alignment_bytes), state_dependencies_(dependencies) {
    unsigned int bank_id = 0;
    for (const auto bank_offset : bank_offsets) {
        bank_id_to_bank_offset_.insert({bank_id, bank_offset});
        bank_id++;
    }
    interleaved_address_limit_ = 0;
    validate_num_banks(bank_id_to_bank_offset_.size(), buffer_type_, disable_interleaved);

    // Initialize allocators across states; sets up state-dependent members
    this->init_allocators_across_states(
        size_bytes, MetalContext::instance().hal().get_alignment(HalMemType::DRAM), alloc_offset);
}

BankManager::BankManager(
    const BufferType& buffer_type,
    const std::unordered_map<uint32_t, int64_t>& bank_id_to_bank_offset,
    DeviceAddr size_bytes,
    DeviceAddr interleaved_address_limit,
    uint32_t alignment_bytes,
    DeviceAddr alloc_offset,
    bool disable_interleaved,
    const StateDependencies& dependencies) :
    buffer_type_(buffer_type),
    bank_id_to_bank_offset_(bank_id_to_bank_offset),
    interleaved_address_limit_(interleaved_address_limit),
    alignment_bytes_(alignment_bytes),
    state_dependencies_(dependencies) {
    validate_num_banks(bank_id_to_bank_offset_.size(), buffer_type_, disable_interleaved);

    // Initialize allocators across states; sets up state-dependent members
    this->init_allocators_across_states(
        size_bytes, MetalContext::instance().hal().get_alignment(HalMemType::DRAM), alloc_offset);
}

uint32_t BankManager::num_banks() const { return bank_id_to_bank_offset_.size(); }

DeviceAddr BankManager::bank_size() const {
    TT_ASSERT(bool(allocators_[0]), "Allocator not initialized!");
    DeviceAddr max_size_bytes_u64 = allocators_[0]->max_size_bytes();
    if (max_size_bytes_u64 > std::numeric_limits<DeviceAddr>::max()) {
        TT_THROW("Bank size {} overflows DeviceAddr", max_size_bytes_u64);
    }
    DeviceAddr max_size_bytes = (DeviceAddr)max_size_bytes_u64;
    return max_size_bytes;
}

int64_t BankManager::bank_offset(uint32_t bank_id) const {
    this->validate_bank_id(bank_id);
    return bank_id_to_bank_offset_.at(bank_id);
}

void BankManager::validate_bank_id(uint32_t bank_id) const {
    TT_FATAL(
        bank_id_to_bank_offset_.find(bank_id) != bank_id_to_bank_offset_.end(),
        "Expected bank {} to be tracked!",
        bank_id,
        bank_id_to_bank_offset_.size());
}

void BankManager::assert_valid_state(BankManager::StateDependencies::StateId state) const {
    TT_FATAL(
        state.value < state_dependencies_.num_states(),
        "Invalid state {} (num_states={})",
        state.value,
        state_dependencies_.num_states());
}

uint64_t BankManager::allocate_buffer(
    DeviceAddr size,
    DeviceAddr page_size,
    bool bottom_up,
    const CoreRangeSet& compute_grid,
    std::optional<uint32_t> num_shards,
    BankManager::StateDependencies::StateId state) {
    this->assert_valid_state(state);
    TT_ASSERT(bool(allocators_[state.value]), "Allocator not initialized!");
    auto* alloc = allocators_[state.value].get();

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
    DeviceAddr size_per_bank = tt::tt_metal::detail::SizeBytesPerBank(size, page_size, num_banks, alignment_bytes_);
    DeviceAddr address_limit = 0;
    if (!is_sharded and buffer_type_ == BufferType::L1) {
        address_limit = interleaved_address_limit_;
        TT_FATAL(address_limit > 0, "Address limit {} needs to be larger than zero.", address_limit);
    }

    // If there are no dependent states, fall back to allocator's native strategy
    const auto& neighbors = state_dependencies_.dependencies[state.value];
    if (neighbors.empty()) {
        auto address = alloc->allocate(size_per_bank, bottom_up, address_limit);
        if (not address.has_value()) {
            TT_THROW(
                "Out of Memory: Not enough space to allocate {} B {} buffer across {} banks, where each bank needs to "
                "store {} B",
                size,
                enchantum::to_string(buffer_type_),
                num_banks,
                size_per_bank);
        }
        allocated_buffers_[state.value].insert(address.value());
        return address.value();
    }

    // Build free ranges for current state and all dependent states and find common free address
    auto clamp_ranges = [address_limit](std::vector<std::pair<DeviceAddr, DeviceAddr>> ranges) {
        if (address_limit == 0) {
            return ranges;
        }
        std::vector<std::pair<DeviceAddr, DeviceAddr>> out;
        out.reserve(ranges.size());
        for (auto r : ranges) {
            if (r.first < address_limit) {
                r.first = address_limit;
            }
            if (r.second > r.first) {
                out.emplace_back(r.first, r.second);
            }
        }
        return out;
    };

    std::vector<std::vector<std::pair<DeviceAddr, DeviceAddr>>> all_free;
    all_free.reserve(1 + neighbors.size());

    // Current state ranges
    all_free.push_back(clamp_ranges(alloc->available_addresses(size_per_bank)));

    // Neighbor state ranges
    for (const auto dep_state : neighbors) {
        auto* dep_alloc = allocators_[dep_state.value].get();
        all_free.push_back(clamp_ranges(dep_alloc->available_addresses(size_per_bank)));
    }

    // Sort each free range list by start address to prepare for intersection
    for (auto& ranges : all_free) {
        std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    }

    // Helper to intersect two sorted, disjoint range lists
    auto intersect_two = [](const std::vector<std::pair<DeviceAddr, DeviceAddr>>& a,
                            const std::vector<std::pair<DeviceAddr, DeviceAddr>>& b) {
        std::vector<std::pair<DeviceAddr, DeviceAddr>> res;
        size_t i = 0, j = 0;
        while (i < a.size() && j < b.size()) {
            DeviceAddr s = std::max(a[i].first, b[j].first);
            DeviceAddr e = std::min(a[i].second, b[j].second);
            if (s < e) {
                res.emplace_back(s, e);
            }
            if (a[i].second < b[j].second) {
                i++;
            } else {
                j++;
            }
        }
        return res;
    };

    // Compute intersection across all lists
    std::vector<std::pair<DeviceAddr, DeviceAddr>> allowed = all_free.front();
    for (size_t k = 1; k < all_free.size(); ++k) {
        allowed = intersect_two(allowed, all_free[k]);
        if (allowed.empty()) {
            break;
        }
    }

    // Choose an address from the allowed ranges respecting alignment and direction
    std::optional<DeviceAddr> chosen;
    if (bottom_up) {
        for (const auto& r : allowed) {
            // DeviceAddr s = alloc->align(r.first);
            DeviceAddr s = r.first;
            if (s + size_per_bank <= r.second) {
                chosen = s;
                break;
            }
        }
    } else {
        for (ssize_t i = static_cast<ssize_t>(allowed.size()) - 1; i >= 0; --i) {
            const auto& r = allowed[static_cast<size_t>(i)];
            DeviceAddr s = r.second - size_per_bank;
            // s = (s / alignment_bytes_) * alignment_bytes_;  // round down to alignment
            if (s >= r.first) {
                chosen = s;
                break;
            }
        }
    }

    if (!chosen.has_value()) {
        TT_THROW(
            "Out of Memory: Not enough space after considering dependencies to allocate {} B {} across {} banks ({} B "
            "per bank)",
            size,
            enchantum::to_string(buffer_type_),
            num_banks,
            size_per_bank);
    } else if (chosen.value() % alignment_bytes_ != 0) {
        TT_THROW("Chosen address {} is not aligned to {} B", chosen.value(), alignment_bytes_);
    }

    auto address = alloc->allocate_at_address(chosen.value(), size_per_bank);
    if (!address.has_value()) {
        TT_THROW("Allocator failed to place at chosen address {}", chosen.value());
    }
    allocated_buffers_[state.value].insert(address.value());
    return address.value();
}

void BankManager::deallocate_buffer(DeviceAddr address, BankManager::StateDependencies::StateId state) {
    this->assert_valid_state(state);
    allocators_[state.value]->deallocate(address);
    allocated_buffers_[state.value].erase(address);
}

void BankManager::deallocate_all(BankManager::StateDependencies::StateId state) {
    this->assert_valid_state(state);
    for (DeviceAddr addr : allocated_buffers_[state.value]) {
        this->deallocate_buffer(addr, state);
    }
}

void BankManager::clear(BankManager::StateDependencies::StateId state) {
    this->assert_valid_state(state);
    if (allocators_[state.value]) {
        allocators_[state.value]->clear();
        allocated_buffers_[state.value].clear();
    }
}

BankManager::~BankManager() {
    for (const auto state : state_dependencies_.states()) {
        this->deallocate_all(state);
    }
    allocators_.clear();
    bank_id_to_bank_offset_.clear();
}

BankManager&& BankManager::operator=(BankManager&& that) noexcept {
    state_dependencies_ = std::move(that.state_dependencies_);
    buffer_type_ = that.buffer_type_;
    allocated_buffers_ = std::move(that.allocated_buffers_);
    bank_id_to_bank_offset_ = std::move(that.bank_id_to_bank_offset_);
    allocators_ = std::move(that.allocators_);
    interleaved_address_limit_ = that.interleaved_address_limit_;
    alignment_bytes_ = that.alignment_bytes_;
    return std::move(*this);
}

std::optional<DeviceAddr> BankManager::lowest_occupied_address(
    uint32_t bank_id, BankManager::StateDependencies::StateId state) const {
    this->assert_valid_state(state);
    if (not allocators_[state.value]) {
        return std::nullopt;
    }
    auto lowest_address = allocators_[state.value]->lowest_occupied_address();
    if (not lowest_address.has_value()) {
        return lowest_address;
    }
    DeviceAddr adjusted_abs_addr = lowest_address.value() + this->bank_offset(bank_id);
    return adjusted_abs_addr;
}

Statistics BankManager::get_statistics(BankManager::StateDependencies::StateId state) const {
    this->assert_valid_state(state);
    return allocators_[state.value] ? allocators_[state.value]->get_statistics() : Statistics();
}

void BankManager::dump_blocks(std::ofstream& out, BankManager::StateDependencies::StateId state) const {
    this->assert_valid_state(state);
    if (allocators_[state.value]) {
        allocators_[state.value]->dump_blocks(out);
    }
}

MemoryBlockTable BankManager::get_memory_block_table(BankManager::StateDependencies::StateId state) const {
    this->assert_valid_state(state);
    if (allocators_[state.value]) {
        return allocators_[state.value]->get_memory_block_table();
    }

    log_warning(tt::LogAlways, "allocator is not initialized, cannot get block table for memory");
    return {};
}

void BankManager::shrink_size(DeviceAddr shrink_size, bool bottom_up, BankManager::StateDependencies::StateId state) {
    this->assert_valid_state(state);
    if (allocators_[state.value]) {
        allocators_[state.value]->shrink_size(shrink_size, bottom_up);
    }
}

void BankManager::reset_size(BankManager::StateDependencies::StateId state) {
    this->assert_valid_state(state);
    if (allocators_[state.value]) {
        allocators_[state.value]->reset_size();
    }
}

}  // namespace tt_metal

}  // namespace tt
