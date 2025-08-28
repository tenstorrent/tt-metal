// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <allocator_types.hpp>
#include <buffer_types.hpp>
#include <stdint.h>
#include <fstream>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <map>
#include <tt_stl/small_vector.hpp>

#include "algorithms/allocator_algorithm.hpp"
#include "core_coord.hpp"
#include "hal_types.hpp"

namespace tt {

namespace tt_metal {
enum class BufferType;
namespace allocator {
class Algorithm;
}  // namespace allocator

class BankManager {
public:
    BankManager() = default;

    struct StateDependencies {
        struct StateId {
            uint32_t value{0};
            bool operator==(const StateId& other) const noexcept { return value == other.value; }
            bool operator!=(const StateId& other) const noexcept { return value != other.value; }
        };

        struct Hasher {
            size_t operator()(const StateId& s) const noexcept { return std::hash<uint32_t>{}(s.value); }
        };

        std::unordered_map<StateId, tt::stl::SmallVector<StateId>, Hasher> adjacency{};

        StateDependencies();
        explicit StateDependencies(const std::unordered_map<StateId, tt::stl::SmallVector<StateId>, Hasher>& deps);

        uint32_t num_states() const;
    };

    BankManager(
        const BufferType& buffer_type,
        const std::vector<int64_t>& bank_descriptors,
        DeviceAddr size_bytes,
        uint32_t alignment_bytes,
        DeviceAddr alloc_offset = 0,
        bool disable_interleaved = false,
        const StateDependencies& dependencies = StateDependencies());
    // Removed num_states-only ctor; use StateDependencies instead
    BankManager(
        const BufferType& buffer_type,
        const std::unordered_map<uint32_t, int64_t>& bank_id_to_descriptor,
        DeviceAddr size_bytes,
        DeviceAddr interleaved_address_limit,
        uint32_t alignment_bytes,
        DeviceAddr alloc_offset = 0,
        bool disable_interleaved = false,
        const StateDependencies& dependencies = StateDependencies());
    // Removed num_states-only ctor; use StateDependencies instead
    BankManager&& operator=(BankManager&& that) noexcept;
    ~BankManager();
    uint32_t num_banks(uint32_t state = 0) const;

    DeviceAddr bank_size(uint32_t state = 0) const;

    int64_t bank_offset(uint32_t bank_id, uint32_t state = 0) const;

    DeviceAddr allocate_buffer(
        DeviceAddr size,
        DeviceAddr page_size,
        bool bottom_up,
        const CoreRangeSet& compute_grid,
        std::optional<uint32_t> num_shards,
        uint32_t state = 0);

    void deallocate_buffer(DeviceAddr address, uint32_t state = 0);
    void deallocate_all(uint32_t state = 0);

    void clear(uint32_t state = 0);

    std::optional<DeviceAddr> lowest_occupied_address(uint32_t bank_id, uint32_t state = 0) const;

    Statistics get_statistics(uint32_t state = 0) const;

    void dump_blocks(std::ofstream& out, uint32_t state = 0) const;

    MemoryBlockTable get_memory_block_table(uint32_t state = 0) const;

    void shrink_size(DeviceAddr shrink_size, bool bottom_up = true, uint32_t state = 0);
    void reset_size(uint32_t state = 0);

private:
    void deallocate_buffer_(DeviceAddr address, uint32_t state);

    // Dependencies between states (also encodes number of states)
    StateDependencies dependencies_{};
    // Reverse edges: for each state, which states depend on it
    std::unordered_map<uint32_t, tt::stl::SmallVector<uint32_t>> dependents_{};

    // Type of buffers allocated in the banks (same across states)
    BufferType buffer_type_{};
    // Track allocations per state: base address -> size_per_bank
    std::vector<std::unordered_map<DeviceAddr, DeviceAddr>> allocated_buffers_{};
    // This is to store offsets for any banks that share a core or node (dram in wh/storage core), so we can view all
    // banks using only bank_id. Set to 0 for cores/nodes with only 1 bank.
    std::unordered_map<uint32_t, int64_t> bank_id_to_bank_offset_{};
    std::vector<std::unique_ptr<allocator::Algorithm>> allocators_{};
    DeviceAddr interleaved_address_limit_{};
    uint32_t alignment_bytes_{};
    // Remember allocator offsets per state
    std::vector<DeviceAddr> allocator_offsets_{};

    // Foreign occupancy overlay per state, represented as difference map
    struct Overlay {
        std::map<DeviceAddr, int64_t> delta;
        void add(DeviceAddr start, DeviceAddr end) {
            if (start >= end) {
                return;
            }
            delta[start] += 1;
            delta[end] -= 1;
            if (delta[start] == 0) {
                delta.erase(start);
            }
            if (delta[end] == 0) {
                delta.erase(end);
            }
        }
        void remove(DeviceAddr start, DeviceAddr end) {
            if (start >= end) {
                return;
            }
            delta[start] -= 1;
            delta[end] += 1;
            if (delta[start] == 0) {
                delta.erase(start);
            }
            if (delta[end] == 0) {
                delta.erase(end);
            }
        }
        std::vector<std::pair<DeviceAddr, DeviceAddr>> occupied() const {
            std::vector<std::pair<DeviceAddr, DeviceAddr>> res;
            int64_t acc = 0;
            bool in_seg = false;
            DeviceAddr seg_start = 0;
            for (const auto& kv : delta) {
                DeviceAddr x = kv.first;
                int64_t next = acc + kv.second;
                if (acc <= 0 && next > 0) {
                    seg_start = x;
                    in_seg = true;
                } else if (acc > 0 && next <= 0) {
                    if (in_seg && seg_start < x) {
                        res.emplace_back(seg_start, x);
                    }
                    in_seg = false;
                }
                acc = next;
            }
            return res;
        }
        std::vector<std::pair<DeviceAddr, DeviceAddr>> subtract(
            const std::vector<std::pair<DeviceAddr, DeviceAddr>>& free_ranges) const {
            auto occ = occupied();
            std::vector<std::pair<DeviceAddr, DeviceAddr>> out;
            size_t j = 0;
            for (const auto& fr : free_ranges) {
                DeviceAddr fs = fr.first, fe = fr.second;
                if (fs >= fe) {
                    continue;
                }
                while (j < occ.size() && occ[j].second <= fs) {
                    j++;
                }
                DeviceAddr cur = fs;
                size_t jj = j;
                while (jj < occ.size() && occ[jj].first < fe) {
                    DeviceAddr os = occ[jj].first, oe = occ[jj].second;
                    if (os > cur) {
                        out.emplace_back(cur, std::min(fe, os));
                    }
                    if (oe >= fe) {
                        cur = fe;
                        break;
                    }
                    cur = std::max(cur, oe);
                    jj++;
                }
                if (cur < fe) {
                    out.emplace_back(cur, fe);
                }
            }
            return out;
        }
    };
    std::vector<Overlay> overlays_{};

    void validate_bank_id(uint32_t bank_id, uint32_t state) const;
    void assert_valid_state(uint32_t state) const;

    void init_allocator(DeviceAddr size_bytes, uint32_t alignment_bytes, DeviceAddr offset, uint32_t state);
    static DeviceAddr align_up(DeviceAddr addr, DeviceAddr alignment) {
        if (alignment == 0) {
            return addr;
        }
        DeviceAddr factor = (addr + alignment - 1) / alignment;
        return factor * alignment;
    }
};

}  // namespace tt_metal

}  // namespace tt
