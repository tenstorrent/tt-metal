#include "tt_metal/impl/allocator/allocator.hpp"

namespace tt {

namespace tt_metal {

namespace allocator {

uint32_t find_max_address(const std::vector<std::pair<uint32_t, uint32_t>> &candidate_addr_ranges) {
    uint32_t max_address = candidate_addr_ranges[0].second;
    for (auto candidate_addr_range : candidate_addr_ranges) {
        max_address = std::max(max_address, candidate_addr_range.second);
    }
    return max_address;
}

uint32_t find_address_of_smallest_chunk(const std::vector<std::pair<uint32_t, uint32_t>> &candidate_addr_ranges) {
    uint32_t smallest_chunk = candidate_addr_ranges[0].second - candidate_addr_ranges[0].first;
    uint32_t address = candidate_addr_ranges[0].first;
    for (auto candidate_addr_range : candidate_addr_ranges) {
        uint32_t range_size = candidate_addr_range.second - candidate_addr_range.first;
        if (range_size < smallest_chunk) {
            smallest_chunk = range_size;
            address = candidate_addr_range.first;
        }
    }
    return address;
}

void populate_candidate_address_ranges(
    std::vector<std::pair<uint32_t, uint32_t>> &candidate_addr_ranges,
    const std::vector<std::pair<uint32_t, uint32_t>> &potential_addr_ranges,
    std::function<bool(const std::pair<uint32_t, uint32_t> &)> filter) {
    if (candidate_addr_ranges.empty()) {
        candidate_addr_ranges = potential_addr_ranges;
        return;
    }
    int i = 0;
    int j = 0;
    std::vector<std::pair<uint32_t, uint32_t>> intersecting_addr_ranges;
    while (i < candidate_addr_ranges.size() and j < potential_addr_ranges.size()) {
        uint32_t lower_addr = std::max(candidate_addr_ranges[i].first, potential_addr_ranges[j].first);
        uint32_t upper_addr = std::min(candidate_addr_ranges[i].second, potential_addr_ranges[j].second);
        if (lower_addr <= upper_addr) {
            std::pair<uint32_t, uint32_t> address_range = {lower_addr, upper_addr};
            if (filter(address_range)) {
                intersecting_addr_ranges.push_back(address_range);
            }
        }
        if (candidate_addr_ranges[i].second < potential_addr_ranges[j].second) {
            i++;
        } else {
            j++;
        }
    }
    candidate_addr_ranges = intersecting_addr_ranges;
}

}  // namespace allocator

}  // namespace tt_metal

}  // namespace tt
