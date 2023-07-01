#include "tt_metal/impl/buffers/circular_buffer.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"

#include "llrt/llrt.hpp"

namespace tt {

namespace tt_metal {

void validate_buffer_indices(const std::set<u32> &buffer_indices) {
    log_assert(buffer_indices.size() <= NUM_CIRCULAR_BUFFERS, "Number of circular buffers requested ({}) exceeds max number of circular buffers allowed on a core ({})", buffer_indices.size(), NUM_CIRCULAR_BUFFERS);
    // check that they're between 0 to NUM_CIRCULAR_BUFFERS - 1
    for (u32 buffer_index : buffer_indices) {
        log_assert(buffer_index < NUM_CIRCULAR_BUFFERS, "Buffer index can only be up to {}", NUM_CIRCULAR_BUFFERS - 1);
    }
}

CircularBuffer::CircularBuffer(
    const CoreRangeSet &core_range_set,
    const std::set<u32> &buffer_indices,
    u32 num_tiles,
    u32 size_in_bytes,
    u32 address,
    DataFormat data_format) :
    core_range_set_(core_range_set), buffer_indices_(buffer_indices), num_tiles_(num_tiles), size_(size_in_bytes), address_(address), data_format_(data_format) {
    validate_buffer_indices(buffer_indices);
}

bool CircularBuffer::is_on_logical_core(const CoreCoord &logical_core) const {
    return this->core_range_set_.core_coord_in_core_ranges(logical_core);
}

}  // namespace tt_metal

}  // namespace tt
