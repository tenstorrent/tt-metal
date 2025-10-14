// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>

namespace tt {

enum CBIndex : std::uint8_t;
enum class DataFormat : std::uint8_t;

namespace tt_metal {

using CBHandle = std::uintptr_t;

class CircularBuffer;
class CircularBufferIndexMetadata;

/**
 * @brief (**Read Only**) Metadata accessor for CircularBuffer
 *
 * Provides a lightweight interface to query circular buffer properties without
 * exposing the full CircularBuffer implementation.
 *
 * The lifetime of this class is that of the underlying Circular Buffer,
 * which is managed by the Program.
 */
class CircularBufferMetadata {
public:
    // Internal constructor
    explicit CircularBufferMetadata(CircularBuffer* _buffer) : buffer(_buffer) {}

    /**
     * @brief Get the identifier of the circular buffer
     */
    CBHandle id() const;

    /**
     * @brief Get the set of core ranges this circular buffer is configured on
     */
    const CoreRangeSet& core_ranges() const;

    /**
     * @brief Get the total size of the circular buffer in bytes
     */
    std::size_t size() const;

    /**
     * @brief Check if the buffer address is globally allocated
     */
    bool globally_allocated() const;

    /**
     * @brief Get metadata for all configured buffer indices
     */
    std::vector<CircularBufferIndexMetadata> indicies() const;

private:
    CircularBuffer* buffer;
};

/**
 * @brief (**Read Only**) Metadata for a specific circular buffer index
 *
 * Contains configuration details for a single buffer index within a circular buffer.
 *
 * The lifetime of this class is that of the underlying Circular Buffer,
 * which is managed by the Program.
 */
class CircularBufferIndexMetadata {
public:
    // Internal constructor
    CircularBufferIndexMetadata(CircularBuffer* buffer, CBIndex id);

    /**
     * @brief Get the buffer index current class represents
     */
    CBIndex index() const;

    /**
     * @brief Get the page size for this buffer index
     */
    std::size_t page_size() const;

    /**
     * @brief Get the number of pages for this buffer index
     */
    std::size_t num_pages() const;

    /**
     * @brief Get the data format for this buffer index
     */
    tt::DataFormat data_format() const;

private:
    CircularBuffer* buffer;
    CBIndex idx;
};

}  // namespace tt_metal
}  // namespace tt
