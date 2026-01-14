// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include <tt-metalium/host_buffer.hpp>

// TODO: this will be moved
#include <tt-metalium/shape.hpp>

namespace tt::tt_metal /*::tensor*/ {

struct Unsure {};

/**
 * HostTensor is a host data class. It has the semantics of a container, and all host <-> device communications are
 * explicit.
 *
 */
class HostTensor {
    /**
     * To avoid disruption to existing users, HostTensor will deviate very little from the existing (host) Tensor
     * semantics. The only significant changes are:
     * - Eliminating implicit data movement APIs.
     * - Remove transformation methods like to_layout and pad from the class methods. These seem better as free
     *   functions that operate on a HostTensor than as methods of HostTensor. (Separation of data storage and data
     *   manipulation.) In the existing Tensor, these are already duplicated as both methods and free functions.
     */

public:
    // Special Member functions

    /**
     * Constructs an empty host tensor.
     */
    HostTensor() = default;
    ~HostTensor() = default;

    HostTensor(const HostTensor&) = default;
    HostTensor& operator=(const HostTensor&) = default;

    HostTensor(HostTensor&&) = default;
    HostTensor& operator=(HostTensor&&) = default;

    // End special member functions

    // constructions:

    // TODO: Spec? maybe better if this is `from_host_buffer`?
    explicit HostTensor(const HostBuffer&);

    static HostBuffer from_borrowed_data(/* */);
    static HostBuffer from_span(/* */);
    static HostBuffer from_vector(/* */);

    // getters
    std::vector<Unsure /* What would this be?? */> to_vector() const;
    // TODO: This method has been removed from original Tensor, see: #32600
    Unsure item() const;

    // TODO: we should just specialize std::to_string
    std::string write_to_string() const;

    // TODO: parity with DistributedHostBuffer
    const HostBuffer& get_host_buffer() const;

    // TODO(River): understand what is sharding better
    bool is_sharded() const;
    // TODO: what is the return type here?
    Shape element_size() const;

    // TODO: dump other getters.

    // reshape transformation, mutating version
    // TODO: figure out what we will be doing for reshape
    void reshape(/* */);
};

}  // namespace tt::tt_metal
