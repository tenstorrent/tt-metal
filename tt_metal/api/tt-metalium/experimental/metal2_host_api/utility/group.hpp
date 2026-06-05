// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>
#include <vector>

namespace tt::tt_metal::experimental {

// A Group<T> is an unordered collection of T: "here is a bunch of these", where the
// element order carries no meaning (unlike a std::vector, whose order is semantic).
//
// Group<T> is just a thin alias over std::vector.
// It has the full vector interface and interoperates freely.
// Use add() as a synonym for push_back for appending an element. 

// Where elements in a group must be unique, that is validated by the API that consumes 
// the Group, not enforced by the container.
template <typename T>
using Group = std::vector<T>;

template <typename T>
T& add(Group<T>& g, T value) {
    g.push_back(std::move(value));
    return g.back();
}

}  // namespace tt::tt_metal::experimental
