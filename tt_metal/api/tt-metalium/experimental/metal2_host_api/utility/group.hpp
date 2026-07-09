// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

namespace tt::tt_metal::experimental {

// A Group<T> is an unordered collection of T: "here is a bunch of these", where the
// element order carries no meaning (unlike a std::vector, whose order is semantic).
//
// For now, Group<T> is just a thin alias over std::vector.
// It has the full vector interface and interoperates freely.
// (Plan: upgrade this to a class and augment with a Group.add(element) method
// synonymous with push_back).
//
// Where elements in a group must be unique, that is validated by the API that
// consumes the Group, not enforced by the container.
template <typename T>
using Group = std::vector<T>;

}  // namespace tt::tt_metal::experimental
