// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/maybe_remote.hpp>

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal::distributed {

/**
 * Concept to check if a type behaves like MaybeRemote.
 *
 * A type satisfies MaybeRemoteLike if it has:
 * - A nested value_type type alias
 * - An is_local() method that returns something convertible to bool
 * - A value() method that returns exactly the value_type
 */
template <typename T>
concept MaybeRemoteLike = requires(T t) {
    typename T::value_type;
    { t.is_local() } -> std::convertible_to<bool>;
    { t.value() } -> std::same_as<typename T::value_type&>;
};

using MaybeRemoteDeviceId = MaybeRemote<int>;
using MaybeRemoteDevice = MaybeRemote<IDevice*>;

/**
 * Extract all local values from a container of MaybeRemote objects.
 *
 * This function filters out remote devices and returns only the local values
 * in a new vector.
 *
 * @tparam Container Any container type whose value_type satisfies MaybeRemoteLike
 * @param container The container of MaybeRemote objects
 * @return std::vector<T> containing only the local values
 */
template <typename Container>
    requires MaybeRemoteLike<typename Container::value_type>
[[nodiscard]] auto extract_locals(const Container& container) {
    using MaybeRemoteType = typename Container::value_type;
    using ValueType = typename MaybeRemoteType::value_type;

    std::vector<ValueType> locals;
    locals.reserve(container.size());

    for (const auto& maybe : container) {
        if (maybe.is_local()) {
            locals.push_back(maybe.value());
        }
    }
    return locals;
}

/**
 * Wraps all local values from a container to a container with MaybeRemote objects.
 *
 * @tparam Container Any container type
 * @param container The container of objects of type T
 * @return std::vector<MaybeRemote<T>> containing the local values wrapped in MaybeRemote
 */
template <typename Container>
[[nodiscard]] auto wrap_to_maybe_remote(const Container& container) {
    using T = typename Container::value_type;
    std::vector<MaybeRemote<T>> wrapped;
    wrapped.reserve(container.size());
    for (const auto& local : container) {
        wrapped.push_back(MaybeRemote<T>::local(local));
    }
    return wrapped;
}

}  // namespace tt::tt_metal::distributed
