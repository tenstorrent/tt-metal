// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <type_traits>
#include <vector>
#include <string>
#include <optional>
#include <string_view>
#include <sstream>
#include <concepts>
#include <tt-metalium/assert.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

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

/**
 * Empty marker type to represent a remote device or coordinate.
 *
 * RemoteDevice is used in MaybeRemote to indicate that a device or coordinate
 * exists on a remote host and is not directly accessible from the current process.
 */
struct RemoteDevice {
    // For RemoteDevice to work properly in STL containers and with std::variant, it needs to have comparison operators defined.
    friend bool operator==(const RemoteDevice&, const RemoteDevice&) = default;
};

/**
 * Type-safe wrapper to distinguish between remote and local values.
 *
 * MaybeRemote<T> provides compile-time type safety when dealing with values
 * that may exist on remote hosts in a distributed system.
 *
 */
template <typename T>
class MaybeRemote {
public:
    using value_type = T;

private:
    std::variant<RemoteDevice, T> value_;

    // Private constructor for RemoteDevice
    explicit MaybeRemote(RemoteDevice);
    explicit MaybeRemote(T value);

    // Helper for throwing remote access errors
    [[noreturn]] static void throw_remote_access_error();

public:
    // No default constructor - must be explicit about remote/local
    MaybeRemote() = delete;

    // Named constructors for clarity
    [[nodiscard]] static MaybeRemote local(T value);
    [[nodiscard]] static MaybeRemote remote();

    // Query methods
    [[nodiscard]] bool is_local() const noexcept;
    [[nodiscard]] bool is_remote() const noexcept;

    // Access methods: throw if remote, return local value if local.
    [[nodiscard]] T& value() &;
    [[nodiscard]] const T& value() const&;

    // Pointer-like operators for convenience
    [[nodiscard]] T& operator*() &;
    [[nodiscard]] const T& operator*() const&;
    [[nodiscard]] T* operator->();
    [[nodiscard]] const T* operator->() const;

    // Simple pattern matching for local/remote cases
    template <typename LocalFunc, typename RemoteFunc>
    [[nodiscard]] auto when(LocalFunc&& on_local, RemoteFunc&& on_remote) const -> decltype(auto);

    // For void return types
    template <typename LocalFunc>
    auto if_local(LocalFunc&& on_local) const
        -> std::enable_if_t<std::is_void_v<std::invoke_result_t<LocalFunc, const T&>>>;

    // For non-void return types
    template <typename LocalFunc>
    [[nodiscard]] auto if_local(LocalFunc&& on_local) const -> std::enable_if_t<
        !std::is_void_v<std::invoke_result_t<LocalFunc, const T&>>,
        std::optional<std::invoke_result_t<LocalFunc, const T&>>>;

    // For void return types
    template <typename RemoteFunc>
    auto if_remote(RemoteFunc&& on_remote) const -> std::enable_if_t<std::is_void_v<std::invoke_result_t<RemoteFunc>>>;

    // For non-void return types
    template <typename RemoteFunc>
    [[nodiscard]] auto if_remote(RemoteFunc&& on_remote) const -> std::
        enable_if_t<!std::is_void_v<std::invoke_result_t<RemoteFunc>>, std::optional<std::invoke_result_t<RemoteFunc>>>;

    // Equality operators
    [[nodiscard]] bool operator==(const MaybeRemote& other) const = default;

    // Debugging support
    [[nodiscard]] std::string to_string() const;

};

// Type aliases for common use cases
using MaybeRemoteDeviceId = MaybeRemote<int>;
using MaybeRemoteDevice = MaybeRemote<IDevice*>;

// ============================================================================
// MaybeRemote Implementation
// ============================================================================

template <typename T>
MaybeRemote<T>::MaybeRemote(RemoteDevice) : value_(RemoteDevice{}) {}

template <typename T>
MaybeRemote<T>::MaybeRemote(T value) : value_(std::move(value)) {}

template <typename T>
[[noreturn]] void MaybeRemote<T>::throw_remote_access_error() {
    TT_THROW("Attempted to access remote device from process that does not have it!");
}

template <typename T>
MaybeRemote<T> MaybeRemote<T>::local(T value) {
    return MaybeRemote(std::move(value));
}

template <typename T>
MaybeRemote<T> MaybeRemote<T>::remote() {
    return MaybeRemote(RemoteDevice{});
}

template <typename T>
bool MaybeRemote<T>::is_local() const noexcept {
    return std::holds_alternative<T>(value_);
}

template <typename T>
bool MaybeRemote<T>::is_remote() const noexcept {
    return std::holds_alternative<RemoteDevice>(value_);
}

template <typename T>
T& MaybeRemote<T>::value() & {
    return *this->operator->();
}

template <typename T>
const T& MaybeRemote<T>::value() const& {
    return *this->operator->();
}

template <typename T>
T& MaybeRemote<T>::operator*() & {
    return value();
}

template <typename T>
const T& MaybeRemote<T>::operator*() const& {
    return value();
}

template <typename T>
T* MaybeRemote<T>::operator->() {
    if (auto* p = std::get_if<T>(&value_); p) {
        return p;  // local -– hand back the pointer
    }
    throw_remote_access_error();  // remote – raise
}

template <typename T>
const T* MaybeRemote<T>::operator->() const {
    if (const auto* p = std::get_if<T>(&value_); p) {
        return p;  // local -– hand back the pointer
    }
    throw_remote_access_error();  // remote – raise
}

template <typename T>
template <typename LocalFunc, typename RemoteFunc>
auto MaybeRemote<T>::when(LocalFunc&& on_local, RemoteFunc&& on_remote) const -> decltype(auto) {
    if (is_local()) {
        return std::invoke(std::forward<LocalFunc>(on_local), std::get<T>(value_));
    } else {
        return std::invoke(std::forward<RemoteFunc>(on_remote));
    }
}

// if_local for void return types
template <typename T>
template <typename LocalFunc>
auto MaybeRemote<T>::if_local(LocalFunc&& on_local) const
    -> std::enable_if_t<std::is_void_v<std::invoke_result_t<LocalFunc, const T&>>> {
    if (is_local()) {
        std::invoke(std::forward<LocalFunc>(on_local), std::get<T>(value_));
    }
}

// if_local for non-void return types
template <typename T>
template <typename LocalFunc>
auto MaybeRemote<T>::if_local(LocalFunc&& on_local) const -> std::enable_if_t<
    !std::is_void_v<std::invoke_result_t<LocalFunc, const T&>>,
    std::optional<std::invoke_result_t<LocalFunc, const T&>>> {
    if (is_local()) {
        return std::invoke(std::forward<LocalFunc>(on_local), std::get<T>(value_));
    }
    return std::nullopt;
}

// if_remote for void return types
template <typename T>
template <typename RemoteFunc>
auto MaybeRemote<T>::if_remote(RemoteFunc&& on_remote) const
    -> std::enable_if_t<std::is_void_v<std::invoke_result_t<RemoteFunc>>> {
    if (is_remote()) {
        std::invoke(std::forward<RemoteFunc>(on_remote));
    }
}

// if_remote for non-void return types
template <typename T>
template <typename RemoteFunc>
auto MaybeRemote<T>::if_remote(RemoteFunc&& on_remote) const -> std::
    enable_if_t<!std::is_void_v<std::invoke_result_t<RemoteFunc>>, std::optional<std::invoke_result_t<RemoteFunc>>> {
    if (is_remote()) {
        return std::invoke(std::forward<RemoteFunc>(on_remote));
    }
    return std::nullopt;
}

template <typename T>
std::string MaybeRemote<T>::to_string() const {
    return when(
        [](const T& value) -> std::string {
            std::ostringstream oss;
            oss << "MaybeRemote{" << value << "}";
            return oss.str();
        },
        []() -> std::string { return "MaybeRemote{remote}"; }
    );
}

// ============================================================================
// Utility Functions
// ============================================================================

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

}  // namespace tt::tt_metal::distributed
