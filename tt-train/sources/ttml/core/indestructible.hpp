// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <type_traits>
#include <utility>

namespace ttml::core {

template <typename T>
class Indestructible {
public:
    // Create (or retrieve) the single indestructible instance of T.
    // This constructs T only once and returns a reference.
    template <typename... Args>
    static T& get_instance(Args&&... args) {
        static bool constructed = false;
        static typename std::aligned_storage<sizeof(T), alignof(T)>::type storage;
        if (!constructed) {
            // Construct the object in the allocated storage
            new (&storage) T(std::forward<Args>(args)...);
            constructed = true;
        }
        // Return a reference to the constructed object
        return *reinterpret_cast<T*>(&storage);
    }
    // Disallow creating instances of Indestructible itself
    Indestructible() = delete;
    Indestructible(const Indestructible&) = delete;
    Indestructible& operator=(const Indestructible&) = delete;

private:
    // Private destructor to prevent external destruction.
    // Even if called, it won't be used since we never create `Indestructible` objects.
    ~Indestructible() = default;
};
}  // namespace ttml::core
