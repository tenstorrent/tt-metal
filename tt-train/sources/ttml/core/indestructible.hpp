// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <utility>

namespace ttml::core {

template <typename T>
class Indestructible {
public:
    template <typename... Args>
    explicit Indestructible(Args&&... args) {
        // Construct T in our aligned storage
        new (&storage) T(std::forward<Args>(args)...);
    }

    T& get() {
        return *reinterpret_cast<T*>(&storage);
    }

    const T& get() const {
        return *reinterpret_cast<const T*>(&storage);
    }

    // Disable copy and assignment
    Indestructible(const Indestructible&) = delete;
    Indestructible& operator=(const Indestructible&) = delete;

    // Destructor does NOT call T's destructor.
    // This leaves the object "indestructible."
    ~Indestructible() = default;

private:
    // A buffer of unsigned char with alignment of T and size of T
    alignas(T) unsigned char storage[sizeof(T)];
};

}  // namespace ttml::core
