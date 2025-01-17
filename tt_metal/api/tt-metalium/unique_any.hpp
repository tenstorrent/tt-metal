// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <any>
#include <array>

#include "concepts.hpp"

namespace tt::stl {

template <auto MAX_STORAGE_SIZE, auto ALIGNMENT>
struct unique_any final {
    using storage_t = std::array<std::byte, MAX_STORAGE_SIZE>;

    template <typename Type, typename BaseType = std::decay_t<Type>>
    unique_any(Type&& object) :
        pointer{new(&type_erased_storage) BaseType{std::move(object)}},
        delete_storage{[](storage_t& self) { reinterpret_cast<BaseType*>(&self)->~BaseType(); }},
        move_storage{[](storage_t& self, void* other) -> void* {
            if constexpr (std::is_move_constructible_v<BaseType>) {
                return new (&self) BaseType{std::move(*reinterpret_cast<BaseType*>(other))};
            } else {
                static_assert(tt::stl::concepts::always_false_v<BaseType>);
            }
        }} {
        static_assert(sizeof(BaseType) <= MAX_STORAGE_SIZE);
        static_assert(ALIGNMENT % alignof(BaseType) == 0);
    }

    void destruct() noexcept {
        if (this->pointer) {
            this->delete_storage(this->type_erased_storage);
        }
        this->pointer = nullptr;
    }

    unique_any(const unique_any& other) = delete;
    unique_any& operator=(const unique_any& other) = delete;

    unique_any(unique_any&& other) :
        pointer{other.pointer ? other.move_storage(this->type_erased_storage, other.pointer) : nullptr},
        delete_storage{other.delete_storage},
        move_storage{other.move_storage} {}

    unique_any& operator=(unique_any&& other) {
        if (other.pointer != this->pointer) {
            this->destruct();
            this->pointer = nullptr;
            if (other.pointer) {
                this->pointer = other.move_storage(this->type_erased_storage, other.pointer);
            }
            this->delete_storage = other.delete_storage;
            this->move_storage = other.move_storage;
        }
        return *this;
    }

    ~unique_any() { this->destruct(); }

    template <typename T>
    T& get() {
        return *reinterpret_cast<T*>(&type_erased_storage);
    }

    template <typename T>
    const T& get() const {
        return *reinterpret_cast<const T*>(&type_erased_storage);
    }

private:
    alignas(ALIGNMENT) void* pointer = nullptr;
    alignas(ALIGNMENT) storage_t type_erased_storage;

    void (*delete_storage)(storage_t&) = nullptr;
    void* (*move_storage)(storage_t& storage, void*) = nullptr;
};

}  // namespace tt::stl
