// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensix_functions.h"

#ifdef MODELT_FOO
#include "l1_cache.h"

template <class T>
class atomic_rwptr {
   public:
    atomic_rwptr(uint address, modelt::l1_cache &l1) : m_address(address), m_l1(l1) {}

    T load() const noexcept { return m_l1.read_word(m_address); }
    void store(T value) noexcept { m_l1.write_word(m_address, value); }

    T operator=(T value) noexcept {
        store(value);
        return value;
    }
    atomic_rwptr &operator=(const atomic_rwptr &) = delete;

    operator T() const noexcept { return load(); }

    // This is not RMW atomic, they assume this core is the only writer.
    T operator+=(T add) noexcept {
        T value = load() + add;
        store(value);
        return value;
    }

   private:
    uint m_address;
    modelt::l1_cache &m_l1;
};

#else
#ifdef TENSIX_FIRMWARE

// Atomic using only atomic load and store for single producer / single consumer FIFO read & write pointers.
template <class T>
class atomic_rwptr {
   public:
    atomic_rwptr() noexcept = default;
    atomic_rwptr(const atomic_rwptr &) = delete;

    T load() const noexcept {
        T value = underlying;
        fence();
        return value;
    }
    void store(T value) noexcept {
        fence();
        underlying = value;
    }

    T operator=(T value) noexcept {
        store(value);
        return value;
    }
    atomic_rwptr &operator=(const atomic_rwptr &) = delete;

    operator T() const noexcept { return load(); }

    // This is not RMW atomic, they assume this core is the only writer.
    T operator+=(T add) noexcept {
        fence();
        T value = underlying + add;
        underlying = value;
        fence();
        return value;
    }

   private:
    void fence() const volatile noexcept { clobber_all_memory(); }

    volatile T underlying;
};

#else

#include <atomic>

template <class T>
using atomic_rwptr = std::atomic<T>;

#endif

template <class T>
inline atomic_rwptr<T> &make_atomic_rwptr(T *p) {
    return reinterpret_cast<atomic_rwptr<T> &>(*p);
}

#endif
