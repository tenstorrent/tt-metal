// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/ndarray.h>

#include "nb_fwd.hpp"
#include "ttnn/tensor/tensor.hpp"

nb::ndarray<nb::numpy> make_numpy_tensor(
    const tt::tt_metal::Tensor& tensor, std::optional<tt::tt_metal::DataType> new_type = std::nullopt);
tt::tt_metal::Tensor make_metal_tensor(
    nb::ndarray<> data,
    tt::tt_metal::Layout layout = tt::tt_metal::Layout::TILE,
    std::optional<tt::tt_metal::DataType> new_type = std::nullopt);

template <typename T>
consteval bool same_as_one_of() {
    return true;
};
template <typename T, typename U, typename... Args>
consteval bool same_as_one_of() {
    return std::same_as<T, U> || same_as_one_of<T, Args...>();
}

[[noreturn]] void throw_exception(
    std::source_location source_location,
    nb::exception_type exception_type,
    auto exception_name,
    auto condition_str,
    auto msg);

#define NB_THROW(exception_type, message, ...)    \
    do {                                          \
        throw_exception(                          \
            std::source_location::current(),      \
            exception_type,                       \
            #exception_type,                      \
            "NB_THROW",                           \
            fmt::format(message, ##__VA_ARGS__)); \
    } while (0)

#define NB_COND_THROW(condition, exception_type, message, ...) \
    do {                                                       \
        if (!condition) {                                      \
            throw_exception(                                   \
                std::source_location::current(),               \
                exception_type,                                \
                #exception_type,                               \
                #condition,                                    \
                fmt::format(message, ##__VA_ARGS__));          \
        }                                                      \
    } while (0)
