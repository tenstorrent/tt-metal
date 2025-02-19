// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <boost/core/span.hpp>

namespace tt::stl {

using boost::dynamic_extent;

namespace detail {

template <class T, std::size_t Extent>
class SpanBase : public boost::span<T, Extent> {
public:
    using boost::span<T, Extent>::span;
};

template <class T, std::size_t Extent>
class SpanBase<const T, Extent> : public boost::span<const T, Extent> {
public:
    using boost::span<const T>::span;

    // expose constructor from initializer_list for const-qualified element_type
    explicit(Extent != dynamic_extent) constexpr SpanBase(std::initializer_list<T> ilist) noexcept :
        boost::span<const T>(ilist) {}
};

}  // namespace detail

template <class T, std::size_t Extent = dynamic_extent>
class Span : detail::SpanBase<T, Extent> {
    using base = detail::SpanBase<T, Extent>;

public:
    // Member types
    using typename base::const_pointer;
    using typename base::const_reference;
    using typename base::difference_type;
    using typename base::element_type;
    using typename base::iterator;
    using typename base::pointer;
    using typename base::reference;
    using typename base::reverse_iterator;
    using typename base::size_type;
    using typename base::value_type;

    // Member constants
    using base::extent;

    using base::base;
    using base::operator=;

    // Iterators
    using base::begin;
    using base::end;
    using base::rbegin;
    using base::rend;

    // Element access
    using base::back;
    using base::front;
    using base::operator[];
    using base::data;

    // Observers
    using base::empty;
    using base::size;
    using base::size_bytes;

    // Subviews
    using base::first;
    using base::last;
    using base::subspan;
};

template <class It, class EndOrSize>
Span(It, EndOrSize) -> Span<std::remove_reference_t<decltype(*std::declval<It&>())>>;

template <class T, std::size_t N>
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
Span(T (&)[N]) -> Span<T, N>;

template <class T, std::size_t N>
Span(std::array<T, N>&) -> Span<T, N>;

template <class T, std::size_t N>
Span(const std::array<T, N>&) -> Span<const T, N>;

template <class R>
Span(R&&) -> Span<std::remove_reference_t<decltype(*std::begin(std::declval<R&>()))>>;

}  // namespace tt::stl
