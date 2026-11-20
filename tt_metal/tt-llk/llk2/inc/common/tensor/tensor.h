// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "llk_assert.h"   // LLK_ASSERT
#include "stdx/algorithm.hpp" // stdx::equal, stdx::ranges::equal
#include "stdx/array.hpp"     // stdx::to_array
#include "stdx/type_traits.hpp" // stdx::remove_cvref_t
#include "tensix_types.h" // DataFormat
#include "wrapper/wrapper.h" // llk::ct, llk::rt, llk::ctrt

namespace llk
{

// ---- tile geometry constants (the same on every arch) ---------------------------------------------------------------
// todo(sstanisic): Most of these are actually arch dependant, only not currently, they should live in some arch dependant geometry or smth like that.
// todo(sstanisic): also tensor is supposed to be an abstract concept that is not tied to the tensix. this tightly couples it.
inline constexpr std::uint32_t face_cols          = 16;                 ///< datums in a face row; every face is this wide
inline constexpr std::uint32_t max_face_rows      = 16;                 ///< rows in a full face; shorter tiles have shorter faces
inline constexpr std::uint32_t default_tile_dim   = 32;                 ///< tile height and width when the caller names none
inline constexpr std::uint32_t dest_row_datums    = 16;                 ///< datums in one Dest row
inline constexpr std::uint32_t min_dest_face_rows = 8;                  ///< every face takes at least an 8-row slot in Dest (Quasar's tiny_face_stride)
inline constexpr std::uint32_t max_tiny_face_rows = min_dest_face_rows; ///< the tallest tiny tile: tiny tiles are 1, 2, 4 or 8 rows
inline constexpr std::uint32_t bfp_block_datums   = 16;                 ///< block float formats share one exponent byte per 16 datums
inline constexpr std::uint32_t l1_word_bytes      = 16;                 ///< L1 word: tiles in L1 are padded to whole words, and the unpacker's and
                                                                        ///< packer's addresses and tile sizes count words
inline constexpr std::uint32_t bits_per_byte      = 8;                  ///< bits in one byte, for datum_bits to bytes
inline constexpr std::uint32_t faces_per_face_row = default_tile_dim / face_cols; ///< faces side by side in a full tile: 2
inline constexpr std::uint32_t faces_per_full_tile =
    faces_per_face_row * (default_tile_dim / max_face_rows); ///< faces in a full 32x32 tile: 2 x 2, stored 0, 1 (top), 2, 3 (bottom)
inline constexpr std::uint32_t default_tile_datums = default_tile_dim * default_tile_dim; ///< datums in one 32x32 tile



enum class Storage : std::uint8_t
{
    L1,
    Dest
};

enum class OwnerThread : std::uint8_t
{
    Compute0,
    Compute1,
    Compute2,
    Compute3,
    None
};

enum class OwnerExu : std::uint8_t
{
    Unpack,
    Fpu,
    Sfpu,
    Pack,
    None
};

// todo(sstanisic): this likely shouldn't live here?
inline void tensor_shape_requirement_failed(const char* why) {
    LLK_ASSERT(false, why);
    (void)why;
}

struct NoRelease
{
    constexpr inline void operator()() const
    {
    }
};

// todo(sstanisic): clean up bellow.

namespace detail
{

/// x as a ct or an rt of T: a ct stays compile time (and must fit T), an rt or a plain value is run time
template <class T, class X>
constexpr auto as(const X& x)
{
    if constexpr (wrapper::is_ct_v<X>)
    {
        return ct<T {wrapper::constant<X>}> {};
    }
    else
    {
        return rt<T>(static_cast<T>(wrapper::value(x)));
    }
}

template <std::size_t Begin, class T, std::size_t... I>
constexpr auto sub_impl(const T& t, std::index_sequence<I...>)
{
    return std::make_tuple(std::get<Begin + I>(t)...);
}

/// Elements [Begin, Begin + Count) of t
template <std::size_t Begin, std::size_t Count, class T>
constexpr auto sub(const T& t)
{
    return sub_impl<Begin>(t, std::make_index_sequence<Count> {});
}

/// t with element Dim removed
template <std::size_t Dim, class T>
constexpr auto remove(const T& t)
{
    return std::tuple_cat(sub<0, Dim>(t), sub<Dim + 1, std::tuple_size<T>::value - Dim - 1>(t));
}

/// t with the elements of `with` in place of the Count elements from Dim (Count 0 inserts)
template <std::size_t Dim, std::size_t Count, class T, class W>
constexpr auto splice(const T& t, const W& with)
{
    return std::tuple_cat(sub<0, Dim>(t), with, sub<Dim + Count, std::tuple_size<T>::value - Dim - Count>(t));
}

/// Each element of t as a ct or an rt of T
template <class T, class Tuple>
constexpr auto elements_as(const Tuple& t)
{
    return std::apply([](const auto&... v) { return std::make_tuple(as<T>(v)...); }, t);
}

/// The product of the elements of t: a ct when all are
template <class T>
constexpr auto product(const T& t)
{
    return std::apply([](const auto&... v) { return (ct<std::size_t {1}>() * ... * v); }, t);
}

/// Each element of t times k
template <class T, class K>
constexpr auto scale(const T& t, const K& k)
{
    return std::apply([&](const auto&... v) { return std::make_tuple((v * k)...); }, t);
}

template <class A, class B, std::size_t... I>
constexpr auto all_equal_impl(const A& a, const B& b, std::index_sequence<I...>)
{
    return (ct<true>() && ... && (std::get<I>(a) == std::get<I>(b)));
}

template <class T, std::size_t... I>
constexpr auto contiguous_strides_impl(const T& sizes, std::index_sequence<I...>)
{
    return std::make_tuple(product(sub<I + 1, sizeof...(I) - 1 - I>(sizes))...);
}

template <class T, std::size_t... I>
constexpr std::array<std::size_t, sizeof...(I)> values_impl(const T& t, std::index_sequence<I...>)
{
    return {{static_cast<std::size_t>(wrapper::value(std::get<I>(t)))...}};
}

/// The first K numbers of t as plain values, for run-time loops
template <std::size_t K, class T>
constexpr std::array<std::size_t, K> values(const T& t)
{
    return values_impl(t, std::make_index_sequence<K> {});
}

/// The larger of a and b: a ct when both are
template <class A, class B>
constexpr auto max_of(const A& a, const B& b)
{
    if constexpr (wrapper::is_ct_v<A> && wrapper::is_ct_v<B>)
    {
        return ct<(wrapper::constant<A> < wrapper::constant<B> ? wrapper::constant<B> : wrapper::constant<A>)>();
    }
    else
    {
        return rt(wrapper::value(a) < wrapper::value(b) ? wrapper::value(b) : wrapper::value(a));
    }
}

/// The smaller of a and b: a ct when both are
template <class A, class B>
constexpr auto min_of(const A& a, const B& b)
{
    if constexpr (wrapper::is_ct_v<A> && wrapper::is_ct_v<B>)
    {
        return ct<(wrapper::constant<B> < wrapper::constant<A> ? wrapper::constant<B> : wrapper::constant<A>)>();
    }
    else
    {
        return rt(wrapper::value(b) < wrapper::value(a) ? wrapper::value(b) : wrapper::value(a));
    }
}

} // namespace detail

/// A ct condition is checked by the compiler, an rt condition when the code runs
template <class C>
constexpr void require(const C& ok, const char* why)
{
    if constexpr (wrapper::is_ct_v<C>)
    {
        static_assert(wrapper::constant<C>, "tensor shape requirement failed: see the instantiation trace");
        (void)ok;
        (void)why;
    }
    else if (!wrapper::value(ok))
    {
        tensor_shape_requirement_failed(why);
    }
}

/// a and b hold the same numbers: a ct<bool> when every pair is compile time
template <class A, class B>
constexpr auto all_equal(const A& a, const B& b)
{
    static_assert(std::tuple_size<A>::value == std::tuple_size<B>::value, "ranks differ");
    return detail::all_equal_impl(a, b, std::make_index_sequence<std::tuple_size<A>::value> {});
}

/// The strides of a dense layout of `sizes`: each the product of the sizes after it
template <class T>
constexpr auto contiguous_strides(const T& sizes)
{
    return detail::contiguous_strides_impl(sizes, std::make_index_sequence<std::tuple_size<T>::value> {});
}

// ---- TensorShape ---------------------------------------------------------------------------------------------------

template <class Sizes = std::tuple<>, class Strides = std::tuple<>>
class TensorShape;

template <class... D, class... S>
class TensorShape<std::tuple<D...>, std::tuple<S...>>
{
    static_assert(sizeof...(D) == sizeof...(S), "one stride per dimension");

    std::size_t offset_ = 0; // in elements, from the start of storage
    std::tuple<ctrt<D, std::size_t>...> size_;
    std::tuple<ctrt<S, std::size_t>...> stride_;

   public:
    static constexpr std::size_t rank = sizeof...(D);

    /// Every size and stride is compile time
    static constexpr bool is_static = (wrapper::is_ct_v<D> && ...) && (wrapper::is_ct_v<S> && ...);

    /// The shape its type describes, at offset 0: only when every size and stride is compile time
    constexpr TensorShape() = default;

    constexpr TensorShape(
        std::size_t offset,
        const std::tuple<ctrt<D, std::size_t>...>& sizes,
        const std::tuple<ctrt<S, std::size_t>...>& strides
    )
        : offset_(offset), size_(sizes), stride_(strides) {}

    // ---- create ------------------------------------------------------------

    template <class... X, Requires<(wrapper::is_anytime_v<X, std::size_t> && ...)> = 0>
    static constexpr auto create(const std::tuple<X...>& sizes)
    {
        return create(0, detail::elements_as<std::size_t>(sizes));
    }

    template <class... X, Requires<(wrapper::is_anytime_v<X, std::size_t> && ...)> = 0>
    static constexpr auto create(std::size_t offset, const std::tuple<X...>& sizes)
    {
        return create(offset, sizes, contiguous_strides(detail::elements_as<std::size_t>(sizes)));
    }

    template <class... X, class... Y, Requires<(wrapper::is_anytime_v<X, std::size_t> && ...)> = 0, Requires<(wrapper::is_anytime_v<Y, std::size_t> && ...)> = 0>
    static constexpr auto create(std::size_t offset, const std::tuple<X...>& sizes, const std::tuple<Y...>& strides)
    {
        auto size   = detail::elements_as<std::size_t>(sizes); // each a ct<V> or an rt<std::size_t>: the forms D and S
        auto stride = detail::elements_as<std::size_t>(strides);
        return TensorShape<decltype(size), decltype(stride)>(offset, size, stride);
    }

    // ---- what it is ----------------------------------------------------------------------------------------------

    constexpr std::size_t offset() const { return offset_; }
    constexpr const auto& sizes() const { return size_; }
    constexpr const auto& strides() const { return stride_; }

    // todo(sstanisic): assert based on DIM
    template <auto Dim>
    constexpr auto size(const ct<Dim>) const { return std::get<Dim>(size_); }

    // todo(sstanisic): assert based on DIM
    template <auto Dim>
    constexpr auto stride(const ct<Dim>) const { return std::get<Dim>(stride_); }

    constexpr auto elements() const { return detail::product(size_); }

    constexpr auto is_contiguous() const { return all_equal(stride_, contiguous_strides(size_)); }

    // ---- views: dimensions are ct arguments; indices, starts and lengths are any form ----------------------------

    /// permute(std::tuple {ct<2>(), ct<0>(), ct<1>()}): dimension i of the result is dimension dims[i] of this
    template <auto... P>
    constexpr auto permute(const std::tuple<ct<P>...>&) const
    {
        // todo(sstanisic): assert that dims is a permutation of 0..N-1
        static_assert(sizeof...(P) == rank, "permute names every dimension");
        return TensorShape<>::create(
            offset_,
            std::make_tuple(std::get<P>(size_)...),
            std::make_tuple(std::get<P>(stride_)...)
        );
    }

    /// transpose(ct<0>(), ct<2>())
    // todo(sstanisic): assert first and second are in range
    template <auto A, auto B>
    constexpr auto transpose(const ct<A>, const ct<B>) const { return transpose_impl<A, B>(std::make_index_sequence<rank> {}); }

    /// narrow(dim, start, length): `length` positions of dim from `start`
    template <auto Dim, class Start, class Length>
    constexpr auto narrow(const ct<Dim>, const Start& start, const Length& length) const
    {
        // todo(sstanisic): assert dim < N and start + length <= size(dim)
        constexpr std::size_t At = Dim;
        require(detail::as<std::size_t>(start) + detail::as<std::size_t>(length) <= std::get<At>(size_), "narrow: past the end");
        return TensorShape<>::create(
            offset_ + wrapper::value(start) * wrapper::value(std::get<At>(stride_)), detail::splice<At, 1>(size_, std::make_tuple(length)), stride_);
    }

    /// slice(dim, start, end, step): every step-th position of dim in [start, end)
    template <auto Dim, class Start, class End, class Step = ct<1>>
    constexpr auto slice(const ct<Dim>, const Start& start, const End& end, const Step& step = Step {}) const
    {
        // todo(sstanisic): assert dim < N, step > 0 and start <= end <= size(dim)
        constexpr std::size_t At = Dim;
        const auto first = detail::as<std::size_t>(start);
        const auto last  = detail::as<std::size_t>(end);
        const auto by    = detail::as<std::size_t>(step);
        require(first <= last && last <= std::get<At>(size_), "slice: out of range");
        return TensorShape<>::create(
            offset_ + wrapper::value(start) * wrapper::value(std::get<At>(stride_)),
            detail::splice<At, 1>(size_, std::make_tuple((last - first + by - ct<std::size_t {1}>()) / by)),
            detail::splice<At, 1>(stride_, std::make_tuple(std::get<At>(stride_) * by)));
    }

    /// select(dim, index): position `index` of dim; the dimension is gone
    template <auto Dim, class Index>
    constexpr auto select(const ct<Dim>, const Index& index) const
    {
        // todo(sstanisic): assert dims are distinct and < N, and indices[k] < size(dims[k])
        constexpr std::size_t At = Dim;
        require(detail::as<std::size_t>(index) < std::get<At>(size_), "select: index out of range");
        return TensorShape<>::create(offset_ + wrapper::value(index) * wrapper::value(std::get<At>(stride_)), detail::remove<At>(size_), detail::remove<At>(stride_));
    }

    /// squeeze(dim): drops a dimension of size 1
    template <auto Dim>
    constexpr auto squeeze(const ct<Dim>) const
    {
        // todo(sstanisic): assert size(dim) == 1
        constexpr std::size_t At = Dim;
        require(std::get<At>(size_) == ct<std::size_t {1}>(), "squeeze: the dimension is not 1");
        return TensorShape<>::create(offset_, detail::remove<At>(size_), detail::remove<At>(stride_));
    }

    /// unsqueeze(dim): a new dimension of size 1 before dim
    template <auto Dim>
    constexpr auto unsqueeze(const ct<Dim>) const
    {
        // todo(sstanisic): assert dim <= N
        constexpr std::size_t At = Dim;
        const auto outer_stride = detail::product(detail::sub<At, rank - At>(size_)); // dense: as for a contiguous shape
        return TensorShape<>::create(
            offset_, detail::splice<At, 0>(size_, std::make_tuple(ct<std::size_t {1}>())), detail::splice<At, 0>(stride_, std::make_tuple(outer_stride)));
    }

    /// expand(dim, length): repeats a dimension of size 1 `length` times (stride 0)
    template <auto Dim, class Length>
    constexpr auto expand(const ct<Dim>, const Length& length) const
    {
        // todo(sstanisic): assert dim < N and (length == size(dim) or size(dim) == 1)
        constexpr std::size_t At = Dim;
        require(std::get<At>(size_) == ct<std::size_t {1}>(), "expand: the dimension is not 1");
        return TensorShape<>::create(
            offset_,
            detail::splice<At, 1>(size_, std::make_tuple(length)),
            detail::splice<At, 1>(stride_, std::make_tuple(ct<std::size_t {0}>())));
    }

    /// unflatten(dim, std::tuple {a, b, ...}): splits dim into dimensions of sizes a, b, ...
    template <auto Dim, class... Parts>
    constexpr auto unflatten(const ct<Dim>, const std::tuple<Parts...>& parts) const
    {
        // todo(sstanisic): assert dim < N and the product of split_size equals size(dim)
        constexpr std::size_t At = Dim;
        const auto split         = detail::elements_as<std::size_t>(parts);
        require(detail::product(split) == std::get<At>(size_), "unflatten: the sizes do not multiply to the dimension");
        return TensorShape<>::create(offset_, detail::splice<At, 1>(size_, split), detail::splice<At, 1>(stride_, detail::scale(contiguous_strides(split), std::get<At>(stride_))));
    }

    /// flatten(start, count): merges `count` dimensions from `start` into one; they must be dense with respect to each other
    template <auto Start, auto Count>
    constexpr auto flatten(const ct<Start>, const ct<Count>) const
    {
        // todo(sstanisic): assert start + K <= N
        constexpr std::size_t First = Start;
        constexpr std::size_t K     = Count;
        const auto group            = detail::sub<First, K>(size_);
        const auto last             = std::get<First + K - 1>(stride_);
        require(all_equal(detail::sub<First, K>(stride_), detail::scale(contiguous_strides(group), last)), "flatten: the dimensions are not dense");
        return TensorShape<>::create(
            offset_, detail::splice<First, K>(size_, std::make_tuple(detail::product(group))), detail::splice<First, K>(stride_, std::make_tuple(last)));
    }

    /// view(std::tuple {a, b, ...}): the same elements under new sizes. Contiguous shapes only
    template <class... Sizes>
    constexpr auto view(const std::tuple<Sizes...>& sizes) const
    {
        // todo(sstanisic): assert the product of view_size equals elements()
        const auto to = detail::elements_as<std::size_t>(sizes);
        require(is_contiguous() && detail::product(to) == elements(), "view: needs a contiguous shape of as many elements");
        return TensorShape<>::create(offset_, to, contiguous_strides(to));
    }

    // ---- walking -------------------------------------------------------------------------------------------------

    /// The positions of the first K dimensions, last fastest; each step is the shape of the remaining dimensions.
    /// The walked sizes can be run time: the loop then runs at run time.
    template <std::size_t K>
    class Walk
    {
        TensorShape shape_;
        std::array<std::size_t, K> size_;
        std::array<std::size_t, K> stride_;

       public:
        class iterator
        {
            const Walk* walk_;
            std::array<std::size_t, K> index_;

           public:
            constexpr iterator(const Walk* walk, const std::array<std::size_t, K>& index) : walk_(walk), index_(index) {}

            constexpr auto operator*() const
            {
                std::size_t offset = walk_->shape_.offset_;
                for (std::size_t k = 0; k < K; ++k)
                {
                    offset += index_[k] * walk_->stride_[k];
                }
                return TensorShape<>::create(offset, detail::sub<K, rank - K>(walk_->shape_.size_), detail::sub<K, rank - K>(walk_->shape_.stride_));
            }

            constexpr iterator& operator++()
            {
                for (std::size_t k = K; k-- > 0;)
                {
                    if (++index_[k] < walk_->size_[k] || k == 0)
                    {
                        break;
                    }
                    index_[k] = 0;
                }
                return *this;
            }

            constexpr bool operator!=(const iterator& other) const
            {
                for (std::size_t k = 0; k < K; ++k)
                {
                    if (index_[k] != other.index_[k])
                    {
                        return true;
                    }
                }
                return false;
            }
        };

        constexpr explicit Walk(const TensorShape& shape) : shape_(shape), size_(detail::values<K>(shape.size_)), stride_(detail::values<K>(shape.stride_)) {}

        constexpr iterator end() const
        {
            std::array<std::size_t, K> index {};
            index[0] = size_[0];
            return iterator(this, index);
        }

        constexpr iterator begin() const
        {
            for (std::size_t k = 0; k < K; ++k)
            {
                if (size_[k] == 0)
                {
                    return end();
                }
            }
            return iterator(this, std::array<std::size_t, K> {});
        }
    };

    /// outer(ct<2>()): walks the first 2 dimensions
    template <auto Count>
    constexpr auto outer(const ct<Count>) const
    {
        // todo(sstanisic): assert dims in `outer` are distinct and < N
        constexpr std::size_t K = Count;
        static_assert(K >= 1 && K <= rank, "outer(k): k dimensions of the shape");
        return Walk<K>(*this);
    }
};

template <class... D, class... S>
constexpr TensorShape<std::tuple<D...>, std::tuple<S...>> make_shape(
    std::size_t offset, const std::tuple<ctrt<D, std::size_t>...>& sizes, const std::tuple<ctrt<S, std::size_t>...>& strides)
{
    return {offset, sizes, strides};
}

// ---- TensorSpec and Tensor -------------------------------------------------------------------------------------------

/// A shape in a format, in a storage. The format and the storage stay compile time, in the type.
template <class Shape, DataFormat F, Storage W>
class TensorSpec
{
    Shape shape_;

   public:
    /// The spec its type describes: only when the shape is compile time
    constexpr TensorSpec() = default;

    constexpr explicit TensorSpec(const Shape& shape) : shape_(shape) {}

    constexpr const Shape& shape() const { return shape_; }
    static constexpr std::size_t rank() { return Shape::rank; }
    static constexpr DataFormat format() { return F; }
    static constexpr Storage storage() { return W; }
};

/**
 * @brief A tensor: its spec and release policy in its type; the address and the owner at run time
 *
 * An L1 address is valid only on the thread that reads or writes the tensor (see llk_lib/thread.h). ReleasePolicy is
 * a base, so an empty one costs nothing. The spec is held by value: its compile-time parts hold no data. Offers the
 * spec's queries and views: a view is a Tensor with another shape and the same address, owner and release.
 */
template <class Spec, class ReleasePolicy = NoRelease>
class Tensor : public ReleasePolicy
{
    Spec spec_;
    std::uint32_t address_; // L1: byte address of datum 0; Dest: the absolute first row
    // todo(sstanisic): move to Tensor, these are runtime
    OwnerThread thread_;    // the owner: its thread, OwnerThread::None until an op writes the tensor
    OwnerExu exu_;          // the owner: its unit

    // This tensor under another shape
    template <class Shape>
    constexpr auto with_shape(const Shape& shape) const
    {
        using NewSpec = TensorSpec<Shape, Spec::format(), Spec::storage()>;
        return Tensor<NewSpec, ReleasePolicy>(NewSpec(shape), address_, thread_, exu_, release_policy());
    }

   public:
    /// A tensor nobody has written yet, of `spec` at address a, released by r
    constexpr Tensor(const Spec& spec, std::uint32_t a, ReleasePolicy r = ReleasePolicy {}) : Tensor(spec, a, OwnerThread::None, OwnerExu::None, r) {}

    /// A tensor of `spec` at address a, owned by (thread, exu), released by r
    constexpr Tensor(const Spec& spec, std::uint32_t a, OwnerThread thread, OwnerExu exu, ReleasePolicy r = ReleasePolicy {})
        : ReleasePolicy(r), spec_(spec), address_(a), thread_(thread), exu_(exu) {}

    /// @return the tensor's spec
    constexpr const Spec& spec() const { return spec_; }

    /// @return L1: the byte address of datum 0; Dest: the absolute first row
    constexpr std::uint32_t address() const { return address_; }

    /// @return the owner's thread
    constexpr OwnerThread thread() const { return thread_; }

    /// @return the owner's unit
    constexpr OwnerExu exu() const { return exu_; }

    /// @return what release(t) runs
    constexpr const ReleasePolicy& release_policy() const { return *this; }

    /// @return this tensor under owner (thread, exu): the same type, address and release
    constexpr Tensor with_owner(OwnerThread thread, OwnerExu exu) const { return Tensor(spec_, address_, thread, exu, release_policy()); }

    // ---- the spec's queries
    constexpr const auto& shape() const { return spec_.shape(); }
    static constexpr DataFormat format() { return Spec::format(); }
    static constexpr Storage storage() { return Spec::storage(); }
    static constexpr std::size_t rank() { return Spec::rank(); }
    constexpr std::size_t offset() const { return shape().offset(); }
    constexpr const auto& sizes() const { return shape().sizes(); }
    constexpr const auto& strides() const { return shape().strides(); }
    constexpr auto elements() const { return shape().elements(); }
    constexpr auto is_contiguous() const { return shape().is_contiguous(); }

    template <class Dim>
    constexpr auto size(const Dim& dim) const { return shape().size(dim); }

    template <class Dim>
    constexpr auto stride(const Dim& dim) const { return shape().stride(dim); }

    // ---- the spec's views: the same address, owner and release
    template <class... P>
    constexpr auto permute(const std::tuple<P...>& dims) const { return with_shape(shape().permute(dims)); }

    template <class A, class B>
    constexpr auto transpose(const A& first, const B& second) const { return with_shape(shape().transpose(first, second)); }

    template <class Dim, class Start, class End, class Step = ct<1>>
    constexpr auto slice(const Dim& dim, const Start& start, const End& end, const Step& step = Step {}) const
    {
        return with_shape(shape().slice(dim, start, end, step));
    }

    template <class Dim, class Start, class Length>
    constexpr auto narrow(const Dim& dim, const Start& start, const Length& length) const { return with_shape(shape().narrow(dim, start, length)); }

    template <class Dim, class Index>
    constexpr auto select(const Dim& dim, const Index& index) const { return with_shape(shape().select(dim, index)); }

    template <class... Sizes>
    constexpr auto view(const std::tuple<Sizes...>& sizes) const { return with_shape(shape().view(sizes)); }

    template <class Dim, class Length>
    constexpr auto expand(const Dim& dim, const Length& length) const { return with_shape(shape().expand(dim, length)); }

    template <class Dim>
    constexpr auto squeeze(const Dim& dim) const { return with_shape(shape().squeeze(dim)); }

    template <class Dim>
    constexpr auto unsqueeze(const Dim& dim) const { return with_shape(shape().unsqueeze(dim)); }

    template <class Dim, class... Parts>
    constexpr auto unflatten(const Dim& dim, const std::tuple<Parts...>& parts) const { return with_shape(shape().unflatten(dim, parts)); }

    template <class Start, class Count>
    constexpr auto flatten(const Start& start, const Count& count) const { return with_shape(shape().flatten(start, count)); }
};

/// Gives t's storage back: pops or pushes its buffer pages, or frees its Dest section. Call it once per storage, on
/// any of the tensors that share it (an op's result shares its output's release).
template <class S, class R>
inline __attribute__((always_inline)) void release(const Tensor<S, R>& t)
{
    static_cast<const R&>(t)();
}

// What a compute-API check becomes: one line, checked by the compiler or at run time depending on the operands
template <class SA, class RA, class SB, class RB>
constexpr void require_same_sizes(const Tensor<SA, RA>& a, const Tensor<SB, RB>& b)
{
    require(all_equal(a.shape().sizes(), b.shape().sizes()), "the tensors must have the same sizes");
}

// ---- formats -------------------------------------------------------------------------------------------------------

/// f is a block float format (BFP8/4/2, A or B exponent).
constexpr bool is_bfp(DataFormat f)
{
    return f == DataFormat::Bfp8_b || f == DataFormat::Bfp4_b || f == DataFormat::Bfp2_b || f == DataFormat::Bfp8 || f == DataFormat::Bfp4 ||
           f == DataFormat::Bfp2;
}

/// Bits of one datum's mantissa (and sign) in L1; 0 for formats no Tensor can be made in yet.
constexpr std::uint32_t datum_bits(DataFormat f)
{
    switch (f)
    {
        case DataFormat::Float32:
        case DataFormat::Int32:
        case DataFormat::UInt32:
            return 32;
        case DataFormat::Float16:
        case DataFormat::Float16_b:
        case DataFormat::UInt16:
            return 16;
        case DataFormat::Bfp8:
        case DataFormat::Bfp8_b:
        case DataFormat::Int8:
        case DataFormat::UInt8:
            return 8;
        case DataFormat::Bfp4:
        case DataFormat::Bfp4_b:
            return 4;
        case DataFormat::Bfp2:
        case DataFormat::Bfp2_b:
            return 2;
        default:
            return 0; // not supported yet: no Tensor can be made in this format
    }
}

/// 32-bit datums: FP32, INT32, UINT32 (the formats a 32-bit Dest holds and the unpacker can write straight to Dest).
constexpr bool is_32bit_format(DataFormat f)
{
    return datum_bits(f) == 32;
}

/// value rounded up to a multiple of multiple.
constexpr std::uint32_t round_up(std::uint32_t value, std::uint32_t multiple)
{
    return (value + multiple - 1) / multiple * multiple;
}

/// Bytes of one tile of `datums` datums in L1: mantissas, plus one exponent byte per block for block formats; each
/// part padded to whole L1 words.
constexpr std::uint32_t tile_bytes(DataFormat f, std::uint32_t datums)
{
    const std::uint32_t exponents = is_bfp(f) ? round_up(datums / bfp_block_datums, l1_word_bytes) : 0;
    return round_up(datums * datum_bits(f) / bits_per_byte + exponents, l1_word_bytes);
}

// ---- tile layout ---------------------------------------------------------------------------------------------------
// The Compute API's tensors are grids of tiles in six dims, outermost first:
//   0 tile row   1 tile column   2 face row in a tile   3 face column in a tile   4 row in a face   5 column in a face

/// Dest strides for tile-layout sizes s: rows of 16 datums, a face in at least an 8-row slot (every arch).
template <class... X>
constexpr auto dest_strides(const std::tuple<X...>& s)
{
    static_assert(sizeof...(X) == 6, "tile-layout sizes: tiles_y, tiles_x, faces_y, faces_x, face_rows, face_cols");
    const auto t5 = ct<std::size_t {1}>(); // datums are contiguous along a row
    const auto t4 = ct<std::size_t {dest_row_datums}>();
    const auto t3 = t4 * detail::max_of(detail::as<std::size_t>(std::get<4>(s)), ct<std::size_t {min_dest_face_rows}>());
    const auto t2 = t3 * std::get<3>(s);
    const auto t1 = t2 * std::get<2>(s);
    const auto t0 = t1 * std::get<1>(s);
    return std::make_tuple(t0, t1, t2, t3, t4, t5);
}

/// n is a power of two.
constexpr bool is_power_of_two(std::uint32_t n)
{
    return n > 0 && (n & (n - 1)) == 0;
}

/// An h x w tile can be laid out: whole faces wide; tiny (a power of two up to max_tiny_face_rows rows) or a whole
/// number of full faces tall.
constexpr bool is_valid_tile(std::uint32_t h, std::uint32_t w)
{
    const bool tiny_height = is_power_of_two(h) && h <= max_tiny_face_rows;
    return w > 0 && w % face_cols == 0 && (tiny_height || (h > 0 && h % max_face_rows == 0));
}

/// Tile-layout sizes of a tiles_y x tiles_x grid of h x w tiles: faces of min(h, 16) rows and face_cols columns.
/// Each argument is a ct or an rt (or a plain value, taken as run time).
template <class TilesY, class TilesX, class H, class W>
constexpr auto tiled(const TilesY& tiles_y, const TilesX& tiles_x, const H& h, const W& w)
{
    const auto rows      = detail::as<std::size_t>(h);
    const auto face_rows = detail::min_of(rows, ct<std::size_t {max_face_rows}>());
    return std::make_tuple(
        detail::as<std::size_t>(tiles_y),
        detail::as<std::size_t>(tiles_x),
        rows / face_rows,
        detail::as<std::size_t>(w) / ct<std::size_t {face_cols}>(),
        face_rows,
        ct<std::size_t {face_cols}>());
}

/// A spec of tile-layout sizes in format F, with the strides of storage St (L1: dense; Dest: dest_strides), offset 0.
template <DataFormat F, Storage St, class... X>
constexpr auto laid_out(const std::tuple<X...>& sizes)
{
    if constexpr (St == Storage::Dest)
    {
        const auto shape = TensorShape<>::create(0, sizes, dest_strides(sizes));
        return TensorSpec<decltype(shape), F, St>(shape);
    }
    else
    {
        const auto shape = TensorShape<>::create(sizes);
        return TensorSpec<decltype(shape), F, St>(shape);
    }
}

/// The spec of a Rows x Cols tensor in format F, cut into H x W tiles, in storage St.
template <std::uint32_t Rows, std::uint32_t Cols, DataFormat F, std::uint32_t H, std::uint32_t W, Storage St>
constexpr auto make_spec()
{
    static_assert(is_valid_tile(H, W), "a tile is 16k wide and 1, 2, 4, 8 or 16k tall: pad the tensor first");
    static_assert(Rows % H == 0 && Cols % W == 0, "the tile must divide the tensor: pad it first");
    return laid_out<F, St>(tiled(ct<Rows / H>(), ct<Cols / W>(), ct<H>(), ct<W>()));
}

/// The spec of a Rows x Cols L1 tensor in format F, in H x W tiles (default: min(dim, 32)).
template <
    std::uint32_t Rows,
    std::uint32_t Cols,
    DataFormat F,
    std::uint32_t H = std::min(Rows, default_tile_dim),
    std::uint32_t W = std::min(Cols, default_tile_dim)>
inline constexpr auto l1_spec = make_spec<Rows, Cols, F, H, W, Storage::L1>();

/// The spec of a Rows x Cols Dest tensor in format F, in H x W tiles (default: min(dim, 32)).
template <
    std::uint32_t Rows,
    std::uint32_t Cols,
    DataFormat F,
    std::uint32_t H = std::min(Rows, default_tile_dim),
    std::uint32_t W = std::min(Cols, default_tile_dim)>
inline constexpr auto dst_spec = make_spec<Rows, Cols, F, H, W, Storage::Dest>();

/// An L1 tensor of l1_spec<Rows, Cols, F, H, W> at byte address `address`.
template <
    std::uint32_t Rows,
    std::uint32_t Cols,
    DataFormat F,
    std::uint32_t H = std::min(Rows, default_tile_dim),
    std::uint32_t W = std::min(Cols, default_tile_dim)>
constexpr auto l1_tensor(std::uint32_t address)
{
    return Tensor(l1_spec<Rows, Cols, F, H, W>, address);
}

/// A Dest tensor of dst_spec<Rows, Cols, F, H, W> from Dest row `row`.
template <
    std::uint32_t Rows,
    std::uint32_t Cols,
    DataFormat F,
    std::uint32_t H = std::min(Rows, default_tile_dim),
    std::uint32_t W = std::min(Cols, default_tile_dim)>
constexpr auto dst_tensor(std::uint32_t row)
{
    return Tensor(dst_spec<Rows, Cols, F, H, W>, row);
}

// ---- a spec as plain values, for the Resolver ------------------------------------------------------------------------

/// A tile-layout spec whose numbers are all compile time, as plain values. The Resolver's functions are reached through
/// pointers, so they cannot be templates over a spec's type: they read this. layout_of<Spec> makes one from a spec type.
class TileLayout
{
    std::size_t offset_ = 0;
    std::array<std::size_t, 6> size_ {};
    std::array<std::size_t, 6> stride_ {};
    DataFormat format_ = DataFormat::Invalid;
    Storage storage_   = Storage::L1;

   public:
    constexpr TileLayout() = default;

    constexpr TileLayout(std::size_t offset, const std::array<std::size_t, 6>& sizes, const std::array<std::size_t, 6>& strides, DataFormat format, Storage storage)
        : offset_(offset), size_(sizes), stride_(strides), format_(format), storage_(storage) {}

    constexpr std::size_t offset() const { return offset_; }
    constexpr std::size_t size(std::size_t dim) const { return size_[dim]; }
    constexpr const std::array<std::size_t, 6>& sizes() const { return size_; }
    constexpr std::size_t stride(std::size_t dim) const { return stride_[dim]; }
    constexpr const std::array<std::size_t, 6>& strides() const { return stride_; }
    constexpr DataFormat format() const { return format_; }
    constexpr Storage storage() const { return storage_; }

    /// @return the number of datums
    constexpr std::size_t elements() const { return size_[0] * size_[1] * tile_elements(); }

    /// @return the datums of one tile
    constexpr std::size_t tile_elements() const { return size_[2] * size_[3] * size_[4] * size_[5]; }

    /// @return true if the strides are those of a dense layout
    constexpr bool is_contiguous() const
    {
        std::size_t expected = 1;
        for (std::size_t i = 6; i-- > 0;)
        {
            if (stride_[i] != expected)
            {
                return false;
            }
            expected *= size_[i];
        }
        return true;
    }
};

/// Dest strides for tile-layout sizes s, as plain values: see dest_strides above.
constexpr std::array<std::size_t, 6> dest_strides(const std::array<std::size_t, 6>& s)
{
    std::array<std::size_t, 6> t {};
    t[5] = 1; // datums are contiguous along a row
    t[4] = dest_row_datums;
    t[3] = dest_row_datums * std::max<std::size_t>(s[4], min_dest_face_rows);
    t[2] = t[3] * s[3];
    t[1] = t[2] * s[2];
    t[0] = t[1] * s[1];
    return t;
}

/// Tile-layout sizes in format f, with the strides of `storage` (L1: dense; Dest: dest_strides), offset 0, as plain values.
constexpr TileLayout laid_out(const std::array<std::size_t, 6>& sizes, DataFormat f, Storage storage)
{
    std::array<std::size_t, 6> dense {};
    std::size_t stride = 1;
    for (std::size_t i = 6; i-- > 0;)
    {
        dense[i] = stride;
        stride *= sizes[i];
    }
    return TileLayout(0, sizes, storage == Storage::Dest ? dest_strides(sizes) : dense, f, storage);
}

namespace detail
{

template <class Spec>
constexpr TileLayout layout_from()
{
    using Shape = stdx::remove_cvref_t<decltype(std::declval<Spec>().shape())>;
    static_assert(Shape::rank == 6, "the Resolver takes tile-layout specs: six dimensions");
    static_assert(Shape::is_static, "the Resolver needs every size and stride at compile time");
    const Spec spec {};
    return TileLayout(
        spec.shape().offset(), values<6>(spec.shape().sizes()), values<6>(spec.shape().strides()), Spec::format(), Spec::storage());
}

} // namespace detail

/// The plain values of spec type Spec, as a compile-time object the Resolver can take
template <class Spec>
inline constexpr TileLayout layout_of = detail::layout_from<stdx::remove_cvref_t<Spec>>();

} // namespace llk
