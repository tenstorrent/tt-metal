#pragma once
#include "entries.hpp"
#ifdef __cpp_impl_three_way_comparison
  #include <compare>
#endif
#include "details/countr_zero.hpp"
#include <cstddef>
#include <cstdint>
#include <utility>

namespace enchantum {
namespace details {

  struct senitiel {};


  template<typename CRTP, std::ptrdiff_t Size>
  struct sized_iterator {
    static_assert(Size < INT16_MAX, "Too many enum entries");
  private:
    using IndexType = std::conditional_t<(Size <= INT8_MAX), std::int8_t, std::int16_t>;
  public:
    IndexType       index{};
    constexpr CRTP& operator+=(const std::ptrdiff_t offset) & noexcept
    {
      index += static_cast<IndexType>(offset);
      return static_cast<CRTP&>(*this);
    }
    constexpr CRTP& operator-=(const std::ptrdiff_t offset) & noexcept
    {
      index -= static_cast<IndexType>(offset);
      return static_cast<CRTP&>(*this);
    }

    constexpr CRTP& operator++() & noexcept
    {
      ++index;
      return static_cast<CRTP&>(*this);
    }
    constexpr CRTP& operator--() & noexcept
    {
      --index;
      return static_cast<CRTP&>(*this);
    }

    [[nodiscard]] constexpr CRTP operator++(int) & noexcept
    {
      auto copy = static_cast<CRTP&>(*this);
      ++*this;
      return copy;
    }
    [[nodiscard]] constexpr CRTP operator--(int) & noexcept
    {
      auto copy = static_cast<CRTP&>(*this);
      --*this;
      return copy;
    }

    [[nodiscard]] constexpr friend CRTP operator+(CRTP it, const std::ptrdiff_t offset) noexcept
    {
      it += offset;
      return it;
    }

    [[nodiscard]] constexpr friend CRTP operator+(const std::ptrdiff_t offset, CRTP it) noexcept
    {
      it += offset;
      return it;
    }

    [[nodiscard]] constexpr friend CRTP operator-(CRTP it, const std::ptrdiff_t offset) noexcept
    {
      it -= offset;
      return it;
    }

    [[nodiscard]] constexpr std::ptrdiff_t operator-(const sized_iterator that) const noexcept
    {
      return index - that.index;
    }

    [[nodiscard]] constexpr std::ptrdiff_t        operator-(senitiel) const noexcept { return index - Size; }
    [[nodiscard]] friend constexpr std::ptrdiff_t operator-(senitiel, sized_iterator it) noexcept
    {
      return Size - it.index;
    }

    [[nodiscard]] constexpr bool operator==(const sized_iterator that) const noexcept { return that.index == index; };
    [[nodiscard]] constexpr bool operator==(senitiel) const noexcept { return Size == index; }

#ifdef __cpp_impl_three_way_comparison
    [[nodiscard]] constexpr auto operator<=>(const sized_iterator that) const noexcept { return index <=> that.index; };
    [[nodiscard]] constexpr auto operator<=>(senitiel) const noexcept { return index <=> Size; }
#else

    [[nodiscard]] constexpr bool operator!=(const sized_iterator that) const noexcept { return that.index != index; };
    [[nodiscard]] constexpr bool operator!=(senitiel) const noexcept { return Size != index; }

    [[nodiscard]] friend constexpr bool operator==(senitiel, const sized_iterator it) noexcept
    {
      return Size == it.index;
    }


    [[nodiscard]] friend constexpr bool operator!=(senitiel, const sized_iterator it) noexcept
    {
      return Size != it.index;
    }


    [[nodiscard]] constexpr bool operator<(const sized_iterator that) const noexcept { return index < that.index; };
    [[nodiscard]] constexpr bool operator>(const sized_iterator that) const noexcept { return index > that.index; };
    [[nodiscard]] constexpr bool operator<=(const sized_iterator that) const noexcept { return index <= that.index; };
    [[nodiscard]] constexpr bool operator>=(const sized_iterator that) const noexcept { return index >= that.index; };

    [[nodiscard]] constexpr bool operator<(senitiel) const noexcept { return index < Size; };
    [[nodiscard]] constexpr bool operator>(senitiel) const noexcept { return index > Size; };
    [[nodiscard]] constexpr bool operator<=(senitiel) const noexcept { return index <= Size; };
    [[nodiscard]] constexpr bool operator>=(senitiel) const noexcept { return index >= Size; };

    [[nodiscard]] friend constexpr bool operator<(senitiel, const sized_iterator it) noexcept
    {
      return Size < it.index;
    };
    [[nodiscard]] friend constexpr bool operator>(senitiel, const sized_iterator it) noexcept
    {
      return Size > it.index;
    };
    [[nodiscard]] friend constexpr bool operator<=(senitiel, const sized_iterator it) noexcept
    {
      return Size <= it.index;
    };
    [[nodiscard]] friend constexpr bool operator>=(senitiel, const sized_iterator it) noexcept
    {
      return Size >= it.index;
    };

#endif
  };

  template<typename E, typename String = string_view, bool NullTerminated = true>
  struct names_generator_t {
    [[nodiscard]] static constexpr std::size_t size() noexcept { return count<E>; }

    struct iterator : sized_iterator<iterator, static_cast<std::ptrdiff_t>(size())> {
      using value_type = String;
      [[nodiscard]] constexpr String operator*() const noexcept
      {
        const auto* const p       = details::reflection_string_indices<E, NullTerminated>.data();
        const auto* const strings = details::reflection_data_string_storage<E, NullTerminated>.data();
        return String(strings + p[this->index], p[this->index + 1] - p[this->index] - NullTerminated);
      }

      [[nodiscard]] constexpr String operator[](const std::ptrdiff_t i) const noexcept { return *(*this + i); }
    };

    [[nodiscard]] static constexpr auto begin() { return iterator{}; }
    [[nodiscard]] static constexpr auto end() { return senitiel{}; }

    [[nodiscard]] constexpr auto operator[](const std::size_t i) const noexcept
    {
      return *(begin() + static_cast<std::ptrdiff_t>(i));
    }
  };

  template<typename E>
  struct values_generator_t {
    [[nodiscard]] static constexpr std::size_t size() noexcept { return count<E>; }

    struct iterator : sized_iterator<iterator, static_cast<std::ptrdiff_t>(size())> {
      using value_type = E;
      [[nodiscard]] constexpr E operator*() const noexcept
      {
        using T = std::underlying_type_t<E>;

        if constexpr (is_contiguous<E>) {
          return static_cast<E>(static_cast<T>(min<E>) + static_cast<T>(this->index));
        }
        else if constexpr (is_contiguous_bitflag<E>) {
          using UT                       = std::make_unsigned_t<T>;
          constexpr auto real_min_offset = details::countr_zero(static_cast<UT>(values<E>[has_zero_flag<E>]));

          if constexpr (has_zero_flag<E>)
            if (this->index == 0)
              return E{};
          return static_cast<E>(UT{1} << (real_min_offset + static_cast<UT>(this->index - has_zero_flag<E>)));
        }
        else {
          return values<E>[static_cast<std::size_t>(this->index)];
        }
      }
      [[nodiscard]] constexpr E operator[](const std::ptrdiff_t i) const noexcept { return *(*this + i); }
    };

    [[nodiscard]] static constexpr auto begin() { return iterator{}; }
    [[nodiscard]] static constexpr auto end() { return senitiel{}; }

    [[nodiscard]] constexpr auto operator[](const std::size_t i) const noexcept
    {
      return *(begin() + static_cast<std::ptrdiff_t>(i));
    }
  };

  template<typename E, typename Pair = std::pair<E, string_view>, bool NullTerminated = true>
  struct entries_generator_t {
    [[nodiscard]] static constexpr std::size_t size() noexcept { return count<E>; }

    struct iterator : sized_iterator<iterator, static_cast<std::ptrdiff_t>(size())> {
      using value_type = Pair;
      [[nodiscard]] constexpr Pair operator*() const noexcept
      {
        return Pair{
          values_generator_t<E>{}[static_cast<std::size_t>(this->index)],
          names_generator_t<E, string_view, NullTerminated>{}[static_cast<std::size_t>(this->index)],
        };
      }
      [[nodiscard]] constexpr Pair operator[](const std::ptrdiff_t i) const noexcept { return *(*this + i); }
    };

    [[nodiscard]] static constexpr auto begin() { return iterator{}; }
    [[nodiscard]] static constexpr auto end() { return senitiel{}; }

    [[nodiscard]] constexpr auto operator[](const std::size_t i) const noexcept
    {
      return *(begin() + static_cast<std::ptrdiff_t>(i));
    }
  };

} // namespace details

template<ENCHANTUM_DETAILS_ENUM_CONCEPT(E)>
inline constexpr details::values_generator_t<E> values_generator{};

#ifdef __cpp_concepts
template<Enum E, typename StringView = string_view, bool NullTerminated = true>
inline constexpr details::names_generator_t<E, StringView, NullTerminated> names_generator{};

template<Enum E, typename Pair = std::pair<E, string_view>, bool NullTerminated = true>
inline constexpr details::entries_generator_t<E, Pair, NullTerminated> entries_generator{};

#else
template<typename E, typename StringView = string_view, bool NullTerminated = true, std::enable_if_t<std::is_enum_v<E>, int> = 0>
inline constexpr details::names_generator_t<E, StringView, NullTerminated> names_generator{};

template<typename E, typename Pair = std::pair<E, string_view>, bool NullTerminated = true, std::enable_if_t<std::is_enum_v<E>, int> = 0>
inline constexpr details::entries_generator_t<E, Pair, NullTerminated> entries_generator{};

#endif

} // namespace enchantum