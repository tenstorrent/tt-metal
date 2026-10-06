#pragma once
#include "enchantum.hpp"
#include <array>
#include <stdexcept>

namespace enchantum {

template<typename E, typename V, typename Container = std::array<V, count<E>>>
class array : public Container {
  static_assert(std::is_enum_v<E>);
public:
  using container_type = Container;
  using index_type     = E;
  using typename Container::const_reference;
  using typename Container::reference;

  using Container::at;
  using Container::operator[];

  [[nodiscard]] constexpr reference at(const E index)
  {
    if (const auto i = enchantum::enum_to_index(index))
      return operator[](*i);
    ENCHANTUM_THROW(std::out_of_range("enchantum::array::at index out of range"), index);
  }

  [[nodiscard]] constexpr const_reference at(const E index) const
  {
    if (const auto i = enchantum::enum_to_index(index))
      return operator[](*i);
    ENCHANTUM_THROW(std::out_of_range("enchantum::array::at: index out of range"), index);
  }

  [[nodiscard]] constexpr reference operator[](const E index) noexcept
  {
    return operator[](*enchantum::enum_to_index(index));
  }

  [[nodiscard]] constexpr const_reference operator[](const E index) const noexcept
  {
    return operator[](*enchantum::enum_to_index(index));
  }
};

} // namespace enchantum
