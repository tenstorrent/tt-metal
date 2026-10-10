#pragma once

#include "bitflags.hpp"
#include "details/format_util.hpp"
#include "enchantum.hpp"
#include <fmt/format.h>

#ifdef __cpp_concepts
template<enchantum::Enum E>
struct fmt::formatter<E>
#else
template<typename E>
struct fmt::formatter<E, char, std::enable_if_t<std::is_enum_v<E>>>
#endif
: fmt::formatter<string_view> {
  template<typename FmtContext>
  constexpr auto format(const E e, FmtContext& ctx) const
  {
    return fmt::formatter<string_view>::format(enchantum::details::format(e), ctx);
  }
};