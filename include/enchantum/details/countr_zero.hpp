#pragma once

#if (__cplusplus >= 202002L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L)) && __has_include(<bit>)
#include <bit>
namespace enchantum{
  namespace details 
  {
    using ::std::countr_zero;
  }
}
#else
namespace enchantum{
  namespace details 
  {
    template <typename T>
    constexpr int countr_zero(T x) {
    if (x == 0) 
      return sizeof(T) * 8;

    int count = 0;
    while ((x & 1) == 0) {
        x >>= 1;
        ++count;
    }
    return count;
  }
  }
}
#endif