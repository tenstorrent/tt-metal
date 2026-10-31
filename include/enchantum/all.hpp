#pragma once
#include "type_name.hpp"
#include "details/optional.hpp"
#include "details/string.hpp"
#include "details/string_view.hpp"
#include "details/shared.hpp"
#include "details/format_util.hpp"
#include "generators.hpp"
#include "algorithms.hpp"
#include "array.hpp"
#include "bitset.hpp"
#include "bitflags.hpp"
#include "bitwise_operators.hpp"
#include "enchantum.hpp"
#include "entries.hpp"
#include "iostream.hpp"
#include "next_value.hpp"
#include "bitset.hpp"

#if __has_include(<fmt/format.h>)
  #include "fmt_format.hpp"
#elif (__cplusplus >= 202002 || (defined(_MSVC_LANG) && _MSVC_LANG >= 202002)) && __has_include(<format>)
  #include "std_format.hpp"
#endif