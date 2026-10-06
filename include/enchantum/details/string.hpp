#pragma once


#ifdef ENCHANTUM_CONFIG_FILE
  #include ENCHANTUM_CONFIG_FILE
#endif

#ifndef ENCHANTUM_ALIAS_STRING
  #include <string>
#endif


namespace enchantum {
#ifdef ENCHANTUM_ALIAS_STRING
ENCHANTUM_ALIAS_STRING;
#else
using ::std::string;
#endif

} // namespace enchantum