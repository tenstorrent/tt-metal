
set(ENV{CPM_SOURCE_CACHE} "${PROJECT_SOURCE_DIR}/.cpmcache")

############################################################################################################################
# Boost
############################################################################################################################

include(${PROJECT_SOURCE_DIR}/cmake/fetch_boost.cmake)

add_library(span INTERFACE)
if(CMAKE_CXX_STANDARD LESS 20)
  fetch_boost_library(core)
  target_link_libraries(span INTERFACE Boost::core)
endif()
fetch_boost_library(smart_ptr)

############################################################################################################################
# yaml-cpp
############################################################################################################################

CPMAddPackage(
  NAME yaml-cpp
  GITHUB_REPOSITORY jbeder/yaml-cpp
  GIT_TAG 0.8.0
  OPTIONS
    "YAML_CPP_BUILD_TESTS OFF"
    "YAML_CPP_BUILD_TOOLS OFF"
    "YAML_BUILD_SHARED_LIBS OFF"
)

if (yaml-cpp_ADDED)
    target_link_libraries(yaml-cpp PRIVATE stdlib)
    set_target_properties(yaml-cpp PROPERTIES DEBUG_POSTFIX "")
endif()

############################################################################################################################
# googletest
############################################################################################################################

CPMAddPackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG v1.13.0
  VERSION 1.13.0
  OPTIONS "INSTALL_GTEST OFF"
)

if (googletest_ADDED)
    target_compile_options(gtest PRIVATE -Wno-implicit-int-float-conversion)
    target_link_libraries(gtest PRIVATE stdlib)
    target_link_libraries(gtest_main PRIVATE stdlib)
endif()

############################################################################################################################
# boost-ext reflect : https://github.com/boost-ext/reflect
############################################################################################################################

CPMAddPackage(
  NAME reflect
  GITHUB_REPOSITORY boost-ext/reflect
  GIT_TAG v1.1.1
)

############################################################################################################################
# magic_enum : https://github.com/Neargye/magic_enum
############################################################################################################################

CPMAddPackage(
  NAME magic_enum
  GITHUB_REPOSITORY Neargye/magic_enum
  GIT_TAG v0.9.6
)

############################################################################################################################
# fmt : https://github.com/fmtlib/fmt
############################################################################################################################

CPMAddPackage(
  NAME fmt
  GITHUB_REPOSITORY fmtlib/fmt
  GIT_TAG 11.0.1
)
