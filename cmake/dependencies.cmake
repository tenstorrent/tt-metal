
set(ENV{CPM_SOURCE_CACHE} "${PROJECT_SOURCE_DIR}/.cpmcache")

############################################################################################################################
# Boost
############################################################################################################################

include(${PROJECT_SOURCE_DIR}/cmake/fetch_boost.cmake)

fetch_boost_library(interprocess)
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
# Packages needed for tt-metal simulator
#   NNG for IPC/TCP communication
#   Google Flatbuffers for serialization
#   libuv for process mgmt
############################################################################################################################
CPMAddPackage(
  NAME nanomsg
  GITHUB_REPOSITORY nanomsg/nng
  GIT_TAG v1.8.0
  OPTIONS
      "BUILD_SHARED_LIBS ON"
      "NNG_TESTS OFF"
      "NNG_TOOLS OFF"
)
CPMAddPackage(
  NAME flatbuffers
  GITHUB_REPOSITORY google/flatbuffers
  GIT_TAG v24.3.25
  OPTIONS
      "FLATBUFFERS_BUILD_FLATC OFF"
      "FLATBUFFERS_BUILD_TESTS OFF"
      "FLATBUFFERS_INSTALL OFF"
      "FLATBUFFERS_BUILD_FLATLIB OFF"
      "FLATBUFFERS_SKIP_MONSTER_EXTRA ON"
      "FLATBUFFERS_STRICT_MODE ON"
)
CPMAddPackage(
  NAME libuv
  GITHUB_REPOSITORY libuv/libuv
  GIT_TAG v1.48.0
  OPTIONS
      "LIBUV_BUILD_TESTS OFF"
)
