############################################################################################################################
# CPM
############################################################################################################################
include(${PROJECT_SOURCE_DIR}/cmake/CPM.cmake)

############################################################################################################################
# Boost
############################################################################################################################

CPMAddPackage(
    NAME Boost
    VERSION 1.86.0
    URL
        https://github.com/boostorg/boost/releases/download/boost-1.86.0/boost-1.86.0-cmake.tar.xz
        URL_HASH
        SHA256=2c5ec5edcdff47ff55e27ed9560b0a0b94b07bd07ed9928b476150e16b0efc57
    OPTIONS
        "BOOST_ENABLE_CMAKE ON"
        "BOOST_SKIP_INSTALL_RULES ON"
        "BUILD_SHARED_LIBS OFF"
        "BOOST_INCLUDE_LIBRARIES core\\\;container\\\;smart_ptr"
)

############################################################################################################################
# yaml-cpp
############################################################################################################################

CPMAddPackage(
    NAME yaml-cpp
    GITHUB_REPOSITORY jbeder/yaml-cpp
    GIT_TAG
        2f86d13775d119edbb69af52e5f566fd65c6953b # 0.8.0 + patches
    OPTIONS
        "YAML_CPP_BUILD_TESTS OFF"
        "YAML_CPP_BUILD_TOOLS OFF"
        "YAML_BUILD_SHARED_LIBS OFF"
)

############################################################################################################################
# googletest
############################################################################################################################

CPMAddPackage(
    NAME googletest
    GITHUB_REPOSITORY google/googletest
    GIT_TAG v1.13.0
    VERSION 1.13.0
    OPTIONS
        "INSTALL_GTEST OFF"
)

############################################################################################################################
# boost-ext reflect : https://github.com/boost-ext/reflect
############################################################################################################################

CPMAddPackage(NAME reflect GITHUB_REPOSITORY boost-ext/reflect GIT_TAG v1.2.6)

############################################################################################################################
# fmt : https://github.com/fmtlib/fmt
############################################################################################################################

CPMAddPackage(NAME fmt GITHUB_REPOSITORY fmtlib/fmt GIT_TAG 11.1.4)

############################################################################################################################
# magic_enum : https://github.com/Neargye/magic_enum
############################################################################################################################

CPMAddPackage(
    NAME enchantum
    GIT_REPOSITORY https://github.com/ZXShady/enchantum.git
    GIT_TAG 8ca5b0eb7e7ebe0252e5bc6915083f1dd1b8294e
    OPTIONS
        "CMAKE_MESSAGE_LOG_LEVEL NOTICE"
)

############################################################################################################################
# nlohmann/json : https://github.com/nlohmann/json
############################################################################################################################

CPMAddPackage(NAME nlohmann_json GITHUB_REPOSITORY nlohmann/json GIT_TAG v3.11.3 OPTIONS "JSON_BuildTests OFF")

CPMAddPackage(NAME xtl GITHUB_REPOSITORY xtensor-stack/xtl GIT_TAG 0.8.0 OPTIONS "XTL_ENABLE_TESTS OFF")

CPMAddPackage(NAME xtensor GITHUB_REPOSITORY xtensor-stack/xtensor GIT_TAG 0.26.0 OPTIONS "XTENSOR_ENABLE_TESTS OFF")

CPMAddPackage(
    NAME xtensor-blas
    GITHUB_REPOSITORY xtensor-stack/xtensor-blas
    GIT_TAG 0.22.0
    OPTIONS
        "XTENSOR_ENABLE_TESTS OFF"
)

include(${PROJECT_SOURCE_DIR}/cmake/fetch_cli11.cmake)

####################################################################################################################
# spdlog
####################################################################################################################

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME spdlog-dev)
CPMAddPackage(
    NAME spdlog
    GITHUB_REPOSITORY gabime/spdlog
    VERSION 1.15.2
    OPTIONS
        "CMAKE_MESSAGE_LOG_LEVEL NOTICE"
        "SPDLOG_FMT_EXTERNAL_HO ON"
        "SPDLOG_INSTALL ON"
)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME ${DEFAULT_COMPONENT_NAME})

####################################################################################################################
# tt-logger
####################################################################################################################
CPMAddPackage(
    NAME tt-logger
    GITHUB_REPOSITORY tenstorrent/tt-logger
    VERSION 1.1.7
    OPTIONS
        "TT_LOGGER_INSTALL ON"
        "TT_LOGGER_BUILD_TESTING OFF"
)
####################################################################################################################
# nanobind
####################################################################################################################
find_package(
    Python
    COMPONENTS
        Development
        Development.Module
        Interpreter
    REQUIRED
)
CPMAddPackage(
    NAME nanobind
    GITHUB_REPOSITORY wjakob/nanobind
    GIT_TAG
        c5a3a378aa61d104c82ca053cb1e367782cd3618 # v2.10.2
    OPTIONS
        "CMAKE_MESSAGE_LOG_LEVEL NOTICE"
        "NB_USE_SUBMODULE_DEPS ON"
)

####################################################################################################################
# simd-everywhere
####################################################################################################################
CPMAddPackage(NAME simd-everywhere GITHUB_REPOSITORY simd-everywhere/simde GIT_TAG v0.8.2)
if(simd-everywhere_ADDED)
    add_library(simde INTERFACE)
    add_library(simde::simde ALIAS simde)
    target_include_directories(simde SYSTEM INTERFACE ${simd-everywhere_SOURCE_DIR})
endif()

############################################################################################################################
# flatbuffers : https://github.com/google/flatbuffers
############################################################################################################################

CPMAddPackage(
    NAME flatbuffers
    GITHUB_REPOSITORY google/flatbuffers
    GIT_TAG v24.3.25
    OPTIONS
        "FLATBUFFERS_BUILD_FLATC ON"
        "FLATBUFFERS_BUILD_TESTS OFF"
        "FLATBUFFERS_SKIP_MONSTER_EXTRA ON"
        "FLATBUFFERS_STRICT_MODE ON"
)

if(flatbuffers_ADDED)
    # Few files including idl_gen_dart.cpp:175:18, Possibly related: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
    target_compile_options(flatc PRIVATE -Wno-restrict)
    target_compile_options(flatbuffers PRIVATE -Wno-restrict)
endif()
