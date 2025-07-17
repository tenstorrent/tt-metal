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
    GIT_TAG 0.8.0
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

include(FetchContent)

FetchContent_Declare(enchantum 
GIT_REPOSITORY https://github.com/ZXShady/enchantum 
GIT_TAG 8ed8888e2ab3dd11313f76943ea6849f8b69a9f6)

FetchContent_MakeAvailable(enchantum)


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

# gersemi: off
CPMAddPackage(
    NAME msgpack
    GIT_REPOSITORY https://github.com/msgpack/msgpack-c.git
    GIT_TAG cpp-6.1.0
    PATCH_COMMAND
        patch --dry-run -p1 -R < ${CMAKE_CURRENT_LIST_DIR}/msgpack.patch || patch -p1 < ${CMAKE_CURRENT_LIST_DIR}/msgpack.patch
    OPTIONS
        "CMAKE_MESSAGE_LOG_LEVEL NOTICE"
        "MSGPACK_BUILD_EXAMPLES OFF"
        "MSGPACK_BUILD_TESTS OFF"
        "MSGPACK_BUILD_DOCS OFF"
        "MSGPACK_ENABLE_CXX ON"
        "MSGPACK_USE_BOOST OFF"
        "MSGPACK_BUILD_HEADER_ONLY ON"
        "MSGPACK_ENABLE_SHARED OFF"
        "MSGPACK_ENABLE_STATIC OFF"
        "MSGPACK_CXX20 ON"
        "MSGPACK_NO_BOOST ON"
)

CPMAddPackage(
    NAME tokenizers-cpp
    GITHUB_REPOSITORY mlc-ai/tokenizers-cpp
    GIT_TAG 5de6f656c06da557d4f0fb1ca611b16d6e9ff11d
    PATCH_COMMAND
        patch --dry-run -p1 -R < ${CMAKE_CURRENT_LIST_DIR}/tokenizers-cpp.patch || patch -p1 < ${CMAKE_CURRENT_LIST_DIR}/tokenizers-cpp.patch
    OPTIONS
        "CMAKE_MESSAGE_LOG_LEVEL NOTICE"
)
# gersemi: on

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
    VERSION 1.1.4
    OPTIONS
        "TT_LOGGER_INSTALL ON"
        "TT_LOGGER_BUILD_TESTING OFF"
)
