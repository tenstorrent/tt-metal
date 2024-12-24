# Shadow the cache variable with a blank value
# Placing a no-op .clang-tidy file at the root of CPM cache is insufficient as some projects may define
# their own .clang-tidy within themselves and still not be clean against it <cough>flatbuffers</cough>
set(CMAKE_C_CLANG_TIDY "")
set(CMAKE_CXX_CLANG_TIDY "")

############################################################################################################################
# Boost
############################################################################################################################

include(${PROJECT_SOURCE_DIR}/cmake/fetch_boost.cmake)

fetch_boost_library(core)
fetch_boost_library(smart_ptr)
fetch_boost_library(container)

add_library(span INTERFACE)
target_link_libraries(span INTERFACE Boost::core)

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

if(yaml-cpp_ADDED)
    set_target_properties(
        yaml-cpp
        PROPERTIES
            DEBUG_POSTFIX
                ""
    )
endif()

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

if(googletest_ADDED)
    target_compile_options(gtest PRIVATE -Wno-implicit-int-float-conversion)
endif()

############################################################################################################################
# boost-ext reflect : https://github.com/boost-ext/reflect
############################################################################################################################

CPMAddPackage(NAME reflect GITHUB_REPOSITORY boost-ext/reflect GIT_TAG v1.1.1)
if(reflect_ADDED)
    add_library(reflect INTERFACE)
    add_library(Reflect::Reflect ALIAS reflect)
    target_include_directories(reflect SYSTEM INTERFACE ${reflect_SOURCE_DIR})
endif()

############################################################################################################################
# magic_enum : https://github.com/Neargye/magic_enum
############################################################################################################################

CPMAddPackage(NAME magic_enum GITHUB_REPOSITORY Neargye/magic_enum GIT_TAG v0.9.7)

############################################################################################################################
# fmt : https://github.com/fmtlib/fmt
############################################################################################################################

CPMAddPackage(NAME fmt GITHUB_REPOSITORY fmtlib/fmt GIT_TAG 11.0.1)

############################################################################################################################
# range-v3 : https://github.com/ericniebler/range-v3
############################################################################################################################

CPMAddPackage(NAME range-v3 GITHUB_REPOSITORY ericniebler/range-v3 GIT_TAG 0.12.0)

############################################################################################################################
# pybind11 : https://github.com/pybind/pybind11
############################################################################################################################

CPMAddPackage(NAME pybind11 GITHUB_REPOSITORY pybind/pybind11 GIT_TAG b8f28551cc3a98ea9fbfc15c05b513c8f2d23e84)

############################################################################################################################
# nlohmann/json : https://github.com/nlohmann/json
############################################################################################################################

CPMAddPackage(NAME json GITHUB_REPOSITORY nlohmann/json GIT_TAG v3.9.1)

############################################################################################################################
# xtensor : https://github.com/xtensor-stack/xtensor
############################################################################################################################

CPMAddPackage(NAME xtl GITHUB_REPOSITORY xtensor-stack/xtl GIT_TAG 0.7.7 OPTIONS "XTL_ENABLE_TESTS OFF")
CPMAddPackage(NAME xtensor GITHUB_REPOSITORY xtensor-stack/xtensor GIT_TAG 0.25.0 OPTIONS "XTENSOR_ENABLE_TESTS OFF")
CPMAddPackage(
    NAME xtensor-blas
    GITHUB_REPOSITORY xtensor-stack/xtensor-blas
    GIT_TAG 0.21.0
    OPTIONS
        "XTENSOR_ENABLE_TESTS OFF"
)

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
