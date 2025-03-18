# Shadow the cache variable with a blank value
# Placing a no-op .clang-tidy file at the root of CPM cache is insufficient as some projects may define
# their own .clang-tidy within themselves and still not be clean against it <cough>flatbuffers</cough>
set(CMAKE_C_CLANG_TIDY "")
set(CMAKE_CXX_CLANG_TIDY "")

############################################################################################################################
# Boost
############################################################################################################################

function(ensureboosttarget boostTarget)
    if(NOT TARGET Boost::${boostTarget})
        add_library(Boost::${boostTarget} INTERFACE IMPORTED GLOBAL)
        target_link_libraries(Boost::${boostTarget} INTERFACE Boost::headers)
        message(STATUS "Defined Boost::${boostTarget} as an INTERFACE target.")
    endif()
endfunction()

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
        "BOOST_INCLUDE_LIBRARIES core\\\;container\\\;smart_ptr\\\;interprocess\\\;asio\\\;lockfree"
    FIND_PACKAGE_ARGUMENTS "CONFIG REQUIRED"
)

ensureboosttarget(algorithm)
ensureboosttarget(asio)
ensureboosttarget(lockfree)
ensureboosttarget(core)
ensureboosttarget(container)
ensureboosttarget(smart_ptr)
ensureboosttarget(interprocess)

add_library(span INTERFACE)
target_link_libraries(span INTERFACE Boost::core)

add_library(small_vector INTERFACE)
target_link_libraries(small_vector INTERFACE Boost::container)

############################################################################################################################
# yaml-cpp
############################################################################################################################

CPMAddPackage(
    NAME yaml-cpp
    GITHUB_REPOSITORY jbeder/yaml-cpp
    GIT_TAG 0.8.0
    PATCHES
        yaml-cpp.patch
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
        "BUILD_SHARED_LIBS OFF"
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

CPMAddPackage(NAME fmt GITHUB_REPOSITORY fmtlib/fmt GIT_TAG 11.0.1 OPTIONS "CMAKE_MESSAGE_LOG_LEVEL NOTICE")

############################################################################################################################
# range-v3 : https://github.com/ericniebler/range-v3
############################################################################################################################

CPMAddPackage(
    NAME range-v3
    GITHUB_REPOSITORY ericniebler/range-v3
    GIT_TAG 0.12.0
    PATCHES
        range-v3.patch
    OPTIONS
        "CMAKE_BUILD_TYPE Release"
        "CMAKE_MESSAGE_LOG_LEVEL NOTICE"
)

############################################################################################################################
# pybind11 : https://github.com/pybind/pybind11
############################################################################################################################

CPMAddPackage(NAME pybind11 GITHUB_REPOSITORY pybind/pybind11 GIT_TAG v2.13.6 OPTIONS "CMAKE_MESSAGE_LOG_LEVEL NOTICE")

############################################################################################################################
# nlohmann/json : https://github.com/nlohmann/json
############################################################################################################################

CPMAddPackage(
    NAME nlohmann_json
    GITHUB_REPOSITORY nlohmann/json
    GIT_TAG v3.11.3
    OPTIONS
        "CMAKE_MESSAGE_LOG_LEVEL NOTICE"
)

############################################################################################################################
# xtensor : https://github.com/xtensor-stack/xtensor
############################################################################################################################

CPMAddPackage(
    NAME xtl
    GITHUB_REPOSITORY xtensor-stack/xtl
    GIT_TAG 0.7.7
    OPTIONS
        "CMAKE_MESSAGE_LOG_LEVEL NOTICE"
        "XTL_ENABLE_TESTS OFF"
)
CPMAddPackage(
    NAME xtensor
    GITHUB_REPOSITORY xtensor-stack/xtensor
    GIT_TAG 0.25.0
    OPTIONS
        "CMAKE_MESSAGE_LOG_LEVEL NOTICE"
        "XTENSOR_ENABLE_TESTS OFF"
)
CPMAddPackage(
    NAME xtensor-blas
    GITHUB_REPOSITORY xtensor-stack/xtensor-blas
    GIT_TAG 0.21.0
    OPTIONS
        "CMAKE_MESSAGE_LOG_LEVEL NOTICE"
        "XTENSOR_ENABLE_TESTS OFF"
)

############################################################################################################################
# benchmark : https://github.com/google/benchmark
############################################################################################################################

CPMAddPackage(
    NAME benchmark
    GITHUB_REPOSITORY google/benchmark
    GIT_TAG v1.9.1
    OPTIONS
        "CMAKE_MESSAGE_LOG_LEVEL NOTICE"
        "BENCHMARK_USE_LIBCXX ${ENABLE_LIBCXX}"
        "BENCHMARK_ENABLE_TESTING OFF"
)

if(benchmark_ADDED)
    set_target_properties(
        benchmark
        PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY
                "${CMAKE_BINARY_DIR}/lib"
    )
endif()

############################################################################################################################
# taskflow : https://github.com/taskflow/taskflow
############################################################################################################################

CPMAddPackage(
    NAME Taskflow
    GITHUB_REPOSITORY taskflow/taskflow
    GIT_TAG v3.7.0
    OPTIONS
        "CMAKE_MESSAGE_LOG_LEVEL NOTICE" # Taskflow's CMakeLists.txt is super noisy
)
if(Taskflow_ADDED AND NOT TARGET Taskflow::Taskflow)
    add_library(Taskflow::Taskflow ALIAS Taskflow)
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
