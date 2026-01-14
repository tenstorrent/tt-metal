# Shadow the cache variable with a blank value
# Placing a no-op .clang-tidy file at the root of CPM cache is insufficient as some projects may define
# their own .clang-tidy within themselves and still not be clean against it <cough>flatbuffers</cough>
set(CMAKE_C_CLANG_TIDY "")
set(CMAKE_CXX_CLANG_TIDY "")
set(CMAKE_VERIFY_INTERFACE_HEADER_SETS FALSE)
set(DEFAULT_COMPONENT_NAME ${CMAKE_INSTALL_DEFAULT_COMPONENT_NAME})

add_compile_options(-Wno-unused-variable)

############################################################################################################################
# NUMA
############################################################################################################################
find_library(NUMA_LIBRARY NAMES numa)
if(NOT NUMA_LIBRARY)
    message(FATAL_ERROR "NUMA library not found. Please install it in your system.")
endif()

############################################################################################################################
# hwloc
############################################################################################################################
find_library(HWLOC_LIBRARY NAMES hwloc)
if(NOT HWLOC_LIBRARY)
    message(FATAL_ERROR "hwloc library not found. Please install it in your system.")
endif()

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

############################################################################################################################
# protobuf
############################################################################################################################

CPMAddPackage(
    NAME protobuf
    GITHUB_REPOSITORY protocolbuffers/protobuf
    GIT_TAG
        v21.12 # Using 21.12 to avoid using abseil and utf8_range
    OPTIONS
        "CMAKE_MESSAGE_LOG_LEVEL NOTICE"
        "protobuf_BUILD_TESTS OFF"
        "protobuf_INSTALL OFF"
        "BUILD_TESTING OFF"
        "protobuf_BUILD_EXAMPLES OFF"
        "BUILD_SHARED_LIBS OFF"
        "CMAKE_POSITION_INDEPENDENT_CODE ON"
    PATCHES
        protobuf_noreturn.patch
)

if(CMAKE_VERSION VERSION_LESS 3.25)
    # CMake 3.24 didn't yet have add_subdirectory(... SYSTEM), libprotobuf doesn't publish its interface dirs as SYSTEM
    # and headers in it run afoul certain compiler warnings.  That all combines to failed builds.  Work around it here.
    get_target_property(PROTOBUF_INCLUDES protobuf::libprotobuf INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(
        libprotobuf
        PROPERTIES
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                "${PROTOBUF_INCLUDES}"
    )
endif()

# Setting visibility to hidden to avoid symbol conflicts with other protobuf libraries when tt-metal package is used
set_target_properties(
    libprotobuf
    PROPERTIES
        CXX_VISIBILITY_PRESET
            "hidden"
        VISIBILITY_INLINES_HIDDEN
            TRUE
)

# Setting link and compile flags to avoid symbol conflicts with other protobuf libraries when tt-metal package is used.
# Ensures that tt-metal package only uses its own protobuf library and not the one from the system.
target_link_options(libprotobuf PRIVATE -Wl,-Bsymbolic)
target_compile_options(
    libprotobuf
    PRIVATE
        -fno-semantic-interposition
        -Wno-deprecated-declarations
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
    CUSTOM_CACHE_KEY "0_8_0_upstream_patched"
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

CPMAddPackage(NAME reflect GITHUB_REPOSITORY boost-ext/reflect GIT_TAG v1.2.6)
if(reflect_ADDED)
    add_library(reflect INTERFACE)
    add_library(Reflect::Reflect ALIAS reflect)
    target_include_directories(reflect SYSTEM INTERFACE "$<BUILD_INTERFACE:${reflect_SOURCE_DIR}>")

    target_sources(
        reflect
        INTERFACE
            FILE_SET api
            TYPE HEADERS
            BASE_DIRS ${reflect_SOURCE_DIR}
            FILES ${reflect_SOURCE_DIR}/reflect
    )
endif()

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
# fmt : https://github.com/fmtlib/fmt
############################################################################################################################

CPMAddPackage(NAME fmt GITHUB_REPOSITORY fmtlib/fmt GIT_TAG 11.1.4 OPTIONS "CMAKE_MESSAGE_LOG_LEVEL NOTICE")

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
    CUSTOM_CACHE_KEY "0_12_0_patched"
)

############################################################################################################################
# nanobind : https://github.com/wjakob/nanobind
############################################################################################################################

CPMAddPackage(
    NAME nanobind
    GITHUB_REPOSITORY wjakob/nanobind
    GIT_TAG
        c5a3a378aa61d104c82ca053cb1e367782cd3618 # v2.10.2
    OPTIONS
        "CMAKE_MESSAGE_LOG_LEVEL NOTICE"
        "NB_USE_SUBMODULE_DEPS ON"
)

############################################################################################################################
# nlohmann/json : https://github.com/nlohmann/json
############################################################################################################################

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME json-dev)
CPMAddPackage(
    NAME nlohmann_json
    GITHUB_REPOSITORY nlohmann/json
    GIT_TAG v3.11.3
    OPTIONS
        "CMAKE_MESSAGE_LOG_LEVEL NOTICE"
        "JSON_Install ON"
        "JSON_BuildTests OFF"
)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME ${DEFAULT_COMPONENT_NAME})

############################################################################################################################
# xtensor : https://github.com/xtensor-stack/xtensor
############################################################################################################################

CPMAddPackage(
    NAME xtl
    GITHUB_REPOSITORY xtensor-stack/xtl
    GIT_TAG 0.8.0
    PATCHES
        xtl.patch
    OPTIONS
        "CMAKE_MESSAGE_LOG_LEVEL NOTICE"
        "XTL_ENABLE_TESTS OFF"
    CUSTOM_CACHE_KEY "0_8_0_patched"
)
CPMAddPackage(
    NAME xtensor
    GITHUB_REPOSITORY xtensor-stack/xtensor
    GIT_TAG 0.26.0
    PATCHES
        xtensor.patch
    OPTIONS
        "CMAKE_MESSAGE_LOG_LEVEL NOTICE"
        "XTENSOR_ENABLE_TESTS OFF"
    CUSTOM_CACHE_KEY "0_26_0_patched"
)
CPMAddPackage(
    NAME xtensor-blas
    GITHUB_REPOSITORY xtensor-stack/xtensor-blas
    GIT_TAG 0.22.0
    PATCHES
        xtensor-blas.patch
    OPTIONS
        "CMAKE_MESSAGE_LOG_LEVEL NOTICE"
        "XTENSOR_ENABLE_TESTS OFF"
    CUSTOM_CACHE_KEY "0_22_0_patched"
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
        "TF_BUILD_TESTS OFF"
        "TF_BUILD_EXAMPLES OFF"
)
if(Taskflow_ADDED AND NOT TARGET Taskflow::Taskflow)
    target_compile_options(
        Taskflow
        INTERFACE
            -isystem
            ${Taskflow_SOURCE_DIR}
    )
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
        "FLATBUFFERS_STRICT_MODE OFF"
)

if(flatbuffers_ADDED)
    # Few files including idl_gen_dart.cpp:175:18, Possibly related: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
    target_compile_options(flatc PRIVATE -Wno-restrict)
    target_compile_options(flatbuffers PRIVATE -Wno-restrict)
endif()

############################################################################################################################
# simd-everywhere/simde : https://github.com/simd-everywhere/simde
############################################################################################################################

CPMAddPackage(NAME simd-everywhere GITHUB_REPOSITORY simd-everywhere/simde GIT_TAG v0.8.2)
if(simd-everywhere_ADDED)
    add_library(simde INTERFACE)
    add_library(simde::simde ALIAS simde)
    target_include_directories(simde SYSTEM INTERFACE ${simd-everywhere_SOURCE_DIR})
endif()

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
# Cap'n Proto : https://github.com/capnproto/capnproto
####################################################################################################################
CPMAddPackage(
    NAME capnproto
    GITHUB_REPOSITORY capnproto/capnproto
    GIT_TAG
        d135c9ca5e15219eaf131dfce1a41afdbaea9aab # v1.2.0
    OPTIONS
        "BUILD_TESTING OFF"
        "CAPNP_LITE OFF"
        "BUILD_SHARED_LIBS OFF"
        "CMAKE_CXX_EXTENSIONS ON"
    PATCHES
        capnproto_capture_this.patch
)
