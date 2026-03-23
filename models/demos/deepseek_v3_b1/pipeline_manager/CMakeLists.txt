cmake_minimum_required(VERSION 3.22)
project(deepseek_v3_b1_pipeline_manager LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)

# ===========================================================================
#  Core library — scheduling logic only, no hardware/socket dependencies.
# ===========================================================================
add_library(deepseek_v3_b1_pipeline_manager_core STATIC pipeline_manager.cpp)
target_include_directories(deepseek_v3_b1_pipeline_manager_core PUBLIC ${REPO_ROOT})

# ===========================================================================
#  Full library + loopback harness — requires TT::Metalium for sockets.
# ===========================================================================
set(TT_METALIUM_PREFIX "" CACHE PATH "Path to the TT-Metalium package config directory")

if(NOT TARGET TT::Metalium)
    if(TT_METALIUM_PREFIX)
        find_package(TT-Metalium CONFIG PATHS "${TT_METALIUM_PREFIX}" NO_DEFAULT_PATH)
    else()
        find_package(TT-Metalium QUIET)
    endif()
endif()

if(TARGET TT::Metalium)
    get_filename_component(TT_METALIUM_BUILD_ROOT "${TT_METALIUM_PREFIX}/../../../" ABSOLUTE)

    add_library(
        deepseek_v3_b1_pipeline_manager
        STATIC
        pipeline_manager.cpp
        socket_connector.cpp
    )
    target_link_libraries(deepseek_v3_b1_pipeline_manager PUBLIC TT::Metalium)
    target_include_directories(
        deepseek_v3_b1_pipeline_manager
        PUBLIC
            ${REPO_ROOT}
            ${REPO_ROOT}/tt_metal/api
            ${REPO_ROOT}/tt_metal/api/tt-metalium
            ${REPO_ROOT}/tt_metal
            ${REPO_ROOT}/tt_metal/common
            ${REPO_ROOT}/tt_stl
            ${TT_METALIUM_BUILD_ROOT}/tt_metal/api
            ${TT_METALIUM_BUILD_ROOT}/tt_metal/impl
    )

    add_executable(deepseek_pipeline_loopback_harness tests/deepseek_pipeline_loopback_harness.cpp)
    target_link_libraries(deepseek_pipeline_loopback_harness PRIVATE deepseek_v3_b1_pipeline_manager)
endif()

# ===========================================================================
#  Tests — uses core library + MockPipeline, no hardware needed.
# ===========================================================================
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.13.0
    FIND_PACKAGE_ARGS
        NAMES
        GTest
)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(test_pipeline_manager tests/test_pipeline_manager.cpp)
target_link_libraries(
    test_pipeline_manager
    PRIVATE
        deepseek_v3_b1_pipeline_manager_core
        gtest
        gtest_main
)
