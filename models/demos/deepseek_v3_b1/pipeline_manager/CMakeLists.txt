cmake_minimum_required(VERSION 3.22)
project(deepseek_v3_b1_pipeline_manager LANGUAGES CXX)

if(NOT TARGET TT::Metalium)
    set(TT_METALIUM_PREFIX "" CACHE PATH "Path to the TT-Metalium package config directory")
    if(TT_METALIUM_PREFIX)
        find_package(TT-Metalium REQUIRED CONFIG PATHS "${TT_METALIUM_PREFIX}" NO_DEFAULT_PATH)
    else()
        find_package(TT-Metalium REQUIRED)
    endif()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
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
