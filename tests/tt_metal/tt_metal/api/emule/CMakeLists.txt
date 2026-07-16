# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Per-fiber ASAN sanitizer-state isolation test. Its own binary because it
# includes the emule kernel-side context header (jit_hw/internal/
# emule_thread_ctx.h) directly, which isn't on the shared api test lib's
# include path and shouldn't be dragged into its unity build / PCH.
add_executable(unit_tests_fiber_asan test_fiber_sanitizer_isolation.cpp)
target_include_directories(
    unit_tests_fiber_asan
    BEFORE
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${TT_EMULE_PATH}/include
)
target_link_libraries(unit_tests_fiber_asan PRIVATE test_metal_common_libs)
target_link_options(unit_tests_fiber_asan PRIVATE -rdynamic)
set_target_properties(
    unit_tests_fiber_asan
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

# Global gtest env for the emule ASAN death tests (the emule/*.cpp listed in
# ../sources.cmake, compiled into unit_tests_api): force "threadsafe" death-test
# style so each EXPECT_DEATH re-exec's instead of fork()ing a live fiber-pool
# process (which hangs the forked child). Compiled straight into the api test
# object lib so the whole binary — not just the regression path — is covered.
target_sources(unit_tests_api_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/death_test_env.cpp)
