# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# emule ASAN sanitizer test sources. This file is emule-team-owned (see CODEOWNERS
# for tests/tt_metal/tt_metal/api/emule/), and is include()'d from ../sources.cmake
# under if(TT_METAL_USE_EMULE). Add a new emule api test here — it needs only an
# emule-team review, not an infra review of the shared api/ CMake files.
#
# Paths are absolute (${CMAKE_CURRENT_LIST_DIR}) because this list is consumed in
# the parent api/ scope, where a bare relative path would resolve against api/.

# EXPECT_DEATH tests that assert on the emule ASAN panic + JIT kernels using
# emule-only defines/intrinsics (EMULE_SEM_BASE, __emule_local_l1_to_ptr). They
# compile/pass only under the emule backend, so they're gated out of the non-emule
# unit_tests_api binary. death_test_env.cpp forces "threadsafe" death-test style so
# they don't fork() a live fiber-pool process. Compiled into unit_tests_api.
list(
    APPEND
    UNIT_TESTS_API_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/death_test_env.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_alignment_writes.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_cb_leak.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_cb_pages.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_host_alignment.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_metadata_size.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_noc_without_barrier.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_padded_write.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_semaphore_write.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_tensor_bad_access.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_valid_mem_wrong_alloc.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_write_beyond_res_pages.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_write_outside_tensor.cpp
)

# Sources for the standalone per-fiber sanitizer-state isolation binary
# (unit_tests_fiber_asan); the target is defined in ../CMakeLists.txt.
set(UNIT_TESTS_FIBER_ASAN_SOURCES ${CMAKE_CURRENT_LIST_DIR}/test_fiber_sanitizer_isolation.cpp)
