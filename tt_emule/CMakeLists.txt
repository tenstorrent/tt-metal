cmake_minimum_required(VERSION 3.16)

# With the real tt_metal library built (and EmulatedChip at UMD level),
# tests compile against real headers and link the real library.
# At runtime, TT_METAL_EMULE_MODE=1 activates emulated mode.
# No fake headers needed.

find_package(Threads REQUIRED)

############################################################
# Tests
############################################################
enable_testing()
include(GoogleTest)

############################################################
# Helper: add an emule test that compiles a real tt-metal
# test source against real headers + real library.
############################################################
function(add_emule_test TEST_NAME TEST_SOURCE)
    set(COPIED_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}.cpp")
    configure_file("${PROJECT_SOURCE_DIR}/${TEST_SOURCE}" "${COPIED_SOURCE}" COPYONLY)

    add_executable(${TEST_NAME} "${COPIED_SOURCE}")

    target_include_directories(
        ${TEST_NAME}
        PRIVATE
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
            ${PROJECT_SOURCE_DIR}/tests
            ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
            ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/api
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(
        ${TEST_NAME}
        PRIVATE
            Metalium::Metal
            GTest::gtest_main
            Threads::Threads
            fmt::fmt-header-only
    )

    set_target_properties(
        ${TEST_NAME}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${CMAKE_BINARY_DIR}/test/tt_emule
    )
endfunction()

############################################################
# Helper: add a host-only test that uses real tt-metal
# headers and links the real library.
############################################################
function(add_host_test TEST_NAME)
    set(SOURCES "")
    foreach(src ${ARGN})
        string(REPLACE "/" "_" FLAT_NAME "${src}")
        set(COPIED "${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}_${FLAT_NAME}")
        configure_file("${PROJECT_SOURCE_DIR}/${src}" "${COPIED}" COPYONLY)
        list(APPEND SOURCES "${COPIED}")
    endforeach()

    add_executable(${TEST_NAME} ${SOURCES})

    target_include_directories(
        ${TEST_NAME}
        PRIVATE
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
            ${PROJECT_SOURCE_DIR}/tests
            ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
            ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/api
            ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/api/core_coord
            ${CMAKE_CURRENT_BINARY_DIR}
    )
    target_link_libraries(
        ${TEST_NAME}
        PRIVATE
            Metalium::Metal
            GTest::gtest_main
    )
    set_target_properties(
        ${TEST_NAME}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${CMAKE_BINARY_DIR}/test/tt_emule
    )
endfunction()

############################################################
# Tier 1a: test_bit_utils (pure bit manipulation)
############################################################

set(BIT_UTILS_COPIED "${CMAKE_CURRENT_BINARY_DIR}/test_bit_utils.cpp")
configure_file("${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/api/test_bit_utils.cpp" "${BIT_UTILS_COPIED}" COPYONLY)
add_executable(test_bit_utils "${BIT_UTILS_COPIED}")
target_include_directories(test_bit_utils PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(test_bit_utils PRIVATE GTest::gtest_main)
set_target_properties(
    test_bit_utils
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${CMAKE_BINARY_DIR}/test/tt_emule
)
gtest_discover_tests(test_bit_utils)

############################################################
# Tier 1b: CoreRange tests (9 files)
############################################################

set(CORE_COORD_TEST_DIR tests/tt_metal/tt_metal/api/core_coord)
set(CORE_COORD_TESTS
    test_CoreRange_adjacent
    test_CoreRange_contains
    test_CoreRange_intersects
    test_CoreRange_merge
    test_CoreRange_iterator
    test_CoreRangeSet_construct
    test_CoreRangeSet_contains
    test_CoreRangeSet_intersects
    test_CoreRangeSet_merge
)

foreach(TEST_NAME ${CORE_COORD_TESTS})
    add_host_test(${TEST_NAME}
        ${CORE_COORD_TEST_DIR}/${TEST_NAME}.cpp
    )
    gtest_discover_tests(${TEST_NAME})
endforeach()

############################################################
# Tier 1c: test_host_buffer (host-only, HostBuffer RAII)
############################################################

add_host_test(test_host_buffer
    tests/tt_metal/tt_metal/api/test_host_buffer.cpp
)
target_link_libraries(test_host_buffer PRIVATE GTest::gmock)
gtest_discover_tests(test_host_buffer)

############################################################
# Tier 1d: test_tilize_untilize, test_blockfloat_common
############################################################

add_host_test(test_tilize_untilize
    tests/tt_metal/tt_metal/api/test_tilize_untilize.cpp
)
gtest_discover_tests(test_tilize_untilize)

add_host_test(test_blockfloat_common
    tests/tt_metal/tt_metal/api/test_blockfloat_common.cpp
)
gtest_discover_tests(test_blockfloat_common)

############################################################
# Tier 2: MeshDeviceFixture buffer tests
############################################################

add_emule_test(test_simple_dram_buffer
    tests/tt_metal/tt_metal/api/test_simple_dram_buffer.cpp
)
gtest_discover_tests(test_simple_dram_buffer)

add_emule_test(test_simple_l1_buffer
    tests/tt_metal/tt_metal/api/test_simple_l1_buffer.cpp
)
gtest_discover_tests(test_simple_l1_buffer)

############################################################
# Tier 3: ttnn tests (require ttnn library)
############################################################
add_subdirectory(ttnn_tests)
