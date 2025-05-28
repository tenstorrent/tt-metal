# Smoke tests (fast, not necessarily thorough)
add_library(unit_tests_dispatch_smoke OBJECT)
add_library(TT::Metalium::Test::Dispatch::Smoke ALIAS unit_tests_dispatch_smoke)
TT_ENABLE_UNITY_BUILD(unit_tests_dispatch_smoke)
target_sources(
    unit_tests_dispatch_smoke
    PRIVATE
        dispatch_device/test_enqueue_read_write_core.cpp
        dispatch_event/test_EnqueueWaitForEvent.cpp
        dispatch_event/test_events.cpp
        dispatch_program/test_dispatch_stress.cpp
        dispatch_program/test_sub_device.cpp
        dispatch_util/test_device_command.cpp
        dispatch_util/test_dispatch_settings.cpp
)
target_include_directories(
    unit_tests_dispatch_smoke
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
target_link_libraries(unit_tests_dispatch_smoke PRIVATE test_metal_common_libs)

# Basic tests (too slow for smoke; will run less frequently)
add_library(unit_tests_dispatch_basic OBJECT)
add_library(TT::Metalium::Test::Dispatch::Basic ALIAS unit_tests_dispatch_basic)
TT_ENABLE_UNITY_BUILD(unit_tests_dispatch_basic)
target_sources(
    unit_tests_dispatch_basic
    PRIVATE
        dispatch_buffer/test_sub_device.cpp
        dispatch_program/test_dispatch.cpp
        dispatch_program/test_global_circular_buffers.cpp
        dispatch_program/test_program_reuse.cpp
        dispatch_trace/test_sub_device.cpp
)
target_include_directories(
    unit_tests_dispatch_basic
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
target_link_libraries(unit_tests_dispatch_basic PRIVATE test_metal_common_libs)

add_library(unit_tests_dispatch_slow OBJECT)
add_library(TT::Metalium::Test::Dispatch::Slow ALIAS unit_tests_dispatch_slow)
TT_ENABLE_UNITY_BUILD(unit_tests_dispatch_slow)
target_sources(
    unit_tests_dispatch_slow
    PRIVATE
        dispatch_buffer/test_EnqueueWriteBuffer_and_EnqueueReadBuffer.cpp # FIXME: Fix the heap-buffer-overflow in here and see if it's fast enough for Basic
        dispatch_program/test_EnqueueProgram.cpp
        dispatch_trace/test_EnqueueTrace.cpp
)
target_include_directories(
    unit_tests_dispatch_slow
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
target_link_libraries(unit_tests_dispatch_slow PRIVATE test_metal_common_libs)

# Target for all Dispatch tests regardless of duration
add_executable(unit_tests_dispatch)

set_target_properties(
    unit_tests_dispatch
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

target_link_libraries(
    unit_tests_dispatch
    PRIVATE
        TT::Metalium::Test::Dispatch::Smoke
        TT::Metalium::Test::Dispatch::Basic
        TT::Metalium::Test::Dispatch::Slow
)
