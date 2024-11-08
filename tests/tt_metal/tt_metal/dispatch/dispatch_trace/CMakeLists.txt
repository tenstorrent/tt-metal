set(UNIT_TESTS_DISPATCH_TRACE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/test_EnqueueTrace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sub_device.cpp
    CACHE INTERNAL
    ""
)

add_executable(unit_tests_dispatch_trace ${UNIT_TESTS_DISPATCH_TRACE_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_dispatch_trace)

target_link_libraries(unit_tests_dispatch_trace PUBLIC test_metal_common_libs)
target_include_directories(
    unit_tests_dispatch_trace
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/dispatch/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(
    unit_tests_dispatch_trace
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
