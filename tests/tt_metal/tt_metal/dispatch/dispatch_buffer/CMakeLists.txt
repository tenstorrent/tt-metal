set(UNIT_TESTS_DISPATCH_BUFFER_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/test_EnqueueWriteBuffer_and_EnqueueReadBuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sub_device.cpp
)

set(UNIT_TESTS_DISPATCH_BUFFER_SRC "${UNIT_TESTS_DISPATCH_BUFFER_SRC}" PARENT_SCOPE)

add_executable(unit_tests_dispatch_buffer ${UNIT_TESTS_DISPATCH_BUFFER_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_dispatch_buffer)

target_link_libraries(unit_tests_dispatch_buffer PUBLIC test_metal_common_libs)
target_include_directories(
    unit_tests_dispatch_buffer
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/dispatch
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/dispatch/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(
    unit_tests_dispatch_buffer
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
