set(UNIT_TESTS_DISPATCH_BUFFER_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/test_EnqueueWriteBuffer_and_EnqueueReadBuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sub_device.cpp
)

add_library(unit_tests_dispatch_buffer_o STATIC ${UNIT_TESTS_DISPATCH_BUFFER_SRC})

target_link_libraries(unit_tests_dispatch_buffer_o PRIVATE test_metal_common_libs)

target_include_directories(
    unit_tests_dispatch_buffer_o
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/dispatch
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/dispatch/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)

add_executable(unit_tests_dispatch_buffer $<TARGET_OBJECTS:unit_tests_dispatch_buffer_o>)

target_link_libraries(unit_tests_dispatch_buffer PRIVATE test_metal_common_libs)

set_target_properties(
    unit_tests_dispatch_buffer
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

TT_ENABLE_UNITY_BUILD(unit_tests_dispatch_buffer)
