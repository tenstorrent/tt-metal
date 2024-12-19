add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/dispatch_buffer)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/dispatch_event)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/dispatch_program)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/dispatch_trace)

add_executable(
    unit_tests_dispatch
    $<TARGET_OBJECTS:unit_tests_dispatch_buffer_o>
    $<TARGET_OBJECTS:unit_tests_dispatch_event_o>
    $<TARGET_OBJECTS:unit_tests_dispatch_program_o>
    $<TARGET_OBJECTS:unit_tests_dispatch_trace_o>
)
TT_ENABLE_UNITY_BUILD(unit_tests_dispatch)

target_link_libraries(unit_tests_dispatch PUBLIC test_metal_common_libs)
target_include_directories(
    unit_tests_dispatch
    PRIVATE
        ${UMD_HOME}
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(
    unit_tests_dispatch
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
