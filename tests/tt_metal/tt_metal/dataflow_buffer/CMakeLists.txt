set(UNIT_TESTS_DATAFLOW_BUFFER_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/host/dataflow_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dfb_test_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_local_dataflow_buffer.cpp
)

add_executable(unit_tests_dataflow_buffer ${UNIT_TESTS_DATAFLOW_BUFFER_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_dataflow_buffer)

target_link_libraries(
    unit_tests_dataflow_buffer
    PUBLIC
        test_metal_common_libs
        FlatBuffers::FlatBuffers
)

target_include_directories(
    unit_tests_dataflow_buffer
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)

set_target_properties(
    unit_tests_dataflow_buffer
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
