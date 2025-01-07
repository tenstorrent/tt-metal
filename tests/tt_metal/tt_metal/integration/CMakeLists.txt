set(UNIT_TESTS_INTEGRATION_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/test_autonomous_relay_streams.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_basic_pipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_flatten.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sfpu_compute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul/test_matmul_large_block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul/test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul/test_matmul_multi_core_multi_dram_inX_mcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul/test_matmul_multi_core_X_dram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul/test_matmul_single_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul/test_matmul_X_tile.cpp
)

add_executable(unit_tests_integration ${UNIT_TESTS_INTEGRATION_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_integration)

target_link_libraries(unit_tests_integration PUBLIC test_metal_common_libs)
target_include_directories(
    unit_tests_integration
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(
    unit_tests_integration
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
