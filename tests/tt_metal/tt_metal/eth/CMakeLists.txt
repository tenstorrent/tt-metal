set(UNIT_TESTS_ETH_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/test_basic_eth.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_buffer_movement_kernels.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_erisc_app_direct_send.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_ring_gather_kernels.cpp
)

add_executable(unit_tests_eth ${UNIT_TESTS_ETH_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_eth)

target_link_libraries(unit_tests_eth PUBLIC test_metal_common_libs)
target_include_directories(
    unit_tests_eth
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(
    unit_tests_eth
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
