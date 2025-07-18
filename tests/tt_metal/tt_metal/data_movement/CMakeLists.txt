set(UNIT_TESTS_DATA_MOVEMENT_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/dm_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dram_unary/test_unary_dram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/one_to_all/test_one_to_all.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/one_to_all/test_multicast_schemes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/loopback/test_loopback.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reshard_hardcoded/test_reshard_hardcoded.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/deinterleave_hardcoded/test_deinterleave_hardcoded.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/conv_hardcoded/test_conv_hardcoded.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/one_to_one/test_one_to_one.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/one_from_one/test_one_from_one.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/one_to_and_from_one/test_one_to_and_from_one.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/one_from_all/test_one_from_all.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_to_all/test_all_to_all.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_from_all/test_all_from_all.cpp
)

add_executable(unit_tests_data_movement ${UNIT_TESTS_DATA_MOVEMENT_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_data_movement)

target_link_libraries(unit_tests_data_movement PUBLIC test_metal_common_libs)
target_include_directories(
    unit_tests_data_movement
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(
    unit_tests_data_movement
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
