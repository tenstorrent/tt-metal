set(UNIT_TESTS_LLK_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/test_broadcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_copy_block_matmul_partials.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cumsum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dropout_sfpu_compute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_golden_impls.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_pack_rows.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reconfig.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sfpu_compute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_single_core_binary_compute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_single_core_matmul_compute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_stochastic_rounding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_single_core_matmul_int8.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_transpose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_unary_broadcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_untilize_tilize.cpp
)

add_executable(unit_tests_llk ${UNIT_TESTS_LLK_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_llk)

target_link_libraries(unit_tests_llk PUBLIC test_metal_common_libs)
target_include_directories(
    unit_tests_llk
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(
    unit_tests_llk
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
