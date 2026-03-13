# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

add_executable(unit_tests_llk ${UNIT_TESTS_LLK_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_llk)
target_precompile_headers(unit_tests_llk REUSE_FROM metal_common_pch)

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
set_source_files_properties(
    test_matmul_fidelity_schedule_bh.cpp
    PROPERTIES
        INCLUDE_DIRECTORIES
            "${PROJECT_SOURCE_DIR}/tt_metal/third_party/tt_llk;${PROJECT_SOURCE_DIR}/tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib;${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/blackhole"
        SKIP_UNITY_BUILD_INCLUSION
            ON
)
set_source_files_properties(
    test_matmul_fidelity_schedule_wh.cpp
    PROPERTIES
        INCLUDE_DIRECTORIES
            "${PROJECT_SOURCE_DIR}/tt_metal/third_party/tt_llk;${PROJECT_SOURCE_DIR}/tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib;${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/wormhole;${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines"
        SKIP_UNITY_BUILD_INCLUSION
            ON
)
set_target_properties(
    unit_tests_llk
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
