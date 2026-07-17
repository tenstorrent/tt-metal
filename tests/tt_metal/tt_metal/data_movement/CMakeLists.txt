# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

add_subdirectory(noc_estimator)

add_executable(unit_tests_data_movement ${UNIT_TESTS_DATA_MOVEMENT_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_data_movement)
target_precompile_headers(unit_tests_data_movement REUSE_FROM metal_common_pch)

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

# Let dm_common.cpp see kEmuleAsanBuild=true so it registers its raw-L1 scratch
# extent with the emule OOB sanitizer (see dm_common.cpp / SANITIZER_CHECKS.md).
# PRIVATE on the metal lib, so the test target must opt in explicitly.
if(TT_METAL_USE_EMULE)
    target_compile_definitions(unit_tests_data_movement PRIVATE TT_METAL_USE_EMULE=1)
endif()
