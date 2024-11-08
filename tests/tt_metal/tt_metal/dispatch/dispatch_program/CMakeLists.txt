set(UNIT_TESTS_DISPATCH_PROGRAM_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_EnqueueProgram.cpp
<<<<<<< HEAD:tests/tt_metal/tt_metal/dispatch_program/CMakeLists.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sub_device.cpp
=======
    CACHE INTERNAL
    ""
>>>>>>> #0: More test structuring:tests/tt_metal/tt_metal/dispatch/dispatch_program/CMakeLists.txt
)

add_executable(unit_tests_dispatch_program ${UNIT_TESTS_DISPATCH_PROGRAM_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_dispatch_program)

target_link_libraries(unit_tests_dispatch_program PUBLIC test_metal_common_libs)
target_include_directories(
    unit_tests_dispatch_program
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/dispatch/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(
    unit_tests_dispatch_program
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
