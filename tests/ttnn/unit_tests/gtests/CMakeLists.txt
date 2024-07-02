
set(TTNN_UNIT_TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/test_add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_repeat_interleave.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_async_runtime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_multiprod_queue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reflect.cpp
)

add_executable(unit_tests_ttnn ${TTNN_UNIT_TESTS_SRC})

target_link_libraries(unit_tests_ttnn PUBLIC test_common_libs ttnn tt_metal)
target_include_directories(unit_tests_ttnn PRIVATE
    ${UMD_HOME}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tt_metal
    ${PROJECT_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(unit_tests_ttnn PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/ttnn)
