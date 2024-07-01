
enable_testing()
include(GoogleTest)
add_library(test_common_libs INTERFACE)
target_link_libraries(test_common_libs INTERFACE pthread gtest gtest_main)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_metal/tt_metal)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_eager)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ttnn/unit_tests/gtests)

set(TESTS_DEPENDS_LIST metal_tests eager_tests unit_tests_ttnn ttnn watcher_dump)
if(WITH_PYTHON_BINDINGS)
    list(APPEND TESTS_DEPENDS_LIST tt_lib)
endif()
add_custom_target(tests DEPENDS ${TESTS_DEPENDS_LIST})
