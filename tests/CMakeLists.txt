
enable_testing()
include(GoogleTest)
add_library(test_common_libs INTERFACE)
target_link_libraries(test_common_libs INTERFACE pthread gtest gtest_main)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_metal/tt_metal)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_eager) # this should go away and be replaced with link to ttnn
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ttnn/unit_tests/gtests)

set(TESTS_DEPENDS_LIST metal_tests eager_tests unit_tests_ttnn test_multi_device ttnn watcher_dump)
add_custom_target(tests DEPENDS ${TESTS_DEPENDS_LIST})
