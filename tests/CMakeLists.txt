
enable_testing()
include(GoogleTest)
add_library(test_common_libs INTERFACE)
target_link_libraries(test_common_libs INTERFACE pthread gtest gtest_main)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_metal/tt_metal)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_eager) # this should go away and be replaced with link to ttnn
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ttnn/unit_tests/gtests)

set(TESTS_DEPENDS_LIST metal_tests eager_tests unit_tests_ttnn unit_tests_ttnn_ccl test_multi_device galaxy_unit_tests_ttnn ttnn watcher_dump umd_tests unit_test_mlir_interface_lib)
add_custom_target(tests DEPENDS ${TESTS_DEPENDS_LIST})
