
add_library(test_common_libs INTERFACE)
target_link_libraries(test_common_libs INTERFACE stdc++fs gtest gtest_main)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_metal/tt_metal)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_eager)

add_custom_target(tests DEPENDS metal_tests eager_tests)
