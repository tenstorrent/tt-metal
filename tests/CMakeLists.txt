
enable_testing()
include(GoogleTest)
add_library(test_common_libs INTERFACE)
target_link_libraries(test_common_libs INTERFACE pthread gtest gtest_main)


option(TT_METAL_BUILD_TESTS "Enables build of tt_metal tests" OFF)
if(TT_METAL_BUILD_TESTS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_metal/tt_metal)
endif(TT_METAL_BUILD_TESTS)

option(TTNN_BUILD_TESTS "Enables build of ttnn tests" OFF)
if(TTNN_BUILD_TESTS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_eager) # this should go away and be replaced with link to ttnn
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ttnn/unit_tests/gtests)
endif(TTNN_BUILD_TESTS)

