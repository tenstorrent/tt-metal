# All programming example targets should link tt_stl
link_libraries(TT::STL)

add_compile_options("$<$<CXX_COMPILER_ID:Clang>:-Wno-c++11-narrowing>")
add_compile_options("$<$<CXX_COMPILER_ID:GNU>:-Wno-narrowing>")

add_compile_definitions(OVERRIDE_KERNEL_PREFIX="tt_metal/multi_device_microbench/")
include(${PROJECT_SOURCE_DIR}/cmake/helper_functions.cmake)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/multi_device_microbench")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/fabric_api_tests)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ccl_examples)

add_custom_target(
    multi_device_microbench
    DEPENDS
        profiler_examples
        ${MULTI_DEVICE_MICROBENCH_TEST_TARGETS}
)

install(
    DIRECTORY
        fabric_api_tests
    DESTINATION "${CMAKE_INSTALL_DATADIR}/tt-metalium/microbenchmarks"
    COMPONENT metalium-microbenchmarks
    REGEX
    "/kernels/"
        EXCLUDE
)

install(
    DIRECTORY
        fabric_api_tests
    DESTINATION "${CMAKE_INSTALL_DATADIR}/tenstorrent/microbenchmarks"
    COMPONENT metalium-microbenchmarks
    FILES_MATCHING
    REGEX
    "/kernels/"
)

install(
    DIRECTORY
        ccl_examples
    DESTINATION "${CMAKE_INSTALL_DATADIR}/tt-metalium/microbenchmarks"
    COMPONENT metalium-microbenchmarks
    REGEX
    "/kernels/"
        EXCLUDE
)

install(
    DIRECTORY
        ccl_examples
    DESTINATION "${CMAKE_INSTALL_DATADIR}/tenstorrent/microbenchmarks"
    COMPONENT metalium-microbenchmarks
    FILES_MATCHING
    REGEX
    "/kernels/"
)
