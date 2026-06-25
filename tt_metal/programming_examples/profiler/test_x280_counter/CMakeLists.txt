# X280 bare-metal counter poll example.
#
# Unlike the other profiler examples (built via CREATE_PGM_EXAMPLES_EXE, which
# restricts includes to the public API), this one drives the low-level Cluster
# to reach the X280 / ARC tiles, so it needs the internal tt_metal include
# roots (impl/, llrt/). The Cluster + MetalContext symbols it calls are
# exported from libtt_metal.so. Transitive include roots (umd, tt_stl,
# hostdevcommon, tt-metalium) propagate from tt_metal's PUBLIC link deps.

add_executable(test_x280_counter ${CMAKE_CURRENT_SOURCE_DIR}/test_x280_counter.cpp)

target_link_libraries(
    test_x280_counter
    PRIVATE
        tt_metal
        m
        pthread
)

target_include_directories(
    test_x280_counter
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal # impl/, llrt/ internal headers
        ${PROJECT_SOURCE_DIR}/tt_metal/api # bare api includes
        ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium # bare "core_coord.hpp" etc.
        ${PROJECT_BINARY_DIR}/tt_metal/api # generated api headers
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(
    test_x280_counter
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/programming_examples/profiler
)
