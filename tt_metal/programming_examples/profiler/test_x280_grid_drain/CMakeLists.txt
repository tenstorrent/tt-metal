# X280 grid drain. Drives the low-level Cluster (internal headers), so it uses a
# custom target with the internal include roots rather than CREATE_PGM_EXAMPLES_EXE.

add_executable(test_x280_grid_drain ${CMAKE_CURRENT_SOURCE_DIR}/test_x280_grid_drain.cpp)

target_link_libraries(
    test_x280_grid_drain
    PRIVATE
        tt_metal
        m
        pthread
)

target_include_directories(
    test_x280_grid_drain
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/api
        ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium
        ${PROJECT_BINARY_DIR}/tt_metal/api
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(
    test_x280_grid_drain
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/programming_examples/profiler
)
