# Tile clock offsets re-read at checkpoints across idle, load, di/dt and AICLK-sweep phases: whether they hold for a
# session.
add_executable(
    test_streaming_profiler_tile_offsets
    ${CMAKE_CURRENT_SOURCE_DIR}/test_streaming_profiler_tile_offsets.cpp
)

target_link_libraries(
    test_streaming_profiler_tile_offsets
    PRIVATE
        tt_metal
        pthread
)

target_include_directories(
    test_streaming_profiler_tile_offsets
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/impl
        ${PROJECT_SOURCE_DIR}/tt_metal/llrt
        ${PROJECT_SOURCE_DIR}/tt_metal/api
        ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
        ${PROJECT_BINARY_DIR}/tt_metal/api
        ${CMAKE_CURRENT_SOURCE_DIR}
)
