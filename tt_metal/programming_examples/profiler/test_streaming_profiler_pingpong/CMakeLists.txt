# Worker-to-worker ping-pong through the streaming profiler: the acceptance test for the per-tile wall-clock offsets.
add_executable(test_streaming_profiler_pingpong ${CMAKE_CURRENT_SOURCE_DIR}/test_streaming_profiler_pingpong.cpp)

target_link_libraries(
    test_streaming_profiler_pingpong
    PRIVATE
        tt_metal
        pthread
)

target_include_directories(
    test_streaming_profiler_pingpong
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/api
        ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
        ${PROJECT_BINARY_DIR}/tt_metal/api
        ${CMAKE_CURRENT_SOURCE_DIR}
)
