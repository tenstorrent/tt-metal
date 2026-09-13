# Worker-to-worker ping-pong across fabric links through the streaming profiler: the cross-chip acceptance test for
# the d2d sync.
add_executable(
    test_streaming_profiler_pingpong_fabric
    ${CMAKE_CURRENT_SOURCE_DIR}/test_streaming_profiler_pingpong_fabric.cpp
)

target_link_libraries(
    test_streaming_profiler_pingpong_fabric
    PRIVATE
        tt_metal
        pthread
)

target_include_directories(
    test_streaming_profiler_pingpong_fabric
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
