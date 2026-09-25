# Broadcast arrivals on every Tensix core through the streaming profiler: a core-to-core check of the device-local timeline.
add_executable(test_streaming_profiler_multicast ${CMAKE_CURRENT_SOURCE_DIR}/test_streaming_profiler_multicast.cpp)

target_link_libraries(
    test_streaming_profiler_multicast
    PRIVATE
        tt_metal
        pthread
)

target_include_directories(
    test_streaming_profiler_multicast
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/api
        ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
        ${PROJECT_BINARY_DIR}/tt_metal/api
        ${CMAKE_CURRENT_SOURCE_DIR}
)
