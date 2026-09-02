# perf-debug profiler workload: emits 10 differently-named DeviceZoneScopedN zones (increasing durations)
# on all 5 RISCs of a small grid. Run with TT_METAL_PERF_DEBUG_PROFILER=1 (+ TT_METAL_DEVICE_PROFILER=1) so
# the PerfDebugProfiler boots at MeshDevice bring-up and captures them. Uses only the public metal host API.
add_executable(test_perf_debug_zones ${CMAKE_CURRENT_SOURCE_DIR}/test_perf_debug_zones.cpp)

target_link_libraries(
    test_perf_debug_zones
    PRIVATE
        tt_metal
        pthread
)

target_include_directories(
    test_perf_debug_zones
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal
        # --clkprobe reaches into llrt/tt_cluster.hpp + the MeshDevice impl (a tile debug-register read is not
        # public API); those headers include their siblings unprefixed, hence tt_metal/impl on the path too.
        ${PROJECT_SOURCE_DIR}/tt_metal/impl
        ${PROJECT_SOURCE_DIR}/tt_metal/api
        ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
        ${PROJECT_BINARY_DIR}/tt_metal/api
        ${CMAKE_CURRENT_SOURCE_DIR}
)
