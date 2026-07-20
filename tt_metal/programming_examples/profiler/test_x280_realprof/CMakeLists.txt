# X280 REAL kernel_profiler capture: a real DeviceZoneScopedN workload (configurable cores/riscs/rate/
# prog-id) drained by the X280 (profzone.bin) + decoded on host into the 2-word + PROG/TIMER/SRC format.
# Needs internal Cluster + x280_driver headers, so it uses a custom CMakeLists like the other test_x280_*.
add_executable(test_x280_realprof ${CMAKE_CURRENT_SOURCE_DIR}/test_x280_realprof.cpp)

target_link_libraries(
    test_x280_realprof
    PRIVATE
        tt_metal
        pthread
        $<$<TARGET_EXISTS:TracyClient>:TracyClient>
)

target_include_directories(
    test_x280_realprof
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/api
        ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium
        ${PROJECT_BINARY_DIR}/tt_metal/api
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tools/x280_bm/include # prof_packet.h (shared 8 B format)
)
