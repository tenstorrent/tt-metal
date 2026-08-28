# Ethernet wall-clock sync: launches the bounded sender/receiver pair on the two ends of a link and solves
# for the device-to-device clock offset. Needs internal headers (eth core lookup, Cluster, HAL), so it is
# built the same way test_perf_debug_zones is.
add_executable(test_eth_wallclock_sync ${CMAKE_CURRENT_SOURCE_DIR}/test_eth_wallclock_sync.cpp)

target_link_libraries(
    test_eth_wallclock_sync
    PRIVATE
        tt_metal
        pthread
)

target_include_directories(
    test_eth_wallclock_sync
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/impl
        ${PROJECT_SOURCE_DIR}/tt_metal/api
        ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium
        ${PROJECT_BINARY_DIR}/tt_metal/api
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
        ${PROJECT_SOURCE_DIR}/tt_metal/llrt
        ${CMAKE_CURRENT_SOURCE_DIR}
)
