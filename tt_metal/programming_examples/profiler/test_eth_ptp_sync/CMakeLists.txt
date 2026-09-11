# Ethernet PTP hardware timestamp probe: the wall-clock sync round trip with the 1588 MAC/RX stamps recorded
# alongside. Needs internal headers, so it is built outside CREATE_PGM_EXAMPLES_EXE.
add_executable(test_eth_ptp_sync ${CMAKE_CURRENT_SOURCE_DIR}/test_eth_ptp_sync.cpp)

target_link_libraries(
    test_eth_ptp_sync
    PRIVATE
        tt_metal
        pthread
)

target_include_directories(
    test_eth_ptp_sync
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
