# The 1588 link layer on real links, with no profiler pipeline: the acceptance test for eth_ptp_link.hpp.
add_executable(test_eth_ptp_link ${CMAKE_CURRENT_SOURCE_DIR}/test_eth_ptp_link.cpp)

target_link_libraries(
    test_eth_ptp_link
    PRIVATE
        tt_metal
        pthread
)

target_include_directories(
    test_eth_ptp_link
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/api
        ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
        ${PROJECT_SOURCE_DIR}/tt_metal/llrt
        ${PROJECT_BINARY_DIR}/tt_metal/api
        ${CMAKE_CURRENT_SOURCE_DIR}
)
