# X280 bare-metal 4-hart grid poll (no DMA). Drives the low-level Cluster.

add_executable(test_x280_poll4 ${CMAKE_CURRENT_SOURCE_DIR}/test_x280_poll4.cpp)

target_link_libraries(
    test_x280_poll4
    PRIVATE
        tt_metal
        m
        pthread
)

target_include_directories(
    test_x280_poll4
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/api
        ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium
        ${PROJECT_BINARY_DIR}/tt_metal/api
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(
    test_x280_poll4
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/programming_examples/profiler
)
