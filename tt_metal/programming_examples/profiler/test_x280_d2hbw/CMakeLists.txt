# X280 -> host D2H write-bandwidth test. Drives the low-level Cluster + sysmem.

add_executable(test_x280_d2hbw ${CMAKE_CURRENT_SOURCE_DIR}/test_x280_d2hbw.cpp)

target_link_libraries(
    test_x280_d2hbw
    PRIVATE
        tt_metal
        m
        pthread
)

target_include_directories(
    test_x280_d2hbw
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/api
        ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium
        ${PROJECT_BINARY_DIR}/tt_metal/api
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(
    test_x280_d2hbw
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/programming_examples/profiler
)
