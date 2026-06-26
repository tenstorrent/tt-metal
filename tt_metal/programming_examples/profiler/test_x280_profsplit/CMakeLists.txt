# X280 reader/relay-hart-split profiler consumer (bench). Low-level Cluster + sysmem.

add_executable(test_x280_profsplit ${CMAKE_CURRENT_SOURCE_DIR}/test_x280_profsplit.cpp)

target_link_libraries(
    test_x280_profsplit
    PRIVATE
        tt_metal
        m
        pthread
)

target_include_directories(
    test_x280_profsplit
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/api
        ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium
        ${PROJECT_BINARY_DIR}/tt_metal/api
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(
    test_x280_profsplit
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/programming_examples/profiler
)
