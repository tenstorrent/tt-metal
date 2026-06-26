# X280 relays device-profiler timestamps to host (close the loop). Low-level Cluster + sysmem.

add_executable(test_x280_profrelay ${CMAKE_CURRENT_SOURCE_DIR}/test_x280_profrelay.cpp)

target_link_libraries(
    test_x280_profrelay
    PRIVATE
        tt_metal
        m
        pthread
)

target_include_directories(
    test_x280_profrelay
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/api
        ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium
        ${PROJECT_BINARY_DIR}/tt_metal/api
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(
    test_x280_profrelay
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/programming_examples/profiler
)
