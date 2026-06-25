# X280 NOC poll-rate probe. Like test_x280_counter, it drives the low-level
# Cluster (internal headers), so it uses a custom target with the internal
# include roots rather than the public-API-only CREATE_PGM_EXAMPLES_EXE macro.

add_executable(test_x280_poll_rate ${CMAKE_CURRENT_SOURCE_DIR}/test_x280_poll_rate.cpp)

target_link_libraries(
    test_x280_poll_rate
    PRIVATE
        tt_metal
        m
        pthread
)

target_include_directories(
    test_x280_poll_rate
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal # impl/, llrt/ internal headers
        ${PROJECT_SOURCE_DIR}/tt_metal/api # bare api includes
        ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium # bare "core_coord.hpp" etc.
        ${PROJECT_BINARY_DIR}/tt_metal/api # generated api headers
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(
    test_x280_poll_rate
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/programming_examples/profiler
)
