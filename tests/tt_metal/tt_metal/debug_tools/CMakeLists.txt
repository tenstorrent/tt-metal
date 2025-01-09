set(UNIT_TESTS_DEBUG_TOOLS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_eth_cores.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_invalid_print_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_mute_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_mute_print_server.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_print_buffering.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_print_all_harts.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_print_before_finish.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_print_hanging.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_print_prepend_device_core_risc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_print_tensix_dest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_print_tiles.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_raise_wait.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/watcher/test_assert.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/watcher/test_link_training.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/watcher/test_noc_sanitize_delays.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/watcher/test_noc_sanitize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/watcher/test_pause.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/watcher/test_ringbuf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/watcher/test_waypoint.cpp
)

add_executable(unit_tests_debug_tools ${UNIT_TESTS_DEBUG_TOOLS_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_debug_tools)

target_link_libraries(unit_tests_debug_tools PRIVATE test_metal_common_libs)
target_include_directories(
    unit_tests_debug_tools
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(
    unit_tests_debug_tools
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
