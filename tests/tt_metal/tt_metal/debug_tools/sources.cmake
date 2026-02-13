# Source files for tt_metal debug_tools tests
# Module owners should update this file when adding/removing/renaming source files

set(UNIT_TESTS_DEBUG_TOOLS_SRC
    device_print/test_compilation_failures.cpp
    device_print/test_format_updates.cpp
    dprint/test_eth_cores.cpp
    dprint/test_invalid_print_core.cpp
    dprint/test_mute_device.cpp
    dprint/test_mute_print_server.cpp
    dprint/test_print_buffering.cpp
    dprint/test_print_all_harts.cpp
    dprint/test_print_before_finish.cpp
    dprint/test_print_prepend_device_core_risc.cpp
    dprint/test_print_tensix_dest.cpp
    dprint/test_print_tile.cpp
    dprint/test_print_tiles_multiple.cpp
    dprint/test_print_config_register.cpp
    watcher/test_assert.cpp
    watcher/test_link_training.cpp
    watcher/test_noc_sanitize_delays.cpp
    watcher/test_sanitize.cpp
    watcher/test_pause.cpp
    watcher/test_ringbuf.cpp
    watcher/test_stack_usage.cpp
    watcher/test_waypoint.cpp
    watcher/test_runtime_args_known_garbage.cpp
)

set(UNIT_TESTS_INSPECTOR_SRC inspector/test_rpc_startup.cpp)
