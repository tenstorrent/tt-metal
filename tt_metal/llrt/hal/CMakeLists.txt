set(DEV_MSGS_CODEGEN_PY ${CMAKE_CURRENT_SOURCE_DIR}/codegen/codegen.py)
set(DEV_MSGS_CODEGEN_SH ${CMAKE_CURRENT_SOURCE_DIR}/codegen/codegen.sh)
set(HAL_GENERATED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generated)

# ── Per-file codegen commands ──────────────────────────────────────────────────
# Split into separate add_custom_command calls so that a change to dev_msgs.h
# only regenerates dev_msgs.hpp/dev_msgs_impl.hpp, not all six generated files.
# This prevents ~4000 ccache misses per unrelated header edit.

# 1) dev_msgs
add_custom_command(
    OUTPUT
        ${HAL_GENERATED_DIR}/dev_msgs.hpp
        ${HAL_GENERATED_DIR}/dev_msgs_impl.hpp
    COMMAND
        ${DEV_MSGS_CODEGEN_SH} ${PROJECT_SOURCE_DIR} ${HAL_GENERATED_DIR} 0
    DEPENDS
        ${DEV_MSGS_CODEGEN_PY}
        ${DEV_MSGS_CODEGEN_SH}
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/hostdev/dev_msgs.h
)

# 2) fabric_telemetry
add_custom_command(
    OUTPUT
        ${HAL_GENERATED_DIR}/fabric_telemetry.hpp
        ${HAL_GENERATED_DIR}/fabric_telemetry_impl.hpp
    COMMAND
        ${DEV_MSGS_CODEGEN_SH} ${PROJECT_SOURCE_DIR} ${HAL_GENERATED_DIR} 1
    DEPENDS
        ${DEV_MSGS_CODEGEN_PY}
        ${DEV_MSGS_CODEGEN_SH}
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/hostdev/fabric_telemetry_msgs.h
)

# 3) realtime_profiler_msgs
add_custom_command(
    OUTPUT
        ${HAL_GENERATED_DIR}/realtime_profiler_msgs.hpp
        ${HAL_GENERATED_DIR}/realtime_profiler_msgs_impl.hpp
    COMMAND
        ${DEV_MSGS_CODEGEN_SH} ${PROJECT_SOURCE_DIR} ${HAL_GENERATED_DIR} 2
    DEPENDS
        ${DEV_MSGS_CODEGEN_PY}
        ${DEV_MSGS_CODEGEN_SH}
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/hostdev/realtime_profiler_msgs.h
)

set(HAL_GENERATED_STRUCT_HEADERS
    ${HAL_GENERATED_DIR}/dev_msgs.hpp
    ${HAL_GENERATED_DIR}/dev_msgs_impl.hpp
    ${HAL_GENERATED_DIR}/fabric_telemetry.hpp
    ${HAL_GENERATED_DIR}/fabric_telemetry_impl.hpp
    ${HAL_GENERATED_DIR}/realtime_profiler_msgs.hpp
    ${HAL_GENERATED_DIR}/realtime_profiler_msgs_impl.hpp
)

add_custom_target(hal_generate_dev_msgs_header DEPENDS ${HAL_GENERATED_STRUCT_HEADERS})
if(TARGET all_generated_files)
    add_dependencies(all_generated_files hal_generate_dev_msgs_header)
endif()

add_subdirectory(tt-1xx)
add_subdirectory(tt-2xx)
