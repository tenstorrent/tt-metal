set(HAL_STRUCT_SOURCES
    ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/dev_msgs.h
    ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/fabric_telemetry_msgs.h
)

set(DEV_MSGS_CODEGEN_PY ${CMAKE_CURRENT_SOURCE_DIR}/codegen/codegen.py)
set(DEV_MSGS_CODEGEN_SH ${CMAKE_CURRENT_SOURCE_DIR}/codegen/codegen.sh)
set(HAL_GENERATED_STRUCT_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/generated/dev_msgs.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/generated/dev_msgs_impl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/generated/fabric_telemetry.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/generated/fabric_telemetry_impl.hpp
)

add_custom_command(
    OUTPUT
        ${HAL_GENERATED_STRUCT_HEADERS}
    COMMAND
        ${DEV_MSGS_CODEGEN_SH} ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/generated
    DEPENDS
        ${DEV_MSGS_CODEGEN_PY}
        ${DEV_MSGS_CODEGEN_SH}
        ${HAL_STRUCT_SOURCES}
)

add_custom_target(hal_generate_dev_msgs_header DEPENDS ${HAL_GENERATED_STRUCT_HEADERS})
if(TARGET all_generated_files)
    add_dependencies(all_generated_files hal_generate_dev_msgs_header)
endif()

add_subdirectory(tt-1xx)
add_subdirectory(tt-2xx)
