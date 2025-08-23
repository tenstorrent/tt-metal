set(DEV_MSGS_H ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/dev_msgs.h)

set(DEV_MSGS_CODEGEN_PY ${CMAKE_CURRENT_SOURCE_DIR}/codegen/codegen.py)
set(DEV_MSGS_CODEGEN_SH ${CMAKE_CURRENT_SOURCE_DIR}/codegen/codegen.sh)
set(DEV_MSGS_GENERATED_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/generated_headers/dev_msgs.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/generated_headers/dev_msgs_impl.hpp
)

add_custom_command(
    OUTPUT
        ${DEV_MSGS_GENERATED_FILES}
    COMMAND
        ${DEV_MSGS_CODEGEN_SH} ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/generated_headers
    DEPENDS
        ${DEV_MSGS_CODEGEN_PY}
        ${DEV_MSGS_CODEGEN_SH}
        ${DEV_MSGS_H}
)

add_custom_target(hal_generate_dev_msgs_header DEPENDS ${DEV_MSGS_GENERATED_FILES})

add_subdirectory(tt-1xx)
