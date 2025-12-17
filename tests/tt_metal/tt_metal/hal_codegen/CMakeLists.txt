set(SOURCES codegen_test.cpp)

add_executable(hal_codegen_test ${SOURCES})
set(INTF_FILE ${CMAKE_CURRENT_BINARY_DIR}/test_msgs_intf.h)
set(IMPL_FILE ${CMAKE_CURRENT_BINARY_DIR}/test_msgs_impl.h)

set(SCHEMA ${CMAKE_CURRENT_SOURCE_DIR}/test_msgs.h)
set(GENERATOR ${PROJECT_SOURCE_DIR}/tt_metal/llrt/hal/codegen/codegen.py)

add_custom_command(
    OUTPUT
        ${INTF_FILE}
        ${IMPL_FILE}
    COMMAND
        ${GENERATOR} --driver_ns tt::tt_metal::hal_structs --driver_include_path "llrt/struct_view_driver.hpp" ${SCHEMA}
        ${INTF_FILE} ${IMPL_FILE}
    DEPENDS
        ${GENERATOR}
        ${SCHEMA}
)

add_custom_target(
    generate_test_msgs_header
    DEPENDS
        ${INTF_FILE}
        ${IMPL_FILE}
)

add_dependencies(hal_codegen_test generate_test_msgs_header)
if(TARGET all_generated_files)
    add_dependencies(all_generated_files generate_test_msgs_header)
endif()

target_include_directories(
    hal_codegen_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(
    hal_codegen_test
    PRIVATE
        enchantum::enchantum
        test_common_libs
)
