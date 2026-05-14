include(sources.cmake)

add_executable(jit_compile_broker ${JIT_COMPILE_BROKER_SOURCES})
target_link_libraries(
    jit_compile_broker
    PRIVATE
        tt_metal
        capnp
        capnp-rpc
)

target_include_directories(
    jit_compile_broker
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_BINARY_DIR}/tt_metal/impl
)

set_target_properties(
    jit_compile_broker
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/tools
)
