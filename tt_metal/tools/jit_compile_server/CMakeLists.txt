include(sources.cmake)

add_executable(jit_compile_server ${JIT_COMPILE_SERVER_SOURCES})
target_link_libraries(
    jit_compile_server
    PRIVATE
        tt_metal
        capnp
        capnp-rpc
        Taskflow::Taskflow
)

target_include_directories(
    jit_compile_server
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_BINARY_DIR}/tt_metal/impl
)

set_target_properties(
    jit_compile_server
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/tools
)

tt_reuse_precompile_headers(jit_compile_server TT::CommonPCH)
