# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

# Include helper functions and generate headers from flatbuffer schemas
include(flatbuffers)

set(FLATBUFFER_SCHEMAS
    ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/light_metal_binary.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/command.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/base_types.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/buffer_types.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/program_types.fbs
)

set(GENERATED_HEADERS)
foreach(FBS_FILE ${FLATBUFFER_SCHEMAS})
    GENERATE_FBS_HEADER(${FBS_FILE})
    list(APPEND GENERATED_HEADERS ${FBS_GENERATED_HEADER_FILE})
endforeach()

list(APPEND IMPL_SRC ${GENERATED_HEADERS})

# Generate Inspector Cap'n Proto headers and RPC server implementation
set(INSPECTOR_RPC_CAPNP ${CMAKE_CURRENT_SOURCE_DIR}/debug/inspector/rpc.capnp)
set(CAPNPC_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/impl/)
file(MAKE_DIRECTORY ${CAPNPC_OUTPUT_DIR})

capnp_generate_cpp(INSPECTOR_RPC_CAPNP_SOURCES INSPECTOR_RPC_CAPNP_HEADER ${INSPECTOR_RPC_CAPNP})
list(APPEND IMPL_SRC ${INSPECTOR_RPC_CAPNP_SOURCES})

# Generate Inspector RPC callbacks server implementation
set(INSPECTOR_RPC_GENERATED_HEADER ${CMAKE_CURRENT_BINARY_DIR}/impl/debug/inspector/rpc_server_generated.hpp)
set(INSPECTOR_RPC_GENERATED_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/impl/debug/inspector/rpc_server_generated.cpp)
add_custom_command(
    OUTPUT
        ${INSPECTOR_RPC_GENERATED_HEADER}
        ${INSPECTOR_RPC_GENERATED_SOURCE}
    COMMAND
        python3 ${CMAKE_CURRENT_SOURCE_DIR}/debug/inspector/generate_rpc_server.py ${INSPECTOR_RPC_CAPNP}
        ${INSPECTOR_RPC_GENERATED_HEADER} ${INSPECTOR_RPC_GENERATED_SOURCE}
    DEPENDS
        ${INSPECTOR_RPC_CAPNP}
        ${CMAKE_CURRENT_SOURCE_DIR}/debug/inspector/generate_rpc_server.py
    COMMENT "Generating RPC server implementation for ${INSPECTOR_RPC_CAPNP}"
)
list(APPEND IMPL_SRC ${INSPECTOR_RPC_GENERATED_SOURCE})

# Disable clang-tidy for generated Inspector files
set_source_files_properties(
    ${INSPECTOR_RPC_GENERATED_SOURCE}
    ${INSPECTOR_RPC_GENERATED_HEADER}
    ${INSPECTOR_RPC_CAPNP_SOURCES}
    PROPERTIES
        SKIP_LINTING
            TRUE
)

# Define dependency for CI small build for clang-tidy
if(TARGET all_generated_files)
    add_custom_target(
        capnp_generated_files
        DEPENDS
            ${INSPECTOR_RPC_CAPNP_HEADER}
            ${INSPECTOR_RPC_GENERATED_HEADER}
    )
    add_dependencies(all_generated_files capnp_generated_files)
endif()

add_library(impl OBJECT ${IMPL_SRC})
add_library(Metalium::Metal::Impl ALIAS impl)

# Enable unity builds for faster compilation
TT_ENABLE_UNITY_BUILD(impl)

target_link_libraries(
    impl
    PUBLIC
        common
        Tracy::TracyClient
    PRIVATE
        FlatBuffers::FlatBuffers
        range-v3::range-v3
        TT::Metalium::Common
        TT::Metalium::HostDevCommon
        Metalium::Metal::Hardware
        Metalium::Metal::LLRT
        capnp
        capnp-rpc
)

target_include_directories(
    impl
    PUBLIC
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/debug
        ${CMAKE_CURRENT_SOURCE_DIR}/profiler
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
        ${CMAKE_CURRENT_BINARY_DIR}
)
target_include_directories(impl SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers)

# Set KJ_NO_EXCEPTIONS=0 to enable exceptions in Cap'n Proto code due to a detection bug with g++-12.
target_compile_options(impl PUBLIC -Wno-int-to-pointer-cast PRIVATE -DKJ_NO_EXCEPTIONS=0)

# Generate Inspector RPC stub file
set(INSPECTOR_RPC_STUB_FILE ${PROJECT_SOURCE_DIR}/tools/triage/inspector_capnp.pyi)
add_custom_command(
    OUTPUT
        ${INSPECTOR_RPC_STUB_FILE}
    COMMAND
        python3 ${CMAKE_CURRENT_SOURCE_DIR}/debug/inspector/generate_rpc_stub.py ${INSPECTOR_RPC_CAPNP}
        ${INSPECTOR_RPC_STUB_FILE}
    DEPENDS
        ${INSPECTOR_RPC_CAPNP}
        ${CMAKE_CURRENT_SOURCE_DIR}/debug/inspector/generate_rpc_stub.py
    COMMENT "Generating RPC stub for ${INSPECTOR_RPC_CAPNP}"
)
add_custom_target(inspector_rpc_stub ALL DEPENDS ${INSPECTOR_RPC_STUB_FILE})

# Optionally, ensure impl depends on the stub file
add_dependencies(impl inspector_rpc_stub)
