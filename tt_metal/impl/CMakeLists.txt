include(sources.cmake)

if(TT_UMD_BUILD_SIMULATION)
    list(APPEND IMPL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/buffers/simulator_direct_write.cpp)
endif()

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
set(JIT_COMPILE_RPC_CAPNP ${CMAKE_CURRENT_SOURCE_DIR}/jit_server/rpc.capnp)
set(CAPNPC_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/impl/)
file(MAKE_DIRECTORY ${CAPNPC_OUTPUT_DIR})

capnp_generate_cpp(INSPECTOR_RPC_CAPNP_SOURCES INSPECTOR_RPC_CAPNP_HEADER ${INSPECTOR_RPC_CAPNP})
list(APPEND IMPL_SRC ${INSPECTOR_RPC_CAPNP_SOURCES})
capnp_generate_cpp(JIT_COMPILE_RPC_CAPNP_SOURCES JIT_COMPILE_RPC_CAPNP_HEADER ${JIT_COMPILE_RPC_CAPNP})
list(APPEND IMPL_SRC ${JIT_COMPILE_RPC_CAPNP_SOURCES})

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
    ${JIT_COMPILE_RPC_CAPNP_SOURCES}
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
            ${JIT_COMPILE_RPC_CAPNP_HEADER}
    )
    add_dependencies(all_generated_files capnp_generated_files)
endif()

# Generate protobuf files for KvChunkAddressTable serialization
include(protobuf)
GENERATE_PROTO_FILES(
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/disaggregation/protobuf/kv_chunk_address_table.proto
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/internal/disaggregation
)
list(APPEND IMPL_SRC ${PROTO_SRCS})

# Serializer free-functions (export_to_protobuf_file / import_from_protobuf_file).
# protobuf::libprotobuf is already linked below and the generated pb.h include
# dir is already on this target, so this is the natural home for it.
list(APPEND IMPL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/internal/disaggregation/kv_chunk_address_table_protobuf.cpp)

# Usage requirements for compiling against impl's headers, without impl's object
# files. impl itself links this PUBLIC (it is the single definition of these
# paths); white-box tests that include impl/... headers link this alongside
# TT::Metalium instead of copying tt_metal's INCLUDE_DIRECTORIES property, which
# would flatten -isystem and other usage requirements into plain -I paths (#57055).
add_library(impl_headers INTERFACE)
add_library(TT::Metalium::Impl::Headers ALIAS impl_headers)
target_include_directories(
    impl_headers
    INTERFACE
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/debug
        ${CMAKE_CURRENT_SOURCE_DIR}/profiler
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
        ${CMAKE_CURRENT_BINARY_DIR}
        # impl headers include public API headers unscoped (e.g. "core_coord.hpp"
        # in kernels/kernel.hpp), mirroring the include_directories() convenience
        # in tt_metal/CMakeLists.txt. FIXME: scope those includes and drop this.
        ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium
)
target_include_directories(impl_headers SYSTEM INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/internal/disaggregation)
# impl headers include llrt headers unqualified (e.g. "tt_cluster.hpp"), and
# streaming_profiler headers include SIMDe. Linking the real targets preserves
# their usage requirements (e.g. SIMDe's -isystem classification).
target_link_libraries(
    impl_headers
    INTERFACE
        llrt_headers
        simde::simde
)

add_library(impl OBJECT ${IMPL_SRC})
add_library(Metalium::Metal::Impl ALIAS impl)

# Keep the reload extension from perturbing the established unity source batches.
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/reload_table.cpp
    PROPERTIES
        SKIP_UNITY_BUILD_INCLUSION
            TRUE
)

# Enable unity builds for faster compilation
TT_ENABLE_UNITY_BUILD(impl)
tt_reuse_precompile_headers(impl metal_common_pch)

target_link_libraries(
    impl
    PUBLIC
        impl_headers
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
        protobuf::libprotobuf
        ttexalens::elf
        Taskflow::Taskflow
)

# Public include directories live on impl_headers (linked PUBLIC above).
# Only requirements for compiling impl's own sources stay here.
target_include_directories(impl SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers)

# Set KJ_NO_EXCEPTIONS=0 to enable exceptions in Cap'n Proto code due to a detection bug with g++-12.
target_compile_options(impl PUBLIC -Wno-int-to-pointer-cast PRIVATE -DKJ_NO_EXCEPTIONS=0)

if(TT_UMD_BUILD_SIMULATION)
    target_compile_definitions(impl PRIVATE TT_UMD_BUILD_SIMULATION)
endif()

if(TT_METAL_USE_EMULE)
    get_target_property(_EMULE_JIT_DIR tt_emule_headers TT_EMULE_JIT_INCLUDE_DIR)
    get_target_property(_EMULE_INC_DIR tt_emule_headers TT_EMULE_INCLUDE_DIR)

    target_compile_definitions(
        impl
        PRIVATE
            TT_METAL_USE_EMULE=1
            TT_EMULE_JIT_INCLUDE_DIR="${_EMULE_JIT_DIR}"
            TT_EMULE_INCLUDE_DIR="${_EMULE_INC_DIR}"
            TT_EMULE_PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
            TT_EMULE_CXX_COMPILER="${CMAKE_CXX_COMPILER}"
            TT_EMULE_CXX_STANDARD="${CMAKE_CXX_STANDARD}"
    )
    target_link_libraries(
        impl
        PRIVATE
            tt-emule::headers
            dl
    )
endif()

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
