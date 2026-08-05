include(sources.cmake)

# Include helper functions and generate headers from flatbuffer schemas
include(flatbuffers)

set(FLATBUFFER_SCHEMAS
    ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/socket_peer_descriptor.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/hd_socket_descriptor.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/h2d_stream_service_descriptor.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/d2h_stream_service_descriptor.fbs
)

# Check if distributed compute is enabled
if(NOT DEFINED ENABLE_DISTRIBUTED)
    set(ENABLE_DISTRIBUTED OFF)
endif()

# Distributed builds require the configured ULFM MPI tree.
if(ENABLE_DISTRIBUTED AND EXISTS "${TT_ULFM_PREFIX}/lib/libmpi.so.40")
    message(STATUS "Using ULFM MPI from ${TT_ULFM_PREFIX}")
    list(APPEND DISTRIBUTED_SRC multihost/mpi_distributed_context.cpp)
    set(ULFM_LIB "${TT_ULFM_PREFIX}/lib/libmpi.so.40")
    set(USE_MPI TRUE)

    add_library(OpenMPI::MPI SHARED IMPORTED GLOBAL)

    set_target_properties(
        OpenMPI::MPI
        PROPERTIES
            EXCLUDE_FROM_ALL
                TRUE
    )

    set_target_properties(
        OpenMPI::MPI
        PROPERTIES
            IMPORTED_LOCATION
                ${ULFM_LIB}
            INTERFACE_INCLUDE_DIRECTORIES
                "${TT_ULFM_PREFIX}/include"
    )
elseif(ENABLE_DISTRIBUTED)
    message(
        FATAL_ERROR
        "ENABLE_DISTRIBUTED requires ULFM MPI at ${TT_ULFM_PREFIX}. Set TT_ULFM_PREFIX or disable distributed support."
    )
else()
    message(STATUS "Multihost compute with MPI disabled, using single host context.")
    set(USE_MPI FALSE)
    list(APPEND DISTRIBUTED_SRC multihost/single_host_context.cpp)
endif()

add_library(distributed OBJECT ${DISTRIBUTED_SRC})

# Enable unity builds for faster compilation
TT_ENABLE_UNITY_BUILD(distributed)
target_precompile_headers(distributed REUSE_FROM metal_common_pch)

foreach(FBS_FILE ${FLATBUFFER_SCHEMAS})
    GENERATE_FBS_HEADER(${FBS_FILE} TARGET distributed)
    target_sources(distributed PRIVATE ${FBS_GENERATED_HEADER_FILE})
endforeach()

target_link_libraries(
    distributed
    PUBLIC
        common
    PRIVATE
        Metalium::Metal::Impl
        Metalium::Metal::LLRT
        TT::Metalium::HostDevCommon
        FlatBuffers::FlatBuffers
)
add_dependencies(distributed metalium_GeneratedHeaders)

if(USE_MPI)
    target_link_libraries(distributed PRIVATE OpenMPI::MPI)
    target_compile_definitions(distributed PRIVATE OPEN_MPI="1")
endif()

if(TT_UMD_BUILD_SIMULATION)
    target_compile_definitions(distributed PRIVATE TT_UMD_BUILD_SIMULATION)
endif()

if(TT_METAL_USE_EMULE)
    # The slow-dispatch mesh command queue routes per-device dispatch through the emule
    # register/run split (emulated_program_runner.hpp) — gated on this define.
    target_compile_definitions(distributed PRIVATE TT_METAL_USE_EMULE=1)
endif()

target_include_directories(distributed SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers)
