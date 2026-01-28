set(DISTRIBUTED_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/distributed.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/distributed_coordinate_translator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_command_queue_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fd_mesh_command_queue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sd_mesh_command_queue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_device_view.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_event.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_socket.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_socket_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_socket_serialization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_trace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_workload.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_workload_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pinned_memory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/system_mesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/system_mesh_translation_map.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/distributed_host_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multihost/distributed_context.cpp
)

# Include helper functions and generate headers from flatbuffer schemas
include(flatbuffers)

set(FLATBUFFER_SCHEMAS ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/socket_peer_descriptor.fbs)

# Check if distributed compute is enabled
if(NOT DEFINED ENABLE_DISTRIBUTED)
    set(ENABLE_DISTRIBUTED OFF)
endif()

# MPI is configured in parent scope (tt_metal/CMakeLists.txt) before this subdirectory
# Variables USE_MPI, TT_METAL_USING_ULFM, TT_METAL_MPI_LIB_DIR, and OpenMPI::MPI target
# are inherited from parent scope

# Add appropriate source files based on MPI configuration
if(NOT ENABLE_DISTRIBUTED)
    message(STATUS "Multihost compute with MPI disabled, using single host context.")
    list(APPEND DISTRIBUTED_SRC ${CMAKE_CURRENT_SOURCE_DIR}/multihost/single_host_context.cpp)
elseif(USE_MPI)
    list(APPEND DISTRIBUTED_SRC ${CMAKE_CURRENT_SOURCE_DIR}/multihost/mpi_distributed_context.cpp)
endif()

add_library(distributed OBJECT ${DISTRIBUTED_SRC})

# Prevent MPI library paths from being added to RPATH when linking distributed
# This is crucial for Fedora's brp-check-rpaths which requires $ORIGIN to be first in RPATH
set_target_properties(
    distributed
    PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH
            FALSE
)

# Enable unity builds for faster compilation
TT_ENABLE_UNITY_BUILD(distributed)

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

target_include_directories(distributed SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers)
