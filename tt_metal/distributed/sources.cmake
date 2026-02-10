# Source files for tt_metal distributed library
# Module owners should update this file when adding/removing/renaming source files

set(DISTRIBUTED_SRC
    distributed.cpp
    distributed_coordinate_translator.cpp
    mesh_buffer.cpp
    mesh_command_queue_base.cpp
    fd_mesh_command_queue.cpp
    sd_mesh_command_queue.cpp
    mesh_device.cpp
    mesh_device_view.cpp
    mesh_event.cpp
    mesh_socket.cpp
    mesh_socket_utils.cpp
    mesh_socket_serialization.cpp
    mesh_trace.cpp
    mesh_workload.cpp
    mesh_workload_utils.cpp
    pinned_memory.cpp
    system_mesh.cpp
    system_mesh_translation_map.cpp
    distributed_host_buffer.cpp
    multihost/distributed_context.cpp
    dispatch_context.cpp
    h2d_socket.cpp
    d2h_socket.cpp
)

# Check if distributed compute is enabled
if(NOT DEFINED ENABLE_DISTRIBUTED)
    set(ENABLE_DISTRIBUTED OFF)
endif()

set(ULFM_PREFIX /opt/openmpi-v5.0.7-ulfm)

# Check if distributed is enabled and ULFM MPI exists, otherwise use system MPI, or fall back to single host
if(ENABLE_DISTRIBUTED AND EXISTS ${ULFM_PREFIX}/lib/libmpi.so.40)
    message(STATUS "Using ULFM MPI from ${ULFM_PREFIX}")
    list(APPEND DISTRIBUTED_SRC multihost/mpi_distributed_context.cpp)
    set(ULFM_LIB ${ULFM_PREFIX}/lib/libmpi.so.40)
    set(USE_MPI TRUE)
elseif(ENABLE_DISTRIBUTED)
    # Try to find system MPI if distributed is enabled
    find_package(MPI QUIET COMPONENTS C)
    if(MPI_FOUND)
        message(STATUS "ULFM MPI not found, using system MPI")
        set(USE_MPI TRUE)
        list(APPEND DISTRIBUTED_SRC multihost/mpi_distributed_context.cpp)
    else()
        message(
            FATAL_ERROR
            "ENABLE_DISTRIBUTED is ON but no MPI implementation found. Please install MPI or disable distributed support."
        )
    endif()
else()
    message(STATUS "Multihost compute with MPI disabled, using single host context.")
    set(USE_MPI FALSE)
    list(APPEND DISTRIBUTED_SRC multihost/single_host_context.cpp)
endif()
