set(DISTRIBUTED_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/coordinate_translation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/distributed.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_command_queue_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fd_mesh_command_queue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sd_mesh_command_queue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_device_view.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_event.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_socket.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_socket_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_trace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_workload.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_workload_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/system_mesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/distributed_host_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multihost/distributed_context.cpp
)

# Check if distributed compute is enabled
if(NOT DEFINED ENABLE_DISTRIBUTED)
    set(ENABLE_DISTRIBUTED OFF)
endif()

set(ULFM_PREFIX /opt/openmpi-v5.0.7-ulfm)

# Check if distributed is enabled and ULFM MPI exists, otherwise use system MPI, or fall back to single host
if(ENABLE_DISTRIBUTED AND EXISTS ${ULFM_PREFIX}/lib/libmpi.so.40)
    message(STATUS "Using ULFM MPI from ${ULFM_PREFIX}")
    list(APPEND DISTRIBUTED_SRC ${CMAKE_CURRENT_SOURCE_DIR}/multihost/mpi_distributed_context.cpp)
    set(ULFM_LIB ${ULFM_PREFIX}/lib/libmpi.so.40)
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
                ${ULFM_PREFIX}/include
    )
elseif(ENABLE_DISTRIBUTED)
    # Try to find system MPI if distributed is enabled
    find_package(MPI QUIET COMPONENTS C)
    if(MPI_FOUND)
        message(STATUS "ULFM MPI not found, using system MPI")
        add_library(OpenMPI::MPI ALIAS MPI::MPI_C)
        set(USE_MPI TRUE)
        list(APPEND DISTRIBUTED_SRC ${CMAKE_CURRENT_SOURCE_DIR}/multihost/mpi_distributed_context.cpp)
    else()
        message(
            FATAL_ERROR
            "ENABLE_DISTRIBUTED is ON but no MPI implementation found. Please install MPI or disable distributed support."
        )
    endif()
else()
    message(STATUS "Multihost compute with MPI disabled, using single host context.")
    set(USE_MPI FALSE)
    list(APPEND DISTRIBUTED_SRC ${CMAKE_CURRENT_SOURCE_DIR}/multihost/single_host_context.cpp)
endif()

add_library(distributed OBJECT ${DISTRIBUTED_SRC})

target_link_libraries(
    distributed
    PUBLIC
        common
    PRIVATE
        Metalium::Metal::Impl
        Metalium::Metal::LLRT
        TT::Metalium::HostDevCommon
        Taskflow::Taskflow
)

if(USE_MPI)
    target_link_libraries(distributed PRIVATE OpenMPI::MPI)
    target_compile_definitions(distributed PRIVATE OPEN_MPI="1")
endif()
