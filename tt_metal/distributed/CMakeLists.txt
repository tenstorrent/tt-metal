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
    ${CMAKE_CURRENT_SOURCE_DIR}/mpi_distributed_context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/distributed_host_buffer.cpp
)

add_library(distributed OBJECT ${DISTRIBUTED_SRC})

set(ULFM_PREFIX /opt/openmpi-v5.0.7-ulfm)
set(ULFM_LIB ${ULFM_PREFIX}/lib/libmpi.so.40)

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

target_link_libraries(
    distributed
    PUBLIC
        common
    PRIVATE
        Metalium::Metal::Impl
        Metalium::Metal::LLRT
        TT::Metalium::HostDevCommon
        OpenMPI::MPI
)
