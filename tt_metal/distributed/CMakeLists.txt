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
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_trace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_workload.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_workload_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/system_mesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mpi_distributed_context.cpp
)

add_library(distributed OBJECT ${DISTRIBUTED_SRC})
add_dependencies(distributed openmpi_ulfm_build)

set(ULFM_PREFIX ${CMAKE_BINARY_DIR}/openmpi-ulfm-install)

add_library(OpenMPI::MPI SHARED IMPORTED)

# Path to the actual library file, not the directory
find_library(ULFM_MPI_LIB NAMES mpi PATHS ${ULFM_PREFIX}/lib NO_DEFAULT_PATH)

set_target_properties(
    OpenMPI::MPI
    PROPERTIES
        IMPORTED_LOCATION
            ${ULFM_MPI_LIB}
        INTERFACE_INCLUDE_DIRECTORIES
            ${ULFM_PREFIX}/include
        INTERFACE_LINK_LIBRARIES
            ${ULFM_PREFIX}/lib/libmpi.so
        INTERFACE_C_INCLUDE_DIRECTORIES
            ${ULFM_PREFIX}/include
        INTERFACE_CXX_INCLUDE_DIRECTORIES
            ${ULFM_PREFIX}/include
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
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

# print OpenMPI include directories
