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

set(ULFM_PREFIX /opt/openmpi-v5.0.7-ulfm)

# MPI search list priorities:
# - conan-provided one
# - If not found, try ULFM MPI
# - Otherwise, use system-provided one

if(ENABLE_DISTRIBUTED)
    # Try to find system or conan-provided MPI if distributed is enabled
    find_package(MPI QUIET COMPONENTS C)
    # conan MPI package provides different target name compared to the apt-provided one,
    # so checking for `openmpi::openmpi` target here allows to prioritize conan-provided
    # package.

    if(TARGET openmpi::openmpi)
        add_library(OpenMPI::MPI ALIAS openmpi::openmpi)
        message(STATUS "Conan-provided MPI is found, using it.")
        set(USE_MPI TRUE)
    elseif(EXISTS ${ULFM_PREFIX}/lib/libmpi.so.40)
        message(STATUS "Using ULFM MPI from ${ULFM_PREFIX}")
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
    elseif(MPI_FOUND)
        message(STATUS "ULFM MPI not found, using system MPI")
        add_library(OpenMPI::MPI ALIAS MPI::MPI_C)
        set(USE_MPI TRUE)
    else()
        message(
            FATAL_ERROR
            "ENABLE_DISTRIBUTED is ON but no MPI implementation found. Please install MPI or disable distributed support."
        )
    endif()

    if(USE_MPI)
        list(APPEND DISTRIBUTED_SRC ${CMAKE_CURRENT_SOURCE_DIR}/multihost/mpi_distributed_context.cpp)
    endif()
else()
    message(STATUS "Multihost compute with MPI disabled, using single host context.")
    set(USE_MPI FALSE)
    list(APPEND DISTRIBUTED_SRC ${CMAKE_CURRENT_SOURCE_DIR}/multihost/single_host_context.cpp)
endif()

add_library(distributed OBJECT ${DISTRIBUTED_SRC})

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
