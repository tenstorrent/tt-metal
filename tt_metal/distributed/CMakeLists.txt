# Include helper functions and generate headers from flatbuffer schemas
include(flatbuffers)

set(FLATBUFFER_SCHEMAS ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/socket_peer_descriptor.fbs)

# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

# Setup MPI library targets if needed
if(ENABLE_DISTRIBUTED AND EXISTS ${ULFM_PREFIX}/lib/libmpi.so.40)
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
elseif(ENABLE_DISTRIBUTED AND MPI_FOUND)
    add_library(OpenMPI::MPI ALIAS MPI::MPI_C)
endif()

add_library(distributed OBJECT ${DISTRIBUTED_SRC})

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
