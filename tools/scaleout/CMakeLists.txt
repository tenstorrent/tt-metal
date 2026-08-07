project(scaleout_tools)

include(sources.cmake)

add_executable(run_cluster_validation)
target_sources(run_cluster_validation PRIVATE ${RUN_CLUSTER_VALIDATION_SRCS})

add_executable(run_fabric_manager)
target_sources(run_fabric_manager PRIVATE ${RUN_FABRIC_MANAGER_SRCS})

# Library target
add_library(scaleout_tools OBJECT)
add_library(TT::ScaleoutTools ALIAS scaleout_tools)

include(protobuf)

# GENERATE_PROTO_FILES feeds these into a custom build command, which needs absolute paths
# (a relative path would be resolved against the build dir, not this source dir).
list(TRANSFORM SCALEOUT_PROTO_SCHEMAS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

set(GENERATED_PROTO_SRCS)
foreach(PROTO_FILE ${SCALEOUT_PROTO_SCHEMAS})
    GENERATE_PROTO_FILES(${PROTO_FILE} OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    list(APPEND GENERATED_PROTO_SRCS ${PROTO_SRCS})
endforeach()

# Ensure protobufs are generated before compiling the executable
add_custom_target(scaleout_generated_protos DEPENDS ${GENERATED_PROTO_SRCS})
add_dependencies(run_cluster_validation scaleout_generated_protos)

# TODO: Should refactor into an API directory
target_sources(
    scaleout_tools
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${SCALEOUT_TOOLS_API_HEADERS}
    PRIVATE
        ${SCALEOUT_TOOLS_SRCS}
        ${GENERATED_PROTO_SRCS}
)

target_include_directories(scaleout_tools PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(scaleout_tools SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_include_directories(run_cluster_validation PRIVATE "$<TARGET_PROPERTY:TT::Metalium,INCLUDE_DIRECTORIES>")
target_include_directories(run_cluster_validation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(run_cluster_validation SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_include_directories(run_fabric_manager PRIVATE "$<TARGET_PROPERTY:TT::Metalium,INCLUDE_DIRECTORIES>")
target_include_directories(run_fabric_manager PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(run_fabric_manager SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(
    scaleout_tools
    PUBLIC
        TT::STL
        umd::tt-umd
    PRIVATE
        enchantum::enchantum
        tt-logger::tt-logger
        $<BUILD_INTERFACE:yaml-cpp::yaml-cpp>
        $<BUILD_INTERFACE:protobuf::libprotobuf>
)

# We are adding yaml-cpp include like this to avoid including
# scaleout_tools headers because that would cause protobuf issues.
target_include_directories(
    run_cluster_validation
    SYSTEM
    PRIVATE
        $<TARGET_PROPERTY:yaml-cpp::yaml-cpp,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(
    run_cluster_validation
    PRIVATE
        enchantum::enchantum
        protobuf::libprotobuf
        tt_metal
        cxxopts::cxxopts
)

# Separate binary to get functional coverage for CI. End user fabric manager tool is productized at https://github.com/tenstorrent/tt-fabric-manager
target_link_libraries(
    run_fabric_manager
    PRIVATE
        TT::ScaleoutTools
        enchantum::enchantum
        protobuf::libprotobuf
        tt_metal
)
target_compile_features(scaleout_tools PUBLIC cxx_std_20)
tt_reuse_precompile_headers(scaleout_tools TT::CommonPCH)

install(TARGETS scaleout_tools EXPORT Metalium LIBRARY COMPONENT metalium-runtime FILE_SET api COMPONENT metalium-dev)

add_executable(2d_big_mesh_cabling_gen)
target_sources(2d_big_mesh_cabling_gen PRIVATE ${SCALEOUT_2D_BIG_MESH_CABLING_GEN_SRCS})

set_target_properties(
    2d_big_mesh_cabling_gen
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${Metalium_BINARY_DIR}/tools/scaleout
)

add_dependencies(2d_big_mesh_cabling_gen scaleout_generated_protos)

target_include_directories(2d_big_mesh_cabling_gen SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(
    2d_big_mesh_cabling_gen
    PRIVATE
        scaleout_tools
        protobuf::libprotobuf
        cxxopts::cxxopts
)

add_executable(run_cabling_generator)
target_sources(run_cabling_generator PRIVATE ${RUN_CABLING_GENERATOR_SRCS})

set_target_properties(
    run_cabling_generator
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${Metalium_BINARY_DIR}/tools/scaleout
)

add_dependencies(run_cabling_generator scaleout_generated_protos)

target_include_directories(run_cabling_generator SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(
    run_cabling_generator
    PRIVATE
        scaleout_tools
        protobuf::libprotobuf
        cxxopts::cxxopts
)

add_executable(run_regen_descriptors)
target_sources(run_regen_descriptors PRIVATE ${RUN_REGEN_DESCRIPTORS_SRCS})

set_target_properties(
    run_regen_descriptors
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${Metalium_BINARY_DIR}/tools/scaleout
)

add_dependencies(run_regen_descriptors scaleout_generated_protos)

target_include_directories(run_regen_descriptors SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(
    run_regen_descriptors
    PRIVATE
        scaleout_tools
        protobuf::libprotobuf
        cxxopts::cxxopts
)

add_executable(generate_cluster_descriptor)
target_sources(generate_cluster_descriptor PRIVATE ${GENERATE_CLUSTER_DESCRIPTOR_SRCS})

set_target_properties(
    generate_cluster_descriptor
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${Metalium_BINARY_DIR}/tools/scaleout
)

add_dependencies(generate_cluster_descriptor scaleout_generated_protos)

target_include_directories(generate_cluster_descriptor SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(
    generate_cluster_descriptor
    PRIVATE
        scaleout_tools
        protobuf::libprotobuf
        cxxopts::cxxopts
)

# Generate MGD library
add_library(generate_mgd_lib OBJECT)
target_sources(generate_mgd_lib PRIVATE ${GENERATE_MGD_LIB_SRCS})

add_dependencies(
    generate_mgd_lib
    scaleout_generated_protos
    fabric_generated_protos
    fabric
)

target_include_directories(
    generate_mgd_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/generate_mgd
)
target_include_directories(
    generate_mgd_lib
    SYSTEM
    PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
        $<TARGET_PROPERTY:fabric,BINARY_DIR>
)

target_link_libraries(
    generate_mgd_lib
    PUBLIC
        scaleout_tools
        protobuf::libprotobuf
)

# Generate MGD CLI tool (standalone, links against fabric for protobuf implementation)
add_executable(generate_mgd)
target_sources(generate_mgd PRIVATE ${GENERATE_MGD_SRCS})

add_dependencies(
    generate_mgd
    scaleout_generated_protos
    fabric_generated_protos
    fabric
)

set_target_properties(
    generate_mgd
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${Metalium_BINARY_DIR}/tools/scaleout
)

target_include_directories(
    generate_mgd
    SYSTEM
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        $<TARGET_PROPERTY:fabric,BINARY_DIR>
)

target_sources(generate_mgd PRIVATE $<TARGET_OBJECTS:fabric>)

target_link_libraries(
    generate_mgd
    PRIVATE
        generate_mgd_lib
        scaleout_tools
        tt_metal
        protobuf::libprotobuf
        cxxopts::cxxopts
)

if(ENABLE_DISTRIBUTED)
    # MPI stress test: ring-based sendrecv packet validation across all ranks
    add_executable(run_mpi_stress_test)
    target_sources(run_mpi_stress_test PRIVATE ${RUN_MPI_STRESS_TEST_SRCS})

    set_target_properties(
        run_mpi_stress_test
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${Metalium_BINARY_DIR}/tools/scaleout
    )

    target_link_libraries(run_mpi_stress_test PRIVATE OpenMPI::MPI)

    # Generate rank bindings tool
    add_executable(generate_rank_bindings)
    target_sources(generate_rank_bindings PRIVATE ${GENERATE_RANK_BINDINGS_SRCS})

    set_target_properties(
        generate_rank_bindings
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${Metalium_BINARY_DIR}/tools/scaleout
            BUILD_RPATH
                "$ORIGIN/../../lib"
            INSTALL_RPATH
                "$ORIGIN/../lib"
    )

    add_dependencies(
        generate_rank_bindings
        scaleout_generated_protos
        fabric_generated_protos
        fabric
    )

    target_include_directories(
        generate_rank_bindings
        PRIVATE
            "$<TARGET_PROPERTY:TT::Metalium,INCLUDE_DIRECTORIES>"
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        SYSTEM
        PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}
            $<TARGET_PROPERTY:fabric,BINARY_DIR>
    )

    target_link_libraries(
        generate_rank_bindings
        PRIVATE
            scaleout_tools
            tt_metal
            OpenMPI::MPI
            protobuf::libprotobuf
            cxxopts::cxxopts
            $<BUILD_INTERFACE:yaml-cpp::yaml-cpp>
    )
    target_compile_definitions(generate_rank_bindings PRIVATE OPEN_MPI=1)

    # Install generate_rank_bindings executable
    if(TT_INSTALL)
        install(
            TARGETS
                generate_rank_bindings
            RUNTIME
                DESTINATION ${CMAKE_INSTALL_BINDIR}
                COMPONENT metalium-runtime
        )

        # PGDs used by find_and_load_physical_grouping_descriptor() in fabric_host_utils (default + arch/cluster-specific under
        # TT_METAL_HOME/tests/tt_metal/tt_fabric/physical_groupings/). Mirror that tree under the install datadir.
        file(
            GLOB _generate_rank_bindings_pgd_files
            CONFIGURE_DEPENDS
            "${CMAKE_SOURCE_DIR}/tests/tt_metal/tt_fabric/physical_groupings/*.textproto"
        )
        if(_generate_rank_bindings_pgd_files)
            install(
                FILES
                    ${_generate_rank_bindings_pgd_files}
                DESTINATION ${CMAKE_INSTALL_DATADIR}/tt-metalium/tests/tt_metal/tt_fabric/physical_groupings
                COMPONENT metalium-runtime
            )
        endif()
    endif()
endif()
