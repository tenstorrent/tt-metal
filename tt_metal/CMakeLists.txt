add_compile_options("$<$<CXX_COMPILER_ID:Clang>:-Wunused-parameter>")

add_library(tt_metal)
add_library(TT::Metalium ALIAS tt_metal)
add_library(Metalium::Metal ALIAS tt_metal) # For backwards compatibility

set_target_properties(
    tt_metal
    PROPERTIES
        EXPORT_NAME
            Metalium
)

include(flatbuffers)
GENERATE_FBS_HEADER(
    api/tt-metalium/serialized_descriptors/mesh_coordinate.fbs
    TARGET TT::Metalium
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/api/tt-metalium/serialized_descriptors/
)
set(API_GENERATED_HEADERS ${FBS_GENERATED_HEADER_FILE})

add_custom_target(
    metalium_GeneratedHeaders
    DEPENDS
        ${API_GENERATED_HEADERS}
    COMMENT "Generating all FlatBuffer headers"
)

set(TT_METAL_PUBLIC_API
    api/tt-metalium/allocator.hpp
    api/tt-metalium/base_types.hpp
    api/tt-metalium/bfloat16.hpp
    api/tt-metalium/bfloat4.hpp
    api/tt-metalium/bfloat8.hpp
    api/tt-metalium/buffer.hpp
    api/tt-metalium/buffer_distribution_spec.hpp
    api/tt-metalium/buffer_page_mapping.hpp
    api/tt-metalium/buffer_types.hpp
    api/tt-metalium/circular_buffer.hpp
    api/tt-metalium/circular_buffer_config.hpp
    api/tt-metalium/circular_buffer_constants.h
    api/tt-metalium/cluster.hpp
    api/tt-metalium/constants.hpp
    api/tt-metalium/experimental/fabric/control_plane.hpp
    api/tt-metalium/core_coord.hpp
    api/tt-metalium/data_types.hpp
    api/tt-metalium/device.hpp
    api/tt-metalium/dispatch_core_common.hpp
    api/tt-metalium/distributed.hpp
    api/tt-metalium/distributed_context.hpp
    api/tt-metalium/distributed_host_buffer.hpp
    api/tt-metalium/experimental/fabric/edm_fabric_counters.hpp
    api/tt-metalium/event.hpp
    api/tt-metalium/experimental/inspector.hpp
    api/tt-metalium/experimental/profiler.hpp
    api/tt-metalium/experimental/fabric/fabric.hpp
    api/tt-metalium/experimental/fabric/fabric_edm_types.hpp
    api/tt-metalium/experimental/fabric/fabric_types.hpp
    api/tt-metalium/global_circular_buffer.hpp
    api/tt-metalium/global_semaphore.hpp
    api/tt-metalium/graph_tracking.hpp
    api/tt-metalium/hal.hpp
    api/tt-metalium/hal_types.hpp
    api/tt-metalium/host_api.hpp
    api/tt-metalium/host_buffer.hpp
    api/tt-metalium/kernel_types.hpp
    api/tt-metalium/experimental/lightmetal/lightmetal_binary.hpp
    api/tt-metalium/experimental/lightmetal/lightmetal_capture_utils.hpp
    api/tt-metalium/experimental/lightmetal/lightmetal_replay.hpp
    api/tt-metalium/experimental/lightmetal/lightmetal_api.hpp
    api/tt-metalium/math.hpp
    api/tt-metalium/maybe_remote.hpp
    api/tt-metalium/memory_pin.hpp
    api/tt-metalium/memory_reporter.hpp
    api/tt-metalium/mesh_buffer.hpp
    api/tt-metalium/mesh_command_queue.hpp
    api/tt-metalium/mesh_config.hpp
    api/tt-metalium/mesh_coord.hpp
    api/tt-metalium/mesh_device.hpp
    api/tt-metalium/mesh_device_view.hpp
    api/tt-metalium/mesh_event.hpp
    api/tt-metalium/experimental/fabric/mesh_graph.hpp
    api/tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp
    api/tt-metalium/experimental/sockets/mesh_socket.hpp
    api/tt-metalium/mesh_trace_id.hpp
    api/tt-metalium/mesh_workload.hpp
    api/tt-metalium/profiler_optional_metadata.hpp
    api/tt-metalium/profiler_types.hpp
    api/tt-metalium/program.hpp
    api/tt-metalium/program_cache.hpp
    api/tt-metalium/program_descriptors.hpp
    api/tt-metalium/queue_id.hpp
    api/tt-metalium/experimental/fabric/routing_table_generator.hpp
    api/tt-metalium/experimental/fabric/topology_mapper_utils.hpp
    api/tt-metalium/runtime_args_data.hpp
    api/tt-metalium/shape.hpp
    api/tt-metalium/shape2d.hpp
    api/tt-metalium/shape_base.hpp
    api/tt-metalium/sub_device.hpp
    api/tt-metalium/sub_device_types.hpp
    api/tt-metalium/system_mesh.hpp
    api/tt-metalium/tensor_accessor_args.hpp
    api/tt-metalium/tile.hpp
    api/tt-metalium/tilize_utils.hpp
    api/tt-metalium/tt_align.hpp
    api/tt-metalium/tt_backend_api_types.hpp
    api/tt-metalium/tt_metal.hpp
    api/tt-metalium/tt_metal_profiler.hpp
    api/tt-metalium/vector_aligned.hpp
    api/tt-metalium/work_split.hpp
    api/tt-metalium/experimental/fabric/fabric_telemetry.hpp
    api/tt-metalium/experimental/fabric/fabric_telemetry_reader.hpp
    api/tt-metalium/experimental/pinned_memory.hpp
)
target_sources(
    tt_metal
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/api ${CMAKE_CURRENT_BINARY_DIR}/api
        FILES ${TT_METAL_PUBLIC_API}
    PRIVATE
        tt_metal.cpp
        graph/graph_tracking.cpp
        hal.cpp
        ${API_GENERATED_HEADERS}
)
set_source_files_properties(
    api/tt-metalium/serialized_descriptors/mesh_coordinate.fbs
    PROPERTIES
        SKIP_LINTING
            ON
)

# TODO(afuller): this should be self-describing modules.
#   For now just cherry-pick all the files I discovered empirically by trying to run a test.
add_library(jitapi INTERFACE)
# These headers are for the device, not host; will require cross compiling to verify.
set_target_properties(
    jitapi
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

file(GLOB_RECURSE COMPUTE_KERNEL_API include/*)
file(GLOB_RECURSE FABRIC_HW_INC fabric/hw/inc/*)
file(
    GLOB_RECURSE TT_LLK_HEADERS
    third_party/tt_llk/tt_llk_wormhole_b0/**/*.h
    third_party/tt_llk/tt_llk_blackhole/**/*.h
)

target_sources(
    jitapi
    INTERFACE
        FILE_SET jit_api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            api/tt-metalium/buffer_types.hpp
            api/tt-metalium/circular_buffer_constants.h
            api/tt-metalium/constants.hpp
            api/tt-metalium/experimental/fabric/fabric_edm_types.hpp
            api/tt-metalium/experimental/fabric/edm_fabric_counters.hpp
            core_descriptors/blackhole_140_arch.yaml
            core_descriptors/blackhole_140_arch_eth_dispatch.yaml
            core_descriptors/blackhole_140_arch_fabric_mux.yaml
            core_descriptors/wormhole_b0_80_arch.yaml
            core_descriptors/wormhole_b0_80_arch_fabric_mux.yaml
            core_descriptors/wormhole_b0_80_arch_eth_dispatch.yaml
            fabric/mesh_graph_descriptors/n150_mesh_graph_descriptor.textproto
            fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.textproto
            fabric/mesh_graph_descriptors/n300_2x2_mesh_graph_descriptor.textproto
            fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto
            fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
            fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto
            fabric/mesh_graph_descriptors/p150_x4_mesh_graph_descriptor.textproto
            fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto
            fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
            fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto
            fabric/mesh_graph_descriptors/single_galaxy_mesh_graph_descriptor.textproto
            fabric/mesh_graph_descriptors/single_galaxy_torus_xy_graph_descriptor.textproto
            fabric/mesh_graph_descriptors/single_galaxy_torus_x_graph_descriptor.textproto
            fabric/mesh_graph_descriptors/single_galaxy_torus_y_graph_descriptor.textproto
            fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto
            fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto
            fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.textproto
            fabric/fabric_edm_packet_header.hpp
            ${FABRIC_HW_INC}
            impl/dispatch/kernels/cq_commands.hpp
            impl/dispatch/kernels/cq_common.hpp
            impl/dispatch/kernels/cq_relay.hpp
            impl/dispatch/kernels/cq_helpers.hpp
            ${COMPUTE_KERNEL_API}
            soc_descriptors/blackhole_140_arch.yaml
            soc_descriptors/wormhole_b0_80_arch.yaml
            tools/profiler/kernel_profiler.hpp
            tools/profiler/fabric_event_profiler.hpp
            tools/profiler/noc_event_profiler.hpp
            tools/profiler/cpp_device_analyses.json
            # Kernel sources
            impl/dispatch/kernels/cq_dispatch.cpp
            impl/dispatch/kernels/cq_dispatch_subordinate.cpp
            impl/dispatch/kernels/cq_prefetch.cpp
            fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp
            fabric/impl/kernels/tt_fabric_mux.cpp
            kernels/compute/blank.cpp
            kernels/compute/eltwise_binary.cpp
            kernels/compute/eltwise_sfpu.cpp
            kernels/dataflow/blank.cpp
            kernels/dataflow/reader_binary_diff_lengths.cpp
            kernels/dataflow/reader_unary.cpp
            kernels/dataflow/writer_unary.cpp
            kernels/dataflow/writer_unary_1.cpp
            ${TT_LLK_HEADERS}
)

include(CheckFunctionExists)

check_function_exists(
    backtrace
    HAVE_BACKTRACE
)

if(NOT HAVE_BACKTRACE)
    find_library(EXECINFO_LIB execinfo)
else()
    set(EXECINFO_LIB "")
endif()

target_link_libraries(
    tt_metal
    PUBLIC
        umd::device
        enchantum::enchantum
        fmt::fmt-header-only
        TracyClient
        nlohmann_json::nlohmann_json
        TT::Metalium::HostDevCommon
        Reflect::Reflect
        TT::STL
        tt-logger::tt-logger
        ${EXECINFO_LIB}
    PRIVATE
        Metalium::Metal::Impl
        metal_common_libs
        common
        jit_build
        llrt
        detail
        distributed
        fabric
        FlatBuffers::FlatBuffers
        TT::Metalium::Logging
        TT::ScaleoutTools
)

target_precompile_headers(tt_metal REUSE_FROM TT::CommonPCH)

target_include_directories(
    tt_metal
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/api>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/api>"
    PRIVATE
        api/tt-metalium # FIXME: Re-home the tests and remove this
)
target_compile_options(tt_metal PUBLIC -Wno-int-to-pointer-cast)
add_dependencies(tt_metal hw_toolchain)

set_target_properties(
    tt_metal
    PROPERTIES
        INSTALL_RPATH
            "/opt/openmpi-v5.0.7-ulfm/lib;${PROJECT_BINARY_DIR}/lib;$ORIGIN"
        ADDITIONAL_CLEAN_FILES
            "${PROJECT_BINARY_DIR}/lib;${PROJECT_BINARY_DIR}/obj"
)

if(BUILD_PROGRAMMING_EXAMPLES)
    add_subdirectory(programming_examples)
endif()

# Allow internal files to access the public API "by default" and without the
# scoping that external consumers must use.  Scoping may still be used if desired.
include_directories(
    api
    api/tt-metalium
    ${CMAKE_CURRENT_BINARY_DIR}/api
)

add_subdirectory(logging)
add_subdirectory(hw)
add_subdirectory(hostdevcommon)
add_subdirectory(common)
add_subdirectory(jit_build)
add_subdirectory(llrt)
add_subdirectory(tools)
add_subdirectory(impl)
add_subdirectory(detail)
add_subdirectory(distributed)
add_subdirectory(fabric)
add_subdirectory(experimental/udm)
if(TT_METAL_BUILD_TESTS)
    add_subdirectory(test)
endif()

if(TT_INSTALL)
    install(
        TARGETS
            tt_metal
        EXPORT Metalium
        LIBRARY
            COMPONENT metalium-runtime
        FILE_SET
        api
            DESTINATION COMPONENT
            metalium-dev
    )
    install(EXPORT Metalium NAMESPACE TT:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tt-metalium COMPONENT metalium-dev)

    install(
        TARGETS
            jitapi
        FILE_SET
        jit_api
            DESTINATION
                ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal # FIXME: fix the include paths for jit_build
            COMPONENT metalium-runtime
    )

    # 3rd party dependencies we export in our public API.
    # We must ship not only these libraries, but their public headers as well
    # ... or refactor our public API.
    install(
        TARGETS
            reflect
            enchantum
            TracyClient
            tt-logger
        EXPORT Metalium
        FILE_SET
        api
            COMPONENT metalium-dev
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/metalium-thirdparty
    )
endif()
