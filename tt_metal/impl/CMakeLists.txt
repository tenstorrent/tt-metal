set(IMPL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/sub_device/sub_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sub_device/sub_device_manager_tracker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sub_device/sub_device_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/context/metal_context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/experimental/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/mock_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/device_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffers/buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffers/buffer_distribution_spec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffers/buffer_page_mapping.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffers/tensor_accessor_args.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffers/dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffers/circular_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffers/circular_buffer_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffers/global_circular_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffers/global_semaphore.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffers/semaphore.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data_format/blockfloat_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data_format/bfloat4.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data_format/bfloat8.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data_format/bfloat16.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data_format/bfloat16_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data_format/tile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data_format/tilize_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel_types.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator/algorithms/free_list_opt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator/allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator/allocator_state.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator/bank_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator/l1_banking_allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/program/program.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/program/dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/program/program_descriptors.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/program/program_device_map.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/profiler/profiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/profiler/tt_metal_profiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/profiler/profiler_analysis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/profiler/profiler_state_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/cq_shared_state.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/debug_tools.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/device_command.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/host_runtime_commands.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/device_command_calculator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/dispatch_query_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/dispatch_core_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/dispatch_core_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/command_queue_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/dispatch_mem_map.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/system_memory_cq_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/system_memory_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/hardware_command_queue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/launch_message_ring_buffer_state.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/worker_config_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/data_collection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/data_collector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/ringbuffer_cache.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/topology.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/kernel_config/fd_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/kernel_config/prefetch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/kernel_config/dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/kernel_config/dispatch_s.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/kernel_config/relay_mux.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/util/dispatch_settings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/debug/dprint_server.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/debug/inspector/data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/debug/inspector/inspector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/debug/inspector/logger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/debug/inspector/rpc_server_controller.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/debug/noc_logging.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/debug/watcher_server.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/debug/watcher_device_reader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/trace/trace_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/trace/dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event/event.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event/dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/base_types_from_flatbuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/base_types_to_flatbuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/buffer_types_from_flatbuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/buffer_types_to_flatbuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/program_types_from_flatbuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/program_types_to_flatbuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lightmetal/lightmetal_replay.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lightmetal/lightmetal_replay_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lightmetal/lightmetal_capture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lightmetal/lightmetal_capture_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lightmetal/host_api_capture_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/debug/dprint_parser.cpp
)

# Include helper functions and generate headers from flatbuffer schemas
include(flatbuffers)

set(FLATBUFFER_SCHEMAS
    ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/light_metal_binary.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/command.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/base_types.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/buffer_types.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/program_types.fbs
)

set(GENERATED_HEADERS)
foreach(FBS_FILE ${FLATBUFFER_SCHEMAS})
    GENERATE_FBS_HEADER(${FBS_FILE})
    list(APPEND GENERATED_HEADERS ${FBS_GENERATED_HEADER_FILE})
endforeach()

list(APPEND IMPL_SRC ${GENERATED_HEADERS})

# Generate Inspector Cap'n Proto headers and RPC server implementation
set(INSPECTOR_RPC_CAPNP ${CMAKE_CURRENT_SOURCE_DIR}/debug/inspector/rpc.capnp)
set(CAPNPC_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/impl/)
file(MAKE_DIRECTORY ${CAPNPC_OUTPUT_DIR})

capnp_generate_cpp(INSPECTOR_RPC_CAPNP_SOURCES INSPECTOR_RPC_CAPNP_HEADER ${INSPECTOR_RPC_CAPNP})
list(APPEND IMPL_SRC ${INSPECTOR_RPC_CAPNP_SOURCES})

# Generate Inspector RPC callbacks server implementation
set(INSPECTOR_RPC_GENERATED_HEADER ${CMAKE_CURRENT_BINARY_DIR}/impl/debug/inspector/rpc_server_generated.hpp)
set(INSPECTOR_RPC_GENERATED_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/impl/debug/inspector/rpc_server_generated.cpp)
add_custom_command(
    OUTPUT
        ${INSPECTOR_RPC_GENERATED_HEADER}
        ${INSPECTOR_RPC_GENERATED_SOURCE}
    COMMAND
        python3 ${CMAKE_CURRENT_SOURCE_DIR}/debug/inspector/generate_rpc_server.py ${INSPECTOR_RPC_CAPNP}
        ${INSPECTOR_RPC_GENERATED_HEADER} ${INSPECTOR_RPC_GENERATED_SOURCE}
    DEPENDS
        ${INSPECTOR_RPC_CAPNP}
        ${CMAKE_CURRENT_SOURCE_DIR}/debug/inspector/generate_rpc_server.py
    COMMENT "Generating RPC server implementation for ${INSPECTOR_RPC_CAPNP}"
)
list(APPEND IMPL_SRC ${INSPECTOR_RPC_GENERATED_SOURCE})

# Disable clang-tidy for generated Inspector files
set_source_files_properties(
    ${INSPECTOR_RPC_GENERATED_SOURCE}
    ${INSPECTOR_RPC_GENERATED_HEADER}
    ${INSPECTOR_RPC_CAPNP_SOURCES}
    PROPERTIES
        SKIP_LINTING
            TRUE
)

# Define dependency for CI small build for clang-tidy
if(TARGET all_generated_files)
    add_custom_target(
        capnp_generated_files
        DEPENDS
            ${INSPECTOR_RPC_CAPNP_HEADER}
            ${INSPECTOR_RPC_GENERATED_HEADER}
    )
    add_dependencies(all_generated_files capnp_generated_files)
endif()

add_library(impl OBJECT ${IMPL_SRC})
add_library(Metalium::Metal::Impl ALIAS impl)

# Enable unity builds for faster compilation
TT_ENABLE_UNITY_BUILD(impl)

target_link_libraries(
    impl
    PUBLIC
        common
        Tracy::TracyClient
    PRIVATE
        FlatBuffers::FlatBuffers
        range-v3::range-v3
        TT::Metalium::Common
        TT::Metalium::HostDevCommon
        Metalium::Metal::Hardware
        Metalium::Metal::LLRT
        capnp
        capnp-rpc
)

target_include_directories(
    impl
    PUBLIC
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/debug
        ${CMAKE_CURRENT_SOURCE_DIR}/profiler
        ${PROJECT_SOURCE_DIR}/tt_metal/include
        ${CMAKE_CURRENT_BINARY_DIR}
)
target_include_directories(impl SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers)

# Set KJ_NO_EXCEPTIONS=0 to enable exceptions in Cap'n Proto code due to a detection bug with g++-12.
target_compile_options(impl PUBLIC -Wno-int-to-pointer-cast PRIVATE -DKJ_NO_EXCEPTIONS=0)

# Generate Inspector RPC stub file
set(INSPECTOR_RPC_STUB_FILE ${PROJECT_SOURCE_DIR}/tools/triage/inspector_capnp.pyi)
add_custom_command(
    OUTPUT
        ${INSPECTOR_RPC_STUB_FILE}
    COMMAND
        python3 ${CMAKE_CURRENT_SOURCE_DIR}/debug/inspector/generate_rpc_stub.py ${INSPECTOR_RPC_CAPNP}
        ${INSPECTOR_RPC_STUB_FILE}
    DEPENDS
        ${INSPECTOR_RPC_CAPNP}
        ${CMAKE_CURRENT_SOURCE_DIR}/debug/inspector/generate_rpc_stub.py
    COMMENT "Generating RPC stub for ${INSPECTOR_RPC_CAPNP}"
)
add_custom_target(inspector_rpc_stub ALL DEPENDS ${INSPECTOR_RPC_STUB_FILE})

# Optionally, ensure impl depends on the stub file
add_dependencies(impl inspector_rpc_stub)
