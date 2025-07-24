set(IMPL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/sub_device/sub_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sub_device/sub_device_manager_tracker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sub_device/sub_device_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/context/metal_context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/device_pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffers/buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffers/buffer_distribution_spec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffers/buffer_page_mapping.cpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/data_format/tile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data_format/tilize_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel_types.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator/algorithms/free_list.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator/algorithms/free_list_opt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator/allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator/bank_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator/l1_banking_allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/program/program.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/program/dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/program/program_descriptors.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/program/program_device_map.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/profiler/profiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/profiler/tt_metal_profiler.cpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/ringbuffer_cache.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/topology.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/kernel_config/fd_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/kernel_config/prefetch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/kernel_config/dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/kernel_config/dispatch_s.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/kernel_config/relay_mux.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/util/dispatch_settings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/debug/dprint_server.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/debug/inspector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/debug/inspector_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/debug/noc_logging.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/debug/watcher_server.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/debug/watcher_device_reader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/trace/trace.cpp
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
add_library(impl OBJECT ${IMPL_SRC})
add_library(Metalium::Metal::Impl ALIAS impl)

target_link_libraries(
    impl
    PUBLIC
        common
        Tracy::TracyClient
    PRIVATE
        Boost::smart_ptr
        FlatBuffers::FlatBuffers
        range-v3::range-v3
        TT::Metalium::Common
        TT::Metalium::HostDevCommon
        Metalium::Metal::Hardware
        Metalium::Metal::LLRT
)

target_include_directories(
    impl
    PUBLIC
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/debug
        ${CMAKE_CURRENT_SOURCE_DIR}/profiler
        ${PROJECT_SOURCE_DIR}/tt_metal/include
)
target_include_directories(impl SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers)

target_compile_options(impl PUBLIC -Wno-int-to-pointer-cast)
