set(IMPL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/device/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/device_handle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/device_pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffers/buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffers/circular_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffers/global_semaphore.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffers/semaphore.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator/algorithms/free_list.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator/allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator/basic_allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator/l1_banking_allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/program/program.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/debug_tools.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/command_queue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/worker_config_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/data_collection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/debug/dprint_server.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/debug/noc_logging.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/debug/watcher_server.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/debug/watcher_device_reader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/trace/trace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/trace/trace_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lightmetal/lightmetal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lightmetal/lightmetal_replay.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lightmetal/lightmetal_capture_context.cpp
)

# Include helper functions and generate headers from flatbuffer schemas
include(${PROJECT_SOURCE_DIR}/cmake/flatbuffers.cmake)

set(FLATBUFFER_SCHEMAS
    ${CMAKE_CURRENT_SOURCE_DIR}/lightmetal/binary.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/tracehost/command.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/tracehost/types.fbs
)
foreach(FBS_FILE ${FLATBUFFER_SCHEMAS})
    GENERATE_FBS_HEADER(${FBS_FILE})
    list(APPEND IMPL_SRC ${FBS_GENERATED_HEADER})
endforeach()

add_library(impl OBJECT ${IMPL_SRC})
add_library(Metalium::Metal::Impl ALIAS impl)

target_link_libraries(
    impl
    PUBLIC
        common
        Metalium::Metal::LLRT
        FlatBuffers::FlatBuffers
    PRIVATE
        Boost::smart_ptr
        range-v3::range-v3
)

target_include_directories(
    impl
    PUBLIC
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal/include
        ${CMAKE_CURRENT_BINARY_DIR} # For generated flatbuffer headers
)

target_compile_options(impl PUBLIC -Wno-int-to-pointer-cast)
