set(IMPL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/device/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/device_handle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/device_pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffers/buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffers/circular_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/buffers/semaphore.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/runtime_args_data.cpp
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
)

add_library(impl OBJECT ${IMPL_SRC})
target_link_libraries(impl PRIVATE Boost::smart_ptr)
target_link_libraries(impl PUBLIC common)

target_include_directories(
    impl
    PUBLIC
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(impl PUBLIC -Wno-int-to-pointer-cast)
