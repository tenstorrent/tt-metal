# Source files for tt_metal dispatch tests
# Module owners should update this file when adding/removing/renaming source files

set(UNIT_TESTS_DISPATCH_SMOKE_SOURCES
    dispatch_device/test_enqueue_read_write_core.cpp
    dispatch_event/test_EnqueueWaitForEvent.cpp
    dispatch_event/test_events.cpp
    dispatch_program/test_dispatch_stress.cpp
    dispatch_program/test_sub_device.cpp
    dispatch_program/test_kernel_config_buffer.cpp
    dispatch_util/test_device_command.cpp
    dispatch_util/test_dispatch_settings.cpp
)

set(UNIT_TESTS_DISPATCH_BASIC_SOURCES
    dispatch_buffer/test_sub_device.cpp
    dispatch_buffer/test_BufferCorePageMapping_Iterator.cpp
    dispatch_program/test_dispatch.cpp
    dispatch_program/test_dataflow_cb.cpp
    dispatch_program/test_global_circular_buffers.cpp
    dispatch_trace/test_sub_device.cpp
    dispatch_util/test_ringbuffer_cache.cpp
)

set(UNIT_TESTS_DISPATCH_SLOW_SOURCES
    dispatch_buffer/test_EnqueueWriteBuffer_and_EnqueueReadBuffer.cpp
    dispatch_buffer/test_large_mesh_buffer.cpp
    dispatch_program/test_EnqueueProgram.cpp
    dispatch_trace/test_EnqueueTrace.cpp
)
