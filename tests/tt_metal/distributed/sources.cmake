# Source files for tt_metal distributed tests
# Module owners should update this file when adding/removing/renaming source files

set(DISTRIBUTED_UNIT_TEST_SOURCES
    test_auto_mpi_init.cpp
    test_control_plane_local_mesh_binding.cpp
    test_distributed_coordinate_translator.cpp
    test_end_to_end_eltwise.cpp
    test_distributed_host_buffer.cpp
    test_maybe_remote.cpp
    test_mesh_buffer.cpp
    test_mesh_coord.cpp
    test_mesh_device.cpp
    test_mesh_device_reshape.cpp
    test_mesh_device_view.cpp
    test_mesh_workload.cpp
    test_mesh_socket.cpp
    test_hd_sockets.cpp
    test_mesh_sub_device.cpp
    test_mesh_allocator.cpp
    test_mesh_events.cpp
    test_mesh_trace.cpp
    test_thread_pool.cpp
    test_dispatch_context.cpp
    utils.cpp
)
# MPI sub-context tests require the MPI distributed build (matches tt_metal/distributed USE_MPI).
if(ENABLE_DISTRIBUTED)
    list(APPEND DISTRIBUTED_UNIT_TEST_SOURCES test_mpi_subcontext.cpp)
else()
    list(APPEND DISTRIBUTED_UNIT_TEST_SOURCES test_single_host_context.cpp)
endif()
