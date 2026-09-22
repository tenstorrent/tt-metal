# The whole chain end to end: a real 2-rank MPI program, so it only builds with MPI.
if(TARGET OpenMPI::MPI)
    add_library(d2h2h2d_socket_volume_d2d_test OBJECT)
    TT_ENABLE_UNITY_BUILD(d2h2h2d_socket_volume_d2d_test)

    target_sources(d2h2h2d_socket_volume_d2d_test PRIVATE test_volume_d2d.cpp)

    target_include_directories(
        d2h2h2d_socket_volume_d2d_test
        PRIVATE
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
            ${PROJECT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/tests
    )

    target_compile_definitions(
        d2h2h2d_socket_volume_d2d_test
        PRIVATE
            TT_METAL_HOST_BRIDGE=1
            TT_DIRECT_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )

    target_link_libraries(
        d2h2h2d_socket_volume_d2d_test
        PRIVATE
            test_metal_common_libs
            OpenMPI::MPI
    )

    add_executable(d2h2h2d_socket_volume_d2d)
    set_target_properties(
        d2h2h2d_socket_volume_d2d
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/test/tt_metal
    )
    target_link_libraries(d2h2h2d_socket_volume_d2d PRIVATE d2h2h2d_socket_volume_d2d_test)
endif()
