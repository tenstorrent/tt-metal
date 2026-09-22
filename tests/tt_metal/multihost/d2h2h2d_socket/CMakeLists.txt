# A real 2-rank MPI program, so it only builds when MPI is available.
if(TARGET OpenMPI::MPI)
    add_library(d2h2h2d_socket_h2h_bw_test OBJECT)
    TT_ENABLE_UNITY_BUILD(d2h2h2d_socket_h2h_bw_test)

    target_sources(d2h2h2d_socket_h2h_bw_test PRIVATE test_h2h_bw.cpp)

    target_include_directories(
        d2h2h2d_socket_h2h_bw_test
        PRIVATE
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
            ${PROJECT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/tests
    )

    target_compile_definitions(d2h2h2d_socket_h2h_bw_test PRIVATE TT_METAL_HOST_BRIDGE=1)

    target_link_libraries(
        d2h2h2d_socket_h2h_bw_test
        PRIVATE
            test_metal_common_libs
            OpenMPI::MPI
    )

    add_executable(d2h2h2d_socket_h2h_bw)
    set_target_properties(
        d2h2h2d_socket_h2h_bw
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/test/tt_metal
    )
    target_link_libraries(d2h2h2d_socket_h2h_bw PRIVATE d2h2h2d_socket_h2h_bw_test)
endif()
