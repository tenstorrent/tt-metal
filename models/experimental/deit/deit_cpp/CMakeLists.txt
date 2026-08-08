add_executable(
    deit_cpp_demo
    tt_cpp/demo_deit_ttnn_inference_perf_e2e_2cq_trace.cpp
    tt_cpp/deit_test_infra.cpp
    tt_cpp/deit_inference.cpp
)

target_include_directories(deit_cpp_demo PRIVATE tt_cpp)

target_link_libraries(
    deit_cpp_demo
    PRIVATE
        TT::Metalium
        TTNN::CPP
)
