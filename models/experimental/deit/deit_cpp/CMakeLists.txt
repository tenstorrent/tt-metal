add_executable(
    deit_cpp_demo
    tt_cpp/demo_deit_ttnn_inference_perf_e2e_2cq_trace.cpp
    tt_cpp/deit_test_infra.cpp
    tt_cpp/deit_inference.cpp
)

target_include_directories(deit_cpp_demo PRIVATE tt_cpp)

if(BUILD_DEIT_OPENCV)
    target_include_directories(deit_cpp_demo PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(
        deit_cpp_demo
        PRIVATE
            TT::Metalium
            TTNN::CPP
            torch::torch
            opencv_core
            opencv_imgproc
            opencv_imgcodecs
    )
else()
    target_link_libraries(
        deit_cpp_demo
        PRIVATE
            TT::Metalium
            TTNN::CPP
    )
endif()
