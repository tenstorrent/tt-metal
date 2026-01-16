add_library(ttnn_mobilenetv2 OBJECT ttnn_mobilenetv2.cpp)
target_link_libraries(ttnn_mobilenetv2
    PUBLIC
        torch::torch
        TTNN::CPP
        TT::Metalium
)

add_executable(mobilenetv2_cpp)
target_sources(mobilenetv2_cpp
    PRIVATE
        test/main.cpp
        helper_funcs.cpp
        inference/mobilenetv2_infra.cpp
)
target_link_libraries(mobilenetv2_cpp PRIVATE ttnn_mobilenetv2)

add_executable(mobilenetv2_e2e)
target_sources(mobilenetv2_e2e
    PRIVATE
        test/e2e_test.cpp
        helper_funcs.cpp
        inference/mobilenetv2_infra.cpp
        inference/mobilenetv2_e2e_performance.cpp
)
target_link_libraries(mobilenetv2_e2e PRIVATE ttnn_mobilenetv2)
