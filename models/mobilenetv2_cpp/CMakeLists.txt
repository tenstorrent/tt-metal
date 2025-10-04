###################################################################################################################
# libtorch : https://pytorch.org/
###################################################################################################################
CPMAddPackage(
    NAME libtorch
    VERSION 2.7.1+cpu
    URL
        https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip
    DOWNLOAD_ONLY ON
    PATCHES
        ${CMAKE_CURRENT_SOURCE_DIR}/script/aten.patch
)

set(TORCH_LIBRARIES
    c10
    torch_cpu
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${libtorch_SOURCE_DIR}/include
    ${libtorch_SOURCE_DIR}/include/torch/csrc/api/include
)

link_directories(${libtorch_SOURCE_DIR}/lib)

add_library(ttnn_mobilenetv2 OBJECT ttnn_mobilenetv2.cpp)
add_dependencies(
    ttnn_mobilenetv2
    ttnncpp
    tt_metal
)
target_link_libraries(
    ttnn_mobilenetv2
    PUBLIC
        ${TORCH_LIBRARIES}
        ttnncpp
)

set(TEST_INFRA
    test/main.cpp
    helper_funcs.cpp
    inference/mobilenetv2_infra.cpp
)
add_executable(mobilenetv2_cpp ${TEST_INFRA})
target_link_libraries(mobilenetv2_cpp PRIVATE ttnn_mobilenetv2)

set(TEST_E2E
    test/e2e_test.cpp
    helper_funcs.cpp
    inference/mobilenetv2_infra.cpp
    inference/mobilenetv2_e2e_performance.cpp
)
add_executable(mobilenetv2_e2e ${TEST_E2E})
target_link_libraries(mobilenetv2_e2e PRIVATE ttnn_mobilenetv2)
