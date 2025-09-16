###################################################################################################################
# libtorch : https://pytorch.org/
###################################################################################################################
CPMAddPackage(
    NAME libtorch
    VERSION 2.7.1+cpu
    URL
        https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip
    DOWNLOAD_ONLY ON
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

link_directories(
    ${libtorch_SOURCE_DIR}/lib
)

# Add test_cpp subdirectory
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test_cpp)