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
        ${CMAKE_CURRENT_SOURCE_DIR}/aten.patch
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

###################################################################################################################
# OpenCV : Use system installed OpenCV
###################################################################################################################
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV using: sudo apt install libopencv-dev")
endif()

###################################################################################################################
# Add test_cpp subdirectory
###################################################################################################################
add_subdirectory(test_cpp)
