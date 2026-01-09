cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_lab_eltwise_binary)

add_executable(metal_example_lab_eltwise_binary)
target_sources(metal_example_lab_eltwise_binary PRIVATE lab_eltwise_binary.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

# TTNN::CPP is defined in the ttnn subdirectory which is processed after tt_metal.
# When building from source, CMake will resolve the target during generation.
if(NOT TARGET TTNN::CPP)
    # Check if ttnn exists as a subdirectory in the source tree (source build)
    # If so, don't call find_package - the target will be defined later
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/ttnn/CMakeLists.txt")
        # Not a source build, try to find installed package
        find_package(TT-NN QUIET)
    endif()
endif()

target_link_libraries(
    metal_example_lab_eltwise_binary
    PUBLIC
        TT::Metalium
        TTNN::CPP
)
