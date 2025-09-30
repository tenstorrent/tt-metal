cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_package_integration)

find_package(OpenCV REQUIRED)
find_package(Protobuf REQUIRED)

add_executable(metal_example_package_integration)
target_sources(metal_example_package_integration PRIVATE package_integration.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(
    metal_example_package_integration
    PUBLIC
        TT::Metalium
        ${OpenCV_LIBS}
        protobuf::libprotobuf
)
