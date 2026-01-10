cmake_minimum_required(VERSION 3.22...3.30)
project(distributed_matmul LANGUAGES CXX)

add_executable(distributed_matmul)
target_sources(distributed_matmul PRIVATE distributed_matmul.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(
    distributed_matmul
    PRIVATE
        TT::Metalium
        TTNN::CPP
)
