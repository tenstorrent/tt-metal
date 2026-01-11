cmake_minimum_required(VERSION 3.22...3.30)
project(distributed_write_sem LANGUAGES CXX)

add_executable(distributed_write_sem)
file(GLOB_RECURSE kernels kernels/*)
message(STATUS "Kerneks = ${kernels}")
target_sources(
    distributed_write_sem
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        distributed_write_sem.cpp
)
if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(
    distributed_write_sem
    PRIVATE
        TT::Metalium
        TTNN::CPP
)
