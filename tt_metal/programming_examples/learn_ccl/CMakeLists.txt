cmake_minimum_required(VERSION 3.22...3.30)
project(learn_ccl LANGUAGES CXX)

add_executable(learn_ccl)
target_sources(learn_ccl PRIVATE learn_ccl.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(
    learn_ccl
    PRIVATE
        TT::Metalium
        TTNN::CPP
)
