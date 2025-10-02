cmake_minimum_required(VERSION 3.22...3.30)
project(nlp_concat_heads_boltz_cpp)

# Build C++ example that uses TTNN C++ API to call the op (basic version)
add_executable(alexp_nlp_concat_heads_boltz_cpp)
target_sources(alexp_nlp_concat_heads_boltz_cpp PRIVATE nlp_concat_heads_boltz_cpp.cpp)

# Build C++ example with sharded tensor (debug version)
add_executable(alexp_nlp_concat_heads_boltz_sharded_cpp)
target_sources(alexp_nlp_concat_heads_boltz_sharded_cpp PRIVATE nlp_concat_heads_boltz_sharded_cpp.cpp)

# Build C++ example with working sharded approach (workaround version)
add_executable(alexp_nlp_concat_heads_boltz_working_sharded_cpp)
target_sources(alexp_nlp_concat_heads_boltz_working_sharded_cpp PRIVATE nlp_concat_heads_boltz_working_sharded_cpp.cpp)

# Build quick debug version
add_executable(quick_debug_sharded)
target_sources(quick_debug_sharded PRIVATE quick_debug_sharded.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

# Link all executables
target_link_libraries(alexp_nlp_concat_heads_boltz_cpp PUBLIC TT::Metalium)
target_link_libraries(alexp_nlp_concat_heads_boltz_sharded_cpp PUBLIC TT::Metalium)
target_link_libraries(alexp_nlp_concat_heads_boltz_working_sharded_cpp PUBLIC TT::Metalium)

# Use the rebuilt TTNN shared library with all symbols
message(STATUS "Using rebuilt TTNN CPP shared library")
target_link_libraries(
    alexp_nlp_concat_heads_boltz_cpp
    PUBLIC
        TT::Metalium
        TTNN::CPP
)
target_link_libraries(
    alexp_nlp_concat_heads_boltz_sharded_cpp
    PUBLIC
        TT::Metalium
        TTNN::CPP
)
target_link_libraries(
    alexp_nlp_concat_heads_boltz_working_sharded_cpp
    PUBLIC
        TT::Metalium
        TTNN::CPP
)
target_link_libraries(
    quick_debug_sharded
    PUBLIC
        TT::Metalium
        TTNN::CPP
)

target_include_directories(alexp_nlp_concat_heads_boltz_cpp PRIVATE ${CMAKE_SOURCE_DIR}/ttnn/cpp)
target_include_directories(alexp_nlp_concat_heads_boltz_sharded_cpp PRIVATE ${CMAKE_SOURCE_DIR}/ttnn/cpp)
target_include_directories(alexp_nlp_concat_heads_boltz_working_sharded_cpp PRIVATE ${CMAKE_SOURCE_DIR}/ttnn/cpp)
target_include_directories(quick_debug_sharded PRIVATE ${CMAKE_SOURCE_DIR}/ttnn/cpp)
