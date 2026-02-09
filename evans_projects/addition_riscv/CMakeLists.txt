set(CMAKE_SYSTEM_PROCESSOR "x86_64")

set(CMAKE_C_COMPILER clang-20 CACHE INTERNAL "C compiler")

set(CMAKE_CXX_COMPILER clang++-20 CACHE INTERNAL "C++ compiler")

# Use for configure time
set(ENABLE_LIBCXX FALSE CACHE INTERNAL "Using clang's libc++")

# Our build is super slow; put a band-aid on it by choosing a linker that can cope better.
# We really need to fix out code, though.
find_program(MOLD ld.mold)
if(MOLD)
    set(CMAKE_LINKER_TYPE MOLD)
else()
    find_program(LLD ld.lld-20)
    if(LLD)
        set(CMAKE_LINKER_TYPE LLD)
    endif()
endif()

cmake_minimum_required(VERSION 3.22...4.2)
project(basic_adding)

add_executable(basic_adding_binary)
target_sources(basic_adding_binary PRIVATE add_in_riscv.cpp)

set(_tt_install "${CMAKE_CURRENT_SOURCE_DIR}/../../build_Release")
set(TT-Metalium_DIR "${_tt_install}/lib/cmake/tt-metalium")
list(PREPEND CMAKE_PREFIX_PATH "${_tt_install}")

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(basic_adding_binary PUBLIC TT::Metalium)
