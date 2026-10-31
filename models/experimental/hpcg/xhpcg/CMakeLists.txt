# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# xhpcg for the P150: HPCG 3.1's own program (main.cpp, setup, validation tests, report), downloaded from its release,
# with its user-replaceable functions replaced by the files in src/ (which call the P150 implementation in tt/).
#
#   cmake -S models/experimental/hpcg/xhpcg -B build_xhpcg -G Ninja -DCMAKE_CXX_COMPILER=clang++-20 \
#         "-DCMAKE_PREFIX_PATH=$TT_METAL_HOME/build/lib/cmake;$TT_METAL_HOME/build/share/cmake"
#   cmake --build build_xhpcg -j
#   TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME TT_METAL_INSPECTOR=0 build_xhpcg/xhpcg --nx=104 --rt=60
#   (run from any directory; it writes its report there)

cmake_minimum_required(VERSION 3.24)
project(xhpcg LANGUAGES CXX)

find_package(TT-NN CONFIG REQUIRED) # TT-Metal's build tree, found through CMAKE_PREFIX_PATH (above)
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
    # clang without its own OpenMP runtime: GCC's (libgomp), which clang supports
    # (only GCC's omp.h is copied in: GCC's other internal headers would shadow clang's own)
    file(GLOB _gomp_headers /usr/lib/gcc/x86_64-linux-gnu/*/include/omp.h)
    list(SORT _gomp_headers)
    list(POP_BACK _gomp_headers _gomp_h)
    # GCC's attribute form __malloc__ (deallocator) is not clang's: dropped (an allocation hint only)
    file(READ ${_gomp_h} _gomp_text)
    string(REGEX REPLACE "__malloc__ *\\([^)]*\\)" "__malloc__" _gomp_text "${_gomp_text}")
    file(WRITE ${CMAKE_BINARY_DIR}/gomp_include/omp.h "${_gomp_text}")
    add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
    set_target_properties(
        OpenMP::OpenMP_CXX
        PROPERTIES
            INTERFACE_COMPILE_OPTIONS
                "-fopenmp=libgomp"
            INTERFACE_INCLUDE_DIRECTORIES
                ${CMAKE_BINARY_DIR}/gomp_include
            INTERFACE_LINK_OPTIONS
                "-fopenmp=libgomp"
    )
endif()

include(FetchContent)
FetchContent_Declare(
    hpcg
    URL
        https://github.com/hpcg-benchmark/hpcg/archive/refs/tags/HPCG-release-3-1-0.tar.gz
    URL_HASH SHA256=be841a30231c09b80d715600846212d2854c6709a56ba85af337e43a26255133
    SOURCE_SUBDIR
        no-cmake # sources only: HPCG's own build files are not used
)
FetchContent_MakeAvailable(hpcg)

# HPCG's files that a port replaces (src/), and the rest of HPCG, unchanged
set(REPLACED
    CG
    ComputeSPMV
    ComputeSYMGS
    ComputeMG
    ComputeDotProduct
    ComputeWAXPBY
    OptimizeProblem
)
file(GLOB HPCG_SOURCES ${hpcg_SOURCE_DIR}/src/*.cpp)
set(OUR_SOURCES)
foreach(f ${REPLACED})
    list(REMOVE_ITEM HPCG_SOURCES ${hpcg_SOURCE_DIR}/src/${f}.cpp)
    list(APPEND OUR_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/${f}.cpp)
endforeach()

# HPCG's sources and the replacements: HPCG's flags (setup/Make.GCC_OMP), one process (no MPI)
add_library(
    hpcg_sources
    OBJECT
    ${HPCG_SOURCES}
    ${OUR_SOURCES}
)
target_include_directories(hpcg_sources PRIVATE ${hpcg_SOURCE_DIR}/src)
target_compile_definitions(hpcg_sources PRIVATE HPCG_NO_MPI)
target_compile_options(
    hpcg_sources
    PRIVATE
        -O3
        -ffast-math
)
target_link_libraries(hpcg_sources PRIVATE OpenMP::OpenMP_CXX)

# the P150 implementation: TT-NN, C++20; HPCG's headers for its data structures (the same HPCG_NO_MPI layout)
add_library(
    hpcg_tt
    OBJECT
    tt/device.cpp
    tt/generate.cpp
    tt/hpcg_tt.cpp
    tt/level.cpp
    tt/ops.cpp
    tt/plan.cpp
)
target_compile_features(hpcg_tt PRIVATE cxx_std_20)
target_include_directories(hpcg_tt PRIVATE ${hpcg_SOURCE_DIR}/src)
target_compile_definitions(hpcg_tt PRIVATE HPCG_NO_MPI)
target_compile_options(hpcg_tt PRIVATE -O2)
target_link_libraries(
    hpcg_tt
    PRIVATE
        TTNN::TTNN
        OpenMP::OpenMP_CXX
)

# the Python package's compiled planners (host/native_setup.cpp), linked in
add_library(native_setup OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/../host/native_setup.cpp)
target_compile_features(native_setup PRIVATE cxx_std_17)
target_compile_options(native_setup PRIVATE -O3)
target_link_libraries(native_setup PRIVATE OpenMP::OpenMP_CXX)

add_executable(xhpcg)
target_link_libraries(
    xhpcg
    PRIVATE
        hpcg_sources
        hpcg_tt
        native_setup
        TTNN::TTNN
        OpenMP::OpenMP_CXX
)
