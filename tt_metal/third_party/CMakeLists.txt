set(CMAKE_C_CLANG_TIDY "")
set(CMAKE_CXX_CLANG_TIDY "")
set(CMAKE_VERIFY_INTERFACE_HEADER_SETS FALSE)
set(TT_UMD_ENABLE_CLANG_TIDY OFF CACHE BOOL "Disable clang-tidy checks for UMD" FORCE)

if(TT_METAL_USE_EMULE)
    set(TT_UMD_BUILD_EMULE ON CACHE BOOL "Enable SWEmuleChip (forwarded from TT_METAL_USE_EMULE)" FORCE)
endif()

add_subdirectory(umd)
add_subdirectory(tracy)

set(NO_FILESELECTOR ON)
add_subdirectory(tracy/csvexport)
add_subdirectory(tracy/capture)

find_program(EMCMAKE emcmake REQUIRED)

ExternalProject_Add(tracy-profiler-wasm
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tracy/profiler
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tracy-profiler-wasm
    CMAKE_ARGS
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_CURRENT_BINARY_DIR}/emsdk/upstream/emscripten/emcmake/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake
        -DCMAKE_BUILD_TYPE=MinSizeRel
    BUILD_ALWAYS FALSE
    INSTALL_COMMAND ""
)
