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

# Tracy profiler WASM viewer (Emscripten).
#
# emsdk is fetched via CPM (just the installer/manager scripts ~250 KB). The
# actual Emscripten toolchain (emcc, Emscripten.cmake, node, ...) is materialized
# alongside the scripts on first configure via emsdk's own installer
# (`./emsdk install/activate`), so neither the source tree nor the shared CPM
# cache is polluted with the install. The CPM GIT_TAG pin determines what
# `latest` resolves to.
#
# CPM_SOURCE_CACHE is shadowed for this single call so CPM falls back to its
# default ${CMAKE_BINARY_DIR}/_deps/<name>-src/ destination. emsdk.py uses
# os.path.realpath(__file__) to locate its install root, so the package must
# live somewhere we can write hundreds of MB into; the build dir is the right
# place, the shared CPM cache is not.
set(_cpm_source_cache_save "${CPM_SOURCE_CACHE}")
set(CPM_SOURCE_CACHE "")
CPMAddPackage(
    NAME emsdk
    GITHUB_REPOSITORY emscripten-core/emsdk
    GIT_TAG
        0dfff153dc3c596948ac088b8853006c570f2609 # 5.0.7-4-g0dfff15
    DOWNLOAD_ONLY YES
)
set(CPM_SOURCE_CACHE "${_cpm_source_cache_save}")
unset(_cpm_source_cache_save)

set(EMSDK_DIR "${emsdk_SOURCE_DIR}")
set(EMSCRIPTEN_TOOLCHAIN_FILE "${EMSDK_DIR}/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake")

if(NOT EXISTS "${EMSCRIPTEN_TOOLCHAIN_FILE}")
    message(
        STATUS
        "Installing Emscripten via emsdk into ${EMSDK_DIR} (first configure only; downloads ~hundreds of MB)..."
    )
    execute_process(
        COMMAND
            ./emsdk install latest
        WORKING_DIRECTORY "${EMSDK_DIR}"
        RESULT_VARIABLE _emsdk_install_rc
    )
    if(NOT _emsdk_install_rc EQUAL 0)
        message(FATAL_ERROR "emsdk install failed (exit ${_emsdk_install_rc})")
    endif()
    execute_process(
        COMMAND
            ./emsdk activate latest
        WORKING_DIRECTORY "${EMSDK_DIR}"
        RESULT_VARIABLE _emsdk_activate_rc
    )
    if(NOT _emsdk_activate_rc EQUAL 0)
        message(FATAL_ERROR "emsdk activate failed (exit ${_emsdk_activate_rc})")
    endif()
endif()

if(NOT EXISTS "${EMSCRIPTEN_TOOLCHAIN_FILE}")
    message(FATAL_ERROR "Emscripten toolchain file not found at ${EMSCRIPTEN_TOOLCHAIN_FILE} after install/activate.")
endif()

# Output path is consumed by tools/tracy/common.py (PROFILER_WASM_DIR), profiler CI tests,
# and the .github/workflows/build-artifact.yaml packaging step. Keep them in sync.
ExternalProject_Add(
    tracy_profiler_wasm
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tracy/profiler
    BINARY_DIR ${CMAKE_BINARY_DIR}/profiler/build_wasm
    CMAKE_ARGS
        -DCMAKE_TOOLCHAIN_FILE=${EMSCRIPTEN_TOOLCHAIN_FILE} -DCMAKE_BUILD_TYPE=MinSizeRel
    BUILD_ALWAYS FALSE
    INSTALL_COMMAND
        ""
)
