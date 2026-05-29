set(CMAKE_C_CLANG_TIDY "")
set(CMAKE_CXX_CLANG_TIDY "")
set(CMAKE_VERIFY_INTERFACE_HEADER_SETS FALSE)
set(TT_UMD_ENABLE_CLANG_TIDY OFF CACHE BOOL "Disable clang-tidy checks for UMD" FORCE)

if(TT_METAL_USE_EMULE)
    set(TT_UMD_BUILD_EMULE ON CACHE BOOL "Enable SWEmuleChip (forwarded from TT_METAL_USE_EMULE)" FORCE)
endif()

add_subdirectory(umd)

# Propagate ENABLE_TRACY into Tracy's option before add_subdirectory(tracy).
# Use CACHE FORCE so Tracy's option() cannot reset it under older CMake policy behavior.
if(ENABLE_TRACY)
    set(TRACY_ENABLE ON CACHE BOOL "Enable Tracy profiling instrumentation" FORCE)
else()
    set(TRACY_ENABLE OFF CACHE BOOL "Enable Tracy profiling instrumentation" FORCE)
endif()

# Avoid hard failures on hosts without invariant TSC.
set(TRACY_TIMER_FALLBACK ON CACHE BOOL "Tracy: use lower-resolution timer fallback" FORCE)

add_subdirectory(tracy)

# Tracy symbols/backtraces must be available to TracyClient consumers.
if(ENABLE_TRACY AND TARGET TracyClient)
    target_compile_options(TracyClient PUBLIC -fno-omit-frame-pointer)
    # Tracy sources emit this warning under our global -Werror; keep suppression local to TracyClient.
    target_compile_options(TracyClient PRIVATE -Wno-unused-variable)
    target_link_options(TracyClient PUBLIC -rdynamic)

    # Keep TracyClient in the standard lib output/install paths so wheel/CPack dependency discovery works.
    set_target_properties(
        TracyClient
        PROPERTIES
            EXCLUDE_FROM_ALL
                TRUE
            LIBRARY_OUTPUT_DIRECTORY
                "${PROJECT_BINARY_DIR}/lib"
            ARCHIVE_OUTPUT_DIRECTORY
                "${PROJECT_BINARY_DIR}/lib"
            OUTPUT_NAME
                "tracy"
    )
    install(
        TARGETS
            TracyClient
        LIBRARY
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT metalium-runtime
        ARCHIVE
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT metalium-runtime
    )
endif()

set(NO_FILESELECTOR ON)

# Build Tracy CLI deps statically so the binaries run from packaged artifacts without build/_deps.
set(_tt_tracy_saved_build_shared_libs "${BUILD_SHARED_LIBS}")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Create shared libraries" FORCE)

# Disable Tracy's -march=native for portable CLI binaries in CI.
set(NO_ISA_EXTENSIONS ON CACHE BOOL "Disable Tracy -march=native for portable CLI tools" FORCE)

# Repair AR/RANLIB when Clang leaves CMAKE_<LANG>_COMPILER_(AR|RANLIB)=NOTFOUND.
# Needed so Tracy's static deps (zstd, etc.) build reliably in CI.
if(NOT WIN32)
    if(NOT CMAKE_AR OR "${CMAKE_AR}" MATCHES "NOTFOUND")
        find_program(
            _tt_resolved_ar
            NAMES
                ar
                llvm-ar
                gcc-ar
                NO_CACHE
        )
        if(_tt_resolved_ar)
            set(CMAKE_AR "${_tt_resolved_ar}" CACHE FILEPATH "Path to archiver for static libraries" FORCE)
        endif()
        unset(_tt_resolved_ar)
    endif()
    if(NOT CMAKE_RANLIB OR "${CMAKE_RANLIB}" MATCHES "NOTFOUND")
        find_program(
            _tt_resolved_ranlib
            NAMES
                ranlib
                llvm-ranlib
                gcc-ranlib
                NO_CACHE
        )
        if(_tt_resolved_ranlib)
            set(CMAKE_RANLIB "${_tt_resolved_ranlib}" CACHE FILEPATH "Path to ranlib for static libraries" FORCE)
        endif()
        unset(_tt_resolved_ranlib)
    endif()
    foreach(_tt_lang IN ITEMS C CXX ASM)
        if(
            CMAKE_AR
            AND NOT "${CMAKE_AR}"
                MATCHES
                "NOTFOUND"
            AND (
                NOT CMAKE_${_tt_lang}_COMPILER_AR
                OR "${CMAKE_${_tt_lang}_COMPILER_AR}"
                    MATCHES
                    "NOTFOUND"
            )
        )
            set(CMAKE_${_tt_lang}_COMPILER_AR "${CMAKE_AR}" CACHE FILEPATH "Archiver for ${_tt_lang}" FORCE)
            set(CMAKE_${_tt_lang}_COMPILER_AR "${CMAKE_AR}")
        endif()
        if(
            CMAKE_RANLIB
            AND NOT "${CMAKE_RANLIB}"
                MATCHES
                "NOTFOUND"
            AND (
                NOT CMAKE_${_tt_lang}_COMPILER_RANLIB
                OR "${CMAKE_${_tt_lang}_COMPILER_RANLIB}"
                    MATCHES
                    "NOTFOUND"
            )
        )
            set(CMAKE_${_tt_lang}_COMPILER_RANLIB "${CMAKE_RANLIB}" CACHE FILEPATH "Ranlib for ${_tt_lang}" FORCE)
            set(CMAKE_${_tt_lang}_COMPILER_RANLIB "${CMAKE_RANLIB}")
        endif()
    endforeach()
    unset(_tt_lang)

    # Clang IPO archive templates may already cache NOTFOUND tools; restamp them for this configure.
    foreach(_tt_lang IN ITEMS C CXX)
        if(CMAKE_AR AND NOT "${CMAKE_AR}" MATCHES "NOTFOUND")
            set(CMAKE_${_tt_lang}_ARCHIVE_CREATE_IPO "\"${CMAKE_AR}\" qc <TARGET> <LINK_FLAGS> <OBJECTS>")
            set(CMAKE_${_tt_lang}_ARCHIVE_APPEND_IPO "\"${CMAKE_AR}\" q <TARGET> <LINK_FLAGS> <OBJECTS>")
        endif()
        if(CMAKE_RANLIB AND NOT "${CMAKE_RANLIB}" MATCHES "NOTFOUND")
            set(CMAKE_${_tt_lang}_ARCHIVE_FINISH_IPO "\"${CMAKE_RANLIB}\" <TARGET>")
        endif()
    endforeach()
    unset(_tt_lang)

    # Persist the same fix so later re-configures do not reintroduce NOTFOUND.
    foreach(_tt_lang_pair IN ITEMS "C;CMakeCCompiler.cmake" "CXX;CMakeCXXCompiler.cmake")
        list(GET _tt_lang_pair 0 _tt_lang)
        list(GET _tt_lang_pair 1 _tt_filename)
        file(GLOB _tt_compiler_info_files "${CMAKE_BINARY_DIR}/CMakeFiles/*/${_tt_filename}")
        foreach(_tt_compiler_info_file IN LISTS _tt_compiler_info_files)
            file(READ "${_tt_compiler_info_file}" _tt_txt)
            if(CMAKE_AR AND NOT "${CMAKE_AR}" MATCHES "NOTFOUND")
                string(
                    REPLACE
                    "set(CMAKE_${_tt_lang}_COMPILER_AR \"CMAKE_${_tt_lang}_COMPILER_AR-NOTFOUND\")"
                    "set(CMAKE_${_tt_lang}_COMPILER_AR \"${CMAKE_AR}\")"
                    _tt_txt
                    "${_tt_txt}"
                )
            endif()
            if(CMAKE_RANLIB AND NOT "${CMAKE_RANLIB}" MATCHES "NOTFOUND")
                string(
                    REPLACE
                    "set(CMAKE_${_tt_lang}_COMPILER_RANLIB \"CMAKE_${_tt_lang}_COMPILER_RANLIB-NOTFOUND\")"
                    "set(CMAKE_${_tt_lang}_COMPILER_RANLIB \"${CMAKE_RANLIB}\")"
                    _tt_txt
                    "${_tt_txt}"
                )
            endif()
            file(WRITE "${_tt_compiler_info_file}" "${_tt_txt}")
        endforeach()
    endforeach()
    unset(_tt_lang)
    unset(_tt_filename)
    unset(_tt_lang_pair)
    unset(_tt_compiler_info_file)
    unset(_tt_compiler_info_files)
    unset(_tt_txt)
endif()

add_subdirectory(tracy/csvexport)
add_subdirectory(tracy/capture)

# Restore global shared-lib preference after Tracy CLI configuration.
set(BUILD_SHARED_LIBS "${_tt_tracy_saved_build_shared_libs}" CACHE BOOL "Create shared libraries" FORCE)
unset(_tt_tracy_saved_build_shared_libs)

# Keep CLI output in the path used by Python tooling and CI packaging.
set(TRACY_CLI_BIN_DIR "${CMAKE_BINARY_DIR}/tools/profiler/bin")
set_target_properties(
    tracy-csvexport
    tracy-capture
    tracy-capture-daemon
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            "${TRACY_CLI_BIN_DIR}"
)

# Force a portable ISA baseline for Tracy server/CLI and key deps on x86_64.
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(_tt_tracy_portable_march -march=x86-64-v2)
    foreach(
        _tt_tracy_portable_target
        IN
        ITEMS
            TracyServer
            tracy-capture
            tracy-csvexport
            tracy-capture-daemon
            capstone
            libzstd
            libzstd_static
            PPQSort
            TracyGetOpt
    )
        if(NOT TARGET ${_tt_tracy_portable_target})
            continue()
        endif()
        get_target_property(_tt_tracy_portable_target_type ${_tt_tracy_portable_target} TYPE)
        if(_tt_tracy_portable_target_type STREQUAL "INTERFACE_LIBRARY")
            continue()
        endif()
        target_compile_options(${_tt_tracy_portable_target} PRIVATE ${_tt_tracy_portable_march})
    endforeach()
    unset(_tt_tracy_portable_march)
    unset(_tt_tracy_portable_target)
    unset(_tt_tracy_portable_target_type)
endif()

# Aggregate target used by CI/artifact workflows.
add_custom_target(
    tracy_profiler_cli_tools
    DEPENDS
        tracy-csvexport
        tracy-capture
)

# Tracy profiler WASM viewer (Emscripten).
# Fetch emsdk via CPM, but install toolchain artifacts under the build tree (not shared CPM cache).
# Note emsdk has to be installed the repo just has the scripts, not the toolchain artifacts.
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

# Keep output path aligned with Python tooling and CI packaging/tests.
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
