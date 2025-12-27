# Compressing the debug sections cuts the size of ttnn.so by half for builds with debug info.
# Check for zstd compressed debug sections support (better compression ratio)
include(CheckCXXCompilerFlag)
include(CheckLinkerFlag)

check_cxx_compiler_flag(
    "-gz=zstd"
    COMPILER_SUPPORTS_GZ_ZSTD
)
check_linker_flag(
    CXX
    "-Wl,--compress-debug-sections=zstd"
    LINKER_SUPPORTS_COMPRESS_ZSTD
)

if(COMPILER_SUPPORTS_GZ_ZSTD AND LINKER_SUPPORTS_COMPRESS_ZSTD)
    message(STATUS "Using zstd compressed debug sections")
    add_compile_options(-gz=zstd)
    add_link_options(-Wl,--compress-debug-sections=zstd)
else()
    # Fallback to default -gz compression (typically zlib)
    check_cxx_compiler_flag(
        "-gz"
        COMPILER_SUPPORTS_GZ
    )
    check_linker_flag(
        CXX
        "-Wl,--compress-debug-sections=zlib"
        LINKER_SUPPORTS_COMPRESS_ZLIB
    )

    if(COMPILER_SUPPORTS_GZ AND LINKER_SUPPORTS_COMPRESS_ZLIB)
        message(STATUS "Using zlib compressed debug sections")
        add_compile_options(-gz)
        add_link_options(-Wl,--compress-debug-sections=zlib)
    else()
        message(STATUS "Compressed debug sections not supported, skipping")
    endif()
endif()

# Use mold by default if it is available and version >= 1.6 (LTO support).
# Fall back to LLD >= 17 if mold is unavailable or too old.
if(NOT DEFINED CMAKE_LINKER_TYPE)
    find_program(MOLD_EXECUTABLE ld.mold)
    if(MOLD_EXECUTABLE)
        # Get mold version (output format: "mold X.Y.Z (...)")
        execute_process(
            COMMAND
                ${MOLD_EXECUTABLE} --version
            OUTPUT_VARIABLE MOLD_VERSION_OUTPUT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(MOLD_VERSION_OUTPUT MATCHES "mold ([0-9]+\\.[0-9]+)")
            set(MOLD_VERSION "${CMAKE_MATCH_1}")
            if(MOLD_VERSION VERSION_GREATER_EQUAL "1.6")
                message(STATUS "Linker not specified. Using mold linker ${MOLD_VERSION}: ${MOLD_EXECUTABLE}")
                set(CMAKE_LINKER_TYPE MOLD)
            else()
                message(STATUS "mold ${MOLD_VERSION} found but version < 1.6 (no LTO support), checking for LLD")
            endif()
        else()
            message(STATUS "Could not determine mold version, checking for LLD")
        endif()
    endif()

    # If mold wasn't selected, try LLD >= 17
    if(NOT CMAKE_LINKER_TYPE)
        find_program(LLD_EXECUTABLE ld.lld)
        if(LLD_EXECUTABLE)
            # Get LLD version (output format: "LLD X.Y.Z (...)" or "Ubuntu LLD X.Y.Z (...)")
            execute_process(
                COMMAND
                    ${LLD_EXECUTABLE} --version
                OUTPUT_VARIABLE LLD_VERSION_OUTPUT
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(LLD_VERSION_OUTPUT MATCHES "LLD ([0-9]+)")
                set(LLD_VERSION "${CMAKE_MATCH_1}")
                if(LLD_VERSION VERSION_GREATER_EQUAL "17")
                    message(STATUS "Linker not specified. Using LLD linker ${LLD_VERSION}: ${LLD_EXECUTABLE}")
                    set(CMAKE_LINKER_TYPE LLD)
                else()
                    message(STATUS "LLD ${LLD_VERSION} found but version < 17, using default linker")
                endif()
            else()
                message(STATUS "Could not determine LLD version, using default linker")
            endif()
        else()
            message(STATUS "Neither mold >= 1.6 nor LLD >= 17 found, using default linker")
        endif()
    endif()
endif()

set(TT_LTO_ENABLED OFF)

if(TT_ENABLE_LTO)
    cmake_policy(SET CMP0069 NEW)
    set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)

    # LTO can be requested but not supported, so handle that case
    if(result)
        message(STATUS "LTO/IPO is supported. Enabling for Release/RelWithDebInfo builds.")

        # Do it this way to play nicer with ninja multi-config builds
        # This enables clang's thinLTO by default.
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)

        set(TT_LTO_ENABLED ON)

        # Check for fat LTO objects support (allows object files to be used with or without LTO)
        check_cxx_compiler_flag(
            "-ffat-lto-objects"
            COMPILER_SUPPORTS_FAT_LTO_OBJECTS
        )
        check_linker_flag(
            CXX
            "-ffat-lto-objects"
            LINKER_SUPPORTS_FAT_LTO_OBJECTS
        )
        if(COMPILER_SUPPORTS_FAT_LTO_OBJECTS AND LINKER_SUPPORTS_FAT_LTO_OBJECTS)
            message(STATUS "Using fat LTO objects")
            add_compile_options(-ffat-lto-objects)
            add_link_options(-ffat-lto-objects)
        endif()

        # Enable one-definition-rule (ODR) warnings for Release/RelWithDebInfo builds.
        # Only works when LTO is enabled.
        add_compile_options(
            $<$<CONFIG:Release,RelWithDebInfo>:-Wodr>
            $<$<CONFIG:Release,RelWithDebInfo>:-Wno-error=odr> # TODO(21850): Relaxing odr to warn until current errors are addressed.
        )

        # Make sure all thinLTO jobs are passed to the linker in bulk.
        # Mold won't properly parallelize thinLTO without it.
        add_link_options($<$<CXX_COMPILER_ID:Clang>:-Wl,--thinlto-jobs=all>)

        # Set up lto cache for faster incremental builds and a nominal limits to keep the size
        # from getting too large over time. A single everthing-enabled Release build results
        # in an lto-cache of ~250M. ~1.7G using RelWithDebInfo.
        add_link_options($<$<CXX_COMPILER_ID:Clang>:-Wl,--thinlto-cache-dir=${CMAKE_BINARY_DIR}/lto-cache>)

        # Limit size of lto cache to the least of 2GB, 10% disk space, or 10000 files.
        # By default pruning happens in 20 minute intervals. Add prune_interval to change.
        add_link_options(
            $<$<CXX_COMPILER_ID:Clang>:-Wl,--thinlto-cache-policy=cache_size_bytes=3g:cache_size=10%:cache_size_files=10000>
        )

        # Mozilla rec for non-pgo builds. Default is 100. pgo builds can get away with 5-10.
        # Reduces some excessive inlining bloat.
        add_link_options($<$<CXX_COMPILER_ID:Clang>:-Wl,-plugin-opt=-import-instr-limit=40>)

        # Enable virtual constant propagation and more aggressive vtable optimizations.
        # Adds some link time but still faster than flto=full
        # FIXME: -fforce-emit-vtables doesn't play nice with AnyIterator for some reason.
        #add_compile_options($<$<CXX_COMPILER_ID:Clang>:-fforce-emit-vtables>)
        add_link_options($<$<CXX_COMPILER_ID:Clang>:-fwhole-program-vtables>)
    else()
        message(WARNING "LTO/IPO is not supported: ${output}")
    endif()
endif()
