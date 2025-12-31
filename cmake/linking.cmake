#[[
    linking.cmake - Linker configuration and optimization settings

    This module handles:
    - Debug section compression for smaller binaries
    - Automatic linker selection (mold > LLD > system ld)
    - Linker version detection and reporting
    - LTO (Link Time Optimization) configuration

    Usage: Include this file in your main CMakeLists.txt after project() declaration.
]]

include(CheckCXXCompilerFlag)
include(CheckLinkerFlag)

#===============================================================================
# FUNCTIONS
#===============================================================================

# Attempts to select mold linker if available and compatible (version >= 1.6)
function(select_mold_linker)
    find_program(MOLD_EXECUTABLE ld.mold)
    if(NOT MOLD_EXECUTABLE)
        return()
    endif()

    execute_process(
        COMMAND
            ${MOLD_EXECUTABLE} --version
        OUTPUT_VARIABLE mold_version_output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(NOT mold_version_output MATCHES "mold ([0-9.]+)")
        message(STATUS "Could not determine mold version, checking for LLD")
        return()
    endif()

    set(mold_version "${CMAKE_MATCH_1}")
    set(MOLD_VERSION_CACHED "${mold_version}" PARENT_SCOPE)

    if(mold_version VERSION_GREATER_EQUAL "1.6")
        message(STATUS "Linker not specified. Using mold linker ${mold_version}: ${MOLD_EXECUTABLE}")
        set(CMAKE_LINKER_TYPE MOLD PARENT_SCOPE)
    else()
        message(STATUS "mold ${mold_version} found but version < 1.6 (no LTO support), checking for LLD")
    endif()
endfunction()

# Attempts to select LLD linker if available and compatible (version >= 17)
function(select_lld_linker)
    find_program(LLD_EXECUTABLE ld.lld)
    if(NOT LLD_EXECUTABLE)
        message(STATUS "Neither mold >= 1.6 nor LLD >= 17 found, using default linker")
        return()
    endif()

    execute_process(
        COMMAND
            ${LLD_EXECUTABLE} --version
        OUTPUT_VARIABLE lld_version_output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(NOT lld_version_output MATCHES "LLD ([0-9.]+)")
        message(STATUS "Could not determine LLD version, using default linker")
        return()
    endif()

    set(lld_version "${CMAKE_MATCH_1}")
    set(LLD_VERSION_CACHED "${lld_version}" PARENT_SCOPE)

    if(lld_version VERSION_GREATER_EQUAL "17")
        message(STATUS "Linker not specified. Using LLD linker ${lld_version}: ${LLD_EXECUTABLE}")
        set(CMAKE_LINKER_TYPE LLD PARENT_SCOPE)
    else()
        message(STATUS "LLD ${lld_version} found but version < 17, using default linker")
    endif()
endfunction()

# Prints version information for a given linker
function(print_linker_version linker_executable regex_pattern use_string_regex)
    find_program(linker_exe ${linker_executable})
    if(NOT linker_exe)
        return()
    endif()

    execute_process(
        COMMAND
            ${linker_exe} --version
        OUTPUT_VARIABLE version_output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(use_string_regex)
        # Use string(REGEX MATCH) for GNU/BFD linker
        string(REGEX MATCH "${regex_pattern}" extracted_version "${version_output}")
        if(extracted_version)
            message(STATUS "Linker version: ${extracted_version}")
        else()
            message(STATUS "Linker version: ${version_output}")
        endif()
    else()
        # Use MATCHES for mold/LLD linkers
        if(version_output MATCHES "${regex_pattern}")
            message(STATUS "Linker version: ${CMAKE_MATCH_1}")
        else()
            message(STATUS "Linker version: ${version_output}")
        endif()
    endif()
endfunction()

#===============================================================================
# DEBUG SECTION COMPRESSION
#===============================================================================

# Check for zstd compressed debug sections support (better compression ratio)
check_cxx_compiler_flag(
    "-gz=zstd"
    COMPILER_SUPPORTS_GZ_ZSTD
)
check_cxx_compiler_flag(
    "-gz"
    COMPILER_SUPPORTS_GZ
)
check_linker_flag(
    CXX
    "-Wl,--compress-debug-sections=zstd"
    LINKER_SUPPORTS_COMPRESS_ZSTD
)
check_linker_flag(
    CXX
    "-Wl,--compress-debug-sections=zlib"
    LINKER_SUPPORTS_COMPRESS_ZLIB
)

if(COMPILER_SUPPORTS_GZ_ZSTD AND LINKER_SUPPORTS_COMPRESS_ZSTD)
    message(STATUS "Using zstd compressed debug sections")
    add_compile_options(-gz=zstd)
    add_link_options(-Wl,--compress-debug-sections=zstd)
elseif(COMPILER_SUPPORTS_GZ AND LINKER_SUPPORTS_COMPRESS_ZLIB)
    message(STATUS "Using zlib compressed debug sections")
    add_compile_options(-gz)
    add_link_options(-Wl,--compress-debug-sections=zlib)
else()
    message(STATUS "Compressed debug sections not supported, skipping")
    message(STATUS "Do you have the clang-tools package installed?")
endif()

#===============================================================================
# LINKER SELECTION
#===============================================================================

# Use mold by default if available and version >= 1.6 (LTO support).
# Fall back to LLD >= 17 if mold is unavailable or too old.
if(NOT DEFINED CMAKE_LINKER_TYPE)
    select_mold_linker()

    # If mold wasn't selected, try LLD >= 17
    if(NOT CMAKE_LINKER_TYPE)
        select_lld_linker()
    endif()
endif()

#===============================================================================
# LINKER VERSION REPORTING
#===============================================================================

# Print the linker being used and its version
if(CMAKE_LINKER_TYPE)
    message(STATUS "Linker type: ${CMAKE_LINKER_TYPE}")

    if(CMAKE_LINKER_TYPE STREQUAL "MOLD")
        # Reuse cached version from detection phase if available
        if(DEFINED MOLD_VERSION_CACHED)
            message(STATUS "Linker version: ${MOLD_VERSION_CACHED}")
        else()
            print_linker_version("ld.mold" "mold ([0-9.]+)" FALSE)
        endif()
    elseif(CMAKE_LINKER_TYPE STREQUAL "LLD")
        # Reuse cached version from detection phase if available
        if(DEFINED LLD_VERSION_CACHED)
            message(STATUS "Linker version: ${LLD_VERSION_CACHED}")
        else()
            print_linker_version("ld.lld" "LLD ([0-9.]+)" FALSE)
        endif()
    elseif(CMAKE_LINKER_TYPE MATCHES "^(GNU|BFD)$")
        print_linker_version("ld" "[0-9]+\\.[0-9]+[0-9.]*" TRUE)
    endif()
else()
    message(STATUS "Linker type: default (system ld)")
    print_linker_version("ld" "[0-9]+\\.[0-9]+[0-9.]*" TRUE)
endif()

#===============================================================================
# LINKER MISC CONFIGURATION
#===============================================================================
add_link_options($<$<CONFIG:Debug,RelWithDebInfo>:-Wl,--gdb-index>)

#===============================================================================
# LTO (LINK TIME OPTIMIZATION) CONFIGURATION
#===============================================================================

set(TT_LTO_ENABLED OFF)

if(TT_ENABLE_LTO)
    # Set IPO policy for better compatibility
    cmake_policy(SET CMP0069 NEW)
    set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

    include(CheckIPOSupported)
    check_ipo_supported(RESULT lto_supported OUTPUT lto_check_output)

    if(lto_supported)
        message(STATUS "LTO/IPO is supported. Enabling for Release/RelWithDebInfo builds.")

        # Enable IPO for Release and RelWithDebInfo builds
        # This plays nicely with ninja multi-config builds and enables clang's thinLTO by default
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

        add_compile_options(
            $<$<CONFIG:Release>:-ffunction-sections>
            $<$<CONFIG:Release>:-fdata-sections>
        )
        add_link_options(
            $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:Clang>>:-Wl,--icf=safe>
            $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:Clang>>:-Wl,--ignore-data-address-equality>
            $<$<CONFIG:Release>:-Wl,--gc-sections>
        )

        # Enable one-definition-rule (ODR) warnings for optimized builds
        # Only works when LTO is enabled
        add_compile_options(
            $<$<CONFIG:Release,RelWithDebInfo>:-Wodr>
            $<$<CONFIG:Release,RelWithDebInfo>:-Wstrict-aliasing>
            $<$<AND:$<CONFIG:Release,RelWithDebInfo>,$<CXX_COMPILER_ID:GNU>>:-Wlto-type-mismatch>
        )

        # Ensure all thinLTO jobs are passed to the linker in bulk
        # Mold won't properly parallelize thinLTO without this
        add_link_options($<$<CXX_COMPILER_ID:Clang>:-Wl,--thinlto-jobs=all>)

        # Set up LTO cache for faster incremental builds
        # Single Release build results in ~250M cache, RelWithDebInfo ~1.7G
        add_link_options($<$<CXX_COMPILER_ID:Clang>:-Wl,--thinlto-cache-dir=${CMAKE_BINARY_DIR}/lto-cache>)

        # Limit LTO cache size to the least of 2GB, 10% disk space, or 10000 files
        # Default pruning happens every 20 minutes
        add_link_options(
            $<$<CXX_COMPILER_ID:Clang>:-Wl,--thinlto-cache-policy=cache_size_bytes=2g:cache_size=10%:cache_size_files=10000>
        )

        # Mozilla recommendation for non-PGO builds (default is 100)
        # PGO builds can use 5-10. Reduces excessive inlining bloat
        add_link_options($<$<CXX_COMPILER_ID:Clang>:-Wl,-plugin-opt=-import-instr-limit=40>)

        # Enable virtual constant propagation and aggressive vtable optimizations
        # Adds some link time but still faster than -flto=full
        # FIXME: -fforce-emit-vtables doesn't play nice with AnyIterator for some reason
        # add_compile_options($<$<CXX_COMPILER_ID:Clang>:-fforce-emit-vtables>)
        add_link_options($<$<CXX_COMPILER_ID:Clang>:-fwhole-program-vtables>)
    else()
        message(WARNING "LTO/IPO is not supported: ${lto_check_output}")
    endif()
endif()
