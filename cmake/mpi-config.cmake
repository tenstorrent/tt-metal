# =============================================================================
# MPI Configuration for TT-Metalium
# =============================================================================
#
# CMake Requirements
# ------------------
# This module requires CMake 3.15+ for:
# - GLOBAL IMPORTED targets (CMake 3.11+)
# - Improved find_package() behavior (CMake 3.15+)
#
# Overview
# --------
# This module configures MPI support for distributed computing. It handles:
# - Detection of custom ULFM MPI (Ubuntu) or system OpenMPI (Fedora)
# - Creation of the OpenMPI::MPI target for linking
# - Extraction of MPI library paths for RPATH configuration
# - Sanitization of MPI link flags to prevent RPATH pollution
#
# Why Call From tt_metal/CMakeLists.txt Instead of distributed/CMakeLists.txt?
# ----------------------------------------------------------------------------
# tt_configure_mpi() sets variables via PARENT_SCOPE, which only propagates
# one level up. If called from distributed/CMakeLists.txt:
#
#   tt_metal/CMakeLists.txt
#   └── add_subdirectory(distributed)
#       └── tt_configure_mpi()  # Sets vars in distributed/ scope
#           ↑ PARENT_SCOPE goes here, NOT to tt_metal/
#
# But tt_metal/CMakeLists.txt needs TT_METAL_MPI_LIB_DIR for RPATH configuration.
# CMake doesn't automatically propagate variables from child to parent scope.
#
# Solution: Call tt_configure_mpi() from tt_metal/CMakeLists.txt BEFORE
# add_subdirectory(distributed). Variables are set in tt_metal/ scope and
# automatically inherited by distributed/ (CMake propagates DOWN, not UP).
#
#   tt_metal/CMakeLists.txt
#   ├── tt_configure_mpi()      # Sets vars in tt_metal/ scope
#   └── add_subdirectory(distributed)  # Inherits vars from parent
#
# MPI Implementations
# -------------------
# - Ubuntu: Custom ULFM build at /opt/openmpi-v5.0.7-ulfm
#   - ULFM (User-Level Failure Mitigation) provides fault tolerance
#   - TT_METAL_USING_ULFM=TRUE, TT_METAL_MPI_LIB_DIR=/opt/.../lib
#
# - Fedora: System OpenMPI 5+ (includes ULFM natively)
#   - MPI found via find_package(MPI)
#   - TT_METAL_USING_ULFM=FALSE, TT_METAL_MPI_LIB_DIR=/usr/lib64/openmpi/lib
#
# Output Variables (set in PARENT_SCOPE)
# --------------------------------------
# - USE_MPI: TRUE if MPI is available and enabled
# - TT_METAL_USING_ULFM: TRUE if using custom ULFM build (Ubuntu)
# - TT_METAL_MPI_LIB_DIR: Directory containing libmpi.so (for RPATH)
# - OpenMPI::MPI: Target for linking against MPI
#
# Why TT_METAL_MPI_LIB_DIR?
# -------------------------
# tt_metal uses BUILD_WITH_INSTALL_RPATH=TRUE (see cmake/packaging.cmake for why).
# This means INSTALL_RPATH is embedded at build time. For tests to find libmpi.so,
# the MPI library directory must be in INSTALL_RPATH. TT_METAL_MPI_LIB_DIR provides
# this path, which is added to tt_metal's INSTALL_RPATH after $ORIGIN.
#
# Debugging Tips
# --------------
# - Check CMake output for "Using ULFM MPI from" or "MPI library directory:"
# - Verify RPATH with: readelf -d build/lib/libtt_metal.so | grep -i path
# - Test MPI linking: ldd build/lib/libtt_metal.so | grep mpi
# - If libmpi.so.40 not found, check TT_METAL_MPI_LIB_DIR is in RPATH
#
# =============================================================================

# Default ULFM prefix - can be overridden via -DULFM_PREFIX=/path/to/ulfm
if(NOT DEFINED ULFM_PREFIX)
    set(ULFM_PREFIX "/opt/openmpi-v5.0.7-ulfm" CACHE PATH "Path to ULFM MPI installation")
endif()
# -----------------------------------------------------------------------------
# tt_configure_mpi - Configure MPI support and create OpenMPI::MPI target
# -----------------------------------------------------------------------------
# Arguments:
#   enable_distributed: Whether distributed compute is enabled (ENABLE_DISTRIBUTED)
#   use_mpi_out_var: Output variable name for MPI availability (e.g., USE_MPI)
#
# Sets in PARENT_SCOPE:
#   ${use_mpi_out_var}: TRUE if MPI is configured, FALSE otherwise
#   TT_METAL_USING_ULFM: TRUE if using custom ULFM build
#   TT_METAL_MPI_LIB_DIR: Directory containing libmpi.so (for RPATH)
#
# Creates:
#   OpenMPI::MPI: Target for linking against MPI
#
function(tt_configure_mpi enable_distributed use_mpi_out_var)
    # Initialize output variables (will be overwritten if MPI is found)
    set(${use_mpi_out_var} FALSE PARENT_SCOPE)
    set(TT_METAL_USING_ULFM FALSE PARENT_SCOPE)
    set(TT_METAL_MPI_LIB_DIR "" PARENT_SCOPE)

    if(NOT ${enable_distributed})
        message(STATUS "Distributed compute disabled, skipping MPI configuration")
        return()
    endif()

    # Guard against multiple calls (e.g., if included from multiple places)
    # The OpenMPI::MPI target is GLOBAL, so it persists across the entire build
    if(TARGET OpenMPI::MPI)
        message(DEBUG "OpenMPI::MPI target already exists, skipping reconfiguration")
        set(${use_mpi_out_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # =========================================================================
    # Strategy 1: Custom ULFM MPI (Ubuntu CI builds)
    # =========================================================================
    # Ubuntu doesn't have OpenMPI 5+ in repos, so CI uses a custom ULFM build.
    # We check for this FIRST because it takes precedence over system MPI.
    #
    if(EXISTS "${ULFM_PREFIX}/lib/libmpi.so.40")
        message(STATUS "Using ULFM MPI from ${ULFM_PREFIX}")
        set(TT_METAL_USING_ULFM TRUE PARENT_SCOPE)
        set(TT_METAL_MPI_LIB_DIR "${ULFM_PREFIX}/lib" PARENT_SCOPE)
        set(${use_mpi_out_var} TRUE PARENT_SCOPE)

        # Create IMPORTED target pointing directly to the ULFM library
        # GLOBAL makes it visible across the entire build (not just this directory)
        add_library(OpenMPI::MPI SHARED IMPORTED GLOBAL)
        set_target_properties(
            OpenMPI::MPI
            PROPERTIES
                EXCLUDE_FROM_ALL
                    TRUE
                IMPORTED_LOCATION
                    "${ULFM_PREFIX}/lib/libmpi.so.40"
                INTERFACE_INCLUDE_DIRECTORIES
                    "${ULFM_PREFIX}/include"
        )
        return()
    endif()

    # =========================================================================
    # Strategy 2: System MPI (Fedora, or Ubuntu with system OpenMPI)
    # =========================================================================
    # Use CMake's FindMPI to locate system-installed MPI.
    # On Fedora, this finds /usr/lib64/openmpi/lib/libmpi.so
    #
    find_package(MPI QUIET COMPONENTS C)
    if(NOT MPI_FOUND)
        message(
            FATAL_ERROR
            "ENABLE_DISTRIBUTED is ON but no MPI implementation found.\n"
            "  - Ubuntu: Install custom ULFM to ${ULFM_PREFIX}, or\n"
            "  - Fedora: Install openmpi-devel package"
        )
    endif()

    set(${use_mpi_out_var} TRUE PARENT_SCOPE)

    # Extract MPI library directory for RPATH
    # Why needed: tt_metal uses BUILD_WITH_INSTALL_RPATH=TRUE, so INSTALL_RPATH
    # is embedded at build time. Tests need the MPI library directory in RPATH
    # to find libmpi.so at runtime without setting LD_LIBRARY_PATH.
    #
    # MPI_C_LIBRARIES typically contains: /usr/lib64/openmpi/lib/libmpi.so
    # We extract the directory: /usr/lib64/openmpi/lib
    #
    if(MPI_C_LIBRARIES)
        list(GET MPI_C_LIBRARIES 0 _first_mpi_lib)
        # Skip if it's empty, a flag (starts with -), or not a valid path
        if(_first_mpi_lib AND NOT _first_mpi_lib STREQUAL "" AND NOT _first_mpi_lib MATCHES "^-")
            get_filename_component(_mpi_lib_dir "${_first_mpi_lib}" DIRECTORY)
            if(_mpi_lib_dir AND NOT _mpi_lib_dir STREQUAL "")
                set(TT_METAL_MPI_LIB_DIR "${_mpi_lib_dir}" PARENT_SCOPE)
                message(STATUS "MPI library directory: ${_mpi_lib_dir}")
            endif()
        endif()
    endif()

    # Check for OpenMPI version and warn if < 5 (missing ULFM)
    # Handle cases where MPI_C_LIBRARY_VERSION_STRING might be empty or malformed
    if(
        MPI_C_LIBRARY_VERSION_STRING
        AND NOT MPI_C_LIBRARY_VERSION_STRING
            STREQUAL
            ""
        AND MPI_C_LIBRARY_VERSION_STRING
            MATCHES
            "Open MPI"
    )
        string(REGEX MATCH "Open MPI v([0-9]+)" _ompi_match "${MPI_C_LIBRARY_VERSION_STRING}")
        if(_ompi_match AND CMAKE_MATCH_1 AND NOT CMAKE_MATCH_1 STREQUAL "")
            # CMAKE_MATCH_1 contains the major version number as a string
            # CMake's numeric comparison works with numeric strings, but we validate it's a number
            if(CMAKE_MATCH_1 MATCHES "^[0-9]+$")
                # Compare as numeric (CMake handles string-to-number conversion)
                if(CMAKE_MATCH_1 GREATER_EQUAL 5)
                    message(STATUS "Using system OpenMPI ${CMAKE_MATCH_1}.x (likely supports ULFM)")
                else()
                    message(WARNING "System OpenMPI ${CMAKE_MATCH_1}.x found but ULFM support requires version 5+")
                endif()
            else()
                message(DEBUG "Could not parse numeric version from: ${CMAKE_MATCH_1}")
            endif()
        else()
            message(DEBUG "Could not parse OpenMPI version from: ${MPI_C_LIBRARY_VERSION_STRING}")
        endif()
    elseif(MPI_C_LIBRARY_VERSION_STRING AND NOT MPI_C_LIBRARY_VERSION_STRING STREQUAL "")
        message(
            WARNING
            "Non-OpenMPI implementation found (${MPI_C_LIBRARY_VERSION_STRING}). ULFM fault tolerance requires OpenMPI 5+"
        )
    else()
        message(DEBUG "MPI version string not available, skipping version check")
    endif()

    # =========================================================================
    # Create Sanitized MPI Interface Target
    # =========================================================================
    # Why sanitize? CMake's FindMPI on some distros includes -Wl,-rpath flags
    # that pollute our RPATH. For example, Fedora's FindMPI adds:
    #
    #   -Wl,-rpath,/usr/lib64/openmpi/lib
    #
    # When this propagates through our build (distributed -> tt_metal), the
    # final binary gets RUNPATH like:
    #
    #   /usr/lib64/openmpi/lib:$ORIGIN
    #
    # This FAILS Fedora's brp-check-rpaths which requires $ORIGIN to be FIRST.
    # Error: "the special '$ORIGIN' RPATHs are appearing after other RPATHs"
    #
    # Solution: Create our own interface target that strips -Wl,-rpath* flags.
    # The MPI library is found via:
    # - Standard library search path (system MPI in /usr/lib64), OR
    # - TT_METAL_MPI_LIB_DIR in our controlled INSTALL_RPATH
    #
    add_library(tt_openmpi_mpi INTERFACE)
    add_library(OpenMPI::MPI ALIAS tt_openmpi_mpi)

    if(DEFINED MPI_C_INCLUDE_DIRS)
        target_include_directories(tt_openmpi_mpi SYSTEM INTERFACE ${MPI_C_INCLUDE_DIRS})
    endif()
    if(DEFINED MPI_C_COMPILE_DEFINITIONS)
        target_compile_definitions(tt_openmpi_mpi INTERFACE ${MPI_C_COMPILE_DEFINITIONS})
    endif()
    if(DEFINED MPI_C_COMPILE_OPTIONS)
        target_compile_options(tt_openmpi_mpi INTERFACE ${MPI_C_COMPILE_OPTIONS})
    endif()

    # Sanitize MPI link flags/options
    set(_mpi_link_opts ${MPI_C_LINK_FLAGS})
    if(_mpi_link_opts)
        # If FindMPI gave us a single string with spaces, split it.
        if("${_mpi_link_opts}" MATCHES " ")
            separate_arguments(_mpi_link_opts)
        endif()
        set(_mpi_link_opts_clean "")
        set(_skip_next FALSE)
        foreach(_opt IN LISTS _mpi_link_opts)
            if(_skip_next)
                set(_skip_next FALSE)
                continue()
            endif()
            if(_opt STREQUAL "-Wl,-rpath" OR _opt STREQUAL "-Wl,-rpath-link")
                set(_skip_next TRUE)
                continue()
            endif()
            if(_opt MATCHES "^-Wl,-rpath(,|=)" OR _opt MATCHES "^-Wl,-rpath-link(,|=)")
                continue()
            endif()
            list(APPEND _mpi_link_opts_clean "${_opt}")
        endforeach()
        if(_mpi_link_opts_clean)
            target_link_options(tt_openmpi_mpi INTERFACE ${_mpi_link_opts_clean})
        endif()
    endif()

    # Sanitize MPI libraries list (some FindMPI implementations put -Wl,-rpath* here)
    set(_mpi_libs ${MPI_C_LIBRARIES})
    set(_mpi_libs_clean "")
    set(_skip_next FALSE)
    foreach(_lib IN LISTS _mpi_libs)
        if(_skip_next)
            set(_skip_next FALSE)
            continue()
        endif()
        if(_lib STREQUAL "-Wl,-rpath" OR _lib STREQUAL "-Wl,-rpath-link")
            set(_skip_next TRUE)
            continue()
        endif()
        if(_lib MATCHES "^-Wl,-rpath(,|=)" OR _lib MATCHES "^-Wl,-rpath-link(,|=)")
            continue()
        endif()
        list(APPEND _mpi_libs_clean "${_lib}")
    endforeach()
    if(_mpi_libs_clean)
        target_link_libraries(tt_openmpi_mpi INTERFACE ${_mpi_libs_clean})
    endif()
endfunction()
