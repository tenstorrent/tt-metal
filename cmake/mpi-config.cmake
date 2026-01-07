# MPI configuration for TT-Metalium
#
# This file centralizes the MPI configuration to avoid duplication
# between tt_metal/CMakeLists.txt and tt_metal/distributed/CMakeLists.txt.
#
# ULFM (User-Level Failure Mitigation) is an extension to OpenMPI that
# provides fault tolerance features. OpenMPI 5.0+ includes ULFM by default,
# while earlier versions require a custom build.
#
# For Ubuntu builds: Custom ULFM build at /opt/openmpi-v5.0.7-ulfm
# For Fedora builds: System OpenMPI 5+ (includes ULFM natively)

# Default ULFM prefix - can be overridden via -DULFM_PREFIX=/path/to/ulfm
# This is where the custom ULFM MPI build is installed (typically on Ubuntu)
if(NOT DEFINED ULFM_PREFIX)
    set(ULFM_PREFIX "/opt/openmpi-v5.0.7-ulfm" CACHE PATH "Path to ULFM MPI installation")
endif()

# Function to configure MPI library support
# Checks for custom ULFM or system MPI, sets up the OpenMPI::MPI target,
# and defines USE_MPI and TT_METAL_USING_ULFM variables.
#
# Arguments:
#   enable_distributed: Whether distributed compute is enabled (ON/OFF)
#   use_mpi_out_var: Variable to store the result (TRUE/FALSE) - set in PARENT_SCOPE
function(tt_configure_mpi enable_distributed use_mpi_out_var)
    set(${use_mpi_out_var} FALSE PARENT_SCOPE)
    set(TT_METAL_USING_ULFM FALSE PARENT_SCOPE)

    # Early exit if distributed compute is disabled
    if(NOT ${enable_distributed})
        return()
    endif()

    # Try Custom ULFM MPI first (Ubuntu)
    if(EXISTS "${ULFM_PREFIX}/lib/libmpi.so.40")
        message(STATUS "Using ULFM MPI from ${ULFM_PREFIX}")
        set(TT_METAL_USING_ULFM TRUE PARENT_SCOPE)
        set(${use_mpi_out_var} TRUE PARENT_SCOPE)

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

    # Try System MPI (Fedora/Others)
    find_package(MPI QUIET COMPONENTS C)
    if(NOT MPI_FOUND)
        message(FATAL_ERROR "ENABLE_DISTRIBUTED is ON but no MPI implementation found.")
    endif()

    set(${use_mpi_out_var} TRUE PARENT_SCOPE)

    # Check for OpenMPI version and warn if < 5 (missing ULFM)
    if(MPI_C_LIBRARY_VERSION_STRING MATCHES "Open MPI")
        string(REGEX MATCH "Open MPI v([0-9]+)" _ompi_match "${MPI_C_LIBRARY_VERSION_STRING}")
        if(_ompi_match AND CMAKE_MATCH_1 GREATER_EQUAL 5)
            message(STATUS "Using system OpenMPI ${CMAKE_MATCH_1}.x (likely supports ULFM)")
        elseif(_ompi_match)
            message(WARNING "System OpenMPI ${CMAKE_MATCH_1}.x found but ULFM support requires version 5+")
        endif()
    else()
        message(WARNING "Non-OpenMPI implementation found. ULFM fault tolerance requires OpenMPI 5+")
    endif()

    # Configure system MPI target
    add_library(OpenMPI::MPI ALIAS MPI::MPI_C)

    # Note: We do NOT export the MPI library directory for RPATH construction because:
    # 1. For Fedora builds, system MPI libraries are in standard paths (/usr/lib64 or /usr/lib64/openmpi/lib)
    #    and will be found via the standard library search path, not via RPATH
    # 2. RPM's brp-check-rpaths requires $ORIGIN to be first in RPATH, and adding system paths violates this
    # 3. The MPI library should be found via the system's library search mechanism, not via RPATH
    # 4. Even with INSTALL_RPATH_USE_LINK_PATH=FALSE, CMake might still add library directories from
    #    imported targets' INTERFACE_LINK_DIRECTORIES, so we rely on system library search instead
endfunction()
