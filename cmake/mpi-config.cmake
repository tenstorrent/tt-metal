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

# Macro to configure MPI support
# Checks for custom ULFM or system MPI, sets up the OpenMPI::MPI target,
# and defines USE_MPI and TT_METAL_USING_ULFM variables.
#
# Arguments:
#   enable_distributed: Whether distributed compute is enabled (ON/OFF)
#   use_mpi_out_var: Variable to store the result (TRUE/FALSE) - set in PARENT_SCOPE if calling from function
#   extra_src_out_var: Variable to append source files to (e.g. mpi_distributed_context.cpp)
macro(tt_configure_mpi enable_distributed use_mpi_out_var extra_src_out_var)
    set(${use_mpi_out_var} FALSE)
    set(TT_METAL_USING_ULFM FALSE PARENT_SCOPE)

    # Early exit if distributed compute is disabled
    if(NOT ${enable_distributed})
        message(STATUS "Multihost compute with MPI disabled, using single host context.")
        list(APPEND ${extra_src_out_var} ${CMAKE_CURRENT_SOURCE_DIR}/multihost/single_host_context.cpp)
        return()
    endif()

    # Try Custom ULFM MPI first (Ubuntu)
    if(EXISTS "${ULFM_PREFIX}/lib/libmpi.so.40")
        message(STATUS "Using ULFM MPI from ${ULFM_PREFIX}")
        set(TT_METAL_USING_ULFM TRUE PARENT_SCOPE)
        set(${use_mpi_out_var} TRUE)
        list(APPEND ${extra_src_out_var} ${CMAKE_CURRENT_SOURCE_DIR}/multihost/mpi_distributed_context.cpp)

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

    set(${use_mpi_out_var} TRUE)
    list(APPEND ${extra_src_out_var} ${CMAKE_CURRENT_SOURCE_DIR}/multihost/mpi_distributed_context.cpp)

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

    # Export MPI lib dir for RPATH construction in tt_metal (Fedora fix)
    get_target_property(_mpi_loc MPI::MPI_C IMPORTED_LOCATION)
    if(NOT _mpi_loc AND MPI_C_LIBRARIES)
        list(GET MPI_C_LIBRARIES 0 _mpi_loc)
    endif()

    if(EXISTS "${_mpi_loc}")
        get_filename_component(_mpi_dir "${_mpi_loc}" DIRECTORY)
        set(TT_METAL_MPI_LIB_DIR "${_mpi_dir}" PARENT_SCOPE)
    endif()
endmacro()
