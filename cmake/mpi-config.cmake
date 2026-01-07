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

    # If MPI was already configured in this CMake run, don't recreate targets.
    # (This can happen if multiple subdirs include mpi-config.cmake.)
    if(TARGET OpenMPI::MPI)
        set(${use_mpi_out_var} TRUE PARENT_SCOPE)
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

    # Configure a sanitized MPI interface target.
    #
    # Root cause of the Fedora RPM failure:
    # - On some distros, CMake's FindMPI provides MPI usage requirements that include linker
    #   rpath flags (e.g. -Wl,-rpath,/usr/lib64/openmpi/lib).
    # - When those propagate (via our OBJECT library `distributed` into `tt_metal`), the
    #   final link line can embed RUNPATH as:
    #       /usr/lib64/openmpi/lib:$ORIGIN
    #   which violates Fedora brp-check-rpaths (it requires $ORIGIN to be first).
    #
    # For system MPI we do NOT need any RPATH entries at all; the loader finds MPI via
    # standard system paths. So we build our own interface target and explicitly strip
    # any -Wl,-rpath* flags from the MPI link requirements.
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
