# CMake helper for detecting Linux distribution information
#
# CMake Requirements
# ------------------
# This module requires CMake 3.5+ for:
# - file(STRINGS) command improvements
# - execute_process() enhancements
#
# This file provides a function to detect distribution ID, ID_LIKE, and version
# from /etc/os-release, with fallback to lsb_release for version detection.
#
# Usage:
#   detect_distro()
#   # After calling, the following variables are set in the parent scope:
#   #   DISTRO_ID - Lowercase distribution ID (e.g., "fedora", "ubuntu", "debian")
#   #   DISTRO_ID_LIKE - Lowercase ID_LIKE value (e.g., "fedora", "debian ubuntu")
#   #   DISTRO_VERSION - Distribution version (e.g., "43", "22.04", "12")
#
# ⚠️  CROSS-COMPILATION WARNING ⚠️
# ==================================
# This function reads /etc/os-release from the BUILD HOST, not the target system.
# For cross-compilation scenarios where the target distro differs from the build host,
# the detected values will reflect the BUILD HOST's distribution, NOT the target.
#
# If you are cross-compiling, you should:
#   1. Override packaging type via -DTT_PACKAGING_TYPE=DEB or -DTT_PACKAGING_TYPE=RPM
#   2. Override version suffix logic if needed for your target distribution
#   3. Be aware that package version suffixes may be incorrect for the target system
#
# Example for cross-compilation:
#   cmake -DTT_PACKAGING_TYPE=RPM -DVERSION_RPM="1.0.0.el9" ...

function(detect_distro)
    set(DISTRO_ID "")
    set(DISTRO_ID_LIKE "")
    set(DISTRO_VERSION "")

    if(EXISTS "/etc/os-release")
        file(STRINGS "/etc/os-release" OS_RELEASE_CONTENTS)
        foreach(LINE ${OS_RELEASE_CONTENTS})
            # Extract ID
            if(LINE MATCHES "^ID=\"?([^\"]+)\"?")
                string(TOLOWER "${CMAKE_MATCH_1}" DISTRO_ID)
            endif()
            # Extract ID_LIKE
            if(LINE MATCHES "^ID_LIKE=\"?([^\"]+)\"?")
                string(TOLOWER "${CMAKE_MATCH_1}" DISTRO_ID_LIKE)
            endif()
            # Extract VERSION_ID
            if(LINE MATCHES "^VERSION_ID=\"?([^\"]+)\"?")
                set(DISTRO_VERSION "${CMAKE_MATCH_1}")
            endif()
        endforeach()
    endif()

    # For backwards compatibility, also check lsb_release for version
    if(NOT DISTRO_VERSION)
        execute_process(
            COMMAND
                lsb_release -sr
            OUTPUT_VARIABLE DISTRO_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()

    # Return values to parent scope
    set(DISTRO_ID "${DISTRO_ID}" PARENT_SCOPE)
    set(DISTRO_ID_LIKE "${DISTRO_ID_LIKE}" PARENT_SCOPE)
    set(DISTRO_VERSION "${DISTRO_VERSION}" PARENT_SCOPE)
endfunction()
