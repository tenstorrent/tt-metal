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

# Function to check if custom ULFM MPI is available
# Returns TRUE if the ULFM library exists at ULFM_PREFIX, FALSE otherwise
function(check_ulfm_available RESULT_VAR)
    if(EXISTS "${ULFM_PREFIX}/lib/libmpi.so.40")
        set(${RESULT_VAR} TRUE PARENT_SCOPE)
    else()
        set(${RESULT_VAR} FALSE PARENT_SCOPE)
    endif()
endfunction()
