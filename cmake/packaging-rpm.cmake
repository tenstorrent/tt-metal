# RPM packaging configuration for RPM-based distros (Fedora, RHEL, openSUSE, etc.)

set(CPACK_GENERATOR RPM)

# Package groups (RPM-specific categorization)
set(CPACK_RPM_METALIUM_PACKAGE_GROUP "System Environment/Libraries")
set(CPACK_RPM_METALIUM-DEV_PACKAGE_GROUP "Development/Libraries")
set(CPACK_RPM_METALIUM-JIT_PACKAGE_GROUP "System Environment/Libraries")
set(CPACK_RPM_METALIUM-EXAMPLES_PACKAGE_GROUP "Documentation")
set(CPACK_RPM_METALIUM-VALIDATION_PACKAGE_GROUP "Applications/System")
set(CPACK_RPM_NN_PACKAGE_GROUP "System Environment/Libraries")
set(CPACK_RPM_NN-DEV_PACKAGE_GROUP "Development/Libraries")
set(CPACK_RPM_NN-EXAMPLES_PACKAGE_GROUP "Documentation")
set(CPACK_RPM_NN-VALIDATION_PACKAGE_GROUP "Applications/System")

set(CPACK_RPM_COMPONENT_INSTALL YES)

# VERSION_RPM is set by cmake/version.cmake with distro-specific suffix (e.g., .fc43)
set(CPACK_RPM_PACKAGE_VERSION "${VERSION_RPM}")
set(CPACK_RPM_FILE_NAME RPM-DEFAULT)

# Enable automatic dependency detection
set(CPACK_RPM_PACKAGE_AUTOREQ YES)
set(CPACK_RPM_PACKAGE_AUTOPROV YES)
# jit-build is cross compiling; autoreq does not find dependencies on the host; it should be self-contained anyway.
set(CPACK_RPM_METALIUM-JIT_PACKAGE_AUTOREQ NO)
set(CPACK_RPM_METALIUM-JIT_PACKAGE_AUTOPROV NO)
# ttml uses internal libraries from metalium/nn packages; dependencies are declared explicitly
set(CPACK_RPM_TTML_PACKAGE_AUTOREQ NO)
set(CPACK_RPM_TTML_PACKAGE_AUTOPROV NO)

# Package dependencies (using Fedora/RHEL package names)
set(CPACK_RPM_METALIUM-DEV_PACKAGE_REQUIRES "json-devel >= 3.10")
set(CPACK_RPM_NN-DEV_PACKAGE_REQUIRES "xtensor-devel >= 0.23.10")
set(CPACK_RPM_TTML_PACKAGE_REQUIRES "python3 >= 3.8")

# RPM-specific settings
set(CPACK_RPM_PACKAGE_LICENSE "Apache-2.0")
set(CPACK_RPM_PACKAGE_VENDOR "Tenstorrent")
set(CPACK_RPM_PACKAGE_URL "https://tenstorrent.com")

# Build the RPM spec macros based on build configuration
# Always disable strip for cross-compiled binaries (Tenstorrent hardware binaries)
set(_RPM_SPEC_MACROS
    "%define __strip /bin/true
%define __brp_strip /bin/true
%define __brp_strip_comment_note /bin/true
%define __brp_strip_static_archive /bin/true
%define debug_package %{nil}"
)

# Only disable __brp_check_rpaths when using custom ULFM MPI
# When using system OpenMPI (Fedora/RHEL with OpenMPI 5+), RPATH checking should pass
# because there are no custom paths like /opt/openmpi-v5.0.7-ulfm/lib
#
# TT_METAL_USING_ULFM is set by tt_metal/distributed/CMakeLists.txt:
#   - TRUE when custom ULFM MPI is found at ULFM_PREFIX (typically Ubuntu builds)
#   - FALSE when using system MPI (typically Fedora builds with OpenMPI 5+)
#
# See: https://docs.fedoraproject.org/en-US/packaging-guidelines/#_rpath_for_internal_libraries
if(TT_METAL_USING_ULFM)
    message(
        STATUS
        "RPM packaging: Disabling __brp_check_rpaths (custom ULFM MPI in use). "
        "The ULFM path ${ULFM_PREFIX}/lib may not exist on the build host."
    )
    set(_RPM_SPEC_MACROS
        "${_RPM_SPEC_MACROS}
%define __brp_check_rpaths /bin/true"
    )
else()
    message(STATUS "RPM packaging: RPATH checking enabled (using system MPI or MPI disabled)")
endif()

set(CPACK_RPM_SPEC_MORE_DEFINE "${_RPM_SPEC_MACROS}")

# Exclude build-id files to avoid conflicts
set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
    /usr/lib/.build-id
    /usr/lib64/.build-id
)
