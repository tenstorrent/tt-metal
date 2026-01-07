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

# Disable brp-strip which fails on cross-compiled binaries for Tenstorrent hardware
# The strip command can't recognize the format of these non-host architecture files
#
# RPATH handling: tt_metal/CMakeLists.txt now correctly sets INSTALL_RPATH with $ORIGIN first,
# followed by the ULFM MPI path (when ULFM is enabled, which is only for Ubuntu builds).
# Fedora builds use system OpenMPI 5+ (with ULFM support) and do not include custom ULFM paths.
#
# TODO: Re-enable __brp_check_rpaths once all builds use system MPI only.
#       Tracking: Create a GitHub issue to track re-enabling this security check.
#       The issue should include:
#       - Verification that all builds (Ubuntu and Fedora) use only system MPI
#       - Removal of custom ULFM path from RPATH
#       - Testing that __brp_check_rpaths passes on all supported distros
#
# Currently disabled because:
#   1. The ULFM path (/opt/openmpi-v5.0.7-ulfm/lib) may not exist on the build host
#      (only on target systems for Ubuntu builds with custom ULFM)
#   2. System MPI paths added by find_package() may vary between build and target hosts
#   3. The check fails when RPATH contains paths that don't exist at build time
#
# Once all builds (including Ubuntu) use only system MPI (no custom ULFM paths in RPATH),
# the __brp_check_rpaths skip can be removed. This is a security/quality check that validates
# RPATHs in binaries and helps prevent security issues from hardcoded paths. It should be
# re-enabled as soon as possible.
set(CPACK_RPM_SPEC_MORE_DEFINE
    "%define __strip /bin/true
%define __brp_strip /bin/true
%define __brp_strip_comment_note /bin/true
%define __brp_strip_static_archive /bin/true
%define __brp_check_rpaths /bin/true
%define debug_package %{nil}"
)

# Exclude build-id files to avoid conflicts
set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
    /usr/lib/.build-id
    /usr/lib64/.build-id
)
