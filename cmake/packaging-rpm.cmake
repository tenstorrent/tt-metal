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

# Use VERSION_RPM if defined, otherwise fall back to project version
if(DEFINED VERSION_RPM)
    set(CPACK_RPM_PACKAGE_VERSION "${VERSION_RPM}")
else()
    set(CPACK_RPM_PACKAGE_VERSION "${PROJECT_VERSION}")
endif()
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

# Exclude build-id files to avoid conflicts
set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION /usr/lib/.build-id)
