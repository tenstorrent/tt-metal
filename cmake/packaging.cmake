# TT-Metal CPack packaging configuration
# Automatically detects distro type and includes appropriate DEB or RPM packaging
#
# =============================================================================
# RPATH / RUNPATH and Library Layout
# =============================================================================
#
# This project uses $ORIGIN-based RPATH to make binaries relocatable. $ORIGIN is
# a special token in ELF binaries that the dynamic linker (ld.so) expands at
# runtime to the directory containing the executable or library itself.
#
# For example, if /work/build/tt-train/tests/ttml_tests has RPATH=$ORIGIN/../../lib,
# then at runtime $ORIGIN becomes /work/build/tt-train/tests, and the linker
# searches /work/build/tt-train/tests/../../lib -> /work/build/lib for libraries.
#
# CI Tar Artifact Layout
# ----------------------
# When the CI tar artifact (ttm_any.tar.zst) is extracted to /work/:
#
#   /work/
#   ├── ttnn/
#   │   └── ttnn/
#   │       ├── _ttnn.so          <- Python bindings (nanobind)
#   │       └── _ttnncpp.so       <- C++ library with Python bindings
#   │
#   ├── build/
#   │   ├── lib/
#   │   │   ├── libtt_metal.so    <- Core TT-Metal library
#   │   │   ├── libtracy.so       <- Tracy profiler (optional)
#   │   │   └── ...
#   │   │
#   │   ├── tt-train/
#   │   │   └── tests/
#   │   │       └── ttml_tests    <- tt-train test executable
#   │   │
#   │   └── test/
#   │       └── ...               <- Other test executables
#   │
#   └── runtime/
#       └── ...                   <- Runtime files (kernels, firmware, etc.)
#
# Library Dependency Chain
# ------------------------
#   ttml_tests -> _ttnncpp.so (in ttnn/ttnn/) -> libtt_metal.so (in build/lib/)
#
# RPATH Configuration
# -------------------
# - _ttnncpp.so and _ttnn.so (in ttnn/ttnn/):
#     INSTALL_RPATH = "$ORIGIN/../../build/lib;$ORIGIN"
#     - $ORIGIN/../../build/lib resolves to build/lib/ from ttnn/ttnn/
#     - $ORIGIN allows finding sibling libraries in the same directory
#
# - tt-train executables (in build/tt-train/tests/):
#     CMAKE_INSTALL_RPATH includes:
#     - $ORIGIN/../../lib           -> build/lib/ (for libtt_metal.so)
#     - $ORIGIN/../../../ttnn/ttnn  -> ttnn/ttnn/ (for _ttnncpp.so)
#
# FHS Package Layout (DEB/RPM)
# ----------------------------
# For proper system packages, all libraries are installed to standard locations
# (e.g., /usr/lib64/), so $ORIGIN alone is sufficient since libraries are
# co-located in the same directory.
#
# Why $ORIGIN instead of LD_LIBRARY_PATH?
# ---------------------------------------
# - Embedded in the binary (no environment setup needed)
# - Works in containers without special configuration
# - Portable across different execution contexts
# - More secure (controlled library search path)
#
# =============================================================================

set(CPACK_PACKAGE_CONTACT "support@tenstorrent.com")
set(CMAKE_PROJECT_HOMEPAGE_URL "https://tenstorrent.com")
set(CPACK_PACKAGE_NAME tt)

# Suppress the summary so that we can have per-component summaries
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "")

# Use project config file to defer build-type-specific configuration to packaging time
# This is necessary for multi-config generators.
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/packaging.d/cpack-project-config.cmake.in"
    "${PROJECT_BINARY_DIR}/cpack-project-config.cmake"
    @ONLY
)
set(CPACK_PROJECT_CONFIG_FILE "${PROJECT_BINARY_DIR}/cpack-project-config.cmake")

set(CPACK_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE
)

# Detect package manager type: DEB vs RPM
# Check for /etc/os-release to determine distro family
set(TT_PACKAGING_TYPE "")

if(EXISTS "/etc/os-release")
    file(STRINGS "/etc/os-release" OS_RELEASE_CONTENTS)
    foreach(LINE ${OS_RELEASE_CONTENTS})
        # Check ID_LIKE first (more reliable for derivatives)
        if(LINE MATCHES "^ID_LIKE=(.*)$")
            string(TOLOWER "${CMAKE_MATCH_1}" ID_LIKE_VALUE)
            # Remove quotes if present
            string(REPLACE "\"" "" ID_LIKE_VALUE "${ID_LIKE_VALUE}")
            if(ID_LIKE_VALUE MATCHES "debian|ubuntu")
                set(TT_PACKAGING_TYPE "DEB")
            elseif(ID_LIKE_VALUE MATCHES "fedora|rhel|centos|suse")
                set(TT_PACKAGING_TYPE "RPM")
            endif()
        endif()
        # Check ID if ID_LIKE didn't match
        if(TT_PACKAGING_TYPE STREQUAL "" AND LINE MATCHES "^ID=(.*)$")
            string(TOLOWER "${CMAKE_MATCH_1}" ID_VALUE)
            # Remove quotes if present
            string(REPLACE "\"" "" ID_VALUE "${ID_VALUE}")
            if(ID_VALUE MATCHES "debian|ubuntu|linuxmint|pop")
                set(TT_PACKAGING_TYPE "DEB")
            elseif(ID_VALUE MATCHES "fedora|rhel|centos|rocky|alma|opensuse|sles")
                set(TT_PACKAGING_TYPE "RPM")
            endif()
        endif()
    endforeach()
endif()

# Fallback: check for package manager executables
if(TT_PACKAGING_TYPE STREQUAL "")
    find_program(DPKG_EXECUTABLE dpkg)
    find_program(RPM_EXECUTABLE rpm)
    if(DPKG_EXECUTABLE)
        set(TT_PACKAGING_TYPE "DEB")
    elseif(RPM_EXECUTABLE)
        set(TT_PACKAGING_TYPE "RPM")
    endif()
endif()

# Allow override via command line: -DTT_PACKAGING_TYPE=DEB or -DTT_PACKAGING_TYPE=RPM
if(DEFINED CACHE{TT_PACKAGING_TYPE} AND NOT TT_PACKAGING_TYPE STREQUAL "")
    # User override takes precedence
    set(TT_PACKAGING_TYPE "${TT_PACKAGING_TYPE}")
endif()

# Default to DEB if detection failed
if(TT_PACKAGING_TYPE STREQUAL "")
    message(WARNING "Could not detect distro package type, defaulting to DEB")
    set(TT_PACKAGING_TYPE "DEB")
endif()

message(STATUS "Packaging type: ${TT_PACKAGING_TYPE}")

# Include the appropriate packaging configuration
if(TT_PACKAGING_TYPE STREQUAL "RPM")
    include(${CMAKE_CURRENT_LIST_DIR}/packaging-rpm.cmake)
else()
    include(${CMAKE_CURRENT_LIST_DIR}/packaging-deb.cmake)
endif()

# Common CMake package config helpers
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/tt-metalium-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/packaging.d/tt-metalium-config.cmake.in
    ${PROJECT_BINARY_DIR}/tt-metalium-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tt-metalium
)
install(
    FILES
        ${PROJECT_BINARY_DIR}/tt-metalium-config.cmake
        ${PROJECT_BINARY_DIR}/tt-metalium-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tt-metalium
    COMPONENT metalium-dev
)

write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/tt-nn-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/packaging.d/tt-nn-config.cmake.in
    ${PROJECT_BINARY_DIR}/tt-nn-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tt-nn
)
install(
    FILES
        ${PROJECT_BINARY_DIR}/tt-nn-config.cmake
        ${PROJECT_BINARY_DIR}/tt-nn-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tt-nn
    COMPONENT ttnn-dev
)

# Filter out components we don't want to package
get_cmake_property(CPACK_COMPONENTS_ALL COMPONENTS)
list(
    REMOVE_ITEM
    CPACK_COMPONENTS_ALL
    tt_pybinds # Wow this one is big!
    tar # TODO: Remove that tarball entirely
    # Deps that define install targets that we can't (or haven't) disabled
    msgpack-cxx
    Headers
    Library
    json-dev
    ttml
    Unspecified # TODO: audit if there's anything we need to ship here
    # Empty placeholder components - these are group "header" components with no install targets.
    # Files are installed to sub-components (e.g., metalium-runtime instead of metalium,
    # ttnn-dev instead of nn-dev).
    #
    # DEB vs RPM behavior difference:
    # - CPack DEB: tolerant of empty components, silently creates empty packages or skips them
    # - CPack RPM: strict, emits "file not found" errors when packaging empty components
    #
    # Removing these is a no-op for DEB (already works) but required for RPM. Empty packages
    # serve no purpose in either format, so we exclude them unconditionally for simplicity.
    #
    # Note: metalium-dev, metalium-examples, metalium-validation have install targets and are NOT empty.
    metalium-jit
    metalium
    nn
    nn-dev
    nn-examples
    nn-validation
    ml
)

# When using system SFPI, the jit-build component has no files
if(TT_USE_SYSTEM_SFPI)
    list(REMOVE_ITEM CPACK_COMPONENTS_ALL jit-build)
endif()

# Component groups and components (common to both DEB and RPM)
cpack_add_component_group(metalium-jit)
cpack_add_component(metalium-jit GROUP metalium-jit DESCRIPTION "TT-Metalium JIT runtime library")
cpack_add_component(jit-build GROUP metalium-jit)

cpack_add_component_group(metalium)
cpack_add_component(metalium GROUP metalium DESCRIPTION "TT-Metalium runtime library")
cpack_add_component(metalium-runtime GROUP metalium)
cpack_add_component(umd-runtime GROUP metalium)
cpack_add_component(tracy GROUP metalium)

cpack_add_component_group(metalium-dev)
cpack_add_component(metalium-dev DEPENDS metalium GROUP metalium-dev DESCRIPTION "TT-Metalium SDK")
cpack_add_component(fmt-core GROUP metalium-dev)
cpack_add_component(enchantum GROUP metalium-dev)
cpack_add_component(umd-dev GROUP metalium-dev)
cpack_add_component(spdlog-dev GROUP metalium-dev)
cpack_add_component(tt-logger-dev GROUP metalium-dev)

cpack_add_component_group(metalium-examples)
cpack_add_component(metalium-examples DEPENDS metalium-dev GROUP metalium-examples DESCRIPTION "TT-Metalium examples")

cpack_add_component_group(metalium-validation)
cpack_add_component(
    metalium-validation
    DEPENDS
        metalium
    GROUP metalium-validation
    DESCRIPTION "TT-Metalium validation tools"
)
cpack_add_component(gtest GROUP metalium-validation)

cpack_add_component_group(nn)
cpack_add_component(nn DEPENDS metalium GROUP nn DESCRIPTION "TT-NN runtime library")
cpack_add_component(ttnn-runtime GROUP nn)

cpack_add_component_group(nn-dev)
cpack_add_component(
    nn-dev
    DEPENDS
        metalium-dev
        nn
    GROUP nn-dev
    DESCRIPTION "TT-NN SDK"
)
cpack_add_component(ttnn-dev GROUP nn-dev)

cpack_add_component_group(nn-examples)
cpack_add_component(nn-examples DEPENDS nn-dev GROUP nn-examples DESCRIPTION "TT-NN examples")
cpack_add_component(ttnn-examples GROUP nn-examples)

cpack_add_component_group(nn-validation)
cpack_add_component(
    nn-validation
    DEPENDS
        nn
        metalium
    GROUP nn-validation
    DESCRIPTION "TT-NN validation tools"
)
cpack_add_component(ttnn-validation GROUP nn-validation)

cpack_add_component_group(ml)
cpack_add_component(
    ml
    DEPENDS
        nn
        metalium
    GROUP ml
    DESCRIPTION "TT-Train runtime library"
)
cpack_add_component(ttml GROUP ml)

include(CPack)
