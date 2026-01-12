# TT-Metal CPack packaging configuration
# Automatically detects distro type and includes appropriate DEB or RPM packaging
#
# =============================================================================
# RPATH / RUNPATH Handling
# =============================================================================
#
# Overview
# --------
# This project uses $ORIGIN-based RPATH to make binaries relocatable. $ORIGIN is
# a special ELF token that the dynamic linker expands at runtime to the directory
# containing the executable or library.
#
# Key Implementation Details
# --------------------------
#
# 1. Core Libraries (tt_metal, tt_stl)
#    - Use BUILD_WITH_INSTALL_RPATH=TRUE: RPATH is correct at build time
#    - Use install(FILES $<TARGET_FILE:...>) instead of install(TARGETS ... LIBRARY)
#      to work around a CMake bug where install(TARGETS) can't find libraries when
#      LIBRARY_OUTPUT_DIRECTORY is set (manifests on Ubuntu but not Fedora)
#    - LIBRARY_OUTPUT_DIRECTORY set to ${CMAKE_BINARY_DIR}/lib for consistent
#      library location (prevents ODR violations in multiprocess tests)
#    - INSTALL_RPATH = "$ORIGIN" (plus MPI path for tt_metal)
#
# 2. MPI Library Path
#    - tt_metal links against MPI (either custom ULFM or system OpenMPI)
#    - MPI library directory is detected in cmake/mpi-config.cmake and exported
#      as TT_METAL_MPI_LIB_DIR
#    - This path is added to tt_metal's INSTALL_RPATH after $ORIGIN
#    - Ensures tests can find libmpi.so at runtime without LD_LIBRARY_PATH
#
# 3. Executables (tests, examples, tt-train)
#    - Use tt_set_runtime_rpath() from cmake/rpath.cmake
#    - BUILD_RPATH contains absolute paths for development
#    - INSTALL_RPATH contains $ORIGIN-relative paths for portability
#
# Ubuntu vs Fedora Differences
# ----------------------------
# - Ubuntu: Uses RPATH (searched before LD_LIBRARY_PATH)
# - Fedora: Uses RUNPATH (searched after LD_LIBRARY_PATH, stricter validation)
# - Fedora's brp-check-rpaths requires $ORIGIN to be FIRST in RPATH
# - Both use the same CMake configuration; differences handled transparently
#
# Directory Layouts Supported
# ---------------------------
#
# 1. CI Tar Artifact Layout (ttm_any.tar.zst extracted to /work/):
#
#    /work/
#    ├── ttnn/ttnn/
#    │   ├── _ttnn.so, _ttnncpp.so     <- Python bindings
#    ├── build/lib/
#    │   ├── libtt_metal.so, libtt_stl.so, libtracy.so
#    ├── build/tt-train/tests/
#    │   └── ttml_tests
#    └── runtime/
#
# 2. Python Wheel Layout:
#
#    {wheel}/ttnn/
#    ├── _ttnn.so
#    ├── build/lib/
#    │   └── libtt_metal.so, libtt_stl.so, ...
#    └── runtime/
#
# 3. FHS Package Layout (DEB/RPM):
#
#    /usr/lib64/
#    └── _ttnn.so, libtt_metal.so, libtt_stl.so, ...
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

set(CPACK_THREADS 0) # Enable multithreading for compression

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

# Helper function to detect packaging type (DEB vs RPM)
# This function detects the appropriate package manager type based on the distribution.
# It can be overridden by setting TT_PACKAGING_TYPE via -D flag (creates a cache variable).
function(detect_packaging_type)
    # Check if user has explicitly set TT_PACKAGING_TYPE via -D flag (cache variable)
    # Command-line -D flags create cache variables, which are checked here
    if(DEFINED TT_PACKAGING_TYPE AND NOT TT_PACKAGING_TYPE STREQUAL "")
        message(STATUS "Using user-specified packaging type: ${TT_PACKAGING_TYPE}")
        set(TT_PACKAGING_TYPE "${TT_PACKAGING_TYPE}" PARENT_SCOPE)
        return()
    endif()

    # Auto-detect from distribution information
    set(TT_PACKAGING_TYPE "")
    include(${CMAKE_CURRENT_LIST_DIR}/detect-distro.cmake)
    detect_distro()

    # Determine package type from distro ID and ID_LIKE
    # Check ID_LIKE first (more reliable for derivatives)
    if(DISTRO_ID_LIKE)
        if(DISTRO_ID_LIKE MATCHES "debian|ubuntu")
            set(TT_PACKAGING_TYPE "DEB")
        elseif(DISTRO_ID_LIKE MATCHES "fedora|rhel|centos|suse")
            set(TT_PACKAGING_TYPE "RPM")
        endif()
    endif()
    # Check ID if ID_LIKE didn't match
    if(TT_PACKAGING_TYPE STREQUAL "" AND DISTRO_ID)
        if(DISTRO_ID MATCHES "debian|ubuntu|linuxmint|pop")
            set(TT_PACKAGING_TYPE "DEB")
        elseif(DISTRO_ID MATCHES "fedora|rhel|centos|rocky|alma|opensuse|sles")
            set(TT_PACKAGING_TYPE "RPM")
        endif()
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

    # Default to DEB if detection failed
    if(TT_PACKAGING_TYPE STREQUAL "")
        message(WARNING "Could not detect distro package type, defaulting to DEB")
        set(TT_PACKAGING_TYPE "DEB")
    endif()

    set(TT_PACKAGING_TYPE "${TT_PACKAGING_TYPE}" PARENT_SCOPE)
endfunction()

# Detect package manager type: DEB vs RPM
# Allow override via command line: -DTT_PACKAGING_TYPE=DEB or -DTT_PACKAGING_TYPE=RPM
detect_packaging_type()

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

# =============================================================================
# Technical Notes: CMake Workarounds and Design Decisions
# =============================================================================
#
# CMake install(TARGETS) Bug with LIBRARY_OUTPUT_DIRECTORY
# --------------------------------------------------------
# When LIBRARY_OUTPUT_DIRECTORY is set, CMake's install(TARGETS ... LIBRARY)
# command fails to locate the library file on some systems (notably Ubuntu).
# The error manifests as "file INSTALL cannot find <path>/libfoo.so".
#
# Root cause: CMake's internal path resolution for install(TARGETS) doesn't
# properly account for LIBRARY_OUTPUT_DIRECTORY in all cases. On Fedora, a
# lib64->lib symlink in the build directory happens to make it work.
#
# Workaround: Use install(FILES $<TARGET_FILE:target> DESTINATION ...) instead
# of install(TARGETS ... LIBRARY). The generator expression correctly resolves
# to the actual library location. However, install(FILES) doesn't do RPATH
# fixup, so we use BUILD_WITH_INSTALL_RPATH=TRUE to embed the correct RPATH
# at build time.
#
# Why LIBRARY_OUTPUT_DIRECTORY is Needed
# --------------------------------------
# Without LIBRARY_OUTPUT_DIRECTORY, libraries go to their default locations
# (e.g., build/tt_metal/libtt_metal.so vs build/tt_stl/libtt_stl.so). This
# causes ODR (One Definition Rule) violations in multiprocess tests where
# different processes might load different copies of the same library.
# Setting LIBRARY_OUTPUT_DIRECTORY to ${CMAKE_BINARY_DIR}/lib ensures all
# libraries are in one location, matching CI tar artifact expectations.
#
# BUILD_WITH_INSTALL_RPATH=TRUE vs FALSE
# --------------------------------------
# - FALSE (default): BUILD_RPATH used during build, INSTALL_RPATH after install
#   - Pro: CMAKE_BUILD_RPATH_USE_LINK_PATH automatically adds linked library
#     directories (e.g., MPI) to BUILD_RPATH
#   - Con: Requires install(TARGETS) or install(CODE) to fix RPATH after copy
#
# - TRUE: INSTALL_RPATH used at both build time and after install
#   - Pro: Works with install(FILES) - no post-install RPATH fixup needed
#   - Con: Must manually add all needed paths (e.g., MPI) to INSTALL_RPATH
#
# We use TRUE for tt_metal and tt_stl because:
# 1. install(FILES) workaround requires RPATH to be correct at build time
# 2. MPI path is explicitly added to INSTALL_RPATH via TT_METAL_MPI_LIB_DIR
# 3. $ORIGIN-based paths work correctly in all deployment scenarios
#
# Historical Note: Build Paths in RPATH
# -------------------------------------
# Previously, INSTALL_RPATH contained absolute build paths to support the
# tt_pybinds component which installs into the SOURCE tree for development.
# This was replaced with $ORIGIN-relative paths that work across all layouts.
#
