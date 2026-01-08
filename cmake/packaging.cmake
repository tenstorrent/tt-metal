set(CPACK_GENERATOR DEB)
set(CPACK_PACKAGE_CONTACT "support@tenstorrent.com")
set(CMAKE_PROJECT_HOMEPAGE_URL "https://tenstorrent.com")
set(CPACK_PACKAGE_NAME tt)

# Suppress the summary so that we can have per-component summaries
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "")
set(CPACK_DEBIAN_METALIUM_PACKAGE_SECTION "libs")
set(CPACK_DEBIAN_METALIUM-DEV_PACKAGE_SECTION "devel")
set(CPACK_DEBIAN_METALIUM-JIT_PACKAGE_SECTION "libs")
set(CPACK_DEBIAN_METALIUM-EXAMPLES_PACKAGE_SECTION "doc")
set(CPACK_DEBIAN_METALIUM-VALIDATION_PACKAGE_SECTION "utils")
set(CPACK_DEBIAN_NN_PACKAGE_SECTION "libs")
set(CPACK_DEBIAN_NN-DEV_PACKAGE_SECTION "devel")
set(CPACK_DEBIAN_NN-EXAMPLES_PACKAGE_SECTION "doc")
set(CPACK_DEBIAN_NN-VALIDATION_PACKAGE_SECTION "utils")

set(CPACK_DEBIAN_COMPRESSION_TYPE zstd)
set(CPACK_THREADS 0) # Enable multithreading for compression
set(CPACK_DEB_COMPONENT_INSTALL YES)
set(CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION TRUE)

# Use project config file to defer build-type-specific configuration to packaging time
# This is necessary for multi-config generators.
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/packaging.d/cpack-project-config.cmake.in"
    "${PROJECT_BINARY_DIR}/cpack-project-config.cmake"
    @ONLY
)
set(CPACK_PROJECT_CONFIG_FILE "${PROJECT_BINARY_DIR}/cpack-project-config.cmake")

set(CPACK_DEBIAN_PACKAGE_VERSION "${VERSION_DEB}")
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)

set(CPACK_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE
)

set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS TRUE)
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS TRUE)
# jit-build is cross compiling; shlibdeps does not find dependencies on the host; it should be self-contained anyway.
set(CPACK_DEBIAN_METALIUM-JIT_PACKAGE_SHLIBDEPS FALSE)
# ttml uses internal libraries from metalium/nn packages; dependencies are declared explicitly
set(CPACK_DEBIAN_TTML_PACKAGE_SHLIBDEPS FALSE)

set(CPACK_DEBIAN_METALIUM-DEV_PACKAGE_DEPENDS "nlohmann-json3-dev (>= 3.10)")
set(CPACK_DEBIAN_NN-DEV_PACKAGE_DEPENDS "libxtensor-dev (>= 0.23.10)")
set(CPACK_DEBIAN_TTML_PACKAGE_DEPENDS "python3 (>= 3.8)")

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
)

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
