set(CPACK_GENERATOR DEB)
set(CPACK_PACKAGE_CONTACT "support@tenstorrent.com")
set(CMAKE_PROJECT_HOMEPAGE_URL "https://tenstorrent.com")
set(CPACK_PACKAGE_NAME tt)

# Suppress the summary so that we can have per-component summaries
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "")
set(CPACK_DEBIAN_METALIUM_PACKAGE_SECTION "libs")
set(CPACK_DEBIAN_METALIUM-DEV_PACKAGE_SECTION "libs")
set(CPACK_DEBIAN_METALIUM-JIT_PACKAGE_SECTION "libs")
set(CPACK_DEBIAN_METALIUM-EXAMPLES_PACKAGE_SECTION "doc")
set(CPACK_DEBIAN_METALIUM-VALIDATION_PACKAGE_SECTION "utils")
set(CPACK_DEBIAN_NN_PACKAGE_SECTION "libs")
set(CPACK_DEBIAN_NN-VALIDATION_PACKAGE_SECTION "utils")

set(CPACK_DEB_COMPONENT_INSTALL YES)
set(CPACK_DEBIAN_PACKAGE_VERSION "${VERSION_DEB}")
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)

set(CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION TRUE)

string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWER)
if(CMAKE_BUILD_TYPE_LOWER STREQUAL "asan" OR CMAKE_BUILD_TYPE_LOWER STREQUAL "tsan")
    set(CPACK_DEBIAN_DEBUGINFO_PACKAGE FALSE)
else()
    set(CPACK_DEBIAN_METALIUM_DEBUGINFO_PACKAGE TRUE)
    set(CPACK_DEBIAN_METALIUM-VALIDATION_DEBUGINFO_PACKAGE TRUE)
    set(CPACK_DEBIAN_METALIUM-DEV_DEBUGINFO_PACKAGE TRUE)
    set(CPACK_DEBIAN_METALIUM-EXAMPLES_DEBUGINFO_PACKAGE TRUE)
    set(CPACK_DEBIAN_JIT-BUILD_DEBUGINFO_PACKAGE FALSE) # Some binaries don't have a Build ID; we cannot split dbgsyms
endif()

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

# FIXME(afuller): Sucks for Ubuntu 22.04, but I'm not about to start packaging Boost.
set(CPACK_DEBIAN_METALIUM-DEV_PACKAGE_DEPENDS "libboost-dev (>= 1.78) | libboost1.81-dev")

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
cpack_add_component(json-dev GROUP metalium-dev)
cpack_add_component(magic-enum-dev GROUP metalium-dev)
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

include(CPack)
