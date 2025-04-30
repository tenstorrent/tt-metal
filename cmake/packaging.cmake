set(CPACK_GENERATOR DEB)
set(CPACK_PACKAGE_CONTACT "support@tenstorrent.com")
set(CMAKE_PROJECT_HOMEPAGE_URL "https://tenstorrent.com")
set(CPACK_PACKAGE_NAME tt)

set(CPACK_COMPONENT_METALIUM_DESCRIPTION "TT-Metalium runtime library")
set(CPACK_DEBIAN_METALIUM_PACKAGE_SECTION "libs")
set(CPACK_COMPONENT_METALIUM-TTNN-DEV_DESCRIPTION "TT-Metalium TTNN development files")

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
set(CPACK_DEBIAN_METALIUM-TTNN-DEV_PACKAGE_SHLIBDEPS FALSE)

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

cpack_add_component(jit-build GROUP metalium-jit)

cpack_add_component(metalium-runtime GROUP metalium)
cpack_add_component(umd-runtime GROUP metalium)
cpack_add_component(tracy GROUP metalium)
cpack_add_component_group(metalium)

cpack_add_component(metalium-dev GROUP metalium-dev)
cpack_add_component(fmt-core GROUP metalium-dev)
cpack_add_component(json-dev GROUP metalium-dev)
cpack_add_component(magic-enum-dev GROUP metalium-dev)
cpack_add_component(umd-dev GROUP metalium-dev)
cpack_add_component_group(metalium-dev)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/ttnn DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tt-metal COMPONENT metalium-ttnn-dev)
install(
    DIRECTORY
        ${CMAKE_SOURCE_DIR}/tt_metal
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tt-metal
    COMPONENT metalium-ttnn-dev
)
install(
    DIRECTORY
        ${CMAKE_SOURCE_DIR}/tt_stl
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tt-metal
    COMPONENT metalium-ttnn-dev
)
install(
    DIRECTORY
        ${CMAKE_SOURCE_DIR}/.cpmcache/reflect/e75434c4c5f669e4a74e4d84e0a30d7249c1e66f/
        ${CMAKE_SOURCE_DIR}/.cpmcache/fmt/69912fb6b71fcb1f7e5deca191a2bb4748c4e7b6/include/
        ${CMAKE_SOURCE_DIR}/.cpmcache/magic_enum/4d76fe0a5b27a0e62d6c15976d02b33c54207096/include/
        ${CMAKE_SOURCE_DIR}/.cpmcache/nlohmann_json/798e0374658476027d9723eeb67a262d0f3c8308/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tt-metal/third-party
    COMPONENT metalium-ttnn-dev
    FILES_MATCHING
    PATTERN
    "*.h"
    PATTERN
    "*.hpp"
)
install(
    FILES
        ${CMAKE_SOURCE_DIR}/.cpmcache/reflect/e75434c4c5f669e4a74e4d84e0a30d7249c1e66f/reflect
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tt-metal/third-party/
    COMPONENT metalium-ttnn-dev
)

install(
    DIRECTORY
        ${CMAKE_SOURCE_DIR}/tt_metal/core_descriptors
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal/core_descriptors
    COMPONENT metalium-ttnn-dev
)
install(
    DIRECTORY
        ${CMAKE_SOURCE_DIR}/generated/watcher
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/
    COMPONENT metalium-ttnn-dev
)

find_library(TT_METAL_LIBRARY NAMES "tt_metal" PATHS "${CMAKE_BINARY_DIR}/tt_metal" NO_DEFAULT_PATH)
find_library(DEVICE_LIBRARY NAMES "device" PATHS "${CMAKE_BINARY_DIR}/lib" NO_DEFAULT_PATH)
find_library(TTNN_LIBRARY NAMES "_ttnn.so" PATHS "${CMAKE_BINARY_DIR}/tnn" NO_DEFAULT_PATH)

message(STATUS "TT_METAL_LIBRARY: ${TT_METAL_LIBRARY}")
message(STATUS "DEVICE_LIBRARY: ${DEVICE_LIBRARY}")
message(STATUS "TTNN_LIBRARY: ${TTNN_LIBRARY}")

install(
    FILES
        ${TT_METAL_LIBRARY}
        ${DEVICE_LIBRARY}
        ${TTNN_LIBRARY}
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT metalium-ttnn-dev
)

cpack_add_component(metalium-ttnn-dev)

cpack_add_component(metalium-validation DEPENDS metalium GROUP metalium-validation)
cpack_add_component(gtest GROUP metalium-validation)
cpack_add_component_group(metalium-validation)

include(CPack)
