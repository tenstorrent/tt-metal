set(CPACK_GENERATOR DEB)
set(CPACK_PACKAGE_CONTACT "support@tenstorrent.com")
set(CMAKE_PROJECT_HOMEPAGE_URL "https://tenstorrent.com")
set(CPACK_PACKAGE_NAME tt)

set(CPACK_COMPONENT_METALIUM_DESCRIPTION "TT-Metalium runtime library")
set(CPACK_DEBIAN_METALIUM_PACKAGE_SECTION "libs")

set(CPACK_DEB_COMPONENT_INSTALL YES)
set(CPACK_DEBIAN_PACKAGE_VERSION "${VERSION_DEB}")
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)

set(CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION TRUE)
# set(CPACK_DEBIAN_DEBUGINFO_PACKAGE TRUE)
set(CPACK_DEBIAN_METALIUM_DEBUGINFO_PACKAGE TRUE)
set(CPACK_DEBIAN_JIT-BUILD_DEBUGINFO_PACKAGE FALSE) # Some binaries don't have a Build ID; we cannot split dbgsyms

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
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS FALSE)

get_cmake_property(CPACK_COMPONENTS_ALL COMPONENTS)
list(
    REMOVE_ITEM
    CPACK_COMPONENTS_ALL
    umd-dev # FIXME: -dev packages will come later
    tt_pybinds # Wow this one is big!
    tar # TODO: Remove that tarball entirely
    # Deps that define install targets that we can't (or haven't) disabled
    msgpack-cxx
    Headers
    Library
    Unspecified # TODO: audit if there's anything we need to ship here
)

# Logically we should ship jit-build with metalium-runtime, but jit-build fails to split dbgsyms for now (lacking a Build ID on the binaries)
cpack_add_component(jit-build GROUP metalium-jit)

cpack_add_component(metalium-runtime GROUP metalium)
cpack_add_component(umd-runtime GROUP metalium)
cpack_add_component(dev GROUP metalium) # FIXME: delete this line when we bump UMD submodule
cpack_add_component_group(metalium)

cpack_add_component(gtest GROUP metalium-validation)
cpack_add_component_group(metalium-validation)

include(CPack)
