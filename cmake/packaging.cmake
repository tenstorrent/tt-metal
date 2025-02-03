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
set(CPACK_DEBIAN_DEBUGINFO_PACKAGE TRUE)

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

get_cmake_property(CPACK_COMPONENTS_ALL COMPONENTS)
list(
    REMOVE_ITEM
    CPACK_COMPONENTS_ALL
    # dev FIXME: uncomment when we bump UMD submodule
    tt_pybinds # Wow this one is big!
    Unspecified # TODO: audit if there's anything we need to ship here
    Headers # TODO: Where is this coming from?
    Library # TODO: Where is this coming from?
    msgpack-cxx # TODO: Where is this coming from?
)

cpack_add_component(metalium GROUP metalium)
cpack_add_component(umd-runtime GROUP metalium)
cpack_add_component(dev GROUP metalium) # FIXME: delete this line when we bump UMD submodule
cpack_add_component_group(metalium)

include(CPack)
