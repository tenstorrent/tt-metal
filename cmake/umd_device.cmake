
### THIS CMAKE IS TO BUILD THE UMD_DEVICE SHARED LIBRARY ###
### All variables/compiler flags declared in this file are passed to umd/device .mk file to build device ###

set(WARNINGS "-Werror -Wdelete-non-virtual-dtor -Wreturn-type -Wswitch -Wuninitialized -Wno-unused-parameter" CACHE STRING "Warnings to enable")

if(CMAKE_BUILD_TYPE STREQUAL "ci")
    set(CONFIG_LDFLAGS "${CONFIG_LDFLAGS} -Wl,--verbose")
elseif(CMAKE_BUILD_TYPE STREQUAL "asan")
    set(CONFIG_LDFLAGS "${CONFIG_LDFLAGS} -fsanitize=address")
elseif(CMAKE_BUILD_TYPE STREQUAL "ubsan")
    set(CONFIG_LDFLAGS "${CONFIG_LDFLAGS} -fsanitize=undefined")
endif()

if(NOT TT_METAL_VERSIM_DISABLED)
    set(UMD_VERSIM_STUB 0)
else()
    set(UMD_VERSIM_STUB 1)
endif()
if($ENV{ENABLE_TRACY})
    set(CONFIG_LDFLAGS "${CONFIG_LDFLAGS} -ltracy -rdynamic")
endif()

# MUST have the RPATH set, or else can't find the tracy lib
set(LDFLAGS_ "-L${CMAKE_BINARY_DIR}/lib -Wl,-rpath,${CMAKE_BINARY_DIR}/lib ${CONFIG_LDFLAGS} -ldl -lz -lboost_thread -lboost_filesystem -lboost_system -lboost_regex -lpthread -latomic -lhwloc -lstdc++")
set(SHARED_LIB_FLAGS_ "-shared -fPIC")
set(STATIC_LIB_FLAGS_ "-fPIC")

set (CMAKE_CXX_FLAGS_ "--std=c++17 -fvisibility-inlines-hidden")

# This will build the shared library libdevice.so in build/lib where tt_metal can then find and link it
include(ExternalProject)
ExternalProject_Add(
    umd_device
    PREFIX ${UMD_HOME}
    SOURCE_DIR ${UMD_HOME}
    BINARY_DIR ${CMAKE_BINARY_DIR}
    INSTALL_DIR ${CMAKE_BINARY_DIR}
    STAMP_DIR "${CMAKE_BINARY_DIR}/tmp/umd_stamp"
    TMP_DIR "${CMAKE_BINARY_DIR}/tmp/umd_tmp"
    DOWNLOAD_COMMAND ""
    CONFIGURE_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_COMMAND
        make -f ${UMD_HOME}/device/module.mk umd_device
        OUT=${CMAKE_BINARY_DIR}
        LIBDIR=${CMAKE_BINARY_DIR}/lib
        OBJDIR=${CMAKE_BINARY_DIR}/obj
        UMD_HOME=${UMD_HOME}
        UMD_VERSIM_STUB=${UMD_VERSIM_STUB}
        UMD_VERSIM_HEADERS=${TT_METAL_VERSIM_ROOT}/versim/
        UMD_USER_ROOT=$ENV{TT_METAL_HOME}
        WARNINGS=${WARNINGS}
        SHARED_LIB_FLAGS=${SHARED_LIB_FLAGS_}
        STATIC_LIB_FLAGS=${STATIC_LIB_FLAGS_}
        LDFLAGS=${LDFLAGS_}
        CXXFLAGS=${CMAKE_CXX_FLAGS_}
)
if($ENV{ENABLE_TRACY})
    add_dependencies(umd_device TracyClient)
endif()

# If in production build for python packaging, need to use objs built by umd_device
if(NOT BUILD_SHARED_LIBS)
    if($ENV{ARCH_NAME} STREQUAL "wormhole_b0")
        set(UMD_OBJS ${CMAKE_BINARY_DIR}/obj/umd/device/wormhole/impl_device.o)
    else()
        set(UMD_OBJS ${CMAKE_BINARY_DIR}/obj/umd/device/$ENV{ARCH_NAME}/impl_device.o)
    endif()
    set(UMD_OBJS
        ${UMD_OBJS}
        ${CMAKE_BINARY_DIR}/obj/umd/device/cpuset_lib.o
        ${CMAKE_BINARY_DIR}/obj/umd/device/tt_cluster_descriptor.o
        ${CMAKE_BINARY_DIR}/obj/umd/device/tt_device.o
        ${CMAKE_BINARY_DIR}/obj/umd/device/tt_emulation_stub.o
        ${CMAKE_BINARY_DIR}/obj/umd/device/tt_silicon_driver_common.o
        ${CMAKE_BINARY_DIR}/obj/umd/device/tt_silicon_driver.o
        ${CMAKE_BINARY_DIR}/obj/umd/device/tt_soc_descriptor.o
        ${CMAKE_BINARY_DIR}/obj/umd/device/tt_versim_stub.o
        ${CMAKE_BINARY_DIR}/obj/umd/device/util.o
    )
    set(UMD_STATIC_LIB ${CMAKE_BINARY_DIR}/lib/libdevice.a)

    # Build static lib with objs created after umd_device is built
    add_custom_command(
        OUTPUT ${UMD_STATIC_LIB}
        COMMAND ar rcs ${UMD_STATIC_LIB} ${UMD_OBJS}
        DEPENDS umd_device
        COMMENT "Creating static device library"
    )
    add_custom_target(
        umd_static_lib_target ALL
        DEPENDS ${UMD_STATIC_LIB}
    )
endif()
