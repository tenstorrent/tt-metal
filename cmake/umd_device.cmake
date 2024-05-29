
### THIS CMAKE IS TO BUILD THE UMD_DEVICE SHARED LIBRARY ###
### All variables/compiler flags declared in this file are passed to umd/device .mk file to build device ###

set(WARNINGS "-Werror -Wdelete-non-virtual-dtor -Wreturn-type -Wswitch -Wuninitialized -Wno-unused-parameter" CACHE STRING "Warnings to enable")

# TODO(joelsmith): do we need to pass the -fsanitize flags to compiler too?
if(CMAKE_BUILD_TYPE STREQUAL "ci")
    set(CONFIG_LDFLAGS "${CONFIG_LDFLAGS} -Wl,--verbose")
elseif(CMAKE_BUILD_TYPE STREQUAL "asan")
    set(CONFIG_LDFLAGS "${CONFIG_LDFLAGS} -fsanitize=address")
elseif(CMAKE_BUILD_TYPE STREQUAL "ubsan")
    set(CONFIG_LDFLAGS "${CONFIG_LDFLAGS} -fsanitize=undefined")
endif()


# TODO(joelsmith): why do we need to link UMD against tracy?
# UMD isn't using it directly..
if($ENV{ENABLE_TRACY})
    set(CONFIG_LDFLAGS "${CONFIG_LDFLAGS} -ltracy -rdynamic")
endif()

# MUST have the RPATH set, or else can't find the tracy lib
set(LDFLAGS_ "-L${CMAKE_BINARY_DIR}/lib -Wl,-rpath,${CMAKE_BINARY_DIR}/lib ${CONFIG_LDFLAGS} -ldl -lz -lpthread -latomic -lhwloc -lstdc++")

set (CMAKE_CXX_FLAGS_ "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden ${WARNINGS}")
foreach(lib ${BoostPackages})
    set(CMAKE_CXX_FLAGS_ "${CMAKE_CXX_FLAGS_} -I${Boost${lib}_SOURCE_DIR}/include")
endforeach()

set(UMD_OUTPUT > /dev/null 2>&1)
if(DEFINED ENV{VERBOSE})
    if($ENV{VERBOSE} STREQUAL 1)
        set(UMD_OUTPUT "")
    endif()
endif()

# This will build the shared library libdevice.so in build/lib where tt_metal can then find and link it
include(ExternalProject)
ExternalProject_Add(
    umd_device
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/umd_device
    SOURCE_DIR ${UMD_HOME}
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/umd
    INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/lib
    CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS_} -DCMAKE_SHARED_LINKER_FLAGS=${LDFLAGS_} ${UMD_OUTPUT}
    BUILD_COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/umd ${UMD_OUTPUT}
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/umd/lib/libdevice.so ${CMAKE_CURRENT_BINARY_DIR}/lib/libdevice.so
)

if($ENV{ENABLE_TRACY})
    add_dependencies(umd_device TracyClient)
endif()

# TODO(joelsmith) - should the following be removed?

# If in production build for python packaging, need to use objs built by umd_device
if(NOT BUILD_SHARED_LIBS)
    set(UMD_OBJS
        ${UMD_OBJS}
        ${CMAKE_BINARY_DIR}/obj/umd/device/architecture_implementation.o
        ${CMAKE_BINARY_DIR}/obj/umd/device/blackhole_implementation.o
        ${CMAKE_BINARY_DIR}/obj/umd/device/cpuset_lib.o
        ${CMAKE_BINARY_DIR}/obj/umd/device/grayskull_implementation.o
        ${CMAKE_BINARY_DIR}/obj/umd/device/tt_cluster_descriptor.o
        ${CMAKE_BINARY_DIR}/obj/umd/device/tt_device.o
        ${CMAKE_BINARY_DIR}/obj/umd/device/tt_emulation_stub.o
        ${CMAKE_BINARY_DIR}/obj/umd/device/tt_silicon_driver_common.o
        ${CMAKE_BINARY_DIR}/obj/umd/device/tt_silicon_driver.o
        ${CMAKE_BINARY_DIR}/obj/umd/device/tt_soc_descriptor.o
        ${CMAKE_BINARY_DIR}/obj/umd/device/tt_versim_stub.o
        ${CMAKE_BINARY_DIR}/obj/umd/device/tlb.o
        ${CMAKE_BINARY_DIR}/obj/umd/device/wormhole_implementation.o
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
