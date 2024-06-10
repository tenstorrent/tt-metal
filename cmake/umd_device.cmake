find_package(yaml-cpp REQUIRED)
add_subdirectory(${UMD_HOME}/third_party/fmt)
include_directories(${UMD_HOME})
set(UMD_SRC
    ${UMD_HOME}/device/architecture_implementation.cpp
    ${UMD_HOME}/device/blackhole_implementation.cpp
    ${UMD_HOME}/device/cpuset_lib.cpp
    ${UMD_HOME}/device/grayskull_implementation.cpp
    ${UMD_HOME}/device/tlb.cpp
    ${UMD_HOME}/device/tt_cluster_descriptor.cpp
    ${UMD_HOME}/device/tt_device.cpp
    ${UMD_HOME}/device/tt_emulation_stub.cpp
    ${UMD_HOME}/device/tt_silicon_driver.cpp
    ${UMD_HOME}/device/tt_silicon_driver_common.cpp
    ${UMD_HOME}/device/tt_soc_descriptor.cpp
    ${UMD_HOME}/device/tt_versim_stub.cpp
    ${UMD_HOME}/device/wormhole_implementation.cpp
)

add_library(umd_device OBJECT ${UMD_SRC})
set_target_properties(umd_device PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(umd_device PRIVATE fmt::fmt-header-only yaml-cpp hwloc rt compiler_flags metal_header_directories)
