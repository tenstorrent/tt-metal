# Provide unique architecture specific translation units and include paths for each ARCH

set(GS_HAL_SRC grayskull/gs_hal.cpp)

set(WH_HAL_SRC
    wormhole/wh_hal.cpp
    wormhole/wh_hal_tensix.cpp
    wormhole/wh_hal_active_eth.cpp
    wormhole/wh_hal_idle_eth.cpp
)

set(BH_HAL_SRC
    blackhole/bh_hal.cpp
    blackhole/bh_hal_tensix.cpp
    blackhole/bh_hal_active_eth.cpp
    blackhole/bh_hal_idle_eth.cpp
)

set(HAL_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tt_metal
    ${PROJECT_SOURCE_DIR}/tt_metal/third_party/umd
)

add_library(gs_hal OBJECT ${GS_HAL_SRC})
add_library(HAL::grayskull ALIAS gs_hal)
target_include_directories(
    gs_hal
    PRIVATE
        ${HAL_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/grayskull
)
target_link_libraries(
    gs_hal
    PRIVATE
        hw
        fmt::fmt-header-only
)

add_library(wh_hal OBJECT ${WH_HAL_SRC})
add_library(HAL::wormhole ALIAS wh_hal)
target_include_directories(
    wh_hal
    PRIVATE
        ${HAL_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/wormhole
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/wormhole/wormhole_b0_defines
)
target_link_libraries(
    wh_hal
    PRIVATE
        hw
        fmt::fmt-header-only
)

add_library(bh_hal OBJECT ${BH_HAL_SRC})
add_library(HAL::blackhole ALIAS bh_hal)
target_include_directories(
    bh_hal
    PRIVATE
        ${HAL_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/blackhole
)
target_link_libraries(
    bh_hal
    PRIVATE
        hw
        fmt::fmt-header-only
)

set(LLRT_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/llrt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rtoptions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tlb_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tt_cluster.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tt_elffile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tt_memory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hal.cpp
)

add_library(llrt STATIC ${LLRT_SRC}) # FIXME(14541): Should be OBJECT, but can't handle circular deps between Object libs
add_library(Metalium::Metal::LLRT ALIAS llrt)

target_link_libraries(
    llrt
    PUBLIC
        Metalium::Metal::Common
        umd::device
        Metalium::Metal::Hardware
    PRIVATE
        HAL::grayskull
        HAL::wormhole
        HAL::blackhole
)
target_compile_options(llrt PRIVATE -Wno-int-to-pointer-cast)
