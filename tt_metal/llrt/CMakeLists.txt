# Provide unique architecture specific translation units and include paths for each ARCH

# FIXME: Don't want to be reaching above to project dir
# Prefer a library target to be explicit about the dependency in CMake
set(HAL_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tt_metal
)

add_library(gs_hal OBJECT)
add_library(HAL::grayskull ALIAS gs_hal)
target_sources(gs_hal PRIVATE grayskull/gs_hal.cpp)
target_include_directories(
    gs_hal
    BEFORE
    PRIVATE
        ${HAL_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/grayskull
        ${PROJECT_SOURCE_DIR}/tt_metal/api/grayskull/tt-metalium
)
target_link_libraries(
    gs_hal
    PRIVATE
        hw
        fmt::fmt-header-only
        umd::device
)

add_library(wh_hal OBJECT)
add_library(HAL::wormhole ALIAS wh_hal)
target_sources(
    wh_hal
    PRIVATE
        wormhole/wh_hal.cpp
        wormhole/wh_hal_tensix.cpp
        wormhole/wh_hal_active_eth.cpp
        wormhole/wh_hal_idle_eth.cpp
)
target_include_directories(
    wh_hal
    BEFORE
    PRIVATE
        ${HAL_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/wormhole
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/wormhole/wormhole_b0_defines
        ${PROJECT_SOURCE_DIR}/tt_metal/api/wormhole/tt-metalium
)
target_link_libraries(
    wh_hal
    PRIVATE
        hw
        fmt::fmt-header-only
        umd::device
)

add_library(bh_hal OBJECT ${BH_HAL_SRC})
add_library(HAL::blackhole ALIAS bh_hal)
target_sources(
    bh_hal
    PRIVATE
        blackhole/bh_hal.cpp
        blackhole/bh_hal_tensix.cpp
        blackhole/bh_hal_active_eth.cpp
        blackhole/bh_hal_idle_eth.cpp
)
target_include_directories(
    bh_hal
    BEFORE
    PRIVATE
        ${HAL_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/blackhole
        ${PROJECT_SOURCE_DIR}/tt_metal/api/blackhole/tt-metalium
)
target_link_libraries(
    bh_hal
    PRIVATE
        hw
        fmt::fmt-header-only
        umd::device
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

add_library(llrt OBJECT ${LLRT_SRC})
add_library(Metalium::Metal::LLRT ALIAS llrt)

target_link_libraries(
    llrt
    PUBLIC
        umd::device
        Metalium::Metal::Hardware
    PRIVATE
        HAL::grayskull
        HAL::wormhole
        HAL::blackhole
        Tracy::TracyClient
        nlohmann_json::nlohmann_json
        Reflect::Reflect
        magic_enum
        span
        Metalium::Metal::Impl
)
target_compile_options(llrt PRIVATE -Wno-int-to-pointer-cast)
