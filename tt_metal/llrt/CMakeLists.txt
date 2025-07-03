# Provide unique architecture specific translation units and include paths for each ARCH

# FIXME: Don't want to be reaching above to project dir
# Prefer a library target to be explicit about the dependency in CMake
set(HAL_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tt_metal
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
        Tracy::TracyClient
        hw
        fmt::fmt-header-only
        umd::device
        tt-logger::tt-logger
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
        Tracy::TracyClient
        hw
        fmt::fmt-header-only
        umd::device
        tt-logger::tt-logger
)

set(LLRT_SRC
    llrt.cpp
    core_descriptor.cpp
    rtoptions.cpp
    tlb_config.cpp
    tt_cluster.cpp
    tt_elffile.cpp
    tt_memory.cpp
    hal.cpp
    utils.cpp
)

add_library(llrt OBJECT ${LLRT_SRC})
add_library(Metalium::Metal::LLRT ALIAS llrt)

target_link_libraries(
    llrt
    PUBLIC
        umd::device
        Metalium::Metal::Hardware
    PRIVATE
        HAL::wormhole
        HAL::blackhole
        Tracy::TracyClient
        nlohmann_json::nlohmann_json
        Reflect::Reflect
        yaml-cpp::yaml-cpp
        magic_enum::magic_enum
        span
        TT::Metalium::Common
        tt-logger::tt-logger
)
target_include_directories(llrt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(llrt PRIVATE -Wno-int-to-pointer-cast)
