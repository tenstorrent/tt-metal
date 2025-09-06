# HAL_1xx - Hardware Abstraction Layer for Wormhole and Blackhole

# FIXME: Don't want to be reaching above to project dir
# Prefer a library target to be explicit about the dependency in CMake
set(HAL_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tt_metal
    ${PROJECT_SOURCE_DIR}/tt_metal/llrt
)

add_library(wh_hal OBJECT)
add_dependencies(wh_hal hal_generate_dev_msgs_header)
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
        TT::STL
        tt-logger::tt-logger
)

add_library(bh_hal OBJECT ${BH_HAL_SRC})
add_dependencies(bh_hal hal_generate_dev_msgs_header)
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
        TT::STL
        tt-logger::tt-logger
)

add_library(hal_1xx STATIC)
add_library(HAL::1xx ALIAS hal_1xx)
target_sources(hal_1xx PRIVATE hal_1xx_common.cpp)
target_include_directories(hal_1xx BEFORE PRIVATE ${HAL_INCLUDE_DIRS})
target_link_libraries(
    hal_1xx
    PUBLIC
        wh_hal
        bh_hal
    PRIVATE
        fmt::fmt-header-only
        hw
        umd::device
        TT::STL
        tt-logger::tt-logger
)
