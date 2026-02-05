# Source files for HAL 1xx (Wormhole and Blackhole)
# Module owners should update this file when adding/removing/renaming source files

set(WH_HAL_SOURCES
    wormhole/wh_hal.cpp
    wormhole/wh_hal_tensix.cpp
    wormhole/wh_hal_active_eth.cpp
    wormhole/wh_hal_idle_eth.cpp
)

set(BH_HAL_SOURCES
    blackhole/bh_hal.cpp
    blackhole/bh_hal_tensix.cpp
    blackhole/bh_hal_active_eth.cpp
    blackhole/bh_hal_idle_eth.cpp
)

set(HAL_1XX_SOURCES hal_1xx_common.cpp)
