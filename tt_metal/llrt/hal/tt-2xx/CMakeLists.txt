# HAL_2xx - Hardware Abstraction Layer for Quasar

# FIXME: Don't want to be reaching above to project dir
# Prefer a library target to be explicit about the dependency in CMake
set(HAL_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tt_metal
    ${PROJECT_SOURCE_DIR}/tt_metal/llrt
)

add_library(qa_hal OBJECT ${QA_HAL_SRC})
target_sources(
    qa_hal
    PRIVATE
        quasar/qa_hal.cpp
        quasar/qa_hal_tensix.cpp
        quasar/qa_hal_active_eth.cpp
        quasar/qa_hal_idle_eth.cpp
)
target_include_directories(
    qa_hal
    BEFORE
    PRIVATE
        ${HAL_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/tt-2xx/quasar
)
target_link_libraries(
    qa_hal
    PRIVATE
        Tracy::TracyClient
        hw
        fmt::fmt-header-only
        umd::device
        TT::STL
        tt-logger::tt-logger
)

add_library(hal_2xx STATIC)
add_library(HAL::2xx ALIAS hal_2xx)
target_sources(hal_2xx PRIVATE hal_2xx_common.cpp)
target_include_directories(hal_2xx BEFORE PRIVATE ${HAL_INCLUDE_DIRS})
target_link_libraries(
    hal_2xx
    PUBLIC
        qa_hal
    PRIVATE
        fmt::fmt-header-only
        hw
        umd::device
        TT::STL
        tt-logger::tt-logger
)
