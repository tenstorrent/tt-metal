set(LLRT_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/llrt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rtoptions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tlb_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tt_cluster.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tt_elffile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tt_memory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/grayskull/gs_hal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wormhole/wh_hal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wormhole/wh_hal_tensix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wormhole/wh_hal_active_eth.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wormhole/wh_hal_idle_eth.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/blackhole/bh_hal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/blackhole/bh_hal_tensix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/blackhole/bh_hal_active_eth.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/blackhole/bh_hal_idle_eth.cpp
)

add_library(llrt STATIC ${LLRT_SRC}) # FIXME(14541): Should be OBJECT, but can't handle circular deps between Object libs
add_library(Metalium::Metal::LLRT ALIAS llrt)

target_link_libraries(
    llrt
    PUBLIC
        Metalium::Metal::Common
        umd::device
        Metalium::Metal::Hardware
)
target_compile_options(llrt PRIVATE -Wno-int-to-pointer-cast)
