set(FIRMWARE_JIT_API_FILES
    blackhole/noc.c
    wormhole/noc.c
    active_erisc.cc
    active_erisc-crt0.cc
    active_erisck.cc
    brisc.cc
    brisck.cc
    erisc.cc
    erisc-crt0.cc
    erisck.cc
    idle_erisc.cc
    idle_erisck.cc
    ncrisc.cc
    ncrisck.cc
    subordinate_erisc.cc
    tdma_xmov.c
    trisc.cc
    trisck.cc
    tt_eth_api.cpp
    # TODO: add erisc_cmac_gw.cpp here (currently at ../erisc_cmac_gw.cpp)
    # once the erisc cross-compiler toolchain target is wired up for this kernel.
)
