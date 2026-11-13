# Keraunos to Mimir GDDR

Working note for the MMK emulator bring-up. The open problem is a read from Keraunos HSIO2 CCE0 back to M1 GDDR. The write in that direction works. Both ways of reading it hang.

## Goal

Run a tt-metal ethernet kernel on Keraunos, on the MMK package (M0, M1, and K0), and reach Mimir memory over D2D.

The link under test is K0 D2D2 to M1 D2D1. HSIO2 owns that D2D. The ethernet core is HSIO2 CCE0, descriptor coordinate `(4, 2)`.

The address a Mimir CCE uses for M1 GDDR is the D2D1-route SPA:

```
0x1002200000000 + offset
```

That is route base `0x1002000000000` plus one 8 GiB instance stride. The host reads the same bytes through the chiplet-local GDDR window on M1.

## What already works

Bring-up is `validation/metal_bringup/bringup_mmk.py` on chippy branch `kstevens/metal_bringup`. It releases all four MMK D2D CPUs, in order: M1 D2D0, M1 D2D1, K0 D2D2, M0 D2D1. HSIO tiles 0–3 are out of subsystem reset. HSIO2 CCE0 has no DMRISC window over the GDDR SPA. A matching window would deliver the store on the CCE's local bus, where Keraunos has no GDDR. The store has to miss and leave on the NOC.

These `KeraunosEmu` tests pass. None of them uses IDMA. Each one is a CPU store plus a fence.

| Test | What it does |
| --- | --- |
| `HartZeroRunsEthernetKernelWritesLocalSram` | Hart 0 writes HSIO0 CCE0 SRAM |
| `AllHartsRunEthernetKernelWritesLocalSram` | All 8 harts write HSIO0 CCE0 SRAM |
| `EveryEthernetCoreWritesLocalSram` | The same local write on every ethernet core |
| `EthernetKernelWritesOtherTileSram` | HSIO0 stores into the other HSIO tiles' SRAM at `0x1206800000 + tile * stride` |
| `EthernetKernelWritesMimirGddr` | HSIO2 CCE0 stores one word at `0x1002200000000 + unreserved offset`. The host reads that word back from M1 GDDR |

The representative Mimir suite also passes on this same bring-up, including the M0 to M1 GDDR copies and the cross-Mimir CCE SRAM stores. Releasing K0 D2D2 used to stall that M0 to M1 path. It no longer does.

Kernels:

- `tests/tt_metal/tt_metal/test_kernels/misc/eth_write_tile_sram.cpp`
- `tests/tt_metal/tt_metal/test_kernels/misc/eth_write_mimir_gddr.cpp`

## The test that hangs

`KeraunosEmu.EthernetKernelReadsMimirGddr` in `tests/tt_metal/tt_metal/context/test_mimir_emu.cpp`.

The host writes one DRAM-alignment block into M1 GDDR, launches an idle ethernet kernel on HSIO2 CCE0, and then reads that core's SRAM. `LaunchProgram` does not return. With `TT_METAL_OPERATION_TIMEOUT_SECONDS` unset, the host polls forever (`timeout_duration_for_operations` defaults to 0, and `wait_until_cores_done` treats 0 as no timeout).

Two kernels have been tried. Both hang.

### 1. CPU load

`eth_read_mimir_gddr.cpp` used to be a `volatile` load of `0x1002200000000 + offset`, then a store of that word into the core's own uncached SRAM.

The hart never finished. A store of the same address does finish, and the data lands in M1 GDDR, so the request direction of that address works. A store retires without waiting for a reply. A load does not resume until data comes back to the hart.

This was not a missed DMRISC window. The earlier store failure (host read back 0) was that window: HSIO2 CCE0 had an identity DMRISC entry over `[0x1002200000000, 0x1002400000000)`, so the store completed locally. Removing it made `EthernetKernelWritesMimirGddr` pass. The load still hung after that.

### 2. IDMA copy

The current `eth_read_mimir_gddr.cpp` issues one IDMA copy on command buffer 0 and waits on `idma_acked_cmdbuf_0()`:

- source: `0x1002200000000 + unreserved GDDR offset`
- destination: HSIO2 CCE0's own SRAM as a global address, `0x1206800000 + hsio * tile_stride + cce * eth_l1_size + unreserved SRAM offset`
- length: one DRAM alignment (the block the host wrote)

The kernel compiled and the test ran. It hangs in the ack wait, or on the copy itself. The host never gets to the SRAM compare.

IDMA was chosen because SysEng's `ker_cce_dma_m2k_remap` reads Mimir SRAM from this same core with IDMA, in both directions (`validation/firmware/apps/sip/k2m_remap/ker_cce_dma_m2k_remap/src/main.c`). That firmware reads M0 CCE0 SRAM and M1 CCE1 SRAM. It does not read GDDR, and it does not use a CPU load. Metal's IDMA helpers are `tt_metal/hw/inc/internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp`. Their existing examples are local L1 copies on other core types.

The cross-HSIO test was not a precedent for this. `EthernetKernelWritesOtherTileSram` is a CPU store, and it only writes.

## What this does and does not show

Shown:

- A CPU store from HSIO2 CCE0 to the M1 D2D1-route GDDR SPA reaches M1 GDDR.
- A CPU load of that address does not let the hart finish.
- An IDMA copy from that address into HSIO2's global SRAM window does not let the hart finish either.
- Mimir CCEs can load the other Mimir's GDDR with an ordinary CPU load (`gddr_cce_read` in `tt_metal/hw/inc/experimental/gddr_cce.h`). That path is M0 D2D1 to M1 D2D0, not K0 D2D2.

Not shown:

- That the read response is dropped. The hang only says the hart is still waiting. The wait could be the IDMA ack, a stalled command issue, or a response that never arrives.
- That IDMA on an idle ethernet hart works at all. Nothing in this suite has completed an IDMA on HSIO2 CCE0, even to its own SRAM.
- That Keraunos can read Mimir SRAM. SysEng's firmware does. This suite has not.

## Not tried

- An IDMA copy whose source and destination are both in HSIO2 CCE0 SRAM. That would show whether IDMA itself runs on this hart before involving D2D.
- An IDMA read of Mimir CCE SRAM, which is the transfer SysEng's firmware actually does, instead of GDDR.
- A Mimir CCE store into HSIO SRAM, the other direction across K0 D2D2.
- Any change to the Keraunos D2D ATT, BARs, or HSIO noc2axi remap for this read. Bring-up still programs the Mimir DRAM ATT entry on K0 D2D2 (`translate_addr=1`, compare `0x1000000000000`, BAR `0x800000000`). That is the table the passing store uses.

## Running it

```
TT_METAL_OPERATION_TIMEOUT_SECONDS=60 TT_METAL_SLOW_DISPATCH_MODE=1 \
  ./build/test/tt_metal/unit_tests_context \
  --gtest_filter=KeraunosEmu.EthernetKernelReadsMimirGddr
```

A hang from the earlier CPU-load attempt can leave the hart mid-transaction. `bringup_mmk.py` re-holds the CCE harts, but it does not clear an outstanding bus transaction. If `EthernetKernelWritesMimirGddr` also hangs, the emulator model needs a restart and a fresh `bringup_mmk.py`.
