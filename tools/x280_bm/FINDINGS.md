# X280 Bare-Metal Profiler-Drain — Findings

Bare-metal firmware for the Blackhole **X280 (L2CPU)** used as a non-invasive
**pull profiler**: drain Tensix per-core data over the NoC into the X280, with no
OS. This documents what we built and measured on **bh-11** (single-chip P100A,
11×10 = 110 worker cores), branch `mo/x280_bm_fw`.

All firmware lives in `tools/x280_bm/` (riscv64 bare-metal, no libc). Each host
driver is a tt-metal **C++ programming example** under
`tt_metal/programming_examples/profiler/` that drives the low-level `Cluster`
(single UMD access path — no pyluwen, which can't share a process with UMD).

---

## TL;DR

- **Bare-metal beats Linux on peak bandwidth: 530 MB/s vs 430 MB/s**, because all
  4 harts are free (Linux must spend the 4th on the OS, where it *collapsed*).
- **530 MB/s is a hard hardware wall** — the shared **mesh-link ingress into the
  L2CPU tile**, downstream of both NIUs. Independent of NoC, NIU, DMA, or read
  width. Nothing X280-side beats it.
- The DMA's value is **core-offload** (drains the grid with the harts asleep),
  not throughput.

| Config | Throughput | Cost |
|---|---|---|
| Linux, 3 harts (prior work) | 430 MB/s | OS takes the 4th hart |
| **bare-metal 4-hart vector poll** | **530 MB/s** | all harts busy — optimal |
| 4 harts split 2/2 across NoC0/NoC1 | 530 MB/s | no gain (mesh-bound) |
| 4 harts NOC0 + 2 DMA on NOC1 | 464 MB/s | worse (mesh contention) |
| bigger vector bursts (8 flits/load) | 274 MB/s | worse (LSU serializes) |
| 1 hart, 2-channel DMA | 111 MB/s | **cores free** |

---

## Build & run

Firmware (auto-fetches a matched `riscv64-unknown-elf` toolchain to
`/localdev/$USER/x280-toolchain`, outside the synced tree):

```sh
tt run bh-11 -- 'make -C tools/x280_bm'
```

Host examples (need a consistent tt-metal build of the current branch):

```sh
tt run bh-11 -- 'cd /localdev/mmemarian/tt-metal && ~/.local/bin/cmake --build build_Release --target <example>'
# run from repo root, libs on the path:
tt run bh-11 -- 'cd /localdev/mmemarian/tt-metal && export TT_METAL_HOME=$PWD && \
  export LD_LIBRARY_PATH=$(find build_Release -name "*.so*" -type f -exec dirname {} \; | sort -u | tr "\n" ":")$LD_LIBRARY_PATH && \
  ./build_Release/programming_examples/profiler/<example> [flags]'
```

Each host does `tt-smi -r` (board reset) before opening the device, so the L2CPU
is in a resettable state and any prior X280 firmware is torn down.

---

## Boot model (shared by all firmware)

The host (C++/UMD `Cluster`) is the **bootloader/supervisor**; the X280 hart
**runs the firmware autonomously**. There is no OS and no resident bootloader.

1. `tt-smi -r` (board reset) — L2CPU must be in reset to be (re)loadable; you
   cannot re-assert reset on a *running* L2CPU (HW: release-from-reset once per
   chip reset).
2. Hold L2CPU in reset: clear bit `(4+idx)` of ARC reset reg `0x80030014`
   (NoC reg write to ARC tile `(8,0)`).
3. NoC-write `*.bin` into L3 LIM at `0x08000000` (tile `(8,3)`).
4. Write the 4 hart reset vectors (`L2CPU_REG_BASE = 0xFFFFF7FEFFF10000`).
5. Step the L2CPU PLL (PLL4 `0x80020500`) to 1000 MHz.
6. Release reset → all 4 harts start at `0x08000000` → `entry.S` → `main(hartid)`.

Supporting files:
- `boot/entry.S` — minimal boot stub (gp, mtvec, feature-enable, `mstatus.VS`,
  per-hart sp at **4 KiB** stride, bss-zero, `call main(hartid)`).
- `ld/x280-lim.ld` — links at LIM `0x08000000`.
- `include/noc.h` — 2 MiB NoC TLB programming + read helpers (vendored).
- `include/dma_engine.h` — Synopsys DW DMAC driver (vendored).

---

## Experiments (chronological)

### 1. Single-hart heartbeat — proof of life
- FW `src/counter.c` · example `test_x280_counter`
- Hart 0 increments a u64 in LIM (`0x08010000`) ~every 1 ms; host polls it.
- **Result:** counter climbs +100/100 ms, monotonic → FW boots and runs.

### 2. Single-counter poll rate — the read latency
- FW `src/poller.c` · example `test_x280_poll_rate`
  (+ producer kernel `test_x280_poll_rate/kernels/brisc_counter.cpp`)
- A BRISC counter runs on Tensix (0,0); the X280 reads it over a NoC TLB window
  as fast as it can.
- **Result: 248.9 ns/read, ~4.0 M reads/s, 16 MB/s** for a single uncached u32
  read — pure NoC round-trip latency; matches the Linux-era ~234–260 ns. The
  in-order X280 stalls one read at a time. **Bug found:** a non-returning BRISC
  kernel must `fence` after its L1 store or the writes never drain to NoC-visible
  L1 (a returning kernel drains via tt-metal's epilogue).

### 3. DMA NoC→LIM validation + re-trigger
- FW `src/dma_probe.c` · example `test_x280_dma_probe`
- The X280 DMAC (`dma_engine_noc_to_x280`, EXTERN→L2) pulls a Tensix L1 region
  into LIM with no core loads.
- **Results:** correct (`rc=0`, bytes match). Per-transfer **setup ≈ 7,305
  cycles is one-time**: with `--repeats`, re-triggering the same descriptor
  (restore block_ts/SAR/DAR + kick) costs only 18,268 cyc for 2 KB (vs 25,573
  with setup) → ~112 MB/s steady. Setup-once gives ~10× for small repeated reads.
  Size sweep: `cycles ≈ 7,600 + ~570/flit`.

### 4. Whole-grid drain (1 channel)
- FW `src/grid_drain.c` · example `test_x280_grid_drain`
- One hart, one DMA channel, walks all 110 cores per round (reprogram NoC TLB +
  re-trigger), 256 B (4 flits) each into per-core LIM slots.
- **Result: 96 MB/s, 110/110 identity + liveness OK**, ~292 µs/full-grid
  snapshot, core-free. **Found:** the X280 NoC TLB needs the **translated/virtual
  coord** for Tensix, not the raw physical one (physical only coincides on the
  unharvested low rows).

### 5. Multi-channel drain — the DMAC has 2 channels
- FW `src/grid_drain4.c` · example `test_x280_grid_drain4`
- Self-tests channels 0–3 (timeout-protected), drains with the working set.
- **Result: chan_mask = 0x3 → only 2 DMA channels exist** (ch1 working validates
  the `0x58` per-channel register stride, so ch2/3 absence is real). 2-channel
  drain: **111 MB/s** (vs 96 single) — only ~1.15×, limited by the shared NIU.

### 6. 4-hart vector poll — beats Linux (530 MB/s)
- FW `src/poll4.c` · example `test_x280_poll4`
- All 4 harts vector-poll (`vle64.v`, one 64 B flit = one NoC transaction) their
  quarter of the grid.
- **Result: 530 MB/s, 110/110 identity + liveness**, ~13.3 µs/full-grid snapshot.
  **Beats Linux's 430 MB/s** because the 4th hart (which collapsed Linux) is free.
  **Critical:** scalar 8×u64 loads serialize (71 MB/s); the single `vle64`
  vector load reads the whole flit in one transaction (the lever). Needs 4 KiB
  per-hart stacks (the old 32 KiB stride put hart 3 below LIM).

### 7. NIU saturation — 530 is the wall (NOC0)
- FW `src/poll6.c` (since deleted) · earlier `test_x280_poll6`
- "6 pipes" = 4 harts + 2 DMA, all on NOC0, fixed duration.
- **Result: 463 MB/s (worse than 530)** — the DMA contends with the harts at the
  single NIU; adding pipes past 4 harts degrades. 4-hart poll already saturates.

### 8. NoC1 reachability probe
- FW `src/noc1_probe.c` · example `test_x280_noc1_probe`
- DMA-reads core 0 three ways (timeout-safe) to learn the NOC1 addressing.
- **Result:** the X280 TLB reaches a tile on **NOC1 via `noc_selector=1` + the
  SAME translated coord**; the explicit physical NOC1 coord (e.g. core(0,0)
  NOC1 = (15,9), both axes mirrored on the 17×12 grid) **times out**. Get coords
  via `soc_desc.translate_coord_to(CoreCoord(.,.,TENSIX,LOGICAL), CoordSystem::NOC1)`.

### 9. Dual-NoC split + burst depth
- FW `src/poll4n1.c` · example `test_x280_poll4n1` (`--noc1 N`, `--flits N`)
- **4-hart 2/2 NoC split: 530 MB/s — no gain.** Each hart is latency-bound at
  ~135 MB/s; 4 harts on one NIU show zero contention, so spreading to the 2nd NIU
  relieves nothing. **Bigger bursts (`vl=64`, 8 flits/load): 274 MB/s — worse.**
  The in-order vector LSU doesn't pipeline multiple outstanding uncached reads.

### 10. NIU-vs-mesh decider — confirms the mesh wall
- FW `src/poll6n1.c` · example `test_x280_poll6n1`
- 4 harts poll NOC0 + 2 DMA channels on **NOC1** (the other NIU), fixed duration.
- **Result: 464 MB/s (worse than 530).** Smoking gun: harts 1–3 (pure NOC0 loads,
  the *other* NIU, untouched) dropped **132 → 105 MB/s** the instant the NOC1 DMA
  ran. If the NIU were the limit, NOC0 harts wouldn't care about NOC1 traffic →
  the contention is **downstream of both NIUs = the shared mesh ingress into the
  L2CPU tile.** **530 MB/s is that ingress ceiling.**

---

## Hardware facts established

- **NoC read ceiling ≈ 530 MB/s** per L2CPU tile, set by the shared mesh-link
  ingress (downstream of both NIUs). Independent of NoC/NIU/mechanism.
- **Single uncached read latency ≈ 249 ns** (NoC round-trip); in-order core keeps
  ~1 outstanding read → ~135 MB/s/hart, doesn't improve with wider vector loads.
- **DMAC has exactly 2 channels** (DW `ahb_dmac`, per-channel regs at
  `CH0 + N*0x58`; global SWHS/INT/CHEN use bit `N` + write-enable bit `8+N`).
- **DMA per-transfer setup ≈ 7,305 cycles is one-time** (re-trigger reuses it).
- **X280 NoC TLB uses translated/virtual coords**; NOC1 = same coord +
  `noc_selector=1`. Tensix `(8,3)`=L2CPU tile, ARC `(8,0)`, src L1 `0x80000`.
- **DMAC at `0x2FF80000`**; LIM at `0x08000000`; NoC TLB cfg `0x2FF00000`,
  2 MiB window data `0x430000000`.

## Gotchas (saved time → don't relearn)

- **`tt-smi -r` does NOT clear LIM SRAM** → a re-run of the same FW sees the prior
  run's `DONE_MAGIC` and reads results prematurely. Host must zero the result/done
  region before releasing the FW.
- A non-returning BRISC kernel must `fence` after L1 stores (see §2).
- `-nostdlib`: aggregate `{0}` struct inits emit a `memset` call; zero fields
  explicitly and build with `-fno-tree-loop-distribute-patterns`.
- The box's `cmake` is `~/.local/bin/cmake` (4.2.3); `/usr/bin/cmake` is absent.
  A stale `build_Release` from a different commit causes Tracy-version / ABI
  mismatches — rebuild consistently for the current branch.

---

## File map

Firmware — `tools/x280_bm/`:
| File | Purpose |
|---|---|
| `boot/entry.S`, `ld/x280-lim.ld` | boot stub + linker (LIM @ 0x08000000) |
| `include/noc.h`, `include/dma_engine.h` | vendored NoC TLB + DMAC drivers |
| `src/counter.c` | §1 heartbeat |
| `src/poller.c` | §2 single-counter poll rate |
| `src/dma_probe.c` | §3 DMA NoC→LIM + re-trigger |
| `src/grid_drain.c` | §4 1-channel grid drain |
| `src/grid_drain4.c` | §5 multi-channel (found 2 channels) |
| `src/poll4.c` | §6 4-hart vector poll (530) |
| `src/noc1_probe.c` | §8 NoC1 reachability |
| `src/poll4n1.c` | §9 dual-NoC split + burst depth |
| `src/poll6n1.c` | §10 4 harts NOC0 + 2 DMA NOC1 |

Host examples — `tt_metal/programming_examples/profiler/`:
`test_x280_counter`, `test_x280_poll_rate` (+`kernels/brisc_counter.cpp`),
`test_x280_dma_probe`, `test_x280_grid_drain`, `test_x280_grid_drain4`,
`test_x280_poll4`, `test_x280_noc1_probe`, `test_x280_poll4n1`,
`test_x280_poll6n1`.
