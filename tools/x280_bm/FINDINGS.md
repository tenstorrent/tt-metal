# X280 Bare-Metal Profiler-Drain — Findings

Bare-metal firmware for the Blackhole **X280 (L2CPU)** used as a non-invasive
**pull profiler**: drain Tensix per-core data over the NoC into the X280, with no
OS. This documents what we built and measured on **bh-11** (single-chip P100A,
11×10 = 110 worker cores), branch `mo/x280_bm_fw`.

**Reproduced on a second box (bh-8 / `yyzo-bh-08`, 2026-06-25):** after a clean
`build_Release` (the box had a stale mixed build — see Gotchas), the boot recipe
and all headline numbers came back identical — 4-hart poll **530 MB/s**
(135/135/135/132 per hart), the chunk-size cliff (1 flit 530 → 2–8 flits ~274),
and the mesh-wall decider **464 MB/s** with the harts dragged down by NOC1 DMA.
The scatter-poll numbers are **not box-specific**. (bh-8 then went further and
broke the "530 wall" with ILP — see §11.)

All firmware lives in `tools/x280_bm/` (riscv64 bare-metal, no libc). Each host
driver is a tt-metal **C++ programming example** under
`tt_metal/programming_examples/profiler/` that drives the low-level `Cluster`
(single UMD access path — no pyluwen, which can't share a process with UMD).

---

## TL;DR

- **⚠️ The "530 MB/s wall" was an artifact. The real profiler does 1533 MB/s.**
  530 was only the *latency-bound* regime: each hart kept **one outstanding NoC
  read** (1 flit/core, read→consume→next). Issue **multiple reads in flight** (ILP)
  and the actual 110-core scatter drain (`gridilp`, §12) hits **1533 MB/s with 2
  harts** (110/110 verified) — **2.9×**, using only half the harts. Even 1 hart ×
  ILP 4 = **860**, beating the old 4-hart number with 3 harts idle. (Synthetic
  sequential streaming peaks a bit higher, ~1.8 GB/s, §11.)
- **The §7/§10 "shared mesh ingress wall" was measuring the 1-outstanding regime,
  not the link.** It was never a bandwidth ceiling.
- **Concurrency is the lever — not port, VC, or read width.** Cached Memory Port was
  *worse* (472 < 793) and static-VC did nothing (§11). What scales is reads-in-flight
  = harts × per-hart ILP, **up to a point**: **≥3 harts with ILP≥4 collapses**
  (resource thrash; 4h×4 → 277). Sweet spot: **few harts (1–2), ILP 4**.
- The DMA's value is **core-offload** (drains the grid with the harts asleep),
  not throughput.
- **Export (D2H) — X280 → host: ~3.0 GB/s sustained, ~11× the Linux 268 ceiling
  (§13).** Posted `vse64` (64 B) writes through the PCIe tile into host pinned
  memory; **1 hart is optimal** (egress-bound — more harts don't help), all data
  verified in host. The old 268 was scalar 8 B stores under Linux.
- **Continuous consumer (the end-to-end profiler drain) — ~1.2 GB/s (§15/§16).** X280
  drains all 110×5 device-profiler L1 rings live (lossless, flow-controlled) and
  relays to host. Best config = **2 readers (NOC0) + 1 relay (NOC1), batched =
  ~1.2 GB/s**, leaving the 4th hart free. The wall is **shared LIM SRAM bandwidth**
  (every flit crosses LIM twice: reader writes in, relay reads out) — not the NoC,
  not egress. Scatter reads (wash), deeper buffers (no gain), a 2nd relay hart (+6%),
  and dropping LIM entirely (`--direct`, *worse*: 753) were all tried and ruled out.

| Config | Throughput | Note |
|---|---|---|
| **X280 → host D2H write, 1 hart, vse64** | **~3.0 GB/s** | export ceiling, ~11× Linux 268 (§13) |
| **2 harts, 110-core scatter drain, ILP 4** | **1533 MB/s** | the real profiler read — 2.9×, 2 harts free (§12) |
| **Continuous consumer: 2 readers + 1 relay, batched** | **~1205 MB/s** | end-to-end read→stage→relay, lossless, chosen config (§16) |
| **1 hart, scatter drain, ILP 4** | **860 MB/s** | beats old 4-hart 530 with 3 harts idle |
| 2–3 harts, seq stream, ILP 8 | ~1.8 GB/s | synthetic peak (§11) |
| ≥3 harts, ILP ≥4 | ~280–460 MB/s | collapse — too many issuers |
| cached Memory Port, 1 hart | 472 MB/s | #3 long shot — worse than System Port |
| Linux, 3 harts (prior work) | 430 MB/s | OS takes the 4th hart |
| bare-metal 4-hart scatter poll, ILP 1 (poll4) | 530 MB/s | the original latency-bound profiler poll |
| 4 harts split 2/2 across NoC0/NoC1 | 530 MB/s | no gain in the 1-outstanding regime |
| 4 harts NOC0 + 2 DMA on NOC1 | 464 MB/s | scatter regime + DMA contention |
| bigger chunks (2–8 flits/*one* load) | 274 MB/s | worse — a wide load serialises (≠ ILP) |
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
  relieves nothing.
- **Chunk-size sweep (flits per `vle64`): a cliff, not a curve.** 1 flit (64 B) =
  530; 2/3/4/6/8 flits all flat at **274** (≈half). The in-order, single-issue
  vector LSU walks a multi-flit load as serial back-to-back NoC round-trips (no
  overlap), so per-byte time doubles at 2 flits; beyond 2 the per-instruction
  overhead is already amortized, hence flat. **64 B / 1 flit is the optimum.**
  Throughput scales with *outstanding transactions* (more harts), not bytes/read —
  one hart stalls on its single load no matter how wide it is.

### 10. NIU-vs-mesh decider — the scatter-regime ceiling (NOT the link)
- FW `src/poll6n1.c` · example `test_x280_poll6n1`
- 4 harts poll NOC0 + 2 DMA channels on **NOC1** (the other NIU), fixed duration.
- **Result: 464 MB/s (worse than 530).** Smoking gun: harts 1–3 (pure NOC0 loads,
  the *other* NIU, untouched) dropped **132 → 105 MB/s** the instant the NOC1 DMA
  ran. At the time this looked like a shared mesh-ingress wall at 530.
- **⚠️ Reinterpreted by §11:** this whole regime keeps only ~1 NoC read outstanding
  per hart (scatter, 1 flit/core, read→consume→next). 530 is the ceiling *of that
  regime*, not the link. Add ILP and the same hardware does ~1.8 GB/s. The §10
  contention is real but it's contention among **latency-bound issuers**, not a
  bandwidth wall.

### 11. Breaking the wall — ILP, cached port (#3), static VC (#4)
- FW `src/pollmp.c` · example `test_x280_pollmp`
  (`--nharts N --memport 0|1 --ilp 1|2|4|8 --vc N|--vc-spread --span B`)
- Each hart **streams** a large distinct linear region from one core's L1 (single
  pass — real NoC traffic on every line, no cache reuse), issuing `ilp` independent
  64 B vector loads per iteration before consuming them.
- **#3 cached Memory Port: failed.** 1 hart via the cacheable Memory Port = 472 MB/s,
  *worse* than the uncached System Port (793). The cache adds overhead and its
  prefetcher doesn't out-pace explicit ILP. **#4 static VC: failed** — pinning each
  hart to its own VC (`--vc-spread`) changed nothing (552 either way).
- **The real lever is ILP (reads in flight), on the plain uncached System Port.**
  Single hart, sequential, System Port: ILP 1→2→4→8 = **246 → 447 → 793 → 914 MB/s**.
  The earlier "1 outstanding read/hart" was a property of poll4's dependent loop,
  not the hardware.
- **nharts × ILP grid (System Port, MB/s):**

  | harts \ ILP | 1 | 2 | 4 | 8 |
  |---|---|---|---|---|
  | 1 | 246 | 447 | 792 | 914 |
  | 2 | 492 | 893 | 1583 | 1782 |
  | 3 | 533 | 1000 | 1765 | **1818** |
  | 4 | 533 | 999 | **276** | 274 |

- **Peak ≈ 1.8 GB/s** (2–3 harts, ILP 8) — **3.4× the old "530 wall".**
- **4 harts at ILP ≥4 collapses to ~275** (all four harts uniformly drop to ~69
  MB/s). The *same* 16 outstanding from 2 harts (2×8) gives 1782, so it's **4
  contending issuers**, not the outstanding count, that thrashes the port. Safe
  recipe: **2–3 harts, ILP 4–8.**
- Caveat: this is **sequential** streaming from a core's L1. The real profiler is
  *scatter* (1 flit from each of 110 cores) — see §12, which applies ILP there.

### 12. The real profiler — scatter drain + ILP (breaks 530 for real)
- FW `src/gridilp.c` · example `test_x280_gridilp`
  (`--nharts N --ilp 1|2|4|8 --nrounds N`)
- poll4's full 110-core drain (each hart owns a slice, 1 flit/core, identity +
  liveness verified against the live BRISC counters), but each hart issues `ilp`
  **independent reads to `ilp` different cores' windows** before draining them.
- **Scatter overlaps across windows just like the sequential stream did** — the
  profiler pattern is *not* inherently latency-bound. **`nharts=4 ilp=1` = 530 MB/s
  exactly reproduces poll4** (harness check). All configs kept **110/110 identity**.
- **nharts × ILP grid (110-core scatter drain, MB/s):**

  | harts \ ILP | 1 | 2 | 4 | 8 |
  |---|---|---|---|---|
  | 1 | 251 | 478 | 860 | 830 |
  | 2 | 487 | 915 | **1533** | 1371 |
  | 3 | 549 | 1051 | 455 | 555 |
  | 4 | 530 | 1043 | 277 | 281 |

- **Best profiler config: 2 harts × ILP 4 = 1533 MB/s (2.9× the old 530), using
  only 2 of 4 harts.** 1 hart × ILP 4 = 860 already beats the old 4-hart number with
  3 harts free.
- Collapse is sharper than the sequential case: **≥3 harts with ILP≥4 craters**
  (3h4=455, 4h4=277). Keep it to **1–2 harts, ILP 4**. (Slightly below §11's
  sequential peaks — scatter pays a little for spreading across 110 windows.)

### 13. Export direction — X280 → host D2H write BW (~3.0 GB/s)
- FW `src/d2hbw.c` · example `test_x280_d2hbw`
  (`--nharts N --ilp 1|2|4|8 --bytes B --nrounds N`)
- The other half of the profiler: the X280 fabricates fake 64 B packets and blasts
  **posted `vse64` NoC writes through the PCIe tile into host pinned memory**
  (sysmem ch 0). Addressing = PCIe tile **translated coord** (derived at runtime:
  `soc_desc.get_cores(PCIE, TRANSLATED)` → enc `0x613` = (19,24)) + host IOVA
  (`get_pcie_base_addr_from_device` + offset), `posted=1`. **Write-only** — a NoC
  *read* through the PCIe tile hangs the in-order hart, so the FW issues none.
  Verified end-to-end: a FOOTER flit is the final posted write; the host polls it
  via `read_sysmem` (host-side, safe), then checks the data pattern landed.
- **Sustained (512 MB single pass, all bytes verified in host): ~2.9–3.0 GB/s.**
  1-hart ILP 1→2→8 = 2660 → 3060 → 3068; **1 hart is optimal** — 2 harts 2472, 3–4
  harts 2286, i.e. *more harts don't help* → **egress-bound** (the PCIe-tile write
  path), not issue-bound. ILP gives only a ~15 % bump then saturates.
- **~11× the Linux D2H ceiling (268 MB/s @ 2 harts).** The gap is mostly the store
  width: Linux used scalar **8 B** stores; this FW uses **64 B `vse64`** flits → 8×
  fewer store instructions / 8× bigger NoC write transactions (268 × 8 ≈ 2.1 GB/s,
  same order), plus no OS jitter and no flow-control overhead.
- BW is computed from the FW's `rdcycle` (issue) window; corroborated as the true
  egress rate by (a) invariance to hart count and size, (b) ILP saturation, (c) the
  512 MB sustained run holding. A host-wall (`release → footer`) cross-check is the
  remaining belt-and-suspenders measurement if an independent number is wanted.

### 14. Closing the loop — relay the real device profiler to host (in progress)
Goal: a normal workload makes **all RISCs on all cores** emit device-profiler
timestamps to L1; the X280 drains those L1 profiler buffers (§12 scatter-read) and
relays them to the host D2H socket (§13 write) — replacing tt-metal's profiler
readback path — then validate via Tracy capture/`csvexport`.
- **Increment 1 DONE (bh-8):** `profiler_test_full_buffer` (built via
  `--build-metal-tests`) run with `TT_METAL_DEVICE_PROFILER=1` → **110/110 cores ×
  all 5 RISCs** (BRISC/NCRISC/TRISC0-2) emit FW + kernel zones; ground truth in
  `generated/profiler/.logs/profile_log_device.csv` (3.2 M markers, chip 1350 MHz).
- **L1 source layout (Blackhole):** `profiler_msg_t` lives in the L1 mailbox region
  (`MEM_MAILBOX_BASE=0x60` + offsetof; query via HAL `HalL1MemAddrType::PROFILER`).
  = `control_vector[32]` (128 B) + 5 × `buffer[risc]` (2048 B) = 10368 B/core.
  Marker = 8 B: `data[i] = 0x80000000 | (timer_id<<12) | time_H`, `data[i+1] =
  time_L` (44-bit wall clock). **Guaranteed FW/kernel start+end markers at fixed
  offsets `buffer[risc]+0x10..0x2F`** — persist until next launch, so the X280
  drains a clean live snapshot. Each RISC L1 buffer is only 2048 B (~256 markers);
  the run's 3.2 M total came from the profiler draining/refilling repeatedly.
- **Increment 2 DONE + VERIFIED (bh-8):** FW `src/profrelay.c` + host
  `test_x280_profrelay`. Host launches the all-5-RISC workload (`full_buffer` kernels
  on RISCV_0/1 + compute, full grid, blocking), leaves L1 intact, queries
  `hal.get_dev_addr(TENSIX, PROFILER)` (= L1 `0xb50`), boots the X280; the FW drains
  all 110 cores' `profiler_msg_t` (10496 B/core) and posted-writes them to host D2H.
  **Result: 1,154,560 B relayed @ 493 MB/s (pull-bound); relayed core-0 == direct
  `read_core` EXACT MATCH; 550/550 (core×RISC) valid FW+kernel markers, ordering
  sane.** The X280 replaces tt-metal's profiler readback, proven bit-exact. Relayed
  bytes dumped to `$TT_METAL_PROFILER_DIR/x280_relayed_profiler.bin` for Tracy.
- **Increment 3 DONE (bh-8):** built `tracy-capture` + `tracy-csvexport`
  (`cmake --build build_Release --target tracy-capture tracy-csvexport` →
  `build_Release/tools/profiler/bin/`). Ran the profiler workload with
  `TT_METAL_DEVICE_PROFILER=1 TRACY_NO_EXIT=1`, `tracy-capture` pulled **3,215,157
  zones** → `.tracy` (20 MB), `tracy-csvexport` → CSV showing all-RISC FW+kernel
  zones (BRISC-FW/KERNEL, NCRISC-FW/KERNEL, TRISC-FW×3 / TRISC-KERNEL×3, TEST-FULL).
- **LOOP CLOSED:** all RISCs/all cores emit profiler timestamps → X280 relays them
  to host D2H (bit-exact vs `read_core`) → Tracy capture/`csvexport` validates the
  same FW/kernel zones. The X280-relayed bytes are bit-identical to the device
  profiler source that feeds Tracy, so the relay reproduces the Tracy-exported data.
  (bh-8 is single-chip / no `tt-run`+MPI, so the literal multi-host smoke test can't
  run multi-rank here; the loop is validated single-host.)

### 15. Continuous SPSC profiler — producer backend + live X280 consumer

§14 relayed a *snapshot*. To make it continuous and lossless, the device profiler's
on-device backend was swapped to an SPSC ring drained live by the X280.

- **Producer backend (`tt_metal/tools/profiler/kernel_profiler.hpp`, SPSC variant).**
  The original DRAM-push version is preserved verbatim as `kernel_profiler_push.hpp`;
  the new one keeps the exact public macro API (`DeviceZoneScopedN`/`MainN`/…) but
  each RISC streams markers into its **per-RISC L1 ring** (reuses the existing
  `profiler_msg_t`; tail = `DEVICE_BUFFER_END_INDEX_BR_ER+r`, head =
  `HOST_BUFFER_END_INDEX_BR_ER+r`, monotonic word counts, storage `% 512`). Each
  append **blocks** (`invalidate_l1_cache` spin) while the ring is full; `quick_push`/
  `finish` do **no DRAM**. So a profiled run now *requires* the X280 draining, and the
  stream is lossless/flow-controlled. (Verified: JIT-compiles for all 5 Tensix RISCs.)
- **Consumer `src/profcons.c` + `test_x280_profcons` — flow control proven.** X280
  drains all 110×5 rings, advances each head (unblocking producers), relays to host.
  Run a workload that **overflows** the rings (LOOP_COUNT 150 → ~608 words/ring > 512
  cap): without a consumer it would deadlock; with it, the workload **completes**,
  **max-outstanding pinned at 512** (never overruns), **334,400 produced == drained
  (lossless)**. That completion is the flow-control proof.
- **Drain throughput (bench: `--bench`/`--bench-ro`, no producers).** Naive single-hart
  fused read+relay = **327 MB/s**; the bottleneck was per-flit interleave + ILP-1.
  Lessons (each is a real lever, in impact order):
  1. **Every stage must be ILP-4** (4 NoC transactions in flight): an ILP-1 stage caps
     the pipe. Read-only sweep: 1h 388 / 2h 748 / 3h 96 (collapse).
  2. **Decouple read from relay onto separate harts** (`src/profcons_split.c`,
     `test_x280_profsplit`): reader harts drain rings → per-reader **LIM staging SPSC
     ring** → dedicated **relay hart** posted-writes to host. No per-flit read↔write
     dependency.
  3. **Split the NoCs**: reads on **NOC0**, relay writes on **NOC1** (`noc_selector=1`;
     NOC1→PCIe→host verified via footer) → ingress and egress don't contend.
- **Result (`profcons_split`, 2 readers ILP-4 NOC0 + 1 relay ILP-4 NOC1): 1097 MB/s**,
  lossless, NOC1-verified — **3.4× the fused 327, past the 748 read-only mark, ~72% of
  gridilp's 1534**. Progression: 327 → 421 (decouple/NOC0) → 406 (reader-ILP4 alone:
  no gain, still relay-bound, proven by 1r≈2r) → **1097 (relay-ILP4 was the unlock)**.
  1 reader = 747; 3 readers = 1070 (2 is the knee). **2 relays is invalid** in this
  design (both consume one per-reader ring → SPMC race; the consistency check catches
  it). The single relay hart is the current cap; correct multi-relay (partition reader
  rings, one consumer each) is the path toward ~1.5 GB/s.

### 16. Consumer throughput — chasing the relay wall (bh-08)

Reproduced §15 on bh-08, then tried to push the split consumer past 1097 toward the
~1.5 GB/s read ceiling. Every lever was characterized with A/B benches added to
`profcons_split` (`--ro`/`--ro-contig` read-only, `--direct` no-LIM). **Net: the wall
is shared LIM SRAM bandwidth, and 2 readers + 1 relay ≈ 1.2 GB/s is the practical
best.**

Every approach tried (end-to-end = read→stage→relay→host, lossless, reps=200; all
`consistent ✓` + NOC1 footer ✓ unless noted):

| # | Approach | Config | MB/s | Verdict |
|---|---|---|---|---|
| a | Fused single-hart (read+relay, per-flit SPSC) | 1 hart | 327 | baseline; per-flit interleave + ILP-1 |
| b | Decouple read↔relay onto separate harts, both NOC0 | 2r+1relay | 421 | helped, but relay still ILP-1 |
| c | + reader ILP-4 (relay still ILP-1) | 2r+1relay | 406 | no gain → relay-bound (1r≈2r) |
| d | + relay ILP-4 **and** relay→NOC1 (per-flit `lim4_to_host4`) | 2r+1relay | **1097** | §15 peak; two-sided ILP + NoC split |
| e | per-flit relay, fewer/more readers | 1r+1relay / 3r+1relay | 747 / 1070 | 2 readers is the knee |
| f | **Reader scatter** (half→4 quarters, 4 cores/group) | 2r+1relay | 1086 | wash vs (d) — read was never the bottleneck |
| g | **Batched relay** (per-ring descriptor + wide `m8` contiguous copy) | 2r+1relay | **1197** | +10%; amortizes dst-read 64→1, contiguous bursts |
| h | + 2-way ILP on the copy | 2r+1relay | 1206 | no gain → relay write-path not the bottleneck |
| i | + deep flit buffer (8→64 rings) | 2r+1relay | 1207 | no gain → not stalls; steady-state bound |
| j | **Two independent pipelines** (1:1 reader↔relay, separate buffers) | 2r+2relay | **1267** | correct multi-relay (no SPMC race), but only +6% |
| k | batched relay, 3 readers | 3r+1relay | 1192 | 3rd reader doesn't help |
| l | **Direct grid→host, no LIM** (`--direct`) | 2 harts | 753 | *worse* — read/write interleave stalls 1 in-order hart |
| m | Direct grid→host, no LIM | 4 harts | 294 | bidirectional NoC congestion |
| n | **DMA relay** (DMAC LIM→host via PCIe tile, `--dma-egress`) | 1 ch egress | 412 | *worse* — 7× slower than hart vse64; DMA = core-offload, not BW |
| **o** | **Batched, compact buffer — CHOSEN** | **2r+1relay** | **~1205** | best value; 4th hart free, +6% not worth a hart |

Read-only references (readers only, no relay — isolates the read path):

| Approach | Config | MB/s |
|---|---|---|
| Contiguous ILP-4 read (`--ro-contig`) | 2 harts | **1495** |
| Quarter-scatter read (`--ro`) | 2 harts | 1423 |
| Contiguous ILP-4 read | 1 hart | 776 |
| Quarter-scatter read | 1 hart | 748 |

What was learned, in order:

- **Scatter reads are a wash; the read was never the bottleneck.** Hypothesis: read
  4 flits from 4 *different* cores per ILP-4 group (split each reader's half-grid into
  4 quarters) would beat 4 consecutive flits at one core. Read-only A/B (no relay):
  contiguous **1495** vs scatter **1423** @ 2 harts — scatter slightly *worse* (extra
  TLB-window churn). The X280 System Port already overlaps multiple outstanding reads
  to the *same* endpoint, so endpoint diversity is no lever. The earlier "748 read
  ceiling" was `profcons --bench-ro` measuring read+full-SPSC-bookkeeping; the **raw**
  read ceiling is ≈ **1490 @ 2 harts** (≈ gridilp's 1532, pattern-independent).
  Production reader reverted to plain contiguous `read4_store4`.
- **Relay batching: 1097 → 1197.** New staging layout — a **contiguous flit ring** +
  a **per-ring descriptor ring** (one host `dst_start` per 64-flit ring). The relay
  reads one descriptor + does one **batched wide contiguous copy** of the whole 4 KB
  ring to host (`copy_contig`, RVV `e64,m8`, 2-way ILP), replacing 64× (per-flit
  dst-read + scattered 64 B posted write). Amortizes bookkeeping 64→1 and turns
  scattered writes into contiguous posted bursts.
- **Two nulls that pinned the diagnosis.** Adding 2-way ILP to the copy → 1206 (no
  gain ⇒ relay host-write path no longer the bottleneck). Deepening the flit buffer
  8→64 rings → 1207 (no gain ⇒ not handoff stalls/burstiness; steady-state bound).
- **Two independent pipelines (partitioned multi-relay) — the *correct* 2-relay.**
  Relay hart `hartid` drains a **disjoint** reader subset (`lo_r=r_idx*nread/nrelay`);
  with nrelay==nread that's a 1:1 reader↔relay pairing — own reader, own LIM buffer,
  own NOC1 window, own host slice, **no shared ring ⇒ no SPMC race** (the correct fix
  for §15's `consistent=NO`). Result: 2r+2relay = **1267, consistent ✓** — but only
  **+6%** over 2r+1relay (1195). Doubling relay harts barely helps ⇒ **not relay-hart
  capacity**.
- **Direct grid→host (drop LIM entirely) — FAILED, and that's the key insight.** Every
  hart reads its cores (NOC0) and posted-writes each flit straight to host (NOC1), no
  staging (`--direct`). Predicted to beat the LIM ceiling; instead **2 harts = 753,
  4 harts = 294** (footer ✓, correct, just slow). Each direct hart runs at ~376 ≈
  *half* the read-only hart rate: interleaving NoC reads + NoC writes on one **in-order**
  hart doesn't overlap — the posted `vse64` (NOC1) issue cost lands in the critical path
  and consumes the ILP-4 that was hiding read latency (~2× per-flit work). 4 harts
  congest the NoC bidirectionally (cf. gridilp 4-hart 276).
  **⇒ LIM staging isn't just a buffer: it lets each hart SPECIALIZE to one NoC direction
  in a tight unidirectional loop** (reader: NoC-read + cheap *local* LIM-store; relay:
  cheap *local* LIM-load + NoC-write). That specialization beats avoiding the 2× LIM
  crossing.
- **DMA relay ruled out (`--dma-egress`).** Tried replacing the relay hart with the
  Synopsys DMAC (LIM→host via the PCIe tile, EXTERN master, `dma_engine_x280_to_noc`).
  It works (rc=0, data lands in host) but only **412 MB/s** — ~7× slower than a hart's
  posted `vse64` egress (d2hbw 1-hart = 2779) and well below the 1205 hart-relay. The
  DMAC's per-block software handshake (enable→start→poll-done) is latency-bound, not
  bandwidth. So 2 readers + DMA-relay would cap at ~412 (≤~800 with both channels) —
  worse than the hart relay. Confirms §3/§5: **the DMA is core-offload, not throughput.**
- **The wall = shared LIM SRAM bandwidth.** The store-and-forward design crosses LIM
  twice (readers write each flit in, relay reads it out). Read-only (writes only) =
  1490; add the relay's concurrent LIM reads and aggregate LIM R+W saturates at
  ~half of ~2.5 GB/s ⇒ end-to-end pins at **~1.2–1.27 GB/s regardless of hart split**
  (1207 deep-buffer-1relay, 1267 two-pipeline). Not egress-bound (d2hbw 1h 3121 / 2h
  2442) nor NoC-read-bound (1490). Beating it needs avoiding the double-LIM-crossing,
  which `--direct` shows the in-order hart can't do.
- **Chosen production config: 2 readers + 1 relay ≈ 1205 MB/s** (lossless, NOC1 footer
  ✓), leaving the 4th hart free — a whole hart for +6% isn't worth it. `--ro`/
  `--ro-contig`/`--direct`/`--dma-egress`/`--latency` kept as documented diagnostic /
  negative-result probes.

### 17. Per-packet latency — L1 → host (bh-08)

Throughput (§16) is not latency. `--latency` (P_MODE 5) times, on hart 0's own rdcycle
clock (pll MHz), the components of one marker's trip; the host-write *landing* is not
device-observable and is estimated.

- **L1→X280 read = 271 ns** (one 64 B flit, NOC0, in-order round-trip) — the one hard
  hop, consistent with §2's 248.9 ns u32 read.
- **Posted host write = fire-and-forget at the hart** (~tens of ns to inject). A
  **non-posted** write was tried to force an ack-stall and measure the landing: it
  returned in **13 ns** (≈ rdcycle noise) — i.e. `vse64` retires on local injection
  **regardless of the posted bit**; the hart never stalls for a remote ack. Useful
  nulls: (a) a non-posted write to the PCIe tile does **not** hang (unlike a read), and
  (b) the **X280→host write-landing latency is not device-measurable** (no completion
  signal a hart can time; host-side timing is swamped by µs-scale driver overhead).

**End-to-end estimate, one marker through an (empty) pipeline:**

| Stage | Latency | Source |
|---|---|---|
| Producer publishes marker to L1 (store + fence) | ~30–50 ns | local, est. |
| Reader detects new tail (1 NoC ctrl read) | ~270 ns | measured |
| Reader reads the flit L1→X280 (NOC0) | **271 ns** | **measured** |
| LIM staging hop (reader store + relay load, local) | ~50–100 ns | local, est. |
| Relay issues host write (NOC1) | ~tens of ns | measured (issue) |
| PCIe write landing (X280→PCIe tile→host DRAM) | ~0.3–1 µs | **estimated** |
| **Total transit** | **≈ 1–1.5 µs** | |

- **Two regimes.** *Transit* (pipeline keeping up) ≈ **1–1.5 µs/marker**, dominated by
  the two NoC reads (~540 ns measured) + the estimated sub-µs PCIe landing. *Under load*
  **queueing dominates**: a marker waits in its L1 ring until the consumer sweeps to it;
  with ~550 rings backed up (~1.1 MB) draining at ~1.2 GB/s, worst-case ≈ **~900 µs**.
  Lightly loaded it collapses back toward the ~1–1.5 µs transit.

### 18. End-to-end to Tracy — the loop closes (bh-08)

The original goal: Tensix markers → X280 → host → **Tracy zones**. Done via direct emit
(Strategy B, modeled on `realtime_profiler_tracy_handler.cpp`) in `test_x280_profcons
--tracy`:

- After a continuous profiled run (`TT_METAL_DEVICE_PROFILER=1 --loop N`), the host reads
  every relayed `(core,risc)` slice from sysmem, parses the 2-word markers (`timer_id`,
  64-bit timestamp, packet type), and pushes them as device zones: `TracyTTContext()` →
  `TracyTTContextPopulate(0, global_min_ts, freq_GHz)` (one context per core, shared
  anchor) → `TracyTTPushStartMarker/EndMarker`. Needs the `ENABLE_TRACY=ON` build; the
  example's CMake links `TracyClient` (brings the `TRACY_ENABLE` define + tracy includes).
- **Markers must be timestamp-sorted per context** — gather a core's 5 RISCs, `stable_sort`
  by timestamp (START-before-END on ties) — else Tracy drops out-of-order device zones
  (this is why tt-metal has `getSortedDeviceMarkersVector`). This was THE bug: raw
  drain-order emit jumps backward at each RISC boundary.
- Headless capture: `tracy-capture -o x280.tracy -f` + run with `TRACY_NO_EXIT=1` (client
  flushes on exit). `tracy-csvexport` is **CPU-zone-only**, so device-zone correctness is
  proven by an in-tool START/END pairing + duration pass, and viewed in the Tracy GUI.
- **Verified (`--loop 50`, 1.35 GHz):** lossless drain 114,400 words; parsed **28,600
  START / 28,600 END → 28,600 matched pairs (0 unmatched)**; zone durations **min 172 /
  mean 577 / max 11,624 ns**; 57,200 markers pushed; `x280.tracy` saved (355 KB).
- First cut: ZONE_START/END only (skips TS_DATA/multi-word packets); zone names synthesized
  `x280_zone_<id>` (no kernel-source name map yet). `--tracy`/`--freq` flags on
  `test_x280_profcons`.

### 19. Productionizing — X280 drainer in device init, via a real D2H socket (bh-08)

Goal: make the SPSC kernel-profiler "just work" on this branch (it BLOCKS producers when a ring
fills, so a normal profiled run with no X280 drainer deadlocks). Plan: bring the X280 up at device
init and feed its drained markers to **Tracy through the existing real-time-profiler (RT) path**,
using a real tt-metal **D2H socket** as the transport. Pieces built + de-risked (all on bh-08):

- **X280 boots mid-session without a chip reset.** On a clean(ish) chip the L2CPU reset toggle
  alone boots + drains losslessly (`--no-reset`); a dirty (left-running) X280 needs a clean halt
  on close. So device-init boot is viable; teardown must assert L2CPU reset.
- **X280 firmware builds in the JIT phase.** Tensix `JitBuildState` is SFPI/rv32/HAL-bound and
  can't build the rv64gcv X280 FW, so a **sibling step** in `BuildEnvManager::build_firmware`
  (`build_env_manager.cpp`) shells the rv64 toolchain into the JIT cache (`.../firmware/x280/
  profcons.bin`), gated on profiler-enabled + Blackhole + L2CPU; toolchain (`TT_METAL_X280_TOOLCHAIN`)
  is a hard precondition. Verified building the `.bin` at device init.
- **`D2HSocket` can drive the X280 L2CPU as a *sender*.** It was Tensix-worker-specific in two
  spots (config write + the `bytes_acked` ack-write); a small `sender_is_l2cpu` extension
  (`d2h_socket.{hpp,cpp}`) targets the L2CPU via phys→virt translation + full LIM address, and
  skips the (nonexistent) static TLB for the dynamic `write_core` ack path. Verified: the
  `sender_socket_md` lands in the X280 LIM (`is_d2h=1`, `fifo_total=4096`, real host FIFO addr).
- **FW D2H-socket sender + BW bench** (`src/profsock.c`, host `--socktest`): the X280 reads the
  socket config from LIM and runs the real `reserve → 64 B page write (PCIe-tile NOC1) → push →
  notify` protocol; host drains via `socket.read()` (which acks back, closing flow control).

**BW measured (200k × 64 B pages, lossless):**

| Notify batch | D2H-socket push BW |
|---|---|
| 1 (per-page) | 773 MB/s |
| **8** | **839 MB/s** (knee) |
| 16 / 32 | 834 / 838 |
| 64 (whole FIFO) | 780 (reserve serializes vs the drain) |

- Batching the notify gains only ~8% ⇒ the notify wasn't the cost. The cap is the **host drain**:
  `socket.read()` reads ≤64 pages/call (4 KB FIFO) → ~3,100 calls for 200k pages, per-call driver
  overhead bounds it at **~840 MB/s** (the FW can push faster; the receiver read-loop is the limit).
- **~840 MB/s is ~70% of the raw-relay 1.2 GB/s, but immaterial in practice** — it's drain capacity,
  far above the actual zone-production rate.
- **⚠️ Revisit these rates:** a future commit will make the **host side faster** (e.g. a larger FIFO
  → more pages per `read()`, lower per-call overhead). The ~840 MB/s here is bounded by today's
  4 KB-FIFO / per-call `read()` cost, **not** the X280 sender — re-measure the socket BW after the
  host-side speedup lands.

Files (uncommitted): `build_env_manager.cpp`, `d2h_socket.{hpp,cpp}`, `src/profsock.c`, `Makefile`,
`test_x280_profcons.cpp` (`--derisk-socket`/`--socktest`/`--no-reset`). NEXT: on-device zone pairing
+ tagged RT page/record (type/core/risc/subdevice) + register the X280 socket with the RT receiver.

---

## Hardware facts established

- **X280 → host D2H write ceiling ≈ 3.0 GB/s** (1 hart, posted 64 B `vse64`,
  egress-bound at the PCIe tile; §13). ~11× the Linux 268 (which was scalar 8 B
  stores). Write-only: PCIe-tile *reads* hang the hart.

- **NoC read ceiling ≈ 1.8 GB/s** per L2CPU tile (2–3 harts, ILP 8, sequential;
  §11). The widely-quoted **530 MB/s is only the latency-bound *scatter* regime**
  (4 harts × ~1 outstanding read, 1 flit/core).
- **Throughput is set by reads-in-flight**, not port/VC/read-width: scales from 246
  (1 outstanding) to ~1.8 GB/s; **4 issuers at deep ILP thrash** (drops to ~275).
- **Single uncached read latency ≈ 249 ns** (NoC round-trip); with only ~1
  outstanding read a hart is latency-bound at ~135–246 MB/s — which is why ILP,
  not wider single loads, is what helps.
- **Cached Memory Port alias** `0x400430000000` works but is slower than the
  uncached System Port `0x430000000` for this access pattern.
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
| `src/poll4.c` | §6 4-hart scatter poll (530) |
| `src/noc1_probe.c` | §8 NoC1 reachability |
| `src/poll4n1.c` | §9 dual-NoC split + burst depth |
| `src/poll6n1.c` | §10 4 harts NOC0 + 2 DMA NOC1 |
| `src/pollmp.c` | §11 ILP / cached port / static VC — breaks the wall (~1.8 GB/s) |
| `src/gridilp.c` | §12 real 110-core scatter drain + ILP (1533 MB/s, 2 harts) |
| `src/d2hbw.c` | §13 X280→host D2H write BW (~3.0 GB/s, posted vse64 via PCIe tile) |
| `src/profrelay.c` | §14 relay device-profiler L1 snapshot → host (bit-exact, all RISCs/cores) |
| `src/profcons.c` | §15 continuous SPSC consumer (lossless flow control) + `--bench` |
| `src/profcons_split.c` | §15 reader/relay-hart split, two-sided ILP-4, NOC split (1097 MB/s) |

Producer backend — `tt_metal/tools/profiler/`:
| File | Purpose |
|---|---|
| `kernel_profiler.hpp` | §15 SPSC ring backend (per-RISC L1 ring, block-on-full, no DRAM) |
| `kernel_profiler_push.hpp` | original DRAM-push backend, preserved verbatim |

Host examples — `tt_metal/programming_examples/profiler/`:
`test_x280_counter`, `test_x280_poll_rate` (+`kernels/brisc_counter.cpp`),
`test_x280_dma_probe`, `test_x280_grid_drain`, `test_x280_grid_drain4`,
`test_x280_poll4`, `test_x280_noc1_probe`, `test_x280_poll4n1`,
`test_x280_poll6n1`, `test_x280_pollmp`, `test_x280_gridilp`, `test_x280_d2hbw`,
`test_x280_profrelay`, `test_x280_profcons`, `test_x280_profsplit`.
