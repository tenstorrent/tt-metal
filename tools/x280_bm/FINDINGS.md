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

### 20. RT-profiler back-pressure — reader blind-polling is the wall (bh-11, 2026-07-10)

The productionized drainer is `src/profzone.c` (2 readers + 1 relay), driven by the
`RealtimeProfilerManager` and pushing through the real D2H socket to the RT Tracy handler.
Measured under a real 64-core workload (`python -m tracy -m -p pytest
tests/ttnn/tracy/test_trace_runs.py::test_with_ops`, `TT_METAL_X280_FORCE_PRIME=1`).

**Symptom:** the drainer sustains only **~250k markers/s (≈16 MB/s)** end-to-end, and because
`kernel_profiler.hpp` is lossless-BLOCKING (compute cores stall on a full L1 ring), the workload's
TRISC-KERNEL zones inflate **~22× avg / up to ~2700×** (min ~7 µs → max ~15–22 ms). Single-core is
clean. So the X280 drain rate directly throttles the compute.

**What was tried on the drain/signal side — all no help (numbers):**
- `bytes_sent` POSTED, batched once per N pages instead of per page: no change. It is NOT PCIe
  write-combining coalescing of *many* writes — host observes `avail>0` only ~160×/run regardless of
  write frequency (per-page 163, per-drain 160).
- `bytes_sent` NON-POSTED via a 2nd `posted=0` window (`WRITE_WIN+1`), batched every 256 pages: SAFE
  (no hang — a non-posted *write* is fine; only PCIe-tile *reads* hang), but no reliable throughput
  change. Reserve-stalls are pure NOISE (48k / 78k / 888k / 1.3M back-to-back, prime-state dependent).
- ILP reader: reader bulk-reads each contiguous ring segment off the System Port in one wide `vle64`
  (LMUL=8) burst into a LIM scratch, then reshapes from LIM. Diagnostic (RES `+0xA0/+0xB0`) confirms
  ILP is ACTIVE — reader 0 avg **230 words/seg** (115-marker bursts) — yet throughput stayed ~250k/s.
- `fence_()` after the `bytes_sent` write: no effect.

**The decisive measurement — the drain path is EXONERATED (~98× headroom).** Reran the isolated
D2H-socket push bench on bh-11: `test_x280_profcons --socktest --npages 200000` **WITHOUT**
`TT_METAL_DEVICE_PROFILER` (else the RT manager auto-boots profzone and steals the X280) →
**1566 MB/s = 24.5M pages/s, lossless, done=1** (batch=16 push + notify). vs production 16 MB/s. So the
socket + PCIe + host-drain path can do 1566 MB/s; reads are also proven fast in isolation (`gridilp`
~1533 MB/s @ 2 harts ILP-4). The 250k/s bottleneck is ENTIRELY in the coupled READER path.

**ROOT CAUSE (rdcycle + count instrumentation of the reader) = blind empty-ring POLLING, NOT reshape.**
Corrected drain rate: reader wall = **1.42 s** for ~640k markers ⇒ **~450k markers/s** (the earlier
"250k/s" used a wrong 2.5 s wall assumption). Clean count-based run (no per-iteration rdcycle):

| reader | wall | markers | core-polls | ns/poll | productive segs | %productive |
|---|---|---|---|---|---|---|
| 0 | 1422 ms | 400 330 | 1 278 695 | 1112 | 3 413 | 0.27% |
| 1 | 1421 ms | 241 114 | 1 322 970 | 1074 | 16 246 | 1.2% |

Where the reader's 1.42 s goes (per reader, count-based —
per-iteration rdcycle does NOT trap/contaminate here, wall unchanged with/without it):
- **~2.6M core-polls (both readers) at ~1100 ns each = essentially the ENTIRE wall.** A "core-poll" =
  bulk-read `ctrl[0..15]` (head+tail for all 5 RISCs) in one `vle` + `fence` + extract 10 values.
- **Only ~20k of the 2.6M polls were productive (0.3%). 99.7% of polls find an empty ring.** The reader
  free-runs (~23k passes × 55 cores), and markers trickle in over the ~1.4 s, so almost every poll is wasted.
- reshape = 148 ms, bulk-read = 3 ms — both NEGLIGIBLE. **The reshape hypothesis was WRONG.** LIM scalar
  ops measured ~52 ns each (reshape: 9 LIM ops/marker × 315k ≈ 148 ms).
- Per-poll ~1100 ns ≈ ~300 ns (bulk-read + `fence`) + ~700 ns (reading the ctrl vector back out of LIM
  scratch, 10 reads × ~68 ns) + loop glue.

**CORRECTION — the reader is NOT the bottleneck; poll rate is irrelevant.** Tested a 10× poll reduction
(idle-backoff: after a fully-empty pass, spin `POLL_BACKOFF_CYC`=500 µs before re-polling; productive
passes skip it). Core-polls dropped as designed: 1.28M → 123k, passes 23k → 2.2k. **But wall (1416 ms),
drain rate, and perturbation (162 µs avg / 14 ms max / 22×) were ALL identical.** So the 1.42 s reader
"wall" is really the elongated WORKLOAD DURATION (the reader runs until teardown) — NOT time spent
polling. Reducing polls just converts poll-time into backoff-time. The "polling is the wall" reading
above conflated wall with busy-time; it's wrong. Reader poll/visit rate does not gate the compute.

**RELAY TIME-SPLIT (2026-07-14) refutes the host-FIFO hypothesis — the WHOLE drain pipeline is ~90% IDLE.**
Instrumented the relay (rdcycle brackets, X280 ~1 GHz): of a 1429 ms wall, **empty-spin = 1287 ms (90%)**,
reserve-stall (host FIFO) = **108 ms (7.5%)**, copy = **33 ms (2%)**. The reserve-stall *count* is ~1M but
the *time* is only 108 ms (~100 ns/spin), so the host FIFO is NOT the bottleneck. The relay is STARVED —
spinning on empty LIM staging 90% of the time. The readers are idle too (reshape ~150 ms + reads 3 ms of
their 1429 ms wall). So reader → relay → host all keep up with huge headroom; markers just arrive slowly.

**⇒ The throttle is UPSTREAM of the drain — the PRODUCER (compute) marker-emission path, not the pipeline.**
The compute emits ~630k markers over ~1.43 s (~440k/s) because the compute itself is slowed when emitting,
not because the drain can't keep up. Consistent with EVERY drain-side lever having zero effect (poll rate
10×, bytes_sent posted/non-posted, 1 MB FIFO, ILP reads). Single-core is clean, so it's a many-core
producer effect. **NEXT: instrument the PRODUCER** — `kernel_profiler.hpp` SPSC emit path
(`ring_ensure_room` block-on-full + `publish_tail` fence per marker) + the `ring_full_wait_count` the
manager already logs at shutdown ("L1 ring hit capacity N times"). If rings rarely fill, the ~22× is
per-marker emission overhead (fence/L1), not ring-full blocking. Then decide fix vs go-lossy.

Files (uncommitted on bh-11, on top of EXPERIMENT `214632ee`): `src/profzone.c` — ILP reader (bulk `vle64`
ring-segment read into LIM scratch `SCRATCH_BASE 0x08012000`, reshape from LIM) + bulk `ctrl[0..15]` poll
+ non-posted `bytes_sent` (2nd `posted=0` window) + count diag at RES `+0x60`=bulk_words, `+0x70`=segs,
`+0x80`=reader wall, `+0x90`=passes, `+0xA0`=core-polls (the earlier per-iteration rdcycle-split brackets
were removed — count-based is uncontaminated). `realtime_profiler_manager.cpp` — host log of the reader
wall / drain rate / poll count / avg seg. See memory `x280-rt-profiler-backpressure`.

**ROOT CAUSE — the reader's per-marker RESHAPE (2026-07-14, supersedes the "wall / producer-emit / blind-poll"
readings above).** The reader "wall" is boot-to-teardown and does NOT represent device-side kernel
elongation (it even rose to 2147 ms on a noisier run while back-pressure fell). The true signal is the
producer's self-emitted `PROFILER_STALL_ZONE` (id `0x7FFF`). Built a HOST raw-marker decoder
(`TT_METAL_X280_DROP=1`: FW bulk-flushes raw 2-word markers, host decodes + tallies stall zones, no core
info needed), validated with a guaranteed per-RISC stall injected in `risc_finished_profiling` (decoder saw
it; since removed). Measured REAL stalls:

| reader mode | stall count | stall dur avg | stall max |
|---|---|---|---|
| reshape (per-marker, 4×w64) | ~1248 | ~2.5 ms | ~12.6 ms |
| **raw bulk-flush (no reshape)** | **546** | **12 µs** | **121 µs** |

⇒ dropping the on-device reshape collapses producer stall **duration ~200×** (and count ~2.3×). A full ring
drains in ~2.5 ms with per-marker reshape vs ~12 µs with a bulk vector copy, so producers block that much
less. The earlier 7→4-store vectorize did nothing because it kept the per-marker *loop*; only the full
per-segment bulk copy collapses it. **FIX = relay raw markers + convey (core,risc) once per ring as a sticky
header, reshape on the (idle) host** — proven to cut perturbation ~200× and it's the DRAM-profiler-style
format (`ID_LL..ID_HH` headers) rather than self-describing-per-page. Raw decoder + `g_x280_raw` live in
`realtime_profiler_manager.cpp`; the non-functional raw-flush is in `src/profzone.c`; functional reshape
version committed at `dfe12d1d`.

---

### 21. Yusuf-branch integration — crash fixed, pipeline reconciled, BW measured (bh-11, 2026-07-15)

Wired the raw-flush + host-translate path into the `x280-on-yusuf` branch (Option A: the receiver
translates raw markers → `WorkerZoneWire` into a `BroadcastRing`; a decoupled `run_consumer` enriches
[noc0 coord + zone name] and pushes to Tracy). Three things resolved. Committed at `a7ea9c67663`.

**(a) The persistent crash was a use-after-free, not corruption or an empty map.** Every run of the
raw-flush path cored ~6 s in (0–389 zones) with `Fatal Python error: Segmentation fault` OR `Floating
point exception` — the SIGFPE/SIGSEGV alternation is the tell. gdb pinned it to `run_consumer`
(frame #1 = libstdc++ thread trampoline ⇒ fault in its own inlined `push_batch`). Root cause:
`boot_x280_drainer(DeviceState& dev_state)` did `x280_dev_ = &dev_state`, but `dev_state` is the init
loop's **local**, immediately `devices_.push_back(std::move(dev_state))`'d (and `devices_` is a
`std::vector` that reallocates per push). So `x280_dev_` dangled at a moved-from, destroyed stack
local; the consumer thread deref'd its moved-from `unordered_map`, whose garbage `bucket_count` gave
`hash % 0` (SIGFPE) or a wild pointer (SIGSEGV). This is why ring size (131072/1M/4M) was irrelevant,
the 6-bit-coord guesses failed, and an `!map.empty()` guard did nothing (`is EMPTY` fired 0× — the map
isn't empty, the *pointer* is dead). **FIX**: don't take the address of the pre-move local; set
`x280_dev_` **after** the init loop from the stable `devices_` element (scan for `x280_active`).
Restored the ring to `kMaxRingCapacity` (~128 MB) for the lossless `DrainThenStop` backlog.
**VALIDATED**: 0 crashes, `test_with_ops` passes, X280 boots (110 cores/111 ctx), `ring-dropped 0`.

**(b) The "reader read 804k, relay relayed 108k" ⇒ 7× loss was a PHANTOM — a mislabeled counter.**
The reader **does** advance the SPSC read pointer every core/segment (`profzone.c:232`
`w32(CTRL_HEAD(r), h)`), and the staging hop is lossless-**blocking** (`profzone.c:213`, reader spins
until the relay frees room), so real loss there is impossible without the reader stalling. The relay's
`total` counter (`profzone.c:354`, `RES 0x00`) increments **once per 64 B page_copy — it counts PAGES,
not markers** — but the manager logged it as "markers". 108,499 pages × 8 = ~868k marker-slots ≈ the
readers' ~804k real markers + ~8% partial-page padding (reader 1's 13k tiny 45-word segments each round
up to a full page). **Consistent, lossless end-to-end.** Fixed the mislabel in
`realtime_profiler_manager.cpp` (now logs pages + bytes + marker-slots).

**(c) BW on `test_with_ops`: ~6.9 MB (108,499 × 64 B) over ~1.3 s ≈ ~5 MB/s — but the pipeline is 99%
idle** (relay empty-spin 1280/1298 ms, **0** host-FIFO reserve-stalls). This workload does NOT exercise
the pipe: it's a 64-core matmul trace (100 capture + 5 replay, 0.12 s of `call`), supply-starved, ~300×
below the isolated capacity ceilings (§13 ~3.0 GB/s D2H, §11–12 ~1.5–1.8 GB/s reads). A real
post-reshape-removal BW number needs a sustained kernel that holds the L1 rings near-full for seconds.

**(d) Reservation churn was self-inflicted (bh-11 kept "losing its ssh key").** My detached scripts
`rm -f /tmp/tt_x280_autoprime_dev*` every run, forcing the boot path's `WarmReset::warm_reset_chip_id`
— a chip warm-reset re-enumerates the Blackhole PCIe device out from under the container, so IRD's
health-check **recreates the container** (new reservation id, wiped `/tmp`, key drops to a password
prompt). Churned 128575→617→637→662. **FIX**: never delete the autoprime marker, never set
`TT_METAL_X280_FORCE_PRIME`; let the chip prime at most once and stay primed. Added `ControlMaster`
multiplexing for `bh-11` so bursts of `tt run` don't hammer sshd. The clean run did not churn.

**Still open**: ~25+ orphan ZONE_ENDs across 18 lanes (guardrail log-capped at 25) — NOT ring loss
(`ring-dropped 0`); it's the STICKY_META forward-fill attribution splitting START/END across lanes.
Known-buggy by design: X280 serializes across two regions/cores so the sticky packet isn't
representative of the full stream, and a Tensix producer can fill its L1 ring entirely inside the
kernel and never return to `brisc.cc` to emit a sticky-meta packet at all. Deferred.

---

### 22. Intermittent trace-replay HANG — root cause: X280 multi-hart boot is flaky (bh-11, 2026-07-16)

`test_trace_runs.py::test_with_ops` under the X280 RT profiler hangs ~30% of runs (X280 off = 100%
stable). Found by on-silicon elimination + a host-side liveness probe (reads X280 LIM every 250 ms:
relay total/loops, single S_PROD/S_CONS, mirror ΣMTAIL/ΣMHEAD, live reader passes, and a **per-hart
boot stage @ RES(0x100+h*8)**).

**Not** a pipeline/lossless bug. Ruled out: PCIe D2H (relay drain-to-null → same rate), host
receiver/consumer threads (disabled → same), producer-block-per-se (drop-not-block only MASKS it —
undrained workers drop instead of blocking, so the trace completes). The probe caught hangs frozen
from t=0 with e.g. `stage(h0=3 h1=0 h2=0 h3=3)` — **harts 1 (reader-1) and 2 (collect) never entered
`main()`**. `release_reset()` does not reliably start all 4 harts; the dead set varies run-to-run.

**Mechanism:** a dead drainer hart never drains its worker cores → their Tensix L1 rings fill →
workers block in `ring_ensure_room` → the trace never completes → host hangs in
`FDMeshCommandQueue::finish_nolock` (device never posts the trace-completion event). **Compounding
gap:** `boot_x280_drainer` only verified **hart 0** (`RES(0x30)==0xB007`), so it ran a crippled drainer
undetected.

**Fix (this commit):** (1) permanent per-hart boot heartbeat — every hart writes stage 1/2/3 to
`RES(0x100+h*8)`; (2) host waits for **all** `nharts` to reach stage 3, and if any is missing
re-releases reset (≤3 retries), refusing to run a crippled drainer.

**Validation caveat:** 8/8 clean passes with `allup=1` (0 hangs vs ~30% baseline), but every completed
run had `boot_retries=0` — the *retry path was not exercised* (no flaky boot happened to occur in the
runs that finished), so recovery-of-a-flaky-boot is not yet directly demonstrated. **Blocker:** bh-11
**reboots the host under the X280 workload itself** — reproduced twice with ZERO `tt-smi -r` from us
(chip stayed primed across host reboots, so no auto-prime either), rebooting mid-run at run 3 and run
7. That is a separate bh-11 hardware/infra instability under X280 load (not our resets); finishing
validation needs a BH box that survives the workload. Repeated `tt-smi -r` earlier ADDED churn (and
the "loses its key / container churn" symptom) but is not the reboot root. See memory
`x280_trace_hang_multihart_boot`.

---

### 23. Pipeline bottleneck under load — the readers are the wall, not downstream (bh-11, 2026-07-16)

`test_with_ops` captures a 100-matmul trace on the 8×8 grid and replays it 5×. Each matmul fires
zone markers across 64 cores × 5 RISCs, so markers are generated in a burst far faster than an op
takes. The buffer chain (L1 rings → LIM mirrors → single SPSC → D2H FIFO) absorbs ~95 ops of slack,
then saturates and back-pressures the Tensix producers — the "X280 stall". This is a **sustained
throughput** problem: the drainer's real drain rate R is below the burst marker rate. Isolated
per-stage BW numbers (§11–16) all showed headroom; they don't set R. This is which stage does.

Per-stage cycle telemetry from one run (1354 ms wall):

| Stage | Busy (real work) | Waiting | Verdict |
|---|---|---|---|
| **Relay** (single SPSC → D2H → host) | copy **17 ms (1.3%)** | empty-spin **1279 ms (94.5%)**; D2H reserve 57 ms; **0 reserve-stall spins** | starved — huge headroom |
| **Collect** (mirrors → reshape → single) | copy **90 ms (6.6%)** | empty-spin **1264 ms (93%)** (mirrors idle) | starved — huge headroom |
| **Readers** (Tensix L1 → LIM mirror, NoC0) | **~100% of wall in the poll sweep** | ~0 | **the wall** |

**The bottleneck is the reader stage (L1→mirror drain).** Everything downstream — collect, relay,
the PCIe D2H export, the host consumer — is 90%+ idle and the host D2H FIFO stalled **0** times, so
they all have large headroom (consistent with the isolated measurements). The readers spend their
whole wall sweeping cores, and **~96% of polls find no data** (reader 0: 4857 productive of 126,610
polls; reader 1: 3595 of 130,405). Reader throughput was **371 k/s (r0) vs 222 k/s (r1)** — the
core→reader split is also unbalanced.

**Why isolated tests misled:** the NoC-read benches (§11–12) hit 1.5–1.8 GB/s using **ILP (many reads
in flight)**. The real reader poll issues **one ctrl read per core and waits** (non-posted round-trip
+ full `fence`), so it is **latency-bound, not bandwidth-bound** — BW headroom is irrelevant when you
pay per-core round-trip latency × 55 cores/sweep.

**OPEN (needs burst-window data):** cumulative telemetry folds in idle gaps + the 500 µs
`POLL_BACKOFF` after empty passes, so the headline "~10.5 µs/poll" can't separate NoC poll latency
from backoff-domination (readers sleeping through part of a burst). Different fixes (ILP the polls vs
retune/kill the backoff vs more reader harts). Next: split the reader wall into ctrl-poll / bulk-read
/ backoff cycle counters to localize the reader sub-cause definitively.

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

## §21 — Linearized stream (sticky-src) + single-hart direct drain (lossless RT profiler)

`src/profstream.c` + `test_x280_stream`. The per-lane acked-ring D2H design died on a
multi-lane host-read/ack coherence wall (see memory `x280_lossless_ring_d2h_race`). This
reverts to the proven single-stream shape and solves lane identity with a **STICKY-SRC**
header: the drainer emits an 8 B `PP_STICKY_SRC` packet (lane id, precomputed LUT) at each
source switch; the host demuxes one stream — a STICKY-SRC sets the current (core,risc), every
marker/meta after it binds to that lane until the next.

**Split (2 readers + 1 relay).** Readers drain worker-L1 → per-reader LIM SPSC; a relay
round-robins them into one host ring. Vectorized copies (`vsetvli e32,m8` word-copy) took the
reader NoC read 249 → 7.9 cyc/word and the relay 52 → ~2 cyc/word. **But it corrupts on
host-ring wrap under saturation** (~tens-of-k seq gaps). Root-caused by elimination: host is
x86_64 (PCIe DMA is coherent — host read not stale), ordered notify didn't fix it (not a
landing race), relay OVERWRITE detector = 0 (flow control sound). Remaining cause = the relay
reads **stale data from the cross-hart LIM SPSC** (reader writes, relay reads; L2CPU L1 holds a
prior generation of a wrapped slot) — the same cross-hart LIM incoherence that forced local
PROD/CONS pointers and killed per-lane HACKED. `cbo` cache maintenance HANGS the X280 and LIM
has no easy uncached alias, so the split can only be kept lossless by never wrapping the ring.

**⚠️ CORRECTION (2026-07-19): the split "loss" above was MISDIAGNOSED — it was NOT cache
coherence, it was a relay FRAMING bug, now FIXED (commit 881e41be20c). Tell: the corrupted runs
had the EXACT marker count (1,100,000 == expected) with ~27 % seq gaps and half the lanes short
— markers MISATTRIBUTED to the wrong lane, not lost/overwritten (coherence staleness would change
counts). Cause: the relay capped each copy at host-ring free space (`run = min(avail, hspace)`),
cutting a reader's data-run mid-frame, then round-robined to the OTHER reader; when the host
freed space, round-robin inserted reader0's next frame BETWEEN reader1's split halves, orphaning
the continuation from its STICKY-SRC → host bound it to the wrong lane. `profcons_split` (§15)
never suffered because it copies only WHOLE rings to descriptor-addressed host slots. FIX: the
relay drains a reader's WHOLE published snapshot (`avail = prod - cn`; PROD only published at
frame boundaries) CONTIGUOUSLY before touching the other reader, publishing HSENT incrementally.
RESULT: the exact 261/550 config → 550/550 lossless, 0 gaps; 4000 mk (2.2M markers) also clean.
Readers 4.3 cyc/word (~1.86 GB/s). So the split is a valid, faster-on-the-read-side design; the
"cross-hart LIM incoherence" story here (and the PROD/CONS local-pointer workarounds it justified)
is suspect. PER-RING HOST DRAIN (done): relay writes reader h → its OWN host ring h (own HSENT/
HACKED, own posted window), N per-ring host consumer threads drain in parallel; relay uses a
NON-blocking round-robin (skip to the other reader when a ring is full — safe now, each reader
owns a ring so partial copies stay contiguous). Result: relay `hostfull` 270k → ~13k (20× less
spinning), wall 16M → 13M, ~666 MB/s end-to-end (beats direct 615 and the single-ring split 539),
lossless at full burst + 2.2M markers. Now READER-bound, not host-bound (readers 4.2 cyc/word,
~1.86 GB/s aggregate); producer stall ~1.5× below direct (640M vs 951M spins). The whole-snapshot
blocking drain (needed for a shared ring) was a throughput regression here and was reverted.**

**TWO-THROTTLE DECOMPOSITION (`--nodrain` diagnostic, 2026-07-19).** "Why doesn't copy dominate at
peak?" Isolated by removing the host sink entirely (relay ignores HACKED + no host consumer thread;
NONCE bit 12; LOSSY on purpose). Reader at full burst, with vs without the sink: `spsc-wait` 4M →
**0**, copy% 34 → **51 %**, wall 13M → 9M, producer spins 640M → 407M. So there are TWO independent
throttles: (1) the HOST SINK — real, ~31 % of reader time, back-pressures via SPSC-full; removing it
zeroes spsc-wait. (2) WASTED POLLING — ~50 % of reader
time even with the sink gone, INDEPENDENT of it. INSTRUMENTED (reader RES +0x28 visits, +0x30 polls):
avg-run = **500 words** (buffers ARE full, ≈ RING_CAP 512 — drains are efficient at ~4.2 cyc/word), but
**83–90 % of the tail-polls hit EMPTY lanes**. The reader cycles its 275 lanes faster than the
producers refill, so 83-90 % of tail-reads find `tail==head`. **BUT the poll is NOT the bottleneck**
(corrected): a per-core ILP tail-gather (1 vle of the 5 contiguous tails/core, 4 cores' NoC reads in flight)
was implemented and REVERTED — it left BW unchanged with the sink and marginally worse without it, and the
`polls` counter TRIPLED (12k→40k) at the SAME wall. So the empty polls overlap with the copy/visit latency
(the X280 LSU already pipelines the 5 scalar `r32` reads); making them cheaper just lets the reader spin
more, not finish sooner. The real non-copy ~50 % is PER-VISIT serialization — the `fence` (drains ~500
posted LIM writes before PROD publishes) + head-advance NoC write + PROD/sticky, ~2000 cyc × ~2k visits.
AND at the real operating point (with the host sink) the SINK dominates anyway (spsc-wait), so reader
overhead is second-order until fan-out. Lever, if pursued: batch the PROD publish/fence across several
lanes (X280-internal LIM→relay, no host-heartbeat coupling — unlike the host-publish batching that failed).

**CANONICAL "ALL BUFFERS FULL" TEST (`--fullread`, 2026-07-19) — the workload we optimize against.**
Current `--proddelay 0` "peak" is NOT all-buffers-full: it's a live producer/reader race that settles at
LOW occupancy (83-97 % of polls find EMPTY lanes — reader out-cycles producers; occupancy is unstable,
run-length dependent). The toughest workload is all 550 rings full simultaneously. Mock it on the READER
side: `--fullread` keeps the per-core tail poll (cost) but IGNORES it and always drains a FULL RING_CAP-2
buffer per (core,risc), advancing head to the real tail so producers stay consistent (over-reads stale past
tail, harmless; LOSSY). Deterministic max-drain load, `0 % empty`, `avg-run=510`. RESULTS (bh-11, 550 lanes):
| all buffers full | reader copy% | reader spsc-wait | relay copy% | relay hostfull | wall |
|---|---|---|---|---|---|
| with host sink | 34-49 % | 6-9M | 53 % | ~38k | 16-18M |
| sink removed (`--nodrain`) | **75 %** | **0** | **94 %** | **0** | **9M** |
Three findings: (1) **the host sink is THE bottleneck** — with it the reader is ~50 % spsc-wait and the relay
~47 % hostfull-spinning; removing it → reader 75 %, relay 94 %, wall HALVES (fan-out is the fix). (2) **work
is imbalanced across readers under the sink** (one drains ~2× the other, flips run-to-run — the single relay
+ per-ring host threads service the two rings unevenly); sink-free they're perfectly balanced. (3) even
sink-free the reader caps ~75 % copy — the remaining ~25 % is the per-visit fence/publish, the reader's true
internal ceiling at full load. So priority: fan-out the sink first, then batch the reader fence/publish. To get copy-dominant at peak needs BOTH a
faster sink (fan-out to more rings/threads than readers) AND the reader-overhead fix (bulk-poll the
tails + batch the PROD publish/fence). Bigger SPSC does NOT help — tried 4096→16384, it made things
2.5× WORSE (burstier back-pressure, less 3-stage overlap; the small buffer is near-optimal).

**PEAK-PATH BUILT: bulkcore + dualrelay + adaptive switch (2026-07-19). BW now (pll=1000, 1 cyc=1 ns):**
- **`--bulkcore` (1 bulk NoC read/core, 5 contiguous rings = 2560 words)**: reader 4.5→3.6 cyc/word = rdrbench
  read rate. Per-core tail round-robin gone. Relay becomes the bottleneck (86 %).
- **`--dualrelay` (1 relay hart PER reader) + runtime SPSC size (bulk=16384, normal=4096, mode-gated)**: both
  needed together — the 2nd relay alone left readers spsc-wait; the bigger SPSC decouples the 2560-word
  handshake. `--bulkcore --dualrelay --nodrain`: readers 88 % copy, spsc-wait 0 → **~1.93 GB/s aggregate
  reader→relay** (lossy bench).
- **`--adaptive` (per-core switch: fullness ≥ 4·RING_CAP → bulk, else per-risc)**: ONE dynamic switch, not
  modes. Full burst → bulk fires (1704 cores); slow producer → per-risc (0 bulk). Since the reader out-cycles
  producers, bulk only fires under real BACKLOG — exactly when amortization pays.
- **LOSSLESS bulk framing**: bulk case frames 5 per-risc STICKY-SRC + valid `[head,tail)` only (host demuxes
  unchanged), but BATCHES the fixed cost — **ONE fence + ONE PROD publish per core vs 5 in per-risc mode**
  (the `fence iorw,iorw` stalls the hart until SPSC writes land before PROD is visible; 1 barrier/core not 5
  is the batching win). `--adaptive`/`--bulkcore` both 550/550 LOSSLESS (1-relay + dual-relay).

**BW SUMMARY (lossless `--adaptive --dualrelay`, 550 lanes, 8000 mk):**
| | per reader | aggregate | note |
|---|---|---|---|
| reader→relay (`--nodrain`, sink ignored) | 609 MB/s | **~1.22 GB/s** | lossless; readers 83 % copy |
| end-to-end (real host consumer) | 108 MB/s | **~215 MB/s** | HOST-SINK BOUND (relay 94 % on hostfull=778k) |

Lossless framing costs ~0.6-0.7 GB/s vs the lossy bulk (1.93→1.22): NOT the sticky bytes (10 words/core,
~0.4 %) but the FRAGMENTATION — 5 per-risc framed reads (~444 words each) lose the single-2560-read NoC
amortization. It doesn't matter yet: reader→relay (1.22) already has ~6× headroom over the host sink (215).
**The host consumer is now THE wall** — it reads HSENT from LIM per poll (~18 µs each, 1085 polls) and bursty
bulk floods it (end-to-end 215 < the steady per-risc split's 666). NEXT = publish HSENT to host sysmem
(coherent ~ns read) / 2 D2H sockets. To claw back the 0.6 GB/s after that: one core-sticky carrying the 5
(head,tail) offsets + host-side split — but that ships stale words to the host, so only worth it once the
sink keeps up. Commits: f29bc7c2279 bulkcore, 0b857b93f82 dualrelay+runtime-SPSC, 40234d9ad48 adaptive,
8acee90bf63 lossless framing.

**HOST-SINK FIXED (Gap 1) — SENT in host sysmem + spin-poll + bigger ring (2026-07-19).** Two changes took
end-to-end from 215 MB/s to the reader ceiling:
1. **SENT pointer in host sysmem** (each ring gets a 64 B trailer; relay/drainer publishes SENT through the
   SAME posted PCIe window as the data, ordered after it). Host polls its own RAM (`read_sysmem` ~µs) instead
   of reading the pointer from device LIM (`drv.read_block` ~18 µs/poll — the wall). HACKED stays in LIM
   (host→device write is posted/fast). Also replaced the consumer's 200 µs sleep-on-empty with a SPIN +
   time-based exit (the sleep let the relay spin on hostfull while the host slept). → 162M→34M wall, ~1.0 GB/s.
2. **Default host ring 32 KB → 256 KB.** After (1), the residual ~15 % drag (reader spsc-wait, relay
   hostfull~25k) was the SMALL RING: the relay stalled on hostfull waiting for HACKED acks (ack-latency
   ping-pong), NOT host-consumer throughput — the relay's raw push (4.5 cyc/word ×2 ≈ 1.78 GB/s) already
   exceeds the readers. A 256 KB ring absorbs bursts + decouples the relay from ack latency (fits 4 rings in
   the 2 MB window). Commits 4b9b55e392c, d73f4ef3e21.

**BW NOW (lossless `--adaptive --dualrelay`, 256 KB ring, 8000 mk): ~1.26 GB/s aggregate end-to-end == the
reader→relay ceiling.** reader 83 % copy, spsc-wait 0; relay ~72 % busy, hostfull 0; host consumer ~30 % busy
on real work. **The push-to-host now OUTPACES the readers with headroom — the READER is the bottleneck** now
(limited by its own per-risc framing/poll overhead, not the host). So further speedups are reader-side:
reduce framing overhead, or reclaim the 0.6 GB/s via core-sticky + host-side split (now viable — the host can
absorb the extra raw bytes). Still TODO (Gap 2, for CPU efficiency + real-time, not BW): direct hugepage
pointer (kills the read_sysmem ~1 µs/call spin) + flusher→MPMC→consumer restructure.

**RAW-BULK reclaim (`PP_BULK_CORE`, 2026-07-19).** Now that the host has headroom, ship raw over-read from
the device to get back the per-risc-framing cost. Reader bulk path = ONE streaming 2560-word NoC read into a
`PP_BULK_CORE` frame [hdr 2w: core,rawn][NRISC meta: head_mod|run][pad][2560 RAW]; host demux dispatches on
packet type and splits per-risc, taking only the valid `[head_mod,head_mod+run)` circular slice of each ring
via a range `insert` (ignores over-read). Adaptive's MIXED stream (bulk + per-risc) demuxes correctly.
RESULT: reader 5.3→4.4 cyc/word, reader→relay ~1.22→~1.5 GB/s, end-to-end ~1.26→~1.47 GB/s; BONUS the
offline demux went 1.34→2.91 GB/s (range-insert vs per-word push). Lossless (all modes). Commit 31c15e25aa4.

**REAL HOST PIPELINE (Gap 2, `--mpmc M`, 2026-07-19, commit 5e0e89cc78b).** Built in the required order:
per-socket flush+demux → device records BEFORE the MPMC (sticky-src is in-order per stream, so demux MUST be
on the flusher; records are self-contained → consumers stateless). 2 flush+demux threads (drain in-order,
dispatch bulk/per-risc, decode + per-lane seq-verify → `Rec{lane,seq,ts,prog}`, batch-push to a BOUNDED MPMC)
→ M consumers pop + emit (`--cwork N` simulates Tracy-emit cost). Old offline path kept (`--mpmc 0`). Two
fixes it forced: (1) relay must publish SENT on WHOLE frames (copy the whole frame-aligned snapshot or
nothing, never `min(avail,hspace)`) — the old partial copy published non-frame-aligned SENT and the streaming
flusher misparsed a split bulk frame → 1.9M gaps under back-pressure (offline masked it by capture-then-parse);
(2) authoritative shutdown via `wait_active_fw_returned`→`device_done`→ drain to `hsent==acked`, not a 500 ms
quiescence timer (quit early under back-pressure → 73k short). RESULT (adaptive+dualrelay 8000mk, all LOSSLESS
4.4M/4.4M, 0 gaps, 0 overflow): cwork 0 ~1.47 GB/s (reader-bound); cwork 300 (heavy) M=1 253M → M=4 74M
(3.4×) → M=8 60M (4.2×). **The pipeline BACK-PRESSURES LOSSLESSLY (bounded MPMC) and scales with consumer
count** — heavy higher stages stall the whole chain without dropping, and M is the lever. Remaining: real
Tracy-context emit in the consumer (replace the cwork sink); direct hugepage pointer for the SENT poll.

**Direct drain (`--direct`, NONCE bit 8).** ONE hart reads worker-L1 over NoC (coherent) and
writes the single host ring DIRECTLY, injecting sticky-src inline — no LIM SPSC, no cross-hart
handoff, no relay. UNCONDITIONALLY LOSSLESS: 550/550 lanes, 0 seq gaps at every rate (proddelay
0..6000), full burst, high volume. Correct + simplest, but NOT the only correct design (the split
is lossless too, post-fix) and slower on the read side (single hart read+write = 615 MB/s).

**BW + stall knee (bh-11, 550 lanes, `pll=1000` → 1 cyc = 1 ns).** The drainer is a steady
**6.5 cyc/word ≈ 615 MB/s raw** (read worker-L1 + write host ring, per 4 B word); end-to-end at
full burst is **~410 MB/s** (drain hart only ~65 % copy-busy, rest is host-ring flow-control
wait). Producers back-pressure whenever the 550 lanes collectively outrun the one hart:

| proddelay (iters/marker) | producers stalled | total spins |
|---|---|---|
| 0 (full burst) | 550/550 | 957 M |
| 500 | 550/550 | 576 M |
| 1000 | 550/550 | 180 M |
| 1200 | 440/550 (tail) | 32 M |
| **1350** | **0/550** | **0** |
| 2000 | 0/550 | 0 |

**Per-word cost decomposition (clean single-path builds — a runtime `if` in the hot loop regresses the
untaken path ~2×, so each mode was measured as its own unconditional build).** Why the drainer sits at
615 MB/s vs the ~2.24 GB/s rdrbench read-only ceiling (2 harts) / ~1.15 GB/s per hart (3.5 cyc/word):

| stage | cyc/word | GB/s | delta |
|---|---|---|---|
| rdrbench pure NoC read (no relay) | 3.5 | 1.15 | baseline read |
| stream READ-ONLY (`vle` + head-advance + fence + poll, discard store) | 4.5 | 0.89 | **+1.0** relay bookkeeping |
| stream READ+WRITE (production drainer) | 6.5 | 0.615 | **+2.0** posted PCIe store |

So the drainer is NOT leaving read bandwidth on the table — it pays the pure-read 3.5, **+1.0** for the
per-visit relay bookkeeping (the head-advance is itself a NoC write back to worker L1, plus a `fence` and the
tail-poll loop), and **+2.0** for the PCIe posted write of the data. The write is ~31 % (not half — a posted
write is cheaper per word than a NoC read round-trip); bookkeeping ~15 %; read ~54 %. 2 harts don't double
because only the read half is parallel — the +2.0 write half funnels through the one shared PCIe endpoint.
Levers: fewer bytes/marker cuts all three. Two lever attempts, both MEASURED and REJECTED:
- **Batched publish** (fence + HSENT notify only every ~½ ring, not per visit): DID cut copy-cost 6.5→6.1
  cyc/word (the +0.4 fence saving is real) but EXPLODED the wall 21M→92M (host-wait 3M→74M) — the HSENT
  notify is the pipeline heartbeat, so publishing in bursts lets the host's 200µs idle-sleep kick in, the
  ring backs up, and the drainer stalls on flow control. Net ~4× worse. The per-visit publish's real job is
  keeping the consumer fed; can't batch it without starving the host. Reverted.
- **Write over NoC1** (`--wnoc1`, posted PCIe store on NoC1 instead of NoC0): NO-OP — 6.5 cyc/word single
  hart, 9.3 two harts, bit-identical to NoC0. The write cost is L2CPU store-issue + the PCIe endpoint, not
  the NoC path; and 2 harts still funnel to the same PCIe tile regardless of NoC. Flag kept (setup-only).

**2-hart scaling levers (all measured, all closed):** `--ndrain 2` (independent slice/ring/HSENT-HACKED/heads
per hart, zero shared LIM) is 550/550 lossless but SUBLINEAR — per-hart 6.5→~9 cyc/word, eff BW ~411→~539
MB/s (+31%), knee ~1300→~900. Sublinear because both harts contend on the shared L2CPU load/store path + the
single PCIe write endpoint. `--rrconsumer` off (one host thread per ring) drives host-wait→0 (best ~680 MB/s)
but is a minor factor — the ceiling is on-device. `--splitnoc` (hart h reads over NoC h&1) is a **NO-OP**
(~identical wall), confirming it's not NoC bandwidth. **Double-buffered copy (software-pipeline the NoC read
of chunk N+1 ahead of the PCIe store of chunk N, ping-pong vreg groups v0/v8) = NO-OP:** clean single-path
build is 6.5 cyc/word, bit-identical to the plain copy. The X280 vector LSU already overlaps the store with
following loads (posted PCIe writes retire immediately — nothing to hide); reverted. So the read-side `vle32.v
m8` streaming is the whole ILP story; the only remaining lever is fewer bytes/marker (matches rdrbench).

**Knee ≈ 1300 iters/marker.** Below it → workers stall; at/above → zero stall. Critically,
**stalling ≠ loss**: every run incl. full burst was 0 seq gaps — back-pressure only perturbs
workload *timing*, never drops markers. At ~615 MB/s / 8 B this is ~77 M markers/s aggregate
(~140k/s/lane) before the knee; real kernels emit at zone boundaries, orders of magnitude
below, so the operating point sits well under the stall threshold. More headroom later = a 2nd
independent drain hart on its own host ring (no shared LIM), ~1.9× per rdrbench (§ RDRBENCH).

## §22 — Real kernel_profiler capture (2-word markers + split stickies) + full-grid BW

Moved the X280 profiler onto the REAL `kernel_profiler.hpp` path (2-word markers; identity/context via
STICKY_PROG/TIMER/SRC — SRC & TIMER now 1 word, PROG 2). Drainer = `profzone` (copy of `profstream`);
new host test = `test_x280_realprof` (real `DeviceZoneScopedN` dm+compute kernels). Run with
`TT_METAL_DEVICE_PROFILER=1 TT_METAL_NO_RT_PROFILER=1`. Dual-relay (2 D2H sockets) is the default.

**Marker spacing (the key finding).** With `WORK_SIZE=0` (empty zone bodies), consecutive markers on a
lane settle at **~48–50 ns apart** (~49 cycles @ 1 GHz) — the raw cost of one `mark_time`: ring
room-check + tear-free 64-bit wall-clock read (retry loop) + 2 ring writes + fenced `publish_tail`. A
back-to-back empty zone (START+END) ≈ **~98 ns**. First 1–2 markers are wider (warmup + leading nested
FW/child-zone STARTs). So the profiler's own overhead sets an effective resolution floor of ~50 ns/marker
(~100 ns/empty-zone) — the observer effect on a real kernel. Per-lane instantaneous rate ≈ 1 marker/49 ns
≈ 160 MB/s/lane of 8-B markers, but bursts don't overlap enough across lanes to saturate the drain.

**Full grid = 550 lanes** (110 cores × 5; the board is 11×10, not 600):
- Real `work=0` workload: LOSSLESS with ~30× headroom — drain <4% busy, 0 stalls, 0 loss, 0 ts
  regressions. Effective end-to-end ~46 MB/s (production/dispatch-limited, NOT drain-limited).
- Saturated ceiling via `--fullread` (lossy microbench; ignores tail to force max drain):
  - SINGLE relay: relay **92.6% busy**, ~1.2–1.3 GB/s end-to-end — relay-funnel-bound.
  - DUAL relay: relays ~59% busy (funnel gone), readers ~77% @ **4.5 cyc/word** → **~1.8 GB/s aggregate
    read** = NoC-read-bound (near the 2-reader ~2.24 GB/s ceiling, § RDRBENCH). Reader ~0.89 GB/s each,
    relay ~3.0–3.5 cyc/word.
- So dual-relay lifts the artificial funnel; the wall becomes the fundamental NoC read. 3rd reader craters
  (§ RDRBENCH), so 2 readers + 2 relays is the sweet spot.

**Two silicon-only bugs fixed** (neither caught by the synthetic bench): (1) hi/lo wall-clock read TEAR →
tear-free `read_wall_clock`; (2) the real ts-regression cause — `mark_time` timestamped BEFORE
`ring_ensure_room`, so a full-ring stall (X280-STALL zone, timestamped later) got written ahead of a
marker carrying its pre-stall time → backwards per-lane jump. Fix: reserve ring room FIRST (take the
stall), read the clock AFTER. Commits (local): `1e9f01113a1`, `a76d615ea18`, `046d151ef7b`.

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
| `src/profsock.c` | §19 D2H-socket sender bench (`--socktest`: 1566 MB/s isolated push on bh-11, §20) |
| `src/profzone.c` | §20 PRODUCTION drainer: 2 readers (ILP `vle64`→LIM scratch, reshape) + 1 relay |
| `src/profstream.c` | §21 linearized sticky-src stream: split (2 rd + 1 relay) + single-hart `--direct` (lossless, ~615 MB/s, stall knee ~1300) |

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
`test_x280_profrelay`, `test_x280_profcons`, `test_x280_profsplit`,
`test_x280_stream` (§21, `--direct` single-hart lossless drain).
