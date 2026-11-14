# dfb_isr_timing_bench: design

Issue #55788, ISR follow-up. Status: design only, nothing implemented.

## Goal

Characterise how DM0's implicit-sync ISR posts and acks credits, and produce a sizing rule:

> For a given number of readers/writers and access pattern, which **NoC transaction size × DFB depth × number of sections** hides the ISR latency, i.e. keeps the stream running at NoC/DRAM speed rather than waiting on the ISR?

Everything is measured on the Quasar emulator (2x3_DISPATCH-userbits, slow dispatch), with DRAM as the read source and write sink.

## How the mechanism works today (origin/main 72c529c7e70)

| Fact | Where |
|---|---|
| Implicit sync exists only for **DM** sides. A TRISC side never raises an interrupt. | `dataflow_buffer.cpp:2893, 2927` (`!producer_is_tensix_only`) |
| The ISR increments whatever tile counters (TCs) the descriptor lists, including **Tensix** TCs. A DM producer feeding a TRISC **unpacker** gets its credits *posted* by the ISR. A DM consumer draining a TRISC **packer** gets the packer's space *acked* by the ISR. | `dataflow_buffer_isr.h` |
| One DM0 ISR serves every DFB on its core. All pending transaction IDs (trids) are handled in one entry, in ctz order. Each Tensix core has its own DM0, so the ISR is per core. | `dataflow_buffer_isr.h` |
| **Sections = trids per side.** The FIFO is split into n sections. Each section has its own trid, and its threshold covers `num_entries / n` entries (÷ block_size for BLOCKED). The host picks the **smallest legal n in 2..4**, which is almost always 2. | `compute_optimal_txn_id_count`, `dataflow_buffer.cpp:1430` |
| A producer cannot reuse a section's trid until the ISR has posted that section's previous batch. So at most n sections are in flight, and the producer stalls on the ISR when the other n−1 sections drain faster than the ISR responds. Consumers behave the same way through acks. | `prepare_implicit_read/write`, `dataflow_buffer.inl:843, 906` |
| Legality: `num_entries % (n × num_prods_or_cons × num_tcs) == 0`; entries per trid and the threshold must each be ≤ 255; BLOCKED needs `per_txn % (block × num_tcs) == 0` (or the split-credit rules). | `dfb_txn_descriptor_violation`, `dataflow_buffer.cpp:1302` |
| NoC transaction size = `entry_size` (STRIDED side) or `block_size × entry_size` (BLOCKED side). One BLOCKED transaction must be ≤ `NOC_MAX_BURST_SIZE` = 256 words × 256 B = 64 KB. | `Noc::async_read`, `validate_implicit_burst_fits_one_packet` |
| DFB trid pool: 24 per core (8–31). With 2 sections per side, at most 12 implicit DFB sides per core. Up to 6 TCs round-robined per hart, at most 18 TCs per trid descriptor. | `dataflow_buffer_config.h` |
| L1 is 4 MB. Free L1 starts at about 0x51440 after kernel config, which bounds the "huge DFB" case. | `dev_mem_map.h`, earlier coherence-test notes |

Model to test:

```
ISR hidden  ⇔  (n − 1) × section_bytes / BW_eff(txn_size, readers)  ≥  L_isr(pending trids, counters per trid)
section_bytes = num_entries × entry_size / n
```

`BW_eff` comes from B0, and `L_isr` comes from B1. B2 to B7 check the model's prediction against measured throughput.

## Common infrastructure

**Binary.** New `tests/tt_metal/tt_metal/api/dataflow_buffer/dfb_isr_timing_bench.cpp`, built by the same CMake target list as `dfb_init_timing_bench`. Each benchmark below is a `--suite`. Each configuration is one row, given on the command line or as a generated sweep list.

**Sections override (host change needed).** Today `n` cannot be chosen. Add a test-only override that forces `n` (2, 3 or 4) for a DFB side and goes through the same `dfb_txn_descriptor_violation` check. Two options:
- (a) an rtoptions env var, `TT_METAL_DFB_FORCE_NUM_TXN_IDS`, applied to every implicit DFB;
- (b) an optional field in `DataflowBufferConfig` (or the experimental Quasar API), so each DFB can differ.

(b) is needed if B3b ever mixes section counts; otherwise (a) is enough. See open question 1.

**Legal-config generator.** The bench computes legality on the host before launching. It mirrors the rules above, logs every skipped configuration with its reason, and never relies on the host's TT_FATAL to filter.

**Measurement: waveforms only (decided 2026-11-03).** Accurate numbers come only from post-processing emulator waveforms with `quasar-isr-latency/`. The device code therefore carries **no timing instrumentation at all**: no ISR probe, no stall counters, no `rdcycle`. Kernels only run the stream (and LOOP runs check their data).

| Metric | Where it comes from in the waveform (`isr_bench_post.py`) |
|---|---|
| ISR latency per credit event | T0 = trid threshold reached (`risc_to_noc_arb.o_[wr_]tiles_to_process_irq`), T1 = DM0 interrupt line, Te = vector-table retire on DM0, T3b = DM0's credit write on that DFB's tile counter (`i_local_wren/addr`), T4 = overlay counter value, T5 = Neo handshake. "queued" = the interrupt was already pending or running when T0 fired. |
| ISR occupancy | DM0 vector → `mret` intervals per launch: entries, busy cycles, credit writes per entry. |
| Producer / consumer stall | Tight polling loops in each worker DM's retire trace (`io_trace_0_*` of `tile_prci_domain_N`), with the hottest loop lines for `addr2line` on the kernel ELF. |
| Throughput | Bytes per launch over the worker DMs' kernel-code window; steady state from threshold events after the first lap. |
| Launch windows | Kernel-code PCs (≥ 0x20000) on the worker DMs; matched in order to the manifest lines. |

**Manifest.** The bench appends one JSON line per launch (`--manifest`): configuration, cores (logical and virtual), and per DFB the producer/consumer txn ids, threshold, entries per txn id, and every risc's tile counters as `[tensix, counter]`. `gen_rc.py` turns a manifest into the export signal list (adds worker-DM retire traces and the tile-counter banks the configurations use); `isr_bench_post.py` uses it to attribute waveform events. Self-test: on `run1.vcd.gz` it reproduces the README (355 / 74 cycles interrupt → Neo).

**Correctness.** LOOP runs with the same transaction size on both sides compare the output tensor with the input every launch. A run is invalid, not fast, if the post-processor's launch-window count does not match the manifest.

**Session structure.** Emulator setup dominates run time, so one device-open runs a group of configurations. Each configuration is its own Program, run as 1 cold launch + R measured launches (R = 8 by default, all reported as discrete states, not mean ± sd).
- The first and last configuration of every group is the same control configuration. If the two differ, the group is flagged for drift (the `--case all` state carry-over problem).
- Cold-ISR numbers come only from launch 0 of a fresh device-open, never from mid-group launches.

**Captures.** One bench process = one emulator session = one waveform. Group several configurations per process (`--filter`, `--launches`), keep each launch's stream short enough for the dump (the bench caps streams at 2 MB per DFB, 8 MB for B7), and post-process with the manifest lines of that process (`--seq-offset`).

## Benchmarks

### B0. Raw transfer baseline (no DFB, no ISR)

Purpose: `BW_eff(txn_size, readers)`, the ceiling every DFB stream is compared against.
- DM reader threads issue plain `noc_async_read` from DRAM into an L1 ring with a trid barrier per section. The same ring geometry is used as B2, but no TCs and no ISR are involved.
- Axes: txn size {64 B, 256 B, 1 KB, 2 KB, 4 KB, 8 KB, 16 KB, 32 KB, 64 KB} × readers {1, 2, 4, 6} × outstanding transactions {1, 2, 4, 8, unlimited}.
- Write mirror: L1 → DRAM `noc_async_write`.
- Also an **explicit-sync DFB** baseline (producer `push_back` after a barrier) at the B2 knee points, showing what implicit sync buys.

### B1. ISR latency microbenchmark

Purpose: `L_isr`, from interrupt raised to the credit being visible in the TC, as a function of load.
- **Base:** one DFB, one DM producer, one DM consumer spinning on the TC's posted count. Small entries (64 B), so the NoC part is negligible. The producer stamps `rdcycle` after issuing the last transaction of a section, and the consumer stamps when the posted count moves.
  - Raise-time correction: subtract the B0 one-transaction read latency at that size.
  - Calibrate against waveforms (interrupt-rise → TC-update) at 3–4 points.
- **B1a, trids pending per ISR entry:** k ∈ {1, 2, 4, 8, 12} DFBs on one core whose sections complete at the same time (the producers release together through a shared start flag). Measures queueing inside one ISR entry, i.e. the latency of the last trid serviced.
- **B1b, TCs per descriptor:** 1, 2, 4, 6, 12, 18 counters per trid, using multi-consumer STRIDED, ALL broadcast and split BLOCKED rings. Measures the per-counter cost.
- **B1c, path:** post (read side) vs ack (write side), and a TRISC TC vs a DM TC as the target.
- **B1d, cold vs warm:** launch 0 vs later, and the first section vs later sections. This connects to `cold_isr/README.md` (vector line, stack lines, handler line).

### B2. Read-side streaming sweep (main deliverable)

DRAM → DFB (DM producer, implicit sync) → consumer.
- **Consumers, both:**
  - (a) a TRISC unpacker that pops as fast as possible, with no math (ISR posts to the Tensix TC);
  - (b) a DM sink using explicit-sync `pop_front`, so only the read-side ISR path runs.
- **Axes:**
  - entry size (= txn size for STRIDED) {256 B, 1 KB, 2 KB, 4 KB, 8 KB, 16 KB};
  - sections n ∈ {2, 3, 4} (forced);
  - entries per section ∈ {1, 2, 4, 8, 16, 32, 64, 128, 255}, so `num_entries = n × entries_per_section`, which sets the threshold and the TC capacity;
  - a total DFB byte cap, to stay in free L1.
- **Tensor:** at least 16 laps of the ring and at least 1 MB. It is grown until the steady-state throughput stops changing (checked once at the smallest and largest DFB).
- **Output:** throughput / B0 ceiling and the ISR-wait stall fraction, against section bytes, one curve per (txn size, n). The **knee** is the smallest section size with ISR-wait < 1% of the window and throughput within 5% of the plateau. Compare it with the model's prediction.

### B3. Reader scaling

Each part runs B2's knee configurations plus one step on each side of them.
- **B3a, threads on one DFB:** 1, 2, 3, 6 DM producer threads (STRIDED; `num_entries` divisible by n × producers).
- **B3b, DFBs on one core sharing DM0:** 1, 2, 4, 6, 12 independent DFBs, each with one reader, all streaming at once. Checks aggregate ISR event rate against `L_isr` from B1a.
- **B3c, Tensix cores:** the same single-DFB stream on 1, 2, 4, 6 cores at once. Each core has its own ISR, so this isolates NoC/DRAM contention from ISR load. Use the 2x3 grid's worker cores.

Output: the knee section size as a function of total readers, per variant.

### B4. Write side

DFB → DRAM through a DM consumer with implicit-sync `async_write` (ISR acks).
- **Producers, both:**
  - (a) a TRISC packer producing as fast as possible (ISR acks to the Tensix TC);
  - (b) a DM producer using explicit sync.
- Same axes as B2, plus writer counts {1, 2, 4, 6} as in B3a/B3b.
- **B4-loop:** DRAM → DM producer (implicit) → DFB → DM consumer (implicit) → DRAM. Both paths share one ISR. This is the old `--case isr` generalised over the B2 axes, showing post and ack events interleaving and competing.

### B5. Access patterns

BLOCKED is not exercised anywhere in this bench (decided 2026-11-03; the one BLOCKED point that ran on the emulator, block 16 = 16 KB transactions, hung). Producers are STRIDED; consumers are STRIDED or ALL.

At fixed bytes per section (32 KB), compare:
- STRIDED → STRIDED (baseline);
- STRIDED → ALL (2 consumers: more counters per ISR event).

Sweep the entry size (= one NoC transaction) ∈ {1, 4, 16 KB} at those fixed section bytes, so the transactions per section change while the threshold bytes and interrupts per byte don't. That separates "bigger transactions" from "fewer interrupts".

Readers: 1 and 4. Consumers: TRISC and DM. 24 configurations.

### B6. Stress: small DFBs, small thresholds

Purpose: the ISR saturation point, which is the floor for any recommendation.
- Entries per section ∈ {1, 2, 4}, small entries (64–256 B), readers 1 → 6 and DFBs 1 → 12 (B3b layout).
- Report ISR time share (probe busy cycles / window) and throughput.
- Saturation is where the time share approaches 100% and throughput stops scaling with readers. From that, give the events per kilocycle DM0 can sustain.

### B7. Huge DFB, large tensor

Purpose: confirm the ISR is fully hidden and find which limit binds instead.
- Largest DFB that fits in free L1 (about 3.5 MB), largest entries (16–64 KB BLOCKED) and the 255 per-trid entry cap, so `num_entries` ≤ n × 255.
- Tensor 8–32 MB streamed in.
- Readers 1 and 6, both consumer kinds.
- Expected result: the throughput plateau equals B0 and the ISR time share is near 0. Any shortfall is attributed to the 8-bit threshold or the 4-section cap.

## Deliverables

1. `L_isr(k pending, c counters, path)` table (B1), cold and warm.
2. `BW_eff(txn size, readers)` table (B0).
3. Knee curves (B2–B5) with the model's predicted knee overlaid.
4. Sizing table: (readers/writers, access pattern, consumer kind) → minimum (txn size, entries per section, n), plus the B6 saturation floor and the B7 ceiling.
5. Raw logs and arm sources kept beside the write-up (`isr-55788/isr_bench/`), as in `cold_isr/`.

## Order of work

B0 + B1 first: they set the model's two inputs, and B1 validates the instrumentation against waveforms. Then B2 (read side), B3, B4, B5, B6, B7. Port the ISR probe and add the sections override before B1.

## Implementation status (2026-11-03, branch shengxiangji/isr-55788, uncommitted)

| Piece | File | State |
|---|---|---|
| Section override | `tt_metal/llrt/rtoptions.{hpp,cpp}` (`TT_METAL_DFB_FORCE_NUM_TXN_IDS` + setter), `tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp` (`compute_optimal_txn_id_count`) | built |
| Bench host | `tests/tt_metal/tt_metal/api/dataflow_buffer/dfb_isr_timing_bench.cpp`, CMake target `dfb_isr_timing_bench` | built; suites b0–b7, `--list`, `--filter`, `--launches`, `--manifest`, `--control`, `--compile-only` |
| DM kernels | `test_kernels/dataflow/dfb_isr_bench_dm_{producer,consumer}.cpp`, `dfb_isr_bench_common.h` | JIT-compiled (mock) |
| TRISC kernel | `test_kernels/compute/dfb_isr_bench_trisc.cpp` | JIT-compiled (mock) |
| Signal list | `quasar-isr-latency/gen_rc.py` | written; needs the logical→RTL tile map for cores other than (0,0) |
| Post-processor | `quasar-isr-latency/isr_bench_post.py` | self-tested on run1.vcd.gz |

Compile check (mock Quasar cluster, real JIT compile + link of firmware and every kernel variant, `--compile-only`): every configuration that passes the host checks compiles. Per suite, compiled / generated: b0 178/180, b1 22/22, b2 300/324, b3 52/52, b4 75/78, b5 24/24 (after removing BLOCKED), b6 48/48, b7 12/12. The rest are over the L1 budget.

Emulator: B5 (old 36-point version, one launch each) ran 16 configurations in 35 s after 81 s of startup (~2.2 s per configuration, 0.4-1.4 s of it in LaunchProgram), then hung on its first block-16 BLOCKED point; BLOCKED was removed afterwards. Untested assumptions to check on the first capture: the worker-DM retire-trace signal names (`tile_prci_domain_N`) and the tile-counter bank names for banks 1–3 in the export; TRISC kernels without `tile_regs_*` handshakes; the explicit-sync DFB baseline for B0 is not implemented.

## Open questions

1. Sections override: should the forced `n` be a global env var (a) or a per-DFB config field (b)? (b) touches the experimental public API.
2. Base branch: main or the PR branch? The PR changes DM0's descriptor pool layout (compact pool, hot-first), which is exactly what the ISR reads. Current assumption: main, rebased onto the PR once it merges.
3. TRISC unpacker/packer "as fast as possible": pop/push with no LLK data movement, or real unpack/pack of tiles? The first isolates sync; the second is realistic but adds Neo time.

### Emulator findings (2026-11-03)

- **Implicit-sync transactions above 8 KB hang.** `NOC_OVERLAY_MAX_BYTES_IN_PACKET` is 8 KB and the overlay splits larger reads into packets. A 16 KB transaction hung twice: once as BLOCKED (16 x 1 KB), once as a STRIDED 16 KB entry. 1, 4 and 8 KB passed. In the watcher, DM0 sat idle, the producer sat in `prepare_implicit_read` (PIRW) and the sink in `wait_front` (WFW), so the ISR never credited the first section. Mechanism (2026-11-04): the overlay spec (tensix/soc/overlay docs/rocc/noc_cmdbufs/tiles_to_process.md) says TR_ACK/WR_SENT count one per NIU packet response, while the host threshold counts transactions (dataflow_buffer.cpp:1409). A 16 KB txn is 2 packets, so the edge-triggered IRQ fires mid-section; the ISR's clear-to-0 and its pending-bit write-0-to-clear then drop counts or edges, and the producer's posted check never catches up. Which race occurred was not captured (no waveform). 8 KB is a firmware choice (noc_cmd_buf_common.h:74, "lower than the 64KB HW default to avoid NOC congestion"), not a hardware limit: with it temporarily raised to 64 KB, on emu release_Quasar_A0_1.0.0_v2026.W42.0, B5 at 16/32 KB ran 12/12 (incl. the config that hung) and B4 write/loop at 16/32 KB ran 43/43 with 0 mismatches (change reverted). An audit found no in-tree implicit-sync user above 8 KB (max exactly 8192 B). main's host check (`validate_implicit_burst_fits_one_packet`) compares against the 64 KB burst, not the 8 KB packet, and STRIDED entries are not checked at all. The bench caps implicit flows at 8 KB, and B0 raw reads keep the full range.
- **DM producer -> DM ALL consumers with implicit sync is unsupported on main.** main's tests skip it ("legacy parity"), and it hung here. The bench rejects it; ALL is exercised only with TRISC consumers.
- **B5 (18 runnable of 24): all 18 pass**, in 198 s wall clock including about 140 s of emulator startup. The 18 programs took 50 s, about 2.8 s each including host work.
- **B2 (2026-11-04, emu W42, soc-l-12).** No waveform: 260 run, 10 skipped (L1 budget), 0 mismatched, 21.5 min. With waveform the emulator runs ~45x slower per launch (~110 s vs ~2.5 s), so the full suite (~16 h) does not fit; the capture used a 24-configuration subset (peer dm/trisc x entry 1/8 KB x sections 2/4 x eps 4/16/64, 2 launches each; zrun.22, 43 GB ztdb, 54 min).
  - ISR latency, threshold → DM0 credit write (T0→T3b), all 1255 warm events: min 62, median 68, p90 82, max 86 DM cycles; identical for DM and TRISC consumers and across txn size, sections and section size. Split: T0→interrupt line 2, interrupt → handler entry 9 (max 15), handler → credit write 51–75. Credit → counter value 1, → Neo handshake 8 (TRISC). The only cold event is the first ISR of the session: 427 (51 to enter + 374 in the handler); later launch-0s are warm. No event was queued behind another.
  - ISR occupancy = sections per cycle window: ~14% of DM0 at 4 KB sections (1 KB txns), ~11% at 32 KB (8 KB txns), ~4% at 16–128 KB, ~1% at 64–512 KB.
  - Throughput (whole launch, warm; B/DM cycle): 1 KB txns 5.2 (4 KB section) / 5.7–6.1 (16 KB) / 6.0–6.4 (64 KB); 8 KB txns 31–34 (32 KB) / 43–45 (128 KB) / 38–48 (512 KB, < 1 lap). Txn size sets the ceiling; section size sets the knee: ≥16 KB sections are within ~5% of the 1 KB plateau, ≥128 KB for 8 KB txns. 4 sections instead of 2 changes nothing at the same section size, so the small-section loss is a per-section cost, not ring depth. B0 (raw ceiling) not yet measured.
  - Post-processing: the W42 DM0 vector slot is 0xef34 (not 0xED00+52) and launches are only ~16.7k cycles apart (Zebu stops the clock while the host works); `isr_bench_post.py` now finds the vector slot from the trace and defaults `--launch-gap 8000`. Its spin detector reports 0% (polling loads take > 1 cycle per retire); by PC, a DM sink spends ~68% in `wait_front` occupancy polling and the producer ~13% in `prepare_implicit_read` posted polling (1 KB, 4 KB sections) — needs a fix before stall numbers are reported.
