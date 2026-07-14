# Regime-A Matmul — Golden Parity Suite (Step 1 oracle)

**Frozen 2026-07-14.** This is the parity oracle for converting the `test_regime_a_mm --unified`
prototype into the independent `ttnn.experimental.regime_a_matmul` op. The op must reproduce these
kernel times **within 5%** at the same manual config (Step 6 acceptance). Do not edit measured numbers;
re-measure and add a new dated block if the prototype changes.

Machine-readable companion: `golden_parity_suite.json`. Re-runner: `golden_parity_suite.py`.

## Measurement methodology
- Binary: `build/test/tt_metal/perf_microbenchmark/regime_a_mm/test_regime_a_mm --unified` (rebuilt
  2026-07-14 14:04 from committed source `d4c8df34b19`).
- Device: single Blackhole p150b (PCI a1, firmware bundle 19.5.0 — the latest UMD-supported BH version
  in this checkout; no mismatch since only one board is enumerated).
- Profiler: `TT_METAL_DEVICE_PROFILER=1`, `generated/profiler/.logs/profile_log_device.csv`.
- Clock `FREQ = 1.35e9`; DRAM peak `PEAK = 512e9` B/s.
- Kernel cycles = **max across all (core, RISC) `*-KERNEL` zones per run, min over runs 1..N** (run 0 is
  warmup). `--num-tests 6`.
- **Effective BW** = logical bytes `(Mt·Kt + Kt·Nt + Mt·Nt)·2048` ÷ kernel time (what the user sees).
- **Delivered BW** = padded physical bytes `(Mts·Kts + Kts·Nts + Mts·Nts)·2048` ÷ kernel time (engine
  quality independent of padding waste).
- ⚠️ Harness must run with **cwd = repo root** (else TT_METAL_HOME detection FATALs "root_dir empty").
- RISC roles: **BRISC = in1 sharded reader (`reader_ring.cpp`)**, **NCRISC = in0 ring all-gather +
  split-K reduction + output write (`in0_ring_writer.cpp`)**, **TRISC = compute (minimal_matmul
  `compute.cpp`, HiFi2, fp32 acc)**.

## Golden configurations

Config tuple = `(Ns, Pk, Sm, kb, nsb)`; ring all-gather in0; Sm=1 throughout (M-split lost 20/20 in the
picker sweep). All picked by the frozen picker: table shapes from `picker_table.py` (oracle-best of the
3262-config sweep), off-table shapes = best of 4 v2-cost candidates (`picker_v2.py`), best-measured frozen.

| # | Case | Shape M×K×N | Mt | Config (Ns,Pk,Sm,kb,nsb) | Cores | Kernel µs | Kernel cyc | Eff BW | Deliv BW | L1/core | Source |
|---|------|-------------|----|--------------------------|-------|-----------|-----------|--------|----------|---------|--------|
| 1 | base       | 32×6144×4608  | 1 | (1,12,1,2,1) | 96 | **118.6** | 160105 | **94.4%** (483 GB/s) | 94.4% | 60 KB  | v2 (won vs 3 alts) |
| 2 | base       | 64×6144×4608  | 2 | (1,6,1,4,2)  | 48 | **122.4** | 165269 | **92.5%** (474 GB/s) | 92.5% | 240 KB | table oracle |
| 3 | base       | 128×6144×4608 | 4 | (1,12,1,2,1) | 96 | **131.1** | 177039 | **88.4%** (453 GB/s) | 88.4% | 192 KB | table oracle |
| 4 | base       | 256×6144×4608 | 8 | (1,12,1,2,1) | 96 | **154.8** | 208983 | **78.4%** (401 GB/s) | 78.4% | 368 KB | v2 (won vs 3 alts) |
| 5 | small-N    | 128×6144×768  | 4 | (1,12,1,2,1) | 96 | **35.2**  | 47554  | **62.1%** (318 GB/s) | 62.1% | 192 KB | table oracle |
| 6 | non-divis  | 32×6080×4640  | 1 | (1,12,1,2,1) | 96 | **125.1** | 168848 | **89.2%** (457 GB/s) | **94.4%** | 60 KB | v2 (won vs 3 alts) |

Kernel cyc column = BRISC (the max RISC); NCRISC/TRISC track within ~1% on every shape (balanced
pipeline — see below).

## Per-RISC balance (max-core cycles)

| Shape | BRISC (in1 read) | NCRISC (in0 ring+reduce+out) | TRISC (compute) | Spread |
|-------|------------------|------------------------------|-----------------|--------|
| 32×6144×4608   | 160105 | 159414 | 159641 | 0.4% |
| 64×6144×4608   | 165269 | 164108 | 164571 | 0.7% |
| 128×6144×4608  | 177039 | 175756 | 176352 | 0.7% |
| 256×6144×4608  | 208983 | 207559 | 208287 | 0.7% |
| 128×6144×768   | 47554  | 46371  | 46890  | 2.5% |
| 32×6080×4640   | 168848 | 167897 | 168356 | 0.6% |

**All three RISCs finish within ~1% of each other** (2.5% on tiny small-N). The engine is co-limited, not
lopsided on any one role — the op must preserve this balance, not just the aggregate time.

## Physical in1 layout assumption (the persistent width-sharded weight)

in1 is DRAM width-sharded across **8 banks**; canonical per-bank shard ≈ `[K_padded, N_padded/8]` tiles:

| Shape | in1 per-bank shard (tiles) | Kt | Nt | Note |
|-------|----------------------------|----|----|------|
| ×4608 shapes | [192, 18] | 192 | 144 | Nt 144 = 8·18, no pad |
| 128×6144×768 | [192, 3]  | 192 | 24  | Nt 24 = 8·3, no pad |
| 32×6080×4640 | [192, 19] | 190 | 145 | Kt 190→192 (config pad), Nt 145→152 = 8·19 (shard pad) |

⚠️ **Step-2 design flag:** the prototype bakes **config-dependent** K-padding (from Pk·kb → `Ktl`) into
the stored in1 buffer (non-divis shape: Kt 190 stored as 192). The op's
`create_regime_a_weight_memory_config` must make the persistent layout a function of **in1 only** — pad
K/N to bank-alignment (`rup(Nt,8)`, K to a fixed alignment), and let each worker handle Pk/kb sub-slice
tails via valid-tile reads + local zero-fill, **not** by storing a wider tensor. The delivered-vs-effective
gap below is the evidence this matters.

## Key observations (the oracle's teeth)

1. **The non-divisible case is the headline lesson.** 32×6080×4640: effective 89.2% but **delivered
   94.4%** — a **5.2-point gap** that is *entirely padding waste* (Kt 190→192 = 1.1%, Nt 145→152 = 4.8%),
   not engine inefficiency. The engine still moves its padded bytes at full 94.4% quality. This is the
   "misleading performance" the plan's balanced floor/ceil tails must remove: the op should read only the
   ~145 valid N-tiles and local-zero-fill the ring/subblock remainder, closing eff toward deliv. **Op
   target for this shape: eff → ~94% (match delivered), i.e. recover the 5.2 points.**

2. **Divisible shapes have eff == deliv** (waste 0) — these are the clean parity anchors for Step 6. The
   op must hit each within 5%: 118.6 / 122.4 / 131.1 / 154.8 / 35.2 µs.

3. **BW declines monotonically with Mt** (94→92→88→78%) as M-dependent work (in0 ring forward, reduction
   depth, compute) grows while the M-independent in1 read stays fixed — consistent with the whole
   investigation. Mt≤4 is the strong-win core; Mt=8 at 78% is solid; small-N (62%) is the known hard
   corner (fixed cost under-amortized by tiny N).

4. **Config family is uniform: Sm=1 always; Pk fills "enough cores to saturate", not all cores** (Mt2
   won at 48c, not 96). kb∈{2,4}, nsb∈{1,2}. This is the config surface the op's planner must accept.

## Reproduction

```bash
cd /localdev/cglagovich/tt-metal
python3 tools/mm_sweep/golden_parity_suite.py   # re-measures all 6, rewrites the JSON
```

Single shape, e.g. #1:
```bash
ARCH_NAME=blackhole TT_METAL_HOME=$PWD TT_METAL_DEVICE_PROFILER=1 \
  build/test/tt_metal/perf_microbenchmark/regime_a_mm/test_regime_a_mm \
  --unified --m 32 --k 6144 --n 4608 --ksplit 12 --kb 2 --nsb 1 --num-tests 6
```
