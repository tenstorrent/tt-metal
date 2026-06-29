# Sampling perf: blaze vs deepseek_v3_b1 (per-stage, trace-replay)

Comparison of the mesh top-K sampling op in **tt-blaze** vs **deepseek_v3_b1 (current
`main`)**, measured the same way (Ambrose's "method 2": fast dispatch + tracy `-r -p`
+ trace replay), reading per-stage `DeviceZoneScopedN` zones from
`profile_log_device.csv` and op-level `DEVICE KERNEL DURATION` from
`ops_perf_results_*.csv`.

## Config (identical both sides)
- 4×2 mesh (8 devices), **101 active cores × 160 = 16160 vocab/device**
- k=32, p=0.95, **temperature=0.6**, seed=2005
- final sink: **mesh_coord (1,1), core_idx 100**
- fast dispatch (`TT_METAL_SLOW_DISPATCH_MODE` unset), single-op trace capture + N replays
- topology no-op (FABRIC_2D == TORUS_X; reduce axis is only 2 devices wide)

## ⚠ CSV column caveat (cost us a detour — note for parsing)
Current-`main` tt-metal `ops_perf_results_*.csv` has **`DEVICE KERNEL DURATION` at
column 20**; `DEVICE FW DURATION` (col 19) is ~2–3× larger (includes op-to-op / idle
overhead). The older tt-blaze tt-metal CSV had kernel duration at **col 19**. Parse the
header, don't assume a fixed index — mixing the two makes deepseek look ~3× slower than
it is. (per-RISC kernel durations are at col 26–30 here, 25–29 on the blaze CSV.)

## Op-level (DEVICE KERNEL DURATION, final/reduce device, trace-replay plateau)
| | total | critical RISC | TRISC compute | exposed past compute |
|---|---|---|---|---|
| deepseek (`main`) | **~22.7 µs** | BRISC 22.7 | ~21.7 µs | ~+1 µs |
| Ambrose's reference | 21.8 µs | — | — | — |
| blaze | **~24.7 µs** | NCRISC 24.7 | ~21.4 µs | **~+3.3 µs** |

- **deepseek reproduces Ambrose's 21.8 µs** (~22.7 µs here; the ~1 µs is config noise).
- **blaze is ~2 µs (~8%) slower**, and the difference is **not compute** (both ~21.5 µs
  TRISC) — it's **data movement on the reduce path**: blaze is NCRISC-bound and its
  reader/reduce NCRISC (24.7 µs) runs ~3.3 µs past when compute finishes; deepseek is
  BRISC-bound and exposes only ~1 µs. blaze NCRISC 24.7 vs deepseek NCRISC 18.3 → blaze's
  reduce-side reads are ~6 µs heavier and sit on its critical path.

## Per-stage compute zones (final core, median over trace replays)
From `profile_log_device.csv` (zone log; unaffected by the col-index issue). NOTE: these
zone durations **overlap** across the 3 TRISCs + NCRISC, so they do NOT sum to the op
total — use them to compare stage-by-stage, not to reconstruct the total.

| Stage | blaze | deepseek (`main`) |
|---|---|---|
| LocalTopK (`SP-PHASE1LLK`) | ~4.5 µs | ~4.3 µs |
| CrossCore merge (`SP-PHASE2LLK`) | ~8.7 µs | ~10.0 µs |
| Cross-device merge | ~3.8 µs | ~10.0 µs (`SP-MESH1LLK` 7.1 + `SP-MESH2LLK` 2.9) |
| Softmax+topP (`SP-TOPP-TRISC`) | ~4.3 µs | ~4.2 µs |
| Multinomial / final | ~2.6 µs | `SP-FINALCORE` ~4.5 µs |

LocalTopK and Softmax are ~equal (blaze ported softmax from deepseek). The compute
stages are comparable overall; the op-level difference is dominated by the NCRISC
reduce-read exposure described above, not by any single compute stage.

## Method note (deepseek code change required to trace)
Current-`main` `SamplingOp.op` (`_op_mesh_topk`) calls `ttnn.create_global_semaphore`
**inside** the op, which `begin/end_trace_capture` forbids (`TT_FATAL: Writes are not
supported during trace capture`). This commit hoists those two semaphores to optional
params (`receiver_global_sem`, `local_ready_global_sem`); the test pre-creates them
before capture. Defaults preserve existing behavior. (Ambrose's original patch traced
fine, so his op version didn't self-allocate these — `main` drifted.)

## Bottom line
- deepseek `main` reproduces ~21.8 µs; blaze is ~24.7 µs → **blaze ~2 µs (~8%) slower**.
- The gap is **blaze's NCRISC reduce-read path** (~+3.3 µs exposed past compute), not the
  compute kernels (TRISC compute is equal). deepseek keeps its critical RISC ~1 µs over
  compute; blaze ~3.3 µs over.
- Optimization target for blaze: overlap/shorten the cross-device reduce reads on NCRISC
  so the reader stops being the exposed long pole.

## Question for Ambrose
- Sanity-check the per-stage zone mapping (SP-* → blaze stages) and whether the
  BRISC-bound (deepseek) vs NCRISC-bound (blaze) critical-path difference matches your
  understanding of the two reducers.
