# BEVFormer encoder — performance

Measured results live here. The backlog lives in
[perf_optimization_candidates.md](perf_optimization_candidates.md); each landed change moves from
there to the table below and gets a report under [`perf_reports/`](perf_reports/).

## Harness

| | |
|---|---|
| test | [`tests/perf/test_layer_perf.py`](../tests/perf/test_layer_perf.py) — **one** encoder layer |
| config | `nuscenes_base`, `bev_size=(100, 100)`, `batch_size=1`, 4 levels, 6 cameras |
| device | Wormhole N150 |
| gate | PCC ≥ 0.997 against the torch reference, asserted inside the perf test |
| metric | `DEVICE KERNEL DURATION` and `OP TO OP LATENCY`, summed over the signposted region |
| iters | 1 warm-up, `DEVICE_PERF_ITERS = 1` — the gap column carries region-entry cost, see below |

```bash
MESH_DEVICE=N150 python -m tracy -p -r -v --op-support-count 20000 -m pytest \
  models/experimental/bevformer/tests/perf/test_layer_perf.py::test_bevformer_layer_perf -sv
```

CSV lands in `generated/profiler/reports/<timestamp>/`. **Keep the CSV** for any number quoted here —
two stages already cite CSVs that were deleted between sessions and cannot be re-audited.

**One layer, not six.** The 6-layer encoder harness
([`test_encoder_perf.py`](../tests/perf/test_encoder_perf.py)) still runs and still gates PCC, but it
cannot be profiled: it emits more device ops than Tracy's per-device buffer holds and
`process_ops_logs` aborts with `Device data missing`. The layer is the repeated unit anyway — encoder
≈ 6 × layer plus one point-sampling pass. Multiply by 6 for an encoder estimate.

**Not traced replay.** `TTSpatialCrossAttention.forward` reads `bev_mask` back to host and the result
decides the shapes of the ops after it, so the encoder is not trace-capturable today. The signposted
region therefore carries host dispatch.

## Results

| # | change | kernel | gap | wall | Δ | ops |
|--:|---|---:|---:|---:|---:|---:|
| 0 | [baseline](perf_reports/00-baseline.md) | 655.6 ms | 2416.5 ms | **3072.1 ms** | — | 131 |
| 1 | [SCA rebatch and scatter-back on device](perf_reports/01-sca-rebatch-on-device.md) | 683.0 ms | 44.4 ms | **727.4 ms** | **−2344.7 ms wall** | 146 |
| 2 | [rebatch plan resolved once per forward](perf_reports/02-rebatch-plan-hoisted.md) | 680.4 ms | 46.3 ms | 726.7 ms | −0.7 ms | 146 |
| 3 | [constant uploads cached](perf_reports/03-constant-uploads-cached.md) | 681.7 ms | 40.5 ms | 722.2 ms | −4.5 ms | 121 |
| 4 | [fused multi_scale_deformable_attn](perf_reports/04-fused-msda.md) | 489.5 ms | — | — | **−191.6 ms kernel** † | 129 |
| 5 | [offset normalizer folded into the Linear](perf_reports/05-offset-normalizer-folded.md) | 456.8 ms | — | — | **−32.7 ms kernel** | 127 |
| 6 | [dead SCA `key` permute deleted](perf_reports/06-sca-key-permute-deleted.md) | 438.4 ms | 14.4 ms | — | **−18.1 ms kernel** ‡ | 126 |
| 7 | [sampling grid built in ROW_MAJOR](perf_reports/07-sampling-grid-in-row-major.md) | 356.2 ms | 17.7 ms | — | **−82.2 ms kernel** ‡ | 131 |
| 8 | [`attn` prepared once per call](perf_reports/08-attn-prepared-once-per-call.md) | 311.3 ms | 24.2 ms | — | **−44.9 ms kernel** ‡ | 113 |
| 9 | [head-major sampling grid](perf_reports/09-head-major-sampling-grid.md) | 286.8 ms | 20.9 ms | — | **−24.5 ms kernel** ‡ | 112 |
| 10 | [`value` head split without the padding](perf_reports/10-value-head-split-unpadded.md) | **280.2 ms** | 14.0 ms | — | **−6.6 ms kernel** ‡ | 106 |

`kernel` = summed `DEVICE KERNEL DURATION`. `gap` = summed `OP TO OP LATENCY`, i.e. device idle
waiting on host dispatch. `wall` = kernel + gap, per layer.

† Row 4's Δ is against a same-session re-measure of row 3 (`2026_08_27_23_03_17`: 681.1 ms kernel,
121 ops), not against row 3's 722.2 ms.
‡ Rows 6–10 (candidate 5's sub-items) each quote a Δ against a same-session re-measure of the
previous stage. Row 6's is `2026_09_01_22_47_34` — 456.5 ms, 127 ops — which reproduced stage 05's
456.8 ms to 0.3 ms and its per-op table exactly, so the harness is stable on kernel time across days.

Rows 1–3 were re-measured 2026-08-27 (`2026_08_27_08_56_58`, `2026_08_27_21_31_29`,
`2026_08_27_13_47_55`), each on
that stage's `tt/` sources — Python-only diffs, no rebuild. PCC 0.999608 in all three.

**Cumulative: 681.1 → 280.2 ms kernel, −58.9%**, all measured on this harness. Rows 1–3 are flat on
kernel by construction — they remove dispatches, not device work; read their value from the
[encoder table](#encoder-level-changes). Rows 4–5 took −224.3 ms by replacing the deformable-attention
decomposition and folding the normalizer that fed it. Rows 6–10 took another −176.3 ms
([candidate 5](perf_optimization_candidates.md#candidate-5--data-movement-vs-compute)) without a new
kernel or an upstream change.

## The gap column is not reliable

Rows 4–10 quote **kernel deltas, not wall deltas**, and rows 0–3's wall figures deserve the same
suspicion. The layer was profiled twice on identical stage-05 code, minutes apart:

| run | kernel | gap | wall |
|---|---:|---:|---:|
| `2026_08_28_10_23_13` | 456.8 ms | 93.4 ms | 550.2 ms |
| `2026_08_28_10_30_24` | 456.8 ms | 151.2 ms | 608.0 ms |

**Kernel reproduces to 0.1 ms; gap differs by 57.8 ms on the same binary** — and stage 04 measured
14.0 ms on the same harness the night before. Taken at face value the wall column would make stage 05
a ~100 ms regression while its kernel dropped 32.7 ms. **Use `DEVICE KERNEL DURATION` for
stage-to-stage comparison.** Stage 01's gap was likewise corrected from 218.3 ms to 44.4 ms on
re-measurement.

### The gap column carries region-entry cost

The first ops after `signpost("start")` are charged the host sync, deallocation and queue refill that
precede them, because the command queue is empty at region entry. At `DEVICE_PERF_ITERS = 1` that
lands inside the reported gap and it is large and noisy — 6.4, 25.7 and 38.1 ms across three runs,
sitting on one or two ops. A `DEVICE_PERF_ITERS = 2` pass separates it:

| tree | iter 1 gap | iter 2 gap | entry cost | CSV |
|---|---:|---:|---:|---|
| stage 1 | 37.5 ms | 30.9 ms | 6.6 ms | `2026_08_27_20_58_53` |
| stage 2 | 45.1 ms | 38.8 ms | 6.3 ms | `2026_08_27_21_04_33` |
| stage 3 | 16.1 ms | **8.9 ms** | 7.2 ms | `2026_08_27_20_44_43` |

**Steady-state gap was 8.9 ms over 121 ops at stage 3** — host dispatch is otherwise fully hidden.
Two trees with identical op counts (stages 1 and 2) differ by 7.9 ms, which puts the **noise floor at
~±8 ms**. More warm-up does not help; only a second measured iteration separates it, at the cost of
doubling the ops in the region, so the harness stays at 1. **Do not read gap deltas below ~40 ms from
the results table.** Raise `DEVICE_PERF_ITERS` to 2 and segment the CSV at its midpoint when a gap
delta is the claim.

Independently confirmed by [host_fallback_gap.md](host_fallback_gap.md), a stage-04 capture on a
different harness: **zero** TTNN host-fallback ops exist on the forward path, and the host residue is
~3.9 ms of `build_rebatch_plan` transfers that the encoder hoists out of the layer entirely.

### Encoder-level changes

Stage 2's win is per-forward and the layer harness takes the unhoisted path — it drives
`TTSpatialCrossAttention` directly, which builds its own `SCARebatchPlan` when none is supplied — so
that stage is only visible end to end.

| # | change | encoder wall (6 layers) | Δ | report |
|--:|---|---:|---:|---|
| — | *before stage 2* | 4385.5 ms median | — | — |
| 2 | rebatch plan resolved once per forward | 4290.9 ms median | −94.6 ms (−2.2%) | [02](perf_reports/02-rebatch-plan-hoisted.md) |
| 3 | constant uploads cached | **4234.5 ms** median | −56.4 ms (−1.3%) | [03](perf_reports/03-constant-uploads-cached.md) |

End-to-end host time over 11 iterations, not a profiled figure — noisier than the layer harness.
Stage 2's median and minimum disagree 3×, so read it as **~1–2%**; stage 3's agree (−56.4 vs
−48.3 ms). Together **−151.0 ms, −3.4%.** Both are **steady-state**: the benchmark warms up before
timing and stage 3 caches across forwards, so the first frame still pays.

## Where the time went, by op, across stages

Kernel ms per op code. Stage 3 is the last pre-fusion tree; stage 10 is HEAD.

| Op | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MSDAOperation | — | 167.6 | 167.8 | 167.9 | 167.9 | 168.1 | 167.9 | **167.9** |
| GridSample | 116.0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| Concat | 115.5 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| ReshapeView | 157.0 | 77.1 | 77.1 | 77.4 | 55.3 | 43.0 | 42.3 | **28.0** |
| Permute | 105.4 | 62.5 | 62.7 | 43.4 | 43.7 | 43.4 | 19.9 | **33.8** |
| BinaryNg | 85.7 | 81.5 | 48.2 | 49.3 | 2.3 | 2.4 | 3.0 | **2.9** |
| UntilizeWithUnpadding | 28.9 | 42.4 | 42.4 | 42.4 | 26.5 | 13.2 | 13.8 | **13.1** |
| TilizeWithValPadding | 14.6 | 17.1 | 17.1 | 17.1 | 17.2 | 0.9 | 0.9 | **0.9** |
| Slice | 29.1 | 13.2 | 13.3 | 13.2 | 13.2 | 11.5 | 11.4 | **11.0** |
| Scatter | 10.5 | 10.5 | 10.5 | 10.5 | 10.5 | 10.5 | 10.5 | **10.5** |
| Transpose | 0.1 | 7.9 | 8.2 | 8.0 | 7.9 | 6.7 | 7.5 | **2.1** |
| Matmul | 4.7 | 4.7 | 4.7 | 4.7 | 4.7 | 4.7 | 4.7 | **4.7** |
| Reduce + FillPad | 9.2 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| **total** | **681.7** | **489.5** | **456.8** | **438.4** | **356.2** | **311.3** | **286.8** | **280.2** |

The total row is the measured layer figure, not the column sum: minor ops (Softmax, LayerNorm, Unary,
Clone, Embeddings, RepeatCodegen, UntilizeCodegen) are omitted and account for the ~5 ms difference.
Each column's source CSV is named in that stage's report.

Three things this table says that no single report does:

- **`MSDAOperation` has not moved since it appeared** — 167.6 → 167.9 ms across seven stages, and it
  is now **59.9% of the layer**. Everything else has been cut in half or deleted. It is an op-level
  question ([candidate 10](perf_optimization_candidates.md#candidate-10--msdaoperation-itself) /
  [12](perf_optimization_candidates.md#candidate-12--one-fused-call-for-all-levels)), not a
  model-level one.
- **`Matmul` and `Scatter` are flat and irrelevant** — 4.7 and 10.5 ms since stage 03. Matmul has
  been ~1% of the layer since the baseline, so nothing in the matmul-tuning playbook applies here.
- **`Permute` went up at stage 10.** That is [5d](perf_reports/10-value-head-split-unpadded.md)
  deliberately trading a 21 ms padded reshape for a 14 ms ROW_MAJOR permute — visible here as
  `ReshapeView` −14.3 against `Permute` +13.9.

Classified at stage 10: **data movement 90.8 ms against 178.8 ms of compute, a ratio of 0.51** where
stage 05 measured 1.00. Compute barely moved (220.7 → 178.8 ms, and ~46 ms of stage 05's "compute"
was elementwise arithmetic on tile padding); essentially all 176 ms of candidate 5 came off the
layout side. `MSDAOperation` is 167.9 of the 178.8 ms of compute.

## Where the baseline time was

Kept because it is the only per-region breakdown of the baseline, and it is what set the ordering of
the whole backlog. 131 ops, 655.6 ms kernel, 2416.5 ms gap.

| region | ops | kernel | gap |
|---|---:|---:|---:|
| SCA — deformable attention (`2484 × 30125`) | 72 | 522.0 ms | 0.3 ms |
| TSA — deformable attention (`10000 × 10000`) | 39 | 91.5 ms | 154.5 ms |
| SCA — rebatch / scatter-back (outside MSDA) | 5 | 40.0 ms | **1917.0 ms** |
| TSA — forward, outside MSDA | 3 | 0.2 ms | 267.7 ms |
| MSDA exit | 5 | 0.4 ms | 68.3 ms |
| FFN | 5 | 1.3 ms | 7.6 ms |
| rest | 2 | 0.2 ms | 1.1 ms |

Two independent problems, and they did not compete: **host round-trips** — a single 1.917 s stall,
62% of wall clock, from ~36 `to_torch`/`from_torch` calls on tensors that never had to leave host
memory (candidate 1, closed) — and **deformable-attention kernel time**, 613 ms of the 656 ms total,
none of it matmul (candidate 2, landed; then 3, 5, and what remains of 10/11/12).

## What was tried and rejected

[perf_reports/DEAD_ENDS.md](perf_reports/DEAD_ENDS.md) — measured, not in the tree, with the reason
each one lost. Entry 3 (a static `max_len` bound) was rejected partly on a DRAM ceiling that
[stage 07](perf_reports/07-sampling-grid-in-row-major.md) has since removed; its +129 ms cost argument
stands on its own and has not been re-priced.

## Report format

One file per landed change, `NN-slug.md`: source commit, the candidate it implements, kernel/ops
after, delta from the previous stage, PCC, what changed and why, and where the time went. Same
numbers, same harness, same signposts every time — so the deltas sum.

Only landed changes get a report. Re-measurements that revise a published figure are corrected in
place, in the report that published it.
