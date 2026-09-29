# Goodput in Pavlo's sim: 16x[2,4] vs 16x[4,2], and goodput per op fix (REPORT_ops.md items 3 and 6)

Sim: `tools/sim_core.js` (branch `philei/m3-traffic-sim` @ 446896d), driven by `tools/run_goodput.js`. Setup: 4 galaxies,
today's kernels (`opEff 0`), decode 180 tok/s, 1800 s window, Pavlo's traffic (`/data/philei/m3_traffic_sim/data`).
Goodput is useful tok/s at the p90 TTFT SLO, found with a concurrency sweep plus cliff bisection (lib/pool.js rule).
Feature sets are the `SETS` in `run_goodput.js`:

* `near` = pool bounded host async batch idxdedup. This is P0+P1 without the variable chunk. It is the closest match to
  Pavlo's "near-term: [4,2] 2-3% worse" (see `pavlo_reference.md`).
* `full` = the g4_k0 study stack, which adds arena, var, idxbf8, msa (the SP-local indexer), srpt, fused and unaligned.
* `p0p1` = `near` + var. It is shown as supplementary data.

No device was used. Everything here is CPU simulation over the zone profiles already in `per_op.csv` and `per_op_4x2.csv`.

## Verdict

**Keep [2,4].** The decision rule fails on (b) and (c):

| rule | status | evidence |
|---|---|---|
| (a) KV PCC on (4,2) passes, or KV-head sharding scoped at <= 1-2 weeks | **pass** | PCC >= 0.9985 vs (2,4) on layers 0-6. Every layer is within 1e-4 of (2,4)'s PCC vs golden. The sharding change is scoped at ~4-6.5 d (`kv_4x2_status.md`). |
| (b) [4,2] >= +5% at the SLO with near-term features | **fail** | Near, our effs, native gather: -0.5% / -0.1% at p90 <= 10 s (W 4096 / 8192), and -10% / -26% at p90 <= 3 s. With the measured per-head copies it is -2.4% / -3.3% and -13% / -58%. |
| (b') ... or with the full stack, if the SP-local indexer and async handoff are committed | not applicable | Full stack: +3.2-3.9% at 4096 / 10 s (below 5%), +8.1-8.7% at 8192 / 10 s, and +13-19% at 3 s. But `msa` and `async` are sim features only. Neither is implemented or committed for M3, so the full-stack column does not count. |
| (c) runner and KV migration handle TP=2 without new blockers | **fail (blocker to scope)** | The 4x(4,2) runner is not ready. It lacks a topology yaml, a chained 4-mesh [4,2] mesh graph descriptor, per-tray `TT_VISIBLE_DEVICES` and a full 60-layer [4,2] weight cache. `kv_chunk_table.py:96` asserts `num_kv_heads == cols`. |

What would reopen it: the SP-local indexer and async handoff get committed, and the runner/migration work (~1-2 d
plus the runner pieces) gets scoped. Under those conditions, [4,2] leads by about 9% at 8k / 10 s and 13-19% at 3 s in the
full stack. The variable chunk alone (`p0p1`, native gather) gives +9.6% / +7.3% at W=8192 but -0.6% at W=4096.

## 1. CAL.effs['4x2'] from per_op_4x2.csv

**Method.** This is the method that made the [2,4] set (`tools/make_cal_effs.py`, which reproduces
`cal_effs_ours_2x4_w*.json` to 1e-5):

* zone ms is the chip mean;
* condition: W single request at h = 139,264; moe = mean of sparse layers 3-6, dense = layer 1;
* eff = `roof x wave / (mean - floor)`, clamped to [0.003, 1];
* the floor is 0.04 ms for CCL ops (x2 for norm_ag) and 0.01 ms otherwise;
* roofline from `roofline_ops.js --mesh 4x2 --segments W:139264 --idx bf8`;
* wave factor `waveFactor(W)/waveFactor(5120)`: 1.0417 at 4096 and 0.9896 at 8192, the same as [2,4];
* misc = layer minus the mapped ops; kv_a2a = dispatch; ring_scan = 0.015 placeholder.

**Difference from [2,4].** The (4,2) runs are prose only, because only prose was profiled on (4,2). The [2,4] set pools prose and code.
`cal_effs_ours_2x4_prose_w*.json` is the like-for-like prose-only [2,4] set, used as a sensitivity check.

**Variants.**

* (a) `cal_effs_ours_4x2_w{4096,8192}.json`: as measured. `ag_kv` includes the harness per-head slice and concat copies.
* (b) `cal_effs_ours_4x2_native_w{4096,8192}.json`: `ag_kv` = `ag_kv_native_est`, with the copies removed from the layer total
  as well. Only `ag_kv` changes (bold below).

**Check.** `roofline_ops.js --mesh 4x2 --detail --effs <file> --segments W:139264 --idx bf8` gives layerMs equal to the measured
chip-mean layers:

| layer | [4,2] W4096 | [4,2] native W4096 | [2,4] W4096 | [4,2] W8192 | [4,2] native W8192 | [2,4] W8192 |
|---|---:|---:|---:|---:|---:|---:|
| sparse | 14.091 ms | 13.569 ms | 13.810 ms | 23.939 ms | 23.413 ms | 25.719 ms |
| dense | 29.407 ms | | 26.241 ms | 50.509 ms | | 53.821 ms |

| kind | op | [4,2] W4096 | [4,2] native W4096 | [4,2] W8192 | [4,2] native W8192 | Pavlo [4,2] | ours [2,4] W4096 / W8192 | target |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| moe | norm_ag | 54.2% | 54.2% | 47.3% | 47.3% | 51.1% | 40.9% / 40.1% | 80% |
| moe | qkv | 72.0% | 72.0% | 80.3% | 80.3% | 87.2% | 79.6% / 87.3% | 70% |
| moe | idx_branch | 5.7% | 5.7% | 6.6% | 6.6% | 6.2% | 3.4% / 3.6% | 70% |
| moe | o_proj | 56.5% | 56.5% | 66.2% | 66.2% | 66.7% | 56.5% / 58.8% | 70% |
| moe | attn_rs | 45.4% | 45.4% | 43.0% | 43.0% | 44.6% | 42.7% / 43.0% | 80% |
| moe | shared | 49.3% | 49.3% | 53.7% | 53.7% | 52.7% | 41.9% / 43.5% | 70% |
| moe | router | 5.5% | 5.5% | 6.0% | 6.0% | 5.0% | 6.0% / 6.2% | 70% |
| moe | dispatch | 27.3% | 27.3% | 27.1% | 27.1% | 29.1% | 16.4% / 16.1% | 80% |
| moe | experts | 78.1% | 78.1% | 74.8% | 74.8% | 44.3% | 76.9% / 74.5% | 70% |
| moe | combine | 33.2% | 33.2% | 34.1% | 34.1% | 22.7% | 38.0% / 39.3% | 80% |
| moe | moe_reduce | 16.1% | 16.1% | 16.1% | 16.1% | 17.6% | 17.0% / 17.4% | 80% |
| moe | ag_kv | 30.1% | **41.1%** | 28.0% | **37.1%** | 26.7% | 45.4% / 33.7% | 80% |
| moe | ag_idx | 46.8% | 46.8% | 46.8% | 46.8% | 100.0% | 54.1% / 53.9% | 80% |
| moe | indexer | 2.1% | 2.1% | 1.9% | 1.9% | 4.2% | 4.1% / 3.7% | 70% |
| moe | sparse | 4.1% | 4.1% | 3.9% | 3.9% | 3.9% | 4.0% / 3.6% | 70% |
| moe | misc | 11.1% | 11.1% | 14.3% | 14.3% | 12.0% | 11.2% / 11.4% | 80% |
| dense | norm_ag | 54.2% | 54.2% | 47.4% | 47.4% | 50.5% | 41.3% / 39.5% | 80% |
| dense | qkv | 71.9% | 71.9% | 80.5% | 80.5% | 86.9% | 79.8% / 87.3% | 70% |
| dense | o_proj | 56.8% | 56.8% | 67.1% | 67.1% | 67.1% | 57.0% / 59.7% | 70% |
| dense | attn_rs | 27.1% | 27.1% | 30.3% | 30.3% | 42.9% | 34.1% / 26.3% | 80% |
| dense | dense_mlp | 58.5% | 58.5% | 65.7% | 65.7% | 67.2% | 59.9% / 61.7% | 70% |
| dense | misc | 14.2% | 14.2% | 17.7% | 17.7% | 14.7% | 13.4% / 13.6% | 80% |
| dense | ring_c | 31.2% | 31.2% | 35.0% | 35.0% | 35.5% | 37.4% / 35.0% | 70% |

**Against Pavlo's [4,2]:**

* **experts** is 75-78% against 44%. It is capped at 70% on both meshes, so both run at target.
* **combine** is 33-34% against 23%.
* **ag_idx** is 47% against a clamped 100%, and **indexer** is about 2% against 4.2%. These are depth effects at 139k
  against his 51k, the same as on [2,4].
* **qkv** is 72% at W=4096, still above target at both W.
* **The dense ring at W=4096** is 31.2% against 35.5%. At 1024 rows per chip the (4,2) ring is 20% slower than the (2,4).
  At W=8192 (2048 rows per chip) the two are equal.
* **ag_kv** is 28-30% with the copies and 37-41% native, against his 26.7%. The native gather still moves 3x the [2,4]
  bytes, and the roofline carries that factor.

### CAL.pipe for the dense ring

[2,4] pipelines take the dense ring-joint efficiency from `CAL.pipe.ringC`. [4,2] pipelines take it from
`CAL.pipe.ringC x effs['4x2'].dense.ring_c / effs['2x4'].dense.ring_c`. The ring_c values in the effs files therefore only
set the ratio.

**Our pipe value.** We have no deep-context pipeline measurement to fit ringC against. The first P1-D/P1-E runner results
(`pipeline_overheads.txt`, `dense_scan_check.txt`) are 4x(2,4), one layer per stage, W=4096, at h=0 and 16k. There the ring
is a small part of the dense stage, so they cannot pin ringC. We therefore keep Pavlo's pipeline/zone correction and scale
it by our zone value:

`pipe.ringC = 0.30248 (his fit) x our [2,4] zone ring_c / 0.35759 (his [2,4] zone ring_c)`

`ringScan` stays at his fit, 0.014013.

* **ringScan barely matters here.** P1-E finds a 1M-slot dense stage within +1-20% (about +1 ms) of a 64k-slot one, where
  the whole-capacity scan model predicts +94 ms. `near` has `bounded`, so its scan is kv_len-bounded. `full` uses
  request-sized arena lanes.
* **Pavlo's block/hop fit is not used by any of these sets.** P1-D measures a blocking send of about 0.8-0.9 ms at W=4096,
  against his fit of 14.5 ms. But `near`, `full` and `p0p1` all include `async`, which replaces the block/hop fit with a
  link-rate transfer.

| W | pipe.ringC ([2,4]) | implied [4,2] | Pavlo: [2,4] / [4,2] |
|---:|---:|---:|---:|
| 4096 | 0.316492 | 0.2636 | 0.3025 / 0.3001 |
| 8192 | 0.296068 | 0.2959 | 0.3025 / 0.3001 |

This is passed as `--pipe '{"ringC":...}'` in every "ours" run. The row "ours native, Pavlo's pipe.ringC" in the sensitivity
table below shows the effect of keeping his value instead.

`CAL.pipe.moeMult` is kept in every run. It was fitted on 1D-fabric zone profiles like ours, but only on [2,4] pipelines,
and the sim applies it to [4,2] as well.

## 2. 16x[2,4] vs 16x[4,2]

The effs files match the budget: the W=4096 files at budget 4096 and the W=8192 files at 8192. The [2,4] column is the same
in both "ours" rows.

| calibration | features | W | SLO p90 | 16x[2,4] | 16x[4,2] | [4,2]/[2,4] |
|---|---|---:|---:|---:|---:|---:|
| Pavlo's | near | 4096 | 10 s | 28.4k | 27.5k | 0.968 (-3.2%) |
| Pavlo's | near | 4096 | 3 s | 20.1k | 16.4k | 0.819 (-18.1%) |
| Pavlo's | near | 8192 | 10 s | 29.3k | 28.2k | 0.964 (-3.6%) |
| Pavlo's | near | 8192 | 3 s | 15.4k | 8.1k | 0.527 (-47.3%) |
| Pavlo's | full | 4096 | 10 s | 38.5k | 40.1k | 1.042 (+4.2%) |
| Pavlo's | full | 4096 | 3 s | 27.8k | 30.7k | 1.105 (+10.5%) |
| Pavlo's | full | 8192 | 10 s | 39.8k | 42.5k | 1.069 (+6.9%) |
| Pavlo's | full | 8192 | 3 s | 24.6k | 29.9k | 1.216 (+21.6%) |
| ours, [4,2] with copies | near | 4096 | 10 s | 29.7k | 28.9k | 0.976 (-2.4%) |
| ours, [4,2] with copies | near | 4096 | 3 s | 24.0k | 20.8k | 0.867 (-13.3%) |
| ours, [4,2] with copies | near | 8192 | 10 s | 30.4k | 29.4k | 0.967 (-3.3%) |
| ours, [4,2] with copies | near | 8192 | 3 s | 20.8k | 8.8k | 0.423 (-57.7%) |
| ours, [4,2] with copies | full | 4096 | 10 s | 40.6k | 41.9k | 1.032 (+3.2%) |
| ours, [4,2] with copies | full | 4096 | 3 s | 31.2k | 37.0k | 1.184 (+18.4%) |
| ours, [4,2] with copies | full | 8192 | 10 s | 42.0k | 45.4k | 1.081 (+8.1%) |
| ours, [4,2] with copies | full | 8192 | 3 s | 28.9k | 32.7k | 1.129 (+12.9%) |
| ours, [4,2] native gather | near | 4096 | 10 s | 29.7k | 29.5k | 0.995 (-0.5%) |
| ours, [4,2] native gather | near | 4096 | 3 s | 24.0k | 21.5k | 0.897 (-10.3%) |
| ours, [4,2] native gather | near | 8192 | 10 s | 30.4k | 30.4k | 0.999 (-0.1%) |
| ours, [4,2] native gather | near | 8192 | 3 s | 20.8k | 15.4k | 0.739 (-26.1%) |
| ours, [4,2] native gather | full | 4096 | 10 s | 40.6k | 42.2k | 1.039 (+3.9%) |
| ours, [4,2] native gather | full | 4096 | 3 s | 31.2k | 36.0k | 1.152 (+15.2%) |
| ours, [4,2] native gather | full | 8192 | 10 s | 42.0k | 45.6k | 1.087 (+8.7%) |
| ours, [4,2] native gather | full | 8192 | 3 s | 28.9k | 34.5k | 1.193 (+19.3%) |

**Supplementary: `p0p1` (near + variable chunk)**

| calibration | W | [4,2]/[2,4] at 10 s | at 3 s |
|---|---:|---:|---:|
| Pavlo's | 4096 / 8192 | -2.1% / +2.6% | -1.6% / +11.0% |
| ours, with copies | 4096 / 8192 | -4.1% / +3.0% | -3.2% / +1.7% |
| ours, native | 4096 / 8192 | -0.6% / +9.6% | -0.6% / +7.3% |

**Sensitivity**

| calibration | near 10 s (4096 / 8192) | near 3 s | full 10 s | full 3 s |
|---|---|---|---|---|
| ours native (main row) | -0.5% / -0.1% | -10.3% / -26.1% | +3.9% / +8.7% | +15.2% / +19.3% |
| ours native, Pavlo's pipe.ringC | -0.4% / -0.1% | -9.9% / -26.1% | +3.2% / +8.9% | +8.8% / +8.7% |
| ours, [2,4] prose only, [4,2] with copies | -2.6% / -2.9% | -10.0% / -55.4% | +4.3% / +10.8% | +15.0% / +16.6% |
| ours, [2,4] prose only, [4,2] native | -0.6% / +0.3% | -6.9% / -22.1% | +5.0% / +11.0% | +16.0% / +22.8% |

**Reading it**

* **Our effs move both meshes the same way.** Near-term at p90 <= 10 s, [4,2] goes from -3.2% / -3.6% (Pavlo) to -2.4% / -3.3%
  (ours, with copies), and to -0.5% / -0.1% with a native multi-head gather. It never reaches +5%.
  * A prose-only [2,4] (like for like with the (4,2) profiles) moves [4,2] by 0.1-0.4 points at near / 10 s.
  * It adds about 1-2 points to [4,2]'s full-stack lead.
* **The native gather is worth about 2-3 points at 10 s and much more at 3 s** (8.8k to 15.4k, near, W=8192). The copies
  cost is the harness workaround (`kv_4x2_status.md` step b, 2-3 d). It is a precondition for any [4,2] number, not a
  [4,2] advantage.
* **The 3 s SLO with near features is a cliff regime.** The unloaded p90 TTFT is already 2.2-2.3 s on [4,2] against
  2.0-2.2 s on [2,4] (W=8192, lowest concurrencies). So a small per-request latency difference decides the 3 s goodput.
  * Treat 3 s cells as about ±5% resolution. Example: in the full stack at W=4096 / 3 s, our faster [2,4] dense ring
    (ringC 0.316 against 0.302) gives 31.2k against 32.9k with Pavlo's pipe. That is a non-monotonic response.
  * The 10 s cells are smooth.
* **Full stack.** [4,2] wins by 3-4% at 4096 and 8-9% at 8192 (10 s), and 13-19% at 3 s. The SP-local indexer (`msa`) is
  what removes [4,2]'s SP=4 ag_kv penalty, as in Pavlo's analysis.

## 3. Goodput per fix, 16x[2,4], our calibration

**Setup.** Base: `cal_effs_ours_2x4_w{W}.json` with our `pipe.ringC`. Each fix merges one extra effs file (`goodput/fix/*.effs.json`)
and reruns 16x[2,4]. Δ is against the base at the same features, W and SLO.

**Fix values:**

* **target**: 70% for matmul-class ops, 80% for DRAM and link ops.
* **min-chip**: `roof / (min-chip ms - floor)`, from sparse layers 3-6, prose+code, h = 139,264. This is the eff at the fastest
  chip's time, with the cross-chip wait on the hot expert's column removed:
  * combine: 78.8% / 87.1%, capped at the 80% target;
  * moe_reduce: 36.3% / 35.9%;
  * dispatch: 18.6% / 18.2%.

| fix (16x[2,4], ours) | near 4096 10 s | near 4096 3 s | near 8192 10 s | near 8192 3 s | full 4096 10 s | full 4096 3 s | full 8192 10 s | full 8192 3 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base goodput | 29.7k | 24.0k | 30.4k | 20.8k | 40.6k | 31.2k | 42.0k | 28.9k |
| 1. sparse: 3.6-4.0% -> 70% (target) | **+17.3%** | +24.5% | **+16.7%** | +36.9% | +14.8% | +30.0% | +14.9% | +26.9% |
| 2. moe_reduce: 17% -> 80% (target) | **+7.8%** | +11.4% | **+5.1%** | +12.8% | +5.1% | +21.0% | +7.0% | +6.2% |
| 2'. moe_reduce -> min-chip 36% (imbalance wait removed only) | +5.3% | +8.2% | +2.6% | +6.1% | +2.7% | +13.7% | +5.1% | +4.9% |
| 3. combine: 38-39% -> min-chip 79% / 80% | **+3.1%** | +4.1% | **+1.3%** | +2.2% | +0.9% | +2.4% | +1.1% | +3.6% |
| 4. experts -> 70% (target) | **+0.0%** | +0.0% | **+0.0%** | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% |
| 4'. experts: roofline imbalance 1.2 -> 1.0 | -0.2% | +0.2% | +0.1% | +0.4% | +0.3% | +5.8% | +0.7% | +1.2% |
| 5. dispatch: 16% -> 80% (target) | **+3.7%** | +6.1% | **+2.1%** | +3.9% | +1.8% | +8.0% | +2.9% | +3.7% |
| 5'. dispatch -> min-chip 18.6% | +2.7% | -0.2% | -0.1% | +0.4% | +0.9% | +3.8% | -0.1% | +0.0% |
| all 5 (rows 1, 2, 3, 4, 5) | **+38.0%** | +45.0% | **+37.1%** | +65.0% | +27.5% | +49.8% | +29.7% | +57.2% |

**Reading it**

* **sparse is the lever.** It gives +15-17% at 10 s on its own, 2-3x the next op, and the ranking in `pavlo_table_ours.md`
  holds.
* **moe_reduce comes second.** It gives +5-8% at target. Half to two thirds of that is recoverable just by removing the
  imbalance wait (row 2'), because the kernel itself is at 36% on the fastest chip.
* **combine and dispatch are 1-4% each.** For combine that is the whole wait: its fastest chip is already at target.
* **experts: no effect (clipping).** Our measured experts eff (77% / 74%) is already above its 70% target. With `opEff = 0` the
  sim uses `min(eff, target)`, so raising it does nothing. The worst-chip experts time (1.5-2.1x the mean) is load
  imbalance, and the sim models imbalance only as the fixed 1.2 in the roofline. Setting that to 1.0 (row 4') is worth
  0-1% at 10 s.
* **Other ops with measured eff above target have no effect** if raised: qkv at W=8192 and experts.
* **The "all 5" gain (+37-38% near, +28-30% full, at 10 s) is larger than the sum of the singles.** Once sparse is fixed, the
  MoE chain becomes a larger share of the critical stage.
* **The 3 s deltas are larger and noisier** (see the cliff note in section 2).
* **Caveat on dispatch.** The dispatch rows leave `kv_a2a` (the var-layout KV all-to-all, a copy of dispatch's eff) unchanged.

## 4. Commands

```bash
cd ~/tt-metal/m3_budget_study/results_ops
export NODE=$(ls ~/.vscode-server/cli/servers/*/server/node | head -1)
# effs (per_op_4x2.csv / per_op.csv -> JSON)
for W in 4096 8192; do
  python3 tools/make_cal_effs.py --mesh 4x2 --W $W --out tools/cal_effs_ours_4x2_w$W.json
  python3 tools/make_cal_effs.py --mesh 4x2 --W $W --native --out tools/cal_effs_ours_4x2_native_w$W.json
  python3 tools/make_cal_effs.py --mesh 2x4 --W $W --inputs prose --out tools/cal_effs_ours_2x4_prose_w$W.json
done
# layer check
$NODE tools/roofline_ops.js --mesh 4x2 --detail --effs tools/cal_effs_ours_4x2_w4096.json --segments 4096:139264 --idx bf8 --layer both
# all sim runs (a few minutes on 16 workers) -> goodput/*.json, goodput/fix/*.json; then the tables
bash tools/goodput_study.sh all
python3 tools/goodput_summary.py
```

`goodput_study.sh` runs these, for W in 4096 and 8192, each with `--budget $W --slo 10,3 --features near,full,p0p1`:

```bash
# Pavlo's calibration
run_goodput.js --topo 2x4,4x2
# ours, [4,2] with copies
run_goodput.js --topo 2x4,4x2 --effs tools/cal_effs_ours_2x4_w$W.json,tools/cal_effs_ours_4x2_w$W.json --pipe '{"ringC":<0.316492|0.296068>}'
# ours, [4,2] native gather
run_goodput.js --topo 2x4,4x2 --effs tools/cal_effs_ours_2x4_w$W.json,tools/cal_effs_ours_4x2_native_w$W.json --pipe '{"ringC":...}'
# per fix, 2x4 only
run_goodput.js --topo 2x4 --features near,full --effs tools/cal_effs_ours_2x4_w$W.json,goodput/fix/<fix>.w$W.effs.json --pipe '{"ringC":...}'
# experts imbalance row
run_goodput.js --topo 2x4 --features near,full --effs tools/cal_effs_ours_2x4_w$W.json --pipe '{"ringC":...}' --set expertImb=1
```

`run_goodput.js --effs` now takes a comma list, merged left to right.

## 5. Caveats

* **Single-stage profiles on both meshes.** The [4,2] costs come from one (4,2) stage (layers 0-6, rows 0-3). That stage
  has the harness KV patches (`tools/profile_4x2.py`), 1D fabric and v1 dispatch/combine. There is no [4,2] pipeline
  measurement, and `moeMult` and the block/hop fits are [2,4]-pipeline fits reused for [4,2].
* **Profile inputs and depth.** The (4,2) profiles are prose only, and the effs are taken at h = 139k. The sim applies them at
  every depth, but the depth ops (indexer, ag_kv, ag_idx) do not scale with depth the way their rooflines do
  (`msa_summary.md`).
* **The native gather is an estimate:** `ag_kv` minus the measured slice and concat copies. It is not a measured native kernel.
* **Features are sim-only.** The features that decide the comparison (`msa`, `async`, `var`) exist only as sim features.
* **Output files.** The `goodput/*.json` files hold every concurrency point, with the plan and cfg of each run.
