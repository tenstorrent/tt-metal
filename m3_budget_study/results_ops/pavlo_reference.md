# Pavlo's per-op efficiency reference (MiniMax-M3 prefill traffic sim)

Source: tt-metal branch `philei/m3-traffic-sim` @ 446896d, `models/demos/minimax_m3/traffic_sim/`
(`sim_core.js`, `calib_data.json`). Copies are in `tools/sim_core.js` and `tools/calib_data.json`.

## Conditions

* Zone profiles: one prefill chunk of 5120 tokens attending a 51,200-token cache. The harness capacity is 56,320.
  The profiles used a 1D fabric, bf4 experts and bf16 index_k (`M3_INDEX_CACHE_BF16=1`).
  Each zone time is the device-kernel time, averaged over the stage's devices.
* Meshes profiled: [2,4] (SP=2, TP=4, EP=8, 16 experts/chip), [4,2] (SP=4, TP=2, EP=8) and [8,4].
  [4,4] is the geometric midpoint of [2,4] and [8,4].
* Efficiency: `eff = roofline / (zone ms - latency floor)`, with a floor of 0.04 ms for CCL ops and 0.01 ms for the rest.
  It is clamped to [0.003, 1]. `misc` = layer zone total minus every mapped zone.
* Target (the `opEff = 1` "fixed kernels" point): 70% for matmul-class ops, 80% for DRAM- and link-class ops.
* Headroom = `target / min(eff, target)`. This is the formula of the artifact's "Per-op efficiency, [2,4] MoE/MSA layer" table.
* Share = zone ms / layer zone ms at the profile point. Pavlo's table has no share column; it is computed here from the same data.
* Pavlo's page only renders the [2,4] MoE/MSA layer. The [4,2] and dense rows below apply the same formula to the same calibration.
* Reproduce with `node tools/roofline_ops.js --check`. It gives model = zone for every op except `ag_idx` (see below).
  The sum of the ops equals `layerMs` to the last digit.

## [2,4] MoE/MSA layer (layer 3), layer total 18.575 ms

| op | zone ms | roofline ms | measured eff | target | headroom | share |
|---|---:|---:|---:|---:|---:|---:|
| norm_ag (x2) | 1.211 | 0.472 | 41.7% | 80% | x1.9 | 6.5% |
| qkv | 0.282 | 0.238 | 87.6% | 70% | x1.0 | 1.5% |
| idx_branch | 0.448 | 0.017 | 3.8% | 70% | x18.5 | 2.4% |
| misc | 1.110 | 0.123 | 11.2% | 80% | x7.2 | 6.0% |
| o_proj | 0.372 | 0.212 | 58.5% | 70% | x1.2 | 2.0% |
| attn_rs | 0.607 | 0.236 | 41.6% | 80% | x1.9 | 3.3% |
| shared | 1.169 | 0.474 | 42.0% | 70% | x1.7 | 6.3% |
| router | 0.207 | 0.013 | 6.7% | 70% | x10.4 | 1.1% |
| dispatch | 1.108 | 0.157 | 14.7% | 80% | x5.4 | 6.0% |
| experts | 3.518 | 1.568 | 44.7% | 70% | x1.6 | 18.9% |
| combine | 1.106 | 0.315 | 29.5% | 80% | x2.7 | 6.0% |
| moe_reduce | 3.210 | 0.359 | 11.3% | 80% | x7.1 | 17.3% |
| ag_kv | 0.235 | 0.077 | 39.3% | 80% | x2.0 | 1.3% |
| ag_idx | 0.091 | 0.072 | 100% (clamped) | 80% | x1.0 | 0.5% |
| indexer | 0.285 | 0.021 | 7.8% | 70% | x9.0 | 1.5% |
| sparse | 3.616 | 0.141 | 3.9% | 70% | x17.9 | 19.5% |

## [4,2] MoE/MSA layer (layer 3), layer total 16.864 ms

| op | zone ms | roofline ms | measured eff | target | headroom | share |
|---|---:|---:|---:|---:|---:|---:|
| norm_ag (x2) | 0.388 | 0.157 | 51.1% | 80% | x1.6 | 2.3% |
| qkv | 0.283 | 0.238 | 87.2% | 70% | x1.0 | 1.7% |
| idx_branch | 0.277 | 0.017 | 6.2% | 70% | x11.3 | 1.6% |
| misc | 1.030 | 0.123 | 12.0% | 80% | x6.6 | 6.1% |
| o_proj | 0.328 | 0.212 | 66.7% | 70% | x1.1 | 1.9% |
| attn_rs | 0.216 | 0.079 | 44.6% | 80% | x1.8 | 1.3% |
| shared | 0.642 | 0.317 | 52.7% | 70% | x1.3 | 3.8% |
| router | 0.143 | 0.007 | 5.0% | 70% | x14.0 | 0.8% |
| dispatch | 1.120 | 0.315 | 29.1% | 80% | x2.7 | 6.6% |
| experts | 3.547 | 1.568 | 44.3% | 70% | x1.6 | 21.0% |
| combine | 2.811 | 0.629 | 22.7% | 80% | x3.5 | 16.7% |
| moe_reduce | 1.184 | 0.202 | 17.6% | 80% | x4.5 | 7.0% |
| ag_kv | 0.899 | 0.230 | 26.8% | 80% | x3.0 | 5.3% |
| ag_idx | 0.144 | 0.108 | 100% (clamped) | 80% | x1.0 | 0.9% |
| indexer | 0.265 | 0.011 | 4.2% | 70% | x16.6 | 1.6% |
| sparse | 3.587 | 0.141 | 4.0% | 70% | x17.7 | 21.3% |

## Dense GQA layer (layer 0)

`ring` is `attn/ring_joint_sdpa`. Its roofline is the compute part `ring_c`; the capacity scan is smaller at a capacity of 56k.

| op | [2,4] zone ms | [2,4] eff | [2,4] share | [4,2] zone ms | [4,2] eff | [4,2] share | target |
|---|---:|---:|---:|---:|---:|---:|---:|
| norm_ag (x2) | 1.170 | 43.3% | 7.4% | 0.391 | 50.5% | 2.8% | 80% |
| qkv | 0.282 | 87.7% | 1.8% | 0.284 | 86.9% | 2.0% | 70% |
| misc | 0.951 | 13.1% | 6.0% | 0.846 | 14.7% | 6.0% | 80% |
| o_proj | 0.370 | 58.8% | 2.4% | 0.326 | 67.1% | 2.3% | 70% |
| attn_rs | 0.606 | 41.7% | 3.9% | 0.223 | 42.9% | 1.6% | 80% |
| dense_mlp | 1.986 | 61.1% | 12.6% | 1.576 | 67.2% | 11.2% | 70% |
| ring (ring_c) | 10.381 | 35.8% | 65.9% | 10.464 | 35.5% | 74.2% | 70% |
| layer total | 15.748 | | | 14.110 | | | |

## CAL.effs (as `SIM.calibrate(calib_data.json)` returns them)

The full-precision values are in `tools/cal_effs_calibrated.json`, which is also the template for a replacement file.
The hosted artifact (claude.ai/artifact/3Yresb3qbWJQpvpQMabaq6) embeds identical `effs`.

| mesh | kind | values |
|---|---|---|
| 2x4 | moe | norm_ag 0.4173, qkv 0.8756, idx_branch 0.0378, o_proj 0.5846, attn_rs 0.4158, shared 0.4201, router 0.0673, dispatch 0.1473, experts 0.4469, combine 0.2952, moe_reduce 0.1132, ag_kv 0.3934, ag_idx 1.0000, indexer 0.0781, sparse 0.0392, misc 0.1117, kv_a2a 0.1473 |
| 2x4 | dense | norm_ag 0.4328, qkv 0.8768, o_proj 0.5883, attn_rs 0.4166, dense_mlp 0.6111, misc 0.1305, ring_c 0.3576, ring_scan 0.0150, kv_a2a 0.1473 |
| 4x2 | moe | norm_ag 0.5113, qkv 0.8723, idx_branch 0.0620, o_proj 0.6666, attn_rs 0.4461, shared 0.5266, router 0.0499, dispatch 0.2913, experts 0.4431, combine 0.2271, moe_reduce 0.1761, ag_kv 0.2675, ag_idx 1.0000, indexer 0.0421, sparse 0.0395, misc 0.1205, kv_a2a 0.2913 |
| 4x2 | dense | norm_ag 0.5054, qkv 0.8689, o_proj 0.6713, attn_rs 0.4293, dense_mlp 0.6722, misc 0.1471, ring_c 0.3547, ring_scan 0.0150, kv_a2a 0.2913 |

`kv_a2a` is copied from `dispatch`. `ring_scan` 0.015 is a placeholder that the pipeline fit replaces.

Pipeline fit (`CAL.pipe`, branch version), fitted on the 16 x [2,4] runs A/B/C on the 2D fabric:

* `moeMult` = 1.0346 + 0.0241 x 5120/T, applied to the whole MoE layer;
* `ringC` = 0.3025 and `ringScan` = 0.0140. These are the dense ring-joint efficiencies the pipeline actually uses: [4,2] gets ringC x 0.3547/0.3576;
* `embed` = 1.54 ms;
* blocking send = 0.59 + 3.37 x T/1000 ms (17.9 ms at 5120, 7.5 ms at 2048);
* hop = 12.09 + 0.77 x T/1000 ms (16.0 / 13.7 ms).

## How the sim uses these (read before replacing them)

* With `opEff = 0` (today), `makePlan` uses `min(measured, target)`. So qkv (87.6%) runs at 70% and ag_idx at 80%.
  A replacement efficiency above its target has no effect.
* Changing `effs['2x4'].dense.ring_c` does not move a [2,4] pipeline. It only rescales [4,2] through the ratio.
  Use `run_goodput.js --pipe '{"ringC":..,"ringScan":..}'` to change the dense layer.
* `moeMult` was fitted against these effs. Keep it when the new effs come from the same kind of 1D-fabric zone profile.
  Use `--no-moe-mult` when the new profile already reflects the pipeline (2D fabric).
* `ag_idx` is clamped to 1.0. Its roofline (0.072 ms on [2,4]) is above the measured 0.091 ms minus the 0.04 ms floor.

## Published goodput and its reproduction (`tools/run_goodput.js`)

Goodput is in useful tok/s, at 4 galaxies with today's kernels (`opEff 0`), decode at 180 tok/s, over a 1800 s window.
The "full stack" is the g4_k0 study stack:

`pool arena host idxdedup batch var async idxbf8 msa srpt fused unaligned`

It runs with a 2M-token lane arena, chunk 2048 and the auto split `1,1,1,4x8,5x5`.
The branch's `results/study.json` holds the SLO-10 s grid values.
SLO 3 s is not in the study; its sweep and bisection are re-run at 3 s by the same rule.

| config | budget | p90 <= 10 s: study.json | p90 <= 10 s: run_goodput.js | p90 <= 3 s: run_goodput.js |
|---|---:|---:|---:|---:|
| 16x[4,2] | 8192 | 42,500 | 42,500 | 29,915 |
| 16x[4,2] | 4096 | 40,108 | 40,108 | 30,747 |
| 16x[2,4] | 8192 | 39,759 | 39,759 | 24,593 |
| 16x[2,4] | 4096 | 38,509 | 38,509 | 27,813 |

This matches the quoted 42.5k vs 39.8k at 8k and p90 <= 10 s. The quoted p90 <= 3 s ranges, 29.9-30.7k and 24.6-27.8k, are the spread over budgets 8192 and 4096.
Pavlo's material never names a "near-term" set. `run_goodput.js` defines `near` as the README's P0+P1 tiers:

`pool bounded host async batch idxdedup var`

It runs with 4 fixed 1M lanes. Edit `SETS` in `run_goodput.js` if a different set is meant.

Note: the hosted artifact and Pavlo's standalone repo (philei-tt/m3-agentx-prefill-lab) are newer than the branch. They add:

* 4x4-torus ring collectives;
* a kv_len-bounded dense gather (tt-metal #47539);
* a 2.75 ms dense fixed cost;
* re-fitted ringC 0.328 and block/hop.

Their study reports other numbers, for example 16x[4,2] with 4 lanes at budget 8192 = 44.4k. The quoted figures come from the branch version, which is the one copied here.
