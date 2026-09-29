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
### "Near-term" feature set

The branch has no "near-term" wording: none in the README, the study code, the page or the commit messages.
The only statement to match is that with near-term features, 16x[4,2] is 2-3% worse than 16x[2,4].
The candidates below were run with `run_goodput.js --features <keys> --budget 4096,8192 --slo 10,3`.
Sets with `arena` use a 2M-token arena; sets with `pool` and no `arena` use 4 fixed 1M lanes.
Cells are goodput in k tok/s as [2,4] / [4,2] (Δ = [4,2] vs [2,4]).

| set (`run_goodput.js` name) | 4k, 10 s | 8k, 10 s | 4k, 3 s | 8k, 3 s |
|---|---|---|---|---|
| **`near`** = P0+P1 − var: pool bounded host async batch idxdedup | 28.4 / 27.5 (**−3.2%**) | 29.3 / 28.2 (**−3.6%**) | 20.1 / 16.4 (−18%) | 15.4 / 8.1 (−47%) |
| P0+P1 − var − async: pool bounded host batch idxdedup | 25.1 / 23.9 (−4.6%) | 26.4 / 25.6 (−3.1%) | 14.9 / 8.4 (−43%) | 8.0 / 5.8 (−28%) |
| `near` + idxbf8 | 29.0 / 28.2 (−3.0%) | 30.4 / 28.6 (−5.7%) | 20.5 / 17.3 (−15%) | 15.4 / 8.7 (−43%) |
| `near` + idxbf8 + srpt | 29.0 / 28.5 (−1.8%) | 31.4 / 30.4 (−3.4%) | 20.6 / 19.6 (−4.7%) | 15.4 / 8.4 (−45%) |
| greedy prefix before var: pool arena host idxdedup batch async | 28.5 / 27.5 (−3.6%) | 29.2 / 28.7 (−1.4%) | 20.0 / 16.4 (−18%) | 15.2 / 7.9 (−48%) |
| `p0p1` = pool bounded host async batch idxdedup var | 32.7 / 32.0 (−2.1%) | 33.4 / 34.3 (+2.6%) | 25.0 / 24.6 (−1.6%) | 22.7 / 25.2 (+11%) |
| P0+P1 − async | 29.5 / 29.4 (−0.6%) | 30.4 / 30.8 (+1.2%) | 19.8 / 19.4 (−2.0%) | 15.0 / 17.5 (+16%) |
| P0+P1 − idxdedup | 25.9 / 29.6 (+14%) | 27.0 / 31.4 (+16%) | 20.2 / 23.7 (+17%) | 18.9 / 23.8 (+26%) |
| P0 only: pool bounded host | 19.2 / 21.4 (+12%) | 14.7 / 17.2 (+17%) | 8.0 / 8.1 (+1%) | 5.3 / 6.4 (+20%) |
| full − msa (async kept) | 37.0 / 36.1 (−2.4%) | 38.9 / 39.4 (+1.4%) | 26.0 / 26.0 (−0.1%) | 23.2 / 26.2 (+13%) |
| full − msa − async | 32.4 / 31.7 (−2.1%) | 34.9 / 35.5 (+1.5%) | 19.6 / 19.8 (+0.7%) | 15.1 / 19.6 (+30%) |
| full − async (msa kept) | 33.7 / 36.4 (+7.9%) | 36.1 / 38.5 (+6.6%) | 21.5 / 25.4 (+18%) | 15.2 / 23.0 (+52%) |
| `full` | 38.5 / 40.1 (+4.2%) | 39.8 / 42.5 (+6.9%) | 27.8 / 30.7 (+11%) | 24.6 / 29.9 (+22%) |

Match: `near` (P0+P1 without the variable chunk) is the closest. It puts [4,2] 3.2% and 3.6% behind at the two budgets at p90 ≤ 10 s.
Other sets that also put [4,2] behind at both budgets spread wider (−1.4% to −5.7%).
It is now the `near` set in `run_goodput.js`, and the old P0+P1 set is kept as `p0p1`.
At p90 ≤ 3 s, `near` puts [4,2] far behind (−18% / −47%), so the "2-3% worse" statement only matches the 10 s SLO.

What drives the [4,2] result:

* **idxdedup** removes [4,2]'s capacity edge: index_k replicated ×2 instead of ×4. Without it, [4,2] wins by 12-26% in every set.
* **msa** (the SP-local indexer) is what gives [4,2] its full-stack lead: full − msa is −2.4% / +1.4%, full is +4.2% / +6.9%.
  It drops the SP=4 prefix all-gathers (ag_kv is 0.90 ms on [4,2] vs 0.24 ms on [2,4]).
* **async** alone does not favour [4,2]: full − async still gives +7.9% / +6.6%.
* **var** is the next largest swing: without it, [4,2] trails at both budgets; with it, [4,2] leads at 8k.

Note: the hosted artifact and Pavlo's standalone repo (philei-tt/m3-agentx-prefill-lab) are newer than the branch. They add:

* 4x4-torus ring collectives;
* a kv_len-bounded dense gather (tt-metal #47539);
* a 2.75 ms dense fixed cost;
* re-fitted ringC 0.328 and block/hop.

Their study reports other numbers, for example 16x[4,2] with 4 lanes at budget 8192 = 44.4k. The quoted figures come from the branch version, which is the one copied here.
