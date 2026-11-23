# MB.9: the 5120 chunk's matmuls re-swept on the 8x4 (MM_TUNED160 off | on)

Model deepseek_v41_flash_d_p, galaxy_8x4, layers 0-20 (DECODER_SKIP=decode), 5120 chunk (160 rows a chip). Attempt 1
(2026-11-23).

## 1. Inventory: every matmul of a real 5120 chunk (agentic_cc, H = 138240), largest first

Device us a chunk: the op-mode real-time profile of the off side (generated/ds41_mm160/ops_single.json, slowest chip;
op-mode chunk total 71.85 ms). Calls and configs: the capture (tests/test_mm160_offon.py MM160_PARTS=capture,
generated/ds41_mm160/calls_mm160.json: 97 dense / grouped matmul calls a chunk, 12 distinct, 1399 KiB L1 free for CBs
at every call). Config source: tt/mm_configs (`GALAXY` = the FIX.15 11 x 10 table, `rule` = the generic rule).

| op / site | x · w | dtypes (x, w -> out), fidelity, fp32 acc | calls | device us / chunk | today's config (source) |
|---|---|---|---|---|---|
| bringup.moe_block.routed_experts_combine (routed experts gate_up + down, bfp4 LoFi, + combine) | 5120x5120 bf16 · 1x384 u32 | fork | 20 | 18607.9 | fork moe_block: blocking fixed at load (flat layout / planner); no call-time knob, not swept |
| bringup.moe_block.moe_head (router + shared expert, fused head) | 640x1280 bf16 · 2661120x32 bf16 | fork | 20 | 6433.5 | fork moe_block: as above, not swept |
| linear attn.wq_b | 160x1280 bfp8 · 1280x32768 bfp8 | BFLOAT8_B, BFLOAT8_B -> BFLOAT16, HiFi2, 1 | 20 | 3958.4 | 1d in0 bw8 ob5x10 sb1x2 (GALAXY) |
| linear attn.wo_b | 160x8192 bfp8 · 8192x5120 bfp8 | BFLOAT8_B, BFLOAT8_B -> BFLOAT16, HiFi2, 1 | 20 | 2836.4 | 1d in0 bw8 ob5x2 sb1x2 (GALAXY) |
| bringup.grouped_matmul attn.wo_a | 8x160x4096 bfp8 · 8x4096x1024 bfp8 | BFLOAT8_B, BFLOAT8_B -> BFLOAT8_B, HiFi2, 1 | 20 | 2664.1 | grouped_matmul bw4 depth4 rt1 (WO_A_GMM_*) |
| linear q_a | 160x5120 bfp8 · 5120x1792 bfp8 | BFLOAT8_B, BFLOAT8_B -> BFLOAT16, HiFi2, 1 | 20 | 950.0 | 1d in0 bw16 pn1 (GALAXY) |
| experimental.minimal_matmul engram_kv | 640x6144 bfp8 · 6144x6400 bfp8 | BFLOAT8_B, BFLOAT8_B -> BFLOAT16, HiFi2, 1 | 2 | 503.7 | minimal M4 K16 N8 sb1x4 (GALAXY) |
| linear compressor | 160x5120 bf16 · 5120x1024 bf16 | BFLOAT16, BFLOAT16 -> FLOAT32, HiFi2, 1 | 3 | 128.0 | 2d 11x5 bw16 (GALAXY) |
| matmul hc.fn | 160x20480 bf16 · 20480x32 fp32 | BFLOAT16, FLOAT32 -> FLOAT32, HiFi2, 1 | 1 | 87.9 | 2d 1x5 bw16 (rule: GALAXY's hc.fn key is 640x5120) |
| linear indexer.wq_b | 160x1280 bfp8 · 1280x4096 bfp8 | BFLOAT8_B, BFLOAT8_B -> BFLOAT16, HiFi2, 1 | 3 | 85.7 | 1d in0 bw8 (GALAXY) |
| linear indexer.weights_proj | 160x5120 bf16 · 5120x32 bf16 | BFLOAT16, BFLOAT16 -> BFLOAT16, HiFi2, 1 | 3 | 49.3 | 2d 1x5 bw16 (GALAXY) |
| linear compressor | 160x5120 bf16 · 5120x512 bf16 | BFLOAT16, BFLOAT16 -> FLOAT32, HiFi2, 1 | 1 | 28.1 | 2d 8x5 bw16 (GALAXY) |
| linear compressor | 320x512 bf16 · 512x128 bf16 | BFLOAT16, BFLOAT16 -> BFLOAT16, HiFi2, 1 | 3 | 12.9 | 2d 4x10 bw16 (GALAXY) |
| linear compressor | 640x512 bf16 · 512x128 bf16 | BFLOAT16, BFLOAT16 -> BFLOAT16, HiFi2, 1 | 1 | 6.7 | 2d 4x10 bw16 (GALAXY) |

Dense + grouped matmuls: 11.31 ms of the 71.85 ms chunk. The two moe_block fork rows (~25 ms) hold the routed / shared
expert and router matmuls inside fused programs whose K / N blocking comes from the load-time flat routed-expert
layout (MoEBlockConfig, FlatLayoutOptions): a call-time switch cannot change it (it needs a reload and its own A/B).
Other fork programs with internal matmul tiles (sparse_sdpa, ring_indexer_score_dsa, mhc_post / mhc_pre) are attention
/ score / residual kernels without a program-config knob; not part of this sweep.

## 2. Tuned table (MM_TUNED160=on: tt/mm_configs.GALAXY160 / GMM160)

Sweep: tests/perf/test_matmul_sweep.py -k test_sweep_card, 12 single-card processes in parallel (P23_SHARD=i/12,
P23_WIDE=1), resident runner stopped, ~14 min wall, over the capture of the real chunk; every candidate on the 11 x 10
matmul grid (FIX.15) and the model's L1 (filler: 1398 KiB for CBs); fidelity, dtypes, accumulation unchanged. Accepted
(OWNER rule 3): deterministic, every chip equal, error vs float64 <= today's x 1.01, within a bf16 step of auto, > 1 %
faster; never "minimal auto" (the card's 12 x 10). Confirmed on the whole mesh (test_mm160_offon confirm: every chip,
2 repeats a side bit-identical, 2 x 20 calls timed a side).

| site | shape | calls / chunk | today config | today us | tuned config | tuned us (sweep / mesh off -> on) | gain us / chunk (sweep) |
|---|---|---|---|---|---|---|---|
| attn.wq_b | 160x1280 bfp8 · 1280x32768 bfp8 | 20 | 1d bw8 ob5x10 | 194.1 | 1d in0 pm5 pn10 ob5x2 bw20 sb1x2 (PS.2's) | 175.6 / 195.9 -> 180.1 | 370 |
| attn.wo_a.gmm | 8x160x4096 bfp8 · 8x4096x1024 bfp8 | 20 | gmm bw4 depth4 rt1 | 131.2 | gmm bw2 depth4 rt1 | 127.1 / 131.5 -> 128.3 | 82 |
| attn.wo_b | 160x8192 bfp8 · 8192x5120 bfp8 | 20 | 1d bw8 sb1x2 | 139.2 | 1d in0 pm5 pn2 bw8 sb1x1 | 136.5 / 139.5 -> 137.0 | 54 |
| q_a | 160x5120 bfp8 · 5120x1792 bfp8 | 20 | 1d bw16 | 46.8 | 1d in0 pm5 pn1 bw32 sb1x1 | 44.2 / 47.2 -> 44.3 | 51 |
| engram_kv | 640x6144 bfp8 · 6144x6400 bfp8 | 2 | minimal M4 K16 N8 | 250.4 | minimal M2 K24 N8 sb1x4 | 236.6 / 250.7 -> 237.4 | 28 |
| hc.fn | 160x20480 bf16 · 20480x32 fp32 | 1 | 2d 1x5 bw16 (rule) | 88.1 | 2d 1x5 pm1 pn1 bw64 sb1x1 | 74.5 / 87.6 -> 74.1 | 14 |
| indexer.weights_proj | 160x5120 bf16 · 5120x32 bf16 | 3 | 2d 1x5 bw16 | 16.1 | 2d 1x5 pm1 pn1 bw40 sb1x1 | 13.4 / 16.1 -> 13.6 | 8 |
| indexer.wq_b | 160x1280 bfp8 · 1280x4096 bfp8 | 3 | 1d bw8 | 26.1 | 1d in0 pm5 pn2 bw20 sb1x2 | 24.7 / 28.2 -> 27.0 | 4 |
| compressor | 160x5120 bf16 · 5120x512 bf16 | 1 | 2d 8x5 bw16 | 27.5 | 2d 8x5 pm1 pn2 bw32 sb1x2 | 26.0 / 28.0 -> 27.8 | 2 |

No faster config: compressor 160x5120x1024 (x3), 320x512x128 (x3), 640x512x128 (x1). Sum over the swept calls 11.11 ->
10.50 ms a chunk (-0.61 ms, sweep times x calls). Every winner was bit-identical to today's config on the sweep's
random inputs (DS41-390: bfp8 x bfp8 sums are exact on random data); on the real chunk the off and on outputs differ
from layer 13 (section 4).

## 3. Device ms off vs on, one load (test_mm160_offon, H = 138240, slowest chip, median of 2 profiled runs a side)

| section | off | on | on - off |
|---|---|---|---|
| attn_hc | 0.256 | 0.242 | -0.014 |
| attn_norm | 0.017 | 0.017 | +0.000 |
| q_a | 1.712 | 1.654 | -0.058 |
| kv | 0.362 | 0.362 | +0.000 |
| attention | 21.734 | 21.451 | -0.283 |
| attn_residual | 2.858 | 3.100 | +0.242 |
| experts | 31.440 | 31.470 | +0.030 |
| ffn_residual | 4.635 | 4.744 | +0.109 |
| engram_rows | 0.905 | 0.865 | -0.040 |
| engram_kv | 0.725 | 0.701 | -0.024 |
| engram_add | 0.567 | 0.568 | +0.001 |
| compressor | 1.605 | 1.636 | +0.031 |
| indexer | 2.104 | 2.054 | -0.050 |
| indexer_topk | 0.819 | 0.819 | +0.000 |
| **chunk (device)** | **66.489** | **66.060** | **-0.43** |

Runs: off [66.402, 66.489], on [66.060, 65.461]. The orchestrator measure's run of the same test (same code): off [65.937, 66.377], on [65.799, 65.852] (attention 21.730 -> 21.186, q_a 1.713 -> 1.653, attn_residual 2.962 -> 2.823). The sections the configs touch move as the sweep says (attention -0.28
for wq_b / wo_a / wo_b, q_a -0.06, hc / indexer / engram_kv -0.09); attn_residual / ffn_residual (+0.35) contain no
re-tuned matmul and move run to run (mhc_post's CCL waits).

## 4. Accuracy and determinism (switch on)

- test_ladder, last rung (orchestrator measure --part last, DS41_MM_TUNED160=on, 40 layers): PASS, all 104 limits.
  err_final_hidden 0.1362 / 0.2422; top5 1.0 / 0.9943; err_layer_L00 0.0189 / 0.0551, L13 0.1131 / 0.2003, L19 0.1376 /
  0.2643, L38 0.1210 / 0.2203, L39 0.7554 / 1.5332 (DS41-188: L39 swings 0.70-0.91 on rounding-only changes).
- determinism (orchestrator measure --part determinism, fresh process): determinism_max_abs_diff 0.0, state 0.0.
- mm160_repeat_max_abs_diff_single = 0 (slot 1 vs slot 2, the same history: 20 layer outputs, 29 served records
  bitwise); mm160_repeat_max_abs_diff_confirm = 0 (every winner, every chip, 2 repeats a side).
- Off vs on (diagnostic, rel L2 over 64 sampled rows a layer): bit-identical on layers 0-12; 7.1e-4 at L13, 1.3e-2 at
  L14-15, 2.9e-2 L16, 3.5e-2 L17, 3.8e-2 L18-19: a real-input summation-order change (DS41-390) carried by later layers;
  both sides are within the ladder's limits (on side: above).

- OV-58: the gate ladder's err_* metrics are bit-equal to MB.8's. Probe tests/test_mm160_ladder_hits.py (the ladder's own
  run_ladder on rung last, the gate's accuracy load, off then on): tuned-config hits off 0, on 220 (wq_b / wo_b / q_a /
  wo_a 40 each, engram_kv 2, indexer 8 + 8, hc.fn 1, compressor 1); all 42 layer outputs byte-identical off vs on. The
  switch takes effect; on that chunk the winners give the same bytes (7 of 9 keep the K order by construction; engram_kv
  K24 and wo_a bw2 regroup L1-accumulated fp32 sums that are exact on that data, DS41-390).
- Proof metric (test_mm160_offon, gate run 2): mm160_tuned_hits_single_off 0, _on 90 (wq_b / wo_b / q_a / wo_a gmm 20
  each, engram_kv 2, indexer.wq_b 3, weights_proj 3, hc.fn 1, compressor 1); device ms off 66.68 / on 65.85.

## 5. Ladder (5120), switch on vs off

The orchestrator's A/B after this step runs the gate's 5120 ladder (test_skip_positions, both sides) and the plain
profile; its table is the off vs on comparison. This attempt measured the on side: `gate MB.9 --check` PASS (twice; run 2 after the OV-58 proof metric: 51200 70.6k,
139264 68.5k, 262144 68.0k, 524288 65.8k, 786432 61.6k, 1024000 62.1k), run 1 steady tok/s 51200 70.1k,
131072 68.8k, 139264 69.1k, 262144 67.6k, 524288 65.5k, 786432 62.7k, 1024000 62.7k (MB.8's gate, other process: 69.1 /
67.9 / 66.1 / 66.7 / 65.6 / 63.0 / 61.6k); `orchestrator measure --task MB.9` rc 0 on every run (determinism, last,
test_mm160_offon, test_skip_positions, test_no_rebuild).

## 6. Not done / notes

- moe_block fork matmuls (~25 ms a chunk, the largest matmul work) not swept: no call-time knob (section 1).
- DS41-306 corrected to the code (GALAXY shipped wq_b bw8 ob5x10; PS.2's bw20 ob5x2 is now GALAXY160's wq_b).
- The 8192 path (PB.MM256's GALAXY256) is on the batching branch only; no 5120 key collides with another chunk's.
