# moe-grouped-topk-accuracy

## Goal
Make the device MoE gate select the same experts as an fp32 torch reference. The score is the percentage
of golden experts the device misses: lower is better. Device kernel time may grow by at most 5% per case.

## What the code does
`ttnn.experimental.deepseek_prefill.moe_grouped_topk` is the device MoE router of the prefill models
(DeepSeek-V3, DeepSeek-V4 Flash/Pro non-hash layers, Kimi-K2.7, Kimi-K3, GLM-5.3). Per token row it takes
the gate logits `[tokens, n_experts]` and the expert correction bias (same shape, broadcast per expert) and:

1. upcasts both to fp32,
2. applies the score function (`sigmoid`, or `sqrt(softplus(x))` for DeepSeek-V4),
3. adds the bias,
4. selects experts by the biased score: with `n_groups > 1` (DeepSeek-V3: 8 groups) it sorts each group,
   sums the top `summed_experts_per_group` scores per group, keeps the top `topk_groups` groups and takes
   the top-k experts among them; with `n_groups == 1` it is a plain top-k over all experts (`blocks::topk`),
5. gathers the *unbiased* activated scores of the chosen experts, normalizes them to sum 1 and scales
   by `route_scale`.

Outputs: weights `[tokens, k]` bf16 (ROW_MAJOR in production for k <= 32), indices `[tokens, k]` uint16
TILE. Rows past `padding_config`'s real-token count get the sentinel index `n_experts`.

Work is split across cores by tile rows of tokens; each core runs the whole pipeline for its 32-token tile
rows. The sorts are the bitonic `topk_local_sort` / `topk_merge` / `topk_rebuild` LLKs on transposed
tiles, carrying uint32 index tiles that the writer generates.

## Where the error comes from (measured before the campaign)
The baseline misses 0.25 to 0.42% of golden experts. The inputs are tie-free bf16 logits, so every miss is a
near-tie between biased scores that the device resolves differently from fp32:
- **Sort key precision.** The factory sets `sort_keys_tf32` and the sort compares keys at TF32 precision.
  On a 384-expert sigmoid case, a golden that *truncates* the biased scores to TF32 (10-bit mantissa)
  reproduces the device's picks at recall 0.9992, against 0.9963 for the exact fp32 golden. So for sigmoid most
  misses are the sort comparing truncated keys. Look at how the topk LLKs load and compare values (TF32 /
  SrcA/SrcB paths, the rank-tag engine `GATE_TAG_BITS`) and at full-fp32 comparison alternatives.
- **Score function precision.** For sqrt(softplus) TF32 truncation explains less (the device's picks match neither fp32 nor TF32
  well), so softplus/sqrt error also matters there. The test's `biased_scores` debug output can be compared with
  `score_activation(logits) + bias` to measure it.
- The grouped path (DeepSeek-V3) also sums group scores and sorts groups, which adds more places to lose precision.

## Where the code is
All under `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk/`:
- `device/moe_grouped_topk_program_factory.cpp`: core split, circular buffers, `sort_keys_tf32`,
  unpack-to-dest modes, compile-time args, compute config (host side; changing it needs a rebuild)
- `device/kernels/compute/moe_grouped_topk.cpp`: the per-tile-row pipeline
- `device/kernels/compute/moe_gate_common_compute.hpp`: stage blocks (upcast, score func, add bias,
  sort/topk, group sums, normalize, scale). **Shared with `moe_hash_gate`.**
- `device/kernels/dataflow/reader_moe_grouped_topk.cpp`, `writer_moe_grouped_topk.cpp`,
  `moe_gate_common_dataflow.hpp`: input reads, index templates, gather of chosen scores, outputs
- `device/moe_grouped_topk_device_operation.cpp`: validation and output specs
Kernels are JIT-compiled at run time; host files need a rebuild.

The Blackhole LLK sources under `tt_metal/hw/ckernels/blackhole/` (topk sort LLKs, SFPU sigmoid/softplus/sqrt,
unpack/pack) are editable too. They are shared with other ops, so prefer a new function or file that only this op
calls. An in-place change must keep the interface and must not reduce accuracy.

## What is measured
`agent_orch/campaigns/moe-grouped-topk-accuracy/eval.py`:
1. Runs `tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/test_moe_grouped_topk_prefill_perf.py`
   under the Tracy perf adapter. There are 6 cases, one per production gate, each called as `TtMoEGatePrefill` calls it:
   640 tokens, bf16 logits in L1, broadcast bf16 bias in DRAM, a padding config, `stable_sort=True`,
   ROW_MAJOR weights. There are 13 calls per case (3 warmup, 10 timed), each on a fresh random input, and accuracy is pooled over
   all 8320 tokens.

   | case | experts | groups (top) | k | score func | miss % | time µs |
   |---|---|---|---|---|---|---|
   | deepseek-v3 | 256 | 8 (4) | 8 | sigmoid | 0.42 | 82.0 |
   | kimi-k2-7 | 384 | 1 | 8 | sigmoid | 0.37 | 76.9 |
   | kimi-k3 | 896 | 1 | 16 | sigmoid | 0.34 | 174.8 |
   | glm-5-3 | 256 | 1 | 8 | sigmoid | 0.35 | 54.0 |
   | dsv4-flash | 256 | 1 | 6 | sqrtsoftplus | 0.25 | 55.9 |
   | dsv4-pro | 384 | 1 | 6 | sqrtsoftplus | 0.34 | 80.4 |

2. Per case it sets `value = miss_pct + 0.001` (a floor that keeps ratios finite), records the device time as
   `duration_us`, and sets `slowdown = duration_us / perf_baseline.json[case]`. Gates: `slowdown <= 1.05`
   and `set_weights_max_abs <= 0.004` (the max weight error on rows whose expert set matches).
3. Runs the correctness suites `test_moe_grouped_topk.py`, `_stable`, `_ties`, `_state`,
   `_row_major_weights` and `test_moe_hash_gate.py` (252 tests, about 2.5 min). Any failure makes the attempt invalid.

`exact_rows` (whether the sorted order also matches) and `weights_pcc` are logged but not scored.

## Constraints
See the campaign rules. Precision may only go up. The op's contracts stay (stable tie order, padding
sentinels, layouts, fp32/bf16 inputs, grouped and single-group routing). `moe_hash_gate` must keep working. No
special-casing of the test's data.
