# zai-org/GLM-5.3-Flash bring-up: breadcrumbs

Append-only log, one section per task attempt: what was done, decisions and why, gotchas, the re-run command, the verdict.

## R.2 reference (attempt 1 of the reference role; the previous run failed at the stock HF loader)

What was done:
- `reference/hf/`: the transformers glm5_next modeling + configuration files (commit 0291458166a6, 5.17 line),
  vendored with relative imports made absolute; five helpers 5.12.1 lacks come from `reference/hf/_compat.py`
  (kernel-hub / accelerate decorators as no-ops, `create_recurrent_attention_mask` = local 2D mask, vision stubs).
  The model code itself is unchanged.
- `reference/hf_oracle.py` (hook `hf_model`): text model only (`Glm5NextTextModel` + lm_head; no vision tower, no
  MTP), built on meta, weights assigned. Routed experts are `PackedExperts`: FP8 read from the safetensors (via the
  index) and dequantized per use, same forward as HF's experts. `num_layers=None` gives bf16 (intake sanity),
  otherwise fp32. fp32 kept for conv1d / dt_bias / A_log / e_score_correction_bias as `_keep_in_fp32_modules_strict`.
  Name mapping: `hc_{attn,ffn}_{fn,base,scale}` -> `{attn,ffn}_hc.*`, `{q,k,v}_conv1d` concatenated into `conv1d`,
  `f_a/f_b/dt_bias/A_log` -> `forget_gate.*`. `generate` is a greedy loop over `GlmCache` (see gotchas).
- `reference/glm_ref.py` (hook `reference`): standalone CPU reference, every block through `run_block`. Graph:
  attn_hc, attn_collapse, attn_norm, [KDA: attention] or [DSA: q_a, indexer, attention], attn_residual, ffn_hc,
  ffn_collapse, ffn_norm, [dense: mlp] or [MoE: router, experts, shared_expert, moe_add], ffn_residual (`out`).
  KDA runs HF's chunk-64 delta rule from the carried state (intra-chunk terms built 8 sub-chunks at a time). The
  indexer caches pooled keys and scores 128-query blocks against them. MLA uses absorbed kv_b and gathers the selected
  latent rows (topk [S, 2051] int32: 512 pools x 4 tokens + 3 tail, -1 = none). Experts are grouped per expert and
  padded to 32 rows (known issue on M-dependent sgemm).
- `reference/weights.py`: index-based loader, FP8 128x128 block dequant (fp32 product, exact).

Decisions:
- Residual-stream boundaries (`in`, `h_mid`, `out`) are `[S * 4, H]`, token-major (row 4t + n = stream n of token t):
  check_hf compares against HF's layer output flattened to `[-1, H]`, and `[S, 4, H]` fails its maxabs print.
- mHC coefficients are one boundary `[S, 24]` fp32 = pre 4 | post 4 | comb 16 (row-major, after Sinkhorn), shared
  by the collapse and residual steps.
- State: DSA `kv_latent [len, 512]`, `index_key [len // 4, 128]` (complete pools only); KDA `kda_recurrent
  [64, 128, 128]` fp32, `kda_conv [3, 24576]` (last 3 pre-conv q|k|v rows); a chunk at start 0 resets the KDA state.
  The indexer asserts chunk starts are multiples of 4. One-shot == chunked also needs starts that are multiples of 64
  (KDA sub-chunk grid); all ladder chunks are.
- Experts dequantized to fp32 at load when a subset has <= 2 MoE layers (58 GB for layers 3-4), else per use.

Gotchas:
- transformers 5.12.1 has linear-attention and indexed cache layers, but with other semantics than the 5.17 code
  expects (`update_conv_state` returns the stored state instead of [state | input]; `conv_states` is a tensor, the
  code indexes `[0]`). A 5.12 DynamicCache would silently corrupt KDA; `forward` therefore defaults to no cache and
  `generate` uses `GlmCache`.
- config.json names DSA layers `deepseek_sparse_attention`; the modeling code's mask table only has
  `indexed_attention` / `linear_attention` (KeyError). The oracle renames layer_types in the config object.
- The whole-model sanity streams ~300 GB of FP8 experts from disk; it took 10 min on this host (page cache warm).

Results (local runs): check_hf --seq 4096: L00-L04 pcc 1.0000000 (maxabs <= 5.7e-6), logits pcc 1.0000000, top-1
match 1.0000. check_hf_sanity: revision ok, smoke '</think>Paris<|user|>' ok, text_top1_acc 0.964.
Chunked (64) vs one-shot at 128 tokens: max abs 0.

Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.intake.check_hf_sanity && PYTHONPATH=$PWD python -m
models.demos.common.bringup.reference.check_hf --seq 4096` (about 14 min).
Also run locally (R.3's check, not this gate): check_reference --seq 4096 --chunk 2048: hidden pcc 1.000000000,
state min pcc 1.000000000, graph replay max abs 0 for kda_dense (L0, 10 steps), dsa_moe (L3, 15), kda_moe (L4, 13);
4 min. Gate run: 13 min 23 s, all R.2 metrics pass.

## PL.1 plan (attempt 1)

What was done:
- `plan.yaml`: placements for every checkpoint tensor (76108, whole-model map from `bringup_trim.json`): vision, MTP
  layer 45 and layers 5-44 skipped; fp8 block scales folded at load. State: MLA latent [56320, 512] and pooled keys
  (seq_stride 4) replicated for layer 3; KDA recurrent (32 heads x 128 x 128 fp32) and conv tail per chip for 0-2, 4.
  Activations 7.0 GiB (8192-token chunk, 4-stream residual replicated). Gate: 13.59 of 27.20 GiB per chip.
- `components.yaml`: all 41 steps (kda_dense 10, dsa_moe 15, kda_moe 13, model 3), NATIVE or COMPOSED, none CPU or
  OPGEN.
- `plan.md`: per-layer tables, per-chip total, collectives, departures. `tasks.yaml` unchanged.

Decisions:
- KDA is TP=2 (axis 1) x SP=2 (axis 0), because ttKDA requires distinct SP / TP axes. Fallback: SP=1 on 1x2 row
  submeshes.
- DSA attention: replicated weights, query rows split 4 ways, all 64 heads per chip (sparse_sdpa needs H % 32 == 0).
  Caches replicated, every chip computes latent and pooled keys for all S rows, one all_gather after o_proj.
- Indexer is composed, not OPGEN: key pooling is reshape + 4 slices + an elementwise softmax. indexer_score_dsa runs
  unmasked (chunk_start_idx = kv_len - 1) plus a constant pool-causal mask on the chunk's own pools. Sentinel
  compaction: for q >= 2047 the pool part has no sentinels, so [pools | tail | pad] ends in sentinels; for q < 2047
  the row is the constant "all tokens 0..q".
- Experts EP=4 via the MiMo 2x2 dispatch (axis 0, groups = columns, 72 per chip), high_precision unified kernel with
  ClampedSiluGlu. Dense MLP and shared expert TP=4 + all_reduce(cluster_axis=None). Router fp32 (MiMo TtRouter).
- mHC: DeepSeek mhc_split_sinkhorn matches GLM's Sinkhorn order and pre/post forms (checked against
  reference/mhc/mhc_reference.py); RMS eps 1e-5 is a config field.

Gotchas / to verify at implement: ttnn.topk at width 288; unified_routed_expert_moe and the dispatch tables with 72
local / 288 global experts; ttKDA's SP segment order under a chunk-aligned actual_start; sparse_sdpa scale must be
passed explicitly as 1/16 (its default is 512^-0.5); KDA recurrence and o_proj default to HiFi2 in the Kimi config
(set HiFi4); core grids from the device (p300c 11x10). The gate builds the CPU reference for layers 0, 3 and 4 (14 s
here, page cache warm).

Result: plan_fits 1, unplaced_tensors 0, plan_errors 0, component_errors 0, ledger_errors 0; plan_approved 0 (waits
for the person's approval).

Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.plan.check_plan`.

## C.kda_dense.attn_hc test (attempt 1)

Reviewed the rendered component test for attn_hc (layer 0, s4096 chunk 1). Kept the gated PCC and added per-part checks,
because whole-output PCC passes several real bugs (measured on the golden with a host script: comb transposed 0.9958,
softmax over the wrong axis 0.9987, 10 Sinkhorn iterations 0.9958, swapped pre/post scales 0.9958, a zeroed row 0.9997).
- Per part (pre / post / comb): rel L2 <= 0.03 / 0.03 / 0.02 and max abs <= 0.15 / 0.2 / 0.1.
- Comb column sums within 0.01 of 1; pre in (0, 1], post in [0, 2], comb >= 0.
- Headroom: bf16, bfp8 or tf32 projection operands give rel <= 0.002. A pessimistic random mix error at PCC 0.9989
  (the TT fp32-matmul ceiling from deepseek_v3_d_p test_mhc.py) gives pre rel 0.025, comb max abs 0.061.
- Not caught (small effect): 19 iterations, hc_eps 0, pre without +1e-6.
- The golden attn_hc is bf16-rounded values stored as fp32. Comb row sums are only 0.92..1.08 even after 20 iterations,
  so don't assert row sums.
- Implement: the output must be [2048, 24] (or reshapeable), comb row-major (comb[i, j] at column 8 + 4i + j).
Results: reference passes (PCC 0.999999, part rel ~0.0013, column sums 1.00000). Stub fails (PCC 0). The gate in
device mode fails with NotImplementedError until the implement step adds the module.
Re-run: `PYTHONPATH=$PWD BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_kda_dense_attn_hc.py`

## C.kda_dense.attn_hc implement (attempt 1)

What was done:
- `tt/mhc.py:TtHcWeights` (+ `build_hc(mesh, loader, cfg, layer, "attn"|"ffn")`): replicated, no CCL. Input
  `[1, 1, S, 4H]` (streams packed along the last dim, the same memory as the reference's token-major `[S * 4, H]`),
  bf16 -> `ttnn.typecast` fp32 -> DeepSeek `tt_mhc._project` (fp32 matmul HiFi4 + fp32 acc, RMS rsqrt applied after
  the linear, eps 1e-5) -> `mhc_split_sinkhorn` (consts from `tt_mhc.build_consts`, n 4, 20 iters, eps 1e-6) ->
  `ttnn.concat` [pre | post | comb] -> `[1, 1, S, 24]` fp32. Weights and consts uploaded once at load.
- `tt/common.py`: `hifi4_config`, `replicate` / `replicated_to_host` (load time and harness boundary only).
- `hooks.py`: `device_component` handles `attn_hc` and `ffn_hc` (same module, other weights); `DEVICE_STEPS`
  (kda_dense: attn_hc) and a `HybridDeviceModel` (CPU reference + DEVICE_STEPS, host in / out per step) as
  `device_model` until assemble.

Decisions: the kernel's comb output is already row-major (entry (i, j) at column 4i + j), as DeepSeek's op test
reshapes it `[T, n, n]` against the row-major reference; no reorder needed. The harness uploads the residual as bf16
(the planned device residual dtype).

Result: pcc_attn_hc_L00 0.999999; rel L2 pre 0.0014 / post 0.0013 / comb 0.0016; max abs <= 4.9e-3; comb column
sums [0.9974, 1.0009].

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_kda_dense_attn_hc.py`

## S.kda_dense.01 test (attempt 1)

Reviewed the rendered swap test (attn_hc on device, layer 0). The gated pcc_swap_out is kept. Asserted extra checks added,
because a CPU sensitivity run (attn_hc output perturbed, rest CPU) showed that block-out PCC passes real bugs: comb
transposed 0.9985, comb rows normalized instead of columns 0.9985, post x1.05 0.9991, pre x1.05 0.99986, last row zeroed
0.99993.
- Block out: rel L2 <= 0.01 (those bugs give 0.070 / 0.070 / 0.045 / 0.017 / 0.012) and per-row norm ratio in [0.97, 1.03].
- attn_hc's own output vs golden (its input is the golden `in`): the component test's per-part rel L2 and max abs, comb
  column sums, and pre/post ranges.
Results: reference passes (out PCC 0.999999, rel 0.0017). Stub fails (PCC 0). Device passes (out PCC 0.999998,
rel 0.0023, row ratio [0.9966, 1.0029]; attn_hc part rel <= 0.0016).
Gotcha: the run_safe_pytest precompile pass prints the whole metric set once with stub values (rel 1.0) before the real
pass. Ignore that block.
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_swap_kda_dense_01_attn_hc.py`
(prefix with `BRINGUP_IMPL=reference` or `BRINGUP_IMPL=stub` to check the other two modes).

## C.kda_dense.attn_collapse test (attempt 1)

Reviewed the rendered component test for attn_collapse (layer 0, s4096 chunk 1). Kept the gated PCC and added checks,
measured on the golden with a CPU-only host script:
- At layer 0 the four streams of `in` are identical (embedding copied), so pre-column or stream order is invisible,
  and PCC passes a dropped last stream (0.9998), output x1.02 (0.999998), and 32 zeroed rows (0.9934).
- Added: rel L2 <= 0.01 and per-token norm ratio in [0.985, 1.015] (those bugs: rel 0.021 / 0.020 / 0.115).
- Added: the same module (the step has no weights) on layer 1's golden `in` / `attn_hc` / `attn_in`, where the
  streams differ; pre reversed gives rel 1.27, pre 0/1 swapped 0.29, stream-major rows 1.28.
- Headroom: bf16 products accumulated in bf16 give rel 0.0036, ratio [0.994, 1.004].
Results: reference passes (PCC 0.999998; rel L2 0.0021 at L00, 0.0026 at L01). Stub fails (PCC 0). Device mode fails
with NotImplementedError until the implement step adds the module.
Implement: the device_component for attn_collapse is called with layer 0's ctx and fed layer 1's inputs too; the
input `in` is token-major [S * 4, H] (row 4s + n), the output is [S, H] (or reshapeable).
Re-run: `PYTHONPATH=$PWD BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_kda_dense_attn_collapse.py`

## C.kda_dense.attn_collapse implement (attempt 1)

What was done:
- `tt/collapse.py:TtHcCollapse` (+ `build_collapse(cfg)`): replicated, no CCL, no weights. `x [1, 1, S, 4H]` (streams
  packed along the last dim = token-major `[S * 4, H]`), `hc [1, 1, S, K]` fp32 (pre = columns 0..3) -> DeepSeek
  `tt_mhc._streams` / `_cols` -> each stream `ttnn.typecast` to fp32 -> `_mix` (multiply + 3 x addcmul, column
  broadcast) in fp32 -> one `ttnn.typecast` to bf16 -> `[1, 1, S, H]`.
- `hooks.py`: `_collapse_host_fn` (harness boundary: x uploaded bf16, hc fp32, chip 0 read back) for `attn_collapse`
  and `ffn_collapse` (same module); `DEVICE_STEPS["kda_dense"]` now has `attn_collapse`.
Decisions: fp32 accumulate with a single bf16 round, matching `glm_ref.hc_collapse` (the component notes).
Result: pcc_attn_collapse_L00 0.999997; L00 rel L2 0.00238, ratio [0.9964, 1.0027]; L01 (distinct streams) rel L2
0.00298, ratio [0.9960, 1.0036].
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_kda_dense_attn_collapse.py`

## S.kda_dense.02 test (attempt 1)

Reviewed the rendered swap test (attn_hc + attn_collapse on device, layer 0). Kept the gated pcc_swap_out. A CPU-only
sensitivity run (attn_collapse output perturbed, rest CPU; `/tmp` script, not kept) showed block out is weak on collapse
bugs because attn_norm follows it: x1.02 passes PCC and rel L2 (0.0072), last stream dropped passes PCC (rel 0.029).
Added asserted checks (informational metrics):
- block out rel L2 <= 0.01, per-row norm ratio [0.97, 1.03]; attn_hc's per-part checks (both as swap 01);
- attn_collapse vs golden attn_in and vs the CPU collapse of its actual inputs (golden in, swapped attn_hc):
  rel L2 <= 0.01, per-token ratio [0.985, 1.015];
- the same collapse module on layer 1's golden (distinct streams), same limits.
Results: reference passes (out rel 0.0017). Stub fails (PCC 0, every check). Device passes: out PCC 0.999997,
rel 0.0026, ratio [0.9951, 1.0044]; collapse vs golden 0.0016, vs CPU same input 0.0017, L01 0.0030.
Note: with the stub, `vs_cpu_same_input` rel is 0 (the CPU collapse of a zero attn_hc is zero); its ratio check fails it.
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_swap_kda_dense_02_attn_collapse.py`
(prefix `BRINGUP_IMPL=reference` / `BRINGUP_IMPL=stub` for the other modes).

## C.kda_dense.attn_norm test (attempt 1)

Reviewed the rendered component test for attn_norm (layer 0, s4096 chunk 1, [2048, 4096] bf16). Kept the gated PCC and
added asserted checks (informational metrics): finite, rel L2 <= 0.01, per-token norm ratio [0.98, 1.02], worst
per-token rel L2 <= 0.03. Tighter than the Gemma/MiMo template (0.03, [0.97, 1.03]) because of what the CPU mutation
run on this golden showed (host script, not kept):
- Layer-0 input rows are tiny (row RMS 0.0023..0.0155), so mean(x^2) is about the size of eps 1e-5 and eps matters:
  eps 1.2e-5 scores PCC 0.99986 / rel 0.027 (passes 0.03), eps 8e-6 rel 0.031.
- LayerNorm-style mean subtraction: PCC 0.99992, rel 0.0149, worst row 0.056. x1.02: rel 0.020.
- Caught by PCC already: sum instead of mean (0.9877). No weight / 1 + w: PCC 0.998, rel ~9.
- Headroom: fp32 reference rel 0.0023; squares accumulated in bf16 rel 0.0056, ratio [0.9865, 1.0149], worst row 0.015.
Results: reference passes (PCC 0.999997, rel 0.0023, ratio [0.9998, 1.0002], worst row 0.0029). Stub fails (PCC 0).
Device mode fails with NotImplementedError until the implement step adds the module.
Implement: keep eps exactly 1e-5 and the reduction of squares in fp32 (fp32 dest acc); bf16 accumulation is close to
the ratio limit.
Re-run: `PYTHONPATH=$PWD BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_kda_dense_attn_norm.py`

## C.kda_dense.attn_norm implement (attempt 1)

- `tt/rms_norm.py:TtRMSNorm` (from `mimo_v2_6_d_p/tt/rms_norm.py`): `ttnn.bringup.rms_norm`, replicated [1, 1, S, 4096]
  bf16, plain w (bf16 TILE gamma [1, 1, 1, H]), eps `cfg.rms_norm_eps` (1e-5), HiFi4 + fp32 dest. `build_norm(mesh,
  loader, cfg, layer, name)` takes any `<name>.weight` of the layer, so post_attention_layernorm can reuse it.
  `GLM_NORM_IMPL=native` selects `ttnn.rms_norm` for comparison.
- hooks: `_NORM_STEPS = {"attn_norm": "input_layernorm"}`, `_norm_host_fn` (bf16 upload, chip-0 readback at the harness
  boundary); `attn_norm` added to `DEVICE_STEPS["kda_dense"]`.
- Gate: PCC 0.999996, rel L2 0.0028, row norm ratio [0.9993, 1.0006], worst row 0.0037 (fp32 CPU reference 0.0023).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_kda_dense_attn_norm.py`

## S.kda_dense.03 test (attempt 1)

Reviewed the rendered swap test (attn_hc + attn_collapse + attn_norm on device, layer 0). Rewrote it from swap 02's
test: kept the gated pcc_swap_out and every swap-02 check (block out rel L2 <= 0.01 / ratio [0.97, 1.03], attn_hc
per-part checks, collapse vs golden / vs CPU same input / layer 1). Added attn_norm checks at the component test's
limits (rel L2 <= 0.01, per-token ratio [0.98, 1.02], worst row rel <= 0.03), both vs golden attn_norm and vs the CPU
norm of the device collapse output it actually got. No new sensitivity run: the reason is in the attn_norm
component review (eps / mean-subtraction bugs hide under rel 0.03) and the known issue that the block out misses norm
bugs at layer 0.
Gotcha: the stock test failed on the device at once (bf16 device output into the fp32 CPU KDA attention matmul). The
test now casts each override's floating output to fp32 (`_f32`).
Results: reference passes (out 0.999999, norm rel 0.0017). Stub fails (PCC 0, every check). Device passes: out PCC
0.999997, rel 0.0026, ratio [0.9954, 1.0047]; attn_norm vs golden rel 0.0028 / [0.9991, 1.0005] / worst row 0.0033,
vs CPU same input 0.0017.
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_swap_kda_dense_03_attn_norm.py`
(prefix `BRINGUP_IMPL=reference` / `BRINGUP_IMPL=stub` for the other modes).

## C.kda_dense.attention test (attempt 1)

Reviewed the rendered component test for the KDA attention (layer 0, s4096 chunk 1: start 2048, [2048, 4096] bf16,
prefix state from the golden snapshot at 2048). Kept the gated PCC (metric `pcc_attention_L00`). A CPU mutation run on
this golden (`/tmp` scripts, not kept) showed that PCC and whole-chunk rel L2 miss the state bugs that matter for a
TP=2 / SP=2 ttKDA:
- zeroed recurrent prefix PCC 0.99935 / rel 0.036; zeroed conv tail 0.99992 / 0.012; transposed recurrent state 0.9982;
  second SP half started from the prefix state 0.99987 / 0.016, or from zero 0.99941; conv halo at the SP split
  zeroed 0.99994 / 0.011; o_norm dropped 0.99942 (ratio 0.003); x1.02 PCC 1.0 / rel 0.020.
- These errors sit in the first rows of the chunk or of an SP segment: the worst 128-row block rel L2 is 0.043..0.15 and
  the worst row 0.38..0.95. Headroom: fp32 reference 0.0020 / 0.0024; bf16 everywhere 0.0047 / 0.0063; +1% element
  noise 0.011 / 0.015; +3% noise 0.030 / 0.039.
- Caught by PCC already: q scale, gate bound, dt_bias, A_log exp, beta, gate activation, o_norm weight, conv silu, TP
  head halves swapped.
Added asserted checks (informational metrics): finite, rel L2 <= 0.02, per-token norm ratio [0.98, 1.02], every
128-row block rel L2 <= 0.03 (prints all 16 blocks), worst per-token rel L2 <= 0.1. Post-chunk state vs the golden
snapshot at start + chunk: recurrent rel <= 0.03 and worst head <= 0.05, conv rel <= 0.02. Always checked in reference
mode. In device mode it is checked only when the module sets `dctx.extra["state_out"] = {"kda_recurrent": [64, 128,
128], "kda_conv": [3, 24576]}` (reference layout, torch, read back at the harness boundary); otherwise the test prints
a note and the ladder's state metrics check the state.
Results: reference passes (PCC 0.999998, rel 0.0020, blocks 0.0020, worst row 0.0024; state rec 0.0008 / head 0.0020,
conv 0.0019). Stub fails (PCC 0). Device mode fails with NotImplementedError until the implement step adds the module.
Implement notes: o_norm eps must be exactly 1e-5. The core output RMS is far below sqrt(eps), so the gated RMSNorm acts
as a scale of about 316 and does not normalize core-output error away. A device error above about 1.5% of the output
fails rel 0.02. Exposing `state_out` is recommended: the SP carry is where a 2x2 ttKDA is most likely to break.
Re-run: `PYTHONPATH=$PWD BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_kda_dense_attention.py`
(`BRINGUP_IMPL=stub` for the stub; no prefix for the device gate).

## C.kda_dense.attention implement (attempt 1)

What was done:
- `tt/kda_attention.py:TtKdaAttention` (+ `build_kda_attention(mesh, loader, cfg, layer, max_seq)`), built from DeepSeek's
  ttKDA. x [1, 1, S, 4096] replicated -> `ttnn.mesh_partition(dim -2, cluster_axis 0)` -> [1, S/2, 4096] -> ttKDA
  (SP 2 on axis 0, TP 2 on axis 1, heads 32c..32c+31 on column c) -> [1, S/2, 2048] -> `all_gather` dim -1 on axis 1,
  then dim -2 on axis 0 -> [1, 1, S, 4096] replicated bf16.
- Every matmul-like program runs at HiFi4: recurrence affine prefix and scan, and output projection (ttKDA defaults
  are HiFi2). The gated-RMS output is fp32. The grouped-scan group count comes from the device grid: the largest g
  dividing the local chunk count with 32 * g <= grid cores, which gives g = 2 for local T = 1024 / 2560 / 4096.
  One ttKDA per chunk length, sharing the KDAWeights.
- actual_start: a replicated uint32 ROW_MAJOR [max_seq / 64 + 1, 1] table of 64-aligned starts, built at load.
  Each chunk takes a `ttnn.slice` of it (a [1, 1] device scalar).
- State: `self.state` and `self.zero_state` are allocated once; a chunk at start 0 reads `zero_state`. The new carries
  are `ttnn.copy`'d into `self.state` (kimi_k3/kda_state.py pattern). `load_state` / `state_torch` convert at the
  harness boundary: recurrent [64, 128, 128] is sharded on dim 1 over axis 1. The reference conv tail [3, q|k|v] is
  reordered to the per-TP-rank [q_c|k_c|v_c].
- hooks: `_KdaHostFn`. If ctx.extra has `state_prefix` it is loaded and `state_out` is set; otherwise the module's
  carried state continues. `attention` is added to `DEVICE_STEPS["kda_dense"]`. In the hybrid model `_RefState`
  routes load_prefix / to_torch of those layers to the device module.
Checked: SP order. chronology.hpp `derive` gives first_rank = (start / (S/2)) % 2 = 0 and no split for every start that
is a multiple of S, so row 0 takes the first half and row 1 the second: the contiguous halves the plan feeds. The SP
fallback was not needed (every block rel L2 is 0.0070-0.0072, no bump at row 1024).
Precision findings (known_issues Proposed):
- First run: output fine (PCC 0.99998, rel 0.0072), state worst head 0.0528 > 0.05, biased low (head 12 scale 0.95),
  on the fast-decay channels.
- The device q/k/v/g/beta match the CPU (rel 0.004 / 0.0018 / 0.0004). Of the prepared terms only k_dec_t is off
  (4.4% on head 12). Cause: the TF32 FPU subtraction `G - G_last/2` in prepare_chunk_recurrence.
- Fix: `_PreciseDecayRecurrence` recomputes k_dec_t (suffix-sum matmul on bf16 g, fp32 exp, fp32 l2-normalized k).
  `GLM_KDA_DECAY=kernel` keeps the kernel's own term. Also the gate in fp32 (`_GlmKDA._compute_gates`); on its own it
  changed nothing (worst head 0.0538).
- Probed exact on device (fp32): transpose, permute, subtract, multiply, typecast->bf16 (RNE). An fp32 matmul operand
  rounds to TF32 (max rel 9.7e-4, unbiased); a bf16 operand is exact.
Result (gate): pcc_attention_L00 0.999985, rel L2 0.0071, row ratio [0.9902, 0.9982], worst block 0.0072, worst row
0.0136; state recurrent rel 0.0134 / worst head 0.0238, conv 0.0017.
Also ran, as a scratch test now deleted: S = 5120 and 8192 at start 0 and start S (random x). They build and give
finite output and state. Warm times 0.08 s / 0.13 s including host readback.
Left for later: output per-token ratio stays low ([0.990, 0.998], scale 0.9954), probably `intra` (scale 0.9957) from the
same kernel subtraction; a fork of prepare_chunk_recurrence would fix k_dec_t and intra at the source.
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_kda_dense_attention.py`

## S.kda_dense.04 test (attempt 1)

Reviewed the rendered swap test (attn_hc + attn_collapse + attn_norm + attention on device, layer 0). Rewrote it from
swap 03's test. Kept the gated pcc_swap_out, every swap-03 check and the `_f32` cast. Added the attention checks at the
component test's limits (rel L2 <= 0.02, per-token ratio [0.98, 1.02], every 128-row block <= 0.03, worst row <= 0.1),
run twice: vs the golden attn_out, and vs the CPU KDA of the device attn_norm, on a fresh reference ctx with the same
golden prefix state. Added the post-chunk KDA state vs the golden snapshot at start + chunk (recurrent 0.03 / worst head
0.05, conv 0.02). On the device it reads `dctx.extra["state_out"]` (set by `_KdaHostFn` because device_ctx carries
`state_prefix`) and fails if that is missing. In reference mode it reads the CPU state.
Sensitivity (CPU host script, not kept; only the attention perturbed): block out rel follows the attention's rel
(x0.99 0.0096, x1.02 0.0185, zeroed prefix state 0.045, zeroed attention 1.60). So the block-out limits stay rel <= 0.01
and ratio [0.97, 1.03].
Results: reference passes (out 0.999999, rel 0.0017; the recurrent state matches the golden exactly because the fp32
chain from `in` repeats the golden run; conv 0.0017, which is bf16 rounding). Stub fails (PCC 0, every check). Device
passes: out PCC 0.999984, rel 0.0062, ratio [0.9827, 1.0078]; attention vs golden 0.0073 / [0.9895, 0.9978] / block
0.0074 / row 0.014, vs CPU same input 0.0068; state recurrent 0.0135 / head 0.024, conv 0.0020.
Next step: the device attention's low scale (0.9954) accounts for most of the block-out rel (0.0062 of 0.01) and sets
the per-token minimum (0.983 of 0.97).
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_swap_kda_dense_04_attention.py`
(prefix `BRINGUP_IMPL=reference` / `BRINGUP_IMPL=stub` for the other modes).

## C.kda_dense.attn_residual test (attempt 1)

Reviewed the rendered component test for `h_mid = post * attn_out + comb^T @ in` ([S * 4, H], token-major). Rewrote
it on the attn_collapse test's pattern: the gated PCC plus asserted checks, run on layer 0 and again (same weightless
module) on layer 1's golden. At layer 0 the streams are identical, so the comb cannot be seen there.
Sensitivity (CPU host script on the goldens, not kept), layer 0 PCC / rel L2 (layer 1 rel): comb not transposed
0.9987 / 0.064 (0.26); post x1.02 0.9998 / 0.023 (0.015); comb x1.05 0.9987 / 0.057 (0.035); last token zeroed 0.99993 /
0.012; identity comb 0.999996 / 0.0030 (layer 1 0.50). Device-like noise: bf16 products and partial sums rel 0.0048,
ratio [0.995, 1.005], per-stream <= 0.0065, worst row 0.014. A bfp8 output gives rel 0.0082, so keep h_mid bf16.
Checks per layer: rel L2 <= 0.01, per-row norm ratio [0.985, 1.015], per-stream rel <= 0.015, worst row rel <= 0.05.
Each term is also checked on its own (out minus the exact other term from the golden inputs, vs post * attn_out and
vs comb^T @ in): coefficient [0.98, 1.02] and rel <= 0.02.
Results: reference passes (PCC 0.999995; L0 rel 0.0032, worst row 0.0104; L1 rel 0.0027). Stub fails (PCC 0). Device
mode fails with NotImplementedError until the implement step adds the module.
Implement notes: the module is called twice, with device_ctx(0) and device_ctx(1) and the matching golden inputs, so
it must not cache anything per layer. comb is row-major 4x4 at attn_hc[:, 8:24], and out stream m uses column m
(sum_n comb[n, m] in[n]).
Re-run: `PYTHONPATH=$PWD BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_kda_dense_attn_residual.py`
(`BRINGUP_IMPL=stub` for the stub; no prefix for the device gate).

## C.kda_dense.attn_residual implement (attempt 1)

What was done: `tt/residual.py:TtHcResidual` (+ `build_residual(cfg)`), weightless, replicated, no CCL. Inputs are
x [1, 1, S, 4H] (packed streams), hc [1, 1, S, 24] fp32 and y [1, 1, S, H]. For each output stream m:
`multiply(post col m, y)`, then 4 x `addcmul(o, x_i, comb col 4i + m)` (comb^T, the same as TtMHCWrap.hc_post), then
`typecast` to bf16 and `concat` dim -1. Uses DeepSeek's `_streams` and width-1 `ttnn.slice` columns of hc.
Precision: x, y and hc are cast to fp32. The mix runs in fp32 and rounds to bf16 once. That matches the fp32 reference,
where post and comb are cast to the activation dtype (fp32). A bf16 mix would give about 0.005 rel (test notes).
hooks: `_residual_host_fn` (x and y uploaded as bf16, hc as fp32, output read back as [S * 4, H]). It serves both
`attn_residual` and `ffn_residual` (`_RESIDUAL_STEPS`, same weightless module). Only `attn_residual` was added to
`DEVICE_STEPS["kda_dense"]`. The module caches nothing per layer, so the test's layer-1 call is safe.
Result (gate): pcc_attn_residual_L00 0.999994. L0: rel 0.0036, ratio [0.9959, 1.0050], worst row 0.0106, both terms
coef 1.0000 / rel 0.0015. L1: rel 0.0031, ratio [0.9965, 1.0035], terms rel 0.0022.
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_kda_dense_attn_residual.py`

## S.kda_dense.05 test (attempt 1)

Reviewed the rendered swap test (attn_hc + attn_collapse + attn_norm + attention + attn_residual on device, layer 0).
Rewrote it from swap 04's test. Kept the gated pcc_swap_out, every swap-04 check (incl. the KDA state) and the `_f32`
cast. Added h_mid checks, because at layer 0 comb^T @ in = in for any column-stochastic comb, so block out cannot see
comb bugs:
- vs the CPU residual of the inputs the step actually got (golden in, device attn_hc, device attn_out), and the same
  module on layer 1's golden. Both use the component test's limits (rel <= 0.01, ratio [0.985, 1.015], per-stream
  <= 0.015, worst row <= 0.05) plus the per-term coefficient / rel checks.
- vs golden h_mid, looser (rel 0.015, ratio [0.97, 1.03], per-stream 0.02), because it carries the attention's
  upstream error.
No new sensitivity runs: the residual bugs are the component review's; the block-out limits are swap 04's.
Results: device passes (out PCC 0.999982, rel 0.0064, ratio [0.9828, 1.0078]; h_mid vs golden 0.0084, streams 0/1
0.0115 (the larger post weight on the attention error); vs CPU same input 0.0017; layer 1 0.0031). Reference passes
(out rel 0.0017). Stub fails (PCC 0).
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_swap_kda_dense_05_attn_residual.py`
(prefix `BRINGUP_IMPL=reference` / `BRINGUP_IMPL=stub` for the other modes).

## C.kda_dense.ffn_hc test (attempt 1)

Reviewed the rendered component test for `ffn_hc` (the attn_hc op with hc_ffn_* weights, on h_mid). Rewrote it from the
attn_hc test: the gated PCC plus asserted per-part rel L2 and max abs, comb column sums, and range checks.
Sensitivity (CPU host script on the golden, not kept), PCC / part rel / max abs. Passing PCC 0.99: comb transposed
0.9945 / 0.113; softmax on the wrong axis 0.030; 10 Sinkhorn iterations 0.054; hc_eps 1e-5 0.0107 / 0.078; comb base
transposed 0.018 / 0.064; comb scale x1.02 0.042; rms eps 1.2e-5 pre 0.045; pre scale x1.02 0.031; post scale x1.02
0.0146; x1.02 on any part 0.020; last row zeroed max abs 0.99. Caught by PCC: h_mid streams differ at layer 0
(rel 0.6..1.5), so streams 0/1 swapped (0.912), stream-major flatten (0.548), attn weights (0.547) and rms eps 1e-6
(0.936). Noise: bf16 mix output 0.0026 / 0.007; 0.3% mix noise 0.0049 / 0.023; 1% 0.016 / 0.070.
Limits: rel L2 <= 0.01 and max abs <= 0.05 for every part (tighter than attn_hc's 0.03 / 0.02 / 0.15..0.2; the
attn_hc pessimistic noise model gives 0.07 here), column sums within 0.01.
Results: reference passes (PCC 0.999998, parts 0.0013 / 0.0017 / 0.0014). Stub fails (PCC 0). Device already passes
through the existing `TtHcWeights` (`build_hc(..., "ffn")`): PCC 0.999994, rel 0.0028 / 0.0022 / 0.0030, max abs
<= 0.0103, column sums [0.9965, 1.0009].
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_kda_dense_ffn_hc.py`
(prefix `BRINGUP_IMPL=reference` / `BRINGUP_IMPL=stub` for the other modes).

## S.kda_dense.06 test (attempt 1)

Reviewed the rendered swap test (attn_hc through attn_residual plus ffn_hc on device, layer 0). Rewrote it from swap 05's
test, keeping every swap-05 check and the gated pcc_swap_out. `_hc_checks` now takes a tag and limits, and the attn_hc
metric names are unchanged. Added:
- ffn_hc vs the CPU ffn_hc of the device h_mid, with the component test's limits (part rel <= 0.01, max abs <= 0.05,
  column sums within 0.01, ranges).
- ffn_hc vs golden, looser because it carries the h_mid error: part rel <= 0.015, max abs <= 0.06.
- Block out vs the CPU tail (ffn_collapse, ffn_norm, mlp, ffn_residual) from the same device h_mid with the CPU ffn_hc.
  This isolates ffn_hc's effect on block out: rel <= 0.005, per-row ratio [0.975, 1.025].
Sensitivity (CPU host script /tmp/s06_sens.py on the layer-0 golden h_mid, not kept), tail rel / ratio: comb base
transposed 0.0061 / [0.953, 1.029], post x1.01 0.0080, post x1.02 0.016, hc_eps 1e-5 0.017 / [0.878, ..], pre x1.02
0.0023 (caught by the part check only). Noise: bf16 mix 0.0018 / [0.989, 1.010], 0.3% mix noise 0.0036 / [0.979, 1.020].
My first ratio limit of [0.99, 1.01] failed the device (ratio [0.9873, 1.0129]), which is noise on small rows, not a bug.
Results: device passes (out PCC 0.999982, rel 0.0069, ratio [0.9797, 1.0087]; ffn_hc vs CPU same input 0.0024 / 0.0013 /
0.0027; vs golden 0.0031 / 0.0027 / 0.0045; tail rel 0.0020, ratio [0.9873, 1.0129]). Reference passes (out rel 0.0017,
tail exact). Stub fails (PCC 0, every check).
Watch: the block-out per-row ratio minimum vs golden drops with each swap (swap 05 0.9828, now 0.9797, limit 0.97). The
ffn steps still to come (ffn_collapse, ffn_norm, mlp, ffn_residual) add noise to this same ratio.
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_swap_kda_dense_06_ffn_hc.py`
(prefix `BRINGUP_IMPL=reference` / `BRINGUP_IMPL=stub` for the other modes).

## C.kda_dense.ffn_collapse test (attempt 1)

Reviewed the rendered component test for ffn_collapse (`sum_n pre[:, n] * h_mid[:, n]`, pre = ffn_hc[:, 0:4]). Rewrote
it on the attn_collapse test's pattern: the gated PCC plus asserted checks, on layer 0 and again (same weightless
module) on layer 1's golden.
Sensitivity (CPU host script on the goldens, not kept), layer 0 PCC / rel L2 / ratio: h_mid's streams already differ
at layer 0 (rel 0.6..1.5), so order bugs fail PCC there (pre reversed 0.919, stream-major rows 0.336, last stream
dropped 0.891, pre 0/1 swapped 0.988). PCC passes mean instead of sum (0.999997 / 0.75), x1.02 (rel 0.020, ratio
[1.018, 1.022]), x1.01 (0.0103, [1.008, 1.012]), last row zeroed (0.99993 / 0.012), and one row's pre reversed (rel
0.0104, worst row 0.38).
Noise: the golden's ffn_hc is bf16, so an exact fp32 reference already scores rel 0.0024. bf16 h_mid with fp32
accumulation gives rel 0.0027 / worst row 0.0041 / ratio [0.9978, 1.0022]; all-bf16 0.0038..0.0047 / <= 0.0076 /
[0.9974, 1.0024].
Checks per layer: rel L2 <= 0.008, worst per-token rel L2 <= 0.03, per-token norm ratio [0.99, 1.01]. These are
tighter than attn_collapse's 0.01 / [0.985, 1.015], because the noise here is small.
Results: reference passes (PCC 0.999997; L0 rel 0.0024 / row 0.0039, L1 0.0032 / 0.0065). Stub fails (PCC 0). The
device gate already passes through `TtHcCollapse` (hooks `_COLLAPSE_STEPS`): PCC 0.999996, L0 rel 0.0027 / row 0.0041 /
[0.9978, 1.0022], L1 0.0036 / 0.0067 / [0.9977, 1.0024].
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_kda_dense_ffn_collapse.py`
(prefix `BRINGUP_IMPL=reference` / `BRINGUP_IMPL=stub` for the other modes).

## S.kda_dense.07 test (attempt 1)

Reviewed the rendered swap test (attn_hc through ffn_hc plus ffn_collapse on device, layer 0). Rewrote it from swap 06's
test, keeping every swap-06 check and the gated pcc_swap_out. swap 06's tail check (metric `rel_l2_swap_out_vs_cpu_tail`)
now uses the CPU ffn_hc plus the CPU ffn_collapse, with the same limits. Added:
- ffn_collapse vs the CPU collapse of the device (h_mid, ffn_hc), and the same module on layer 1's golden, at the
  component test's limits (rel <= 0.008, ratio [0.99, 1.01], worst row <= 0.03).
- ffn_collapse vs golden ffn_in, looser because it carries the upstream error: rel <= 0.02, ratio [0.97, 1.03],
  worst row <= 0.05. The device scores 0.0118, so 0.015 would have left too little margin.
- ffn_collapse's share of block out: block out vs the CPU tail from the CPU collapse of the same device inputs.
  Limits: rel <= 0.003, ratio [0.995, 1.005].
Sensitivity (CPU host script /tmp/s07_sens.py on the layer-0 golden, not kept), tail rel / ratio: x1.01 0.0012 /
[0.9967, 1.0118]; one row's pre reversed 0.0001 / [0.9952, 1.0080], which passes the collapse checks and is caught only
here; last row zeroed 0.0075. Noise: bf16 output 0.0015 / [0.9989, 1.0015]; all-bf16 0.0019 / [0.9983, 1.0017].
ffn_norm does not remove the collapse's scale errors (small-RMS rows, eps).
Results: device passes (out PCC 0.999981, rel 0.0071, ratio [0.9797, 1.0088]; collapse vs CPU 0.0017, L1 0.0036, vs
golden 0.0118; cpu tail 0.0025 / [0.9873, 1.0130]; collapse-share tail 0.0015 / [0.9989, 1.0009]). Reference passes
(out rel 0.0017). Stub fails (PCC 0, every check).
Watch: the block-out ratio minimum vs golden is unchanged from swap 06 (0.9797, limit 0.97). ffn_norm, mlp and
ffn_residual are still to come.
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_swap_kda_dense_07_ffn_collapse.py`
(prefix `BRINGUP_IMPL=reference` / `BRINGUP_IMPL=stub` for the other modes).

## C.kda_dense.ffn_norm test (attempt 1)

Reviewed the rendered component test for ffn_norm (post_attention_layernorm, layer 0, [2048, 4096] bf16). Rewrote it
from the attn_norm test: kept the gated PCC, added asserted checks (informational metrics): finite, rel L2 <= 0.01,
per-token norm ratio [0.99, 1.01], worst per-token rel L2 <= 0.015. The ratio and worst-row limits are tighter than
attn_norm's because of the CPU mutation run on this golden (host script /tmp/ffn_norm_sens.py, not kept):
- Input row RMS 0.0034..0.0236 (mean square 1.2e-5..5.6e-4, about eps 1e-5); w in [0.028, 0.157].
- Hidden under PCC here: LayerNorm-style mean subtraction scores rel 0.0082 and worst row 0.029, so it passes
  attn_norm's limits (0.01 / 0.03); the 0.015 worst-row limit catches it. x1.01: rel 0.0103, ratio [1.008, 1.012].
  eps 1.2e-5: rel 0.0149, ratio [0.957, ..]; eps 8e-6: rel 0.0162, ratio [.., 1.050].
- PCC already catches no weight / 1 + w / attn_norm's weight / w reversed (0.81..0.84), eps 1e-4 (0.983), and last 32
  rows zeroed (0.992). Sum instead of mean passes PCC (0.9979) but gives rel 0.98.
- Headroom: fp32 CPU rel 0.0024 / ratio [0.9982, 1.0019] / worst row 0.0038; bf16 output 0.0029 / 0.0052. Squares
  accumulated in bf16 (rel 0.0062, ratio [0.983, 1.023], worst row 0.023) FAIL the new limits.
Results: reference passes (PCC 0.999997, rel 0.0024, ratio [0.9982, 1.0019], worst row 0.0038). Stub fails (PCC 0).
Device mode fails with NotImplementedError until the implement step adds the module.
Implement: reuse `tt/rms_norm.py:build_norm(..., "post_attention_layernorm")` (add `ffn_norm` to `_NORM_STEPS` and
`DEVICE_STEPS`), keeping eps 1e-5 and fp32 accumulation of the squares (fp32 dest acc). attn_norm's device module
scored worst row 0.0037 there, so it should have margin here.
Gotcha: the log shows `FAIL pcc_ffn_norm_L00: pcc=0.000000` before the real result. That line is from the runner's
up-front compile collect pass, not from the test.
Re-run: `PYTHONPATH=$PWD BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_kda_dense_ffn_norm.py`
(`BRINGUP_IMPL=stub` for the stub; no prefix for the device gate).

## C.kda_dense.ffn_norm implement (attempt 1)

- Reused `tt/rms_norm.py:TtRMSNorm` / `build_norm` unchanged: added `"ffn_norm": "post_attention_layernorm"` to
  `_NORM_STEPS` in `hooks.py`. Same op as attn_norm: `ttnn.bringup.rms_norm`, replicated [1, 1, S, 4096], plain `w`
  gamma, eps 1e-5 from `cfg.rms_norm_eps`, HiFi4 + fp32 dest acc. `GLM_NORM_IMPL=native` still selects `ttnn.rms_norm`.
- `DEVICE_STEPS["kda_dense"]` now also lists `ffn_hc` and `ffn_collapse` (the earlier implement steps did not add
  them, although both passed their component gates and swap 06/07) as well as `ffn_norm`, so the ladder's hybrid model
  runs every validated step on the device. Swap tests do not read this set; they call `device_component` directly.
- Gate results: PCC 0.999996, rel L2 0.0029, per-token norm ratio [0.9974, 1.0027], worst row rel L2 0.0052 (limits
  0.01 / [0.99, 1.01] / 0.015).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_kda_dense_ffn_norm.py`

## S.kda_dense.08 test (attempt 1)

Reviewed the rendered swap test (attn_hc through ffn_collapse plus ffn_norm on device, layer 0). Rewrote it from swap
07's test. It keeps every swap-07 check and the gated pcc_swap_out. Added:
- ffn_norm vs the CPU norm of the device ffn_in, and the module on layer 1's golden ffn_in (layer-0 weights on both
  sides), at the component test's limits (rel <= 0.01, ratio [0.99, 1.01], worst row <= 0.015).
- ffn_norm vs golden, which carries the upstream ffn_in error (0.0118): rel <= 0.015, ratio [0.98, 1.02], worst row
  <= 0.03. Device: 0.0080 / [0.9951, 1.0032] / 0.0130.
- ffn_norm's share of block out: block out vs the CPU tail (mlp, ffn_residual) from the CPU norm of the device ffn_in.
  Limits: rel <= 0.004, ratio [0.99, 1.01]. Device: 0.0016 / [0.9977, 1.0023].
- The swap-06 tail (metric `rel_l2_swap_out_vs_cpu_tail`) now runs CPU ffn_hc + collapse + norm. Limits unchanged;
  device 0.0029 / [0.9874, 1.0130].
- The swap-07 collapse-share tail (`cpu_fc_tail`) now puts the device norm on the CPU collapse, so only the collapse
  differs. Limits unchanged. Device 7e-6: the device norm rounds its input to bf16.
Sensitivity (CPU host scripts /tmp/s08_sens.py and /tmp/s08_fc.py, not kept): the mlp amplifies a norm scale error
into block out by about 1.6x. Norm x1.005 passes every norm check (ratio 1.005) and gives tail rel 0.008, ratio 1.0157.
Mean subtraction: tail rel 0.0116. eps 1.2e-5: 0.011. Last row x1.05: ratio 1.0178. Noise: bf16 output 0.0015;
0.2% per-row rsqrt noise 0.0034 / [0.9836, 1.0184], which fails the ratio. Collapse bugs through the device-norm tail:
x1.01 ratio 1.0125; one row's pre reversed 1.0080 (both fail [0.995, 1.005]); all-bf16 collapse passes (0.0024).
Results: device passes (out PCC 0.999980, rel 0.0072, ratio [0.9797, 1.0087]). Reference passes (out rel 0.0017).
Stub fails (PCC 0, every check).
Watch: the block-out ratio minimum vs golden is still 0.9797 (limit 0.97). mlp and ffn_residual are next. The mlp
amplifies upstream error, so check the block-out margin there first.
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_swap_kda_dense_08_ffn_norm.py`
(prefix `BRINGUP_IMPL=reference` / `BRINGUP_IMPL=stub` for the other modes).

## C.kda_dense.mlp test (attempt 1)

Reviewed the rendered component test for mlp (dense clamped SwiGLU 12288, layer 0, [2048, 4096] bf16 golden). I rewrote
it from the ffn_norm test. It keeps the gated PCC and adds asserted checks (informational metrics):
- vs golden: finite, rel L2 <= 0.015, per-token norm ratio [0.99, 1.01], worst row rel L2 <= 0.02, and the global scale
  coefficient <got, want> / <want, want> in [0.995, 1.005].
- Clamp probe: the module on 48 * ffn_norm vs the CPU reference on the same input: rel <= 0.01, ratio [0.99, 1.01],
  worst row <= 0.03. The golden never reaches the limit (|gate| <= 0.70, |up| <= 0.66), so without the probe no
  clamp bug is visible.
Sensitivity (CPU host scripts /tmp/mlp_sens/s{1,2,3}.py, not kept):
- Headroom: fp32 CPU 0.0026 / [0.9986, 1.0013] / 0.0031. bf16 weights + intermediates + output: 0.0039 / worst row
  0.0044 / coef 0.9998. bfp8 weights: 0.0068..0.0089 / worst row 0.0078..0.0105.
- Fail: bfp8 x and h, 0.025 / worst row 0.034 (outlier channels, |x| up to 1.53). Keep activations bf16.
  gelu instead of silu: 0.031 / 0.046. gate/up swapped: 0.059. x1.01: ratio max 1.0113. Row zeroed: worst row 1.0.
  7-bit weight truncation (HiFi2-like): coef 0.9938, although its ratio [0.9925, 0.9952] passes. TP shard doubled or
  dropped: PCC 0.94..0.98.
- Probe at 48x (gate > 10 on 0.007%): bf16 0.0030 / worst row 0.0037. No gate clamp 0.27 (ratio 1.108), no up clamp
  0.19, up clamped above only 0.070, limit 7 0.19. gate clamped to [-10, 10] and min(silu(g), 10) are numerically
  equivalent and pass. Limit 9.5 scores 0.029, just inside.
Results: reference passes (PCC 0.999997, rel 0.0026, coef 0.99998, probe 0). Stub fails (PCC 0). The device gate fails
with NotImplementedError until the implement step adds the module.
Implement: plan per components.yaml (bf16 weights, fp32 gate/up out, HiFi4 + fp32 acc). Check that ttnn.silu is not an
approximate mode that biases the output: the coefficient check allows only 0.5%.
Re-run: `PYTHONPATH=$PWD BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_kda_dense_mlp.py`
(`BRINGUP_IMPL=stub` for the stub; no prefix for the device gate).

## C.kda_dense.mlp implement (attempt 1)

Added `tt/mlp.py:TtDenseMLP` + `build_mlp` (from `mimo_v2_6_d_p_2x2/tt/mlp.py`), registered in `hooks._device_step`
(step `mlp`, non-MoE layers, wrapped by `_norm_host_fn`) and added `mlp` to `DEVICE_STEPS["kda_dense"]`.
- TP=4 over the intermediate: chip d holds gate/up columns and down rows [3072d, 3072d + 3072). Weights: the loader's
  fp8 block dequant (`loader.weight`), stored bf16. All three linears HiFi4 + fp32 acc (`common.hifi4_config`).
- Forward: gate/up linear fp32 out -> `ttnn.minimum(g, 10)` -> `ttnn.silu` -> `ttnn.clamp(u, -10, 10)` ->
  `ttnn.multiply` (fp32) -> down linear fp32 out -> `ttnn.all_reduce(cluster_axis=None)` in fp32 -> typecast bf16.
  No host work in the forward.
- Gotcha: with bf16 down partials the all_reduce biased the output up (coef 1.0018, ratio [1.0005, 1.0032]; stage
  probe (tt-probe, not kept: outside this step's paths): all_reduce vs host sum of the same
  partials coef 1.0019). fp32 all_reduce fixes it (known issues, Proposed). silu/minimum/clamp on fp32 are exact
  (rel 6e-8 on a separate probe).
- Gate: PCC 0.999994; golden rel L2 0.0036, ratio [0.9977, 1.0004], worst row 0.0043, coef 0.99917; clamp probe rel
  0.0031, ratio [0.9980, 1.0004], worst row 0.0039.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_kda_dense_mlp.py`

## S.kda_dense.09 test (attempt 1)

Reviewed the rendered swap test (attn_hc through ffn_norm plus mlp on device, layer 0). I rewrote it from swap 08's test.
It keeps every swap-08 check and the gated pcc_swap_out. Added:
- mlp vs the CPU mlp of the device ffn_norm, and the module on layer 1's golden ffn_norm, at the component test's
  limits (rel <= 0.015, ratio [0.99, 1.01], worst row <= 0.02, coef [0.995, 1.005]). The clamp probe (48 * device
  ffn_norm vs CPU) uses the component limits (0.01 / [0.99, 1.01] / 0.03).
- mlp vs golden mlp_out, looser because it carries the upstream error: rel <= 0.02, ratio [0.98, 1.02], worst row
  <= 0.03, coef [0.99, 1.01]. Device: 0.0138 / [0.9938, 1.0041] / 0.0243 / 0.99779.
- mlp's share of block out: block out vs ffn_residual(h_mid, device ffn_hc, CPU mlp of device ffn_norm). Limits: rel
  <= 0.0035, ratio [0.995, 1.005]. Device: 0.0020 / [0.9986, 1.0004].
- swap 08's collapse- and norm-share tails now run the device mlp (`muts["mlp"]`) instead of the CPU mlp, so each still
  isolates one step. Limits unchanged. Device: 8e-5 and 0.0016 / [0.9971, 1.0025]. The all-CPU ffn tail (cpu_tail)
  is unchanged; device 0.0035 / [0.9874, 1.0130] (limit 0.005 / [0.975, 1.025]).
Sensitivity (CPU host script /tmp/s09/sens.py, not kept): an mlp error reaches block out at about 0.8x. Results as tail
rel / ratio: x1.005 0.0040 / 1.0079; 7-bit weight truncation (HiFi2-like) 0.0052 / 0.9905; 0.2% per-row noise
[0.9913, 1.0068]; row zeroed [1.0, 1.39]. All of these fail. bf16 everything 0.0018 / [0.9995, 1.0002] passes.
Results: device passes (out PCC 0.999978, rel 0.0075, ratio [0.9797, 1.0083]). Reference passes (out rel 0.0017).
Stub fails (PCC 0 and every golden check). The same-input mlp checks score 0 on the stub, because the CPU mlp of a zero
input is zero.
Watch: cpu_tail margin is shrinking (0.0025 -> 0.0029 -> 0.0035 of 0.005), and mlp vs golden worst row is 0.0243 of
0.03. ffn_residual is next.
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_swap_kda_dense_09_mlp.py`
(prefix `BRINGUP_IMPL=reference` / `BRINGUP_IMPL=stub` for the other modes).

## C.kda_dense.ffn_residual test (attempt 1)

Reviewed the rendered component test for `out = post * mlp_out + comb^T @ h_mid` ([S * 4, H], token-major; inputs
h_mid, ffn_hc, mlp_out). Rewrote it from the attn_residual test: the gated PCC plus asserted checks, run on layer 0
and again (same weightless module) on layer 1's golden. Unlike attn_residual, h_mid's streams already differ at
layer 0, so the comb is visible there (comb not transposed: rel 0.098 at layer 0, 0.038 at layer 1).
Sensitivity (CPU host script /tmp/ffnres/sens.py, not kept), layer 0 PCC / rel L2 / ratio: post x1.01 (= mlp_out
x1.01) 0.99996 / 0.0087 / [0.9925, 1.0205]; post x1.02 0.99987 / 0.016 / worst row 0.053; comb x1.02 0.024; last row
zeroed 0.99998 / 0.0062 / ratio min 0. The fp32 reference itself is at rel 0.0034 / [0.9949, 1.0052] vs the bf16
golden; all-bf16 mix 0.0044 / [0.9945, 1.0053] / term rel up to 0.0082.
Limits tightened from attn_residual: ratio [0.99, 1.01], per-stream rel <= 0.012, worst row <= 0.03, term coefficient
[0.99, 1.01], term rel <= 0.015; rel L2 <= 0.01 kept. The layer-1 metrics are tagged `L01`.
Results: reference passes (L0 rel 0.0034, L1 0.0028). Stub fails (PCC 0). Device passes already, because
`_residual_host_fn` serves ffn_residual too: PCC 0.999993, L0 rel 0.0037 / [0.9949, 1.0052] / worst row 0.0096, term
rel 0.0021 / 0.0014; L1 rel 0.0032, post term rel 0.0055.
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_kda_dense_ffn_residual.py`
(prefix `BRINGUP_IMPL=reference` / `BRINGUP_IMPL=stub` for the other modes).

## S.kda_dense.10 test (attempt 1)

Reviewed the rendered swap test (all ten kda_dense steps on device, layer 0). I rewrote it from swap 09's test. It keeps
every swap-09 check (limits unchanged) and the gated pcc_swap_out. Changes:
- The collapse-, norm- and mlp-share tails now end in the device ffn_residual (`muts["ffn_residual"]`) instead of the
  CPU one, so each still isolates one step. Device: 1e-4, 0.0017 / [0.9971, 1.0026], 0.0025 / [0.9985, 1.0005].
  cpu_tail (all-CPU ffn tail, now five device steps' share) is unchanged at 0.005 / [0.975, 1.025]; device 0.0038 /
  [0.9873, 1.0131].
- New: ffn_residual vs the fp32 CPU residual of the (h_mid, ffn_hc, mlp_out) it actually got, and the module on
  layer 1's golden inputs vs the CPU residual of those inputs. Both check rel L2, per-row ratio, per-stream rel, worst
  row, and each term on its own. Limits: rel <= 0.0035, ratio [0.9975, 1.0025], stream <= 0.005, row <= 0.008,
  term coefficient [0.998, 1.002], term rel <= 0.012. These are tighter than the component test's because the
  reference is exact fp32, not the bf16 golden.
Sensitivity (CPU host script /tmp/s10/sens.py, not kept), vs the exact residual: bf16 output (what the device does)
0.0017 / [0.9998, 1.0002]; all-bf16 mix 0.0029 / [0.9980, 1.0018] / row 0.0054 / L1 term rel 0.0082 (passes). These
fail: post x1.005 (L0 ratio 1.0079, coefficient 1.005 on both layers), comb x1.005 (0.0059), 7-bit output (ratio
0.994), last row x1.02 (ratio 1.02), and last row zeroed.
Results: device passes (out PCC 0.999977, rel 0.0077, ratio [0.9797, 1.0082]; ffn_residual same input 0.0017, coef
1.0000; L1 0.0017). Reference passes (out rel 0.0017, same-input checks 0). Stub fails (PCC 0 and every golden check).
On layer 0 the same-input checks score 0 for the stub, because the CPU steps of zero inputs are zero; layer 1 fails.
Watch: cpu_tail is at 0.0038 of 0.005 (0.0035 in swap 09); block out rel 0.0077 of 0.01.
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_swap_kda_dense_10_ffn_residual.py`
(prefix `BRINGUP_IMPL=reference` / `BRINGUP_IMPL=stub` for the other modes).

## C.dsa_moe.attn_hc test (attempt 1)

Reviewed the rendered component test for attn_hc at layer 3 (the same op as kda_dense attn_hc, with layer 3's weights).
I rewrote it from the kda_dense attn_hc test: the gated PCC plus asserted part checks. The limits are re-measured on
the layer-3 golden, because the layer-0 limits do not fit here:
- At layer 3 the streams differ (rel ~1.0 from stream 0), so PCC catches stream-order bugs (0.60..0.64). Every
  coefficient bug still scores PCC >= 0.9967, for example comb transposed 0.99926.
- post is saturated near 0: every entry is <= 0.024, and column 6 is ~1e-8. So post max abs is ~1e-4, and the layer-0
  limit of 0.2 means nothing here.
- Limits: part rel L2 <= 0.01; max abs pre 0.02, post 5e-4, comb 0.02; worst single-column rel L2 <= 0.07 (new);
  comb column sums within 0.01; range checks.
Sensitivity (CPU host script /tmp/dsahc/sens.py, not kept; numbers in the test docstring). Caught: comb transposed,
wrong softmax axis, 10/18/19 iterations, hc_eps 1e-5 and 0, comb base transposed, rms eps 1e-6, scales swapped,
x1.02 on any scale or part, last row zeroed. Not caught: rms eps 1.2e-5, and 0.3% mix noise (device-like).
Results: device mode already passes, because `_device_step` builds tt/mhc.py for any layer. It scores PCC 0.999998,
part rel 0.0021 / 0.0030 / 0.0019, max abs 4.9e-3 / 1.4e-4 / 6.8e-3, worst column 0.034 (column 9, a comb entry
<= 3e-4). Reference passes (0.0013 / 0.0017 / 0.0012, worst column 0.0035). Stub fails (PCC 0).
Watch: the worst-column margin is about 2x (0.034 of 0.07). The next step, implement, only needs to add attn_hc to
`DEVICE_STEPS["dsa_moe"]`.
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_dsa_moe_attn_hc.py`
(prefix `BRINGUP_IMPL=reference` / `BRINGUP_IMPL=stub` for the other modes).

## S.dsa_moe.01 test (attempt 1)

Reviewed the rendered swap test (dsa_moe layer 3, attn_hc on device). I rewrote it from swap kda_dense 01. It keeps
the gated pcc_swap_out and takes the attn_hc part checks from the layer-3 component test (part rel 0.01; max abs
0.02 / 5e-4 / 0.02; worst column 0.07; column sums; ranges). The attn_hc input here is the golden `in`.
New at layer 3: the MoE is in the block, and near-tie top-8 flips break per-row checks against the golden. The fp32
CPU reference flips 36 tokens against the golden, with ratio up to 1.033 on those rows. Checks:
- block out vs golden: rel L2 <= 0.01; per-row ratio [0.975, 1.025] on rows with the golden's top-8, [0.9, 1.1] on
  flipped rows (selection taken from the nonzero pattern of `router`).
- block out vs the all-CPU block of the same golden `in` (a second CPU run, 5 s; it differs only in attn_hc): router
  flips <= 64 tokens; rel L2 <= 0.005 and ratio [0.98, 1.02] on same-routing rows.
- `_f32` wraps the override output (known issue: device step feeding a CPU step).
Sensitivity (CPU host script /tmp/dsas01/sens2.py, not kept; the numbers are in the test docstring). Caught: comb
transposed, comb x1.01 / x1.02, post x1.02 / x1.05, 10 iterations, rms eps 1e-6, last row zeroed, hc_eps 1e-5 (CPU
ratio 0.985). The attn_hc part checks catch hc_eps 0 and 19 iterations. Not caught in block out: a pre scale (attn_norm
removes it; the part checks catch x1.02) and rms eps 1.2e-5.
Results: device passes (PCC 0.999994, rel 0.0038; golden same-routing rows [0.9922, 1.0071], 46 flips; vs CPU 29
flips, rel 0.0020, [0.9924, 1.0075]; attn_hc parts 0.0021 / 0.0030 / 0.0019, worst column 0.034). Reference passes
(rel 0.0028, vs CPU 0). Stub fails (PCC 0 and every check).
Watch: the worst-column margin is 2x (0.034 of 0.07). Later dsa_moe swaps should keep the CPU-block comparison, but
run the CPU tail from the device inputs of the step they add (see the kda_dense swaps 06-10).
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_swap_dsa_moe_01_attn_hc.py`
(prefix `BRINGUP_IMPL=reference` / `BRINGUP_IMPL=stub` for the other modes).

## C.dsa_moe.attn_collapse test (attempt 1)

Reviewed the rendered component test for attn_collapse at layer 3 (the same weightless op as kda_dense attn_collapse).
I rewrote it from the kda_dense test and dropped the second-layer run: at layer 3 the streams already differ (rel ~1.0
from stream 0) and pre spans 9e-4..0.98. So PCC catches stream and pre order bugs here (pre reversed, pre 0/1 swapped,
stream-major rows, last stream dropped, post instead of pre, unweighted mean: PCC <= 0.79).
Checks: the gated PCC; finite; rel L2 <= 0.01; per-token norm ratio [0.99, 1.01] (tighter than layer 0's
[0.985, 1.015], so x1.01 fails at 1.0128); worst per-token rel L2 <= 0.008 (new).
Sensitivity (CPU host script /tmp/dsacol/sens.py, not kept; numbers in the test docstring). These pass PCC 0.99 and
fail the extra checks: pre normalized to sum 1, output x1.01 / x1.02, pre column 0 x1.01 (worst row 0.0103), last row
or last 32 rows zeroed, last 32 columns zeroed, one row's pre reversed. Noise: fp32 reference 0.0027 / worst row
0.0037; all-bf16 products and accumulation 0.0039 / 0.0047; 0.3% element noise 0.0040 / 0.0047.
Results: device passes already, because `_device_step` builds tt/collapse.py for any layer. It scores PCC 0.999995,
rel 0.0031, ratio [0.9974, 1.0028], worst row 0.0042. Reference passes (0.0027 / [0.9975, 1.0028] / 0.0037). Stub
fails (PCC 0).
Watch: the worst-row margin is about 1.9x (0.0042 of 0.008). The next step, implement, only needs to add attn_collapse
to `DEVICE_STEPS["dsa_moe"]`.
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_dsa_moe_attn_collapse.py`
(prefix `BRINGUP_IMPL=reference` / `BRINGUP_IMPL=stub` for the other modes).

## S.dsa_moe.02 test (attempt 1)

Reviewed the rendered swap test (dsa_moe layer 3, attn_hc + attn_collapse on device). I rewrote it from swap dsa_moe
01, which keeps every swap 01 check: block out vs golden split by routing; block out vs the all-CPU block of the golden
`in` (now both device steps' share); the attn_hc part checks. Added:
- attn_collapse vs golden attn_in: rel 0.01, ratio [0.985, 1.015], worst row 0.02. The component test's worst-row
  limit (0.008) does not fit here, because the input is the device attn_hc (0.3% attn_hc noise gives 0.0084).
- attn_collapse vs the fp32 CPU collapse of its own inputs (golden `in`, device attn_hc): rel 0.0045, ratio
  [0.996, 1.004], worst row 0.0065. This is the sharp check: x1.005 and pre column 0 x1.01 fail, and bf16 accumulation
  (0.0031 / [0.9970, 1.0026]) passes.
- The collapse share of block out: a third CPU block run with attn_hc fixed to the device output and the CPU
  collapse. Limits: flips <= 24, same-routing rel 0.0015, ratio [0.995, 1.005].
- Limits are written `not x <= lim`, so NaN fails (under the stub, the same-input rel is 0/0).
Sensitivity (CPU host script /tmp/dsas02/sens.py, not kept; numbers in the test docstring). attn_norm removes a
collapse scale error, so block out cannot see x1.01 (share rel 0.0001). The gate (PCC) catches only pre reversed and
stream-major rows. Share catches: pre col 0 x1.01 (40 flips, 0.9939), one row's pre reversed (ratio 1.0059), zeroed
rows and columns.
Results: device passes (PCC 0.999994; collapse same-input 0.0017 / [0.9998, 1.0001] / 0.0017, which is just the bf16
output rounding; vs golden 0.0039 / worst row 0.0070; share 4 flips / 0.0002 / [0.9995, 1.0004]; vs CPU 27 flips /
0.0020). Reference passes (every same-input check 0). Stub fails (PCC 0 and every check).
Note: `DEVICE_STEPS["dsa_moe"]` in hooks.py is still empty. The swap tests call `device_component` directly, so the
next implement step only needs to register attn_hc / attn_collapse there.
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_swap_dsa_moe_02_attn_collapse.py`
(prefix `BRINGUP_IMPL=reference` / `BRINGUP_IMPL=stub` for the other modes).

## C.dsa_moe.attn_norm test (attempt 1)

Reviewed the rendered component test for attn_norm at layer 3 (input_layernorm, [2048, 4096] bf16). Rewrote it from
the kda_dense attn_norm test (LAYER = 3), with ffn_norm's tighter limits: the gated PCC; finite; rel L2 <= 0.01;
per-token norm ratio [0.99, 1.01]; worst per-token rel L2 <= 0.015. Limits are written `not x <= lim` so NaN fails.
Sensitivity (CPU host script /tmp/dsanorm/sens.py, not kept; numbers in the test docstring):
- Layer-3 input rows are smaller than layer 0's (row RMS 0.0015..0.0099, mean square 2.1e-6..9.7e-5, at or below
  eps 1e-5), and w is small (0.0126..0.0286). So eps dominates the small rows: eps 1.1e-5 / 9e-6 score PCC ~1.0 but
  rel 0.021 / 0.022 and ratio 0.961 / 1.044.
- Hidden under PCC and caught only by the worst-row limit: LayerNorm-style mean subtraction (rel 0.0141, worst row
  0.043). x1.01 fails rel (0.0103) and ratio (1.0102); x1.005 passes everything (0.0055).
- PCC catches sum instead of mean, eps 0 / 1e-6 / 1e-4, no weight, 1 + w, ffn_norm's or layer 0's weight, w reversed
  (0.994), last 32 rows zeroed.
- Noise: fp32 CPU 0.0023 / [0.9998, 1.0002] / 0.0025; bf16 input, weight and output 0.0028 / 0.0030; 0.3% element
  noise 0.0042 / 0.0045 (pass). Squares accumulated in bf16 (ratio [0.989, 1.025], worst row 0.025) and 5e-3 rsqrt
  row error (0.021) fail.
Results: device already passes: PCC 0.999996, rel 0.0028, ratio [0.9990, 1.0005], worst row 0.0031. `_device_step`
builds `tt/rms_norm.py` for any layer. Reference passes (0.999997 / 0.0023 / [0.9998, 1.0002] / 0.0025). Stub fails
(PCC 0).
Next (implement): the module needs no change. Add `attn_norm` to `DEVICE_STEPS["dsa_moe"]` (still empty; attn_hc and
attn_collapse are not there either). Keep eps exactly 1e-5 and fp32 accumulation of the squares.
Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_dsa_moe_attn_norm.py`
(prefix `BRINGUP_IMPL=reference` / `BRINGUP_IMPL=stub` for the other modes).
